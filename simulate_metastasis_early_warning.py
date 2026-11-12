import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.tumor_microenvironment_dynamics import TumorMicroenvironmentSystem

def run_metastasis_simulation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    
    system = TumorMicroenvironmentSystem(
        r_stem=0.35,
        r_diff=0.10,
        r_norm=0.25,
        kd0=0.03,
        gamma_diff=1.5,
        kr0=0.10,
        d_stem=0.005,
        d_diff0=0.08,
        alpha_cyt=0.08,
        beta_mat=1.5,
        eta_inv=0.0003,
        delta_stress=0.04,
        s_cyt=0.15,
        rho_cyt=0.08,
        alpha_ag=100.0,
        mu_supp=0.03,
        kappa_ici=3.0,
        mu_metab=0.05,
        d_cyt=0.05,
        s_supp=0.05,
        rho_supp=0.06,
        d_supp=0.04,
        beta_glyc=0.0015,
        beta_hypox=0.05,
        d_metab=0.20,
        gamma_mat=0.0008,
        d_mat=0.05
    )
    
    t_span = (0.0, 120.0)
    t_eval = np.linspace(0.0, 120.0, 1201)
    
    y0_localized = [25.0, 120.0, 750.0, 2.0, 10.0, 0.40, 0.40]
    
    def emt_transition(t, state):
        trigger = 0.0005 * (1.0 + t / 40.0)
        res = system.vector_field(t, state, u_diff=0.0, u_chemo=0.0, u_ici=0.0, u_mat=0.0)
        res[5] += trigger
        return res

    sol = solve_ivp(
        emt_transition,
        t_span,
        y0_localized,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    
    n_pts = len(sol.t)
    dt = sol.t[1] - sol.t[0]
    
    omega1_cancer = np.zeros(n_pts)
    raychaudhuri_theta = np.zeros(n_pts)
    
    c_stem_arr = sol.y[0]
    sigma_arr = sol.y[5]
    mat_arr = sol.y[6]
    
    window_pts = 40
    for k in range(n_pts):
        start_idx = max(0, k - window_pts + 1)
        if k - start_idx < 10:
            omega1_cancer[k] = 0.08
            raychaudhuri_theta[k] = -0.15
            continue
            
        y_slice = np.vstack([
            c_stem_arr[start_idx : k + 1],
            sigma_arr[start_idx : k + 1],
            mat_arr[start_idx : k + 1]
        ])
        
        y_mean = np.mean(y_slice, axis=1, keepdims=True)
        y_normed = y_slice - y_mean
        
        y0 = y_normed[:, :-1]
        y1 = y_normed[:, 1:]
        
        reg_eye = 1e-4 * np.eye(3)
        cov0 = y0 @ y0.T + reg_eye
        cross_cov = y1 @ y0.T
        
        try:
            a_mat = cross_cov @ np.linalg.inv(cov0)
            eig_vals = np.linalg.eigvals(a_mat)
            cont_eigs = np.log(np.maximum(1e-6, np.abs(eig_vals))) / dt
            
            omega1_cancer[k] = - float(np.max(cont_eigs))
            raychaudhuri_theta[k] = float(np.sum(cont_eigs))
        except Exception:
            omega1_cancer[k] = omega1_cancer[k - 1]
            raychaudhuri_theta[k] = raychaudhuri_theta[k - 1]

    smooth_span = 25
    kernel = np.ones(smooth_span) / smooth_span
    omega1_smooth = np.convolve(omega1_cancer, kernel, mode="same")
    
    ctc_dissemination = np.maximum(
        0.0,
        (sol.y[0] - 250.0) * sol.y[6] * (sol.y[5] / (0.5 + sol.y[5])) * 0.20
    )
    
    df_meta = pd.DataFrame({
        "time_days": sol.t,
        "stem_cells": sol.y[0],
        "diff_cells": sol.y[1],
        "norm_cells": sol.y[2],
        "metab_stress": sol.y[5],
        "matrix_stiffness": sol.y[6],
        "ctc_dissemination": ctc_dissemination,
        "omega1_cancer": omega1_smooth,
        "raychaudhuri_theta": raychaudhuri_theta
    })
    
    csv_path = os.path.join(res_dir, "metastasis_early_warning_simulation.csv")
    df_meta.to_csv(csv_path, index=False)
    
    clinical_metastasis_day = None
    meta_mask = ctc_dissemination > 5.0
    if np.any(meta_mask):
        clinical_metastasis_day = float(sol.t[np.argmax(meta_mask)])
        
    warning_day = None
    warn_mask = (omega1_smooth < 0.02) & (sol.t > 15.0)
    if np.any(warn_mask):
        warning_day = float(sol.t[np.argmax(warn_mask)])
        
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(sol.t, ctc_dissemination, "r-", linewidth=2.5, label="Circulating Tumor Cells (CTCs)")
    plt.axhline(5.0, color="red", linestyle="--", label="Clinical Dissemination Threshold")
    if clinical_metastasis_day:
        plt.axvline(clinical_metastasis_day, color="red", linewidth=2.0, alpha=0.7, label=f"Clinical Metastasis (Day {clinical_metastasis_day:.1f})")
    if warning_day:
        plt.axvline(warning_day, color="blue", linewidth=2.0, linestyle="--", label=f"Omega_1 Softening Alert (Day {warning_day:.1f})")
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Circulating Tumor Cell Flux", fontsize=12)
    plt.title("Metastatic Cascade and Hematogenous Dissemination", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    
    plt.subplot(2, 2, 2)
    plt.plot(sol.t, sol.y[0], "m-", linewidth=2.2, label="Cancer Stem Cells (CSCs)")
    plt.plot(sol.t, sol.y[5], "orange", linewidth=2.0, label="Metabolic Stress")
    plt.plot(sol.t, sol.y[6], "brown", linestyle="-.", linewidth=2.0, label="Matrix Stiffness")
    if warning_day:
        plt.axvline(warning_day, color="blue", linewidth=2.0, linestyle="--")
    if clinical_metastasis_day:
        plt.axvline(clinical_metastasis_day, color="red", linewidth=2.0, alpha=0.7)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Microenvironment State Index", fontsize=12)
    plt.title("Pre-Metastatic Niche Remodeling (EMT Triggers)", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    
    plt.subplot(2, 2, 3)
    plt.plot(sol.t, raychaudhuri_theta, "g-", linewidth=2.0, label="Raychaudhuri Expansion Theta")
    plt.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    plt.axhline(0.05, color="orange", linestyle=":", label="Divergence Boundary (UNSAT)")
    if warning_day:
        plt.axvline(warning_day, color="blue", linewidth=2.0, linestyle="--")
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Theta (Bundle Expansion Rate)", fontsize=12)
    plt.title("Phase-Space Geometric Bundle Expansion", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    
    plt.subplot(2, 2, 4)
    plt.plot(sol.t, omega1_smooth, "b-", linewidth=2.5, label="Sturm-Liouville Gap Omega_1")
    plt.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    plt.axhline(0.02, color="orange", linestyle=":", label="Critical Softening Threshold")
    if warning_day and clinical_metastasis_day:
        lead_days = clinical_metastasis_day - warning_day
        plt.axvspan(warning_day, clinical_metastasis_day, color="green", alpha=0.15, label=f"Early Warning Lead Time: {lead_days:.1f} Days")
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Turbulence Gap Omega_1", fontsize=12)
    plt.title("Precursor Phase Transition: Spectral Gap Softening", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(res_dir, "metastasis_gap_softening_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("Metastasis Early Warning Simulation Completed.")
    print(f"Results saved to {csv_path} and {plot_path}")
    if warning_day and clinical_metastasis_day:
        print(f"Pre-Metastatic Early Warning Lead Time: {clinical_metastasis_day - warning_day:.1f} days before clinical dissemination.")
    else:
        print(f"Warning Day: {warning_day}, Clinical Metastasis Day: {clinical_metastasis_day}")

if __name__ == "__main__":
    run_metastasis_simulation()