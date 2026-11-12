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

def compute_raychaudhuri_expansion(sol, dt=0.1, window_pts=25):
    n_pts = len(sol.t)
    theta_arr = np.zeros(n_pts)
    y_target = sol.y[[0, 1, 3], :]
    
    for k in range(n_pts):
        start_idx = max(0, k - window_pts + 1)
        if k - start_idx < 8:
            theta_arr[k] = -0.15
            continue
            
        y_slice = y_target[:, start_idx : k + 1]
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
            theta_arr[k] = float(np.sum(cont_eigs))
        except Exception:
            theta_arr[k] = theta_arr[k - 1]
            
    smooth_span = 15
    kernel = np.ones(smooth_span) / smooth_span
    theta_smooth = np.convolve(theta_arr, kernel, mode="same")
    return theta_smooth

def run_checkpoint_simulation():
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
        rho_cyt=0.12,
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
    
    t_span = (0.0, 180.0)
    t_eval = np.linspace(0.0, 180.0, 1801)
    
    y0_exhausted = [40.0, 280.0, 500.0, 2.5, 15.0, 0.80, 0.70]
    
    def responder_dynamics(t, state):
        if 14.0 <= t <= 70.0:
            u_ici = 3.0
            u_mat = 2.0
            u_diff = 2.5
        else:
            u_ici = 0.0
            u_mat = 0.0
            u_diff = 0.0
        return system.vector_field(t, state, u_diff=u_diff, u_chemo=0.0, u_ici=u_ici, u_mat=u_mat)
        
    def non_responder_dynamics(t, state):
        if 14.0 <= t <= 70.0:
            u_ici = 1.0
            u_mat = 0.0
            u_diff = 0.0
        else:
            u_ici = 0.0
            u_mat = 0.0
            u_diff = 0.0
        return system.vector_field(t, state, u_diff=u_diff, u_chemo=0.0, u_ici=u_ici, u_mat=u_mat)

    sol_resp = solve_ivp(
        responder_dynamics,
        t_span,
        y0_exhausted,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    
    sol_non = solve_ivp(
        non_responder_dynamics,
        t_span,
        y0_exhausted,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    
    dt = t_eval[1] - t_eval[0]
    theta_resp = compute_raychaudhuri_expansion(sol_resp, dt=dt)
    theta_non = compute_raychaudhuri_expansion(sol_non, dt=dt)
    
    tumor_resp = sol_resp.y[0] + sol_resp.y[1]
    tumor_non = sol_non.y[0] + sol_non.y[1]
    
    df_res = pd.DataFrame({
        "time_days": t_eval,
        "stem_resp": sol_resp.y[0],
        "diff_resp": sol_resp.y[1],
        "norm_resp": sol_resp.y[2],
        "tumor_resp": tumor_resp,
        "cyt_resp": sol_resp.y[3],
        "supp_resp": sol_resp.y[4],
        "theta_resp": theta_resp,
        "stem_non": sol_non.y[0],
        "diff_non": sol_non.y[1],
        "norm_non": sol_non.y[2],
        "tumor_non": tumor_non,
        "cyt_non": sol_non.y[3],
        "supp_non": sol_non.y[4],
        "theta_non": theta_non
    })
    
    csv_path = os.path.join(res_dir, "checkpoint_bifurcation_simulation.csv")
    df_res.to_csv(csv_path, index=False)
    
    bifurcation_day_ray = None
    div_mask = (theta_non - theta_resp > 0.04) & (t_eval >= 15.0)
    if np.any(div_mask):
        bifurcation_day_ray = float(t_eval[np.argmax(div_mask)])
        
    base_tumor = df_res[df_res["time_days"] == 14.0]["tumor_resp"].iloc[0]
    recist_pd_mask = (t_eval >= 14.0) & (tumor_non >= 1.20 * base_tumor)
    recist_pd_day = float(t_eval[np.argmax(recist_pd_mask)]) if np.any(recist_pd_mask) else None
    
    standard_ct_scan_day = 56.0
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(t_eval, tumor_resp, "b-", linewidth=2.5, label="Responder Cohort (Optimal Steering)")
    plt.plot(t_eval, tumor_non, "r--", linewidth=2.5, label="Non-Responder Cohort (Standard Monotherapy)")
    plt.axvspan(14.0, 70.0, color="gray", alpha=0.15, label="Treatment Window (Days 14-70)")
    if recist_pd_day:
        plt.axvline(recist_pd_day, color="orange", linestyle="--", linewidth=1.8, label=f"RECIST PD (+20% Growth, Day {recist_pd_day:.1f})")
    plt.axvline(standard_ct_scan_day, color="darkred", linestyle=":", linewidth=2.0, label=f"Standard Clinical CT Assessment (Day {standard_ct_scan_day:.1f})")
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Total Tumor Burden (Cells)", fontsize=12)
    plt.title("Tumor Trajectory Under Immune/Microenvironment Steering", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    
    plt.subplot(2, 2, 2)
    plt.plot(t_eval, sol_resp.y[3], "b-", linewidth=2.2, label="Cytotoxic T/NK (Responder)")
    plt.plot(t_eval, sol_non.y[3], "r--", linewidth=2.2, label="Cytotoxic T/NK (Non-Responder)")
    plt.plot(t_eval, sol_resp.y[4], "cyan", linestyle="-.", linewidth=1.8, label="Immunosuppressive (Responder)")
    plt.plot(t_eval, sol_non.y[4], "orange", linestyle="-.", linewidth=1.8, label="Immunosuppressive (Non-Responder)")
    plt.axvspan(14.0, 70.0, color="gray", alpha=0.15)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Immune Cell Density", fontsize=12)
    plt.title("Immune Microenvironment Reprogramming", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    
    plt.subplot(2, 2, 3)
    plt.plot(t_eval, theta_resp, "b-", linewidth=2.5, label="Raychaudhuri Theta (Responder: SAT)")
    plt.plot(t_eval, theta_non, "r--", linewidth=2.5, label="Raychaudhuri Theta (Non-Responder: UNSAT)")
    plt.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    plt.axvspan(14.0, 70.0, color="gray", alpha=0.15)
    if bifurcation_day_ray:
        plt.axvline(bifurcation_day_ray, color="green", linewidth=2.0, linestyle="--", label=f"Raychaudhuri Early Triage (Day {bifurcation_day_ray:.1f})")
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Theta = Tr(L_k) (Bundle Expansion Rate)", fontsize=12)
    plt.title("Raychaudhuri Phase Space Geometric Convergence", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=9)
    
    plt.subplot(2, 2, 4)
    if bifurcation_day_ray and recist_pd_day:
        lead_time_pd = recist_pd_day - bifurcation_day_ray
        lead_time_scan = standard_ct_scan_day - bifurcation_day_ray
        labels = ["Raychaudhuri Early Triage", "RECIST PD (+20%)", "Standard CT Scan (6 Weeks)"]
        days = [bifurcation_day_ray, recist_pd_day, standard_ct_scan_day]
        colors = ["green", "orange", "darkred"]
        bars = plt.barh(labels, days, color=colors, height=0.5, alpha=0.8)
        plt.xlabel("Days Post Diagnosis", fontsize=12)
        plt.title(f"Early Triage Advantage: {lead_time_scan:.1f} Days Ahead of Routine CT", fontsize=13)
        for bar in bars:
            w = bar.get_width()
            plt.text(w + 1.0, bar.get_y() + bar.get_height()/2.0, f"Day {w:.1f}", ha="left", va="center", fontsize=11, fontweight="bold")
        plt.xlim(0.0, max(days) + 15.0)
        plt.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plot_path = os.path.join(res_dir, "checkpoint_raychaudhuri_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("Checkpoint Bifurcation Simulation Completed.")
    print(f"Results saved to {csv_path} and {plot_path}")
    if bifurcation_day_ray and recist_pd_day:
        print(f"Raychaudhuri Early Triage Day: {bifurcation_day_ray:.1f}")
        print(f"RECIST PD Confirmation Day: {recist_pd_day:.1f}")
        print(f"Standard Clinical CT Assessment Day: {standard_ct_scan_day:.1f}")
        print(f"Lead Time Advantage Over Routine CT: {standard_ct_scan_day - bifurcation_day_ray:.1f} days ahead.")

if __name__ == "__main__":
    run_checkpoint_simulation()