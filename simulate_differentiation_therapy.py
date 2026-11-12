import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.tumor_microenvironment_dynamics import TumorMicroenvironmentSystem

def run_differentiation_simulation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    
    system = TumorMicroenvironmentSystem()
    
    t_span = (0.0, 180.0)
    t_eval = np.linspace(0.0, 180.0, 1801)
    
    y0_tumor = [80.0, 450.0, 250.0, 5.0, 15.0, 1.2, 0.8]
    
    def chemo_protocol(t, state):
        if 14.0 <= t <= 70.0:
            u_chemo = 1.8
        else:
            u_chemo = 0.0
        return system.vector_field(
            t, state,
            u_diff=0.0,
            u_chemo=u_chemo,
            u_ici=0.0,
            u_mat=0.0
        )

    def differentiation_protocol(t, state):
        if 14.0 <= t <= 70.0:
            u_diff = 5.0
            u_mat = 2.5
            u_ici = 2.0
        else:
            u_diff = 0.0
            u_mat = 0.0
            u_ici = 0.0
        return system.vector_field(
            t, state,
            u_diff=u_diff,
            u_chemo=0.0,
            u_ici=u_ici,
            u_mat=u_mat
        )

    sol_chemo = solve_ivp(
        chemo_protocol,
        t_span,
        y0_tumor,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    
    sol_diff = solve_ivp(
        differentiation_protocol,
        t_span,
        y0_tumor,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    
    total_tumor_chemo = sol_chemo.y[0] + sol_chemo.y[1]
    total_tumor_diff = sol_diff.y[0] + sol_diff.y[1]
    
    df_res = pd.DataFrame({
        "time_days": sol_diff.t,
        "stem_diff": sol_diff.y[0],
        "diff_diff": sol_diff.y[1],
        "norm_diff": sol_diff.y[2],
        "total_tumor_diff": total_tumor_diff,
        "cyt_diff": sol_diff.y[3],
        "supp_diff": sol_diff.y[4],
        "metab_diff": sol_diff.y[5],
        "mat_diff": sol_diff.y[6],
        "stem_chemo": sol_chemo.y[0],
        "diff_chemo": sol_chemo.y[1],
        "norm_chemo": sol_chemo.y[2],
        "total_tumor_chemo": total_tumor_chemo,
        "cyt_chemo": sol_chemo.y[3],
        "supp_chemo": sol_chemo.y[4],
        "metab_chemo": sol_chemo.y[5],
        "mat_chemo": sol_chemo.y[6]
    })
    
    csv_path = os.path.join(res_dir, "differentiation_therapy_simulation.csv")
    df_res.to_csv(csv_path, index=False)
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(sol_chemo.t, total_tumor_chemo, "r--", linewidth=2.0, label="Cytotoxic Chemotherapy (Relapse)")
    plt.plot(sol_diff.t, total_tumor_diff, "b-", linewidth=2.5, label="Differentiation Pulse Steering (Remission)")
    plt.axvspan(14.0, 70.0, color="gray", alpha=0.15, label="Active Therapy Window (8 Weeks)")
    plt.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Total Tumor Burden (Stem + Diff)", fontsize=12)
    plt.title("Total Tumor Burden: Cytotoxic Selection vs Differentiation", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    
    plt.subplot(2, 2, 2)
    plt.plot(sol_chemo.t, sol_chemo.y[0], "r--", linewidth=2.0, label="Cancer Stem Cells (Chemo)")
    plt.plot(sol_diff.t, sol_diff.y[0], "b-", linewidth=2.5, label="Cancer Stem Cells (Diff Pulse)")
    plt.plot(sol_diff.t, sol_diff.y[1], "c-.", linewidth=1.8, label="Differentiated Cells (Diff Pulse)")
    plt.axvspan(14.0, 70.0, color="gray", alpha=0.15)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Cell Population Density", fontsize=12)
    plt.title("Stem vs Differentiated Pool Kinetics", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    
    plt.subplot(2, 2, 3)
    plt.plot(sol_chemo.y[0], sol_chemo.y[1], "r--", linewidth=2.0, label="Chemotherapy Trajectory")
    plt.plot(sol_diff.y[0], sol_diff.y[1], "b-", linewidth=2.5, label="Differentiation Steering Trajectory")
    plt.scatter([y0_tumor[0]], [y0_tumor[1]], color="black", s=80, zorder=5, label="Initial Malignant State")
    plt.scatter([sol_diff.y[0][-1]], [sol_diff.y[1][-1]], color="blue", marker="*", s=160, zorder=5, label="Tumor-Free Healthy Basin")
    plt.scatter([sol_chemo.y[0][-1]], [sol_chemo.y[1][-1]], color="red", marker="x", s=100, zorder=5, label="Relapsed Malignant Attractor")
    plt.xlabel("Cancer Stem Cells C_stem", fontsize=12)
    plt.ylabel("Differentiated Tumor Cells C_diff", fontsize=12)
    plt.title("Phase-Space Attractor Escape Trajectory", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    
    plt.subplot(2, 2, 4)
    storage_chemo = 0.5 * (sol_chemo.y[0]**2 + sol_chemo.y[1]**2) / 1000.0
    storage_diff = 0.5 * (sol_diff.y[0]**2 + sol_diff.y[1]**2) / 1000.0
    plt.plot(sol_chemo.t, storage_chemo, "r--", linewidth=2.0, label="Storage S(X) (Chemotherapy)")
    plt.plot(sol_diff.t, storage_diff, "b-", linewidth=2.5, label="Storage S(X) (Differentiation Steering)")
    plt.axvspan(14.0, 70.0, color="gray", alpha=0.15)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Willems Tumor Storage Energy S(X)", fontsize=12)
    plt.title("Dissipative Tumor Energy Decay", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(res_dir, "differentiation_therapy_phase_portrait.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("Differentiation Therapy Simulation Completed.")
    print(f"Results saved to {csv_path} and {plot_path}")

if __name__ == "__main__":
    run_differentiation_simulation()
