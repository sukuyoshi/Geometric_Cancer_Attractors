import numpy as np

class TumorMicroenvironmentSystem:
    def __init__(
        self,
        k_tissue=1000.0,
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
    ):
        self.k_tissue = k_tissue
        self.r_stem = r_stem
        self.r_diff = r_diff
        self.r_norm = r_norm
        self.kd0 = kd0
        self.gamma_diff = gamma_diff
        self.kr0 = kr0
        self.d_stem = d_stem
        self.d_diff0 = d_diff0
        self.alpha_cyt = alpha_cyt
        self.beta_mat = beta_mat
        self.eta_inv = eta_inv
        self.delta_stress = delta_stress
        self.s_cyt = s_cyt
        self.rho_cyt = rho_cyt
        self.alpha_ag = alpha_ag
        self.mu_supp = mu_supp
        self.kappa_ici = kappa_ici
        self.mu_metab = mu_metab
        self.d_cyt = d_cyt
        self.s_supp = s_supp
        self.rho_supp = rho_supp
        self.d_supp = d_supp
        self.beta_glyc = beta_glyc
        self.beta_hypox = beta_hypox
        self.d_metab = d_metab
        self.gamma_mat = gamma_mat
        self.d_mat = d_mat

    def vector_field(
        self,
        t,
        state,
        u_diff=0.0,
        u_chemo=0.0,
        u_ici=0.0,
        u_mat=0.0
    ):
        c_stem, c_diff, c_norm, i_cyt, i_supp, sigma_metab, m_mat = state
        
        c_stem = max(0.0, c_stem)
        c_diff = max(0.0, c_diff)
        c_norm = max(0.0, c_norm)
        i_cyt = max(0.01, i_cyt)
        i_supp = max(0.01, i_supp)
        sigma_metab = max(0.001, sigma_metab)
        m_mat = max(0.01, m_mat)
        
        n_total = c_stem + c_diff + c_norm
        space_factor = max(0.0, 1.0 - n_total / self.k_tissue)
        
        k_diff = self.kd0 + self.gamma_diff * u_diff
        k_dediff = self.kr0 * (sigma_metab / (0.8 + sigma_metab))
        
        cyt_eff = self.alpha_cyt * i_cyt / (1.0 + self.beta_mat * m_mat)
        
        dc_stem_dt = (
            self.r_stem * c_stem * space_factor
            - k_diff * c_stem
            + k_dediff * c_diff
            - cyt_eff * c_stem
            - self.d_stem * c_stem
            - 0.001 * u_chemo * c_stem
        )
        
        d_diff = self.d_diff0 * (1.0 + 0.3 * sigma_metab)
        dc_diff_dt = (
            k_diff * c_stem
            - k_dediff * c_diff
            + self.r_diff * c_diff * space_factor
            - cyt_eff * c_diff
            - d_diff * c_diff
            - 0.80 * u_chemo * c_diff
        )
        
        dc_norm_dt = (
            self.r_norm * c_norm * space_factor
            - self.delta_stress * (sigma_metab / (0.5 + sigma_metab)) * c_norm
            - self.eta_inv * (c_stem + c_diff) * c_norm
            - 0.10 * u_chemo * c_norm
        )
        
        tumor_antigen = c_stem + c_diff
        imm_stim = (self.rho_cyt * tumor_antigen / (self.alpha_ag + tumor_antigen)) * i_cyt
        supp_factor = (self.mu_supp * i_supp / (1.0 + self.kappa_ici * u_ici)) * i_cyt
        
        di_cyt_dt = (
            self.s_cyt
            + imm_stim
            - supp_factor
            - self.mu_metab * (sigma_metab / (1.0 + sigma_metab)) * i_cyt
            - self.d_cyt * i_cyt
            - 0.50 * u_chemo * i_cyt
        )
        
        supp_stim = (self.rho_supp * sigma_metab / (0.5 + sigma_metab)) * i_supp
        di_supp_dt = self.s_supp + supp_stim - self.d_supp * i_supp
        
        dsigma_dt = (
            self.beta_glyc * tumor_antigen
            + self.beta_hypox * (m_mat / (1.0 + m_mat))
            - self.d_metab * sigma_metab
        )
        
        dm_mat_dt = (
            self.gamma_mat * tumor_antigen * (sigma_metab / (1.0 + sigma_metab)) * max(0.0, 1.0 - m_mat / 2.5)
            - self.d_mat * m_mat
            - u_mat * m_mat
        )
        
        return np.array([
            dc_stem_dt,
            dc_diff_dt,
            dc_norm_dt,
            di_cyt_dt,
            di_supp_dt,
            dsigma_dt,
            dm_mat_dt
        ])
