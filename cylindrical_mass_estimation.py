'''
"""
=============================================================================
DELTA MASS ESTIMATOR AT THE BASE OF A CYLINDER
Using the Wahr-like cylindrical harmonic inversion formula
=============================================================================

PHYSICAL SETUP
--------------
A cylinder of radius R*, height L, scaling parameter α.
A thin mass layer Δσ(ρ,φ) [kg/m²] sits at the BASE z = 0.

The gravitational potential INSIDE the cylinder (0 ≤ ρ ≤ R*, 0 ≤ z ≤ L)
is represented by the expansion:

    U(ρ,φ,z) = Σ_{m,n} J_m(k_mn ρ) e^{-k_mn z}
                         [A_mn cos(mφ) + B_mn sin(mφ)]

    where  k_mn = j_{mn} / (α R*)   and  j_{mn} = n-th zero of J_m

INVERSION PIPELINE
------------------
Given: gravity measurements (potential U or acceleration g) at N_pts
       inside the cylinder.

Step 1 — FIT cylindrical harmonic coefficients {A_mn, B_mn} from data
         via weighted least squares:
            min || W (A x - b) ||²
         where A is the design matrix of basis functions evaluated at
         measurement points, b is the data vector.

Step 2 — APPLY the Wahr-like inversion formula:
            Δσ(ρ,φ) = - 1/(2πG α R*) Σ j_mn J_m(k_mn ρ)
                                         [A_mn cos(mφ) + B_mn sin(mφ)]

Step 3 — INTEGRATE over the disk to get total DELTA MASS:
            ΔM = ∫∫ Δσ(ρ,φ) ρ dρ dφ
               = - 1/(2πG α R*) Σ j_mn A_{0n} ∫₀^{R*} J_0(k_0n ρ) ρ dρ · 2π
         (only m=0 terms survive the φ integration)

         Using ∫₀^{R*} J_0(k_0n ρ) ρ dρ = R*/k_0n · J_1(j_0n/α):
            ΔM = - 1/(G α) Σ_n J_1(j_{0n}/α) / j_{0n} · A_{0n} · R*²

MATHEMATICAL DERIVATION NOTES
------------------------------
The key steps are:
  (a) Neumann BC:    ∂U/∂z|_{z=0} = 2πG Δσ  →  A_mn = -2πG α R*/j_mn · C^σ_mn
  (b) Inversion:     C^σ_mn = -j_mn/(2πG α R*) · A_mn
  (c) φ-integration: ∫₀^{2π} cos(mφ) dφ = 2π δ_{m0}  →  only m=0 survives
  (d) ρ-integration: ∫₀^{R*} J_0(k_0n ρ) ρ dρ = (R*)²/j_{0n} · J_1(j_{0n}/α) / α
      [from ∫ x J_0(ax) dx = x/a J_1(ax)]

INPUTS (configurable at bottom of file)
---------
  - Cylinder geometry: R*, L, α
  - Measurement points: auto-generated or user-supplied
  - Which data type: potential only / acceleration only / both (augmented LS)
  - True Δσ (optional, for validation when ground truth is known)

OUTPUTS
-------
  - Fitted A_mn, B_mn coefficients
  - Δσ(ρ,φ) map on the base disk
  - Total ΔM [kg]
  - Uncertainty map σ_{Δσ}(ρ,φ) from covariance propagation
  - Diagnostic plots
=============================================================================
"""

import numpy as np
from scipy.special import jv as BesselJ, jn_zeros
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────
# MATPLOTLIB SETTINGS
# ─────────────────────────────────────────────────────────────────────────
mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 13
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["axes.titlesize"] = 13
mpl.rcParams["legend.fontsize"] = 11
mpl.rcParams["xtick.labelsize"] = 11
mpl.rcParams["ytick.labelsize"] = 11
COLORS = ["#E6001A", "#F08C00", "#0077BB", "#2c7bb6", "#009933", "#7B2D8B"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLORS)

# ─────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────
G = 6.674e-11  # m³ kg⁻¹ s⁻²


# ═════════════════════════════════════════════════════════════════════════
# SECTION 1 — CYLINDER BASIS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════


class CylinderBasis:
    """
    Precomputes and evaluates the cylindrical harmonic basis:
        φ^C_mn(ρ,φ,z) = J_m(k_mn ρ) exp(-k_mn z) cos(mφ)
        φ^S_mn(ρ,φ,z) = J_m(k_mn ρ) exp(-k_mn z) sin(mφ)

    and all their spatial derivatives needed for the gravity signal.

    Parameters
    ----------
    R_star : float — cylinder radius [m]
    alpha  : float — scaling parameter (k_mn = j_mn / (alpha * R_star))
    m_max  : int   — max azimuthal order (0 … m_max-1)
    n_max  : int   — max radial index (1 … n_max)
    """

    def __init__(self, R_star, alpha, m_max, n_max):
        self.R_star = R_star
        self.alpha = alpha
        self.R_alpha = alpha * R_star
        self.m_max = m_max
        self.n_max = n_max
        self.n_params = 2 * m_max * n_max  # A_mn and B_mn for each (m,n)

        # Pre-cache Bessel zeros: zeros[m][n-1] = j_{m,n}
        self.zeros = {}
        for m in range(m_max):
            self.zeros[m] = jn_zeros(m, n_max)

    def j(self, m, n):
        """j_{m,n} — n-th positive zero of J_m."""
        return self.zeros[m][n - 1]

    def k(self, m, n):
        """Eigenwavenumber k_{mn} = j_{mn} / (alpha * R*)."""
        return self.j(m, n) / self.R_alpha

    def mode_index(self, m, n):
        """
        Map (m, n) pair to column index in the design matrix.
        Column 2*(m*n_max + n-1)   → A_mn coefficient
        Column 2*(m*n_max + n-1)+1 → B_mn coefficient
        """
        return 2 * (m * self.n_max + (n - 1))

    def potential_row(self, rho, phi, z):
        """
        Design matrix row for POTENTIAL at point (ρ,φ,z).
        U = Σ_{m,n} J_m(k_mn ρ) exp(-k_mn z) [A_mn cos(mφ) + B_mn sin(mφ)]
        Returns: array of shape (n_params,)
        """
        row = np.zeros(self.n_params)
        for m in range(self.m_max):
            for n in range(1, self.n_max + 1):
                kmn = self.k(m, n)
                base = BesselJ(m, kmn * rho) * np.exp(-kmn * z)
                idx = self.mode_index(m, n)
                row[idx] = base * np.cos(m * phi)
                row[idx + 1] = base * np.sin(m * phi)
        return row

    def acceleration_rows(self, rho, phi, z):
        """
        Three design matrix rows for ACCELERATION components
        g = -∇U at point (ρ,φ,z), in cylindrical coordinates (g_ρ, g_φ, g_z).

        MATH:
          g_ρ = -∂U/∂ρ  = -Σ k_mn J_m'(k_mn ρ) exp(-k_mn z) [A cos + B sin]
          g_φ = -(1/ρ)∂U/∂φ = Σ m/ρ J_m(k_mn ρ) exp(-k_mn z) [A sin - B cos]
          g_z = -∂U/∂z  =  Σ k_mn J_m(k_mn ρ) exp(-k_mn z) [A cos + B sin]
        where J_m'(x) = dJ_m/dx = (J_{m-1}(x) - J_{m+1}(x)) / 2
        Returns: array of shape (3, n_params)
        """
        rows = np.zeros((3, self.n_params))
        for m in range(self.m_max):
            for n in range(1, self.n_max + 1):
                kmn = self.k(m, n)
                x = kmn * rho
                Jm = BesselJ(m, x)
                dJm = 0.5 * (BesselJ(m - 1, x) - BesselJ(m + 1, x))  # J_m'(x)
                Ez = np.exp(-kmn * z)
                cp = np.cos(m * phi)
                sp = np.sin(m * phi)
                idx = self.mode_index(m, n)

                # g_ρ = -∂U/∂ρ = -k_mn * J_m'(k_mn ρ) * exp(-k_mn z) * [A cos + B sin]
                rows[0, idx] = -kmn * dJm * Ez * cp
                rows[0, idx + 1] = -kmn * dJm * Ez * sp

                # g_φ = -(1/ρ) ∂U/∂φ = m/ρ * J_m * exp(-k_mn z) * [A sin - B cos]
                rho_safe = rho if rho > 1e-12 else 1e-12
                rows[1, idx] = m / rho_safe * Jm * Ez * sp
                rows[1, idx + 1] = -m / rho_safe * Jm * Ez * cp

                # g_z = -∂U/∂z = k_mn * J_m * exp(-k_mn z) * [A cos + B sin]
                rows[2, idx] = kmn * Jm * Ez * cp
                rows[2, idx + 1] = kmn * Jm * Ez * sp

        return rows


# ═════════════════════════════════════════════════════════════════════════
# SECTION 2 — SYNTHETIC DATA GENERATOR
# Generates U and g from a KNOWN Δσ(ρ,φ), for testing the full pipeline.
# ═════════════════════════════════════════════════════════════════════════


def sigma_to_potential_coefficients(
    sigma_fn, basis: CylinderBasis, num_rho=500, num_phi=600
):
    """
    Compute potential coefficients {A_mn, B_mn} from a known surface mass
    function Δσ(ρ,φ) at z=0, using the FORWARD Wahr formula:

        A_mn = -(2πG α R*) / j_mn · C^σ_mn
        C^σ_mn = 1/(π_m N²_mn) ∫∫ Δσ J_m(k_mn ρ) cos(mφ) ρ dρ dφ

    MATH NOTE on π_m:
        π_m = 2π for m=0, π for m≥1
        This is the direct azimuthal normalization ∫ cos²(mφ) dφ.
        NOT the Neumann factor ε_m·π (which is the OPPOSITE).
    """
    R = basis.R_star
    rho_g = np.linspace(0, R, num_rho + 1)[1:]
    phi_g = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)
    drho = R / num_rho
    dphi = 2 * np.pi / num_phi
    RHO, PHI = np.meshgrid(rho_g, phi_g, indexing="ij")
    dA = RHO * drho * dphi  # area element ρ dρ dφ

    sig = sigma_fn(RHO, PHI)

    coeffs = np.zeros(basis.n_params)
    for m in range(basis.m_max):
        pi_m = 2 * np.pi if m == 0 else np.pi  # azimuthal norm
        for n in range(1, basis.n_max + 1):
            kmn = basis.k(m, n)
            jmn = basis.j(m, n)
            N2 = 0.5 * R**2 * BesselJ(m + 1, jmn / basis.alpha) ** 2
            bess = BesselJ(m, kmn * RHO)
            pref = 1.0 / (pi_m * N2)

            C_mn = pref * np.sum(sig * bess * np.cos(m * PHI) * dA)
            S_mn = pref * np.sum(sig * bess * np.sin(m * PHI) * dA) if m > 0 else 0.0

            # Forward Wahr formula: A_mn = -2πG α R* / j_mn · C^σ_mn
            fac = -2 * np.pi * G * basis.R_alpha / jmn
            idx = basis.mode_index(m, n)
            coeffs[idx] = fac * C_mn
            coeffs[idx + 1] = fac * S_mn

    return coeffs


def generate_synthetic_data(
    basis: CylinderBasis,
    sigma_fn,
    points_cyl,  # (N, 3): (ρ, φ, z) in cylinder frame
    noise_sigma_pot=0.0,
    noise_sigma_acc=0.0,
    use_potential=True,
    use_acceleration=True,
):
    """
    Generate synthetic potential + acceleration from a known Δσ.

    Returns
    -------
    A_des : design matrix (n_data × n_params)
    b_obs : observation vector (n_data,)
    true_coeffs : array (n_params,) — ground-truth coefficients
    """
    np.random.seed(42)
    true_coeffs = sigma_to_potential_coefficients(sigma_fn, basis)

    rho_pts = points_cyl[:, 0]
    phi_pts = points_cyl[:, 1]
    z_pts = points_cyl[:, 2]
    N = len(rho_pts)

    rows_list = []
    b_list = []

    if use_potential:
        A_pot = np.zeros((N, basis.n_params))
        for i in range(N):
            A_pot[i] = basis.potential_row(rho_pts[i], phi_pts[i], z_pts[i])
        b_pot = A_pot @ true_coeffs
        if noise_sigma_pot > 0:
            b_pot += np.random.randn(N) * noise_sigma_pot
        rows_list.append(A_pot)
        b_list.append(b_pot)

    if use_acceleration:
        A_acc = np.zeros((3 * N, basis.n_params))
        b_acc_clean = np.zeros(3 * N)
        for i in range(N):
            rows = basis.acceleration_rows(rho_pts[i], phi_pts[i], z_pts[i])
            A_acc[3 * i : 3 * i + 3] = rows
        b_acc_clean = A_acc @ true_coeffs
        if noise_sigma_acc > 0:
            b_acc_clean += np.random.randn(3 * N) * noise_sigma_acc
        rows_list.append(A_acc)
        b_list.append(b_acc_clean)

    A_des = np.vstack(rows_list)
    b_obs = np.hstack(b_list)

    return A_des, b_obs, true_coeffs


# ═════════════════════════════════════════════════════════════════════════
# SECTION 3 — LEAST SQUARES FITTING
# ═════════════════════════════════════════════════════════════════════════


def fit_coefficients(A_des, b_obs, rcond=None, return_covariance=True):
    """
    Solve the least squares problem:
        min || A x - b ||²

    Returns
    -------
    coeffs      : fitted {A_mn, B_mn}  (n_params,)
    residuals   : b - A @ coeffs
    cov_coeffs  : covariance matrix (n_params × n_params) or None
    """
    coeffs, res, rank, sv = lstsq(A_des, b_obs, cond=rcond)

    residuals = b_obs - A_des @ coeffs

    cov = None
    if return_covariance:
        # Estimate noise variance from residuals
        n_data, n_par = A_des.shape
        dof = max(n_data - n_par, 1)
        sigma2 = np.dot(residuals, residuals) / dof

        # C = sigma² (AᵀA)⁻¹  via SVD for numerical stability
        # A = U S Vᵀ  →  (AᵀA)⁻¹ = V S⁻² Vᵀ
        _, s, Vt = np.linalg.svd(A_des, full_matrices=False)
        tol = rcond if rcond else sv[0] * max(A_des.shape) * np.finfo(float).eps
        inv_s2 = np.where(s > tol, 1.0 / s**2, 0.0)
        cov = sigma2 * (Vt.T * inv_s2) @ Vt

    return coeffs, residuals, cov


# ═════════════════════════════════════════════════════════════════════════
# SECTION 4 — WAHR-LIKE INVERSION: coefficients → Δσ map
# ═════════════════════════════════════════════════════════════════════════


def invert_to_surface_mass(
    coeffs, basis: CylinderBasis, rho_grid, phi_grid, cov_coeffs=None
):
    """
    Apply the Wahr-like inversion formula:

        Δσ(ρ,φ) = -1/(2πG α R*) Σ_{m,n} j_mn J_m(k_mn ρ)
                                          [A_mn cos(mφ) + B_mn sin(mφ)]

    MATH:
        Derived from: -k_mn A_mn = 2πG C^σ_mn  (Neumann BC at z=0)
        → C^σ_mn = -j_mn/(2πG α R*) A_mn
        Reconstruction: Δσ = Σ C^σ_mn J_m(k_mn ρ) cos(mφ)

    Parameters
    ----------
    rho_grid, phi_grid : 2D arrays — evaluation grid on z=0 disk
    cov_coeffs : covariance of fitted coefficients (for uncertainty propagation)

    Returns
    -------
    sigma_map  : Δσ(ρ,φ) in [kg/m²]
    sigma_std  : 1σ uncertainty map [kg/m²] (None if cov_coeffs not given)
    """
    prefac = -1.0 / (2.0 * np.pi * G * basis.R_alpha)
    sigma = np.zeros_like(rho_grid, dtype=float)

    # Jacobian: ∂(Δσ)/∂A_mn  — needed for uncertainty propagation
    if cov_coeffs is not None:
        J_mat = np.zeros((rho_grid.size, basis.n_params))

    for m in range(basis.m_max):
        for n in range(1, basis.n_max + 1):
            jmn = basis.j(m, n)
            kmn = basis.k(m, n)
            bess = BesselJ(m, kmn * rho_grid)
            idx = basis.mode_index(m, n)

            A_mn = coeffs[idx]
            B_mn = coeffs[idx + 1]
            cp = np.cos(m * phi_grid)
            sp = np.sin(m * phi_grid)

            contrib = prefac * jmn * bess * (A_mn * cp + B_mn * sp)
            sigma += contrib

            if cov_coeffs is not None:
                # ∂(Δσ)/∂A_mn = prefac * j_mn * J_m(k_mn ρ) cos(mφ)
                # ∂(Δσ)/∂B_mn = prefac * j_mn * J_m(k_mn ρ) sin(mφ)
                flat = rho_grid.flatten()
                J_mat[:, idx] = (
                    prefac
                    * jmn
                    * BesselJ(m, kmn * flat)
                    * np.cos(m * phi_grid.flatten())
                )
                J_mat[:, idx + 1] = (
                    prefac
                    * jmn
                    * BesselJ(m, kmn * flat)
                    * np.sin(m * phi_grid.flatten())
                )

    sigma_std = None
    if cov_coeffs is not None:
        # Var(Δσ) = J · Cov(coeffs) · Jᵀ  — diagonal only (pointwise variance)
        var_flat = np.einsum("ij,jk,ik->i", J_mat, cov_coeffs, J_mat)
        sigma_std = np.sqrt(np.abs(var_flat)).reshape(rho_grid.shape)

    return sigma, sigma_std


# ═════════════════════════════════════════════════════════════════════════
# SECTION 5 — TOTAL DELTA MASS
# ═════════════════════════════════════════════════════════════════════════


def compute_total_mass(coeffs, basis: CylinderBasis, cov_coeffs=None):
    """
    Integrate Δσ over the disk to get total ΔM [kg]:

        ΔM = ∫₀^{2π} ∫₀^{R*} Δσ(ρ,φ) ρ dρ dφ

    MATH:
        φ-integral: ∫₀^{2π} cos(mφ) dφ = 2π δ_{m0}
        → only m=0 terms survive!

        ρ-integral for m=0:
          ∫₀^{R*} J_0(k_0n ρ) ρ dρ = [x/k · J_1(k x)]₀^{R*} / k
                                     = R*/k_0n · J_1(k_0n R*)
                                     = R*² / j_{0n} · J_1(j_{0n}/α)  [since k=j/αR*]

        Therefore:
          ΔM = -1/(2πG α R*) · 2π · Σ_n j_0n · A_{0n}
                                           · R*²/j_0n · J_1(j_0n/α)
             = -R*²/(G α) · Σ_n J_1(j_{0n}/α) · A_{0n}

        Analogy with Wahr total mass (GRACE):
          ΔM_Wahr = a² ρ̄_E / (3 ρ̄_w) · 4π · Σ_n (2n+1)/(1+k_n) · ΔC̄_n0 / √(2n+1)
          The (2n+1) weight ↔ j_mn; the Legendre polynomials integrate to
          a known closed form just as J_0 integrates analytically here.

    Returns
    -------
    delta_M      : total mass change [kg]
    delta_M_std  : 1σ uncertainty [kg] (if cov_coeffs provided)
    contribution : array of per-mode contributions Σ_n ΔM_{0n}
    """
    R = basis.R_star
    al = basis.alpha

    delta_M = 0.0
    per_mode = []
    grad_M = np.zeros(basis.n_params)  # ∂(ΔM)/∂coeffs

    for n in range(1, basis.n_max + 1):
        j0n = basis.j(0, n)
        J1 = BesselJ(1, j0n / al)
        # Radial integral: ∫₀^{R*} J_0(k_0n ρ) ρ dρ = R*²/j_{0n} · J_1(j_0n/α)
        rho_integral = R**2 / j0n * J1

        # ΔM contribution from mode (m=0, n):
        # ΔM_{0n} = -1/(2πG α R*) · 2π · j_0n · A_{0n} · rho_integral
        #         = -R*² / (G α) · J_1(j_0n/α) · A_{0n}
        prefac = -R / (G * al)  # correct: -R*/(G*alpha), see derivation
        idx = basis.mode_index(0, n)
        A0n = coeffs[idx]
        dm = prefac * J1 * A0n
        delta_M += dm
        per_mode.append(dm)
        grad_M[idx] = prefac * J1  # ∂(ΔM)/∂A_{0n} = -(R*/(G*alpha)) * J_1(j_0n/alpha)

    delta_M_std = None
    if cov_coeffs is not None:
        # Var(ΔM) = grad_M · Cov · grad_M^T
        var_M = grad_M @ cov_coeffs @ grad_M
        delta_M_std = np.sqrt(abs(var_M))

    return delta_M, delta_M_std, np.array(per_mode)


def compute_total_mass_numerical(sigma_map, rho_grid, phi_grid):
    """
    Cross-check: integrate the reconstructed sigma map numerically.

    ΔM = ∫∫ Δσ(ρ,φ) ρ dρ dφ  ≈  Σ_{i,j} Δσ(ρ_i,φ_j) ρ_i Δρ Δφ

    This is independent of the Bessel series convergence and provides
    a direct check on the reconstructed sigma map quality.
    """
    drho = np.abs(rho_grid[1, 0] - rho_grid[0, 0]) if rho_grid.shape[0] > 1 else 0
    dphi = np.abs(phi_grid[0, 1] - phi_grid[0, 0]) if phi_grid.shape[1] > 1 else 0
    dA = rho_grid * drho * dphi
    return np.sum(sigma_map * dA)


# ═════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTTING
# ═════════════════════════════════════════════════════════════════════════


def plot_results(
    basis,
    sigma_true_map,
    sigma_est_map,
    sigma_std_map,
    rho_grid,
    phi_grid,
    coeffs_true,
    coeffs_fit,
    per_mode_mass,
    delta_M,
    delta_M_std,
    delta_M_true=None,
    residuals=None,
):
    """6-panel diagnostic figure."""

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.38)

    X = rho_grid * np.cos(phi_grid)
    Y = rho_grid * np.sin(phi_grid)
    theta_c = np.linspace(0, 2 * np.pi, 300)
    scale = (
        np.max(np.abs(sigma_true_map))
        if sigma_true_map is not None
        else np.max(np.abs(sigma_est_map))
    )

    # ── Panel 1: True sigma (if known) or estimated sigma ────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if sigma_true_map is not None:
        data1 = sigma_true_map / scale
        title1 = "True surface mass Delta_sigma [normalized]"
    else:
        data1 = sigma_est_map / scale
        title1 = "Estimated Delta_sigma (no ground truth)"
    c1 = ax1.pcolormesh(
        X, Y, data1, cmap="RdBu_r", vmin=-1.3, vmax=1.3, shading="gouraud"
    )
    fig.colorbar(c1, ax=ax1)
    ax1.plot(
        basis.R_star * np.cos(theta_c),
        basis.R_star * np.sin(theta_c),
        "k--",
        lw=1,
        alpha=0.6,
    )
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title(title1)

    # ── Panel 2: Estimated sigma ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    c2 = ax2.pcolormesh(
        X,
        Y,
        sigma_est_map / scale,
        cmap="RdBu_r",
        vmin=-1.3,
        vmax=1.3,
        shading="gouraud",
    )
    fig.colorbar(c2, ax=ax2)
    ax2.plot(
        basis.R_star * np.cos(theta_c),
        basis.R_star * np.sin(theta_c),
        "k--",
        lw=1,
        alpha=0.6,
    )
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Estimated Delta_sigma via Wahr-like formula")

    # ── Panel 3: Uncertainty OR error map ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if sigma_true_map is not None:
        err_map = np.abs(sigma_est_map - sigma_true_map) / scale
        c3 = ax3.pcolormesh(
            X,
            Y,
            np.log10(err_map + 1e-8),
            cmap="hot_r",
            vmin=-5,
            vmax=0,
            shading="gouraud",
        )
        fig.colorbar(c3, ax=ax3, label="log10 relative error")
        ax3.set_title("log10 |estimated - true| / max|true|")
    elif sigma_std_map is not None:
        c3 = ax3.pcolormesh(
            X, Y, sigma_std_map / scale, cmap="Oranges", shading="gouraud"
        )
        fig.colorbar(c3, ax=ax3, label="1-sigma / max|sigma|")
        ax3.set_title("1-sigma uncertainty map (normalized)")
    ax3.plot(
        basis.R_star * np.cos(theta_c),
        basis.R_star * np.sin(theta_c),
        "w--",
        lw=1,
        alpha=0.7,
    )
    ax3.set_aspect("equal")
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")

    # ── Panel 4: Coefficient comparison ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    n_show = min(40, len(coeffs_fit))
    ax4.bar(
        np.arange(n_show) - 0.2,
        np.abs(coeffs_true[:n_show]) if coeffs_true is not None else np.zeros(n_show),
        0.4,
        label="True",
        color=COLORS[2],
        alpha=0.8,
    )
    ax4.bar(
        np.arange(n_show) + 0.2,
        np.abs(coeffs_fit[:n_show]),
        0.4,
        label="Fitted",
        color=COLORS[0],
        alpha=0.8,
    )
    ax4.set_yscale("log")
    ax4.set_xlabel("Coefficient index")
    ax4.set_ylabel("|A_mn| or |B_mn|")
    ax4.set_title("Fitted vs true potential coefficients")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Per-mode mass contribution ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    n_modes = len(per_mode_mass)
    n_idx = np.arange(1, n_modes + 1)
    colors_bar = [COLORS[2] if dm >= 0 else COLORS[0] for dm in per_mode_mass]
    ax5.bar(n_idx, per_mode_mass, color=colors_bar, alpha=0.85)
    ax5.axhline(0, color="k", lw=0.8)
    ax5.set_xlabel("Radial index n  (m=0 modes)")
    ax5.set_ylabel("Delta_M contribution [kg]")
    ax5.set_title("Per-mode mass at base (m=0 terms only)")
    ax5.grid(True, alpha=0.3)
    total_str = f"Total DeltaM = {delta_M:.4e} kg"
    if delta_M_std:
        total_str += f"\n+/- {delta_M_std:.2e} kg"
    if delta_M_true is not None:
        total_str += f"\nTrue DeltaM = {delta_M_true:.4e} kg"
        total_str += f"\nError = {abs(delta_M-delta_M_true)/abs(delta_M_true)*100:.2f}%"
    ax5.text(
        0.05,
        0.95,
        total_str,
        transform=ax5.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(fc="lightyellow", ec="gray", alpha=0.9),
    )

    # ── Panel 6: LS residuals ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if residuals is not None:
        ax6.plot(np.abs(residuals), ".", ms=2, alpha=0.5, color=COLORS[2])
        ax6.set_yscale("log")
        ax6.set_xlabel("Observation index")
        ax6.set_ylabel("|residual|")
        rms = np.sqrt(np.mean(residuals**2))
        ax6.set_title(f"LS residuals  (RMS = {rms:.3e})")
        ax6.axhline(rms, color=COLORS[0], lw=1.5, ls="--", label=f"RMS={rms:.2e}")
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)

    fig.suptitle(
        "Delta Mass Estimation at Cylinder Base\n"
        "Wahr-like inversion:  D_sigma = -1/(2piG*alpha*R*) "
        "sum_mn j_mn J_m(k_mn*rho) [A cos + B sin]",
        fontsize=12,
    )

    # #plt.savefig('/mnt/user-data/outputs/delta_mass_estimation.png',
    #            dpi=150, bbox_inches='tight')
    # #plt.savefig('/mnt/user-data/outputs/delta_mass_estimation.pdf',
    #            dpi=150, bbox_inches='tight')
    # print("Figure saved.")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════


def run_pipeline(
    # ── Cylinder geometry ──────────────────────────────────────────────
    R_star=0.1,  # cylinder radius [m]
    L=0.5,  # cylinder height [m]
    alpha=1.0,  # scaling parameter α
    # ── Basis truncation ───────────────────────────────────────────────
    m_max=6,  # azimuthal orders 0 … m_max-1
    n_max=6,  # radial indices 1 … n_max
    # ── Measurement points ─────────────────────────────────────────────
    N_pts=500,  # number of random interior points
    # ── Noise levels ───────────────────────────────────────────────────
    noise_potential=0.0,  # [m²/s²] — set > 0 for realistic scenario
    noise_acceleration=0.0,  # [m/s²]
    # ── Which data to use for fitting ──────────────────────────────────
    use_potential=True,
    use_acceleration=True,
    # ── True surface mass (set to None to invert from real data) ───────
    # Any callable sigma(rho, phi) -> kg/m²
    sigma_fn=None,
    # ── Evaluation grid on disk ────────────────────────────────────────
    n_rho_grid=80,
    n_phi_grid=160,
    verbose=True,
):
    """
    Full pipeline: generate data → fit coefficients → invert → ΔM.

    Returns
    -------
    dict with keys: delta_M, delta_M_std, delta_M_true,
                    sigma_map, sigma_std_map, coeffs_fit, basis
    """
    if verbose:
        print("=" * 65)
        print("DELTA MASS ESTIMATOR — CYLINDRICAL HARMONIC INVERSION")
        print("=" * 65)
        print(f"  Cylinder:  R* = {R_star} m,  L = {L} m,  alpha = {alpha}")
        print(f"  Basis:     m in [0,{m_max-1}],  n in [1,{n_max}]")
        print(f"  N_params:  {2*m_max*n_max}")
        print(f"  N_pts:     {N_pts}")
        print()

    # ── Default test mass: paraboloid + dipole ─────────────────────────
    if sigma_fn is None:
        SIGMA0 = 1e6  # kg/m²

        def sigma_fn(rho, phi):
            """
            Default test mass at z=0:
              σ(ρ,φ) = σ₀ (1-(ρ/R*)²)(1 + 0.4 cos(φ) + 0.2 sin(2φ))
            Paraboloid radial profile with m=1,2 azimuthal variation.
            Vanishes at ρ=R* (consistent with Bessel BC for α=1).
            """
            radial = 1.0 - (rho / R_star) ** 2
            angular = 1.0 + 0.4 * np.cos(phi) + 0.2 * np.sin(2 * phi)
            return SIGMA0 * radial * angular

    # ── True total mass (numerical) ────────────────────────────────────
    rho_int = np.linspace(0, R_star, 1000)[1:]
    phi_int = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    RI, PI = np.meshgrid(rho_int, phi_int, indexing="ij")
    delta_M_true = np.sum(sigma_fn(RI, PI) * RI) * (R_star / 1000) * (2 * np.pi / 800)
    if verbose:
        print(f"  True total ΔM = {delta_M_true:.6e} kg")

    # ── Build basis ────────────────────────────────────────────────────
    basis = CylinderBasis(R_star, alpha, m_max, n_max)

    # ── Generate random measurement points inside cylinder ────────────
    np.random.seed(1)
    rho_pts = np.sqrt(np.random.uniform(0, R_star**2, N_pts))  # uniform in area
    phi_pts = np.random.uniform(0, 2 * np.pi, N_pts)
    z_pts = np.random.uniform(0.01 * L, L, N_pts)  # avoid z=0 exactly
    points = np.column_stack([rho_pts, phi_pts, z_pts])

    # ── Generate synthetic observations from true sigma ────────────────
    if verbose:
        print("  Generating synthetic observations ...")
    A_des, b_obs, true_coeffs = generate_synthetic_data(
        basis,
        sigma_fn,
        points,
        noise_sigma_pot=noise_potential,
        noise_sigma_acc=noise_acceleration,
        use_potential=use_potential,
        use_acceleration=use_acceleration,
    )
    if verbose:
        print(f"  Design matrix shape: {A_des.shape}")

    # ── Least squares fit ─────────────────────────────────────────────
    if verbose:
        print("  Fitting cylindrical harmonic coefficients ...")
    coeffs_fit, residuals, cov = fit_coefficients(A_des, b_obs)
    rms_res = np.sqrt(np.mean(residuals**2))
    if verbose:
        coeff_err = np.max(np.abs(coeffs_fit - true_coeffs)) / (
            np.max(np.abs(true_coeffs)) + 1e-30
        )
        print(f"  Fit RMS residual:    {rms_res:.3e}")
        print(f"  Max coeff rel error: {coeff_err:.3e}")

    # ── Wahr-like inversion → Δσ map ──────────────────────────────────
    if verbose:
        print("  Inverting to surface mass map ...")
    rho_1d = np.linspace(0.005 * R_star, R_star * 0.99, n_rho_grid)
    phi_1d = np.linspace(0, 2 * np.pi, n_phi_grid, endpoint=False)
    RHO, PHI = np.meshgrid(rho_1d, phi_1d, indexing="ij")

    sigma_est, sigma_std = invert_to_surface_mass(coeffs_fit, basis, RHO, PHI, cov)
    sigma_true_map = sigma_fn(RHO, PHI)

    rms_sigma = np.sqrt(np.mean((sigma_est - sigma_true_map) ** 2))
    scale = np.max(np.abs(sigma_true_map))
    if verbose:
        print(
            f"  Sigma RMS error:  {rms_sigma:.3e} kg/m²  "
            f"({rms_sigma/scale*100:.2f}% of peak)"
        )

    # ── Integrate to total ΔM ─────────────────────────────────────────
    if verbose:
        print("  Computing total delta mass ...")
    delta_M, delta_M_std, per_mode = compute_total_mass(coeffs_fit, basis, cov)
    delta_M_numerical = compute_total_mass_numerical(sigma_est, RHO, PHI)
    if verbose:
        print()
        print("─" * 50)
        print(f"  RESULT:  ΔM (Wahr formula)   = {delta_M:.6e} kg")
        print(f"           ΔM (numerical grid)  = {delta_M_numerical:.6e} kg")
        if delta_M_std:
            print(f"           uncertainty         = {delta_M_std:.2e} kg  (1-sigma)")
        print(f"           ΔM (true)            = {delta_M_true:.6e} kg")
        err_f = abs(delta_M - delta_M_true) / abs(delta_M_true) * 100
        err_n = abs(delta_M_numerical - delta_M_true) / abs(delta_M_true) * 100
        print(f"           Error (formula)      = {err_f:.3f}%")
        print(f"           Error (numerical)    = {err_n:.3f}%")
        print("─" * 50)

    # ── Plot ──────────────────────────────────────────────────────────
    plot_results(
        basis,
        sigma_true_map,
        sigma_est,
        sigma_std,
        RHO,
        PHI,
        coeffs_true=true_coeffs,
        coeffs_fit=coeffs_fit,
        per_mode_mass=per_mode,
        delta_M=delta_M,
        delta_M_std=delta_M_std,
        delta_M_true=delta_M_true,
        residuals=residuals,
    )

    return dict(
        delta_M=delta_M,
        delta_M_std=delta_M_std,
        delta_M_true=delta_M_true,
        sigma_map=sigma_est,
        sigma_std_map=sigma_std,
        sigma_true_map=sigma_true_map,
        coeffs_fit=coeffs_fit,
        coeffs_true=true_coeffs,
        basis=basis,
        per_mode_mass=per_mode,
        residuals=residuals,
    )


# ═════════════════════════════════════════════════════════════════════════
# SECTION 8 — NOISE + TRUNCATION SENSITIVITY STUDY
# ═════════════════════════════════════════════════════════════════════════


def sensitivity_study(R_star=0.1, alpha=1.0, L=0.5, sigma_fn=None, N_pts=500):
    """
    Sweep noise levels and truncation orders, reporting ΔM error.
    Useful for choosing m_max, n_max for a given measurement quality.
    """
    if sigma_fn is None:
        S0 = 1e6

        def sigma_fn(rho, phi):
            return S0 * (1 - (rho / R_star) ** 2)

    # True mass
    rho_int = np.linspace(0, R_star, 800)[1:]
    phi_int = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    RI, PI = np.meshgrid(rho_int, phi_int, indexing="ij")
    dM_true = np.sum(sigma_fn(RI, PI) * RI) * (R_star / 800) * (2 * np.pi / 4)

    print("\nSENSITIVITY STUDY")
    print("─" * 55)

    # 1) Truncation sweep (no noise)
    print("\n(A) Truncation sweep (noise-free):")
    print(f"    {'N_max':>6}  {'ΔM estimated':>16}  {'Error %':>9}")
    for N in [2, 4, 6, 8, 10]:
        basis = CylinderBasis(R_star, alpha, N, N)
        np.random.seed(1)
        rho_pts = np.sqrt(np.random.uniform(0, R_star**2, N_pts))
        phi_pts = np.random.uniform(0, 2 * np.pi, N_pts)
        z_pts = np.random.uniform(0.01 * L, L, N_pts)
        pts = np.column_stack([rho_pts, phi_pts, z_pts])
        A_des, b_obs, _ = generate_synthetic_data(
            basis, sigma_fn, pts, noise_sigma_pot=0, noise_sigma_acc=0
        )
        cf, _, cov = fit_coefficients(A_des, b_obs)
        dM, dM_std, _ = compute_total_mass(cf, basis, cov)
        err = abs(dM - dM_true) / abs(dM_true) * 100
        std_str = f"+/-{dM_std:.1e}" if dM_std else ""
        print(f"    {N:>6}  {dM:>16.4e}  {err:>8.4f}%  {std_str}")

    # 2) Noise sweep (fixed N=6)
    print("\n(B) Noise sweep (N=6 fixed, potential only):")
    print(f"    {'noise [m²/s²]':>14}  {'ΔM estimated':>16}  {'Error %':>9}")
    basis = CylinderBasis(R_star, alpha, 6, 6)
    # Typical potential scale: G*M/R ~ 6.67e-11 * (M_Eros~6.7e15) / 16km ~ 28 m²/s²
    noise_levels = [0.0, 1e-13, 1e-12, 1e-11, 1e-10]
    for ns in noise_levels:
        np.random.seed(1)
        rho_pts = np.sqrt(np.random.uniform(0, R_star**2, N_pts))
        phi_pts = np.random.uniform(0, 2 * np.pi, N_pts)
        z_pts = np.random.uniform(0.01 * L, L, N_pts)
        pts = np.column_stack([rho_pts, phi_pts, z_pts])
        A_des, b_obs, _ = generate_synthetic_data(
            basis,
            sigma_fn,
            pts,
            noise_sigma_pot=ns,
            use_potential=True,
            use_acceleration=False,
        )
        cf, _, cov = fit_coefficients(A_des, b_obs)
        dM, dM_std, _ = compute_total_mass(cf, basis, cov)
        err = abs(dM - dM_true) / abs(dM_true) * 100
        print(f"    {ns:>14.1e}  {dM:>16.4e}  {err:>8.4f}%")

    print(f"\n  True ΔM = {dM_true:.6e} kg")


# ═════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── CONFIGURATION ────────────────────────────────────────────────
    # Geometry (match your Eros cylinder setup)
    R_STAR = 0.1  # [m]
    L = 0.5  # [m]
    ALPHA = 1.0  # scaling (use optimizer from your original code for real data)

    # Basis truncation (increase for more accuracy, costs more computation)
    M_MAX = 8
    N_MAX = 8

    # Measurement setup
    N_PTS = 600

    # Noise (set to 0 for noiseless validation, realistic values for simulation)
    NOISE_POT = 0.0  # [m²/s²]
    NOISE_ACC = 0.0  # [m/s²]

    # Custom surface mass function (None → use default paraboloid test)
    # Example: uncomment and define your own:
    #
    # def MY_SIGMA(rho, phi):
    #     """Annular mass ring at rho ~ 0.5 R*."""
    #     sigma0 = 2e6
    #     return sigma0 * np.exp(-((rho - 0.5*R_STAR)**2) / (0.01*R_STAR**2))
    #
    SIGMA_FN = None  # ← replace with MY_SIGMA to use custom mass

    # ── RUN MAIN PIPELINE ────────────────────────────────────────────
    results = run_pipeline(
        R_star=R_STAR,
        L=L,
        alpha=ALPHA,
        m_max=M_MAX,
        n_max=N_MAX,
        N_pts=N_PTS,
        noise_potential=NOISE_POT,
        noise_acceleration=NOISE_ACC,
        use_potential=True,
        use_acceleration=True,
        sigma_fn=SIGMA_FN,
    )

    # ── SENSITIVITY STUDY ─────────────────────────────────────────────
    sensitivity_study(R_star=R_STAR, alpha=ALPHA, L=L, N_pts=N_PTS)

    print("\nDone. Check delta_mass_estimation.png for plots.")


# WITH CALUDE: i dont like the test you did cause as inverse model you use the same
# formila.... you should create the data with another method and then
# see if the formula an estimate total mass change. Also you have wro
# ng conclusions... you say " The Bessel basis with J_m(j_mn rho/R*)
# forces exactly this zero condition at rho = R*" that's why there
# is alpha! if yuou put the 0 is at alpha*R^*. so try again code
# with newton potential as foward model

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jn_zeros
from scipy.linalg import lstsq

G = 6.674e-11


# ============================================================
# CYLINDRICAL HARMONIC BASIS
# ============================================================


class CylinderBasis:
    def __init__(self, R_star, alpha, m_max, n_max):
        self.R_star = R_star
        self.alpha = alpha
        self.Ra = alpha * R_star
        self.m_max = m_max
        self.n_max = n_max
        self.n_params = 2 * m_max * n_max
        self.zeros = {m: jn_zeros(m, n_max) for m in range(m_max)}

    def j(self, m, n):
        return self.zeros[m][n - 1]

    def k(self, m, n):
        return self.j(m, n) / self.Ra

    def idx(self, m, n):
        return 2 * (m * self.n_max + (n - 1))

    def potential_row(self, rho, phi, z):
        row = np.zeros(self.n_params)
        for m in range(self.m_max):
            cp, sp = np.cos(m * phi), np.sin(m * phi)
            for n in range(1, self.n_max + 1):
                kmn = self.k(m, n)
                b = jv(m, kmn * rho) * np.exp(-kmn * z)
                i = self.idx(m, n)
                row[i], row[i + 1] = b * cp, b * sp
        return row

    def acceleration_rows(self, rho, phi, z):
        rows = np.zeros((3, self.n_params))
        rho_safe = max(rho, 1e-12)

        for m in range(self.m_max):
            cp, sp = np.cos(m * phi), np.sin(m * phi)
            for n in range(1, self.n_max + 1):
                kmn = self.k(m, n)
                x = kmn * rho
                Jm = jv(m, x)
                dJm = 0.5 * (jv(m - 1, x) - jv(m + 1, x))
                Ez = np.exp(-kmn * z)
                i = self.idx(m, n)

                rows[0, i] = -kmn * dJm * Ez * cp
                rows[0, i + 1] = -kmn * dJm * Ez * sp

                rows[1, i] = (m / rho_safe) * Jm * Ez * sp
                rows[1, i + 1] = -(m / rho_safe) * Jm * Ez * cp

                rows[2, i] = kmn * Jm * Ez * cp
                rows[2, i + 1] = kmn * Jm * Ez * sp

        return rows


# ============================================================
# INDEPENDENT TRUTH MODEL: DIRECT NEWTONIAN QUADRATURE
# ============================================================


def disk_quadrature_truth(
    points_cyl,
    sigma_fn,
    R_star,
    n_r=160,
    n_phi=240,
    return_potential=True,
    return_acceleration=True,
):
    """
    Generate synthetic gravity from direct Newtonian integration over the
    base disk z=0. This is independent of the cylindrical-harmonic model.

    Disk cells:
        dm = sigma(rho',phi') * rho' dr dphi

    Potential at x:
        U(x) = G ∫ dm / |x-x'|

    Acceleration:
        g(x) = -G ∫ dm (x-x') / |x-x'|^3
    """
    rho = (np.arange(n_r) + 0.5) * R_star / n_r
    phi = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    dr, dphi = R_star / n_r, 2 * np.pi / n_phi
    RHO, PHI = np.meshgrid(rho, phi, indexing="ij")

    xq = RHO * np.cos(PHI)
    yq = RHO * np.sin(PHI)
    zq = np.zeros_like(xq)

    sigma = sigma_fn(RHO, PHI)
    dm = sigma * RHO * dr * dphi

    xq = xq.ravel()
    yq = yq.ravel()
    zq = zq.ravel()
    dm = dm.ravel()

    N = len(points_cyl)
    U = np.zeros(N) if return_potential else None
    g_cyl = np.zeros((N, 3)) if return_acceleration else None

    for i, (rp, phip, zp) in enumerate(points_cyl):
        xp, yp = rp * np.cos(phip), rp * np.sin(phip)

        dx = xp - xq
        dy = yp - yq
        dz = zp - zq
        r2 = dx * dx + dy * dy + dz * dz
        r = np.sqrt(r2)
        r3 = r2 * r

        if return_potential:
            U[i] = G * np.sum(dm / r)

        if return_acceleration:
            gx = -G * np.sum(dm * dx / r3)
            gy = -G * np.sum(dm * dy / r3)
            gz = -G * np.sum(dm * dz / r3)

            er = np.array([np.cos(phip), np.sin(phip), 0.0])
            eph = np.array([-np.sin(phip), np.cos(phip), 0.0])

            gvec = np.array([gx, gy, gz])
            g_cyl[i, 0] = gvec @ er
            g_cyl[i, 1] = gvec @ eph
            g_cyl[i, 2] = gz

    return U, g_cyl


def generate_synthetic_data_independent(
    basis,
    sigma_fn,
    points_cyl,
    noise_sigma_pot=0.0,
    noise_sigma_acc=0.0,
    use_potential=True,
    use_acceleration=True,
    truth_n_r=160,
    truth_n_phi=240,
    seed=42,
):
    rng = np.random.default_rng(seed)

    U_true, g_true = disk_quadrature_truth(
        points_cyl,
        sigma_fn,
        basis.R_star,
        n_r=truth_n_r,
        n_phi=truth_n_phi,
        return_potential=use_potential,
        return_acceleration=use_acceleration,
    )

    blocks, obs = [], []
    N = len(points_cyl)

    if use_potential:
        A_pot = np.vstack([basis.potential_row(*p) for p in points_cyl])
        y = U_true.copy()
        if noise_sigma_pot > 0:
            y += rng.normal(0.0, noise_sigma_pot, size=N)
        blocks.append(A_pot)
        obs.append(y)

    if use_acceleration:
        A_acc = np.vstack([basis.acceleration_rows(*p) for p in points_cyl])
        y = g_true.reshape(-1).copy()
        if noise_sigma_acc > 0:
            y += rng.normal(0.0, noise_sigma_acc, size=3 * N)
        blocks.append(A_acc)
        obs.append(y)

    return np.vstack(blocks), np.hstack(obs)


# ============================================================
# LEAST SQUARES
# ============================================================


def fit_coefficients(A, b, rcond=None, return_cov=True):
    c, *_ = lstsq(A, b, cond=rcond)
    res = b - A @ c

    cov = None
    if return_cov:
        n_data, n_par = A.shape
        dof = max(n_data - n_par, 1)
        sigma2 = (res @ res) / dof
        _, s, Vt = np.linalg.svd(A, full_matrices=False)
        tol = (s[0] * max(A.shape) * np.finfo(float).eps) if rcond is None else rcond
        inv_s2 = np.where(s > tol, 1.0 / s**2, 0.0)
        cov = sigma2 * (Vt.T * inv_s2) @ Vt

    return c, res, cov


# ============================================================
# WAHR-LIKE INVERSION: COEFFS -> SURFACE MASS
# ============================================================


def invert_to_surface_mass(coeffs, basis, rho_grid, phi_grid, cov_coeffs=None):
    """
    Δσ(ρ,φ) = -1/(2πG α R*) Σ_{m,n} j_mn J_m(k_mn ρ)[A_mn cos(mφ)+B_mn sin(mφ)]
    """
    pref = -1.0 / (2.0 * np.pi * G * basis.Ra)
    sigma = np.zeros_like(rho_grid)
    Jmap = np.zeros((rho_grid.size, basis.n_params)) if cov_coeffs is not None else None

    flat_rho, flat_phi = rho_grid.ravel(), phi_grid.ravel()

    for m in range(basis.m_max):
        cp = np.cos(m * phi_grid)
        sp = np.sin(m * phi_grid)
        cp_f = np.cos(m * flat_phi)
        sp_f = np.sin(m * flat_phi)

        for n in range(1, basis.n_max + 1):
            jmn, kmn = basis.j(m, n), basis.k(m, n)
            Jm = jv(m, kmn * rho_grid)
            Jm_f = jv(m, kmn * flat_rho)
            i = basis.idx(m, n)

            sigma += pref * jmn * Jm * (coeffs[i] * cp + coeffs[i + 1] * sp)

            if Jmap is not None:
                Jmap[:, i] = pref * jmn * Jm_f * cp_f
                Jmap[:, i + 1] = pref * jmn * Jm_f * sp_f

    sigma_std = None
    if Jmap is not None:
        var = np.einsum("ij,jk,ik->i", Jmap, cov_coeffs, Jmap)
        sigma_std = np.sqrt(np.maximum(var, 0.0)).reshape(rho_grid.shape)

    return sigma, sigma_std


# ============================================================
# TOTAL MASS
# ============================================================


def compute_total_mass(coeffs, basis, cov_coeffs=None):
    """
    Only m=0 survives:
        ΔM = -R*^2/(G α) Σ_n [J_1(j_0n/α)/j_0n] A_0n
    """
    R, a = basis.R_star, basis.alpha
    dM = 0.0
    grad = np.zeros(basis.n_params)
    per_mode = []

    for n in range(1, basis.n_max + 1):
        j0n = basis.j(0, n)
        w = -(R**2 / (G * a)) * jv(1, j0n / a) / j0n
        i = basis.idx(0, n)
        dm = w * coeffs[i]
        dM += dm
        grad[i] = w
        per_mode.append(dm)

    dM_std = None
    if cov_coeffs is not None:
        dM_std = np.sqrt(np.maximum(grad @ cov_coeffs @ grad, 0.0))

    return dM, dM_std, np.array(per_mode)


def compute_total_mass_numerical(sigma_map, rho_grid, phi_grid):
    drho = rho_grid[1, 0] - rho_grid[0, 0]
    dphi = phi_grid[0, 1] - phi_grid[0, 0]
    return np.sum(sigma_map * rho_grid) * drho * dphi


def true_total_mass(sigma_fn, R_star, n_r=1000, n_phi=720):
    rho = (np.arange(n_r) + 0.5) * R_star / n_r
    phi = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    dr, dphi = R_star / n_r, 2 * np.pi / n_phi
    RHO, PHI = np.meshgrid(rho, phi, indexing="ij")
    return np.sum(sigma_fn(RHO, PHI) * RHO) * dr * dphi


# ============================================================
# PIPELINE
# ============================================================


def run_pipeline(
    R_star=0.1,
    L=0.5,
    alpha=1.0,
    m_max=8,
    n_max=8,
    N_pts=600,
    noise_potential=0.0,
    noise_acceleration=0.0,
    use_potential=True,
    use_acceleration=True,
    sigma_fn=None,
    n_rho_grid=80,
    n_phi_grid=160,
    truth_n_r=160,
    truth_n_phi=240,
    seed=1,
):
    if sigma_fn is None:
        S0 = 1e6

        def sigma_fn(rho, phi):
            return (
                S0
                * (1 - (rho / R_star) ** 2)
                * (1 + 0.4 * np.cos(phi) + 0.2 * np.sin(2 * phi))
            )

    basis = CylinderBasis(R_star, alpha, m_max, n_max)

    rng = np.random.default_rng(seed)
    rho_pts = np.sqrt(rng.uniform(0, R_star**2, N_pts))
    phi_pts = rng.uniform(0, 2 * np.pi, N_pts)
    z_pts = rng.uniform(0.01 * L, L, N_pts)
    points = np.column_stack((rho_pts, phi_pts, z_pts))

    A, b = generate_synthetic_data_independent(
        basis,
        sigma_fn,
        points,
        noise_sigma_pot=noise_potential,
        noise_sigma_acc=noise_acceleration,
        use_potential=use_potential,
        use_acceleration=use_acceleration,
        truth_n_r=truth_n_r,
        truth_n_phi=truth_n_phi,
    )

    coeffs, residuals, cov = fit_coefficients(A, b)

    rho_1d = np.linspace(0.005 * R_star, 0.99 * R_star, n_rho_grid)
    phi_1d = np.linspace(0, 2 * np.pi, n_phi_grid, endpoint=False)
    RHO, PHI = np.meshgrid(rho_1d, phi_1d, indexing="ij")

    sigma_est, sigma_std = invert_to_surface_mass(coeffs, basis, RHO, PHI, cov)
    sigma_true = sigma_fn(RHO, PHI)

    dM_true = true_total_mass(sigma_fn, R_star)
    dM_formula, dM_std, per_mode = compute_total_mass(coeffs, basis, cov)
    dM_num = compute_total_mass_numerical(sigma_est, RHO, PHI)

    print("\n" + "=" * 64)
    print("CYLINDRICAL HARMONIC INVERSION WITH INDEPENDENT TRUTH MODEL")
    print("=" * 64)
    print(f"Design matrix shape        : {A.shape}")
    print(f"Residual RMS               : {np.sqrt(np.mean(residuals**2)):.3e}")
    print(f"True ΔM                    : {dM_true:.6e} kg")
    print(f"Estimated ΔM (formula)     : {dM_formula:.6e} kg")
    print(f"Estimated ΔM (numerical)   : {dM_num:.6e} kg")
    if dM_std is not None:
        print(f"Estimated ΔM std           : {dM_std:.3e} kg")
    print(
        f"Formula mass error         : {100*abs(dM_formula-dM_true)/abs(dM_true):.3f}%"
    )
    print(f"Grid mass error            : {100*abs(dM_num-dM_true)/abs(dM_true):.3f}%")

    # Minimal plots
    X, Y = RHO * np.cos(PHI), RHO * np.sin(PHI)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    s = max(np.max(np.abs(sigma_true)), 1e-30)

    im0 = ax[0].pcolormesh(X, Y, sigma_true / s, shading="gouraud", cmap="RdBu_r")
    ax[0].set_title("True $\\Delta\\sigma$")
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].pcolormesh(X, Y, sigma_est / s, shading="gouraud", cmap="RdBu_r")
    ax[1].set_title("Estimated $\\Delta\\sigma$")
    fig.colorbar(im1, ax=ax[1])

    im2 = ax[2].pcolormesh(
        X,
        Y,
        np.log10(np.abs(sigma_est - sigma_true) / s + 1e-12),
        shading="gouraud",
        cmap="hot_r",
    )
    ax[2].set_title("log10 relative error")
    fig.colorbar(im2, ax=ax[2])

    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")

    plt.show()

    return {
        "basis": basis,
        "coeffs_fit": coeffs,
        "cov": cov,
        "residuals": residuals,
        "sigma_est": sigma_est,
        "sigma_true": sigma_true,
        "sigma_std": sigma_std,
        "delta_M": dM_formula,
        "delta_M_std": dM_std,
        "delta_M_true": dM_true,
        "delta_M_num": dM_num,
        "per_mode_mass": per_mode,
    }


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":
    R_STAR, L, ALPHA = 0.1, 0.5, 100.0

    def sigma_fn(rho, phi):
        s0 = 1e6
        ring = np.exp(-(((rho - 0.45 * R_STAR) / (0.12 * R_STAR)) ** 2))
        blob = np.exp(-(((rho - 0.72 * R_STAR) / (0.10 * R_STAR)) ** 2)) * np.cos(
            phi - 0.6
        )
        return s0 * (0.8 * ring + 0.35 * blob)

    run_pipeline(
        R_star=R_STAR,
        L=L,
        alpha=ALPHA,
        m_max=18,
        n_max=18,
        N_pts=1000,
        noise_potential=0.0,
        noise_acceleration=1e-6,
        use_potential=True,
        use_acceleration=True,
        sigma_fn=sigma_fn,
        truth_n_r=180,
        truth_n_phi=260,
    )
'''

"""
=============================================================================
CYLINDRICAL HARMONIC MASS CHANGE ESTIMATOR
Polyhedral mesh → gravity field fits → ΔM + Δσ(ρ,φ) map
=============================================================================

WORKFLOW
--------
1. Load "before" body as a polyhedral mesh  (e.g. asteroid, rock sample)
2. Define a surface mass layer Δσ(ρ,φ) at the cylinder base z = 0
   (positive = mass added, negative = mass removed)
3. Generate synthetic "after" gravity = "before" gravity + gravity of Δσ
4. At each state, fit cylindrical harmonic coefficients {A_mn} to
   gravity measurements (U, gρ, gφ, gz) at field points INSIDE the cylinder
5. Compute ΔA_mn = A_mn^after − A_mn^before
6. Apply Wahr-like inversion formula to ΔA_mn:
      ΔM    = (R*/G) · Σ_n  J_1(j_{0n}/α) · ΔA_{0n}
      Δσ(ρ,φ) = 1/(2πGαR*) · Σ_{m,n} j_{mn} J_m(k_{mn}ρ) [ΔA cos mφ + ΔB sin mφ]

SIGN CONVENTIONS (verified numerically)
----------------------------------------
  g = +∇U   (geodesy convention, U = G∫σ/D dA)
  gρ = ∂U/∂ρ = Σ k_mn J_m'(k_mn ρ) exp(-k_mn z) [A cos + B sin]
  gφ = (1/ρ)∂U/∂φ = Σ (m/ρ) J_m exp(-k_mn z) [-A sin + B cos]
  gz = ∂U/∂z = Σ (−k_mn) J_m(k_mn ρ) exp(-k_mn z) [A cos + B sin]   ← NEGATIVE sign
  Neumann BC: ∂U/∂z|_{z=0+} = −2πGσ   (negative for positive mass layer)
  Forward:    A_{mn} = +2πGαR*/j_{mn} × C^σ_{mn}   (positive)
  Wahr ΔM:    ΔM = +R*/G × Σ_n J_1(j_{0n}/α) × ΔA_{0n}   (positive)

BASIS ORTHOGONALITY
-------------------
  {J_m(k_{mn}ρ)} is orthogonal on [0, αR*] (NOT on [0, R*]).
  N²_{mn} = (αR*)² / 2 · J_{m+1}(j_{mn})²
  This means σ can be nonzero at ρ = R* — α > 1 allows free edge BC.
  Zeros of J_m are at ρ = αR*, not ρ = R*.

USAGE
-----
  # With your own mesh:
  import trimesh
  mesh = trimesh.load("your_body.stl")
  result = run(mesh_before=mesh, density=2700.0)

  # With synthetic data (default):
  result = run()

  # To load an Eros-style .pk mesh:
  import pickle
  with open("eros.pk","rb") as f: data = pickle.load(f)
  mesh = trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"])
  result = run(mesh_before=mesh, density=2700.0)

REQUIREMENTS
------------
  trimesh, scipy, numpy, matplotlib
=============================================================================
"""

import numpy as np
import trimesh
from scipy.special import jv as BesselJ, jn_zeros
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import warnings, time, os

G = 6.674e-11
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
    }
)
COLORS = ["#E6001A", "#0077BB", "#F08C00", "#2c7bb6", "#009933", "#7B2D8B"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLORS)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — POLYHEDRAL GRAVITY  (Tsoulis / Werner signed-tetrahedra fan)
# ═══════════════════════════════════════════════════════════════════════════


def poly_gravity(vertices, faces, field_pts, rho_body):
    """
    Gravitational potential U and acceleration g = ∇U for a constant-density
    polyhedron via signed-tetrahedra fan from the coordinate origin.

    Algorithm
    ---------
    Decompose the polyhedron into signed tetrahedra (origin + each face).
    Integrate G·ρ/|r−r'| over each tetrahedron using the 4-point Dunavant
    Gauss rule on the reference tetrahedron (exact for polynomials up to degree 2).

    Physical integral over physical tet of signed volume V_f:
        I_f ≈ Σ_q  w_q · f(x_q) · V_f
    where  Σ_q w_q = 1  (weights normalised to unit-tet volume = 1).
    The signed volume ensures correct orientation for inward-facing faces.

    Parameters
    ----------
    vertices  : (Nv, 3) float   — vertex coordinates
    faces     : (Nf, 3) int     — face vertex indices
    field_pts : (N, 3) float    — field points (must be OUTSIDE the body)
    rho_body  : float           — body bulk density [kg/m³]

    Returns
    -------
    U, gx, gy, gz : each (N,) — potential and gravity in Cartesian coords
                    (g = ∇U, so gz < 0 above a mass at z < 0)
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int32)
    P = np.asarray(field_pts, dtype=np.float64)
    Np = len(P)

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    # Signed volume of tet (origin, v0, v1, v2): vol = det[v0,v1,v2]/6
    vols = np.einsum("fi,fi->f", v0, np.cross(v1, v2)) / 6.0  # (Nf,)

    # 4-point Dunavant Gauss rule on reference tet with vertices at
    # (1,0,0),(0,1,0),(0,0,1),(0,0,0). Barycentric weights normalised to 1.
    a = 0.13819660112501052
    b = 0.58541019662496847
    xi = np.array(
        [[a, a, a], [b, a, a], [a, b, a], [a, a, b]]
    )  # (4,3): coords of v0,v1,v2
    wt = np.full(4, 0.25)  # weights sum to 1

    U = np.zeros(Np)
    gx = np.zeros(Np)
    gy = np.zeros(Np)
    gz = np.zeros(Np)

    for xq, wq in zip(xi, wt):
        # Source points for all faces at this Gauss point (origin has weight 1−Σxq)
        src = xq[0] * v0 + xq[1] * v1 + xq[2] * v2  # (Nf, 3)

        for i in range(Np):
            dr = P[i] - src  # (Nf, 3)
            d2 = np.einsum("fi,fi->f", dr, dr)  # (Nf,)
            d1 = np.sqrt(d2)
            d3 = d2 * d1
            mask = d1 > 1e-14
            sv = vols[mask] * wq  # signed weighted vol
            U[i] += G * rho_body * np.sum(sv / d1[mask])
            gx[i] += G * rho_body * np.sum(sv * (-dr[mask, 0]) / d3[mask])
            gy[i] += G * rho_body * np.sum(sv * (-dr[mask, 1]) / d3[mask])
            gz[i] += G * rho_body * np.sum(sv * (-dr[mask, 2]) / d3[mask])

    return U, gx, gy, gz


def surface_mass_gravity(sigma_fn, R_star, field_pts, n_rho=180, n_phi=240):
    """
    Gravity of a thin surface mass layer Δσ(ρ,φ) at z = 0 on a disk of radius R*.
    Uses Gauss-Legendre × uniform trapezoidal quadrature.

    g = ∇U convention, U = G ∫∫ σ(r')/|r−r'| dA'

    Returns U, gx, gy, gz each of shape (N,).
    """
    gl, gw = np.polynomial.legendre.leggauss(n_rho)
    rho_s = 0.5 * R_star * (gl + 1.0)
    w_rho = 0.5 * R_star * gw
    phi_s = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    RS, PS = np.meshgrid(rho_s, phi_s, indexing="ij")
    WR, _ = np.meshgrid(w_rho, phi_s, indexing="ij")
    dA = RS * WR * dphi
    SIG = sigma_fn(RS, PS)

    P = np.asarray(field_pts, dtype=np.float64)
    Np = len(P)
    U = np.zeros(Np)
    gx = np.zeros(Np)
    gy = np.zeros(Np)
    gz = np.zeros(Np)

    XS = RS * np.cos(PS)
    YS = RS * np.sin(PS)

    for i in range(Np):
        dx = P[i, 0] - XS
        dy = P[i, 1] - YS
        dz = P[i, 2]
        D = np.sqrt(dx**2 + dy**2 + dz**2)
        D3 = D**3
        U[i] += G * np.sum(SIG / D * dA)
        gx[i] += G * np.sum(SIG * (-dx) / D3 * dA)
        gy[i] += G * np.sum(SIG * (-dy) / D3 * dA)
        gz[i] += G * np.sum(SIG * (-dz) / D3 * dA)

    return U, gx, gy, gz


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — CYLINDRICAL HARMONIC BASIS
# ═══════════════════════════════════════════════════════════════════════════


def build_design_matrix(rho_pts, phi_pts, z_pts, R_alpha, m_max, n_max):
    """
    Build the design matrix A such that A @ coeffs ≈ [U, gρ, gφ, gz] at each point.

    Basis: U = Σ_{m,n} J_m(k_mn ρ) exp(−k_mn z) [A_mn cos(mφ) + B_mn sin(mφ)]
           k_mn = j_{mn} / (αR*)   (zeros of J_m at ρ = αR*, NOT ρ = R*)

    Column ordering: for m=0..m_max-1, n=1..n_max:
      col 2*(m*n_max+(n-1))   = A_{mn} coefficient (cosine)
      col 2*(m*n_max+(n-1))+1 = B_{mn} coefficient (sine)

    Row ordering per field point i (rows 4i, 4i+1, 4i+2, 4i+3):
      U row  : J_m exp [cos, sin]
      gρ row : k_mn J_m'(k_mn ρ) exp [cos, sin]        ← ∂U/∂ρ
      gφ row : (m/ρ) J_m exp [−sin, cos]               ← (1/ρ)∂U/∂φ
      gz row : (−k_mn) J_m exp [cos, sin]              ← ∂U/∂z  (NEGATIVE k)

    Returns
    -------
    A      : (4N, 2·m_max·n_max) design matrix
    zeros  : dict {m: j_mn zeros array}
    """
    zeros = {m: jn_zeros(m, n_max) for m in range(m_max)}
    N = len(rho_pts)
    N_par = 2 * m_max * n_max
    A = np.zeros((4 * N, N_par))

    for i, (rh, ph, z) in enumerate(zip(rho_pts, phi_pts, z_pts)):
        rs = max(rh, 1e-12)
        for m in range(m_max):
            for n in range(1, n_max + 1):
                jmn = zeros[m][n - 1]
                kmn = jmn / R_alpha
                x = kmn * rh
                Ez = np.exp(-kmn * z)
                Jm = BesselJ(m, x)
                dJm = 0.5 * (BesselJ(m - 1, x) - BesselJ(m + 1, x))
                cp = np.cos(m * ph)
                sp = np.sin(m * ph)
                c = 2 * (m * n_max + (n - 1))

                A[4 * i, c] = Jm * Ez * cp
                A[4 * i, c + 1] = Jm * Ez * sp  # U
                A[4 * i + 1, c] = kmn * dJm * Ez * cp
                A[4 * i + 1, c + 1] = kmn * dJm * Ez * sp  # gρ
                A[4 * i + 2, c] = -m / rs * Jm * Ez * sp
                A[4 * i + 2, c + 1] = m / rs * Jm * Ez * cp  # gφ
                A[4 * i + 3, c] = -kmn * Jm * Ez * cp
                A[4 * i + 3, c + 1] = -kmn * Jm * Ez * sp  # gz

    return A, zeros


def fit_coefficients(A_design, U_obs, gr_obs, gphi_obs, gz_obs):
    """
    Fit cylindrical harmonic coefficients via least squares.

    Assembles observation vector [U, gρ, gφ, gz] per point and solves
    A @ coeffs ≈ b via scipy lstsq.

    Returns coeffs (N_par,), residual RMS, relative RMS.
    """
    N = len(U_obs)
    b = np.zeros(4 * N)
    for i in range(N):
        b[4 * i] = U_obs[i]
        b[4 * i + 1] = gr_obs[i]
        b[4 * i + 2] = gphi_obs[i]
        b[4 * i + 3] = gz_obs[i]
    coeffs, _, _, _ = lstsq(A_design, b)
    rms = np.sqrt(np.mean((A_design @ coeffs - b) ** 2))
    rel_rms = rms / (np.sqrt(np.mean(b**2)) + 1e-30)
    return coeffs, rms, rel_rms


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — WAHR-LIKE INVERSION
# ═══════════════════════════════════════════════════════════════════════════


def wahr_invert(delta_coeffs, R_star, alpha, n_max, m_max, zeros, n_rho=80, n_phi=120):
    """
    Recover ΔM and Δσ(ρ,φ) from ΔA_mn = A_mn^after − A_mn^before.

    ΔM formula (only m=0 contributes):
        ΔM = (R*/G) · Σ_{n=1}^{N} J_1(j_{0n}/α) · ΔA_{0n}

    Δσ spatial map (Wahr-like series):
        Δσ(ρ,φ) = 1/(2πGαR*) · Σ_{m,n} j_{mn} J_m(k_{mn}ρ)
                    × [ΔA_{mn} cos(mφ) + ΔB_{mn} sin(mφ)]

    Sign convention: positive ΔA_{0n} → positive ΔM (mass added at base).

    Parameters
    ----------
    delta_coeffs : (N_par,) — ΔA_mn coefficients from before/after fit difference
    R_star       : float — cylinder/disk radius [m]
    alpha        : float — Bessel extension parameter (α > 1 recommended)
    n_max, m_max : int — truncation orders
    zeros        : dict from build_design_matrix

    Returns
    -------
    delta_M   : float [kg]
    sigma_map : (n_rho, n_phi) array [kg/m²]
    RHO, PHI  : meshgrid arrays for the map
    """
    R_alpha = alpha * R_star
    delta_M = 0.0

    # ΔM from m=0 terms only
    for n in range(1, n_max + 1):
        j0n = zeros[0][n - 1]
        c = 2 * (0 * n_max + (n - 1))
        delta_M += R_star / G * BesselJ(1, j0n / alpha) * delta_coeffs[c]

    # Δσ spatial map
    rho_1d = np.linspace(0.02 * R_star, 0.98 * R_star, n_rho)
    phi_1d = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    RHO, PHI = np.meshgrid(rho_1d, phi_1d, indexing="ij")
    sigma_map = np.zeros_like(RHO)
    pref = 1.0 / (2 * np.pi * G * R_alpha)

    for m in range(m_max):
        for n in range(1, n_max + 1):
            jmn = zeros[m][n - 1]
            kmn = jmn / R_alpha
            c = 2 * (m * n_max + (n - 1))
            bess = BesselJ(m, kmn * RHO)
            sigma_map += (
                pref
                * jmn
                * bess
                * (
                    delta_coeffs[c] * np.cos(m * PHI)
                    + delta_coeffs[c + 1] * np.sin(m * PHI)
                )
            )

    return delta_M, sigma_map, RHO, PHI


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — FIELD POINT PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════


def make_field_points(
    R_star,
    H,
    N=400,
    z_min_frac=0.01,
    z_max_frac=0.6,
    seed=1,
    distribution="three_bands",
):
    """
    Generate field points inside the cylinder, suitable for coefficient fitting.

    IMPORTANT: field points should span a range of heights for good
    conditioning of the coefficient recovery. Near-surface points (small z)
    are most sensitive to the mass layer; mid-range points help recover
    higher-degree azimuthal structure.

    Parameters
    ----------
    z_min_frac : float — min z as fraction of R_star (avoid near z=0 singularity)
    distribution : "three_bands" | "uniform"
    """
    np.random.seed(seed)

    if distribution == "three_bands":
        N1, N2, N3 = N // 3, N // 3, N - 2 * (N // 3)
        bands = [
            (N1, z_min_frac * R_star, 0.05 * R_star),  # near-base
            (N2, 0.05 * R_star, 0.20 * R_star),  # mid
            (N3, 0.20 * R_star, z_max_frac * H),  # far
        ]
        rp = []
        pp = []
        zp = []
        for Nb, z_lo, z_hi in bands:
            rp.append(np.sqrt(np.random.uniform(0, R_star**2, Nb)))
            pp.append(np.random.uniform(0, 2 * np.pi, Nb))
            zp.append(np.random.uniform(z_lo, z_hi, Nb))
        return (np.concatenate(rp), np.concatenate(pp), np.concatenate(zp))
    else:
        rp = np.sqrt(np.random.uniform(0, R_star**2, N))
        pp = np.random.uniform(0, 2 * np.pi, N)
        zp = np.random.uniform(z_min_frac * R_star, z_max_frac * H, N)
        return rp, pp, zp


def _to_cartesian(rp, pp, zp):
    return np.column_stack([rp * np.cos(pp), rp * np.sin(pp), zp])


def _to_cylindrical_g(gx, gy, phi_pts):
    gr = gx * np.cos(phi_pts) + gy * np.sin(phi_pts)
    gphi = -gx * np.sin(phi_pts) + gy * np.cos(phi_pts)
    return gr, gphi


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def run(
    mesh_before=None,  # trimesh.Trimesh — the "before" body; None→ synthetic cylinder
    density=2700.0,  # body bulk density [kg/m³]
    R_star=None,  # cylinder radius [m]; None→ auto from mesh bounding box
    H=None,  # cylinder height [m]; None→ auto from mesh
    alpha=2.0,  # Bessel extension parameter (α > 1: zeros at αR*, NOT R*)
    m_max=5,  # max azimuthal order
    n_max=8,  # max radial index
    N_field=400,  # number of field points
    sigma_fn=None,  # Δσ(ρ,φ) [kg/m²] surface mass; None → default test pattern
    verbose=True,
    outdir="/mnt/user-data/outputs",
):
    """
    Full pipeline: mesh → coefficient fits (before/after) → ΔM + Δσ map.

    The "after" state is simulated by adding surface mass gravity (sigma_fn)
    on top of the "before" polyhedral gravity at the same field points.
    No second mesh is needed — the change is purely a thin mass layer.

    This matches real-world usage: you have one body mesh, measure its gravity
    at field points (orbiting spacecraft, surface sensors), then measure again
    after a mass redistribution event (landing, regolith shift, etc.),
    and recover ΔM and the spatial map Δσ(ρ,φ).

    Returns dict with all results.
    """
    sep = "=" * 62

    # ── MESH SETUP ─────────────────────────────────────────────────────
    if mesh_before is None:
        # Synthetic: solid cylinder
        R_default = 0.10
        H_default = 0.50
        if R_star is None:
            R_star = R_default
        if H is None:
            H = H_default
        mesh_before = trimesh.creation.cylinder(radius=R_star, height=H, sections=64)
        mesh_before.apply_translation([0, 0, H / 2])
        if verbose:
            print(sep)
            print("Using SYNTHETIC CYLINDER mesh (no mesh_before provided)")
            print(f"  R* = {R_star} m,  H = {H} m,  density = {density} kg/m³")
    else:
        # Auto-detect R* and H from mesh bounding box
        bb = mesh_before.bounding_box.extents
        if R_star is None:
            R_star = 0.5 * max(bb[0], bb[1])
        if H is None:
            H = bb[2]
        if verbose:
            print(sep)
            print(
                f"Loaded mesh: {len(mesh_before.vertices)} vertices, "
                f"{len(mesh_before.faces)} faces"
            )
            print(f"  Auto-detected: R* = {R_star:.4f} m, H = {H:.4f} m")

    R_alpha = alpha * R_star

    if verbose:
        print(f"  Basis: m ∈ [0,{m_max-1}], n ∈ [1,{n_max}],  α = {alpha}")
        print(f"  Bessel zeros at αR* = {R_alpha:.4f} m  (not at R*)")

    # ── SIGMA (surface mass layer) ─────────────────────────────────────
    if sigma_fn is None:
        S0 = -5e-2  # kg/m²

        def sigma_fn(rho, phi):
            """
            Default test mass: smooth paraboloid + cos + sin2 variation.
            Nonzero at ρ = R* to test α > 1 boundary freedom.
            """
            return (
                S0
                * (0.5 + 0.5 * (1.0 - (rho / R_star) ** 2))
                * (1.0 + 0.30 * np.cos(phi) + 0.20 * np.sin(2 * phi))
            )

    # True ΔM (analytical integration)
    rho_q = np.linspace(0, R_star, 3000)[1:]
    phi_q = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    RI, PI = np.meshgrid(rho_q, phi_q, indexing="ij")
    dM_true = np.sum(sigma_fn(RI, PI) * RI) * (R_star / 3000) * (2 * np.pi / 600)

    if verbose:
        print(f"\n  True ΔM = {dM_true:.6e} kg  (ground truth for validation)")

    # ── FIELD POINTS ───────────────────────────────────────────────────
    rp, pp, zp = make_field_points(
        R_star,
        H,
        N=N_field,
        z_min_frac=0.04,  # ≥4% of R_star above base
        z_max_frac=0.7,
    )
    pts_cart = _to_cartesian(rp, pp, zp)

    if verbose:
        print(
            f"\n  Field points: {len(rp)}  " f"(z: {zp.min():.4f} – {zp.max():.4f} m)"
        )

    # ── GRAVITY: BEFORE STATE (polyhedral body only) ──────────────────
    if verbose:
        print(
            f"\n  Computing BEFORE gravity (polyhedral, "
            f"{len(mesh_before.faces)} faces)..."
        )
    t0 = time.time()
    U_b, gx_b, gy_b, gz_b = poly_gravity(
        mesh_before.vertices, mesh_before.faces, pts_cart, density
    )
    gr_b, gphi_b = _to_cylindrical_g(gx_b, gy_b, pp)
    if verbose:
        print(f"    Done in {time.time()-t0:.1f}s")

    # ── GRAVITY: DELTA from surface mass layer ─────────────────────────
    if verbose:
        print(f"  Computing DELTA gravity (surface mass layer)...")
    t0 = time.time()
    dU, dgx, dgy, dgz = surface_mass_gravity(sigma_fn, R_star, pts_cart)
    dgr, dgphi = _to_cylindrical_g(dgx, dgy, pp)
    if verbose:
        print(f"    Done in {time.time()-t0:.1f}s")
        sig_ratio = np.std(dU) / (np.std(U_b) + 1e-30)
        print(f"    Signal ratio ΔU/U_body = {sig_ratio:.3e}")

    # ── AFTER STATE ────────────────────────────────────────────────────
    U_a = U_b + dU
    gr_a = gr_b + dgr
    gphi_a = gphi_b + dgphi
    gz_a = gz_b + dgz

    # ── FIT CYLINDRICAL HARMONIC COEFFICIENTS ─────────────────────────
    if verbose:
        print(f"\n  Building design matrix ({4*len(rp)} × {2*m_max*n_max})...")
    A_des, zeros_d = build_design_matrix(rp, pp, zp, R_alpha, m_max, n_max)

    if verbose:
        print("  Fitting BEFORE coefficients...")
    c_b, rms_b, rel_b = fit_coefficients(A_des, U_b, gr_b, gphi_b, gz_b)

    if verbose:
        print("  Fitting AFTER coefficients...")
    c_a, rms_a, rel_a = fit_coefficients(A_des, U_a, gr_a, gphi_a, gz_a)

    if verbose:
        print(f"    Before fit RMS: {rms_b:.3e}  (relative: {rel_b:.4f})")
        print(f"    After  fit RMS: {rms_a:.3e}  (relative: {rel_a:.4f})")

    # ── DELTA COEFFICIENTS & WAHR INVERSION ───────────────────────────
    d_coeffs = c_a - c_b

    dM_est, sigma_map, RHO, PHI = wahr_invert(
        d_coeffs, R_star, alpha, n_max, m_max, zeros_d, n_rho=80, n_phi=120
    )

    sigma_true_map = sigma_fn(RHO, PHI)
    sigma_rms = np.sqrt(np.mean((sigma_map - sigma_true_map) ** 2))
    err_M = abs(dM_est - dM_true) / abs(dM_true) * 100
    err_sig = sigma_rms / (np.amax(np.abs(sigma_true_map)) + 1e-30) * 100

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  True  ΔM = {dM_true:.6e} kg")
        print(f"  Est.  ΔM = {dM_est:.6e} kg    [{err_M:.2f}% error]")
        print(f"  Δσ RMS   = {sigma_rms:.3e} kg/m²  [{err_sig:.1f}% of peak]")
        print(f"{'─'*55}")

    return dict(
        mesh_before=mesh_before,
        density=density,
        R_star=R_star,
        H=H,
        alpha=alpha,
        m_max=m_max,
        n_max=n_max,
        rp=rp,
        pp=pp,
        zp=zp,
        U_before=U_b,
        gz_before=gz_b,
        dU=dU,
        dgz=dgz,
        coeffs_before=c_b,
        coeffs_after=c_a,
        delta_coeffs=d_coeffs,
        rms_before=rms_b,
        rms_after=rms_a,
        dM_true=dM_true,
        dM_est=dM_est,
        err_pct=err_M,
        sigma_map=sigma_map,
        sigma_true=sigma_true_map,
        sigma_fn=sigma_fn,
        zeros=zeros_d,
        RHO=RHO,
        PHI=PHI,
        sigma_rms=sigma_rms,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — DIAGNOSTIC PLOTS
# ═══════════════════════════════════════════════════════════════════════════


def plot_results(res, outdir="/mnt/user-data/outputs"):
    """Six-panel diagnostic figure."""
    R = res["R_star"]
    RHO = res["RHO"]
    PHI = res["PHI"]
    X = RHO * np.cos(PHI)
    Y = RHO * np.sin(PHI)
    tc = np.linspace(0, 2 * np.pi, 300)

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    sc = max(np.abs(res["sigma_true"]).max(), 1e-10)
    vlim = 1.15

    # ── True Δσ ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    c1 = ax1.pcolormesh(
        X,
        Y,
        res["sigma_true"] / sc,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        shading="gouraud",
    )
    fig.colorbar(c1, ax=ax1, label="normalised")
    ax1.plot(R * np.cos(tc), R * np.sin(tc), "k--", lw=1, alpha=0.5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("True Δσ at base z=0\n(nonzero at ρ=R*, allowed by α>1)")

    # ── Estimated Δσ ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    c2 = ax2.pcolormesh(
        X,
        Y,
        res["sigma_map"] / sc,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        shading="gouraud",
    )
    fig.colorbar(c2, ax=ax2, label="normalised")
    ax2.plot(R * np.cos(tc), R * np.sin(tc), "k--", lw=1, alpha=0.5)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title(
        f"Recovered Δσ  (Wahr inversion)\n"
        f'ΔM = {res["dM_est"]:.3e} kg  [{res["err_pct"]:.2f}%]'
    )

    # ── Residual Δσ ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    resid = res["sigma_map"] - res["sigma_true"]
    c3 = ax3.pcolormesh(
        X, Y, resid / sc, cmap="PiYG", vmin=-0.3, vmax=0.3, shading="gouraud"
    )
    fig.colorbar(c3, ax=ax3, label="norm. residual")
    ax3.plot(R * np.cos(tc), R * np.sin(tc), "k--", lw=1, alpha=0.5)
    ax3.set_aspect("equal")
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_title(f'Residual Δσ (est − true)\nRMS = {res["sigma_rms"]:.2e} kg/m²')

    # ── Delta coefficients spectrum ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    dc = res["delta_coeffs"]
    n_modes = len(dc) // 2
    mode_idx = np.arange(n_modes)
    mode_amp = np.sqrt(dc[0::2] ** 2 + dc[1::2] ** 2)
    ax4.bar(mode_idx, mode_amp, color=COLORS[0], alpha=0.7)
    ax4.set_xlabel("Mode index (m·n_max + n)")
    ax4.set_ylabel("|ΔA| + |ΔB| amplitude")
    ax4.set_title("ΔCoefficient spectrum\n(m=0 block carries ΔM signal)")
    # Mark m=0 block
    ax4.axvspan(-0.5, res["n_max"] - 0.5, alpha=0.12, color=COLORS[1], label="m=0")
    ax4.legend()

    # ── Field point coverage ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sc5 = ax5.scatter(
        res["rp"], res["zp"], c=res["dU"] * 1e6, cmap="plasma", s=8, alpha=0.7
    )
    fig.colorbar(sc5, ax=ax5, label="ΔU [μJ/kg]")
    ax5.set_xlabel("ρ [m]")
    ax5.set_ylabel("z [m]")
    ax5.set_title(
        f"Field point distribution\n" f'N={len(res["rp"])}, coloured by ΔU signal'
    )

    # ── Azimuthal profile at mid-radius ──────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    r_mid = 0.5 * res["R_star"]
    i_mid = np.argmin(np.abs(res["RHO"][:, 0] - r_mid))
    phi_1d = PHI[i_mid, :]
    ax6.plot(
        np.degrees(phi_1d), res["sigma_true"][i_mid, :] / 1e3, "k-", lw=2, label="True"
    )
    ax6.plot(
        np.degrees(phi_1d),
        res["sigma_map"][i_mid, :] / 1e3,
        "--",
        color=COLORS[0],
        lw=2,
        label="Recovered",
    )
    ax6.set_xlabel("φ [degrees]")
    ax6.set_ylabel("Δσ [kg/m² × 10³]")
    ax6.set_title(f"Azimuthal profile at ρ = {r_mid:.3f} m")
    ax6.legend()

    fig.suptitle(
        "Cylindrical Harmonic Mass Change Estimator\n"
        f'Mesh: {len(res["mesh_before"].faces)} faces  |  '
        f'α={res["alpha"]}  |  '
        f'm_max={res["m_max"]}, n_max={res["n_max"]}  |  '
        f'ΔM error = {res["err_pct"]:.2f}%',
        fontsize=11,
    )

    # os.makedirs(outdir, exist_ok=True)
    # out_png = os.path.join(outdir, "cylindrical_mass_change.png")
    # out_pdf = os.path.join(outdir, "cylindrical_mass_change.pdf")
    # plt.savefig(out_png, dpi=150, bbox_inches="tight")
    # plt.savefig(out_pdf, dpi=150, bbox_inches="tight")
    # print(f"Figure saved to {out_png}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — SENSITIVITY STUDIES
# ═══════════════════════════════════════════════════════════════════════════


def study_alpha(
    mesh,
    density,
    R_star,
    H,
    sigma_fn,
    alpha_vals=None,
    m_max=5,
    n_max=8,
    N_field=400,
    dM_true=None,
):
    """Sweep α values and report ΔM error."""
    if alpha_vals is None:
        alpha_vals = [1.2, 1.5, 2.0, 3.0, 5.0]
    print(f"\nα sweep (m={m_max}, n={n_max}, N={N_field}):")
    print(f"  {'alpha':>6}  {'ΔM_est':>14}  {'err%':>8}")
    results = []
    for al in alpha_vals:
        r = run(
            mesh_before=mesh,
            density=density,
            R_star=R_star,
            H=H,
            alpha=al,
            m_max=m_max,
            n_max=n_max,
            N_field=N_field,
            sigma_fn=sigma_fn,
            verbose=False,
        )
        dM_t = r["dM_true"] if dM_true is None else dM_true
        err = abs(r["dM_est"] - dM_t) / abs(dM_t) * 100
        print(f"  {al:>6.2f}  {r['dM_est']:>14.4e}  {err:>8.3f}%")
        results.append((al, r["dM_est"], err))
    return results


def study_N_field(
    mesh, density, R_star, H, sigma_fn, N_vals=None, alpha=2.0, m_max=5, n_max=8
):
    """Sweep number of field points."""
    if N_vals is None:
        N_vals = [100, 200, 400, 800]
    print(f"\nN_field sweep (α={alpha}, m={m_max}, n={n_max}):")
    print(f"  {'N':>5}  {'ΔM_est':>14}  {'err%':>8}  {'rms_rel':>9}")
    results = []
    for N in N_vals:
        r = run(
            mesh_before=mesh,
            density=density,
            R_star=R_star,
            H=H,
            alpha=alpha,
            m_max=m_max,
            n_max=n_max,
            N_field=N,
            sigma_fn=sigma_fn,
            verbose=False,
        )
        err = r["err_pct"]
        print(f"  {N:>5}  {r['dM_est']:>14.4e}  {err:>8.3f}%  {r['rms_after']:>9.3e}")
        results.append((N, r["dM_est"], err))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── CONFIG ───────────────────────────────────────────────────────
    # To use your own mesh, replace this block:
    #
    import mesh_utility

    vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, int)
    MESH = trimesh.Trimesh(vertices=vertices, faces=faces)
    DENSITY = 1.0  # kg/LU³ (Eros bulk density)
    R_STAR = 0.1  # [LU] — set to your cylinder radius
    H = 0.5  # [LU] — cylinder height

    # Basis parameters
    ALPHA = 100.0  # Bessel extension: zeros at 2R* → σ can be nonzero at R*
    M_MAX = 10  # azimuthal orders 0..4
    N_MAX = 10  # radial modes 1..8

    # Surface mass (replace with your own callable σ(ρ,φ) → kg/m²):
    SIGMA_FN = None  # None → built-in test pattern

    # ── MAIN RUN ─────────────────────────────────────────────────────
    result = run(
        mesh_before=MESH,
        density=DENSITY,
        R_star=R_STAR,
        H=H,
        alpha=ALPHA,
        m_max=M_MAX,
        n_max=N_MAX,
        N_field=400,
        sigma_fn=SIGMA_FN,
        verbose=True,
    )

    # ── SENSITIVITY SWEEPS ──────────────────────────────────────────
    mesh = result["mesh_before"]
    R_star = result["R_star"]
    H_val = result["H"]

    alpha_study = study_alpha(mesh, DENSITY, R_star, H_val, SIGMA_FN)
    N_study = study_N_field(mesh, DENSITY, R_star, H_val, SIGMA_FN)

    # ── FIGURE ──────────────────────────────────────────────────────
    plot_results(result)

    print("\nDone.")
