"""
Bennu Pre-Tag / After-Tag  — Cylindrical Harmonic Gravity Fitting
==================================================================
Pipeline
--------
1.  Load BENNU_preTag.obj  and  BENNU_afterTag.obj
2.  Normalise both meshes (centre + scale to unit sphere)
3.  Compute polyhedral gravity at cylinder field points for BOTH states
4.  Fit cylindrical harmonic coefficients (before & after) via least squares
5.  Form ΔA = A_after − A_before
6.  Wahr-like inversion  →  ΔM  and  Δσ(ρ,φ) surface-density map
7.  Diagnostic plots

Formula reference (Section 2 of companion code)
------------------------------------------------
Basis:
    U(ρ,φ,z) = Σ_{m,n}  J_m(k_mn ρ) exp(−k_mn z)
               × [A_mn cos(mφ) + B_mn sin(mφ)]
    k_mn = j_{m,n} / (α R*)     (zeros of J_m placed at αR*)

Radial gradient:
    ∂U/∂ρ  = Σ k_mn J_m'(k_mn ρ) exp(−k_mn z) [A cos + B sin]

Azimuthal gradient:
    (1/ρ) ∂U/∂φ = Σ (m/ρ) J_m exp(−k_mn z) [−A sin + B cos]

Axial gradient:
    ∂U/∂z  = Σ (−k_mn) J_m exp(−k_mn z) [A cos + B sin]

Mass recovery (Wahr):
    ΔM = (R*/G) Σ_{n=1}^{N} J_1(j_{0n}/α) ΔA_{0n}

Surface density map:
    Δσ(ρ,φ) = 1/(2π G α R*) Σ_{m,n} j_{mn} J_m(k_{mn}ρ)
               × [ΔA_{mn} cos(mφ) + ΔB_{mn} sin(mφ)]
"""

import numpy as np
import trimesh
from scipy.special import jv as BesselJ, jn_zeros
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches
from matplotlib.gridspec import GridSpec
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
from tqdm import tqdm
import time, os, sys

# ── physical / plotting constants ─────────────────────────────────────────
G_SI = 6.674e-11  # m³ kg⁻¹ s⁻²  (used for real-unit estimates)
G_LU = 6.674e-11  # same value; keep symbolic distinction

COLOR_PALETTE = ["#d7191c", "#fdae61", "#2c7bb6", "#1a9641", "#762a83", "#e66101"]
mpl.rcParams.update(
    {
        "text.usetex": False,  # set True if LaTeX is available
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=COLOR_PALETTE),
    }
)

SEP = "=" * 65
DASH = "─" * 65


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — MESH UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


def load_and_normalise(path: str, rotate_x_deg: float = -90.0) -> trimesh.Trimesh:
    """
    Load an OBJ (or any trimesh-supported format), centre at origin,
    scale so the longest bounding-box axis = 1 LU, then apply an
    optional rotation about the X-axis (matching the original script).

    Returns the normalised trimesh.Trimesh.
    """
    mesh = trimesh.load(path, force="mesh")
    # Centre
    mesh.apply_translation(-mesh.centroid)
    # Scale to unit bounding box (longest axis → 1 LU)
    scale = 1.0 / np.max(mesh.bounding_box.extents)
    mesh.apply_scale(scale)
    # Rotate around X-axis (same convention as original script's Y rotation)
    if rotate_x_deg != 0.0:
        angle = np.radians(rotate_x_deg)
        R4 = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(angle), -np.sin(angle), 0],
                [0, np.sin(angle), np.cos(angle), 0],
                [0, 0, 0, 1],
            ]
        )
        mesh.apply_transform(R4)
    return mesh


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — POLYHEDRAL GRAVITY  (polyhedral_gravity package)
# ═══════════════════════════════════════════════════════════════════════════


def make_evaluable(vertices, faces, density):
    """
    Build a GravityEvaluable from a mesh + density using the
    polyhedral_gravity package (Tsoulis / Werner analytic method).

    Parameters
    ----------
    vertices : (Nv, 3) array  — vertex coordinates [LU]
    faces    : (Nf, 3) int    — triangle face indices
    density  : float          — bulk density [kg/LU³]

    Returns
    -------
    evaluable : GravityEvaluable — callable for potential/acceleration/tensor
    """
    poly = Polyhedron(
        polyhedral_source=(
            np.asarray(vertices, dtype=float),
            np.asarray(faces, dtype=int),
        ),
        density=density,
        integrity_check=PolyhedronIntegrity.DISABLE,
    )
    return GravityEvaluable(poly)


def eval_gravity(evaluable, field_pts):
    """
    Evaluate gravitational potential and acceleration at each field point.

    Calls the GravityEvaluable one point at a time (parallel=False) with a
    tqdm progress bar, exactly as in the original Bennu/Eros scripts.

    Parameters
    ----------
    evaluable : GravityEvaluable
    field_pts : (N, 3) array — Cartesian query points [LU]

    Returns
    -------
    U, gx, gy, gz : each (N,) — potential and Cartesian acceleration components
    """
    N = len(field_pts)
    U = np.zeros(N)
    gx = np.zeros(N)
    gy = np.zeros(N)
    gz = np.zeros(N)

    for i, pt in enumerate(tqdm(field_pts, desc="  evaluating gravity", leave=False)):
        potential, acceleration, _ = evaluable(computation_points=pt, parallel=False)
        U[i] = potential
        gx[i] = acceleration[0]
        gy[i] = acceleration[1]
        gz[i] = acceleration[2]

    return U, gx, gy, gz


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — FIELD POINT GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def make_cylinder_field_points(
    center: np.ndarray,
    R_star: float,
    H: float,
    N: int = 1000,
    seed: int = 1,
    z_min_frac: float = 0.04,
    z_max_frac: float = 0.70,
    distribution: str = "else",
):
    """
    Generate N random field points INSIDE a cylinder of radius R_star and
    height H, centred at `center`.

    Three-band distribution concentrates points near the base (most sensitive
    to surface mass changes) while retaining mid- and far-field coverage for
    good conditioning.

    Returns
    -------
    rp, pp, zp : cylindrical coords (N,) arrays
    pts_cart   : (N, 3) Cartesian  [LU], shifted by center
    """
    np.random.seed(seed)
    z_lo = z_min_frac * R_star
    z_hi = z_max_frac * H

    if distribution == "three_bands":
        N1 = N // 3
        N2 = N // 3
        N3 = N - 2 * (N // 3)
        bands = [
            (N1, z_lo, 0.05 * R_star),
            (N2, 0.05 * R_star, 0.20 * R_star),
            (N3, 0.20 * R_star, z_hi),
        ]
        rp_list, pp_list, zp_list = [], [], []
        for Nb, za, zb in bands:
            rp_list.append(np.sqrt(np.random.uniform(0, R_star**2, Nb)))
            pp_list.append(np.random.uniform(0, 2 * np.pi, Nb))
            zp_list.append(np.random.uniform(za, zb, Nb))
        rp = np.concatenate(rp_list)
        pp = np.concatenate(pp_list)
        zp = np.concatenate(zp_list)
    else:
        rp = np.sqrt(np.random.uniform(0, R_star**2, N))
        pp = np.random.uniform(0, 2 * np.pi, N)
        zp = np.random.uniform(z_lo, z_hi, N)

    x = rp * np.cos(pp) + center[0]
    y = rp * np.sin(pp) + center[1]
    z = zp + center[2]
    pts_cart = np.column_stack([x, y, z])
    return rp, pp, zp, pts_cart


def cart_to_cyl_g(gx, gy, phi_pts):
    """Convert Cartesian (gx, gy) → cylindrical (gρ, gφ)."""
    gr = gx * np.cos(phi_pts) + gy * np.sin(phi_pts)
    gphi = -gx * np.sin(phi_pts) + gy * np.cos(phi_pts)
    return gr, gphi


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — CYLINDRICAL HARMONIC DESIGN MATRIX  (Bessel-Fourier basis)
# ═══════════════════════════════════════════════════════════════════════════


def build_design_matrix(rho_pts, phi_pts, z_pts, R_alpha, m_max, n_max):
    """
    Assemble 4N × 2·m_max·n_max design matrix for simultaneous fitting of
    U, gρ, gφ, gz at each field point.

    Basis:
        U = Σ_{m,n} J_m(k_mn ρ) exp(−k_mn z) [A_mn cos(mφ) + B_mn sin(mφ)]
        k_mn = j_{m,n} / (α R*)

    Column order (2 cols per (m,n) pair):
        col 2*(m*n_max + n-1)     → A_mn  (cosine amplitude)
        col 2*(m*n_max + n-1) + 1 → B_mn  (sine  amplitude)

    Row order per point i  (rows 4i..4i+3):
        4i   → U
        4i+1 → gρ = ∂U/∂ρ
        4i+2 → gφ = (1/ρ) ∂U/∂φ
        4i+3 → gz = ∂U/∂z
    """
    zeros_dict = {m: jn_zeros(m, n_max) for m in range(m_max)}
    N = len(rho_pts)
    N_par = 2 * m_max * n_max
    A = np.zeros((4 * N, N_par))

    for i, (rh, ph, z) in enumerate(zip(rho_pts, phi_pts, z_pts)):
        rs = max(rh, 1e-12)  # guard against ρ = 0
        for m in range(m_max):
            for n in range(1, n_max + 1):
                jmn = zeros_dict[m][n - 1]
                kmn = jmn / R_alpha
                x = kmn * rh
                Ez = np.exp(-kmn * z)
                Jm = BesselJ(m, x)
                dJm = 0.5 * (BesselJ(m - 1, x) - BesselJ(m + 1, x))
                cp = np.cos(m * ph)
                sp = np.sin(m * ph)
                col = 2 * (m * n_max + (n - 1))

                # U
                A[4 * i, col] = Jm * Ez * cp
                A[4 * i, col + 1] = Jm * Ez * sp
                # gρ
                A[4 * i + 1, col] = kmn * dJm * Ez * cp
                A[4 * i + 1, col + 1] = kmn * dJm * Ez * sp
                # gφ
                A[4 * i + 2, col] = -m / rs * Jm * Ez * sp
                A[4 * i + 2, col + 1] = m / rs * Jm * Ez * cp
                # gz
                A[4 * i + 3, col] = -kmn * Jm * Ez * cp
                A[4 * i + 3, col + 1] = -kmn * Jm * Ez * sp

    return A, zeros_dict


def assemble_obs_vector(U, gr, gphi, gz):
    """Interleave [U, gρ, gφ, gz] per point into (4N,) vector."""
    N = len(U)
    b = np.zeros(4 * N)
    b[0::4] = U
    b[1::4] = gr
    b[2::4] = gphi
    b[3::4] = gz
    return b


def fit_coefficients(A_des, U_obs, gr_obs, gphi_obs, gz_obs):
    """
    Least-squares fit:  A_des @ coeffs ≈ b

    Returns coeffs, fit-rms, relative-rms.
    """
    b = assemble_obs_vector(U_obs, gr_obs, gphi_obs, gz_obs)
    coeffs, _, _, _ = lstsq(A_des, b)
    resid = A_des @ coeffs - b
    rms = np.sqrt(np.mean(resid**2))
    rel = rms / (np.sqrt(np.mean(b**2)) + 1e-30)
    return coeffs, rms, rel


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — WAHR-LIKE INVERSION  (ΔM and Δσ recovery)
# ═══════════════════════════════════════════════════════════════════════════


def wahr_invert(
    delta_coeffs, R_star, alpha, n_max, m_max, zeros_dict, n_rho=80, n_phi=120
):
    """
    Recover mass change ΔM [kg or LU units] and surface-density map Δσ(ρ,φ)
    from ΔA = A_after − A_before coefficient vector.

    Mass formula  (m=0 modes only):
    ─────────────────────────────
        ΔM = (R* / G) Σ_{n=1}^{N} J_1(j_{0n}/α) · ΔA_{0n}

    Derivation: integrate the m=0 basis function over the disk of radius R*,
    use ∫_0^R* J_0(k_0n ρ) ρ dρ = (R*/k_0n) J_1(k_0n R*)
    and k_0n R* = j_{0n}/α, then relate U(z→0⁺) to surface density via
    U_layer = G σ/|r−r'| → σ = U / (2πG) in the thin-layer limit.

    Surface density map:
    ────────────────────
        Δσ(ρ,φ) = 1/(2π G α R*) Σ_{m,n} j_{mn} J_m(k_{mn}ρ)
                  × [ΔA_{mn} cos(mφ) + ΔB_{mn} sin(mφ)]

    Parameters
    ----------
    delta_coeffs : (N_par,) array  — ΔA coefficients
    R_star       : float           — cylinder radius [LU]
    alpha        : float           — Bessel extension (α > 1)
    n_max, m_max : int             — truncation
    zeros_dict   : dict {m: zeros array}

    Returns
    -------
    delta_M   : float         — estimated mass change [same units as G·rho·LU³]
    sigma_map : (n_rho,n_phi) — surface density map [kg/LU²]
    RHO, PHI  : meshgrid arrays for the map
    """
    R_alpha = alpha * R_star

    # ── ΔM (m=0 only) ──────────────────────────────────────────────────
    delta_M = 0.0
    for n in range(1, n_max + 1):
        j0n = zeros_dict[0][n - 1]
        col = 2 * (0 * n_max + (n - 1))  # m=0 column index
        delta_M += (R_star / G_LU) * BesselJ(1, j0n / alpha) * delta_coeffs[col]

    # ── Δσ map ─────────────────────────────────────────────────────────
    rho_1d = np.linspace(0.02 * R_star, 0.98 * R_star, n_rho)
    phi_1d = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    RHO, PHI = np.meshgrid(rho_1d, phi_1d, indexing="ij")
    sigma_map = np.zeros_like(RHO)
    pref = 1.0 / (2.0 * np.pi * G_LU * R_alpha)

    for m in range(m_max):
        for n in range(1, n_max + 1):
            jmn = zeros_dict[m][n - 1]
            kmn = jmn / R_alpha
            col = 2 * (m * n_max + (n - 1))
            bess = BesselJ(m, kmn * RHO)
            sigma_map += (
                pref
                * jmn
                * bess
                * (
                    delta_coeffs[col] * np.cos(m * PHI)
                    + delta_coeffs[col + 1] * np.sin(m * PHI)
                )
            )

    return delta_M, sigma_map, RHO, PHI


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def run_bennu(
    path_pre: str = "3dmeshes/BENNU_preTag.obj",
    path_post: str = "3dmeshes/BENNU_afterTag.obj",
    density: float = 1.0,  # [kg/LU³]  ~1260 kg/m³ for Bennu
    # Cylinder parameters (relative to normalised mesh, LU)
    cyl_center=np.array([0.0, 0.0, 0.1]),
    R_star: float = 0.10,  # cylinder radius [LU]
    H: float = 0.50,  # cylinder height [LU]
    # Basis parameters
    alpha: float = 2.0,  # Bessel extension α > 1
    m_max: int = 5,  # azimuthal orders 0..m_max-1
    n_max: int = 8,  # radial modes  1..n_max
    N_field: int = 1000,
    verbose: bool = True,
):
    """
    Full before/after pipeline for Bennu.

    Steps
    -----
    1. Load & normalise BENNU_preTag  and  BENNU_afterTag meshes
    2. Generate cylinder field points
    3. Evaluate polyhedral gravity at field points for BOTH meshes
    4. Fit cylindrical harmonic coefficients via LS for each state
    5. Form ΔA = c_after − c_before
    6. Wahr inversion → ΔM and Δσ(ρ,φ)
    7. Estimate effective density change Δρ within cylinder volume

    Returns a dict with all intermediate and final results.
    """
    if verbose:
        print(SEP)
        print("  BENNU  Pre/After Tag — Cylindrical Harmonic Gravity Fitting")
        print(SEP)

    # ── 1. LOAD MESHES ─────────────────────────────────────────────────
    if verbose:
        print(f"\n[1] Loading meshes …")
    mesh_pre = load_and_normalise(path_pre, rotate_x_deg=90.0)
    mesh_post = load_and_normalise(path_post, rotate_x_deg=90.0)
    if verbose:
        print(
            f"    Pre-tag : {len(mesh_pre.vertices):>6} verts, "
            f"{len(mesh_pre.faces):>6} faces"
        )
        print(
            f"    Post-tag: {len(mesh_post.vertices):>6} verts, "
            f"{len(mesh_post.faces):>6} faces"
        )

    R_alpha = alpha * R_star
    if verbose:
        print(f"\n    Cylinder  centre={cyl_center}, R*={R_star}, H={H}")
        print(f"    Basis     m_max={m_max}, n_max={n_max}, α={alpha}")
        print(f"    Bessel zeros at α R* = {R_alpha:.4f} LU")

    # ── 2. FIELD POINTS ─────────────────────────────────────────────────
    if verbose:
        print(f"\n[2] Generating {N_field} field points …")
    rp, pp, zp, pts_cart = make_cylinder_field_points(
        center=cyl_center,
        R_star=R_star,
        H=H,
        N=N_field,
        z_min_frac=0.04,
        z_max_frac=0.70,
    )
    if verbose:
        print(
            f"    ρ: [{rp.min():.4f}, {rp.max():.4f}]  "
            f"z: [{zp.min():.4f}, {zp.max():.4f}]"
        )

    # ── 3. POLYHEDRAL GRAVITY (polyhedral_gravity package) ─────────────
    if verbose:
        print(f"\n[3] Computing polyhedral gravity …")

    if verbose:
        print("    Building GravityEvaluable for pre-tag  …", end=" ", flush=True)
    eval_pre = make_evaluable(mesh_pre.vertices, mesh_pre.faces, density)
    if verbose:
        print("done")

    if verbose:
        print("    Building GravityEvaluable for post-tag …", end=" ", flush=True)
    eval_post = make_evaluable(mesh_post.vertices, mesh_post.faces, density)
    if verbose:
        print("done")

    if verbose:
        print("    Evaluating pre-tag  gravity …")
    t0 = time.time()
    U_pre, gx_pre, gy_pre, gz_pre = eval_gravity(eval_pre, pts_cart)
    gr_pre, gphi_pre = cart_to_cyl_g(gx_pre, gy_pre, pp)
    if verbose:
        print(f"    done  ({time.time()-t0:.1f}s)")

    if verbose:
        print("    Evaluating post-tag gravity …")
    t0 = time.time()
    U_post, gx_post, gy_post, gz_post = eval_gravity(eval_post, pts_cart)
    gr_post, gphi_post = cart_to_cyl_g(gx_post, gy_post, pp)
    if verbose:
        print(f"    done  ({time.time()-t0:.1f}s)")

    # ── 4. DESIGN MATRIX & LS FIT ───────────────────────────────────────
    N_par = 2 * m_max * n_max
    if verbose:
        print(
            f"\n[4] Building design matrix  ({4*N_field} × {N_par}) …",
            end=" ",
            flush=True,
        )
    A_des, zeros_dict = build_design_matrix(rp, pp, zp, R_alpha, m_max, n_max)
    if verbose:
        print("done")

    if verbose:
        print("    Fitting pre-tag  coefficients …", end=" ", flush=True)
    t0 = time.time()
    c_pre, rms_pre, rel_pre = fit_coefficients(A_des, U_pre, gr_pre, gphi_pre, gz_pre)
    if verbose:
        print(f"done ({time.time()-t0:.1f}s)  " f"RMS={rms_pre:.3e}  rel={rel_pre:.4f}")

    if verbose:
        print("    Fitting post-tag coefficients …", end=" ", flush=True)
    t0 = time.time()
    c_post, rms_post, rel_post = fit_coefficients(
        A_des, U_post, gr_post, gphi_post, gz_post
    )
    if verbose:
        print(
            f"done ({time.time()-t0:.1f}s)  " f"RMS={rms_post:.3e}  rel={rel_post:.4f}"
        )

    # ── 5. DELTA COEFFICIENTS ──────────────────────────────────────────
    d_coeffs = c_post - c_pre

    # ── 6. WAHR INVERSION ──────────────────────────────────────────────
    if verbose:
        print(f"\n[5] Wahr inversion …", end=" ", flush=True)
    dM_est, sigma_map, RHO, PHI = wahr_invert(
        d_coeffs, R_star, alpha, n_max, m_max, zeros_dict, n_rho=80, n_phi=120
    )
    if verbose:
        print("done")

    # ── 7. DENSITY CHANGE ESTIMATE ─────────────────────────────────────
    # Effective volume of cylinder  V_cyl = π R*² H
    # Effective density change:  Δρ = ΔM / V_cyl
    V_cyl = np.pi * R_star**2 * H
    delta_rho = dM_est / V_cyl  # [kg/LU³] or dimensionless

    # Gravity signal change magnitude
    dU_rms = np.std(U_post - U_pre)
    dgz_rms = np.std(gz_post - gz_pre)
    U_pre_rms = np.sqrt(np.mean(U_pre**2))
    sig_ratio = dU_rms / (U_pre_rms + 1e-30)

    # Coefficient L2 norm ratio
    c_pre_norm = np.linalg.norm(c_pre)
    d_coeffs_norm = np.linalg.norm(d_coeffs)
    coeff_ratio = d_coeffs_norm / (c_pre_norm + 1e-30)

    if verbose:
        print(f"\n{DASH}")
        print(f"  RESULTS")
        print(DASH)
        print(f"  Estimated ΔM          = {dM_est:.6e}  [kg·LU³/m³ units]")
        print(f"  Cylinder volume       = {V_cyl:.6e}  LU³")
        print(f"  Estimated Δρ          = {delta_rho:.6e}  [kg/LU³]")
        print(f"  ΔU RMS (field pts)    = {dU_rms:.3e}")
        print(f"  Δgz RMS (field pts)   = {dgz_rms:.3e}")
        print(f"  Signal ratio ΔU/U     = {sig_ratio:.4e}")
        print(f"  ||Δc|| / ||c||        = {coeff_ratio:.4e}")
        print(DASH)

    return dict(
        # meshes
        mesh_pre=mesh_pre,
        mesh_post=mesh_post,
        density=density,
        # cylinder geometry
        cyl_center=cyl_center,
        R_star=R_star,
        H=H,
        alpha=alpha,
        m_max=m_max,
        n_max=n_max,
        R_alpha=R_alpha,
        # field points
        rp=rp,
        pp=pp,
        zp=zp,
        pts_cart=pts_cart,
        # gravity
        U_pre=U_pre,
        gz_pre=gz_pre,
        U_post=U_post,
        gz_post=gz_post,
        dU=U_post - U_pre,
        dgz=gz_post - gz_pre,
        # cylindrical gravity
        gr_pre=gr_pre,
        gphi_pre=gphi_pre,
        gr_post=gr_post,
        gphi_post=gphi_post,
        # coefficients
        design_matrix=A_des,
        zeros_dict=zeros_dict,
        c_pre=c_pre,
        rms_pre=rms_pre,
        rel_pre=rel_pre,
        c_post=c_post,
        rms_post=rms_post,
        rel_post=rel_post,
        d_coeffs=d_coeffs,
        # inversion
        dM_est=dM_est,
        sigma_map=sigma_map,
        RHO=RHO,
        PHI=PHI,
        # derived
        V_cyl=V_cyl,
        delta_rho=delta_rho,
        dU_rms=dU_rms,
        dgz_rms=dgz_rms,
        sig_ratio=sig_ratio,
        coeff_ratio=coeff_ratio,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — DIAGNOSTIC PLOTS
# ═══════════════════════════════════════════════════════════════════════════


def plot_results(res, outdir=None):
    """
    Same inputs/outputs as before:
        input  : res, outdir=None
        output : fig1, fig2, fig3

    Cleaner plotting:
      - no black background
      - visible Bennu mesh edges
      - clearer cylinder and field points
      - better color scaling
    """
    import os
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    np.random.seed(0)

    R = res["R_star"]
    H = res["H"]
    ctr = res["cyl_center"]
    RHO = res["RHO"]
    PHI = res["PHI"]
    X = RHO * np.cos(PHI)
    Y = RHO * np.sin(PHI)

    rp = res["rp"]
    pp = res["pp"]
    zp = res["zp"]
    dU = res["dU"]
    dgz = res["dgz"]
    U_pre = res["U_pre"]
    gz_pre = res["gz_pre"]
    U_post = res["U_post"]
    gz_post = res["gz_post"]
    sm = res["sigma_map"]

    os.makedirs(outdir or ".", exist_ok=True)

    def _nice_3d_axes(ax):
        ax.set_facecolor("white")
        ax.grid(False)

        # remove the 3D box panes
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor("white")

        # remove box/grid lines
        ax.xaxis._axinfo["grid"]["linewidth"] = 0
        ax.yaxis._axinfo["grid"]["linewidth"] = 0
        ax.zaxis._axinfo["grid"]["linewidth"] = 0

        ax.tick_params(labelsize=8)
        ax.set_xlabel("$X$ [LU]", fontsize=8)
        ax.set_ylabel("$Y$ [LU]", fontsize=8)
        ax.set_zlabel("$Z$ [LU]", fontsize=8)

        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

        ax.view_init(elev=22, azim=38)

    def _draw_mesh_visible(ax, mesh, face_color=COLOR_PALETTE[-1], alpha=0.30):
        if hasattr(mesh, "vertices"):
            V = np.asarray(mesh.vertices)
        elif hasattr(mesh, "points"):
            V = np.asarray(mesh.points)
        else:
            V = np.asarray(mesh[0])

        if hasattr(mesh, "faces"):
            F = np.asarray(mesh.faces)
        elif hasattr(mesh, "triangles"):
            F = np.asarray(mesh.triangles)
        else:
            F = np.asarray(mesh[1])

        poly = Poly3DCollection(
            V[F],
            alpha=0.3,
            edgecolor="k",
            linewidths=0.15,
            facecolor=COLOR_PALETTE[-1],
        )
        ax.add_collection3d(poly)

        max_range = np.ptp(V, axis=0).max()
        mid = V.mean(axis=0)
        pad = 0.58 * max_range
        ax.set_xlim(mid[0] - pad, mid[0] + pad)
        ax.set_ylim(mid[1] - pad, mid[1] + pad)
        ax.set_zlim(mid[2] - pad, mid[2] + pad)

        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

    def _draw_cylinder_wireframe_clean(ax, ctr, R, H, color="tab:cyan", alpha=0.9):
        th = np.linspace(0, 2 * np.pi, 160)

        # ctr is the center of the lower disk, not the middle of the cylinder
        z0 = ctr[2]
        z1 = ctr[2] + H

        for z in [z0, z1]:
            ax.plot(
                ctr[0] + R * np.cos(th),
                ctr[1] + R * np.sin(th),
                z * np.ones_like(th),
                color=color,
                lw=1.6,
                alpha=alpha,
            )

        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            x = ctr[0] + R * np.cos(a)
            y = ctr[1] + R * np.sin(a)
            ax.plot([x, x], [y, y], [z0, z1], color=color, lw=0.9, alpha=0.65)

    # ================================================================
    # FIGURE 1 — geometry
    # ================================================================
    fig1 = plt.figure(figsize=(18, 6))
    gs1 = GridSpec(1, 3, figure=fig1, wspace=0.12)

    titles_3d = [
        ("Bennu PRE-tag", res["mesh_pre"], "#6baed6"),
        ("Bennu POST-tag", res["mesh_post"], "#fdae6b"),
        (
            "Field points inside cylinder\ncolored by |gz| pre-tag",
            res["mesh_pre"],
            "#bdbdbd",
        ),
    ]

    for col, (ttl, mesh, mc) in enumerate(titles_3d):
        ax = fig1.add_subplot(gs1[col], projection="3d")
        _nice_3d_axes(ax)

        mesh_alpha = 0.32 if col < 2 else 0.12
        _draw_mesh_visible(ax, mesh, face_color=mc, alpha=mesh_alpha)
        _draw_cylinder_wireframe_clean(ax, ctr, R, H)

        if col == 2:
            mag = np.abs(gz_pre)
            lo, hi = np.percentile(mag, [2, 98])
            sc = ax.scatter(
                rp * np.cos(pp) + ctr[0],
                rp * np.sin(pp) + ctr[1],
                zp + ctr[2],
                c=mag,
                cmap="plasma",
                s=10,
                alpha=0.85,
                edgecolors="none",
                vmin=lo,
                vmax=hi,
                depthshade=True,
            )
            cb = fig1.colorbar(sc, ax=ax, pad=0.08, shrink=0.7)
            cb.set_label(r"$|g_z|$ pre-tag", fontsize=9)

        ax.set_title(ttl, fontsize=11, fontweight="bold", pad=8)

    fig1.suptitle(
        "Bennu Measurement Geometry\n"
        f"Cylinder: R* = {R:.3f} LU, H = {H:.3f} LU, "
        f"N_field = {len(rp)}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    if outdir:
        p = os.path.join(outdir, "fig1_geometry.png")
        fig1.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved → {p}")

    # ================================================================
    # FIGURE 2 — gravity change
    # ================================================================
    fig2, axes2 = plt.subplots(
        2, 3, figsize=(16, 10), gridspec_kw={"hspace": 0.42, "wspace": 0.35}
    )
    fig2.suptitle(
        "Gravity Signal at Field Points — Pre-Tag vs Post-Tag",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    def _scatter_cyl(ax, vals, title, cmap, label, symmetric=False):
        if symmetric:
            vmax = np.percentile(np.abs(vals), 98)
            vmin = -vmax
        else:
            vmin, vmax = np.percentile(vals, [2, 98])

        sc = ax.scatter(
            rp,
            zp,
            c=vals,
            cmap=cmap,
            s=13,
            alpha=0.9,
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
        )
        cb = fig2.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(label, fontsize=9)

        ax.set_xlabel(r"$\rho$ [LU]", fontsize=10)
        ax.set_ylabel("z [LU]", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_facecolor("#fafafa")
        ax.grid(True, alpha=0.25)

    _scatter_cyl(axes2[0, 0], U_pre, "Potential U — PRE-tag", "viridis", "U [LU²/s²]")
    _scatter_cyl(axes2[0, 1], U_post, "Potential U — POST-tag", "viridis", "U [LU²/s²]")
    _scatter_cyl(
        axes2[0, 2],
        dU,
        r"Potential change $\Delta U$",
        "RdBu_r",
        "ΔU [LU²/s²]",
        symmetric=True,
    )

    _scatter_cyl(
        axes2[1, 0],
        gz_pre,
        r"Vertical gravity $g_z$ — PRE-tag",
        "coolwarm",
        "gz [LU/s²]",
        symmetric=True,
    )
    _scatter_cyl(
        axes2[1, 1],
        gz_post,
        r"Vertical gravity $g_z$ — POST-tag",
        "coolwarm",
        "gz [LU/s²]",
        symmetric=True,
    )
    _scatter_cyl(
        axes2[1, 2],
        dgz,
        r"Vertical gravity change $\Delta g_z$",
        "RdBu_r",
        "Δgz [LU/s²]",
        symmetric=True,
    )

    txt = (
        f"Mean ΔU  = {np.mean(dU):+.3e}\n"
        f"Mean Δgz = {np.mean(dgz):+.3e}\n"
        f"RMS  ΔU  = {res['dU_rms']:.3e}\n"
        f"RMS  Δgz = {res['dgz_rms']:.3e}\n"
        f"ΔU/U     = {res['sig_ratio']:.3e}"
    )
    axes2[1, 2].text(
        0.03,
        0.97,
        txt,
        transform=axes2[1, 2].transAxes,
        fontsize=8.5,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, ec="0.7"),
    )

    if outdir:
        p = os.path.join(outdir, "fig2_gravity_change.png")
        fig2.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved → {p}")

    # ================================================================
    # FIGURE 3 — recovered mass change
    # ================================================================
    fig3 = plt.figure(figsize=(17, 11))
    gs3 = GridSpec(2, 3, figure=fig3, hspace=0.48, wspace=0.42)

    fig3.suptitle(
        f"Mass-Change Recovery via Cylindrical Harmonic Inversion\n"
        f"Estimated ΔM = {res['dM_est']:.4e}  |  "
        f"Estimated Δρ = {res['delta_rho']:.4e}  |  "
        f"α = {res['alpha']}  m_max = {res['m_max']}  n_max = {res['n_max']}",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )

    tc = np.linspace(0, 2 * np.pi, 300)

    ax3A = fig3.add_subplot(gs3[0, 0])
    vmax = np.percentile(np.abs(sm), 98)
    cA = ax3A.pcolormesh(X, Y, sm, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
    fig3.colorbar(cA, ax=ax3A, label="Δσ [kg/LU²]")
    ax3A.plot(R * np.cos(tc), R * np.sin(tc), "k--", lw=1.2, alpha=0.65)
    ax3A.set_aspect("equal")
    ax3A.set_xlabel("x [LU]")
    ax3A.set_ylabel("y [LU]")
    ax3A.set_title(r"Recovered $\Delta\sigma(\rho,\phi)$", fontweight="bold")
    ax3A.grid(True, alpha=0.25)

    ax3B = fig3.add_subplot(gs3[0, 1], projection="3d")
    surf = ax3B.plot_surface(
        X,
        Y,
        sm,
        cmap="RdBu_r",
        rstride=1,
        cstride=1,
        alpha=0.95,
        linewidth=0.15,
        edgecolor="0.35",
        antialiased=True,
    )
    fig3.colorbar(surf, ax=ax3B, shrink=0.65, pad=0.08, label="Δσ [kg/LU²]")
    _nice_3d_axes(ax3B)
    ax3B.set_title(r"3D recovered $\Delta\sigma$", fontweight="bold")

    ax3C = fig3.add_subplot(gs3[0, 2])
    rho_1d = RHO[:, 0]
    sm_mean = sm.mean(axis=1)
    sm_max = sm.max(axis=1)
    sm_min = sm.min(axis=1)
    ax3C.fill_between(rho_1d, sm_min, sm_max, alpha=0.25, label="min-max range")
    ax3C.plot(rho_1d, sm_mean, lw=2.2, label="azimuthal mean")
    ax3C.axhline(0, color="k", lw=0.8, ls="--")
    ax3C.axvline(R, color="0.5", lw=1, ls=":", label=f"R* = {R:.3f}")
    ax3C.set_xlabel(r"$\rho$ [LU]")
    ax3C.set_ylabel("Δσ [kg/LU²]")
    ax3C.set_title("Radial profile", fontweight="bold")
    ax3C.grid(True, alpha=0.25)
    ax3C.legend(fontsize=8)

    ax3D = fig3.add_subplot(gs3[1, 0])
    phi_deg = np.degrees(PHI[0, :])
    for frac in [0.25, 0.50, 0.75]:
        i_r = np.argmin(np.abs(rho_1d - frac * R))
        ax3D.plot(phi_deg, sm[i_r, :], lw=2, label=f"ρ = {rho_1d[i_r]:.3f} LU")
    ax3D.axhline(0, color="k", lw=0.8, ls="--")
    ax3D.set_xlabel("φ [deg]")
    ax3D.set_ylabel("Δσ [kg/LU²]")
    ax3D.set_title("Azimuthal profiles", fontweight="bold")
    ax3D.grid(True, alpha=0.25)
    ax3D.legend(fontsize=8)

    ax3E = fig3.add_subplot(gs3[1, 1])
    dc = res["d_coeffs"]
    n_max = res["n_max"]
    m_max = res["m_max"]
    mode_amp = np.sqrt(dc[0::2] ** 2 + dc[1::2] ** 2)

    x_pos = 0
    tick_pos, tick_lbl = [], []
    cmap_m = plt.cm.tab10
    for m in range(m_max):
        amps = mode_amp[m * n_max : (m + 1) * n_max]
        xs = np.arange(x_pos, x_pos + n_max)
        ax3E.bar(xs, amps, color=cmap_m(m / max(m_max - 1, 1)), alpha=0.85)
        tick_pos.append(x_pos + n_max / 2 - 0.5)
        tick_lbl.append(f"m={m}")
        x_pos += n_max + 1
        ax3E.axvline(x_pos - 1, color="0.85", lw=0.8)

    ax3E.set_xticks(tick_pos)
    ax3E.set_xticklabels(tick_lbl, fontsize=9)
    ax3E.set_ylabel(r"Amplitude $\sqrt{\Delta A^2+\Delta B^2}$")
    ax3E.set_title("Coefficient spectrum by order m", fontweight="bold")
    ax3E.grid(True, axis="y", alpha=0.25)

    ax3F = fig3.add_subplot(gs3[1, 2])
    ax3F.axis("off")

    summary = (
        f"CYLINDER GEOMETRY\n"
        f"  Radius R*    = {R:.4f} LU\n"
        f"  Height H     = {H:.4f} LU\n"
        f"  Volume       = {res['V_cyl']:.4e} LU³\n\n"
        f"LS FIT QUALITY\n"
        f"  RMS pre      = {res['rms_pre']:.3e}\n"
        f"  RMS post     = {res['rms_post']:.3e}\n"
        f"  ||Δc||/||c|| = {res['coeff_ratio']:.4e}\n\n"
        f"MASS CHANGE ESTIMATE\n"
        f"  ΔM           = {res['dM_est']:+.6e}\n"
        f"  Δρ           = {res['delta_rho']:+.6e} kg/LU³\n\n"
        f"GRAVITY SIGNAL CHANGE\n"
        f"  RMS ΔU       = {res['dU_rms']:.3e}\n"
        f"  RMS Δgz      = {res['dgz_rms']:.3e}\n"
        f"  ΔU/U         = {res['sig_ratio']:.3e}"
    )

    ax3F.text(
        0.05,
        0.95,
        summary,
        transform=ax3F.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", fc="#f8f8f8", ec="0.75"),
    )
    ax3F.set_title("Summary", fontsize=10, fontweight="bold")

    if outdir:
        p = os.path.join(outdir, "fig3_mass_change.png")
        fig3.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved → {p}")

    plt.show()
    return fig1, fig2, fig3


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── USER CONFIG ──────────────────────────────────────────────────────
    PATH_PRE = "3dmeshes/BENNU_preTag.obj"
    PATH_POST = "3dmeshes/BENNU_afterTag.obj"

    DENSITY = 1.0  # kg/LU³  (normalised units)
    CYL_CENTER = np.array([0.0, 0.0, 0.00])  # cylinder base centre [LU]
    R_STAR = 0.20  # cylinder radius       [LU]
    H = 0.50  # cylinder height       [LU]

    ALPHA = 2.0  # Bessel extension parameter  (α > 1 recommended)
    M_MAX = 5  # azimuthal orders 0 .. M_MAX-1
    N_MAX = 5  # radial modes     1 .. N_MAX
    N_FLD = 1000  # number of field points

    # ── RUN ──────────────────────────────────────────────────────────────
    result = run_bennu(
        path_pre=PATH_PRE,
        path_post=PATH_POST,
        density=DENSITY,
        cyl_center=CYL_CENTER,
        R_star=R_STAR,
        H=H,
        alpha=ALPHA,
        m_max=M_MAX,
        n_max=N_MAX,
        N_field=N_FLD,
        verbose=True,
    )

    # ── PLOT ─────────────────────────────────────────────────────────────
    fig1, fig2, fig3 = plot_results(result)

    print("\nDone.")
