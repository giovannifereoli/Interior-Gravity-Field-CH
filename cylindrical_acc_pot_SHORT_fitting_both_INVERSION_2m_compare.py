"""
Two-mascon TAG-style forward + inverse (GLOBAL SPHERICAL HARMONICS, from scratch)

✅ Full pipeline:
   (1) Forward synth field: constant-density polyhedron + TWO fixed-position mascon anomalies (Δmu1, Δmu2)
   (2) Fit GLOBAL spherical harmonics coefficients (linear LS) to that synthetic field (acc + pot)
       - points are all around the asteroid (spherical shell)
   (3) Given fitted SH coeffs + known polyhedron, estimate ONLY Δmu1, Δmu2 (positions fixed)
       - do LSQ (scipy least_squares) + MCMC (emcee)
✅ ALL plots at the END (no plt.show inside solvers)

Notes / conventions:
- SH expansion is real (C_lm, S_lm) about an origin "center".
- Potential model basis (no explicit GM factor): V = Σ a_p * basis_p(r,lat,lon)
  so coefficients have your LU/TU potential units built-in.
- Acceleration is a = -∇V computed in spherical components then mapped to Cartesian.

Dependencies:
  numpy, scipy, matplotlib, emcee (optional), corner (optional),
  polyhedral_gravity, mesh_utility
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.optimize import least_squares
from scipy.special import lpmv  # associated Legendre P_lm(x), x in [-1,1]

import mesh_utility
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable


# ======================================================================================
# Geometry + sampling (global points all around)
# ======================================================================================


def bounding_sphere_radius(vertices: np.ndarray, center: np.ndarray) -> float:
    return float(np.linalg.norm(vertices - center[None, :], axis=1).max())


def sample_points_in_spherical_shell(
    num_points: int,
    r_min: float,
    r_max: float,
    center: np.ndarray,
    seed: int = 1,
) -> np.ndarray:
    """
    Uniform in volume in a spherical shell: r in [r_min, r_max].
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, num_points)
    r = (r_min**3 + u * (r_max**3 - r_min**3)) ** (1.0 / 3.0)

    cos_th = rng.uniform(-1.0, 1.0, num_points)  # colatitude theta
    th = np.arccos(cos_th)
    lon = rng.uniform(0.0, 2.0 * np.pi, num_points)

    # spherical to cart (relative)
    x = r * np.sin(th) * np.cos(lon)
    y = r * np.sin(th) * np.sin(lon)
    z = r * np.cos(th)

    return np.column_stack((x, y, z)) + center[None, :]


def cart_to_sph(
    points: np.ndarray, center: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (r, lat, lon) where:
      lat = geocentric latitude in [-pi/2, pi/2]
      lon in [0, 2pi)
    """
    p = points - center[None, :]
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    lon = np.arctan2(y, x) % (2.0 * np.pi)
    lat = np.arcsin(np.clip(z / (r + 1e-30), -1.0, 1.0))
    return r, lat, lon


# ======================================================================================
# Poly gravity evaluation (baseline constant-density poly)
# ======================================================================================


def eval_poly_gravity(
    vertices, faces, density: float, points: np.ndarray, parallel: bool = False
):
    poly = Polyhedron(
        polyhedral_source=(np.asarray(vertices), np.asarray(faces)),
        density=density,
        integrity_check=PolyhedronIntegrity.DISABLE,
    )
    eval_poly = GravityEvaluable(poly)

    N = points.shape[0]
    pot = np.zeros(N, float)
    acc = np.zeros((N, 3), float)
    for i, p in enumerate(points):
        V, a, _ = eval_poly(computation_points=p, parallel=parallel)
        pot[i] = float(np.squeeze(V))
        acc[i] = np.squeeze(a)
    return pot, acc


# ======================================================================================
# Mascon anomalies (point masses)
# ======================================================================================


def mascon_potential_acc(
    points: np.ndarray, r_m: np.ndarray, mu: float, softening: float = 0.0
):
    """
    V = -mu / r
    a = -mu * dr / r^3
    """
    dr = points - r_m[None, :]
    r2 = np.sum(dr * dr, axis=1) + softening**2
    r = np.sqrt(r2)
    V = -mu / (r + 1e-30)
    a = -mu * dr / ((r[:, None] ** 3) + 1e-30)
    return V, a


import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.special import lpmv
from math import factorial

# ======================================================================================
# GLOBAL spherical harmonics (FULLY NORMALIZED, real C/S): parameterization + design matrices
#   - Uses scipy.special.lpmv(m,l,x), which INCLUDES the Condon–Shortley phase (-1)^m.
#   - Normalization here is the standard "fully normalized" geodesy/gravity one:
#       P̄_lm(x) = N_lm * P_lm(x)
#       N_lm = sqrt( (2 - δ_m0) (2l+1) (l-m)! / (l+m)! )
# ======================================================================================


@dataclass
class SHSpec:
    L: int  # max degree
    R_ref: float  # reference radius (LU)
    center: np.ndarray  # expansion center (3,)


def sh_num_params(L: int) -> int:
    return (L + 1) ** 2


def sh_param_indexing(L: int):
    """
    index -> (l,m,kind) with kind in {"C","S"}.
    Order:
      l=0: C00
      for l=1..L:
        m=0: C_l0
        m=1..l: C_lm, S_lm
    """
    ls, ms, ks = [0], [0], ["C"]
    for l in range(1, L + 1):
        ls.append(l)
        ms.append(0)
        ks.append("C")
        for m in range(1, l + 1):
            ls.append(l)
            ms.append(m)
            ks.append("C")
            ls.append(l)
            ms.append(m)
            ks.append("S")
    return np.array(ls, int), np.array(ms, int), np.array(ks, object)


def _N_lm(l: int, m: int) -> float:
    """Fully-normalization factor N_lm."""
    # (2 - δ_m0)
    fac = 1.0 if m == 0 else 2.0
    return np.sqrt(fac * (2 * l + 1) * factorial(l - m) / factorial(l + m))


def _Pbar_lm(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Fully-normalized associated Legendre P̄_lm(x) = N_lm * P_lm(x)."""
    return _N_lm(l, m) * lpmv(m, l, x)


def _dP_lm_dx(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """
    Derivative w.r.t x of associated Legendre P_lm(x) (UNnormalized).
    Stable identity:
      d/dx P_lm(x) = [l x P_lm(x) - (l+m) P_{l-1,m}(x)] / (x^2 - 1)
    """
    if l == 0:
        return np.zeros_like(x)
    P_lm = lpmv(m, l, x)
    P_lm1 = lpmv(m, l - 1, x)
    denom = x * x - 1.0
    denom = np.where(np.abs(denom) < 1e-14, np.sign(denom) * 1e-14, denom)
    return (l * x * P_lm - (l + m) * P_lm1) / denom


def _dPbar_lm_dx(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Derivative of fully-normalized P̄_lm(x): just multiply by N_lm."""
    return _N_lm(l, m) * _dP_lm_dx(l, m, x)


# --- you already have this somewhere ---
# def cart_to_sph(points: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
#     returns r, lat, lon  (lat in radians, lon in radians)


def sh_basis_potential_and_acc(
    spec: SHSpec,
    points: np.ndarray,
    l: int,
    m: int,
    kind: str,  # "C" or "S"
):
    """
    Returns:
      V_basis (N,)
      a_basis (N,3)   for coefficient amplitude = 1
    Using fully-normalized P̄_lm(sin(lat)).
    """
    r, lat, lon = cart_to_sph(points, spec.center)
    x = np.sin(lat)  # argument for P̄_lm
    coslat = np.cos(lat)

    # radial factor: (R_ref^l / r^(l+1))
    R = spec.R_ref
    k = (R**l) / ((r + 1e-30) ** (l + 1))

    Pbar = _Pbar_lm(l, m, x)

    if m == 0:
        trig = np.ones_like(lon)
        dtrig_dlon = np.zeros_like(lon)
    else:
        if kind == "C":
            trig = np.cos(m * lon)
            dtrig_dlon = -m * np.sin(m * lon)
        else:
            trig = np.sin(m * lon)
            dtrig_dlon = m * np.cos(m * lon)

    # Potential
    V = k * Pbar * trig

    # spherical derivatives
    dV_dr = -(l + 1) * V / (r + 1e-30)

    dPbar_dx = _dPbar_lm_dx(l, m, x)
    dV_dlat = k * dPbar_dx * coslat * trig

    dV_dlon = k * Pbar * dtrig_dlon

    # Accel spherical components
    a_r = -dV_dr
    a_lat = -(dV_dlat / (r + 1e-30))
    a_lon = -(dV_dlon / ((r + 1e-30) * (coslat + 1e-30)))

    # Convert to Cartesian
    cl = np.cos(lat)
    sl = np.sin(lat)
    co = np.cos(lon)
    so = np.sin(lon)

    e_r = np.column_stack((cl * co, cl * so, sl))
    e_lat = np.column_stack((-sl * co, -sl * so, cl))
    e_lon = np.column_stack((-so, co, 0.0 * so))

    a_cart = a_r[:, None] * e_r + a_lat[:, None] * e_lat + a_lon[:, None] * e_lon
    return V, a_cart


def build_sh_design_matrices(
    spec: SHSpec,
    points: np.ndarray,
    use_potential: bool = True,
    use_acc: bool = True,
):
    """
    Linear system:
      V(points) ≈ A_pot @ c
      a(points) ≈ A_acc @ c   (stacked ax,ay,az)
    Coefficients c are for FULLY NORMALIZED real SH (C/S).
    """
    L = spec.L
    P = sh_num_params(L)
    N = points.shape[0]

    ls, ms, ks = sh_param_indexing(L)

    A_pot = np.zeros((N, P), float) if use_potential else None
    A_acc = np.zeros((3 * N, P), float) if use_acc else None

    for j in range(P):
        l = int(ls[j])
        m = int(ms[j])
        kind = str(ks[j])
        Vb, ab = sh_basis_potential_and_acc(spec, points, l, m, kind)
        if use_potential:
            A_pot[:, j] = Vb
        if use_acc:
            A_acc[0::3, j] = ab[:, 0]
            A_acc[1::3, j] = ab[:, 1]
            A_acc[2::3, j] = ab[:, 2]

    return A_pot, A_acc


def sh_eval_from_coeffs(
    spec: SHSpec, points: np.ndarray, coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    A_pot, A_acc = build_sh_design_matrices(
        spec, points, use_potential=True, use_acc=True
    )
    V = A_pot @ coeffs
    a = (A_acc @ coeffs).reshape((-1, 3), order="C")
    return V, a


# ======================================================================================
# (1) FORWARD: poly + fixed mascon anomalies -> fit SH coeffs (linear LS)
# ======================================================================================


def fit_sh_for_poly_plus_2fixedmascons(
    vertices,
    faces,
    density: float,
    shspec: SHSpec,
    points: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    mu1: float,
    mu2: float,
    parallel: bool = False,
    softening: float = 0.0,
    w_acc: float = 1.0,
    w_pot: float = 1.0,
):
    pot_poly, acc_poly = eval_poly_gravity(
        vertices, faces, density, points, parallel=parallel
    )

    pot1, acc1 = mascon_potential_acc(points, r1, mu1, softening=softening)
    pot2, acc2 = mascon_potential_acc(points, r2, mu2, softening=softening)

    pot_tot = pot_poly + pot1 + pot2
    acc_tot = acc_poly + acc1 + acc2

    # Build design matrices for SH
    A_pot, A_acc = build_sh_design_matrices(
        shspec, points, use_potential=True, use_acc=True
    )

    b_pot = pot_tot
    b_acc = acc_tot.reshape(-1, order="C")

    aug_A = np.vstack([w_acc * A_acc, w_pot * A_pot])
    aug_b = np.hstack([w_acc * b_acc, w_pot * b_pot])

    coeffs, *_ = np.linalg.lstsq(aug_A, aug_b, rcond=None)

    return dict(
        coeffs=coeffs,
        points=points,
        pot_poly=pot_poly,
        acc_poly=acc_poly,
        pot_tot=pot_tot,
        acc_tot=acc_tot,
        mu_true=np.array([mu1, mu2], float),
        r1=r1,
        r2=r2,
    )


# ======================================================================================
# (2) INVERSE: given SH coeffs + known poly -> estimate ONLY mu1, mu2 (LSQ + MCMC)
# ======================================================================================


def estimate_2mus_from_shcoeffs_lsq(
    coeffs_sh: np.ndarray,
    vertices,
    faces,
    density: float,
    shspec: SHSpec,
    points: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    x0_mu: np.ndarray,
    bounds_mu: Tuple[np.ndarray, np.ndarray],
    parallel: bool = False,
    softening: float = 0.0,
    use_potential: bool = True,
    w_acc: float = 1.0,
    w_pot: float = 1.0,
):
    pot_target, acc_target = sh_eval_from_coeffs(shspec, points, coeffs_sh)

    pot_poly, acc_poly = eval_poly_gravity(
        vertices, faces, density, points, parallel=parallel
    )

    pot_resid = (pot_target - pot_poly) if use_potential else None
    acc_resid = acc_target - acc_poly

    def fun(mu_vec):
        mu1, mu2 = float(mu_vec[0]), float(mu_vec[1])
        pot1, acc1 = mascon_potential_acc(points, r1, mu1, softening=softening)
        pot2, acc2 = mascon_potential_acc(points, r2, mu2, softening=softening)
        pot_m = pot1 + pot2
        acc_m = acc1 + acc2

        r_acc = (acc_m - acc_resid).reshape(-1, order="C") * w_acc
        if use_potential:
            r_pot = (pot_m - pot_resid) * w_pot
            return np.hstack([r_acc, r_pot])
        return r_acc

    lb, ub = bounds_mu
    return least_squares(
        fun,
        x0=x0_mu,
        bounds=(lb, ub),
        method="trf",
        jac="2-point",
        ftol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
        max_nfev=20000,
        x_scale="jac",
        loss="linear",
        diff_step=1e-6,
        tr_solver="exact",
        verbose=2,
    )


def estimate_2mus_from_shcoeffs_mcmc(
    coeffs_sh: np.ndarray,
    vertices,
    faces,
    density: float,
    shspec: SHSpec,
    points: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    x0_mu: np.ndarray,
    bounds_mu: Tuple[np.ndarray, np.ndarray],
    parallel: bool = False,
    softening: float = 0.0,
    use_potential: bool = True,
    w_acc: float = 1.0,
    w_pot: float = 1.0,
    sigma_like: Optional[float] = None,
    nwalkers: int = 48,
    n_burn: int = 1500,
    n_steps: int = 4000,
    thin: int = 10,
    seed: int = 123,
):
    from types import SimpleNamespace

    try:
        import emcee
    except Exception as e:
        raise ImportError("emcee is required: pip install emcee") from e

    # optional corner (NO plt.show here)
    try:
        import corner

        _HAS_CORNER = True
    except Exception:
        _HAS_CORNER = False

    pot_target, acc_target = sh_eval_from_coeffs(shspec, points, coeffs_sh)
    pot_poly, acc_poly = eval_poly_gravity(
        vertices, faces, density, points, parallel=parallel
    )

    pot_resid = (pot_target - pot_poly) if use_potential else None
    acc_resid = acc_target - acc_poly

    y_acc = acc_resid.reshape(-1, order="C")
    y = (
        np.hstack([w_acc * y_acc, w_pot * pot_resid])
        if use_potential
        else (w_acc * y_acc)
    )

    lb, ub = np.asarray(bounds_mu[0], float), np.asarray(bounds_mu[1], float)

    def log_prior(mu_vec):
        mu_vec = np.asarray(mu_vec, float)
        if np.any(mu_vec < lb) or np.any(mu_vec > ub):
            return -np.inf
        if mu_vec[0] <= 0.0 or mu_vec[1] <= 0.0:
            return -np.inf
        return 0.0

    def model_vec(mu_vec):
        mu1, mu2 = float(mu_vec[0]), float(mu_vec[1])
        pot1, acc1 = mascon_potential_acc(points, r1, mu1, softening=softening)
        pot2, acc2 = mascon_potential_acc(points, r2, mu2, softening=softening)
        pot_m = pot1 + pot2
        acc_m = acc1 + acc2

        m_acc = acc_m.reshape(-1, order="C")
        return (
            np.hstack([w_acc * m_acc, w_pot * pot_m])
            if use_potential
            else (w_acc * m_acc)
        )

    if sigma_like is None:
        y_rms = np.sqrt(np.mean(y**2)) + 1e-30
        sigma_like = 0.01 * y_rms
    inv_sigma2 = 1.0 / (sigma_like**2)

    def log_prob(mu_vec):
        lp = log_prior(mu_vec)
        if not np.isfinite(lp):
            return -np.inf
        r = model_vec(mu_vec) - y
        return lp - 0.5 * np.sum(r * r) * inv_sigma2

    ndim = 2
    rng = np.random.default_rng(seed)

    scale = np.array([0.2 * max(x0_mu[0], 1e-16), 0.2 * max(x0_mu[1], 1e-16)], float)
    scale = np.maximum(scale, np.array([1e-12, 1e-12]))
    p0 = x0_mu[None, :] + rng.normal(size=(nwalkers, ndim)) * scale[None, :]

    for i in range(nwalkers):
        p0[i] = np.minimum(np.maximum(p0[i], lb), ub)
        p0[i, 0] = max(p0[i, 0], 1e-12)
        p0[i, 1] = max(p0[i, 1], 1e-12)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, n_burn, progress=True)
    sampler.reset()
    sampler.run_mcmc(None, n_steps, progress=True)

    samples = sampler.get_chain(flat=True, thin=thin)
    logp = sampler.get_log_prob(flat=True, thin=thin)

    mu_hat = np.median(samples, axis=0)
    cov_hat = np.cov(samples, rowvar=False)

    i_map = int(np.argmax(logp))
    mu_map = samples[i_map]
    r_map = model_vec(mu_map) - y
    cost_hat = 0.5 * float(np.sum(r_map**2))

    corner_fig = None
    if _HAS_CORNER:
        corner_fig = corner.corner(
            samples,
            labels=[r"$\Delta\mu_1$", r"$\Delta\mu_2$"],
            truths=mu_hat,
            show_titles=True,
            title_fmt=".3e",
            quantiles=[0.16, 0.50, 0.84],
        )

    return SimpleNamespace(
        x=mu_hat,
        cov=cov_hat,
        cost=cost_hat,
        success=True,
        message="MCMC posterior median estimate (mu1, mu2 only)",
        samples=samples,
        log_prob=logp,
        sampler=sampler,
        corner_fig=corner_fig,
        sigma_like=sigma_like,
        mu_map=mu_map,
    )


# ======================================================================================
# Plot helpers (ONLY called at the end)
# ======================================================================================


def sh_degree_rms(coeffs: np.ndarray, L: int) -> np.ndarray:
    """
    RMS per degree: sqrt( sum_{m} (C_lm^2 + S_lm^2) )
    with our packed ordering.
    """
    ls, ms, ks = sh_param_indexing(L)
    rms = np.zeros(L + 1, float)
    for l in range(L + 1):
        idx = np.where(ls == l)[0]
        rms[l] = np.sqrt(np.sum(coeffs[idx] ** 2))
    return rms


def plot_degree_spectrum(coeffs: np.ndarray, L: int, title: str):
    deg = np.arange(L + 1)
    rms = sh_degree_rms(coeffs, L)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(deg, rms, marker="o")
    ax.set_xlabel("Degree $l$")
    ax.set_ylabel(r"RMS $\sqrt{\sum (C_{lm}^2 + S_{lm}^2)}$")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    return fig


def plot_mu_comparison(mu_true: np.ndarray, mu_lsq: np.ndarray, mu_mcmc: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array([0, 1])
    w = 0.25
    ax.bar(x - w, mu_true, width=w, label="True")
    ax.bar(x, mu_lsq, width=w, label="LSQ")
    ax.bar(x + w, mu_mcmc, width=w, label="MCMC median")
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\Delta\mu_1$", r"$\Delta\mu_2$"])
    ax.set_ylabel(r"$\Delta\mu$ (LU$^3$/TU$^2$)")
    ax.set_title("Mascon anomaly recovery (global SH fit)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return fig


def plot_residual_norm_hist(acc_resid: np.ndarray, acc_model: np.ndarray):
    nr = np.linalg.norm(acc_resid, axis=1)
    nm = np.linalg.norm(acc_model - acc_resid, axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(nr, bins=60, alpha=0.6, density=True, label=r"$\|\mathbf{a}_{res}\|$")
    ax.hist(
        nm,
        bins=60,
        alpha=0.6,
        density=True,
        label=r"$\|\mathbf{a}_{model}-\mathbf{a}_{res}\|$",
    )
    ax.set_xlabel("Acceleration norm (Cartesian)")
    ax.set_ylabel("PDF")
    ax.set_title("Residual acceleration vs post-fit mismatch (acc only)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return fig


def plot_point_cloud(points: np.ndarray, r1: np.ndarray, r2: np.ndarray, title: str):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, alpha=0.35)
    ax.scatter([r1[0]], [r1[1]], [r1[2]], s=80, marker="x")
    ax.scatter([r2[0]], [r2[1]], [r2[2]], s=80, marker="x")
    ax.set_title(title)
    ax.set_xlabel("X (LU)")
    ax.set_ylabel("Y (LU)")
    ax.set_zlabel("Z (LU)")
    return fig


# ======================================================================================
# MAIN
# ======================================================================================

if __name__ == "__main__":
    # ---------------------------
    # Load mesh (EROS)
    # ---------------------------
    vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    DENSITY = 1.0  # constant-density poly baseline

    # Expansion center: use mesh centroid (or your body-fixed origin)
    center = vertices.mean(axis=0)
    Rb = bounding_sphere_radius(vertices, center=center)

    # ---------------------------
    # GLOBAL fit points: spherical shell all around the asteroid
    # ---------------------------
    N_FIT = 10000
    r_min = 1.05 * Rb
    r_max = 3.0 * Rb
    points = sample_points_in_spherical_shell(
        N_FIT, r_min, r_max, center=center, seed=1
    )

    # ---------------------------
    # Fixed mascon positions (must be inside body)
    # ---------------------------
    r1 = center + np.array([0.20, -0.05, 0.10])
    r2 = center + np.array([-0.10, 0.12, -0.02])

    # True anomalies (Δmu wrt constant-density poly)
    mu_true = np.array([5e-9, 2e-9], float)

    # ---------------------------
    # Spherical Harmonics setup
    # ---------------------------
    L = 9  # increase if you want (cost ~ O(N * L^2))
    R_ref = Rb  # typical choice: body reference radius
    shspec = SHSpec(L=L, R_ref=R_ref, center=center)

    # weights between acc and pot in the LS fit
    w_acc_fit = 1.0
    w_pot_fit = 1.0

    # ---------------------------
    # (1) Forward synth -> fit SH coeffs
    # ---------------------------
    fwd = fit_sh_for_poly_plus_2fixedmascons(
        vertices,
        faces,
        DENSITY,
        shspec,
        points,
        r1=r1,
        r2=r2,
        mu1=float(mu_true[0]),
        mu2=float(mu_true[1]),
        parallel=False,
        softening=0.0,
        w_acc=w_acc_fit,
        w_pot=w_pot_fit,
    )
    coeffs_sh = fwd["coeffs"]

    # ---------------------------
    # (2a) Inverse LSQ: estimate mu only
    # ---------------------------
    x0_mu = np.array([0, 0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1e-6, 1e-6])

    res_lsq = estimate_2mus_from_shcoeffs_lsq(
        coeffs_sh,
        vertices,
        faces,
        DENSITY,
        shspec,
        points,
        r1=r1,
        r2=r2,
        x0_mu=x0_mu,
        bounds_mu=(lb, ub),
        parallel=False,
        softening=0.0,
        use_potential=True,
        w_acc=1.0,
        w_pot=1.0,
    )
    mu_lsq = res_lsq.x

    # ---------------------------
    # (2b) Inverse MCMC: posterior on mu
    # ---------------------------
    res_mcmc = estimate_2mus_from_shcoeffs_mcmc(
        coeffs_sh,
        vertices,
        faces,
        DENSITY,
        shspec,
        points,
        r1=r1,
        r2=r2,
        x0_mu=mu_lsq,
        bounds_mu=(lb, ub),
        parallel=False,
        softening=0.0,
        use_potential=True,
        w_acc=1.0,
        w_pot=1.0,
        sigma_like=None,
        nwalkers=48,
        n_burn=1500,
        n_steps=4000,
        thin=10,
        seed=123,
    )
    mu_mcmc = res_mcmc.x

    # ---------------------------
    # Diagnostics: residual accel attributed to anomalies
    # ---------------------------
    pot_target, acc_target = sh_eval_from_coeffs(shspec, points, coeffs_sh)
    pot_poly, acc_poly = eval_poly_gravity(
        vertices, faces, DENSITY, points, parallel=False
    )

    acc_resid = acc_target - acc_poly

    _, acc1h = mascon_potential_acc(points, r1, float(mu_mcmc[0]), softening=0.0)
    _, acc2h = mascon_potential_acc(points, r2, float(mu_mcmc[1]), softening=0.0)
    acc_model = acc1h + acc2h

    # ==================================================================================
    # PLOTS (ONLY HERE)
    # ==================================================================================
    figs = []
    figs.append(
        plot_point_cloud(
            points,
            r1,
            r2,
            "Global fit points (spherical shell) + fixed mascon locations",
        )
    )
    figs.append(
        plot_degree_spectrum(
            coeffs_sh, L, "SH coefficient degree RMS spectrum (poly + anomalies)"
        )
    )
    figs.append(plot_mu_comparison(mu_true=mu_true, mu_lsq=mu_lsq, mu_mcmc=mu_mcmc))
    figs.append(plot_residual_norm_hist(acc_resid, acc_model))
    if res_mcmc.corner_fig is not None:
        figs.append(res_mcmc.corner_fig)

    # Summary
    print("\n=== TRUE Δmu ===", mu_true)
    print("=== LSQ  Δmu ===", mu_lsq)
    print("=== MCMC Δmu ===", mu_mcmc)
    print("LSQ cost =", res_lsq.cost)
    print("MCMC cost =", res_mcmc.cost, "(MAP sample)")
    print("MCMC sigma_like =", res_mcmc.sigma_like)
    print("L =", L, "N_fit =", points.shape[0], "R_ref =", R_ref)

    plt.show()
