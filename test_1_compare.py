"""
Two-mascon TAG-style forward + inverse (GLOBAL SPHERICAL HARMONICS) WITH MASS CONSERVATION
— parameterized in MASS FRACTIONS (beta1,beta2) and residuals in COEFFICIENT SPACE.

YOU ASKED:
1) Optimize β1, β2 (mass fractions), not Δμ.
   Δμ_j = β_j * μ_tot
   β̃   = 1 - (β1+β2)

2) Residuals must be differences of spherical-harmonics coefficients (as in your notebook):
   ΔCS_obs = CS_T - CS_CD
   Model:  ΔCS(β) = (β̃-1) CS_CD + Σ CS_j(Δμ_j) = -(β1+β2) CS_CD + CS_1(β1 μ_tot)+CS_2(β2 μ_tot)

Implementation strategy (robust + minimal physics assumptions):
- Cache baseline poly field (pot+acc) at sampling points once.
- Fit SH coefficients for:
    (a) baseline: CS_CD  from poly field
    (b) unit-mascon signatures: c1, c2 such that CS_j(Δμ)=Δμ * c_j
        computed by fitting SH to point-mass field with μ=1 at the same points.
- Forward truth:
    Choose (β1_true, β2_true). Then:
      CS_T = β̃ CS_CD + (β1 μ_tot)c1 + (β2 μ_tot)c2
  (equivalently, you could generate field and refit; this coefficient-space construction is exact
   because the mapping field->coeffs is linear, and it matches your notebook.)
- Inverse:
    Observed ΔCS_obs = CS_T - CS_CD
    Model ΔCS(β) = -(β1+β2) CS_CD + (β1 μ_tot)c1 + (β2 μ_tot)c2
    Solve β via LSQ + MCMC.

✅ No solver does plt.show; plots only at end.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
from math import factorial

from scipy.optimize import least_squares
from scipy.special import lpmv

import mesh_utility
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable


# ======================================================================================
# Geometry + sampling
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
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, num_points)
    r = (r_min**3 + u * (r_max**3 - r_min**3)) ** (1.0 / 3.0)

    cos_th = rng.uniform(-1.0, 1.0, num_points)
    th = np.arccos(cos_th)
    lon = rng.uniform(0.0, 2.0 * np.pi, num_points)

    x = r * np.sin(th) * np.cos(lon)
    y = r * np.sin(th) * np.sin(lon)
    z = r * np.cos(th)

    return np.column_stack((x, y, z)) + center[None, :]


def cart_to_sph(
    points: np.ndarray, center: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = points - center[None, :]
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    lon = np.arctan2(y, x) % (2.0 * np.pi)
    lat = np.arcsin(np.clip(z / (r + 1e-30), -1.0, 1.0))
    return r, lat, lon


# ======================================================================================
# Poly gravity evaluation
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


def estimate_mu_tot_from_poly(
    pot_poly: np.ndarray, points: np.ndarray, center: np.ndarray
) -> float:
    """
    Sign-agnostic μ_tot magnitude estimator (robust to potential sign convention):
      |U| ≈ μ/r  => μ ≈ median(|U| r)
    """
    r = np.linalg.norm(points - center[None, :], axis=1)
    mu_mag = np.median(np.abs(pot_poly) * r)
    if not np.isfinite(mu_mag) or mu_mag <= 0.0:
        raise ValueError(f"Bad mu_tot estimate: {mu_mag}")
    return float(mu_mag)


# ======================================================================================
# Mascon anomalies (point masses)
# ======================================================================================


def mascon_potential_acc(
    points: np.ndarray, r_m: np.ndarray, mu: float, softening: float = 0.0
):
    dr = points - r_m[None, :]
    r2 = np.sum(dr * dr, axis=1) + softening**2
    r = np.sqrt(r2)
    V = -mu / (r + 1e-30)
    a = mu * dr / ((r[:, None] ** 3) + 1e-30)
    return V, a


# ======================================================================================
# Mass bookkeeping in beta-space
# ======================================================================================


def beta_tilde(beta: np.ndarray) -> float:
    beta = np.asarray(beta, float)
    return float(1.0 - np.sum(beta))


def beta_feasible(beta: np.ndarray) -> bool:
    """
    Minimal constraints for redistribution with nonnegative baseline:
      beta1>=0, beta2>=0, beta1+beta2 <= 1
    """
    beta = np.asarray(beta, float)
    if beta.shape != (2,):
        return False
    if np.any(~np.isfinite(beta)):
        return False
    if np.any(beta < 0.0):
        return False
    return (beta[0] + beta[1]) <= 1.0


# ======================================================================================
# GLOBAL spherical harmonics (FULLY NORMALIZED, real C/S)
# ======================================================================================


@dataclass
class SHSpec:
    L: int
    R_ref: float
    center: np.ndarray


def sh_num_params(L: int) -> int:
    return (L + 1) ** 2


def sh_param_indexing(L: int):
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
    fac = 1.0 if m == 0 else 2.0
    return np.sqrt(fac * (2 * l + 1) * factorial(l - m) / factorial(l + m))


def _Pbar_lm(l: int, m: int, x: np.ndarray) -> np.ndarray:
    return _N_lm(l, m) * lpmv(m, l, x)


def _dP_lm_dx(l: int, m: int, x: np.ndarray) -> np.ndarray:
    if l == 0:
        return np.zeros_like(x)
    P_lm = lpmv(m, l, x)
    P_lm1 = lpmv(m, l - 1, x)
    denom = x * x - 1.0
    denom = np.where(np.abs(denom) < 1e-14, np.sign(denom) * 1e-14, denom)
    return (l * x * P_lm - (l + m) * P_lm1) / denom


def _dPbar_lm_dx(l: int, m: int, x: np.ndarray) -> np.ndarray:
    return _N_lm(l, m) * _dP_lm_dx(l, m, x)


def sh_basis_potential_and_acc(
    spec: SHSpec, points: np.ndarray, l: int, m: int, kind: str
):
    r, lat, lon = cart_to_sph(points, spec.center)
    x = np.sin(lat)
    coslat = np.cos(lat)

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

    V = k * Pbar * trig

    dV_dr = -(l + 1) * V / (r + 1e-30)
    dPbar_dx = _dPbar_lm_dx(l, m, x)
    dV_dlat = k * dPbar_dx * coslat * trig
    dV_dlon = k * Pbar * dtrig_dlon

    a_r = dV_dr
    a_lat = dV_dlat / (r + 1e-30)
    a_lon = dV_dlon / ((r + 1e-30) * (coslat + 1e-30))

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
    spec: SHSpec, points: np.ndarray, use_potential: bool = True, use_acc: bool = True
):
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


def fit_sh_to_field(
    shspec: SHSpec,
    points: np.ndarray,
    pot: np.ndarray,
    acc: np.ndarray,
    w_acc: float = 1.0,
    w_pot: float = 1.0,
) -> np.ndarray:
    A_pot, A_acc = build_sh_design_matrices(
        shspec, points, use_potential=True, use_acc=True
    )
    b_pot = pot
    b_acc = acc.reshape(-1, order="C")
    aug_A = np.vstack([w_acc * A_acc, w_pot * A_pot])
    aug_b = np.hstack([w_acc * b_acc, w_pot * b_pot])
    coeffs, *_ = np.linalg.lstsq(aug_A, aug_b, rcond=None)
    return coeffs


# ======================================================================================
# Coefficient-space forward model pieces
# ======================================================================================


def build_coefficient_signatures(
    vertices,
    faces,
    density: float,
    shspec: SHSpec,
    points: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    parallel: bool = False,
    softening: float = 0.0,
    w_acc_fit: float = 1.0,
    w_pot_fit: float = 1.0,
):
    """
    Returns:
      CS_CD  : baseline SH coeffs from poly at density
      c1,c2  : unit-mu coefficient signatures for mascons at r1,r2
      mu_tot : magnitude estimate from poly potential
    """
    pot_poly, acc_poly = eval_poly_gravity(
        vertices, faces, density, points, parallel=parallel
    )
    mu_tot = estimate_mu_tot_from_poly(pot_poly, points, center=shspec.center)

    CS_CD = fit_sh_to_field(
        shspec, points, pot_poly, acc_poly, w_acc=w_acc_fit, w_pot=w_pot_fit
    )

    # unit-mass mascon signatures: CS_j(Δμ) = Δμ * c_j
    pot1u, acc1u = mascon_potential_acc(points, r1, mu=1.0, softening=softening)
    pot2u, acc2u = mascon_potential_acc(points, r2, mu=1.0, softening=softening)

    c1 = fit_sh_to_field(shspec, points, pot1u, acc1u, w_acc=w_acc_fit, w_pot=w_pot_fit)
    c2 = fit_sh_to_field(shspec, points, pot2u, acc2u, w_acc=w_acc_fit, w_pot=w_pot_fit)

    return CS_CD, c1, c2, mu_tot


def forward_truth_coeffs_from_betas(
    beta_true: np.ndarray,
    mu_tot: float,
    CS_CD: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
):
    """
    CS_T = β̃ CS_CD + (β1 μ_tot)c1 + (β2 μ_tot)c2
    """
    beta_true = np.asarray(beta_true, float)
    bt = beta_tilde(beta_true)
    mu1 = beta_true[0] * mu_tot
    mu2 = beta_true[1] * mu_tot
    CS_T = bt * CS_CD + mu1 * c1 + mu2 * c2
    return CS_T


def delta_cs_obs(CS_T: np.ndarray, CS_CD: np.ndarray) -> np.ndarray:
    return np.asarray(CS_T, float) - np.asarray(CS_CD, float)


def delta_cs_model(
    beta: np.ndarray, mu_tot: float, CS_CD: np.ndarray, c1: np.ndarray, c2: np.ndarray
) -> np.ndarray:
    """
    ΔCS(β) = CS_T(β) - CS_CD
           = (β̃-1) CS_CD + (β1 μ_tot)c1 + (β2 μ_tot)c2
           = -(β1+β2) CS_CD + (β1 μ_tot)c1 + (β2 μ_tot)c2
    """
    beta = np.asarray(beta, float)
    return (
        (-(beta[0] + beta[1]) * CS_CD)
        + (beta[0] * mu_tot) * c1
        + (beta[1] * mu_tot) * c2
    )


# ======================================================================================
# Inverse in coefficient space: LSQ + MCMC
# ======================================================================================


def estimate_betas_from_deltaCS_lsq(
    deltaCS_obs: np.ndarray,
    mu_tot: float,
    CS_CD: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    x0_beta: np.ndarray,
    bounds_beta: Tuple[np.ndarray, np.ndarray],
    W: Optional[np.ndarray] = None,
):
    """
    Solve min || W^{1/2} (ΔCS_model(beta) - ΔCS_obs) ||^2
    If W is None: identity (unweighted).
    """
    y = np.asarray(deltaCS_obs, float)
    lb, ub = np.asarray(bounds_beta[0], float), np.asarray(bounds_beta[1], float)

    if W is None:

        def whiten(v):  # identity
            return v

    else:
        # assume W is covariance (Sigma); use Cholesky to whiten
        # W = Sigma -> whiten by L^{-1}, where Sigma = L L^T
        L = np.linalg.cholesky(W + 1e-30 * np.eye(W.shape[0]))
        Linv = np.linalg.inv(L)

        def whiten(v):
            return Linv @ v

    def fun(beta):
        beta = np.asarray(beta, float)
        if (np.any(beta < lb) or np.any(beta > ub)) or (not beta_feasible(beta)):
            return 1e6 * np.ones_like(y)
        r = delta_cs_model(beta, mu_tot, CS_CD, c1, c2) - y
        return whiten(r)

    return least_squares(
        fun,
        x0=np.asarray(x0_beta, float),
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


def estimate_betas_from_deltaCS_mcmc(
    deltaCS_obs: np.ndarray,
    mu_tot: float,
    CS_CD: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    x0_beta: np.ndarray,
    bounds_beta: Tuple[np.ndarray, np.ndarray],
    Sigma: Optional[np.ndarray] = None,
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

    try:
        import corner

        _HAS_CORNER = True
    except Exception:
        _HAS_CORNER = False

    y = np.asarray(deltaCS_obs, float)
    lb, ub = np.asarray(bounds_beta[0], float), np.asarray(bounds_beta[1], float)

    # Likelihood: either full covariance (Sigma) or isotropic sigma_like
    if Sigma is not None:
        L = np.linalg.cholesky(Sigma + 1e-30 * np.eye(Sigma.shape[0]))
        Linv = np.linalg.inv(L)

        def quad_form(res):
            z = Linv @ res
            return float(np.dot(z, z))

    else:
        if sigma_like is None:
            y_rms = np.sqrt(np.mean(y**2)) + 1e-30
            sigma_like = 0.01 * y_rms
        inv_sigma2 = 1.0 / (sigma_like**2)

        def quad_form(res):
            return float(np.sum(res * res) * inv_sigma2)

    def log_prior(beta):
        beta = np.asarray(beta, float)
        if np.any(beta < lb) or np.any(beta > ub):
            return -np.inf
        if not beta_feasible(beta):
            return -np.inf
        return 0.0

    def log_prob(beta):
        lp = log_prior(beta)
        if not np.isfinite(lp):
            return -np.inf
        res = delta_cs_model(beta, mu_tot, CS_CD, c1, c2) - y
        return lp - 0.5 * quad_form(res)

    ndim = 2
    rng = np.random.default_rng(seed)

    scale = np.maximum(0.2 * np.abs(np.asarray(x0_beta, float)), np.array([1e-6, 1e-6]))
    p0 = (
        np.asarray(x0_beta, float)[None, :]
        + rng.normal(size=(nwalkers, ndim)) * scale[None, :]
    )

    # project to bounds + feasibility
    for i in range(nwalkers):
        p0[i] = np.minimum(np.maximum(p0[i], lb), ub)
        for _ in range(50):
            if beta_feasible(p0[i]):
                break
            p0[i] *= 0.5

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, n_burn, progress=True)
    sampler.reset()
    sampler.run_mcmc(None, n_steps, progress=True)

    samples = sampler.get_chain(flat=True, thin=thin)
    logp = sampler.get_log_prob(flat=True, thin=thin)

    beta_hat = np.median(samples, axis=0)
    cov_hat = np.cov(samples, rowvar=False)

    i_map = int(np.argmax(logp))
    beta_map = samples[i_map]
    res_map = delta_cs_model(beta_map, mu_tot, CS_CD, c1, c2) - y
    cost_hat = 0.5 * float(np.sum(res_map**2))

    corner_fig = None
    if _HAS_CORNER:
        corner_fig = corner.corner(
            samples,
            labels=[r"$\beta_1$", r"$\beta_2$"],
            truths=beta_hat,
            show_titles=True,
            title_fmt=".3e",
            quantiles=[0.16, 0.50, 0.84],
        )

    return SimpleNamespace(
        x=beta_hat,
        cov=cov_hat,
        cost=cost_hat,
        success=True,
        message="MCMC posterior median estimate (beta1,beta2) in coefficient space",
        samples=samples,
        log_prob=logp,
        sampler=sampler,
        corner_fig=corner_fig,
        sigma_like=sigma_like,
        beta_map=beta_map,
        mu_tot=mu_tot,
    )


# ======================================================================================
# Plot helpers
# ======================================================================================


def sh_degree_rms(coeffs: np.ndarray, L: int) -> np.ndarray:
    ls, _, _ = sh_param_indexing(L)
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


def plot_beta_comparison(
    beta_true: np.ndarray, beta_lsq: np.ndarray, beta_mcmc: np.ndarray
):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array([0, 1])
    w = 0.25
    ax.bar(x - w, beta_true, width=w, label="True")
    ax.bar(x, beta_lsq, width=w, label="LSQ")
    ax.bar(x + w, beta_mcmc, width=w, label="MCMC median")
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\beta_1$", r"$\beta_2$"])
    ax.set_ylabel(r"Mass fraction $\beta$")
    ax.set_title(
        r"Mass-fraction recovery (mass conservation: $\tilde{\beta}=1-\beta_1-\beta_2$)"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
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

    DENSITY = 1.0
    center = vertices.mean(axis=0)
    Rb = bounding_sphere_radius(vertices, center=center)

    # ---------------------------
    # Fit points: spherical shell
    # ---------------------------
    N_FIT = 10000
    points = sample_points_in_spherical_shell(
        N_FIT, 1.05 * Rb, 3.0 * Rb, center=center, seed=1
    )

    # mascon locations
    r1 = np.array([0.35, -0.02, 0.05])
    r2 = np.array([-0.10, 0.08, 0.02])

    # ---------------------------
    # SH setup
    # ---------------------------
    L = 10
    shspec = SHSpec(L=L, R_ref=Rb, center=center)

    # weights for LS coefficient fits (field -> coeffs)
    w_acc_fit = 1.0
    w_pot_fit = 1.0

    # ---------------------------
    # Build baseline coeffs + mascon coefficient signatures
    # ---------------------------
    CS_CD, c1, c2, mu_tot = build_coefficient_signatures(
        vertices,
        faces,
        DENSITY,
        shspec,
        points,
        r1,
        r2,
        parallel=False,
        softening=0.0,
        w_acc_fit=w_acc_fit,
        w_pot_fit=w_pot_fit,
    )

    # ---------------------------
    # Forward truth (in coefficient space) with betas
    # ---------------------------
    beta_true = np.array([0.02, 0.01], float)  # 2% and 1% of total mass
    if not beta_feasible(beta_true):
        raise ValueError("beta_true infeasible; need beta>=0 and beta1+beta2<=1")

    CS_T = forward_truth_coeffs_from_betas(beta_true, mu_tot, CS_CD, c1, c2)
    dCS_obs = delta_cs_obs(CS_T, CS_CD)

    # ---------------------------
    # Inverse LSQ in coefficient space
    # ---------------------------
    x0_beta = np.array([0.0, 0.0], float)

    # bounds in beta-space (hard constraints enforced also by feasibility)
    lb = np.array([0.0, 0.0], float)
    ub = np.array([1.0, 1.0], float)

    # If you have OD covariance on coefficients, pass it here as Sigma (size P×P):
    Sigma_dCS = None

    res_lsq = estimate_betas_from_deltaCS_lsq(
        deltaCS_obs=dCS_obs,
        mu_tot=mu_tot,
        CS_CD=CS_CD,
        c1=c1,
        c2=c2,
        x0_beta=x0_beta,
        bounds_beta=(lb, ub),
        W=Sigma_dCS,  # if not None, used as covariance for whitening
    )
    beta_lsq = res_lsq.x

    # ---------------------------
    # Inverse MCMC in coefficient space
    # ---------------------------
    res_mcmc = estimate_betas_from_deltaCS_mcmc(
        deltaCS_obs=dCS_obs,
        mu_tot=mu_tot,
        CS_CD=CS_CD,
        c1=c1,
        c2=c2,
        x0_beta=beta_lsq,
        bounds_beta=(lb, ub),
        Sigma=Sigma_dCS,
        sigma_like=None,
        nwalkers=48,
        n_burn=1500,
        n_steps=4000,
        thin=10,
        seed=123,
    )
    beta_mcmc = res_mcmc.x

    # ---------------------------
    # Some derived quantities for print
    # ---------------------------
    bt_true = beta_tilde(beta_true)
    bt_lsq = beta_tilde(beta_lsq)
    bt_mcmc = beta_tilde(beta_mcmc)

    dmu_true = beta_true * mu_tot
    dmu_lsq = beta_lsq * mu_tot
    dmu_mcmc = beta_mcmc * mu_tot

    # ==================================================================================
    # PLOTS (ONLY HERE)
    # ==================================================================================
    figs = []
    figs.append(plot_degree_spectrum(CS_CD, L, "Baseline SH degree RMS (CS_CD)"))
    figs.append(
        plot_degree_spectrum(
            dCS_obs, L, "Observed coefficient discrepancy ΔCS_obs degree RMS"
        )
    )
    figs.append(plot_beta_comparison(beta_true, beta_lsq, beta_mcmc))
    if res_mcmc.corner_fig is not None:
        figs.append(res_mcmc.corner_fig)

    # Summary
    print("\n=== mu_tot (magnitude) ===", mu_tot)
    print("=== TRUE beta ===", beta_true, " beta_tilde =", bt_true)
    print("=== LSQ  beta ===", beta_lsq, " beta_tilde =", bt_lsq)
    print("=== MCMC beta ===", beta_mcmc, " beta_tilde =", bt_mcmc)
    print("\n=== TRUE Δμ ===", dmu_true)
    print("=== LSQ  Δμ ===", dmu_lsq)
    print("=== MCMC Δμ ===", dmu_mcmc)
    print("LSQ cost =", res_lsq.cost)
    print("MCMC cost =", res_mcmc.cost, "(MAP sample)")
    print("MCMC sigma_like =", getattr(res_mcmc, "sigma_like", None))

    plt.show()
