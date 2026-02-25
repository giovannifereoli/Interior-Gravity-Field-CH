"""
Two-mascon TAG-style forward + inverse (TWO CYLINDERS, CYLINDRICAL-HARMONICS FIT)
WITH MASS CONSERVATION — REDONE THE RIGHT WAY (beta-parameterization + coeff-space residuals)

YOU ASKED (mirror of the spherical fix):
1) Optimization variables are NOT Δμ, but β1, β2 (mass fractions):
      Δμ1 = β1 * μ_tot
      Δμ2 = β2 * μ_tot
      β̃  = 1 - (β1 + β2)   (baseline scale)

   Constraints (hard):
      β1 >= 0, β2 >= 0, β1+β2 <= 1

2) Residuals are NOT in physical space (fields), but in COEFFICIENT SPACE, like your notebook:
      Δc_obs,i = c_T,i  - c_CD,i     (for each cylinder i)
   Model:
      c_T,i(β) = β̃ c_CD,i + Δμ1 c1_i + Δμ2 c2_i
   Therefore
      Δc_model,i(β) = c_T,i(β) - c_CD,i
                    = (β̃-1)c_CD,i + Δμ1 c1_i + Δμ2 c2_i
                    = -(β1+β2)c_CD,i + (β1 μ_tot)c1_i + (β2 μ_tot)c2_i

   Stack across cylinders:
      r(β) = stack_i( W_i^{1/2} (Δc_model,i(β) - Δc_obs,i) )

Pipeline:
A) Per cylinder i:
   - sample points_i in cylinder_i
   - evaluate baseline poly at density ρ0 -> (pot_poly_i, acc_poly_i)
   - fit baseline cylindrical coeffs: c_CD,i  (LS on [cyl_acc + pot])
   - compute μ_tot once (robust magnitude estimate from poly potential)

   - compute unit-mascon signatures in coeff-space:
        c1_i: fit coeffs to mascon field at r1 with μ=1
        c2_i: fit coeffs to mascon field at r2 with μ=1
     so that coeffs from mascon j with Δμ are:  Δμ * c_j,i

B) Forward truth (in coeff space):
   pick beta_true -> Δμ_true -> build:
        c_T,i = β̃ c_CD,i + Δμ1 c1_i + Δμ2 c2_i
   define observed coefficient residuals:
        Δc_obs,i = c_T,i - c_CD,i

C) Inverse:
   LSQ + MCMC on β = [β1,β2] using coefficient-space residuals.

Dependencies:
  numpy, scipy, matplotlib, emcee (optional), corner (optional),
  polyhedral_gravity, mesh_utility
  + your cylindrical utilities (imported with *):
      CylinderSpec
      generate_points_in_cylinder
      cart_to_cyl_acc
      prepare_linear_system_for_cyl_acc
      prepare_linear_system_for_cyl_pot
      zero_B0n
      plot_rms_spectrum
      plot_shape_and_mascons_matplotlib
      plot_mu_comparison
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
from scipy.optimize import least_squares

import mesh_utility
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable

from VariousExperiments.cylindrical_acc_pot_SHORT_fitting_both_INVERSION_2m import *  # noqa: F403, F401


# ======================================================================================
# Poly gravity evaluation (baseline constant-density poly) + mu_tot estimate
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
    Robust magnitude estimate that DOES NOT care about sign convention:
        |U| ~ μ/r  =>  μ ~ median(|U| * r)
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
    Redistribution constraints:
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
# Cylindrical fit wrappers (field -> coeffs) for your utilities
# ======================================================================================


def fit_cyl_coeffs_from_field(
    spec: "CylinderSpec",
    points: np.ndarray,
    pot: np.ndarray,
    acc_cart: np.ndarray,
    n_n: int,
    n_m: int,
    enforce_B0n_flag: bool = True,
) -> np.ndarray:
    """
    Fit cylindrical-harmonics coeffs by linear LS stacking (cyl_acc + pot).
    """
    cyl_acc = cart_to_cyl_acc(spec, points, acc_cart)  # noqa: F405

    A_acc, b_acc = prepare_linear_system_for_cyl_acc(
        spec, points, cyl_acc, n_n, n_m
    )  # noqa: F405
    A_pot, b_pot = prepare_linear_system_for_cyl_pot(
        spec, points, pot, n_n, n_m
    )  # noqa: F405

    aug_A = np.vstack([A_acc, A_pot])
    aug_b = np.hstack([b_acc, b_pot])

    coeffs, *_ = np.linalg.lstsq(aug_A, aug_b, rcond=None)

    if enforce_B0n_flag:
        zero_B0n(coeffs, n_n=n_n)  # noqa: F405

    return coeffs


# ======================================================================================
# Build per-cylinder coefficient signatures (baseline + unit mascons)
# ======================================================================================


def build_cylinder_signatures(
    vertices,
    faces,
    density: float,
    center: np.ndarray,
    specs: List["CylinderSpec"],
    points_list: List[np.ndarray],
    n_n: int,
    n_m: int,
    r1: np.ndarray,
    r2: np.ndarray,
    parallel: bool = False,
    softening: float = 0.0,
    enforce_B0n_flag: bool = True,
) -> Dict:
    """
    For each cylinder i, compute:
      cCD_i : coeffs fit to baseline poly field
      c1_i  : coeffs fit to unit mascon at r1 (mu=1)
      c2_i  : coeffs fit to unit mascon at r2 (mu=1)

    Also compute mu_tot once from baseline poly (robust magnitude), using all cylinders' samples.
    """
    assert len(specs) == len(points_list)

    cCD_list = []
    c1_list = []
    c2_list = []
    mu_est_list = []

    for spec, pts in zip(specs, points_list):
        pot_poly, acc_poly = eval_poly_gravity(
            vertices, faces, density, pts, parallel=parallel
        )

        # per-cylinder baseline coeffs
        cCD = fit_cyl_coeffs_from_field(
            spec, pts, pot_poly, acc_poly, n_n, n_m, enforce_B0n_flag=enforce_B0n_flag
        )
        cCD_list.append(cCD)

        # per-cylinder mu_tot estimate (magnitude)
        mu_est_list.append(estimate_mu_tot_from_poly(pot_poly, pts, center=center))

        # unit mascon signatures (mu=1)
        pot1u, acc1u = mascon_potential_acc(pts, r1, mu=1.0, softening=softening)
        pot2u, acc2u = mascon_potential_acc(pts, r2, mu=1.0, softening=softening)

        c1 = fit_cyl_coeffs_from_field(
            spec, pts, pot1u, acc1u, n_n, n_m, enforce_B0n_flag=enforce_B0n_flag
        )
        c2 = fit_cyl_coeffs_from_field(
            spec, pts, pot2u, acc2u, n_n, n_m, enforce_B0n_flag=enforce_B0n_flag
        )
        c1_list.append(c1)
        c2_list.append(c2)

    mu_tot = float(np.median(np.asarray(mu_est_list)))
    if not np.isfinite(mu_tot) or mu_tot <= 0.0:
        raise ValueError(f"Invalid mu_tot from cylinder estimates: {mu_tot}")

    return dict(
        mu_tot=mu_tot,
        cCD_list=cCD_list,
        c1_list=c1_list,
        c2_list=c2_list,
    )


# ======================================================================================
# Forward truth in coefficient space (per cylinder), and observed Δc_obs
# ======================================================================================


def forward_truth_coeffs_from_betas_multi(
    beta_true: np.ndarray,
    mu_tot: float,
    cCD_list: List[np.ndarray],
    c1_list: List[np.ndarray],
    c2_list: List[np.ndarray],
) -> List[np.ndarray]:
    beta_true = np.asarray(beta_true, float)
    if not beta_feasible(beta_true):
        raise ValueError(f"beta_true infeasible: {beta_true}")

    bt = beta_tilde(beta_true)
    dmu1 = beta_true[0] * mu_tot
    dmu2 = beta_true[1] * mu_tot

    cT_list = []
    for cCD, c1, c2 in zip(cCD_list, c1_list, c2_list):
        cT = bt * cCD + dmu1 * c1 + dmu2 * c2
        cT_list.append(cT)
    return cT_list


def delta_coeffs_obs_multi(
    cT_list: List[np.ndarray], cCD_list: List[np.ndarray]
) -> List[np.ndarray]:
    return [
        np.asarray(cT, float) - np.asarray(cCD, float)
        for cT, cCD in zip(cT_list, cCD_list)
    ]


def delta_coeffs_model_multi(
    beta: np.ndarray,
    mu_tot: float,
    cCD_list: List[np.ndarray],
    c1_list: List[np.ndarray],
    c2_list: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Δc_model,i(β) = cT_i(β) - cCD_i
                  = (β̃-1)cCD_i + (β1 μ_tot)c1_i + (β2 μ_tot)c2_i
                  = -(β1+β2)cCD_i + (β1 μ_tot)c1_i + (β2 μ_tot)c2_i
    """
    beta = np.asarray(beta, float)
    if beta.shape != (2,):
        raise ValueError("beta must be shape (2,)")

    out = []
    for cCD, c1, c2 in zip(cCD_list, c1_list, c2_list):
        dc = (
            (-(beta[0] + beta[1]) * cCD)
            + (beta[0] * mu_tot) * c1
            + (beta[1] * mu_tot) * c2
        )
        out.append(dc)
    return out


def stack_coeffs_multi(dclist: List[np.ndarray]) -> np.ndarray:
    return np.hstack([dc.ravel(order="C") for dc in dclist])


# ======================================================================================
# Inverse in coefficient space: LSQ + MCMC
# ======================================================================================


def estimate_betas_lsq_multi_coeffspace(
    dc_obs_list: List[np.ndarray],
    mu_tot: float,
    cCD_list: List[np.ndarray],
    c1_list: List[np.ndarray],
    c2_list: List[np.ndarray],
    x0_beta: np.ndarray,
    bounds_beta: Tuple[np.ndarray, np.ndarray],
    Sigma: Optional[np.ndarray] = None,
):
    """
    Solve min || W^{1/2} (Δc_model(beta) - Δc_obs) ||^2  with stacked cylinders.
    If Sigma provided: it is covariance of stacked Δc (size MxM) used to whiten.
    """
    y = stack_coeffs_multi(dc_obs_list)
    lb, ub = np.asarray(bounds_beta[0], float), np.asarray(bounds_beta[1], float)

    if Sigma is None:

        def whiten(v):  # identity
            return v

    else:
        L = np.linalg.cholesky(Sigma + 1e-30 * np.eye(Sigma.shape[0]))
        Linv = np.linalg.inv(L)

        def whiten(v):
            return Linv @ v

    def fun(beta):
        beta = np.asarray(beta, float)
        if (np.any(beta < lb) or np.any(beta > ub)) or (not beta_feasible(beta)):
            return 1e6 * np.ones_like(y)
        dc_model_list = delta_coeffs_model_multi(
            beta, mu_tot, cCD_list, c1_list, c2_list
        )
        r = stack_coeffs_multi(dc_model_list) - y
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


def estimate_betas_mcmc_multi_coeffspace(
    dc_obs_list: List[np.ndarray],
    mu_tot: float,
    cCD_list: List[np.ndarray],
    c1_list: List[np.ndarray],
    c2_list: List[np.ndarray],
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

    y = stack_coeffs_multi(dc_obs_list)
    lb, ub = np.asarray(bounds_beta[0], float), np.asarray(bounds_beta[1], float)

    # likelihood
    if Sigma is not None:
        L = np.linalg.cholesky(Sigma + 1e-30 * np.eye(Sigma.shape[0]))
        Linv = np.linalg.inv(L)

        def quad(res):
            z = Linv @ res
            return float(np.dot(z, z))

    else:
        if sigma_like is None:
            y_rms = np.sqrt(np.mean(y**2)) + 1e-30
            sigma_like = 0.01 * y_rms
        inv_sigma2 = 1.0 / (sigma_like**2)

        def quad(res):
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
        dc_model_list = delta_coeffs_model_multi(
            beta, mu_tot, cCD_list, c1_list, c2_list
        )
        r = stack_coeffs_multi(dc_model_list) - y
        return lp - 0.5 * quad(r)

    ndim = 2
    rng = np.random.default_rng(seed)

    scale = np.maximum(0.2 * np.abs(np.asarray(x0_beta, float)), np.array([1e-6, 1e-6]))
    p0 = (
        np.asarray(x0_beta, float)[None, :]
        + rng.normal(size=(nwalkers, ndim)) * scale[None, :]
    )

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
    r_map = (
        stack_coeffs_multi(
            delta_coeffs_model_multi(beta_map, mu_tot, cCD_list, c1_list, c2_list)
        )
        - y
    )
    cost_hat = 0.5 * float(np.sum(r_map**2))

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
        message="MCMC posterior median estimate (beta1,beta2) using BOTH cylinders in coefficient space",
        samples=samples,
        log_prob=logp,
        sampler=sampler,
        corner_fig=corner_fig,
        sigma_like=sigma_like,
        beta_map=beta_map,
        mu_tot=mu_tot,
    )


# ======================================================================================
# MAIN (example)
# ======================================================================================

if __name__ == "__main__":
    # ---------------------------
    # Load mesh (EROS)
    # ---------------------------
    vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    DENSITY = 1.0
    center = vertices.mean(axis=0)

    # ---------------------------
    # Define TWO cylinders
    # ---------------------------
    spec1 = CylinderSpec(  # noqa: F405
        center=np.array([0.0, 0.0, 0.28]),
        radius=0.10,
        height=0.50,
        rotation=np.eye(3),
        alpha=100.0,
    )
    spec2 = CylinderSpec(  # noqa: F405
        center=np.array([0.10, -0.05, 0.20]),
        radius=0.10,
        height=0.50,
        rotation=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        alpha=100.0,
    )
    specs = [spec1, spec2]

    # sample points per cylinder
    points_list = [
        generate_points_in_cylinder(spec1, 1200, seed=1),  # noqa: F405
        generate_points_in_cylinder(spec2, 1200, seed=2),  # noqa: F405
    ]

    # truncation
    n_n, n_m = 5, 5

    # fixed mascon positions
    r1 = np.array([0.35, -0.02, 0.05])
    r2 = np.array([-0.10, 0.08, 0.02])

    # ---------------------------
    # Build coefficient signatures for BOTH cylinders
    # ---------------------------
    sig = build_cylinder_signatures(
        vertices=vertices,
        faces=faces,
        density=DENSITY,
        center=center,
        specs=specs,
        points_list=points_list,
        n_n=n_n,
        n_m=n_m,
        r1=r1,
        r2=r2,
        parallel=False,
        softening=0.0,
        enforce_B0n_flag=True,
    )
    mu_tot = sig["mu_tot"]
    cCD_list = sig["cCD_list"]
    c1_list = sig["c1_list"]
    c2_list = sig["c2_list"]

    # ---------------------------
    # FORWARD TRUTH (in coeff space)
    # ---------------------------
    beta_true = np.array([0.02, 0.01], float)  # choose feasible
    cT_list = forward_truth_coeffs_from_betas_multi(
        beta_true, mu_tot, cCD_list, c1_list, c2_list
    )
    dc_obs_list = delta_coeffs_obs_multi(cT_list, cCD_list)

    # Derived Δμ truth (for print/plots)
    dmu_true = beta_true * mu_tot

    # ---------------------------
    # INVERSE LSQ on betas (coeff-space residuals, stacked cylinders)
    # ---------------------------
    x0_beta = np.array([0.005, 0.005], float)
    lb = np.array([0.0, 0.0], float)
    ub = np.array([1.0, 1.0], float)

    # If you have OD covariance on stacked coeffs, pass Sigma here (M×M).
    Sigma = None

    res_lsq = estimate_betas_lsq_multi_coeffspace(
        dc_obs_list=dc_obs_list,
        mu_tot=mu_tot,
        cCD_list=cCD_list,
        c1_list=c1_list,
        c2_list=c2_list,
        x0_beta=x0_beta,
        bounds_beta=(lb, ub),
        Sigma=Sigma,
    )
    beta_lsq = res_lsq.x
    dmu_lsq = beta_lsq * mu_tot

    # ---------------------------
    # INVERSE MCMC on betas (coeff-space)
    # ---------------------------
    res_mcmc = estimate_betas_mcmc_multi_coeffspace(
        dc_obs_list=dc_obs_list,
        mu_tot=mu_tot,
        cCD_list=cCD_list,
        c1_list=c1_list,
        c2_list=c2_list,
        x0_beta=beta_lsq,
        bounds_beta=(lb, ub),
        Sigma=Sigma,
        sigma_like=None,
        nwalkers=48,
        n_burn=1500,
        n_steps=4000,
        thin=10,
        seed=123,
    )
    beta_mcmc = res_mcmc.x
    dmu_mcmc = beta_mcmc * mu_tot

    # ---------------------------
    # PLOTS (ONLY AT END)
    # ---------------------------
    figs = []

    # show spectra for each cylinder (baseline and delta)
    for i, (cCD, dc_obs) in enumerate(zip(cCD_list, dc_obs_list)):
        figs.append(
            plot_rms_spectrum(
                cCD, n_n, n_m, f"Cylinder #{i+1}: baseline coeff spectrum"
            )
        )  # noqa: F405
        figs.append(
            plot_rms_spectrum(
                dc_obs, n_n, n_m, f"Cylinder #{i+1}: Δcoeff spectrum (obs)"
            )
        )  # noqa: F405

    mascons = np.vstack([r1.reshape(1, 3), r2.reshape(1, 3)])
    fig_shape, _ = plot_shape_and_mascons_matplotlib(  # noqa: F405
        vertices,
        faces,
        mascons,
        title="EROS shape + fixed mascons",
        face_alpha=0.20,
        decimate_faces=5,
    )
    figs.append(fig_shape)

    figs.append(
        plot_mu_comparison(mu_true=dmu_true, mu_lsq=dmu_lsq, mu_mcmc=dmu_mcmc)
    )  # noqa: F405

    if getattr(res_mcmc, "corner_fig", None) is not None:
        figs.append(res_mcmc.corner_fig)

    # ---------------------------
    # Summary
    # ---------------------------
    print("\n=== mu_tot (magnitude) ===", mu_tot)
    print("=== TRUE beta ===", beta_true, " beta_tilde =", beta_tilde(beta_true))
    print("=== LSQ  beta ===", beta_lsq, " beta_tilde =", beta_tilde(beta_lsq))
    print("=== MCMC beta ===", beta_mcmc, " beta_tilde =", beta_tilde(beta_mcmc))
    print("\n=== TRUE Δμ ===", dmu_true)
    print("=== LSQ  Δμ ===", dmu_lsq)
    print("=== MCMC Δμ ===", dmu_mcmc)
    print("LSQ cost =", res_lsq.cost)
    print("MCMC cost =", res_mcmc.cost)
    print("MCMC sigma_like =", getattr(res_mcmc, "sigma_like", None))

    plt.show()
