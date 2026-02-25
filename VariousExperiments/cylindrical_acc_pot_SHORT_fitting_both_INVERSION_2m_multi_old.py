"""
PATCH: extend pipeline to TWO cylinders in the forward model, and use BOTH cylinders
simultaneously in the inverse (LSQ + MCMC) to estimate the SAME (mu1, mu2).

Key idea:
- Forward: for each cylinder i, sample points_i inside cylinder_i, evaluate (poly + mascons)
  at points_i, then fit cylinder coeffs_i by LS on that cylinder’s basis.
- Inverse: stack residual vectors from BOTH cylinders into ONE big residual. Solve mu1, mu2
  once, shared across cylinders.

You can drop this into your script by:
  1) replacing the single-cylinder forward+inverse functions with the multi-cylinder ones below
  2) updating __main__ to define spec_list and points_list
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.optimize import least_squares
from VariousExperiments.cylindrical_acc_pot_SHORT_fitting_both_INVERSION_2m import *


# TODO: let's add realistic noise on coefficient to simulate OD. You can sample them from their covariance for instance
# or you can put something on the design matrix to simulate correlated and colored noise and re-fit.

# ------------------------------------------------------------
# NEW: multi-cylinder forward
# ------------------------------------------------------------


def fit_cylinders_for_poly_plus_2fixedmascons(
    vertices,
    faces,
    density: float,
    specs: List["CylinderSpec"],
    points_list: List[np.ndarray],
    n_n: int,
    n_m: int,
    r1: np.ndarray,
    r2: np.ndarray,
    mu1: float,
    mu2: float,
    parallel: bool = False,
    enforce_B0n_flag: bool = True,
    softening: float = 0.0,
) -> Dict:
    assert len(specs) == len(points_list), "specs and points_list must have same length"

    coeffs_list = []
    truth_blocks = []

    for spec, points in zip(specs, points_list):
        # poly (constant density)
        pot_poly, acc_poly = eval_poly_gravity(
            vertices, faces, density, points, parallel=parallel
        )

        # mascon anomalies
        pot1, acc1 = mascon_potential_acc(points, r1, mu1, softening=softening)
        pot2, acc2 = mascon_potential_acc(points, r2, mu2, softening=softening)

        pot_tot = pot_poly + pot1 + pot2
        acc_tot = acc_poly + acc1 + acc2

        cyl_acc_tot = cart_to_cyl_acc(spec, points, acc_tot)

        A_acc, b_acc = prepare_linear_system_for_cyl_acc(
            spec, points, cyl_acc_tot, n_n, n_m
        )
        A_pot, b_pot = prepare_linear_system_for_cyl_pot(
            spec, points, pot_tot, n_n, n_m
        )

        aug_A = np.vstack([A_acc, A_pot])
        aug_b = np.hstack([b_acc, b_pot])

        coeffs, *_ = np.linalg.lstsq(aug_A, aug_b, rcond=None)
        if enforce_B0n_flag:
            zero_B0n(coeffs, n_n=n_n)

        coeffs_list.append(coeffs)
        truth_blocks.append(
            dict(
                pot_poly=pot_poly,
                acc_poly=acc_poly,
                pot_tot=pot_tot,
                acc_tot=acc_tot,
                cyl_acc_tot=cyl_acc_tot,
            )
        )

    return dict(
        coeffs_list=coeffs_list,
        truth_blocks=truth_blocks,
        mu_true=np.array([mu1, mu2], float),
        r1=r1,
        r2=r2,
    )


# ------------------------------------------------------------
# NEW: helpers to build stacked residual vector across cylinders
# ------------------------------------------------------------


def _targets_from_coeffs_multi(
    coeffs_list: List[np.ndarray],
    specs: List["CylinderSpec"],
    points_list: List[np.ndarray],
    n_n: int,
    n_m: int,
    use_potential: bool,
):
    cyl_acc_targets = []
    pot_targets = [] if use_potential else None
    for coeffs, spec, points in zip(coeffs_list, specs, points_list):
        cyl_acc_targets.append(cyl_acc_from_coeffs(spec, points, coeffs, n_n, n_m))
        if use_potential:
            pot_targets.append(cyl_pot_from_coeffs(spec, points, coeffs, n_n, n_m))
    return cyl_acc_targets, pot_targets


def _poly_fields_multi(
    vertices,
    faces,
    density: float,
    specs: List["CylinderSpec"],
    points_list: List[np.ndarray],
    parallel: bool,
):
    pot_polys, acc_polys, cyl_acc_polys = [], [], []
    for spec, points in zip(specs, points_list):
        pot_poly, acc_poly = eval_poly_gravity(
            vertices, faces, density, points, parallel=parallel
        )
        pot_polys.append(pot_poly)
        acc_polys.append(acc_poly)
        cyl_acc_polys.append(cart_to_cyl_acc(spec, points, acc_poly))
    return pot_polys, acc_polys, cyl_acc_polys


def _stack_residual_observation(
    cyl_acc_targets: List[np.ndarray],
    pot_targets: Optional[List[np.ndarray]],
    cyl_acc_polys: List[np.ndarray],
    pot_polys: List[np.ndarray],
    use_potential: bool,
    w_acc: float,
    w_pot: float,
):
    # residual attributed to anomalies, stacked across cylinders
    y_parts = []
    for i in range(len(cyl_acc_targets)):
        cyl_acc_resid = cyl_acc_targets[i] - cyl_acc_polys[i]
        y_acc = cyl_acc_resid.reshape(-1, order="C")
        y_parts.append(w_acc * y_acc)

        if use_potential:
            pot_resid = pot_targets[i] - pot_polys[i]
            y_parts.append(w_pot * pot_resid)

    y = np.hstack(y_parts)
    return y


def _stack_model_vector_multi(
    mu_vec: np.ndarray,
    specs: List["CylinderSpec"],
    points_list: List[np.ndarray],
    r1: np.ndarray,
    r2: np.ndarray,
    softening: float,
    use_potential: bool,
    w_acc: float,
    w_pot: float,
):
    mu1, mu2 = float(mu_vec[0]), float(mu_vec[1])

    m_parts = []
    for spec, points in zip(specs, points_list):
        pot1, acc1 = mascon_potential_acc(points, r1, mu1, softening=softening)
        pot2, acc2 = mascon_potential_acc(points, r2, mu2, softening=softening)

        pot_m = pot1 + pot2
        acc_m = acc1 + acc2
        cyl_acc_m = cart_to_cyl_acc(spec, points, acc_m)

        m_acc = cyl_acc_m.reshape(-1, order="C")
        m_parts.append(w_acc * m_acc)
        if use_potential:
            m_parts.append(w_pot * pot_m)

    return np.hstack(m_parts)


# ------------------------------------------------------------
# NEW: multi-cylinder inverse (LSQ)
# ------------------------------------------------------------


def estimate_2mus_from_coeffs_lsq_multi(
    coeffs_list: List[np.ndarray],
    vertices,
    faces,
    density: float,
    specs: List["CylinderSpec"],
    points_list: List[np.ndarray],
    n_n: int,
    n_m: int,
    r1: np.ndarray,
    r2: np.ndarray,
    x0_mu: np.ndarray,  # (2,)
    bounds_mu: Tuple[np.ndarray, np.ndarray],
    parallel: bool = False,
    softening: float = 0.0,
    use_potential: bool = True,
    w_acc: float = 1.0,
    w_pot: float = 1.0,
):
    assert len(coeffs_list) == len(specs) == len(points_list)

    # targets from coeffs (each cylinder)
    cyl_acc_targets, pot_targets = _targets_from_coeffs_multi(
        coeffs_list, specs, points_list, n_n, n_m, use_potential
    )

    # known poly fields
    pot_polys, _, cyl_acc_polys = _poly_fields_multi(
        vertices, faces, density, specs, points_list, parallel=parallel
    )

    # stacked observed residual vector (what mascons must explain)
    y = _stack_residual_observation(
        cyl_acc_targets=cyl_acc_targets,
        pot_targets=pot_targets,
        cyl_acc_polys=cyl_acc_polys,
        pot_polys=pot_polys,
        use_potential=use_potential,
        w_acc=w_acc,
        w_pot=w_pot,
    )

    def fun(mu_vec):
        m = _stack_model_vector_multi(
            mu_vec=np.asarray(mu_vec, float),
            specs=specs,
            points_list=points_list,
            r1=r1,
            r2=r2,
            softening=softening,
            use_potential=use_potential,
            w_acc=w_acc,
            w_pot=w_pot,
        )
        return m - y

    lb, ub = bounds_mu
    res = least_squares(
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
    return res


# ------------------------------------------------------------
# NEW: multi-cylinder inverse (MCMC)
# ------------------------------------------------------------


def estimate_2mus_from_coeffs_mcmc_multi(
    coeffs_list: List[np.ndarray],
    vertices,
    faces,
    density: float,
    specs: List["CylinderSpec"],
    points_list: List[np.ndarray],
    n_n: int,
    n_m: int,
    r1: np.ndarray,
    r2: np.ndarray,
    x0_mu: np.ndarray,  # (2,)
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

    try:
        import corner

        _HAS_CORNER = True
    except Exception:
        _HAS_CORNER = False

    assert len(coeffs_list) == len(specs) == len(points_list)

    # targets from coeffs
    cyl_acc_targets, pot_targets = _targets_from_coeffs_multi(
        coeffs_list, specs, points_list, n_n, n_m, use_potential
    )

    # known poly fields
    pot_polys, _, cyl_acc_polys = _poly_fields_multi(
        vertices, faces, density, specs, points_list, parallel=parallel
    )

    # stacked observed residual vector
    y = _stack_residual_observation(
        cyl_acc_targets=cyl_acc_targets,
        pot_targets=pot_targets,
        cyl_acc_polys=cyl_acc_polys,
        pot_polys=pot_polys,
        use_potential=use_potential,
        w_acc=w_acc,
        w_pot=w_pot,
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
        return _stack_model_vector_multi(
            mu_vec=np.asarray(mu_vec, float),
            specs=specs,
            points_list=points_list,
            r1=r1,
            r2=r2,
            softening=softening,
            use_potential=use_potential,
            w_acc=w_acc,
            w_pot=w_pot,
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
        message="MCMC posterior median estimate (mu1, mu2) using BOTH cylinders",
        samples=samples,
        log_prob=logp,
        sampler=sampler,
        corner_fig=corner_fig,
        sigma_like=sigma_like,
        mu_map=mu_map,
    )


# ------------------------------------------------------------
# OPTIONAL: residual histogram, per-cylinder (for plots at end)
# ------------------------------------------------------------


def build_cyl_acc_residual_and_model_per_cyl(
    coeffs_list: List[np.ndarray],
    vertices,
    faces,
    density: float,
    specs: List["CylinderSpec"],
    points_list: List[np.ndarray],
    n_n: int,
    n_m: int,
    r1: np.ndarray,
    r2: np.ndarray,
    mu_hat: np.ndarray,
    parallel: bool = False,
    softening: float = 0.0,
):
    cyl_acc_targets, _ = _targets_from_coeffs_multi(
        coeffs_list, specs, points_list, n_n, n_m, use_potential=False
    )

    pot_polys, _, cyl_acc_polys = _poly_fields_multi(
        vertices, faces, density, specs, points_list, parallel=parallel
    )

    out = []
    for spec, points, cyl_acc_t, cyl_acc_p in zip(
        specs, points_list, cyl_acc_targets, cyl_acc_polys
    ):
        cyl_acc_resid = cyl_acc_t - cyl_acc_p

        pot1, acc1 = mascon_potential_acc(
            points, r1, float(mu_hat[0]), softening=softening
        )
        pot2, acc2 = mascon_potential_acc(
            points, r2, float(mu_hat[1]), softening=softening
        )
        cyl_acc_model = cart_to_cyl_acc(spec, points, acc1 + acc2)

        out.append((cyl_acc_resid, cyl_acc_model))
    return out


# ======================================================================================
# __main__ CHANGES (example)
# ======================================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load mesh (EROS)
    vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    DENSITY = 1.0

    # --- define TWO cylinders ---
    spec1 = CylinderSpec(
        center=np.array([0.0, 0.0, 0.28]),
        radius=0.10,
        height=0.50,
        rotation=np.eye(3),
        alpha=100.0,
    )
    spec2 = CylinderSpec(
        center=np.array([0.10, -0.05, 0.20]),
        radius=0.1,
        height=0.5,
        rotation=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        alpha=100.0,
    )
    spec3 = CylinderSpec(
        center=np.array([-1.26, 0, 0]),
        radius=0.1,
        height=0.5,
        rotation=np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        alpha=100.0,
    )

    specs = [spec1]  # , spec2, spec3]

    # sample points in each cylinder
    NUM_POINTS_1 = 1200
    NUM_POINTS_2 = 1200
    NUM_POINTS_3 = 1200
    points1 = generate_points_in_cylinder(spec1, NUM_POINTS_1, seed=1)
    points2 = generate_points_in_cylinder(spec2, NUM_POINTS_2, seed=2)
    points3 = generate_points_in_cylinder(spec3, NUM_POINTS_3, seed=3)
    points_list = [points1]  # , points2, points3]

    # truncation
    n_n, n_m = 5, 5

    # fixed mascon positions
    r1 = np.array([0.35, -0.02, 0.05])
    r2 = np.array([-0.10, 0.08, 0.02])

    mu_true = np.array([5e-9, 2e-9], float)
    mu1_true, mu2_true = float(mu_true[0]), float(mu_true[1])

    # (1) Forward: poly + anomalies -> fit coeffs for BOTH cylinders
    fwd = fit_cylinders_for_poly_plus_2fixedmascons(
        vertices,
        faces,
        DENSITY,
        specs=specs,
        points_list=points_list,
        n_n=n_n,
        n_m=n_m,
        r1=r1,
        r2=r2,
        mu1=mu1_true,
        mu2=mu2_true,
        parallel=False,
        enforce_B0n_flag=True,
        softening=0.0,
    )
    coeffs_list = fwd["coeffs_list"]

    # (2a) Inverse LSQ: use BOTH cylinders together
    x0_mu = np.array([0, 0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1e-6, 1e-6])

    res_lsq = estimate_2mus_from_coeffs_lsq_multi(
        coeffs_list=coeffs_list,
        vertices=vertices,
        faces=faces,
        density=DENSITY,
        specs=specs,
        points_list=points_list,
        n_n=n_n,
        n_m=n_m,
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

    # (2b) Inverse MCMC: posterior using BOTH cylinders together
    res_mcmc = estimate_2mus_from_coeffs_mcmc_multi(
        coeffs_list=coeffs_list,
        vertices=vertices,
        faces=faces,
        density=DENSITY,
        specs=specs,
        points_list=points_list,
        n_n=n_n,
        n_m=n_m,
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

    # ==========================
    # PLOTS (ONLY AT THE END)
    # ==========================
    figs = []

    # spectrum per cylinder
    figs.append(
        plot_rms_spectrum(
            coeffs_list[0], n_n, n_m, "Cylinder #1 RMS spectrum (poly + anomalies)"
        )
    )
    # figs.append(
    #    plot_rms_spectrum(
    #        coeffs_list[1], n_n, n_m, "Cylinder #2 RMS spectrum (poly + anomalies)"
    #    )
    # )
    # figs.append(
    #    plot_rms_spectrum(
    #        coeffs_list[2], n_n, n_m, "Cylinder #3 RMS spectrum (poly + anomalies)"
    #    )
    # )

    mascons = np.vstack([r1.reshape(1, 3), r2.reshape(1, 3)])
    fig_shape, ax_shape = plot_shape_and_mascons_matplotlib(
        vertices,
        faces,
        mascons,
        title="EROS shape + fixed mascons",
        face_alpha=0.20,
        decimate_faces=5,  # bump up if too slow
    )
    figs.append(fig_shape)

    # mu comparison
    figs.append(
        plot_mu_comparison(
            mu_true=np.array([mu1_true, mu2_true]),
            mu_lsq=mu_lsq,
            mu_mcmc=mu_mcmc,
        )
    )

    # residual hist per cylinder (acc-only mismatch)
    res_blocks = build_cyl_acc_residual_and_model_per_cyl(
        coeffs_list=coeffs_list,
        vertices=vertices,
        faces=faces,
        density=DENSITY,
        specs=specs,
        points_list=points_list,
        n_n=n_n,
        n_m=n_m,
        r1=r1,
        r2=r2,
        mu_hat=mu_mcmc,
        parallel=False,
        softening=0.0,
    )
    figs.append(
        plot_residual_norms(
            specs[0], points_list[0], res_blocks[0][0], res_blocks[0][1]
        )
    )
    # figs.append(
    #    plot_residual_norms(
    #        specs[1], points_list[1], res_blocks[1][0], res_blocks[1][1]
    #    )
    # )

    if res_mcmc.corner_fig is not None:
        figs.append(res_mcmc.corner_fig)

    # Print summary
    print("\n=== TRUE Δmu ===", np.array([mu1_true, mu2_true]))
    print("=== LSQ  Δmu ===", mu_lsq)
    print("=== MCMC Δmu ===", mu_mcmc)
    print("LSQ cost =", res_lsq.cost)
    print("MCMC cost =", res_mcmc.cost, " (using MAP sample)")
    print("MCMC sigma_like =", res_mcmc.sigma_like)

    pct_lsq = 100.0 * (mu_lsq - mu_true) / (np.abs(mu_true))
    pct_mcmc = 100.0 * (mu_mcmc - mu_true) / (np.abs(mu_true))

    print("\n=== % ERROR (LSQ)  ===", pct_lsq)
    print("=== % ERROR (MCMC) ===", pct_mcmc)
    print("=== |%| (LSQ)  ===", np.abs(pct_lsq))
    print("=== |%| (MCMC) ===", np.abs(pct_mcmc))

    plt.show()
