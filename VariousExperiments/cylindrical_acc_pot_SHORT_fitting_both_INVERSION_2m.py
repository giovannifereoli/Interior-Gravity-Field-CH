"""
Two-mascon TAG-style forward + inverse (from scratch) — but NOW:

✅ Keep the whole pipeline (forward synth -> cylinder-coeff LS -> inverse LS + MCMC)
✅ Do ALL plots at the END (no plt.show() inside solver functions)
✅ Estimate ONLY mascon "density variation" (i.e., Δmu1, Δmu2) with FIXED mascon positions
   -> positions are treated as known (e.g., you pick where TAG disturbed regolith)
   -> only solve for mu1, mu2 (highly reduces correlations / label switching)

Interpretation:
  polyhedron has constant density (known).
  mascons represent *additional* mass anomaly (Δmu = G Δm) relative to that constant-density poly.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import matplotlib.pyplot as plt
from scipy.special import jv as BesselJ, jvp as BesselJp, jn_zeros
from scipy.optimize import least_squares

import mesh_utility
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable


# ======================================================================================
# Cylinder spec + sampling + coordinate transforms
# ======================================================================================


@dataclass
class CylinderSpec:
    center: np.ndarray  # (3,)
    radius: float
    height: float
    rotation: np.ndarray  # (3,3) local->global
    alpha: float  # R_alpha = alpha*radius


def generate_points_in_cylinder(
    spec: CylinderSpec, num_points: int, seed: int = 1
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, num_points)
    r = np.sqrt(rng.uniform(0.0, spec.radius**2, num_points))
    z = rng.uniform(0.0, spec.height, num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    local = np.column_stack((x, y, z))
    return local @ spec.rotation.T + spec.center


def _cyl_coords(
    spec: CylinderSpec, points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pl = (points - spec.center) @ spec.rotation.T
    rho = np.sqrt(pl[:, 0] ** 2 + pl[:, 1] ** 2)
    phi = np.arctan2(pl[:, 1], pl[:, 0])
    z = pl[:, 2]
    return rho, phi, z


def cart_to_cyl_acc(
    spec: CylinderSpec, points: np.ndarray, acc_cart: np.ndarray
) -> np.ndarray:
    rho, phi, _ = _cyl_coords(spec, points)
    acc_l = acc_cart @ spec.rotation.T
    ax, ay, az = acc_l[:, 0], acc_l[:, 1], acc_l[:, 2]
    a_rho = ax * np.cos(phi) + ay * np.sin(phi)
    a_phi = -ax * np.sin(phi) + ay * np.cos(phi)
    return np.column_stack((a_rho, a_phi, az))


# ======================================================================================
# Point-mass mascon model
# ======================================================================================


def mascon_potential_acc(
    points: np.ndarray, r_m: np.ndarray, mu: float, softening: float = 0.0
):
    """
    V = -mu/r
    a = -mu * dr / r^3
    """
    dr = points - r_m[None, :]
    r2 = np.sum(dr * dr, axis=1) + softening**2
    r = np.sqrt(r2)
    V = -mu / r
    a = -mu * dr / (r[:, None] ** 3 + 1e-30)
    return V, a


# ======================================================================================
# Cylindrical basis: linear systems + forward eval from coeffs
# ======================================================================================


def prepare_linear_system_for_cyl_acc(
    spec: CylinderSpec, points: np.ndarray, cyl_acc: np.ndarray, n_n: int, n_m: int
) -> Tuple[np.ndarray, np.ndarray]:
    rho, phi, z = _cyl_coords(spec, points)
    N = points.shape[0]
    P = 2 * n_n * n_m
    A = np.zeros((3 * N, P), float)
    b = cyl_acc.reshape(-1, order="C")

    def k_mn(m, n):  # n starts at 1
        return jn_zeros(m, n)[n - 1]

    R_alpha = spec.alpha * spec.radius

    idx = 0
    r0 = np.arange(N) * 3
    r1 = r0 + 1
    r2 = r0 + 2

    for m in range(n_m):
        cos_m = np.cos(m * phi)
        sin_m = np.sin(m * phi)
        for n in range(1, n_n + 1):
            k = k_mn(m, n)
            exp_term = np.exp(-k * z / R_alpha)
            J = BesselJ(m, k * rho / R_alpha)
            Jp = BesselJp(m, k * rho / R_alpha)

            dV_drho = (k / R_alpha) * exp_term * Jp
            dV_dphi = exp_term * J * (m / (rho + 1e-14))
            dV_dz = (-k / R_alpha) * exp_term * J

            A[r0, idx] = dV_drho * cos_m
            A[r0, idx + 1] = dV_drho * sin_m

            A[r1, idx] = dV_dphi * (-sin_m)
            A[r1, idx + 1] = dV_dphi * (cos_m)

            A[r2, idx] = dV_dz * cos_m
            A[r2, idx + 1] = dV_dz * sin_m

            idx += 2

    return A, b


def prepare_linear_system_for_cyl_pot(
    spec: CylinderSpec, points: np.ndarray, pot: np.ndarray, n_n: int, n_m: int
) -> Tuple[np.ndarray, np.ndarray]:
    rho, phi, z = _cyl_coords(spec, points)
    N = points.shape[0]
    P = 2 * n_n * n_m
    A = np.zeros((N, P), float)
    b = pot.astype(float)

    def k_mn(m, n):
        return jn_zeros(m, n)[n - 1]

    R_alpha = spec.alpha * spec.radius

    idx = 0
    for m in range(n_m):
        cos_m = np.cos(m * phi)
        sin_m = np.sin(m * phi)
        for n in range(1, n_n + 1):
            k = k_mn(m, n)
            exp_term = np.exp(-k * z / R_alpha)
            J = BesselJ(m, k * rho / R_alpha)
            A[:, idx] = exp_term * J * cos_m
            A[:, idx + 1] = exp_term * J * sin_m
            idx += 2

    return A, b


def zero_B0n(coeffs: np.ndarray, n_n: int):
    # enforce B_0n = 0 for m=0
    for n0 in range(n_n):
        coeffs[2 * n0 + 1] = 0.0


def cyl_acc_from_coeffs(
    spec: CylinderSpec, points: np.ndarray, coeffs: np.ndarray, n_n: int, n_m: int
) -> np.ndarray:
    A_acc, _ = prepare_linear_system_for_cyl_acc(
        spec, points, np.zeros((points.shape[0], 3)), n_n, n_m
    )
    pred = A_acc @ coeffs
    return pred.reshape((-1, 3), order="C")


def cyl_pot_from_coeffs(
    spec: CylinderSpec, points: np.ndarray, coeffs: np.ndarray, n_n: int, n_m: int
) -> np.ndarray:
    A_pot, _ = prepare_linear_system_for_cyl_pot(
        spec, points, np.zeros(points.shape[0]), n_n, n_m
    )
    return A_pot @ coeffs


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


# ======================================================================================
# (1) FORWARD: poly + TWO fixed-position mascons -> fit cylinder coeffs (LS)
# ======================================================================================


def fit_cylinder_for_poly_plus_2fixedmascons(
    vertices,
    faces,
    density: float,
    spec: CylinderSpec,
    points: np.ndarray,
    n_n: int,
    n_m: int,
    r1: np.ndarray,
    r2: np.ndarray,
    mu1: float,
    mu2: float,
    parallel: bool = False,
    enforce_B0n_flag: bool = True,
    softening: float = 0.0,
):
    # poly (constant density)
    pot_poly, acc_poly = eval_poly_gravity(
        vertices, faces, density, points, parallel=parallel
    )

    # mascon anomalies (Δmu wrt constant-density poly)
    pot1, acc1 = mascon_potential_acc(points, r1, mu1, softening=softening)
    pot2, acc2 = mascon_potential_acc(points, r2, mu2, softening=softening)

    pot_tot = pot_poly + pot1 + pot2
    acc_tot = acc_poly + acc1 + acc2

    cyl_acc_tot = cart_to_cyl_acc(spec, points, acc_tot)

    A_acc, b_acc = prepare_linear_system_for_cyl_acc(
        spec, points, cyl_acc_tot, n_n, n_m
    )
    A_pot, b_pot = prepare_linear_system_for_cyl_pot(spec, points, pot_tot, n_n, n_m)

    aug_A = np.vstack([A_acc, A_pot])
    aug_b = np.hstack([b_acc, b_pot])

    coeffs, *_ = np.linalg.lstsq(aug_A, aug_b, rcond=None)
    if enforce_B0n_flag:
        zero_B0n(coeffs, n_n=n_n)

    # also return some truth fields for plotting
    return dict(
        coeffs=coeffs,
        pot_poly=pot_poly,
        acc_poly=acc_poly,
        pot_tot=pot_tot,
        acc_tot=acc_tot,
        mu_true=np.array([mu1, mu2], float),
        r1=r1,
        r2=r2,
    )


# ======================================================================================
# (2) INVERSE: given coeffs + known poly, estimate ONLY mu1, mu2 (LS + MCMC)
# ======================================================================================


def estimate_2mus_from_coeffs_lsq(
    coeffs: np.ndarray,
    vertices,
    faces,
    density: float,
    spec: CylinderSpec,
    points: np.ndarray,
    n_n: int,
    n_m: int,
    r1: np.ndarray,
    r2: np.ndarray,
    x0_mu: np.ndarray,  # (2,) initial guess [mu1, mu2]
    bounds_mu: Tuple[np.ndarray, np.ndarray],  # (lb, ub) each (2,)
    parallel: bool = False,
    softening: float = 0.0,
    use_potential: bool = True,
    w_acc: float = 1.0,
    w_pot: float = 1.0,
):
    # target from coeffs
    cyl_acc_target = cyl_acc_from_coeffs(spec, points, coeffs, n_n, n_m)
    pot_target = (
        cyl_pot_from_coeffs(spec, points, coeffs, n_n, n_m) if use_potential else None
    )

    # known poly
    pot_poly, acc_poly = eval_poly_gravity(
        vertices, faces, density, points, parallel=parallel
    )
    cyl_acc_poly = cart_to_cyl_acc(spec, points, acc_poly)

    # residual field attributed to anomalies (mascons)
    cyl_acc_resid = cyl_acc_target - cyl_acc_poly
    pot_resid = (pot_target - pot_poly) if use_potential else None

    def fun(mu_vec):
        mu1, mu2 = float(mu_vec[0]), float(mu_vec[1])
        pot1, acc1 = mascon_potential_acc(points, r1, mu1, softening=softening)
        pot2, acc2 = mascon_potential_acc(points, r2, mu2, softening=softening)

        pot_m = pot1 + pot2
        acc_m = acc1 + acc2
        cyl_acc_m = cart_to_cyl_acc(spec, points, acc_m)

        r_acc = (cyl_acc_m - cyl_acc_resid).reshape(-1, order="C") * w_acc
        if use_potential:
            r_pot = (pot_m - pot_resid) * w_pot
            return np.hstack([r_acc, r_pot])
        return r_acc

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


def estimate_2mus_from_coeffs_mcmc(
    coeffs: np.ndarray,
    vertices,
    faces,
    density: float,
    spec: CylinderSpec,
    points: np.ndarray,
    n_n: int,
    n_m: int,
    r1: np.ndarray,
    r2: np.ndarray,
    x0_mu: np.ndarray,  # (2,)
    bounds_mu: Tuple[np.ndarray, np.ndarray],  # (lb, ub) each (2,)
    parallel: bool = False,
    softening: float = 0.0,
    use_potential: bool = True,
    w_acc: float = 1.0,
    w_pot: float = 1.0,
    sigma_like: Optional[float] = None,  # if None, auto from RMS
    nwalkers: int = 48,
    n_burn: int = 1500,
    n_steps: int = 4000,
    thin: int = 10,
    seed: int = 123,
):
    from types import SimpleNamespace

    # optional corner
    try:
        import corner

        _HAS_CORNER = True
    except Exception:
        _HAS_CORNER = False

    try:
        import emcee
    except Exception as e:
        raise ImportError("emcee is required: pip install emcee") from e

    # target from coeffs
    cyl_acc_target = cyl_acc_from_coeffs(spec, points, coeffs, n_n, n_m)
    pot_target = (
        cyl_pot_from_coeffs(spec, points, coeffs, n_n, n_m) if use_potential else None
    )

    # known poly
    pot_poly, acc_poly = eval_poly_gravity(
        vertices, faces, density, points, parallel=parallel
    )
    cyl_acc_poly = cart_to_cyl_acc(spec, points, acc_poly)

    # residual field attributed to anomalies
    cyl_acc_resid = cyl_acc_target - cyl_acc_poly
    pot_resid = (pot_target - pot_poly) if use_potential else None

    y_acc = cyl_acc_resid.reshape(-1, order="C")
    if use_potential:
        y = np.hstack([w_acc * y_acc, w_pot * pot_resid])
    else:
        y = w_acc * y_acc

    lb, ub = np.asarray(bounds_mu[0], float), np.asarray(bounds_mu[1], float)

    def log_prior(mu_vec):
        mu_vec = np.asarray(mu_vec, float)
        if np.any(mu_vec < lb) or np.any(mu_vec > ub):
            return -np.inf
        # enforce positivity (also via bounds, but keep explicit)
        if mu_vec[0] <= 0.0 or mu_vec[1] <= 0.0:
            return -np.inf
        return 0.0

    def model_vec(mu_vec):
        mu1, mu2 = float(mu_vec[0]), float(mu_vec[1])
        pot1, acc1 = mascon_potential_acc(points, r1, mu1, softening=softening)
        pot2, acc2 = mascon_potential_acc(points, r2, mu2, softening=softening)

        pot_m = pot1 + pot2
        acc_m = acc1 + acc2
        cyl_acc_m = cart_to_cyl_acc(spec, points, acc_m)

        m_acc = cyl_acc_m.reshape(-1, order="C")
        if use_potential:
            return np.hstack([w_acc * m_acc, w_pot * pot_m])
        return w_acc * m_acc

    # likelihood scale
    if sigma_like is None:
        y_rms = np.sqrt(np.mean(y**2)) + 1e-30
        sigma_like = 0.01 * y_rms  # knob
    inv_sigma2 = 1.0 / (sigma_like**2)

    def log_prob(mu_vec):
        lp = log_prior(mu_vec)
        if not np.isfinite(lp):
            return -np.inf
        r = model_vec(mu_vec) - y
        return lp - 0.5 * np.sum(r * r) * inv_sigma2

    ndim = 2
    rng = np.random.default_rng(seed)

    # init walkers around x0_mu
    scale = np.array([0.2 * max(x0_mu[0], 1e-16), 0.2 * max(x0_mu[1], 1e-16)], float)
    scale = np.maximum(scale, np.array([1e-12, 1e-12]))
    p0 = x0_mu[None, :] + rng.normal(size=(nwalkers, ndim)) * scale[None, :]

    # clip to bounds + positivity
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
            truths=mu_hat,  # use mu_hat as a reference marker; replace with mu_true if you have it
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
# Plot helpers (called ONLY at the end)
# ======================================================================================


def rms_spectrum_by_order(coeffs: np.ndarray, n_n: int, n_m: int) -> np.ndarray:
    A = np.zeros((n_m, n_n))
    B = np.zeros((n_m, n_n))
    k = 0
    for m in range(n_m):
        for n in range(n_n):
            A[m, n] = coeffs[k]
            B[m, n] = coeffs[k + 1]
            k += 2
    return np.sqrt(np.sum(A * A + B * B, axis=1))


def plot_rms_spectrum(coeffs: np.ndarray, n_n: int, n_m: int, title: str):
    m = np.arange(n_m)
    rms = rms_spectrum_by_order(coeffs, n_n, n_m)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(m, rms, marker="o", linestyle="-")
    ax.set_xlabel("Order $m$")
    ax.set_ylabel(r"RMS $\sqrt{\sum_n (A_{mn}^2+B_{mn}^2)}$")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.minorticks_on()
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
    ax.set_title("Mascon anomaly recovery")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return fig


def plot_residual_norms(
    spec: CylinderSpec,
    points: np.ndarray,
    cyl_acc_resid: np.ndarray,
    cyl_acc_model: np.ndarray,
):
    # show norm of residual field vs model mismatch (cylindrical)
    nr = np.linalg.norm(cyl_acc_resid, axis=1)
    nm = np.linalg.norm(cyl_acc_model - cyl_acc_resid, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(nr, bins=50, alpha=0.6, density=True, label=r"$\|\mathbf{a}_{res}\|$")
    ax.hist(
        nm,
        bins=50,
        alpha=0.6,
        density=True,
        label=r"$\|\mathbf{a}_{model}-\mathbf{a}_{res}\|$",
    )
    ax.set_xlabel("Acceleration norm (cyl units)")
    ax.set_ylabel("PDF")
    ax.set_title("Residual field vs post-fit mismatch (acc only)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return fig


# ======================================================================================
# Main
# ======================================================================================

if __name__ == "__main__":
    # ---------------------------
    # Load mesh (EROS)
    # ---------------------------
    vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    DENSITY = 1.0  # constant-density poly baseline

    # ---------------------------
    # Cylinder (your same geometry)
    # ---------------------------
    spec = CylinderSpec(
        center=np.array([0.0, 0.0, 0.28]),
        radius=0.10,
        height=0.50,
        rotation=np.eye(3),
        alpha=100.0,
    )

    # ---------------------------
    # Sampling points in cylinder
    # ---------------------------
    NUM_POINTS = 1200
    points = generate_points_in_cylinder(spec, NUM_POINTS, seed=1)

    # ---------------------------
    # Cyl basis truncation
    # ---------------------------
    n_n, n_m = 25, 25

    # ---------------------------
    # FIXED mascon positions (you choose; must be inside body)
    # ---------------------------
    r1 = np.array([0.35, -0.02, 0.05])
    r2 = np.array([0.10, 0.18, 0.02])

    # TRUE anomalies (Δmu wrt constant-density poly)
    mu_true = np.array([5e-9, 2e-9], float)  # tune to your LU/TU scaling
    mu1_true, mu2_true = float(mu_true[0]), float(mu_true[1])

    # ---------------------------
    # (1) Forward: synth field poly + anomalies -> fit cylinder coeffs
    # ---------------------------
    fwd = fit_cylinder_for_poly_plus_2fixedmascons(
        vertices,
        faces,
        DENSITY,
        spec,
        points,
        n_n,
        n_m,
        r1=r1,
        r2=r2,
        mu1=mu1_true,
        mu2=mu2_true,
        parallel=False,
        enforce_B0n_flag=True,
        softening=0.0,
    )
    coeffs = fwd["coeffs"]

    # ---------------------------
    # (2a) Inverse LSQ: estimate mu1, mu2 only
    # ---------------------------
    x0_mu = np.array([1e-9, 1e-9])  # initial guess for anomalies
    lb = np.array([0.0, 0.0])
    ub = np.array([1e-6, 1e-6])
    res_lsq = estimate_2mus_from_coeffs_lsq(
        coeffs,
        vertices,
        faces,
        DENSITY,
        spec,
        points,
        n_n,
        n_m,
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
    # (2b) Inverse MCMC: posterior on mu1, mu2
    #      start from LSQ estimate (good)
    # ---------------------------
    res_mcmc = estimate_2mus_from_coeffs_mcmc(
        coeffs,
        vertices,
        faces,
        DENSITY,
        spec,
        points,
        n_n,
        n_m,
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
    # Build fields for plotting residuals (acc-only mismatch histogram)
    # ---------------------------
    # residual field attributed to anomalies:
    cyl_acc_target = cyl_acc_from_coeffs(spec, points, coeffs, n_n, n_m)
    pot_poly, acc_poly = eval_poly_gravity(
        vertices, faces, DENSITY, points, parallel=False
    )
    cyl_acc_poly = cart_to_cyl_acc(spec, points, acc_poly)
    cyl_acc_resid = cyl_acc_target - cyl_acc_poly

    # model using recovered mu (MCMC median)
    pot1h, acc1h = mascon_potential_acc(points, r1, mu_mcmc[0], softening=0.0)
    pot2h, acc2h = mascon_potential_acc(points, r2, mu_mcmc[1], softening=0.0)
    cyl_acc_model = cart_to_cyl_acc(spec, points, acc1h + acc2h)

    # ==================================================================================
    # PLOTS (ONLY HERE)
    # ==================================================================================
    figs = []

    figs.append(
        plot_rms_spectrum(
            coeffs, n_n, n_m, "Cylinder coefficient RMS spectrum (poly + anomalies)"
        )
    )
    figs.append(
        plot_mu_comparison(
            mu_true=np.array([mu1_true, mu2_true]), mu_lsq=mu_lsq, mu_mcmc=mu_mcmc
        )
    )
    figs.append(plot_residual_norms(spec, points, cyl_acc_resid, cyl_acc_model))

    if res_mcmc.corner_fig is not None:
        figs.append(res_mcmc.corner_fig)

    # Print summary
    print("\n=== TRUE Δmu ===", np.array([mu1_true, mu2_true]))
    print("=== LSQ  Δmu ===", mu_lsq)
    print("=== MCMC Δmu ===", mu_mcmc)
    print("LSQ cost =", res_lsq.cost)
    print("MCMC cost =", res_mcmc.cost, " (using MAP sample)")
    print("MCMC sigma_like =", res_mcmc.sigma_like)

    plt.show()
