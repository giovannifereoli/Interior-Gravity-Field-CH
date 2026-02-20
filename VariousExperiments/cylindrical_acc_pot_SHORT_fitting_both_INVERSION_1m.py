"""
Forward + inverse test (TAG-style “hidden mass”):
  1) Generate synthetic gravity in a cylinder from: constant-density polyhedron + ONE big interior mascon
  2) Fit cylindrical-harmonic coefficients (linear LS) to that synthetic gravity (acc + pot)
  3) Given the fitted cylinder coefficients AND the known polyhedron, estimate back the mascon (x,y,z,mu)

Notes:
- I treat the mascon as a point mass with parameter mu = G*m in your working LU/TU units.
- The inverse problem is nonlinear -> solved with scipy.optimize.least_squares.
- I do the inversion in CYLINDRICAL components (rho,phi,z) inside your cylinder.

Dependencies:
  numpy, scipy, polyhedral_gravity, mesh_utility
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

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
    alpha: float  # ALPHA scaling (R_alpha = alpha*radius)


def generate_points_in_cylinder(
    spec: CylinderSpec, num_points: int, seed: int = 1
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, num_points)
    r = np.sqrt(rng.uniform(0, spec.radius**2, num_points))
    z = rng.uniform(0, spec.height, num_points)
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


def cyl_to_cart_acc(
    spec: CylinderSpec, points: np.ndarray, acc_cyl: np.ndarray
) -> np.ndarray:
    """
    Convert cylindrical acceleration components (rho,phi,z) in cylinder local frame to CARTESIAN global.
    """
    _, phi, _ = _cyl_coords(spec, points)
    a_rho, a_phi, a_z = acc_cyl[:, 0], acc_cyl[:, 1], acc_cyl[:, 2]

    # local Cartesian components
    ax_l = a_rho * np.cos(phi) - a_phi * np.sin(phi)
    ay_l = a_rho * np.sin(phi) + a_phi * np.cos(phi)
    az_l = a_z
    acc_l = np.column_stack((ax_l, ay_l, az_l))

    # rotate local->global
    return acc_l @ spec.rotation


# ======================================================================================
# Mascon model (point mass)
# ======================================================================================


def mascon_potential_acc(
    points: np.ndarray, r_m: np.ndarray, mu: float, softening: float = 0.0
):
    """
    Point mass potential and acceleration.
      V = -mu / r
      a = -mu * (r_vec) / r^3
    softening: adds eps^2 to r^2 to avoid singularities (optional).
    """
    dr = points - r_m[None, :]
    r2 = np.sum(dr * dr, axis=1) + softening**2
    r = np.sqrt(r2)
    V = -mu / r
    a = -mu * dr / (r[:, None] ** 3 + 1e-30)
    return V, a


# ======================================================================================
# Cylindrical basis: linear systems + forward eval from coefficients
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


def zero_B0n(params: np.ndarray, n_n: int):
    # B_0n are at indices 2*(n-1)+1 for n=1..n_n -> i = 2*n + 1 if n is 0-indexed
    for n0 in range(n_n):
        params[2 * n0 + 1] = 0.0


def cyl_acc_from_coeffs(
    spec: CylinderSpec, points: np.ndarray, coeffs: np.ndarray, n_n: int, n_m: int
) -> np.ndarray:
    A_acc, _ = prepare_linear_system_for_cyl_acc(
        spec, points, np.zeros((points.shape[0], 3)), n_n, n_m
    )
    pred = A_acc @ coeffs
    return pred.reshape((-1, 3), order="C")


# ======================================================================================
# Poly gravity evaluation (baseline)
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
# 1) FORWARD: poly + mascon -> fit cylinder coefficients
# ======================================================================================


def fit_cylinder_for_poly_plus_mascon(
    vertices,
    faces,
    density: float,
    spec: CylinderSpec,
    points: np.ndarray,
    n_n: int,
    n_m: int,
    mascon_r: np.ndarray,
    mascon_mu: float,
    parallel: bool = False,
    enforce_B0n: bool = True,
):
    # poly
    pot_poly, acc_poly = eval_poly_gravity(
        vertices, faces, density, points, parallel=parallel
    )

    # mascon
    pot_m, acc_m = mascon_potential_acc(points, mascon_r, mascon_mu)

    # total field
    pot_tot = pot_poly + pot_m
    acc_tot = acc_poly + acc_m

    # build LS on CYL acceleration + potential
    cyl_acc_tot = cart_to_cyl_acc(spec, points, acc_tot)

    A_acc, b_acc = prepare_linear_system_for_cyl_acc(
        spec, points, cyl_acc_tot, n_n, n_m
    )
    A_pot, b_pot = prepare_linear_system_for_cyl_pot(spec, points, pot_tot, n_n, n_m)

    aug_A = np.vstack([A_acc, A_pot])
    aug_b = np.hstack([b_acc, b_pot])

    coeffs, *_ = np.linalg.lstsq(aug_A, aug_b, rcond=None)
    if enforce_B0n:
        zero_B0n(coeffs, n_n=n_n)

    return coeffs, (pot_poly, acc_poly), (pot_tot, acc_tot)


# ======================================================================================
# 2) INVERSE: cylinder coeffs + known poly -> estimate mascon (x,y,z,mu)
# ======================================================================================


def estimate_mascon_from_coeffs(
    coeffs: np.ndarray,
    vertices,
    faces,
    density: float,
    spec: CylinderSpec,
    points: np.ndarray,
    n_n: int,
    n_m: int,
    x0: np.ndarray,  # initial guess [x,y,z,mu]
    bounds: Tuple[np.ndarray, np.ndarray],
    parallel: bool = False,
    softening: float = 0.0,
    use_potential: bool = True,
    w_acc: float = 1.0,
    w_pot: float = 1.0,
):
    """
    Same signature as your least_squares version, but runs MCMC (emcee) and
    produces a corner plot.

    Returns:
      dict with keys:
        - "sampler": emcee.EnsembleSampler
        - "samples": (Nsamp, 4) posterior samples after burn-in/thinning
        - "log_prob": (Nsamp,) log posterior for those samples
        - "summary": dict with median and 16/84 percentiles
        - "corner_fig": matplotlib Figure (if corner is installed)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Optional but recommended for corner plots
    try:
        import corner

        _HAS_CORNER = True
    except Exception:
        _HAS_CORNER = False

    try:
        import emcee
    except Exception as e:
        raise ImportError("emcee is required: pip install emcee") from e

    # -----------------------------
    # Build the same residual target you used in LS
    # -----------------------------
    cyl_acc_target = cyl_acc_from_coeffs(spec, points, coeffs, n_n, n_m)  # (N,3)
    pot_target = None
    if use_potential:
        A_pot, _ = prepare_linear_system_for_cyl_pot(
            spec, points, np.zeros(points.shape[0]), n_n, n_m
        )
        pot_target = A_pot @ coeffs  # (N,)

    pot_poly, acc_poly = eval_poly_gravity(
        vertices, faces, density, points, parallel=parallel
    )
    cyl_acc_poly = cart_to_cyl_acc(spec, points, acc_poly)

    cyl_acc_resid = cyl_acc_target - cyl_acc_poly
    pot_resid = (pot_target - pot_poly) if use_potential else None

    # Flattened "data" vector we want mascon to match in cylindrical coords
    y_acc = cyl_acc_resid.reshape(-1, order="C")  # (3N,)
    if use_potential:
        y = np.hstack([w_acc * y_acc, w_pot * pot_resid])
    else:
        y = w_acc * y_acc

    # -----------------------------
    # Likelihood model
    # -----------------------------
    # We need a noise scale to convert residual to log-likelihood.
    # If you don't have an actual measurement sigma, use a robust proxy:
    # sigma_like is set from RMS of target signal (so it's unit-consistent).
    # You can tighten/loosen it to control posterior width.
    y_rms = np.sqrt(np.mean(y**2)) + 1e-30
    sigma_like = 0.1 * y_rms  # <-- knob: 1% of RMS signal (adjust if needed)

    inv_sigma2 = 1.0 / (sigma_like**2)

    # -----------------------------
    # Prior
    # -----------------------------
    lb, ub = np.asarray(bounds[0], float), np.asarray(bounds[1], float)

    def log_prior(theta: np.ndarray) -> float:
        # Uniform prior inside bounds
        if np.any(theta < lb) or np.any(theta > ub):
            return -np.inf

        # Optional: weak log-prior to keep mu > 0 well-behaved near 0
        # (still uniform, but prevents emcee getting stuck at exactly 0)
        mu = theta[3]
        if mu <= 0.0:
            return -np.inf
        return 0.0

    # -----------------------------
    # Forward model (mascon only)
    # -----------------------------
    def model_vec(theta: np.ndarray) -> np.ndarray:
        r_m = theta[:3]
        mu = theta[3]
        pot_m, acc_m = mascon_potential_acc(points, r_m, mu, softening=softening)
        cyl_acc_m = cart_to_cyl_acc(spec, points, acc_m)
        m_acc = cyl_acc_m.reshape(-1, order="C")
        if use_potential:
            return np.hstack([w_acc * m_acc, w_pot * pot_m])
        return w_acc * m_acc

    # -----------------------------
    # Posterior
    # -----------------------------
    def log_prob(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        r = model_vec(theta) - y
        # Gaussian iid likelihood
        return lp - 0.5 * np.sum(r * r) * inv_sigma2

    # -----------------------------
    # emcee configuration
    # -----------------------------
    ndim = 4
    # walkers: rule of thumb >= 2*ndim, better 32-64
    nwalkers = 48

    # Initialize walkers in a small Gaussian ball around x0, clipped to bounds
    rng = np.random.default_rng(123)
    p0 = x0[None, :] + rng.normal(
        scale=[1e-3, 1e-3, 1e-3, 0.1 * max(x0[3], 1e-8)], size=(nwalkers, ndim)
    )

    # enforce bounds on init
    for i in range(nwalkers):
        p0[i] = np.minimum(np.maximum(p0[i], lb), ub)
        # ensure mu positive
        if p0[i, 3] <= 0:
            p0[i, 3] = max(1e-12, min(ub[3], abs(p0[i, 3]) + 1e-12))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

    # Run
    n_burn = 1500
    n_steps = 4000
    thin = 10

    # burn-in
    sampler.run_mcmc(p0, n_burn, progress=True)
    sampler.reset()

    # production
    sampler.run_mcmc(None, n_steps, progress=True)

    # Extract samples
    samples = sampler.get_chain(flat=True, thin=thin)  # (Nsamp, ndim)
    log_prob_flat = sampler.get_log_prob(flat=True, thin=thin)  # (Nsamp,)

    # -----------------------------
    # Summaries
    # -----------------------------
    def quantiles(x):
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        return q50, q16, q84

    names = ["x", "y", "z", "mu"]
    summary = {}
    for j, name in enumerate(names):
        med, q16, q84 = quantiles(samples[:, j])
        summary[name] = dict(
            median=med, p16=q16, p84=q84, minus=med - q16, plus=q84 - med
        )

    # -----------------------------
    # Corner plot
    # -----------------------------
    corner_fig = None
    if _HAS_CORNER:
        corner_fig = corner.corner(
            samples,
            labels=[r"$x$", r"$y$", r"$z$", r"$\mu$"],
            truths=x0,
            show_titles=True,
            title_fmt=".3e",
            quantiles=[0.16, 0.50, 0.84],
        )
        corner_fig.suptitle("Mascon posterior (emcee)", y=1.02)
        plt.tight_layout()
        plt.show()

    from types import SimpleNamespace

    # --- point estimate: posterior median ---
    x_hat = np.median(samples, axis=0)

    # --- covariance estimate from posterior ---
    cov_hat = np.cov(samples, rowvar=False)

    # --- negative log-likelihood cost (LS-style) ---
    # cost ≈ 0.5 * sum(r^2) using MAP sample
    i_map = np.argmax(log_prob_flat)
    theta_map = samples[i_map]
    r_map = model_vec(theta_map) - y
    cost_hat = 0.5 * np.sum(r_map**2)

    res = SimpleNamespace(
        x=x_hat,  # LS-style solution vector
        cov=cov_hat,  # posterior covariance
        cost=cost_hat,  # LS-like cost
        success=True,
        message="MCMC posterior median estimate",
        samples=samples,
        sampler=sampler,
        corner_fig=corner_fig,
    )

    return res


def estimate_mascon_from_coeffs2(
    coeffs: np.ndarray,
    vertices,
    faces,
    density: float,
    spec: CylinderSpec,
    points: np.ndarray,
    n_n: int,
    n_m: int,
    x0: np.ndarray,  # initial guess [x,y,z,mu]
    bounds: Tuple[np.ndarray, np.ndarray],
    parallel: bool = False,
    softening: float = 0.0,
    use_potential: bool = True,
    w_acc: float = 1.0,
    w_pot: float = 1.0,
):
    """
    We match the *field represented by the cylinder coefficients* with:
        poly_known + mascon_unknown.

    Target:
      cyl_coeff_field(points) - poly(points)  ≈  mascon(points)
    """

    # compute target from coefficients (in cylindrical components)
    cyl_acc_target = cyl_acc_from_coeffs(spec, points, coeffs, n_n, n_m)  # (N,3)
    pot_target = None
    if use_potential:
        A_pot, _ = prepare_linear_system_for_cyl_pot(
            spec, points, np.zeros(points.shape[0]), n_n, n_m
        )
        pot_target = A_pot @ coeffs  # (N,)

    # known poly at those points
    pot_poly, acc_poly = eval_poly_gravity(
        vertices, faces, density, points, parallel=parallel
    )
    cyl_acc_poly = cart_to_cyl_acc(spec, points, acc_poly)

    # residual that should be explained by mascon
    cyl_acc_resid = cyl_acc_target - cyl_acc_poly
    pot_resid = (pot_target - pot_poly) if use_potential else None

    def fun(p):
        r_m = p[:3]
        mu = p[3]

        pot_m, acc_m = mascon_potential_acc(points, r_m, mu, softening=softening)
        cyl_acc_m = cart_to_cyl_acc(spec, points, acc_m)

        r_acc = (cyl_acc_m - cyl_acc_resid).reshape(-1, order="C") * w_acc
        if use_potential:
            r_pot = (pot_m - pot_resid) * w_pot
            return np.hstack([r_acc, r_pot])
        return r_acc

    res = least_squares(
        fun,
        x0=x0,
        bounds=bounds,
        method="trf",
        jac="2-point",  # or "3-point" for slightly more accurate (slower)
        ftol=1e-14,  # tighter than default (1e-8)
        xtol=1e-14,  # tighter step tolerance
        gtol=1e-14,  # tighter optimality (gradient) tolerance
        max_nfev=20000,  # allow more work before giving up
        x_scale="jac",  # automatic parameter scaling (important with [pos, mu] mixed units)
        loss="linear",  # keep least-squares as-is; try "soft_l1" if you get outlier issues
        f_scale=1.0,  # only used for robust losses; keep 1.0 for linear
        diff_step=1e-6,  # finite-difference step; tune if your LU scaling is weird
        tr_solver="exact",  # more accurate than default for moderate parameter count
        verbose=2,
    )
    return res


# ======================================================================================
# Example main
# ======================================================================================

if __name__ == "__main__":
    # ---------------------------
    # Mesh
    # ---------------------------
    vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    DENSITY = 1.0

    # ---------------------------
    # Cylinder
    # ---------------------------
    CYLINDER_CENTER = np.array([0.0, 0.0, 0.28])
    CYLINDER_RADIUS = 0.10
    CYLINDER_HEIGHT = 0.50
    CYLINDER_ROTATION = np.eye(3)
    ALPHA = 100.0

    spec = CylinderSpec(
        center=CYLINDER_CENTER,
        radius=CYLINDER_RADIUS,
        height=CYLINDER_HEIGHT,
        rotation=CYLINDER_ROTATION,
        alpha=ALPHA,
    )

    # ---------------------------
    # Sample points in cylinder
    # ---------------------------
    NUM_POINTS = 1200
    seed_points = 1
    points = generate_points_in_cylinder(spec, NUM_POINTS, seed=seed_points)

    # ---------------------------
    # (1) FORWARD: poly + big mascon
    # ---------------------------
    n_n, n_m = 25, 25

    # Choose a “big” interior mascon (position must be inside your body)
    mascon_r_true = np.array([0.4, -0.01, 0.05])
    mascon_mu_true = 5e-8  # you tune this to get a visible signature in your units

    coeffs, (pot_poly, acc_poly), (pot_tot, acc_tot) = (
        fit_cylinder_for_poly_plus_mascon(
            vertices,
            faces,
            DENSITY,
            spec,
            points,
            n_n=n_n,
            n_m=n_m,
            mascon_r=mascon_r_true,
            mascon_mu=mascon_mu_true,
            parallel=False,
            enforce_B0n=True,
        )
    )

    print("\nFitted cylinder coeffs computed (total field = poly + mascon).")

    # ---------------------------
    # (2) INVERSE: estimate back mascon from coeffs
    # ---------------------------
    # initial guess
    x0 = np.array([0.0, 0.0, 0.0, 0])

    # bounds (keep mu >= 0, and keep position near body center-ish; adjust to your LU)
    lb = np.array([-1.0, -1.0, -1.0, 0.0])
    ub = np.array([+1.0, +1.0, +1.0, 1.0])

    res = estimate_mascon_from_coeffs(
        coeffs=coeffs,
        vertices=vertices,
        faces=faces,
        density=DENSITY,
        spec=spec,
        points=points,
        n_n=n_n,
        n_m=n_m,
        x0=x0,
        bounds=(lb, ub),
        parallel=False,
        softening=0.0,
        use_potential=True,  # include potential too (helps with conditioning)
        w_acc=1.0,
        w_pot=1.0,
    )

    est = res.x
    print("\n=== TRUE mascon ===")
    print("r_true  =", mascon_r_true)
    print("mu_true =", mascon_mu_true)

    print("\n=== ESTIMATED mascon ===")
    print("r_hat   =", est[:3])
    print("mu_hat  =", est[3])
