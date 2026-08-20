"""
Interior-potential fitting for a polyhedral body using the closed-form solution of
Poisson's equation with polynomial radial density modes.

Convention (matches the LaTeX section):

    laplacian(phi) = -4 pi G rho          phi > 0 outside a positive mass

    rho(r,th,ph) = sum_{l,m,n} rho_lmn r^n Ybar_lm(th,ph)
    phi(r,th,ph) = 4 pi G sum_{l,m,n} rho_lmn B_ln(r) Ybar_lm(th,ph)

with the collapsed radial basis

    B_ln(r) = R^(n-l+2) r^l / [(2l+1)(n-l+2)]  -  r^(n+2) / [(n-l+2)(n+l+3)]     (n != l-2)
    B_ln(r) = r^l/(2l+1) * [ ln(R/r) + 1/(2l+1) ]                                (n == l-2)
    B_ln(r) = R^(n+l+3) r^-(l+1) / [(2l+1)(n+l+3)]                               (r >= R)

Ybar_lm are orthonormal real spherical harmonics built from 4pi-normalized (geodesy)
associated Legendre functions:  Ybar = Pbar_lm(cos th) * {cos,sin}(m ph) / sqrt(4 pi).

Numerically the basis is evaluated in dimensionless form,  B_ln = R^(n+2) * bhat_ln(r/R),
and the fitted coefficient is  c_lmn = 4 pi G rho_lmn R^(n+2),  so that

    rho_lmn = c_lmn / (4 pi G R^(n+2)).
"""

import os

import numpy as np
import trimesh
from scipy.linalg import qr, solve_triangular
from scipy.stats import norm
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
from trimesh.intersections import mesh_plane

from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable

import mesh_utility

# =============================================================================
# 0.  Configuration
# =============================================================================

MESH_PATH = "3dmeshes/eros.pk"
DENSITY = 1.0  # kg/m^3, uniform truth density of the polyhedron
G_NEWTON = 6.67430e-11  # m^3 kg^-1 s^-2
FOUR_PI_G = 4.0 * np.pi * G_NEWTON

R_SCALE = 2.0  # support radius R = R_SCALE * bounding_radius
L_MAX = 10  # max spherical-harmonic degree
N_MAX = 10  # max radial polynomial order

# Restrict to modes with n >= l and (n - l) even.  These are exactly the modes of a
# density that is smooth at the origin; they also eliminate the n = l - 2 logarithmic
# case entirely.  Set False to use the full monomial set (and exercise the log branch).
REGULARITY_FILTER = True

NUM_POINTS = 10_000  # interior sample points for the fit
RIDGE = 0.0  # Tikhonov parameter on the column-normalized system
DATASET_PATH = "spherical_gravity_dataset.npz"
RNG_SEED = 0


# =============================================================================
# 1.  Mesh, bounding sphere, and the forward polyhedral model
# =============================================================================

vertices, faces = mesh_utility.read_pk_file(MESH_PATH)
vertices = np.asarray(vertices, dtype=float)
faces = np.asarray(faces, dtype=int)

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

center = vertices.mean(axis=0)
bounding_radius = np.max(np.linalg.norm(vertices - center, axis=1))
R_REF = R_SCALE * bounding_radius

print(f"bounding radius = {bounding_radius:.3f} m")
print(f"support radius R = {R_REF:.3f} m")
print(f"mesh volume      = {mesh.volume:.6e} m^3   (watertight={mesh.is_watertight})")

polyhedron = Polyhedron(
    polyhedral_source=(vertices, faces),
    density=DENSITY,
    integrity_check=PolyhedronIntegrity.DISABLE,
)
gravity_model = GravityEvaluable(polyhedron)


def eval_polyhedral_potential(pts, desc="polyhedral potential"):
    """Evaluate the polyhedral potential at (N,3) points, batched if supported."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    try:
        out = gravity_model(computation_points=pts, parallel=True)
        # Batch call returns either a list of (pot, acc, tensor) or a tuple of arrays.
        if isinstance(out, (list, tuple)) and len(out) == len(pts) and len(pts) != 3:
            return np.array([o[0] for o in out], dtype=float)
        return np.asarray(out[0], dtype=float).reshape(-1)
    except Exception:
        vals = np.empty(len(pts))
        for i, p in enumerate(tqdm(pts, desc=desc)):
            vals[i] = gravity_model(computation_points=p, parallel=False)[0]
        return vals


def calibrate_potential_scale():
    """
    Determine the multiplicative factor mapping the library's potential onto the
    convention  phi = G * int rho / |r - r'| dV  > 0.

    Far from the body, phi -> G M / r.  Comparing the library output at a far point
    to G M / r detects both a missing gravitational constant and a sign flip.
    """
    r_far = 50.0 * bounding_radius
    probes = center + r_far * np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    u_lib = eval_polyhedral_potential(probes, desc="calibration")
    mass = DENSITY * mesh.volume
    phi_expected = G_NEWTON * mass / r_far
    scale = phi_expected / np.mean(u_lib)
    print(
        f"potential calibration: mean(U_lib)={np.mean(u_lib):.6e}, "
        f"GM/r={phi_expected:.6e}, scale={scale:.6e}"
    )
    if not np.isfinite(scale) or scale == 0.0:
        raise RuntimeError("Potential calibration failed; check mesh/units.")
    return scale


POTENTIAL_SCALE = calibrate_potential_scale()


# =============================================================================
# 2.  Sampling
# =============================================================================


def generate_points_in_sphere(
    center, radius, num_points, r_min_frac=1e-3, seed=RNG_SEED
):
    """Uniform points in a ball, with the origin excluded (r=0 is a coordinate
    singularity and, for n = l - 2, a genuine logarithmic singularity of B_ln)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform((r_min_frac) ** 3, 1.0, num_points)
    costheta = rng.uniform(-1.0, 1.0, num_points)
    phi = rng.uniform(0.0, 2.0 * np.pi, num_points)

    theta = np.arccos(costheta)
    r = radius * np.cbrt(u)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack((x, y, z), axis=-1) + center


def generate_points_outside_mesh(center, radius, num_points, mesh, seed=RNG_SEED):
    """Points in the ball that lie outside the polyhedron (Laplace region)."""
    rng = np.random.default_rng(seed)
    accepted = []
    while len(accepted) < num_points:
        batch = generate_points_in_sphere(
            center, radius, 2 * num_points, seed=rng.integers(1 << 31)
        )
        accepted.extend(batch[~mesh.contains(batch)].tolist())
    return np.array(accepted[:num_points])


if os.path.exists(DATASET_PATH):
    data = np.load(DATASET_PATH)
    points = data["points"]
    potentials = data["potential"]
    print(f"loaded {len(points)} samples from {DATASET_PATH}")
else:
    points = generate_points_in_sphere(center, bounding_radius, NUM_POINTS)
    potentials = POTENTIAL_SCALE * eval_polyhedral_potential(points)
    np.savez(
        DATASET_PATH,
        points=points,
        potential=potentials,
        center=center,
        bounding_radius=bounding_radius,
        R_ref=R_REF,
    )
    print(f"saved {len(points)} samples to {DATASET_PATH}")


# =============================================================================
# 3.  Angular basis: orthonormal real spherical harmonics
# =============================================================================


def normalized_legendre(l_max, cos_theta, sin_theta):
    """
    4pi-normalized (geodesy) associated Legendre functions Pbar_lm, computed by the
    standard stable recursion.  Returns {(l, m): array}.

    Pbar_lm = sqrt((2 - delta_m0)(2l+1)(l-m)!/(l+m)!) * P_lm,  P_lm without the
    Condon-Shortley phase.  Unlike scipy.special.lpmv these do not overflow with l.
    """
    P = {(0, 0): np.ones_like(cos_theta)}
    if l_max >= 1:
        P[(1, 1)] = np.sqrt(3.0) * sin_theta
        P[(1, 0)] = np.sqrt(3.0) * cos_theta
    for l in range(2, l_max + 1):
        P[(l, l)] = np.sqrt((2 * l + 1) / (2 * l)) * sin_theta * P[(l - 1, l - 1)]
        for m in range(l):
            if m == l - 1:
                P[(l, m)] = np.sqrt(2 * l + 1) * cos_theta * P[(l - 1, l - 1)]
            else:
                a = np.sqrt((2 * l - 1) * (2 * l + 1) / ((l - m) * (l + m)))
                b = np.sqrt(
                    (2 * l + 1)
                    * (l + m - 1)
                    * (l - m - 1)
                    / ((l - m) * (l + m) * (2 * l - 3))
                )
                P[(l, m)] = a * cos_theta * P[(l - 1, m)] - b * P[(l - 2, m)]
    return P


def real_harmonics(l_max, theta, phi):
    """Orthonormal real spherical harmonics: {(l, m, 'c'|'s'): array}."""
    ct, st = np.cos(theta), np.sin(theta)
    P = normalized_legendre(l_max, ct, st)
    norm4pi = 1.0 / np.sqrt(4.0 * np.pi)
    Y = {}
    for l in range(l_max + 1):
        for m in range(l + 1):
            Y[(l, m, "c")] = norm4pi * P[(l, m)] * np.cos(m * phi)
            if m > 0:
                Y[(l, m, "s")] = norm4pi * P[(l, m)] * np.sin(m * phi)
    return Y


# =============================================================================
# 4.  Radial basis: the closed-form B_ln, in dimensionless form
# =============================================================================


def bhat(x, l, n):
    """
    Dimensionless radial basis, B_ln(r) = R^(n+2) * bhat(r/R, l, n).

    Interior (x <= 1), n != l-2:   x^l/[(2l+1)(n-l+2)] - x^(n+2)/[(n-l+2)(n+l+3)]
    Interior (x <= 1), n == l-2:   x^l/(2l+1) * [ -ln x + 1/(2l+1) ]
    Exterior (x >= 1):             x^-(l+1) / [(2l+1)(n+l+3)]

    The two branches agree in value and first derivative at x = 1.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    e = n - l + 2
    inner = x <= 1.0
    outer = ~inner

    xi = x[inner]
    if e != 0:
        out[inner] = xi**l / ((2 * l + 1) * e) - xi ** (n + 2) / (e * (n + l + 3))
    else:
        out[inner] = xi**l / (2 * l + 1) * (-np.log(xi) + 1.0 / (2 * l + 1))

    xo = x[outer]
    out[outer] = xo ** (-(l + 1)) / ((2 * l + 1) * (n + l + 3))
    return out


def bhat_prime(x, l, n):
    """d/dx of bhat, used only by the continuity self-test."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    e = n - l + 2
    inner, outer = x <= 1.0, x > 1.0

    xi = x[inner]
    if e != 0:
        out[inner] = l * xi ** (l - 1) / ((2 * l + 1) * e) - (n + 2) * xi ** (n + 1) / (
            e * (n + l + 3)
        )
    else:
        out[inner] = l * xi ** (l - 1) / (2 * l + 1) * (
            -np.log(xi) + 1.0 / (2 * l + 1)
        ) - xi ** (l - 1) / (2 * l + 1)

    xo = x[outer]
    out[outer] = -(l + 1) * xo ** (-(l + 2)) / ((2 * l + 1) * (n + l + 3))
    return out


def mode_list(l_max, n_max, regularity=REGULARITY_FILTER):
    """Enumerate (l, m, trig, n) modes and their string labels."""
    modes, labels = [], []
    for l in range(l_max + 1):
        for m in range(l + 1):
            trigs = ["c"] if m == 0 else ["c", "s"]
            for n in range(n_max + 1):
                if regularity and (n < l or (n - l) % 2 != 0):
                    continue
                for t in trigs:
                    modes.append((l, m, t, n))
                    labels.append(f"{'a' if t == 'c' else 'b'}_{l}_{m}_{n}")
    return modes, labels


MODES, LABELS = mode_list(L_MAX, N_MAX)
print(
    f"model: l_max={L_MAX}, n_max={N_MAX}, regularity={REGULARITY_FILTER}, "
    f"{len(MODES)} coefficients"
)


def to_spherical(points, center):
    v = np.asarray(points, dtype=float) - center
    r = np.linalg.norm(v, axis=1)
    r_safe = np.maximum(r, 1e-12)
    theta = np.arccos(np.clip(v[:, 2] / r_safe, -1.0, 1.0))
    phi = np.arctan2(v[:, 1], v[:, 0])
    return r, theta, phi


def design_matrix(points, center, R, modes, l_max):
    """
    Columns of the potential model.  Column (l,m,t,n) equals bhat_ln(r/R) * Ybar_lmt,
    so the fitted coefficient is c_lmn = 4 pi G rho_lmn R^(n+2).
    """
    r, theta, phi = to_spherical(points, center)
    x = r / R
    Y = real_harmonics(l_max, theta, phi)

    bcache = {}
    cols = []
    for l, m, t, n in modes:
        if (l, n) not in bcache:
            bcache[(l, n)] = bhat(x, l, n)
        cols.append(bcache[(l, n)] * Y[(l, m, t)])
    return np.vstack(cols).T


def evaluate_potential(points, center, R, coeffs, modes, l_max):
    """Model potential; valid inside and outside R via the two branches of bhat."""
    return design_matrix(points, center, R, modes, l_max) @ coeffs


def evaluate_density(points, center, R, coeffs, modes, l_max):
    """
    rho = sum rho_lmn r^n Ybar,  with rho_lmn = c_lmn / (4 pi G R^(n+2)).
    Substituting gives rho = (1/(4 pi G R^2)) sum c_lmn x^n Ybar.  Zero outside R.
    """
    r, theta, phi = to_spherical(points, center)
    x = r / R
    Y = real_harmonics(l_max, theta, phi)

    rho = np.zeros_like(r)
    for c, (l, m, t, n) in zip(coeffs, modes):
        rho += c * x**n * Y[(l, m, t)]
    rho /= FOUR_PI_G * R**2
    rho[x > 1.0] = 0.0
    return rho


def model_mass(coeffs, R, modes):
    """M = int rho dV.  Only (l=0, m=0) modes survive the angular integral."""
    total = 0.0
    for c, (l, m, t, n) in zip(coeffs, modes):
        if l == 0 and m == 0:
            rho_lmn = c / (FOUR_PI_G * R ** (n + 2))
            total += np.sqrt(4.0 * np.pi) * rho_lmn * R ** (n + 3) / (n + 3)
    return total


# =============================================================================
# 5.  Self-tests on the closed form (cheap regression checks)
# =============================================================================


def selftest_continuity(l_max, n_max, tol=1e-10):
    """B and B' must match across r = R for every (l, n), including the log case."""
    worst = 0.0
    for l in range(l_max + 1):
        for n in range(n_max + 1):
            xm = np.array([1.0 - 1e-7])
            xp = np.array([1.0 + 1e-7])
            expected_val = 1.0 / ((2 * l + 1) * (n + l + 3))
            expected_der = -(l + 1) / ((2 * l + 1) * (n + l + 3))
            worst = max(
                worst,
                abs(bhat(xm, l, n)[0] - expected_val),
                abs(bhat(xp, l, n)[0] - expected_val),
                abs(bhat_prime(xm, l, n)[0] - expected_der),
                abs(bhat_prime(xp, l, n)[0] - expected_der),
            )
    print(f"[selftest] C1 matching at r=R: max deviation {worst:.3e}")
    assert worst < 1e-5, "B_ln is not C1 across the support boundary"


def selftest_poisson(l_max, n_max, R=1.0, h=2e-3, seed=1):
    """
    Finite-difference check that the model satisfies laplacian(phi) = -4 pi G rho
    for random coefficients, at random interior points.
    """
    rng = np.random.default_rng(seed)
    modes, _ = mode_list(l_max, n_max)
    coeffs = rng.normal(size=len(modes))

    c0 = np.zeros(3)
    u = rng.normal(size=(200, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rad = R * rng.uniform(0.25, 0.75, size=(200, 1))
    pts = u * rad

    def phi(p):
        return evaluate_potential(p, c0, R, coeffs, modes, l_max)

    lap = -6.0 * phi(pts)
    for k in range(3):
        step = np.zeros(3)
        step[k] = h * R
        lap = lap + phi(pts + step) + phi(pts - step)
    lap /= (h * R) ** 2

    rhs = -FOUR_PI_G * evaluate_density(pts, c0, R, coeffs, modes, l_max)
    rel = np.linalg.norm(lap - rhs) / np.linalg.norm(rhs)
    print(f"[selftest] Poisson residual (relative, FD): {rel:.3e}")
    assert rel < 5e-3, "Model does not satisfy Poisson's equation"


def selftest_uniform_sphere(R=1.0, rho0=1.0):
    """phi(r) = 2 pi G rho0 (R^2 - r^2/3) for a uniform ball."""
    modes = [(0, 0, "c", 0)]
    rho_000 = np.sqrt(4.0 * np.pi) * rho0
    coeffs = np.array([FOUR_PI_G * rho_000 * R**2])
    r = np.linspace(0.05, 1.0, 9) * R
    pts = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=1)
    got = evaluate_potential(pts, np.zeros(3), R, coeffs, modes, 0)
    want = 2.0 * np.pi * G_NEWTON * rho0 * (R**2 - r**2 / 3.0)
    err = np.max(np.abs(got - want) / np.abs(want))
    print(f"[selftest] uniform sphere: max relative error {err:.3e}")
    assert err < 1e-12


selftest_continuity(L_MAX, N_MAX)
selftest_poisson(L_MAX, N_MAX)
selftest_uniform_sphere()


# =============================================================================
# 6.  Least-squares fit (column-normalized, QR)
# =============================================================================

A = design_matrix(points, center, R_REF, MODES, L_MAX)
b = potentials

col_scale = np.linalg.norm(A, axis=0)
col_scale[col_scale == 0.0] = 1.0
A_s = A / col_scale

if RIDGE > 0.0:
    A_aug = np.vstack([A_s, np.sqrt(RIDGE) * np.eye(A_s.shape[1])])
    b_aug = np.concatenate([b, np.zeros(A_s.shape[1])])
else:
    A_aug, b_aug = A_s, b

Qf, Rf = qr(A_aug, mode="economic")
params_s = solve_triangular(Rf, Qf.T @ b_aug)
fitted_params = params_s / col_scale

fitted_potentials = A @ fitted_params
residuals = b - fitted_potentials

dof = max(len(b) - len(fitted_params), 1)
sigma_squared = np.sum(residuals**2) / dof
Rinv = solve_triangular(Rf, np.eye(Rf.shape[0]))
cov_s = sigma_squared * (Rinv @ Rinv.T)
cov_matrix = cov_s / np.outer(col_scale, col_scale)

print(f"cond(A_scaled)   = {np.linalg.cond(A_s):.3e}   (raw: {np.linalg.cond(A):.3e})")
print(f"RMS residual     = {np.sqrt(np.mean(residuals**2)):.6e}")
print(
    f"RMS relative     = {np.sqrt(np.mean(residuals**2)) / np.sqrt(np.mean(b**2)):.6e}"
)

M_true = DENSITY * mesh.volume
M_fit = model_mass(fitted_params, R_REF, MODES)
print(
    f"mass: fitted={M_fit:.6e} kg, polyhedron={M_true:.6e} kg, "
    f"ratio={M_fit / M_true:.4f}"
)


# =============================================================================
# 7.  Residual distribution
# =============================================================================

percentage_error = 100.0 * residuals / np.sqrt(np.mean(b**2))

plt.figure(figsize=(10, 6))
plt.hist(
    percentage_error, bins=50, color="#2c7bb6", alpha=0.7, edgecolor="k", density=True
)
mu, std = norm.fit(percentage_error)
xg = np.linspace(percentage_error.min(), percentage_error.max(), 1000)
plt.plot(
    xg,
    norm.pdf(xg, mu, std),
    color="#d7191c",
    linestyle="--",
    linewidth=2,
    label=rf"$\mu={mu:.4f},\ \sigma={std:.4f}$",
)
plt.xlabel("Residual (% of RMS potential)")
plt.ylabel("Density")
plt.title("Potential fit residuals, interior samples")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 8.  Error on the body surface
# =============================================================================


def shift_vertices_outward(vertices, center, eps_frac=1e-6):
    d = vertices - center
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return vertices + eps_frac * bounding_radius * d / np.maximum(n, 1e-12)


shifted_vertices = shift_vertices_outward(vertices, center)

true_surface = POTENTIAL_SCALE * eval_polyhedral_potential(
    shifted_vertices, desc="true potential at surface"
)
fitted_surface = evaluate_potential(
    shifted_vertices, center, R_REF, fitted_params, MODES, L_MAX
)
vertex_errors = 100.0 * (fitted_surface - true_surface) / np.abs(true_surface)
face_errors = np.mean(vertex_errors[faces], axis=1)

vmax = np.max(np.abs(face_errors))
cnorm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
face_colors = plt.cm.seismic(cnorm(face_errors))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
ax.add_collection3d(
    Poly3DCollection(
        vertices[faces],
        facecolors=face_colors,
        edgecolor="k",
        linewidths=0.2,
        alpha=1.0,
    )
)
span = (vertices.max(axis=0) - vertices.min(axis=0)).max()
mid = vertices.mean(axis=0)
ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
ax.set_xlabel("X (m)"), ax.set_ylabel("Y (m)"), ax.set_zlabel("Z (m)")
ax.set_title("Surface potential error")
mappable = plt.cm.ScalarMappable(cmap="seismic", norm=cnorm)
mappable.set_array(face_errors)
plt.colorbar(mappable, ax=ax, shrink=0.6).set_label("Percentage error (%)")
plt.tight_layout()
plt.show()


# =============================================================================
# 9.  Coefficient power spectrum with uncertainty
# =============================================================================


def parse_labels(labels):
    out = []
    for lab in labels:
        typ, l, m, n = lab.split("_")
        out.append((int(l), int(m), int(n), typ))
    return out


def plot_power_spectrum(fitted_params, labels, covariance):
    parsed = parse_labels(labels)
    max_ln = max(l + n for l, _, n, _ in parsed) + 1

    power = np.zeros(max_ln)
    var_power = np.zeros(max_ln)
    sigma_power = np.zeros(max_ln)

    for idx, (l, m, n, typ) in enumerate(parsed):
        k = l + n
        a = fitted_params[idx]
        s = np.sqrt(max(covariance[idx, idx], 0.0))
        power[k] += a**2
        var_power[k] += (2.0 * a * s) ** 2
        sigma_power[k] += s**2

    err = np.sqrt(var_power)
    ks = np.arange(max_ln)

    plt.figure(figsize=(10, 6))
    plt.semilogy(
        ks, np.maximum(power, 1e-300), marker="o", color="#2c7bb6", label="Power"
    )
    plt.fill_between(
        ks,
        np.maximum(power - err, 1e-300),
        power + err,
        color="#2c7bb6",
        alpha=0.3,
        label=r"$\pm1\sigma$ (propagated)",
    )
    plt.semilogy(
        ks,
        np.maximum(sigma_power, 1e-300),
        marker="x",
        linestyle="--",
        color="#fdae61",
        label=r"$\sigma$-only power",
    )
    plt.xlabel(r"Total order $l+n$")
    plt.ylabel("Power / uncertainty")
    plt.title("Radial-harmonic power spectrum with uncertainty bands")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_power_spectrum(fitted_params, LABELS, cov_matrix)


# =============================================================================
# 10.  Surface potential maps
# =============================================================================

rvec = shifted_vertices - center
r_s = np.linalg.norm(rvec, axis=1)
theta_s = np.arccos(np.clip(rvec[:, 2] / r_s, -1.0, 1.0))
phi_s = np.arctan2(rvec[:, 1], rvec[:, 0])
phi_s[phi_s < 0] += 2.0 * np.pi

lat = 90.0 - np.degrees(theta_s)
lon = np.degrees(phi_s)

N_LAT, N_LON = 180, 360
grid_lon, grid_lat = np.meshgrid(
    np.linspace(0, 360, N_LON), np.linspace(-90, 90, N_LAT)
)
grid_vals = griddata(
    (lon, lat), fitted_surface, (grid_lon, grid_lat), method="linear", fill_value=np.nan
)

plt.figure(figsize=(12, 6))
plt.imshow(
    grid_vals, extent=(0, 360, -90, 90), origin="lower", cmap="viridis", aspect="auto"
)
plt.xlabel("Longitude (deg)")
plt.ylabel("Latitude (deg)")
plt.title("Fitted surface potential")
plt.colorbar().set_label(r"Potential (m$^2$/s$^2$)")
plt.tight_layout()
plt.show()

try:
    import cartopy.crs as ccrs

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Mollweide())
    cf = ax.contourf(
        grid_lon,
        grid_lat,
        grid_vals,
        levels=60,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
    )
    ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_global()
    plt.colorbar(cf, orientation="horizontal", pad=0.05).set_label(
        r"Potential (m$^2$/s$^2$)"
    )
    plt.title("Fitted surface potential (Mollweide)")
    plt.tight_layout()
    plt.show()
except ImportError:
    print("cartopy not available; skipping Mollweide projection")


# =============================================================================
# 11.  Recovered density
# =============================================================================


def generate_grid_points(center, radius, num_per_axis, mesh):
    ax_ = np.linspace(-radius, radius, num_per_axis)
    X, Y, Z = np.meshgrid(ax_, ax_, ax_, indexing="ij")
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T + center
    pts = pts[np.linalg.norm(pts - center, axis=1) <= radius]
    return pts[mesh.contains(pts)]


NUM_POINTS_PER_AXIS = 50
grid_points = generate_grid_points(center, bounding_radius, NUM_POINTS_PER_AXIS, mesh)

density_values = evaluate_density(
    grid_points, center, R_REF, fitted_params, MODES, L_MAX
)

neg_frac = np.mean(density_values < 0.0)
print(
    f"recovered density: mean={density_values.mean():.4f}, "
    f"min={density_values.min():.4f}, max={density_values.max():.4f}, "
    f"negative fraction={neg_frac:.3f}  (truth = {DENSITY})"
)
if neg_frac > 0.01:
    print(
        "  note: negative density indicates the fit is under-constrained; "
        "consider RIDGE > 0, a lower N_MAX, or a positivity-constrained solve."
    )


def plot_density_slices(points, values, center, radius, mesh, num_per_axis=60):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x0, y0, z0 = center
    tol = radius / (num_per_axis / 2.0)
    res = max(num_per_axis // 2, 20)

    cmin, cmax = np.percentile(values, [1, 99])
    levels = np.linspace(cmin, cmax, 21)

    specs = [
        (0, (0, 1), [0, 0, 1], ("X", "Y"), (x0, y0), 2),
        (1, (0, 2), [0, 1, 0], ("X", "Z"), (x0, z0), 1),
        (2, (1, 2), [1, 0, 0], ("Y", "Z"), (y0, z0), 0),
    ]

    for ax_i, (i, j), normal, (lab_i, lab_j), (c_i, c_j), slice_axis in specs:
        ax = axes[ax_i]
        mask = np.abs(points[:, slice_axis] - center[slice_axis]) < tol
        if mask.sum() > 3:
            gi = np.linspace(c_i - radius, c_i + radius, res)
            gj = np.linspace(c_j - radius, c_j + radius, res)
            GI, GJ = np.meshgrid(gi, gj)
            grid = griddata(
                points[mask][:, [i, j]],
                values[mask],
                (GI, GJ),
                method="linear",
                fill_value=np.nan,
            )
            cf = ax.contourf(GI, GJ, grid, levels=levels, cmap="viridis", extend="both")
            plt.colorbar(cf, ax=ax, label=r"Density (kg/m$^3$)")
            for line in mesh_plane(mesh, np.array(normal), np.array(center)):
                ax.plot(line[:, i], line[:, j], "k-", linewidth=1, alpha=0.6)
        ax.set_xlabel(f"{lab_i} (m)")
        ax.set_ylabel(f"{lab_j} (m)")
        ax.set_title(f"{lab_i}{lab_j}-plane slice")
        ax.set_xlim(c_i - radius, c_i + radius)
        ax.set_ylim(c_j - radius, c_j + radius)
        ax.set_aspect("equal")

    plt.suptitle("Recovered density, slices through the body centre")
    plt.tight_layout()
    plt.show()


plot_density_slices(
    grid_points, density_values, center, bounding_radius, mesh, NUM_POINTS_PER_AXIS
)
