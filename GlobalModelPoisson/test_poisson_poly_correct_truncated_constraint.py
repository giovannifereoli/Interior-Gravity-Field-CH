import numpy as np
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
import mesh_utility
from tqdm import tqdm
import trimesh
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import null_space
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.special import lpmv, gammaln
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy
from scipy.interpolate import griddata
from trimesh.intersections import mesh_plane

# Meshes from https://github.com/darioizzo/geodesyNets/tree/master/3dmeshes
vertices, faces = mesh_utility.read_pk_file("3dmeshes/bennu.pk")
# vertices, faces = mesh_utility.read_pk_file("3dmeshes/bennu_lp.pk")
# vertices, faces = np.array(vertices), np.array(faces)

# Load the mesh for inside/outside checks
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# mesh = trimesh.load("3dmeshes/phobos_lowlowres.obj")
# vertices = mesh.vertices
# faces = mesh.faces

# Compute bounding sphere
center = np.mean(vertices, axis=0)
radii = np.linalg.norm(vertices - center, axis=1)
bounding_radius = 1 * np.max(radii)
R_REF = 2 * bounding_radius  # radius of bounding sphere

# Define asteroid density
DENSITY = 1.0  # arbitrary units
G = 6.67430 * 1e-11  # m^3 kg^-1 s^-2, gravitational constant

# Initialize polyhedron model
polyhedron = Polyhedron(
    polyhedral_source=(vertices, faces),
    density=DENSITY,
    integrity_check=PolyhedronIntegrity.DISABLE,
)

# Evaluable wrapper for gravity potential
gravity_model = GravityEvaluable(polyhedron)


# Generate random points inside the bounding sphere
def generate_grid_points(center, radius, num_points_per_axis):
    """
    Generate a 3D grid of points within the bounding sphere, filtered to include
    only points inside the mesh.

    Args:
        center (ndarray): (3,) center of the body.
        radius (float): Bounding sphere radius.
        num_points_per_axis (int): Number of points per axis in the grid.

    Returns:
        points (ndarray): (N, 3) array of points inside the mesh.
    """
    # Create a 3D grid
    x = np.linspace(-radius, radius, num_points_per_axis)
    y = np.linspace(-radius, radius, num_points_per_axis)
    z = np.linspace(-radius, radius, num_points_per_axis)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Filter points within the bounding sphere
    distances = np.linalg.norm(points - center, axis=1)
    mask_sphere = distances <= radius
    points = points[mask_sphere]

    # Filter points inside the mesh
    inside = mesh.contains(points)
    points_inside = points[inside]

    return points_inside


def generate_points_in_sphere(center, radius, num_points):
    np.random.seed(0)
    u = np.random.uniform(0, 1, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)

    theta = np.arccos(costheta)
    r = radius * np.cbrt(u)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    points = np.stack((x, y, z), axis=-1) + center
    return points


NUM_POINTS = 10000
sample_points = generate_points_in_sphere(center, bounding_radius, NUM_POINTS)

# Evaluate gravity potentials at each point
results = []
for pt in tqdm(sample_points, desc="Evaluating gravity potentials"):
    potential, acceleration, tensor = gravity_model(
        computation_points=pt, parallel=False
    )
    results.append(
        {
            "point": pt,
            "potential": potential,
            "acceleration": acceleration,
            "tensor": tensor,
        }
    )

# Save dataset
dataset = {
    "points": np.array([r["point"] for r in results]),
    "potential": np.array([r["potential"] for r in results]),
    "acceleration": np.array([r["acceleration"] for r in results]),
    "tensor": np.array([r["tensor"] for r in results]),
    "center": center,
    "bounding_radius": bounding_radius,
}

np.savez("spherical_gravity_dataset.npz", **dataset)


# Load the spherical dataset
data = np.load("spherical_gravity_dataset.npz")
points = data["points"]
potentials = data["potential"]
center = data["center"]
bounding_radius = data["bounding_radius"]


# Define real spherical harmonics radial basis model


def sh_norm(l, m):
    # N_{l0} and N_{lm} (m>0) with Condon–Shortley phase in lpmv
    base = (2 * l + 1) / (4 * np.pi)
    # factorial ratio via gammaln for stability
    fr = np.exp(gammaln(l - m + 1) - gammaln(l + m + 1))
    if m == 0:
        return np.sqrt(base)
    return np.sqrt(2.0 * base * fr)


def prepare_spherical_poly_basis(points, center, l_max, n_max):
    """
    Build a physically-consistent interior potential basis for Poisson's equation
    with finite support at radius R.

    Args:
        points (ndarray): (N,3) evaluation points.
        center (ndarray): (3,) center of the body.
        R (float): support radius (max radius of density).
        l_max (int): maximum spherical harmonic degree.
        n_max (int): maximum radial polynomial order.

    Returns:
        A (ndarray): design matrix shape (N, num_coeffs).
        coeff_labels (list): list of labels for each column in A.
    """
    # Shift to body-centered coords and spherical angles
    v = points - center
    r = np.maximum(np.finfo(float).tiny, np.linalg.norm(v, axis=1))
    theta = np.arccos(np.clip(v[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(v[:, 1], v[:, 0])
    R = R_REF

    columns = []
    coeff_labels = []

    for l in range(l_max + 1):
        # Precompute Legendre polynomials for this l
        Plm_vals = [lpmv(m, l, np.cos(theta)) for m in range(l + 1)]

        for m in range(l + 1):
            Plm = Plm_vals[m]
            cos_mphi = np.cos(m * phi)
            sin_mphi = np.sin(m * phi) if m > 0 else None
            Nlm = sh_norm(l, m)

            for n in range(n_max + 1):
                exp = n - l + 2
                # Radial integral from r to R
                if exp == 0:
                    radial = r**l * np.log(R / r) / (2 * l + 1)
                else:
                    radial = r**l * (R**exp - r**exp) / (exp * (2 * l + 1))
                radial += r ** (n + l + 3) / ((l + n + 3) * (2 * l + 1)) / r ** (l + 1)

                # Cosine term
                col_c = Nlm * radial * Plm * cos_mphi
                columns.append(col_c)
                coeff_labels.append(f"a_{l}_{m}_{n}")

                # Sine term if m > 0
                if m > 0:
                    col_s = Nlm * radial * Plm * sin_mphi
                    columns.append(col_s)
                    coeff_labels.append(f"b_{l}_{m}_{n}")

    # Stack columns to form design matrix
    A = np.vstack(columns).T
    return A, coeff_labels


# Set model complexity
l_max, n_max = 6, 6


def prepare_density_constraint_matrix(points, center, l_max, n_max):
    """
    Build the constraint matrix C for enforcing rho >= 0 at given points.

    Args:
        points (ndarray): (N, 3) array of constraint points.
        center (ndarray): (3,) center of the body.
        l_max (int): Maximum spherical harmonic degree.
        n_max (int): Maximum radial polynomial order.

    Returns:
        C (ndarray): Constraint matrix of shape (N, num_coeffs).
        coeff_labels (list): Labels for each column in C (e.g., 'a_l_m_n', 'b_l_m_n').
    """
    v = points - center
    r = np.maximum(np.finfo(float).tiny, np.linalg.norm(v, axis=1))
    theta = np.arccos(np.clip(v[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(v[:, 1], v[:, 0])

    columns = []
    coeff_labels = []

    for l in range(l_max + 1):
        Plm_vals = [lpmv(m, l, np.cos(theta)) for m in range(l + 1)]
        for m in range(l + 1):
            Plm = Plm_vals[m]
            cos_mphi = np.cos(m * phi)
            sin_mphi = np.sin(m * phi) if m > 0 else None
            Nlm = sh_norm(l, m)

            for n in range(n_max + 1):
                radial = r**n  # Density basis: r^n
                col_c = radial * Nlm * Plm * cos_mphi
                columns.append(col_c)
                coeff_labels.append(f"a_{l}_{m}_{n}")

                if m > 0:
                    col_s = radial * Nlm * Plm * sin_mphi
                    columns.append(col_s)
                    coeff_labels.append(f"b_{l}_{m}_{n}")

    C = np.vstack(columns).T / (4 * np.pi * G)
    return C, coeff_labels


def evaluate_density(points, center, coeffs, labels, l_max, n_max):
    """
    Evaluate the density rho(r, theta, phi) at given points using the
    spherical harmonic and polynomial expansion.

    Args:
        points (ndarray): (N, 3) array of evaluation points.
        center (ndarray): (3,) center of the body.
        coeffs (ndarray): Fitted coefficients (rho_lmn).
        labels (list): Coefficient labels (e.g., 'a_l_m_n', 'b_l_m_n').
        l_max (int): Maximum spherical harmonic degree.
        n_max (int): Maximum radial polynomial order.

    Returns:
        rho (ndarray): (N,) array of density values.
    """
    # Convert to spherical coordinates
    v = points - center
    r = np.maximum(np.finfo(float).tiny, np.linalg.norm(v, axis=1))
    theta = np.arccos(np.clip(v[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(v[:, 1], v[:, 0])

    rho = np.zeros_like(r)
    idx = 0

    for l in tqdm(range(l_max + 1), desc="Evaluating density (l loop)"):
        # Precompute Legendre polynomials
        Plm_vals = [lpmv(m, l, np.cos(theta)) for m in range(l + 1)]
        for m in range(l + 1):
            Plm = Plm_vals[m]
            cos_mphi = np.cos(m * phi)
            sin_mphi = np.sin(m * phi) if m > 0 else None
            Nlm = sh_norm(l, m)

            for n in range(n_max + 1):
                # Radial polynomial term: r^n
                radial = r**n

                # Cosine term (a_l_m_n)
                rho += coeffs[idx] * radial * Nlm * Plm * cos_mphi
                idx += 1

                # Sine term (b_l_m_n) if m > 0
                if m > 0:
                    rho += coeffs[idx] * radial * Nlm * Plm * sin_mphi
                    idx += 1

    return rho / (4 * np.pi * G)


# Parameters
NUM_POINTS_PER_AXIS = 50  # Adjust for resolution vs. speed
grid_points = generate_grid_points(center, bounding_radius, NUM_POINTS_PER_AXIS)
constraint_points = grid_points

# Build design matrix for potential (from original script)
A, labels = prepare_spherical_poly_basis(points, center, l_max, n_max)
b = potentials

# Build constraint matrix for density
C, _ = prepare_density_constraint_matrix(constraint_points, center, l_max, n_max)


# Objective: 0.5 * ||A x - b||^2
def obj(x, A, b):
    r = A @ x - b
    return 0.5 * np.dot(r, r)


# Gradient: A^T (A x - b)
def grad(x, A, b):
    return A.T @ (A @ x - b)


# Hessian: A^T A
def hess(x, A, b):
    return A.T @ A


# Linear inequality: C x >= 0 → lower=0, upper=+inf
lin_con = LinearConstraint(C, lb=0, ub=np.inf)

# Initial guess (zeros or normal equations solution if feasible)
Q, R = scipy.linalg.qr(A, mode="economic")
x0 = np.linalg.solve(R, Q.T @ b)
progress = []


def callback(xk, state=None):
    # Store or print progress in normalized space
    fval = obj(xk, A, b)
    cval = np.min(C @ xk)  # Min constraint margin
    print(f"Iter: {len(progress)}, f={fval:.6e}, min(Cx)={cval:.3e}")
    progress.append((fval, cval, xk.copy()))


# Run optimization with normalized A and b
res = minimize(
    fun=obj,
    x0=x0,  # Use least-squares solution as initial guess
    args=(A, b),
    method="SLSQP",  # Changed to SLSQP for potentially better convergence on this problem
    jac=grad,
    # hess=hess,  # Commented out: SLSQP does not use the Hessian
    constraints=[lin_con],
    options=dict(maxiter=1000, ftol=1e-14, disp=True),
    callback=callback,
)

# Rescale the solution back to original space
fitted_params = res.x

"""
# pip install cvxpy osqp
import numpy as np
import cvxpy as cp

# Given
# A: (m,n), b: (m,), C: (p,n), d: (p,)
# Example shapes:
# A = np.random.randn(200, 50)
# b = np.random.randn(200)
# C = np.random.randn(20, 50)
# d = np.random.randn(20)

n = A.shape[1]
x = cp.Variable(n)

# Objective: 1/2 ||A x - b||^2
obj = 0.5 * cp.sum_squares(A @ x - b)

# Constraints: Cx >= d  (CVXPY uses elementwise >=)
cons = [C @ x >= 0]

prob = cp.Problem(cp.Minimize(obj), cons)

# Choose a fast QP solver (OSQP is solid for large LSQ)
prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, verbose=True)
fitted_params = x.value
"""

# Compute fitted potentials and residuals in original space
fitted_potentials = A @ fitted_params  # Use original A
residuals = b - fitted_potentials

# Compute variance of residuals
dof = max(len(b) - np.linalg.matrix_rank(A), 1)
sigma_squared = float((residuals @ residuals) / dof)

# --- Covariance approximation under active-set KKT (optional but useful) ---
# Treat currently-active inequalities (Cx ≈ 0) as equalities and compute
# covariance on the subspace tangent to those constraints:
AATA = A.T @ A

# Detect active constraints (tight at solution). Tolerance can be tuned.
active = np.isclose(C @ fitted_params, 0.0, atol=1e-10)
C_active = C[active, :]

if C_active.shape[0] == 0:
    # No active constraints → standard OLS covariance
    cov_matrix = sigma_squared * np.linalg.pinv(AATA)
else:
    # Nullspace Z of active constraints: C_active @ Z = 0
    Z = null_space(C_active)
    if Z.size == 0:
        # All directions constrained → fallback tiny covariance
        cov_matrix = np.zeros((A.shape[1], A.shape[1]))
    else:
        # Cov in reduced coordinates, then lift back: Cov ≈ σ² * Z (Zᵀ AᵀA Z)^(-1) Zᵀ
        ATA_red = Z.T @ AATA @ Z
        cov_red = np.linalg.pinv(ATA_red)
        cov_matrix = sigma_squared * (Z @ cov_red @ Z.T)

# Compute percentage error
percentage_error = 100 * (fitted_potentials - b) / np.abs(b)


# Evaluate the fitted potential on the asteroid surface
def evaluate_potential_on_surface(
    vertices,
    center,
    coeffs,
    l_max,
    n_max,
):
    """
    Evaluate the fitted potential at surface vertices using the
    finite-support interior solution basis.

    Args:
      vertices (ndarray): (M,3) surface points.
      center   (ndarray): (3,) body center.
      coeffs   (ndarray): fitted rho-coeff vector.
      l_max    (int): max degree.
      n_max    (int): max radial order.
      R        (float): support radius.

    Returns:
      Phi_eval (ndarray): (M,) potentials at each vertex.
    """
    # spherical coords
    v = vertices - center
    r = np.maximum(np.finfo(float).tiny, np.linalg.norm(v, axis=1))
    theta = np.arccos(np.clip(v[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(v[:, 1], v[:, 0])
    R = R_REF

    Phi_eval = np.zeros_like(r)
    idx = 0

    # loop degrees/orders
    for l in range(l_max + 1):
        # precompute Plm for this l
        Plm_vals = [lpmv(m, l, np.cos(theta)) for m in range(l + 1)]
        for m in range(l + 1):
            Plm = Plm_vals[m]
            cos_mphi = np.cos(m * phi)
            sin_mphi = np.sin(m * phi) if m > 0 else None
            Nlm = sh_norm(l, m)

            for n in range(n_max + 1):
                exp = n - l + 2
                # Radial integral from r to R
                if exp == 0:
                    radial = r**l * np.log(R / r) / (2 * l + 1)
                else:
                    radial = r**l * (R**exp - r**exp) / (exp * (2 * l + 1))
                radial += r ** (n + l + 3) / ((l + n + 3) * (2 * l + 1)) / r ** (l + 1)

                # cosine term
                Phi_eval += coeffs[idx] * (radial * Nlm * Plm * cos_mphi)
                idx += 1

                # sine term, if exists
                if m > 0:
                    Phi_eval += coeffs[idx] * (radial * Nlm * Plm * sin_mphi)
                    idx += 1

    return Phi_eval


# Shift vertices outward from center
def slightly_shift_vertices(vertices, center, epsilon=1e-6):
    directions = vertices - center
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_dirs = directions / np.maximum(norms, 1e-12)
    return vertices + epsilon * unit_dirs


# Shift each vertex slightly outward
shifted_vertices = slightly_shift_vertices(vertices, center, epsilon=1e-6)

# Evaluate at each shifted vertex
true_potentials = []
for pt in tqdm(shifted_vertices, desc="True potentials at shifted surface"):
    pot, _, _ = gravity_model(pt, parallel=False)
    true_potentials.append(pot)
true_potentials = np.array(true_potentials)

# Evaluate fitted potential
fitted_potentials = evaluate_potential_on_surface(
    shifted_vertices, center, fitted_params, l_max, n_max
)
vertex_errors = 100 * (fitted_potentials - true_potentials) / np.abs(true_potentials)

# Assign per-face error as mean of its 3 vertex errors
face_errors = np.mean(vertex_errors[faces], axis=1)
face_colors = plt.cm.seismic(face_errors / np.max(face_errors))

# Plot mesh with per-face error coloring
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

mesh = Poly3DCollection(
    vertices[faces], facecolors=face_colors, edgecolor="k", linewidths=0.2, alpha=1.0
)
ax.add_collection3d(mesh)

# Force axis limits from mesh vertices
max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max()
mid_x, mid_y, mid_z = vertices.mean(axis=0)
ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

ax.set_title("Asteroid Surface Potential Error (Evaluated at Shifted Vertices)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# Add colorbar correctly
mappable = plt.cm.ScalarMappable(cmap="seismic")
mappable.set_array(face_errors)
cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
cbar.set_label("Percentage Error (%)")
plt.tight_layout()
plt.show()


def plot_spherical_power_spectrum_with_uncertainty(fitted_params, labels, covariance):
    # Parse labels into (l, m, n, type)
    parsed = []
    for label in labels:
        parts = label.split("_")
        if len(parts) == 4:
            typ = parts[0]
            l, m, n = map(int, parts[1:])
            parsed.append((l, m, n, typ))

    max_l = max(p[0] for p in parsed)
    max_n = max(p[2] for p in parsed)
    max_ln = max_l + max_n + 1

    power_ln = np.zeros(max_ln)
    sigma_power_ln_sq = np.zeros(max_ln)
    sigma_only_power_ln = np.zeros(max_ln)

    # Compute grouped power and propagated uncertainty
    for idx, (l, m, n, typ) in enumerate(parsed):
        ln = l + n
        a = fitted_params[idx]
        sigma = np.sqrt(np.abs(covariance[idx, idx]))
        power_ln[ln] += a**2
        sigma_power_ln_sq[ln] += (2 * a * sigma) ** 2
        sigma_only_power_ln[ln] += sigma**2  # purely uncertainty-based power

    sigma_power_ln = np.sqrt(sigma_power_ln_sq)
    ln_indices = np.arange(max_ln)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(
        ln_indices, power_ln, marker="o", linestyle="-", color="#2c7bb6", label="Power"
    )
    plt.fill_between(
        ln_indices,
        power_ln - sigma_power_ln,
        power_ln + sigma_power_ln,
        color="#2c7bb6",
        alpha=0.3,
        label="±1σ (propagated)",
    )
    plt.semilogy(
        ln_indices,
        sigma_only_power_ln,
        marker="x",
        linestyle="--",
        color="#fdae61",
        label="σ-only Power",
    )

    plt.xlabel(r"Total order $l+n$")
    plt.ylabel(r"Power / Uncertainty")
    plt.title("Spherical Harmonic Radial Power Spectrum with Uncertainty Bands")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Call the function with uncertainty
plot_spherical_power_spectrum_with_uncertainty(fitted_params, labels, cov_matrix)


# Convert vertices to spherical coordinates
rvec = shifted_vertices - center
x, y, z = rvec[:, 0], rvec[:, 1], rvec[:, 2]
r = np.linalg.norm(rvec, axis=1)
theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # [0, pi]
phi = np.arctan2(y, x)  # [-pi, pi]
phi[phi < 0] += 2 * np.pi  # [0, 2pi]

# Flatten to lat-lon
lat = 90 - np.degrees(theta)  # latitude from +90 (north) to -90 (south)
lon = np.degrees(phi)  # longitude from 0 to 360

# Interpolate onto grid
N_LAT, N_LON = 180, 360
grid_lat = np.linspace(-90, 90, N_LAT)
grid_lon = np.linspace(0, 360, N_LON)
lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

grid_vals = griddata(
    points=(lon, lat),
    values=fitted_potentials,  # or vertex_errors, etc.
    xi=(lon_grid, lat_grid),
    method="linear",
    fill_value=np.nan,
)

# Plot
plt.figure(figsize=(12, 6))
plt.imshow(
    grid_vals, extent=(0, 360, -90, 90), origin="lower", cmap="seismic", aspect="auto"
)
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.title("Fitted Gravitational Potential on Asteroid Surface")
cbar = plt.colorbar()
cbar.set_label("Potential (arbitrary units)")
plt.tight_layout()
plt.show()


# Mollweide projection plot of fitted potentials
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.Mollweide())
cf = ax.contourf(
    grid_lon,
    grid_lat,
    grid_vals,
    levels=60,
    transform=ccrs.PlateCarree(),
    cmap="seismic",
)
ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_global()
cbar = plt.colorbar(cf, orientation="horizontal", pad=0.05)
cbar.set_label("Potential (arbitrary units)")
plt.title("Fitted Gravitational Potential on Asteroid Surface (Mollweide Projection)")
plt.tight_layout()
plt.show()


# Load the mesh for inside/outside checks
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)


# Evaluate density at grid points
density_values = evaluate_density(
    grid_points, center, fitted_params, labels, l_max, n_max
)


def plot_density_slices(points, values, center, radius, mesh, num_points_per_axis=60):
    """
    Plot 2D contour slices of the density in xy, xz, and yz planes, with the asteroid mesh outline overlaid.

    Args:
        points (ndarray): (N, 3) array of points inside the mesh.
        values (ndarray): (N,) array of density values.
        center (ndarray): (3,) center of the body.
        radius (float): Bounding sphere radius.
        mesh (trimesh.Trimesh): Asteroid mesh for outline projection.
        num_points_per_axis (int): Number of points per axis for interpolation grid.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    x0, y0, z0 = center
    tolerance = radius / (num_points_per_axis / 2)  # Slice thickness

    # Define 2D grid for interpolation
    grid_res = num_points_per_axis // 2  # Adjust for performance
    x_grid = np.linspace(x0 - radius, x0 + radius, grid_res)
    y_grid = np.linspace(y0 - radius, y0 + radius, grid_res)
    z_grid = np.linspace(z0 - radius, z0 + radius, grid_res)
    X, Y = np.meshgrid(x_grid, y_grid)

    # XY slice (z ≈ z0)
    mask_xy = np.abs(points[:, 2] - z0) < tolerance
    if np.sum(mask_xy) > 0:
        # Interpolate density onto 2D grid
        points_xy = points[mask_xy, :2]  # (x, y)
        values_xy = values[mask_xy]
        grid_xy = griddata(points_xy, values_xy, (X, Y), method="linear", fill_value=0)

        # Plot filled contours
        cf1 = ax1.contourf(X, Y, grid_xy, levels=20, cmap="viridis")
        plt.colorbar(cf1, ax=ax1, label="Density (kg/m³)")

        # Overlay mesh outline
        plane_origin = np.array([x0, y0, z0])
        plane_normal = np.array([0, 0, 1])
        lines = mesh_plane(mesh, plane_normal, plane_origin)
        for line in lines:
            ax1.plot(line[:, 0], line[:, 1], "k-", linewidth=1, alpha=0.5)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title(f"Density Slice: XY-plane (z ≈ {z0:.1f} m)")
    ax1.set_xlim(x0 - radius, x0 + radius)
    ax1.set_ylim(y0 - radius, y0 + radius)
    ax1.set_aspect("equal")

    # XZ slice (y ≈ y0)
    mask_xz = np.abs(points[:, 1] - y0) < tolerance
    if np.sum(mask_xz) > 0:
        points_xz = points[mask_xz][:, [0, 2]]  # (x, z)
        values_xz = values[mask_xz]
        X_xz, Z_xz = np.meshgrid(x_grid, z_grid)
        grid_xz = griddata(
            points_xz, values_xz, (X_xz, Z_xz), method="linear", fill_value=0
        )

        cf2 = ax2.contourf(X_xz, Z_xz, grid_xz, levels=20, cmap="viridis")
        plt.colorbar(cf2, ax=ax2, label="Density (kg/m³)")

        plane_origin = np.array([x0, y0, z0])
        plane_normal = np.array([0, 1, 0])
        lines = mesh_plane(mesh, plane_normal, plane_origin)
        for line in lines:
            ax2.plot(line[:, 0], line[:, 2], "k-", linewidth=1, alpha=0.5)

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title(f"Density Slice: XZ-plane (y ≈ {y0:.1f} m)")
    ax2.set_xlim(x0 - radius, x0 + radius)
    ax2.set_ylim(z0 - radius, z0 + radius)
    ax2.set_aspect("equal")

    # YZ slice (x ≈ x0)
    mask_yz = np.abs(points[:, 0] - x0) < tolerance
    if np.sum(mask_yz) > 0:
        points_yz = points[mask_yz][:, [1, 2]]  # (y, z)
        values_yz = values[mask_yz]
        Y_yz, Z_yz = np.meshgrid(y_grid, z_grid)
        grid_yz = griddata(
            points_yz, values_yz, (Y_yz, Z_yz), method="linear", fill_value=0
        )

        cf3 = ax3.contourf(Y_yz, Z_yz, grid_yz, levels=20, cmap="viridis")
        plt.colorbar(cf3, ax=ax3, label="Density (kg/m³)")

        plane_origin = np.array([x0, y0, z0])
        plane_normal = np.array([1, 0, 0])
        lines = mesh_plane(mesh, plane_normal, plane_origin)
        for line in lines:
            ax3.plot(line[:, 1], line[:, 2], "k-", linewidth=1, alpha=0.5)

    ax3.set_xlabel("Y (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_title(f"Density Slice: YZ-plane (x ≈ {x0:.1f} m)")
    ax3.set_xlim(y0 - radius, y0 + radius)
    ax3.set_ylim(z0 - radius, z0 + radius)
    ax3.set_aspect("equal")

    plt.suptitle("Density Distribution Slices with Asteroid Outline")
    plt.tight_layout()
    plt.show()


# Plot the density slices with contours and mesh outline
plot_density_slices(
    grid_points, density_values, center, bounding_radius, mesh, NUM_POINTS_PER_AXIS
)
