import numpy as np
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
import mesh_utility
from tqdm import tqdm
import trimesh

# Meshes from https://github.com/darioizzo/geodesyNets/tree/master/3dmeshes
# vertices, faces = mesh_utility.read_pk_file("3dmeshes/bennu.pk")
# vertices_lp, faces_lp = mesh_utility.read_pk_file("3dmeshes/bennu_lp.pk")
# vertices, faces = np.array(vertices), np.array(faces)

mesh = trimesh.load("3dmeshes/phobos_lowlowres.obj")
vertices = mesh.vertices
faces = mesh.faces

# Compute bounding sphere
center = np.mean(vertices, axis=0)
radii = np.linalg.norm(vertices - center, axis=1)
bounding_radius = 1 * np.max(radii)

# Define asteroid density
DENSITY = 1.0  # arbitrary units
G = 6.67430 * 1e-11  # m^3 kg^-1 s^-2, gravitational constant
EPS = 4 * np.pi * G  # constant factor for potential

# Initialize polyhedron model
polyhedron = Polyhedron(
    polyhedral_source=(vertices, faces),
    density=DENSITY,
    integrity_check=PolyhedronIntegrity.DISABLE,
)

# Evaluable wrapper for gravity potential
gravity_model = GravityEvaluable(polyhedron)


# Generate random points inside the bounding sphere
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


def generate_points_outside_mesh(center, radius, num_points, mesh):
    np.random.seed(0)
    accepted_points = []
    batch_size = num_points * 2  # Oversample to avoid too many rejections

    while len(accepted_points) < num_points:
        # Generate batch of points
        u = np.random.uniform(0, 1, batch_size)
        costheta = np.random.uniform(-1, 1, batch_size)
        phi = np.random.uniform(0, 2 * np.pi, batch_size)

        theta = np.arccos(costheta)
        r = radius * np.cbrt(u)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        points = np.stack((x, y, z), axis=-1) + center

        # Check which points are outside the mesh
        inside = mesh.contains(points)
        outside_points = points[~inside]

        accepted_points.extend(outside_points.tolist())
        print(f"Generated {len(accepted_points)} accepted points so far.")

    return np.array(accepted_points[:num_points])


NUM_POINTS = 10000
sample_points = generate_points_in_sphere(center, bounding_radius, NUM_POINTS)
# sample_points = generate_points_outside_mesh(center, bounding_radius, NUM_POINTS, mesh)


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


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv
from scipy.optimize import lsq_linear
from scipy.stats import norm
from tqdm import tqdm

# Load the spherical dataset
data = np.load("spherical_gravity_dataset.npz")
points = data["points"]
potentials = data["potential"]
center = data["center"]
bounding_radius = data["bounding_radius"]


# Define real spherical harmonics radial basis model
def prepare_spherical_poly_basis(points, center, l_max, n_max):
    transformed = points - center
    x, y, z = transformed[:, 0], transformed[:, 1], transformed[:, 2]
    r = np.linalg.norm(transformed, axis=1)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)

    A = []
    coeff_labels = []

    for l in range(l_max + 1):
        for m in range(0, l + 1):
            Plm = lpmv(m, l, np.cos(theta))
            cos_mphi = EPS * (1 / (2 * l + 1)) * np.cos(m * phi)
            sin_mphi = EPS * (1 / (2 * l + 1)) * np.sin(m * phi)
            for n in range(n_max + 1):
                r_pow = r ** (l + n)
                A.append(r_pow * Plm * cos_mphi)
                coeff_labels.append(f"a_{l}_{m}_{n}")
                if m > 0:
                    A.append(r_pow * Plm * sin_mphi)
                    coeff_labels.append(f"b_{l}_{m}_{n}")

    A = np.vstack(A).T
    return A, coeff_labels


# Set model complexity
l_max, n_max = 3, 3

# Build design matrix and perform least-squares fitting
A, labels = prepare_spherical_poly_basis(points, center, l_max, n_max)
b = potentials
result = np.linalg.lstsq(A, b, rcond=None)
fitted_params = result[0]  # result.x
fitted_potentials = A @ fitted_params

# Compute residuals
residuals = b - fitted_potentials
sigma_squared = np.sum(residuals**2) / (len(b) - len(fitted_params))
cov_matrix = sigma_squared * np.linalg.pinv(A.T @ A)

# Compute percentage error
percentage_error = 100 * np.abs((fitted_potentials - b) / np.maximum(np.abs(b), 1e-10))

# Plot histogram of percentage error
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(
    percentage_error, bins=50, color="#2c7bb6", alpha=0.7, edgecolor="k", density=True
)
mu, std = norm.fit(percentage_error)
x = np.linspace(min(percentage_error), max(percentage_error), 1000)
p = norm.pdf(x, mu, std)
plt.plot(
    x,
    p,
    color="#d7191c",
    linestyle="--",
    linewidth=2,
    label=rf"$\mu={mu:.2f}, \sigma={std:.2f}$",
)
plt.xlabel("Percentage Error (%)")
plt.ylabel("Density")
plt.title("Histogram of Percentage Error in Potential Fit")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

"""
# Scatter error on sphere
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mesh_utility

# Plot percentage error and asteroid mesh
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot asteroid surface
mesh = Poly3DCollection(
    vertices[faces], alpha=0.4, facecolor="lightgrey", edgecolor="k", linewidths=0.2
)
ax.add_collection3d(mesh)

# Plot scatter error
sc = ax.scatter(
    points[:, 0], points[:, 1], points[:, 2], c=percentage_error, cmap="viridis", s=10
)
cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
cbar.set_label("Percentage Error (%)")

# Axis labels and layout
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Spatial Distribution of Percentage Error with Asteroid Mesh")
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
plt.tight_layout()
plt.show()
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.special import lpmv


# Evaluate the fitted potential on the asteroid surface
def evaluate_potential_on_surface(vertices, center, coeffs, l_max, n_max):
    rvec = vertices - center
    x, y, z = rvec[:, 0], rvec[:, 1], rvec[:, 2]
    r = np.linalg.norm(rvec, axis=1)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)

    Phi_eval = np.zeros_like(r)
    idx = 0

    for l in range(l_max + 1):
        for m in range(0, l + 1):
            Plm = lpmv(m, l, np.cos(theta))
            cos_mphi = EPS * (1 / (2 * l + 1)) * np.cos(m * phi)
            sin_mphi = EPS * (1 / (2 * l + 1)) * np.sin(m * phi)
            for n in range(n_max + 1):
                r_pow = r ** (l + n)
                Phi_eval += coeffs[idx] * r_pow * Plm * cos_mphi
                idx += 1
                if m > 0:
                    Phi_eval += coeffs[idx] * r_pow * Plm * sin_mphi
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
vertex_errors = 100 * np.abs(
    (fitted_potentials - true_potentials) / np.maximum(np.abs(true_potentials), 1e-10)
)

# Assign per-face error as mean of its 3 vertex errors
face_errors = np.mean(vertex_errors[faces], axis=1)
face_colors = plt.cm.viridis(face_errors / np.max(face_errors))

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
mappable = plt.cm.ScalarMappable(cmap="viridis")
mappable.set_array(face_errors)
cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
cbar.set_label("Percentage Error (%)")

plt.tight_layout()
plt.show()

"""'
# Plot spherical harmonic coefficient magnitudes by (l, n)
def plot_spherical_coefficients(fitted_params, labels):
    # Parse labels and store (l, m, n, type)
    parsed = []
    for label in labels:
        parts = label.split("_")
        if len(parts) == 4:
            typ = parts[0]  # 'a' or 'b'
            l, m, n = map(int, parts[1:])
        else:  # fallback for malformed label
            continue
        parsed.append((l, m, n, typ))

    # Build a grid of coefficient magnitudes
    max_l = max(p[0] for p in parsed)
    max_n = max(p[2] for p in parsed)
    coeff_mag_a = np.zeros((max_l + 1, max_n + 1))
    coeff_mag_b = np.zeros((max_l + 1, max_n + 1))

    for idx, (l, m, n, typ) in enumerate(parsed):
        mag = np.abs(fitted_params[idx])
        if typ == "a":
            coeff_mag_a[l, n] += mag  # accumulate across m
        elif typ == "b":
            coeff_mag_b[l, n] += mag

    # Plot heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axs[0].imshow(coeff_mag_a, origin="lower", aspect="auto", cmap="viridis")
    axs[0].set_title("Cosine Coefficient Magnitudes (aₗₘₙ)")
    axs[0].set_xlabel("Radial order n")
    axs[0].set_ylabel("Degree l")
    fig.colorbar(im0, ax=axs[0], label="|aₗₘₙ|")

    im1 = axs[1].imshow(coeff_mag_b, origin="lower", aspect="auto", cmap="viridis")
    axs[1].set_title("Sine Coefficient Magnitudes (bₗₘₙ)")
    axs[1].set_xlabel("Radial order n")
    axs[1].set_ylabel("Degree l")
    fig.colorbar(im1, ax=axs[1], label="|bₗₘₙ|")

    plt.tight_layout()
    plt.show()


# Call the function
plot_spherical_coefficients(fitted_params, labels)


# Plot spherical harmonic coefficient spectrum as a power spectrum
def plot_spherical_power_spectrum(fitted_params, labels):
    parsed = []
    for label in labels:
        parts = label.split("_")
        if len(parts) == 4:
            typ = parts[0]  # 'a' or 'b'
            l, m, n = map(int, parts[1:])
        else:
            continue
        parsed.append((l, m, n, typ))

    max_l = max(p[0] for p in parsed)
    max_n = max(p[2] for p in parsed)

    # Power spectrum: sum of squared coefficients grouped by (l+n)
    max_ln = max_l + max_n + 1
    power_ln = np.zeros(max_ln)

    for idx, (l, m, n, typ) in enumerate(parsed):
        ln = l + n
        power_ln[ln] += fitted_params[idx] ** 2

    ln_indices = np.arange(max_ln)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(ln_indices, power_ln, marker="o", linestyle="-", color="#2c7bb6")
    plt.xlabel(r"Total order $l+n$")
    plt.ylabel(r"Power $\sum |a_{lmn}|^2 + |b_{lmn}|^2$")
    plt.title("Spherical Harmonic Radial Power Spectrum")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# Call the function
plot_spherical_power_spectrum(fitted_params, labels)
"""


def plot_spherical_power_spectrum_with_uncertainty(fitted_params, labels, covariance):
    import matplotlib.pyplot as plt
    import numpy as np

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


# Convert fitted potentials to latitude-longitude grid for visualization
from scipy.interpolate import griddata

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
    grid_vals, extent=(0, 360, -90, 90), origin="lower", cmap="viridis", aspect="auto"
)
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.title("Fitted Gravitational Potential on Asteroid Surface")
cbar = plt.colorbar()
cbar.set_label("Potential (arbitrary units)")
plt.tight_layout()
plt.show()
