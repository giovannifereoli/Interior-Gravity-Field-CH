#########################################################################################################################
# Small Body Characterization - Cylinder Gravity LS Fitting
# Author: Giovanni Fereoli (The University of Coloradto at Boulder)
# Advisor: Dr. McMahon (The University of Colorado at Boulder)
# Acknowledgement: None
# Date: 2024-09-30
#########################################################################################################################

## Initialization

# Import necessary libraries
import numpy as np
from polyhedral_gravity import (
    Polyhedron,
    PolyhedronIntegrity,
    GravityEvaluable,
)
import mesh_utility
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.special import (
    jv as BesselJ,
    jvp as BesselJp,
    iv as BesselI,
    jn_zeros,
    factorial,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
from scipy.optimize import lsq_linear
from scipy.integrate import solve_ivp
from scipy.stats import lognorm
import trimesh


# Use a colorblind-friendly color palette
COLOR_PALETTE = ["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_PALETTE)

# Set LaTeX formatting
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"


# Load the OBJ file
mesh = trimesh.load("3dmeshes/BENNU_preTag.obj")
mesh = trimesh.load("3dmeshes/BENNU_afterTag.obj")

# Normalize the mesh (scale to fit in unit sphere)
mesh.apply_translation(-mesh.centroid)  # Center the mesh at the origin
scale_factor = 1.0 / np.max(mesh.bounding_box.extents)  # Scale to fit in unit sphere
mesh.apply_scale(scale_factor)

# Define a 90-degree rotation matrix around the y-axis
angle = np.radians(-90)
rotation_matrix = np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1],
    ]
)

# Apply the rotation
mesh.apply_transform(rotation_matrix.T)

# Extract vertices and faces
vertices = mesh.vertices  # (N, 3) array of vertex coordinates
faces = mesh.faces  # (M, 3) array of triangle indices

# Define asteroid density
DENSITY = 1.0

# 1) No rotation
CYLINDER_CENTER = np.array([0.0, 0.0, 0.1])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.eye(3)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate

# Hyperparameters
ALPHA = 100  # Scaling parameter for the cylinder

# Initialize the polyhedron object
bennu = Polyhedron(
    polyhedral_source=(vertices, faces),
    density=DENSITY,
    integrity_check=PolyhedronIntegrity.DISABLE,
)

# Create an evaluable object for gravity calculations
evaluable_bennu = GravityEvaluable(bennu)


# Function to generate random points within a cylinder
def generate_points_in_cylinder(center, radius, height, rotation, num_points):
    np.random.seed(1)  # Set seed for reproducibility
    theta = np.random.uniform(0, 2 * np.pi, num_points)  # Random angles
    r = np.sqrt(np.random.uniform(0, radius**2, num_points))  # Random radii
    z = np.random.uniform(0, height, num_points)  # Random heights
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    local_points = np.column_stack((x, y, z))  # Points in local cylinder coordinates
    rotated_points = local_points @ rotation.T  # Apply rotation matrix
    translated_points = rotated_points + center  # Translate to the specified center
    return translated_points


# Generate dataset points within the cylindrical volume
cylinder_points = generate_points_in_cylinder(
    CYLINDER_CENTER, CYLINDER_RADIUS, CYLINDER_HEIGHT, CYLINDER_ROTATION, NUM_POINTS
)

# Evaluate gravity at each point
results = []
for point in tqdm(cylinder_points, desc="Evaluating gravity at points"):
    potential, acceleration, tensor = evaluable_bennu(
        computation_points=point, parallel=False
    )
    results.append(
        {
            "point": point,
            "potential": potential,
            "acceleration": acceleration,
            "tensor": tensor,
        }
    )

# Convert results to a structured numpy array for easier processing
structured_results = {
    "points": np.array([res["point"] for res in results]),
    "potential": np.array([res["potential"] for res in results]),
    "acceleration": np.array([res["acceleration"] for res in results]),
    "tensor": np.array([res["tensor"] for res in results]),
}

# Save the dataset to a file
np.savez("cylindrical_gravity_dataset.npz", **structured_results)
print("Dataset generated and saved as 'cylindrical_gravity_dataset.npz'")

# Rotation matrix from Cartesian to Cylindrical is:
# R = np.array([
#    [np.cos(phi), np.sin(phi), 0],
#    [-np.sin(phi), np.cos(phi), 0],
#    [0, 0, 1]
# ])


# Convert Cartesian acceleration components to cylindrical coordinates
def cartesian_to_cylindrical_acceleration(points, accelerations):
    """
    Convert Cartesian acceleration components to cylindrical coordinates.

    Args:
        points: Cartesian points (N, 3).
        accelerations: Cartesian accelerations (N, 3).

    Returns:
        Cylindrical accelerations (N, 3): [a_rho, a_phi, a_z].
    """
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])

    transformed_accelerations = accelerations @ CYLINDER_ROTATION.T

    a_x, a_y, a_z = (
        transformed_accelerations[:, 0],
        transformed_accelerations[:, 1],
        transformed_accelerations[:, 2],
    )
    a_rho = a_x * np.cos(phi) + a_y * np.sin(phi)
    a_phi = -a_x * np.sin(phi) + a_y * np.cos(phi)
    a_z = a_z  # Axial acceleration remains the same

    return np.column_stack((a_rho, a_phi, a_z))


# Fit cylindrical acceleration components using least squares
def prepare_linear_system_for_cylindrical_acceleration(points, accelerations, n_n, n_m):
    """
    Prepare the matrix A and vector b for fitting cylindrical acceleration components.

    Args:
        points: Cartesian points (N, 3).
        accelerations: Cylindrical accelerations (N, 3).
        n_n, n_m: Truncation parameters.

    Returns:
        A: Design matrix for least squares fitting.
        b: Flattened vector of acceleration components.
    """
    R, L = CYLINDER_RADIUS, CYLINDER_HEIGHT
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
    z = transformed_points[:, 2]

    num_points = len(points)
    num_params = 2 * n_n * n_m  # Updated for A and B coefficients
    A = np.zeros((3 * num_points, num_params))
    b = accelerations.flatten(order="C")  # Ensure row-major flattening for consistency

    k = lambda m, n: jn_zeros(m, n)[-1]
    R_alpha = ALPHA * CYLINDER_RADIUS

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            # Useful coefficients
            k_mn = k(m, n)
            bessel_j = BesselJ(m, k_mn / R_alpha * rho)
            bessel_j_derivative = BesselJp(m, k_mn / R_alpha * rho)
            exp_term = np.exp(-k_mn / R_alpha * z)
            cos_m_phi = np.cos(m * phi)
            sin_m_phi = np.sin(m * phi)

            # Radial (rho) component
            dV_drho = (k_mn / R_alpha) * exp_term * bessel_j_derivative

            # Angular (phi) component
            dV_dphi = exp_term * bessel_j * (m / (rho + 1e-14))

            # Axial (z) component
            dV_dz = (-k_mn / R_alpha) * exp_term * bessel_j

            for i in range(num_points):
                # Radial (rho)
                A[3 * i, idx] = dV_drho[i] * cos_m_phi[i]
                A[3 * i, idx + 1] = dV_drho[i] * sin_m_phi[i]

                # Angular (phi)
                A[3 * i + 1, idx] = dV_dphi[i] * -sin_m_phi[i]
                A[3 * i + 1, idx + 1] = dV_dphi[i] * cos_m_phi[i]

                # Axial (z)
                A[3 * i + 2, idx] = dV_dz[i] * cos_m_phi[i]
                A[3 * i + 2, idx + 1] = dV_dz[i] * sin_m_phi[i]

            idx += 2

    return A, b


# Fit cylindrical potential using least squares
def prepare_linear_system_for_cylindrical_potential(points, potentials, n_n, n_m):
    R, L = CYLINDER_RADIUS, CYLINDER_HEIGHT
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
    z = transformed_points[:, 2]

    num_points = len(points)
    num_params = 2 * n_n * n_m  # Updated for A and B coefficients
    A = np.zeros((num_points, num_params))
    b = potentials

    k = lambda m, n: jn_zeros(m, n)[-1]
    R_alpha = ALPHA * CYLINDER_RADIUS

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            # Compute terms
            k_mn = k(m, n)
            exp_term = np.exp(-k_mn * z / R_alpha)
            bessel_j = BesselJ(m, k_mn * rho / R_alpha)

            # Populate matrix A with coefficients for A_mn and B_mn
            A[:, idx] = exp_term * bessel_j * np.cos(m * phi)
            A[:, idx + 1] = exp_term * bessel_j * np.sin(m * phi)

            idx += 2

    return A, b


# Generate the matrix A and vector b
n_n, n_m = 25, 25  # Truncation parameters
num_params = 2 * n_n * n_m  # Updated for two coefficients per term (A and B)
points = structured_results["points"]
accelerations = structured_results["acceleration"]
potentials = structured_results["potential"]

# Transform Cartesian accelerations to cylindrical
cylindrical_accelerations = cartesian_to_cylindrical_acceleration(points, accelerations)

# Prepare the system for cylindrical acceleration fitting
A_acc, b_acc = prepare_linear_system_for_cylindrical_acceleration(
    points, cylindrical_accelerations, n_n, n_m
)

# Prepare the system for cylindrical potential fitting
A_pot, b_pot = prepare_linear_system_for_cylindrical_potential(
    points, potentials, n_n, n_m
)

# Define regularization parameters
M = num_params  # Number of parameters
alpha = 1e-3  # Regularization strength
order_weights = np.zeros(M)
idx = 0
for m in range(n_m):
    for n in range(1, n_n + 1):
        # Assign weights that increase with n and m
        weight = n + m  # Simple linear scaling based on n and m
        order_weights[idx : idx + 2] = alpha * weight  # Apply to each block of 4 terms
        idx += 2
A_reg = np.diag(order_weights)
b_reg = np.zeros(M)

# Define LSQ fitting
aug_A = np.vstack([A_acc, A_pot])
aug_b = np.hstack([b_acc, b_pot])

##  Extract fitted parameters
# 1) SciPy's lsq_linear
# bounds = (-10, 10)  # Constrain coefficients
# result = lsq_linear(aug_A, aug_b, bounds=bounds, verbose=2)
# fitted_params = result.x

# 2) Numpy's lstsq
result = np.linalg.lstsq(aug_A, aug_b, rcond=None)
fitted_params = result[0]

# 3) Regularized least squares
# fitted_params = np.linalg.solve(
#    aug_A.T @ aug_A + 1e-18 * np.eye(aug_A.shape[1]), aug_A.T @ aug_b
# )

# 4) SVD-based least squares (as 2ish)
# U, Sigma, Vt = np.linalg.svd(aug_A, full_matrices=False)
# Sigma_inv = np.diag(1 / Sigma * (Sigma > 1e-10))
# fitted_params = Vt.T @ Sigma_inv @ U.T @ aug_b

print("Fitted parameters for cylindrical acceleration fitting:", fitted_params)


# Function to print fitted parameters in scientific notation
def print_fitted_parameters(fitted_params, n_n, n_m):
    """
    Print the fitted parameters (A_mn, B_mn, etc.) in scientific notation.

    Args:
        fitted_params: Flattened array of fitted coefficients (1D array).
        n_n: Number of terms in the n series (truncation parameter).
        n_m: Number of terms in the m series (truncation parameter).

    Returns:
        A, B: Separate arrays of A_mn and B_mn coefficients.
    """
    A = np.zeros((n_m, n_n))
    B = np.zeros((n_m, n_n))

    print("Fitted Parameters (in scientific notation):")
    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            # Extract alternating coefficients for A_mn and B_mn
            A[m, n - 1] = fitted_params[idx]
            B[m, n - 1] = fitted_params[idx + 1]
            print(f"  A[m={m}, n={n}] = {A[m, n - 1]:.6e}")
            print(f"  B[m={m}, n={n}] = {B[m, n - 1]:.6e}")
            idx += 2

    return A, B


# Function to plot coefficients in semilogarithmic scale with distinct colors
def plot_coefficients_semilogy(A, B, n_n, n_m):
    """
    Plot coefficients A and B in semilogarithmic scale.

    Args:
        A, B: Coefficient matrices of shape (n_m, n_n).
        n_n: Number of terms in the n series.
        n_m: Number of terms in the m series.
    """
    plt.figure(figsize=(12, 8))

    # Define colors for each coefficient type
    colors = {
        "A": "#d7191c",  # Red
        "B": "#fdae61",  # Orange
    }

    # Helper function to scatter coefficients
    def scatter_coefficients(coefficients, label, color):
        for m in range(n_m):
            # Extract coefficients for order m
            coeffs_m = coefficients[m, :]
            # Scatter plot with m repeated for each n
            x = np.full_like(coeffs_m, m, dtype=int)
            plt.scatter(
                x,
                np.abs(coeffs_m),
                color=color,
                label=f"{label}" if m == 0 else None,
                alpha=0.7,
            )

    # Scatter coefficients for A, B
    scatter_coefficients(A, "A", colors["A"])
    scatter_coefficients(B, "B", colors["B"])

    # Set y-axis to logarithmic scale
    plt.yscale("log")
    plt.xlabel("Order m (-)", fontsize=12)
    plt.ylabel("Coefficient Magnitude (-)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    # Add legend with color information
    plt.legend(loc="upper right", fontsize=10)


# Function to display coefficients in matrix form
def plot_coefficient_matrices(A, B):
    """
    Plot the coefficients A and B in matrix form.

    Args:
        A, B: Coefficient matrices of shape (n_m, n_n).
    """
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    # Plot each coefficient matrix
    matrices = [
        (A, "A", axs[0]),
        (B, "B", axs[1]),
    ]

    for matrix, label, ax in matrices:
        c = ax.imshow(np.abs(matrix), aspect="auto", cmap="viridis")
        fig.colorbar(c, ax=ax)
        ax.set_title(f"Matrix of {label} Coefficients", fontsize=14)
        ax.set_xlabel("Order n (-)", fontsize=12)
        ax.set_ylabel("Order m (-)", fontsize=12)
        ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])


# Call the functions
A, B = print_fitted_parameters(fitted_params, n_n, n_m)
plot_coefficients_semilogy(A, B, n_n, n_m)
plot_coefficient_matrices(A, B)

## Plot percentage error


def plot_histogram_with_gaussian(percentage_error, title="Percentage Error"):
    plt.figure(figsize=(12, 8))

    # Plot histogram
    n, bins, patches = plt.hist(
        percentage_error,
        bins=50,
        color=COLOR_PALETTE[-1],
        alpha=0.7,
        edgecolor="black",
        density=True,
        label="Percentage Error",
    )

    # Fit and plot Gaussian curve
    mu, std = norm.fit(percentage_error)
    x = np.linspace(min(percentage_error), max(percentage_error), 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(
        x,
        p,
        color=COLOR_PALETTE[0],
        linestyle="--",
        linewidth=2,
        label=rf"Gaussian Fit: $\mu={mu:.6f}, \sigma={std:.6f}$",
    )

    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel("Percentage Error (-)", labelpad=10)
    plt.ylabel("Frequency Density (-)", labelpad=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    from scipy.stats import lognorm


## Plot percentage error on the cylinder
def plot_error_on_cylinder(cylinder_points, percentage_error, title="Percentage Error"):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot asteroid mesh
    mesh = Poly3DCollection(
        vertices[faces],
        alpha=0.3,
        edgecolor="k",
        linewidths=0.3,
        facecolor=COLOR_PALETTE[-1],
    )
    ax.add_collection3d(mesh)

    # Scatter plot of the percentage error
    scatter = ax.scatter(
        cylinder_points[:, 0],
        cylinder_points[:, 1],
        cylinder_points[:, 2],
        c=percentage_error,
        cmap="viridis",
        s=5,
        label="Percentage Error",
        depthshade=False,
        zorder=10,
    )

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Percentage Error (-)")

    # Set labels and title
    plt.title(title, fontsize=16)
    ax.set_xlabel("$X$ (m)")
    ax.set_ylabel("$Y$ (m)")
    ax.set_zlabel("$Z$ (m)")
    ax.set_aspect("equal")


def compute_fitted_cylindrical_acceleration(points, fitted_params, n_n, n_m):
    """
    Compute the fitted acceleration in cylindrical coordinates using the fitted parameters.

    Args:
        points: Cartesian points (N, 3).
        fitted_params: Fitted coefficients (1D array).
        n_n, n_m: Truncation parameters.

    Returns:
        Fitted accelerations in cylindrical coordinates (N, 3): [a_rho, a_phi, a_z].
    """
    R, L = CYLINDER_RADIUS, CYLINDER_HEIGHT
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
    z = transformed_points[:, 2]

    num_points = len(points)
    fitted_acceleration = np.zeros((num_points, 3))  # Cylindrical: [a_rho, a_phi, a_z]

    k = lambda m, n: jn_zeros(m, n)[-1]
    alpha = 100  # Replace with a physically driven parameter if available.
    R_alpha = alpha * CYLINDER_RADIUS

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            k_mn = k(m, n)

            # Cylindrical basis function components
            exp_term = np.exp(-k_mn * z / R_alpha)
            bessel_j = BesselJ(m, k_mn * rho / R_alpha)
            bessel_j_derivative = BesselJp(m, k_mn * rho / R_alpha)
            cos_m_phi = np.cos(m * phi)
            sin_m_phi = np.sin(m * phi)

            # Compute contributions to acceleration components
            dV_drho = (k_mn / R_alpha) * exp_term * bessel_j_derivative
            dV_dphi = (m / (rho + 1e-14)) * exp_term * bessel_j
            dV_dz = (-k_mn / R_alpha) * exp_term * bessel_j

            # Add contributions from coefficients
            fitted_acceleration[:, 0] += (
                dV_drho * fitted_params[idx] * cos_m_phi
                + dV_drho * fitted_params[idx + 1] * sin_m_phi
            )
            fitted_acceleration[:, 1] += (
                dV_dphi * fitted_params[idx] * -sin_m_phi
                + dV_dphi * fitted_params[idx + 1] * cos_m_phi
            )
            fitted_acceleration[:, 2] += (
                dV_dz * fitted_params[idx] * cos_m_phi
                + dV_dz * fitted_params[idx + 1] * sin_m_phi
            )

            idx += 2

    return fitted_acceleration


# Compute fitted cylindrical acceleration
fitted_acceleration = compute_fitted_cylindrical_acceleration(
    cylinder_points, fitted_params, n_n, n_m
)

# Calculate error in acceleration
acceleration_error = np.linalg.norm(
    fitted_acceleration - cylindrical_accelerations, axis=1
)
percentage_acceleration_error = (
    100 * acceleration_error / np.linalg.norm(cylindrical_accelerations, axis=1)
)

# Plot histogram of acceleration error
plot_histogram_with_gaussian(percentage_acceleration_error, title="Acceleration Error")


# Plot error distribution in the cylinder
plot_error_on_cylinder(
    cylinder_points, percentage_acceleration_error, title="Acceleration Error"
)

## Study on potential

# Calculate percentage error
fitted_potentials = A_pot @ fitted_params
percentage_error = 100 * np.abs((fitted_potentials - b_pot) / b_pot)
print("Percentage Error:", percentage_error)

# Call the function to plot the histogram with Gaussian fit
plot_histogram_with_gaussian(percentage_error, title="Potential Error")


# Call the function to plot the percentage error
plot_error_on_cylinder(cylinder_points, percentage_error, title="Potential Error")


## Analyze Contributions


def extract_top_coefficients(fitted_params, n_n, n_m, top_n=20):
    """
    Extract the top N largest coefficients (A_mn, B_mn) sorted by magnitude.

    Args:
        fitted_params: Flattened array of fitted coefficients (1D array).
        n_n: Number of terms in the n series (truncation parameter).
        n_m: Number of terms in the m series.
        top_n: Number of top coefficients to extract.

    Returns:
        top_coefficients: List of tuples [(m, n, type, magnitude)] sorted by magnitude.
    """
    coefficients = []
    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            # Extract A_mn and B_mn
            A_mn = fitted_params[idx]
            B_mn = fitted_params[idx + 1]
            coefficients.append((m, n, "A", A_mn))
            coefficients.append((m, n, "B", B_mn))
            idx += 2

    # Sort by magnitude
    coefficients = sorted(coefficients, key=lambda x: abs(x[3]), reverse=True)
    return coefficients[:top_n]


def plot_top_coefficients(top_coefficients):
    """
    Plot the top coefficients with explicit labels.

    Args:
        top_coefficients: List of tuples [(m, n, type, magnitude)].
    """
    # Extract data for plotting
    labels = [f"${t}_{{{m},{n}}}$" for m, n, t, _ in top_coefficients]
    magnitudes = [abs(c[3]) for c in top_coefficients]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(magnitudes)), magnitudes, color="#2c7bb6", alpha=0.8)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("Coefficient ($A_{m,n}$, $B_{m,n}$)", fontsize=12)
    plt.ylabel("Magnitude (-)", fontsize=12)
    plt.title("Top Coefficients Sorted by Magnitude", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# Extract the top 20 coefficients
top_coefficients = extract_top_coefficients(fitted_params, n_n, n_m, top_n=20)

# Plot the top coefficients
plot_top_coefficients(top_coefficients)

# Print the top coefficients for inspection
print("Top Coefficients:")
for m, n, t, value in top_coefficients:
    print(f"{t}_{m},{n}: {value:.6e}")
