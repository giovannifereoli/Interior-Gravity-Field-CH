#########################################################################################################################
# Small Body Characterization - Cylinder Gravity LS Fitting
# Author: Giovanni Fereoli (The University of Coloradto at Boulder)
# Advisor: Dr. McMahon (The University of Colorado at Boulder)
# Acknowledgement: None
# Date: 2024-09-30
#
# TODO: 1) SOMETHING OFF IN THE TRANSLATION/ROTATON....
#########################################################################################################################

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

# Use a colorblind-friendly color palette
COLOR_PALETTE = ["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_PALETTE)

# Set LaTeX formatting
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

# Meshes from https://github.com/darioizzo/geodesyNets/tree/master/3dmeshes
vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
vertices_lp, faces_lp = mesh_utility.read_pk_file("3dmeshes/eros_lp.pk")
vertices, faces = np.array(vertices), np.array(faces)

# Define asteroid density
DENSITY = 1.0

# Define cylinder parameters
CYLINDER_CENTER = np.array([0.0, 0.0, 0.35])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.eye(3)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 2000  # Number of points to generate
ALPHA = 100  # Scaling parameter for the cylinder

# Define cylinder parameters
"""'
CYLINDER_CENTER = np.array([0.0, -0.8, 0])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate
"""

# Initialize the polyhedron object
eros = Polyhedron(
    polyhedral_source=(vertices, faces),
    density=DENSITY,
    integrity_check=PolyhedronIntegrity.DISABLE,
)

# Create an evaluable object for gravity calculations
evaluable_eros = GravityEvaluable(eros)


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
    potential, acceleration, tensor = evaluable_eros(
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

    a_x, a_y, a_z = accelerations[:, 0], accelerations[:, 1], accelerations[:, 2]
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
            dV_dphi = exp_term * bessel_j * (m / rho)

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


# Generate the matrix A and vector b
n_n, n_m = 30, 30  # Truncation parameters
num_params = 2 * n_n * n_m  # Updated for two coefficients per term (A and B)
points = structured_results["points"]
accelerations = structured_results["acceleration"]
potentials = structured_results["potential"]

# Transform Cartesian accelerations to cylindrical
cylindrical_accelerations = cartesian_to_cylindrical_acceleration(points, accelerations)

# Prepare the system for cylindrical acceleration fitting
A, b = prepare_linear_system_for_cylindrical_acceleration(
    points, cylindrical_accelerations, n_n, n_m
)

# Define regularization parameters
# M = A.shape[1]  # Number of parameters
# alpha = 1e-3  # Regularization strength
# Generate weights for parameters based on increasing n and m
# Assume the parameter indexing matches the expansion order
# order_weights = np.zeros(M)
# idx = 0
# for m in range(n_m):
#     for n in range(1, n_n + 1):
#         # Assign weights that increase with n and m
#         weight = n + m  # Simple linear scaling based on n and m
#         order_weights[idx : idx + 4] = alpha * weight  # Apply to each block of 4 terms
#         idx += 2
# regularization_matrix = np.diag(order_weights)
# aug_A = np.vstack([A, regularization_matrix])
# aug_b = np.hstack([b, np.zeros(M)])
# Solve the least squares problem
# bounds = (-1, 1)  # Constrain coefficients
# result = lsq_linear(A, b, bounds=bounds, verbose=2)
# fitted_params = result.x

# Extract fitted parameters
fitted_params = np.linalg.solve(A.T @ A + 1e-14 * np.eye(A.shape[1]), A.T @ b)
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
    plt.show()


# Function to display coefficients in matrix form
def plot_coefficient_matrices(A, B):
    """
    Plot the coefficients A and B in matrix form.

    Args:
        A, B: Coefficient matrices of shape (n_m, n_n).
    """
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

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
    plt.show()


# Call the functions
A, B = print_fitted_parameters(fitted_params, n_n, n_m)
plot_coefficients_semilogy(A, B, n_n, n_m)
plot_coefficient_matrices(A, B)


## Plot percentage error


def plot_histogram_with_gaussian(percentage_error):
    plt.figure(figsize=(10, 6))

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
    plt.xlabel("Percentage Error (-)", labelpad=10)
    plt.ylabel("Frequency Density (-)", labelpad=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()


## Plot percentage error on the cylinder
def plot_error_on_cylinder(cylinder_points, percentage_error):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot asteroid mesh
    mesh = Poly3DCollection(
        vertices[faces],
        alpha=0.8,
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
    )

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Percentage Error (-)")

    # Set labels and title
    ax.set_xlabel("$X$ (m)")
    ax.set_ylabel("$Y$ (m)")
    ax.set_zlabel("$Z$ (m)")
    ax.set_aspect("equal")
    plt.show()


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
            dV_dphi = m / rho * exp_term * bessel_j
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
plot_histogram_with_gaussian(percentage_acceleration_error)

# Plot error distribution in the cylinder
plot_error_on_cylinder(cylinder_points, percentage_acceleration_error)


"""'
## Study on potential


# Prepare the matrix A and vector b for the linear least squares fitting with the new formulation
def prepare_linear_system_new(points, potentials, n_n, n_m):
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
A_pot, b_pot = prepare_linear_system_new(points, potentials, n_n, n_m)


# Calculate percentage error
fitted_potentials = A_pot @ fitted_params
percentage_error = 100 * np.abs((fitted_potentials - b_pot) / b_pot)
print("Percentage Error:", percentage_error)

# Call the function to plot the histogram with Gaussian fit
plot_histogram_with_gaussian(percentage_error)


# Call the function to plot the percentage error
plot_error_on_cylinder(cylinder_points, percentage_error)
"""
