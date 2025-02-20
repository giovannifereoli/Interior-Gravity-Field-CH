#########################################################################################################################
# Small Body Characterization - Cylinder Gravity LS Fitting
# Author: Giovanni Fereoli (The University of Coloradto at Boulder)
# Advisor: Dr. McMahon (The University of Colorado at Boulder)
# Acknowledgement: None
# Date: 2024-09-30
#
# TODO: 1) Fai accelerazioni
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
from scipy.special import jv as BesselJ, iv as BesselI, jn_zeros, factorial
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
CYLINDER_CENTER = np.array([0.0, 0.0, 0.27])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.eye(3)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate
ALPHA = 100  # Scaling parameter

# Define cylinder parameters (2)
"""
CYLINDER_CENTER = np.array([0.0, -0.3, 0])  # Center of the cylinder base in XYZ
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
            k_mn = k(m, n)
            norm_coeff = 1  # np.sqrt((2 * n + 1) * factorial(n - m) / factorial(n + m))

            # Compute terms
            exp_term = np.exp(-k_mn * z / R_alpha)
            bessel_j = BesselJ(m, k_mn * rho / R_alpha)

            # Populate matrix A with coefficients for A_mn and B_mn
            A[:, idx] = norm_coeff * exp_term * bessel_j * np.cos(m * phi)
            A[:, idx + 1] = norm_coeff * exp_term * bessel_j * np.sin(m * phi)

            idx += 2

    return A, b


# Generate the matrix A and vector b
n_n, n_m = 20, 20  # Truncation parameters
num_params = 2 * n_n * n_m  # Updated for two coefficients per term (A and B)
points = structured_results["points"]
potentials = structured_results["potential"]
A, b = prepare_linear_system_new(
    points, potentials, n_n, n_m
)  # Using the updated function

"""
# Define regularization parameters
M = A.shape[1]  # Number of parameters
alpha = 1e-3  # Regularization strength
order_weights = np.zeros(M)
idx = 0
for m in range(n_m):
    for n in range(1, n_n + 1):
        # Assign weights that increase with n and m
        weight = n + m  # Simple linear scaling based on n and m
        order_weights[idx : idx + 2] = alpha * weight  # Apply to each block of 4 terms
        idx += 2
reg_A = np.diag(order_weights)
reg_b = np.zeros(M)
aug_A = np.vstack([A, regularization_matrix])
aug_b = np.hstack([b, np.zeros(M)])
"""

# Add noise to b
# b_noisy = b + np.random.normal(loc=0, scale=0.1 * np.mean(np.abs(b)), size=b.shape)


# Solve the linear least squares problem using lsq_linear
bounds = (-1, 1)  # Lower and upper bounds for parameters
result = lsq_linear(A, b, bounds=bounds, verbose=2)  # , tol=1e-14, method="bvls"

# Extract fitted parameters
fitted_params = result.x
# fitted_params = np.linalg.solve(
#    A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ b_noisy
# )  # result.x
print("Fitted parameters:", fitted_params)

# Calculate percentage error
fitted_potentials = A @ fitted_params
percentage_error = 100 * np.abs((fitted_potentials - b) / b)
print("Percentage Error:", percentage_error)


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


# Call the function to plot the histogram with Gaussian fit
plot_histogram_with_gaussian(percentage_error)


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


# Call the function to plot the percentage error
plot_error_on_cylinder(cylinder_points, percentage_error)


## Plot the asteroid and the cylinder


def plot_cylinder_and_asteroid(vertices, faces, cylinder_points):
    # Create a 3D plot
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

    # Plot cylinder points
    ax.scatter(
        cylinder_points[:, 0],
        cylinder_points[:, 1],
        cylinder_points[:, 2],
        c=COLOR_PALETTE[0],
        s=2,
        label="Cylinder Points",
    )

    # Set labels and title with LaTeX formatting
    ax.set_xlabel("$X$ (m)", labelpad=10)
    ax.set_ylabel("$Y$ (m)", labelpad=10)
    ax.set_zlabel("$Z$ (m)", labelpad=10)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    plt.show()


# Call the function to plot the cylinder above the asteroid
plot_cylinder_and_asteroid(vertices, faces, cylinder_points)

# Numerical gradient calculation
gradient_numerical = np.zeros_like(points)
h = 1e-2


def compute_potential(A, fitted_params):
    return A @ fitted_params


for i in range(3):  # Iterate over x, y, z
    perturbed_points = points.copy()
    perturbed_points[:, i] += h  # Positive perturbation
    A_perturbed, _ = prepare_linear_system_new(perturbed_points, potentials, n_n, n_m)
    b_perturbed = compute_potential(A_perturbed, fitted_params)
    gradient_numerical[:, i] = (b_perturbed - b) / h  # Finite difference

# Output gradient
print("Numerical Gradient:")
print(gradient_numerical)

print("Analytical Gradient:")
print(structured_results["acceleration"])

'''
## Study error on Acceleration


# Function to calculate acceleration from potential
def calculate_acceleration_from_potential(points, fitted_params, n_n, n_m):
    """
    Computes the acceleration from the fitted potential model.

    Args:
        points: Cartesian points where acceleration is evaluated.
        fitted_params: Parameters of the fitted potential model.
        n_n, n_m: Truncation parameters.

    Returns:
        Array of Cartesian accelerations.
    """
    R, L = CYLINDER_RADIUS, CYLINDER_HEIGHT
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
    z = transformed_points[:, 2]

    k = lambda m, n: jn_zeros(m, n)[-1]

    acceleration_cylindrical = np.zeros_like(transformed_points)

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            k_mn = k(m, n)
            exp_term = np.exp(-k_mn / R * z)
            bessel_j = BesselJ(m, k_mn / R * rho)

            # Calculate contributions to the potential
            cos_m_phi = np.cos(m * phi)
            sin_m_phi = np.sin(m * phi)

            dV_drho = (
                -k_mn
                / R
                * exp_term
                * bessel_j
                * (fitted_params[idx] * cos_m_phi + fitted_params[idx + 1] * sin_m_phi)
            )
            dV_dphi = (
                -m
                * exp_term
                * bessel_j
                * (-fitted_params[idx] * sin_m_phi + fitted_params[idx + 1] * cos_m_phi)
                / rho
            )
            dV_dz = (
                k_mn
                / R
                * exp_term
                * bessel_j
                * (fitted_params[idx] * cos_m_phi + fitted_params[idx + 1] * sin_m_phi)
            )

            acceleration_cylindrical[:, 0] += dV_drho  # radial
            acceleration_cylindrical[:, 1] += dV_dphi  # angular
            acceleration_cylindrical[:, 2] += dV_dz  # axial

            idx += 2

    # Convert cylindrical acceleration to Cartesian
    acceleration_cartesian = np.zeros_like(points)
    acceleration_cartesian[:, 0] = acceleration_cylindrical[:, 0] * np.cos(
        phi
    ) - acceleration_cylindrical[:, 1] * np.sin(phi)
    acceleration_cartesian[:, 1] = acceleration_cylindrical[:, 0] * np.sin(
        phi
    ) + acceleration_cylindrical[:, 1] * np.cos(phi)
    acceleration_cartesian[:, 2] = acceleration_cylindrical[:, 2]

    return acceleration_cartesian


# Calculate acceleration from the fitted model
fitted_acceleration = calculate_acceleration_from_potential(
    points, fitted_params, n_n, n_m
)

# Compare with ground truth acceleration
ground_truth_acceleration = structured_results["acceleration"]

# Calculate error in acceleration
acceleration_error = np.linalg.norm(
    fitted_acceleration - ground_truth_acceleration, axis=1
)
percentage_acceleration_error = (
    100 * acceleration_error / np.linalg.norm(ground_truth_acceleration, axis=1)
)

# Plot histogram of acceleration error
plot_histogram_with_gaussian(percentage_acceleration_error)

# Plot error distribution in the cylinder
plot_error_on_cylinder(cylinder_points, percentage_acceleration_error)


# Function to set very small values to zero
def threshold_to_zero(values, epsilon=np.finfo(float).eps):
    """
    Replace values below epsilon with zero.

    Parameters:
        values (array-like): Array of values to threshold.
        epsilon (float): Threshold below which values are set to zero.

    Returns:
        array-like: Array with small values set to zero.
    """
    values = np.where(np.abs(values) < epsilon, 0, values)
    return values
    
# Function to calculate dynamical matrix (Hessian) from potential
def calculate_dynamical_matrix_from_potential(points, fitted_params, n_n, n_m):
    """
    Computes the Hessian (dynamical matrix) from the fitted potential model.

    Args:
        points: Cartesian points where the Hessian is evaluated.
        fitted_params: Parameters of the fitted potential model.
        n_n, n_m: Truncation parameters.

    Returns:
        Array of dynamical matrices at each point.
    """
    R, L = CYLINDER_RADIUS, CYLINDER_HEIGHT
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0]**2 + transformed_points[:, 1]**2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
    z = transformed_points[:, 2]

    k = lambda m, n: jn_zeros(m, n)[-1]

    dynamical_matrix_cartesian = np.zeros((len(points), 3, 3))

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            k_mn = k(m, n)
            bessel_j = BesselJ(m, k_mn / R * rho)
            bessel_j_derivative = BesselJ(m - 1, k_mn / R * rho) - m / (k_mn / R * rho) * bessel_j
            bessel_i = BesselI(m, n * np.pi / L * rho)
            exp_term = np.exp(-k_mn / R * z)
            sin_term = np.sin(n * np.pi / L * z)

            cos_m_phi = np.cos(m * phi)
            sin_m_phi = np.sin(m * phi)

            # Second derivatives
            d2V_drho2 = -(k_mn / R)**2 * exp_term * bessel_j * cos_m_phi * fitted_params[idx]
            d2V_dphi2 = -m**2 * exp_term * bessel_j * cos_m_phi * fitted_params[idx + 1] / rho**2
            d2V_dz2 = -(n * np.pi / L)**2 * sin_term * bessel_i * cos_m_phi * fitted_params[idx + 2]

            # Build the Hessian in cylindrical coordinates
            hessian_cylindrical = np.zeros((3, 3))
            hessian_cylindrical[0, 0] = d2V_drho2
            hessian_cylindrical[1, 1] = d2V_dphi2 / rho**2
            hessian_cylindrical[2, 2] = d2V_dz2

            # Convert to Cartesian coordinates
            rotation_matrix = np.array([
                [np.cos(phi), -rho * np.sin(phi), 0],
                [np.sin(phi), rho * np.cos(phi), 0],
                [0, 0, 1]
            ])
            hessian_cartesian = rotation_matrix @ hessian_cylindrical @ rotation_matrix.T

            dynamical_matrix_cartesian += hessian_cartesian
            idx += 4

    return dynamical_matrix_cartesian

# Calculate dynamical matrix from the fitted potential
fitted_dynamical_matrix = calculate_dynamical_matrix_from_potential(points, fitted_params, n_n, n_m)

# Compare with ground truth dynamical matrix
ground_truth_dynamical_matrix = structured_results["tensor"]

# Calculate error in the dynamical matrix
dynamical_matrix_error = np.linalg.norm(fitted_dynamical_matrix - ground_truth_dynamical_matrix, axis=(1, 2))
percentage_dynamical_matrix_error = 100 * dynamical_matrix_error / np.linalg.norm(ground_truth_dynamical_matrix, axis=(1, 2))

# Plot histogram of dynamical matrix error
plot_histogram_with_gaussian(percentage_dynamical_matrix_error)

# Plot error distribution on the cylinder
plot_error_on_cylinder(cylinder_points, percentage_dynamical_matrix_error)

print("Dynamical matrix comparison and error analysis completed.")
'''
