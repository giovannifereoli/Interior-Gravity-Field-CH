#########################################################################################################################
# Small Body Characterization - Cylinder Gravity LS Fitting
# Author: Giovanni Fereoli (The University of Coloradto at Boulder)
# Advisor: Dr. McMahon (The University of Colorado at Boulder)
# Acknowledgement: None
# Date: 2021-09-30
#
# TODO: 1) Fai accelerazioni 2) fitted a and B extraction e sbagliato?
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
    ivp as BesselIp,
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
CYLINDER_CENTER = np.array([0.0, 0.0, 0.27])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.eye(3)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate

# Define cylinder parameters
"""
CYLINDER_CENTER = np.array([0.0, -0.3, 0])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
)  # np.eye(3)  # Rotation matrix (identity matrix by default)
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


# Prepare the matrix A and vector b
def prepare_linear_system_for_acceleration(points, accelerations, n_n, n_m):
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
    num_params = 4 * n_n * n_m  # 4 coefficients per (m, n) pair
    A = np.zeros((3 * num_points, num_params))  # For all acceleration components
    b = accelerations.flatten()  # Flattened acceleration vector

    k = lambda m, n: jn_zeros(m, n)[-1]

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            # Useful coefficients
            k_mn = k(m, n)
            bessel_j = BesselJ(m, k_mn / R * rho)
            bessel_j_derivative = BesselJp(m, k_mn / R * rho)
            bessel_i = BesselI(m, n * np.pi / L * rho)
            bessel_i_derivative = BesselIp(m, n * np.pi / L * rho)
            exp_term = np.exp(-k_mn / R * z)
            sin_term = np.sin(n * np.pi / L * z)
            cos_term = np.cos(n * np.pi / L * z)
            cos_m_phi = np.cos(m * phi)
            sin_m_phi = np.sin(m * phi)

            # Radial (rho) component
            dV_drho_1 = (k_mn / R) * exp_term * bessel_j_derivative
            dV_drho_2 = (n * np.pi / L) * sin_term * bessel_i_derivative

            # Angular (phi) component
            dV_dphi_1 = exp_term * bessel_j * (m / rho)
            dV_dphi_2 = sin_term * bessel_i * (m / rho)

            # Axial (z) component
            dV_dz_1 = (-k_mn / R) * exp_term * bessel_j
            dV_dz_2 = (n * np.pi / L) * cos_term * bessel_i

            for i in range(num_points):
                # Radial (rho)
                A[3 * i, idx] = dV_drho_1[i] * cos_m_phi[i]
                A[3 * i, idx + 1] = dV_drho_1[i] * sin_m_phi[i]
                A[3 * i, idx + 2] = dV_drho_2[i] * cos_m_phi[i]
                A[3 * i, idx + 3] = dV_drho_2[i] * sin_m_phi[i]

                # Angular (phi)
                A[3 * i + 1, idx] = dV_dphi_1[i] * -sin_m_phi[i]
                A[3 * i + 1, idx + 1] = dV_dphi_1[i] * cos_m_phi[i]
                A[3 * i + 1, idx + 2] = dV_dphi_2[i] * -sin_m_phi[i]
                A[3 * i + 1, idx + 3] = dV_dphi_2[i] * cos_m_phi[i]

                # Axial (z)
                A[3 * i + 2, idx] = dV_dz_1[i] * cos_m_phi[i]
                A[3 * i + 2, idx + 1] = dV_dz_1[i] * sin_m_phi[i]
                A[3 * i + 2, idx + 2] = dV_dz_2[i] * cos_m_phi[i]
                A[3 * i + 2, idx + 3] = dV_dz_2[i] * sin_m_phi[i]

            idx += 4

    return A, b


# Generate the matrix A and vector b
n_n, n_m = 30, 30  # Truncation parameters
num_params = 4 * n_n * n_m  # Updated for two coefficients per term (A and B)
points = structured_results["points"]
accelerations = structured_results["acceleration"]

# Transform Cartesian accelerations to cylindrical
cylindrical_accelerations = cartesian_to_cylindrical_acceleration(points, accelerations)

# Prepare the system for cylindrical acceleration fitting
A, b = prepare_linear_system_for_acceleration(
    points, cylindrical_accelerations, n_n, n_m
)

# Solve the least squares problem
result = np.linalg.lstsq(A, b, rcond=None)
fitted_params = result[0]
print("Fitted parameters for cylindrical acceleration fitting:", fitted_params)


# Function to print fitted parameters in scientific notation
def print_fitted_parameters(fitted_params, n_n, n_m):
    """
    Print the fitted parameters (A_mn, B_mn, C_mn, D_mn) in scientific notation.

    Args:
        fitted_params: Flattened array of fitted coefficients (1D array).
        n_n: Number of terms in the n series (truncation parameter).
        n_m: Number of terms in the m series (truncation parameter).

    Returns:
        A, B, C, D: Separate matrices of coefficients (n_m, n_n).
    """
    A = np.zeros((n_m, n_n))
    B = np.zeros((n_m, n_n))
    C = np.zeros((n_m, n_n))
    D = np.zeros((n_m, n_n))

    print("Fitted Parameters (in scientific notation):")
    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            A[m, n - 1] = fitted_params[idx]
            B[m, n - 1] = fitted_params[idx + 1]
            C[m, n - 1] = fitted_params[idx + 2]
            D[m, n - 1] = fitted_params[idx + 3]
            idx += 4

            print(f"  A[m={m}, n={n}] = {A[m, n - 1]:.6e}")
            print(f"  B[m={m}, n={n}] = {B[m, n - 1]:.6e}")
            print(f"  C[m={m}, n={n}] = {C[m, n - 1]:.6e}")
            print(f"  D[m={m}, n={n}] = {D[m, n - 1]:.6e}")

    return A, B, C, D


# Function to plot coefficients in semilogarithmic scale with distinct colors
def plot_coefficients_semilogy(A, B, C, D, n_n, n_m):
    """
    Plot coefficients A, B, C, and D in semilogarithmic scale.

    Args:
        A, B, C, D: Coefficient matrices of shape (n_m, n_n).
        n_n: Number of terms in the n series.
        n_m: Number of terms in the m series.
    """
    plt.figure(figsize=(12, 8))

    # Define colors for each coefficient type
    colors = {
        "A": "#d7191c",  # Red
        "B": "#fdae61",  # Orange
        "C": "#abd9e9",  # Light Blue
        "D": "#2c7bb6",  # Dark Blue
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

    # Scatter coefficients for A, B, C, D
    scatter_coefficients(A, "A", colors["A"])
    scatter_coefficients(B, "B", colors["B"])
    scatter_coefficients(C, "C", colors["C"])
    scatter_coefficients(D, "D", colors["D"])

    # Set y-axis to logarithmic scale
    plt.yscale("log")
    plt.xlabel("Order m (-)", fontsize=12)
    plt.ylabel("Coefficient Magnitude (-)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    # Add legend with color information
    plt.legend(loc="upper right", fontsize=10)
    plt.show()


# Function to display coefficients in matrix form
def plot_coefficient_matrices(A, B, C, D):
    """
    Plot the coefficients A, B, C, and D in matrix form.

    Args:
        A, B, C, D: Coefficient matrices of shape (n_m, n_n).
    """
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot each coefficient matrix
    matrices = [
        (A, "A", axs[0, 0]),
        (B, "B", axs[0, 1]),
        (C, "C", axs[1, 0]),
        (D, "D", axs[1, 1]),
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


# Print and plot the fitted parameters
A, B, C, D = print_fitted_parameters(fitted_params, n_n, n_m)
plot_coefficients_semilogy(A, B, C, D, n_n, n_m)
plot_coefficient_matrices(A, B, C, D)


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

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            # Useful coefficients
            k_mn = k(m, n)
            bessel_j = BesselJ(m, k_mn / R * rho)
            bessel_j_derivative = BesselJp(m, k_mn / R * rho)
            bessel_i = BesselI(m, n * np.pi / L * rho)
            bessel_i_derivative = BesselIp(m, n * np.pi / L * rho)
            exp_term = np.exp(-k_mn / R * z)
            sin_term = np.sin(n * np.pi / L * z)
            cos_term = np.cos(n * np.pi / L * z)

            # Radial (rho) component
            dV_drho_1 = (k_mn / R) * exp_term * bessel_j_derivative
            dV_drho_2 = (n * np.pi / L) * sin_term * bessel_i_derivative

            fitted_acceleration[:, 0] += (
                dV_drho_1 * fitted_params[idx] * np.cos(m * phi)
                + dV_drho_1 * fitted_params[idx + 1] * np.sin(m * phi)
                + dV_drho_2 * fitted_params[idx + 2] * np.cos(m * phi)
                + dV_drho_2 * fitted_params[idx + 3] * np.sin(m * phi)
            )

            # Angular (phi) component
            dV_dphi_1 = exp_term * bessel_j * (m / rho)
            dV_dphi_2 = sin_term * bessel_i * (m / rho)

            fitted_acceleration[:, 1] += (
                dV_dphi_1 * fitted_params[idx] * -np.sin(m * phi)
                + dV_dphi_1 * fitted_params[idx + 1] * np.cos(m * phi)
                + dV_dphi_2 * fitted_params[idx + 2] * -np.sin(m * phi)
                + dV_dphi_2 * fitted_params[idx + 3] * np.cos(m * phi)
            )

            # Axial (z) component
            dV_dz_1 = (-k_mn / R) * exp_term * bessel_j
            dV_dz_2 = (n * np.pi / L) * cos_term * bessel_i

            fitted_acceleration[:, 2] += (
                dV_dz_1 * fitted_params[idx] * np.cos(m * phi)
                + dV_dz_1 * fitted_params[idx + 1] * np.sin(m * phi)
                + dV_dz_2 * fitted_params[idx + 2] * np.cos(m * phi)
                + dV_dz_2 * fitted_params[idx + 3] * np.sin(m * phi)
            )

            idx += 4

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
