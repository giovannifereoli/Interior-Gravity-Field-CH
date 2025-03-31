#########################################################################################################################
# Small Body Characterization - Cylinder Gravity LS Fitting
# Author: Giovanni Fereoli (The University of Coloradto at Boulder)
# Advisor: Dr. McMahon (The University of Colorado at Boulder)
# Acknowledgement: None
# Date: 2024-09-30
#########################################################################################################################

# TODO: prova commit

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
    jn_zeros,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
from scipy.integrate import solve_ivp
from datetime import datetime
from scipy.stats import lognorm


# Use a colorblind-friendly color palette

COLOR_PALETTE = ["#E6001A", "#F08C00", "#0077BB", "#2c7bb6"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_PALETTE)

# Set LaTeX formatting
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 16  # Increase font size for better readability in papers
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14

# Meshes from https://github.com/darioizzo/geodesyNets/tree/master/3dmeshes
vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
vertices_lp, faces_lp = mesh_utility.read_pk_file("3dmeshes/eros_lp.pk")
vertices, faces = np.array(vertices), np.array(faces)

# Define asteroid density
DENSITY = 1.0

# 1) No rotation
CYLINDER_CENTER = np.array([0.0, 0.0, 0.28])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.eye(3)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate

# 2) as before,Rotation x-axis
"""'
CYLINDER_CENTER = np.array([-0.1, -0.28, 0])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate
"""

# 3) as before,Rotation x-axis
"""
CYLINDER_CENTER = np.array([-1.26, 0, 0])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.array(
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate
"""

# Hyperparameters
ALPHA = 100  # Scaling parameter for the cylinder

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

# Numpy's lstsq
result = np.linalg.lstsq(aug_A, aug_b, rcond=None)
fitted_params = result[0]

# Compute residuals
# NOTE: Since you're not adding noise, the sigma is almost epsilon.
residuals = aug_b - aug_A @ fitted_params
sigma_squared = np.sum(residuals**2) / (len(aug_b) - len(fitted_params))
cov_matrix = sigma_squared * np.linalg.pinv(aug_A.T @ aug_A)

# To remove the B_0n coefficients (hard-coded)
for n in range(n_n):
    fitted_params[2 * n] = fitted_params[2 * n]  # A_0n, keep
    fitted_params[2 * n + 1] = 0.0  # B_0n, zero out
idx = 0
for m in range(n_m):
    for n in range(1, n_n + 1):
        if m == 0:
            B_idx = idx + 1  # B_0n index
            cov_matrix[B_idx, :] = 0.0
            cov_matrix[:, B_idx] = 0.0
        idx += 2

print("Fitted parameters for cylindrical acceleration fitting:", fitted_params)


def compute_and_plot_covariance(cov_matrix, n_n, n_m):
    """
    Compute standard deviations from the covariance matrix and plot them as a scatter plot for each coefficient.

    Args:
        fitted_params: Flattened array of fitted coefficients (1D array).
        cov_matrix: Covariance matrix of fitted parameters.
        n_n: Number of terms in the n series (truncation parameter).
        n_m: Number of terms in the m series (truncation parameter).
    """
    # Extract standard deviations from covariance matrix (square root of diagonal elements)
    std_devs = np.sqrt(np.diag(cov_matrix))

    # Reshape standard deviations into A_mn and B_mn structures
    sigma_A = np.zeros((n_m, n_n))
    sigma_B = np.zeros((n_m, n_n))

    m_values = []
    sigma_values_A = []
    sigma_values_B = []

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            sigma_A[m, n - 1] = std_devs[idx]
            sigma_B[m, n - 1] = std_devs[idx + 1]
            m_values.append(m)
            sigma_values_A.append(std_devs[idx])
            sigma_values_B.append(std_devs[idx + 1])
            idx += 2

    # Plot the standard deviations as a scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(
        m_values,
        sigma_values_A,
        label=r"$\sigma_{A_{mn}}$",
        marker="o",
        alpha=0.7,
        color=COLOR_PALETTE[1],
        edgecolor="black",
    )
    plt.scatter(
        m_values,
        sigma_values_B,
        label=r"$\sigma_{B_{mn}}$",
        marker="s",
        alpha=0.7,
        color=COLOR_PALETTE[2],
        edgecolor="black",
    )
    plt.xlabel("Order m (-)", labelpad=10)
    plt.yscale("log")
    plt.ylabel("Standard Deviation (-)", labelpad=10)
    plt.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        edgecolor="black",
        fontsize=14,
    )
    plt.grid(True, linestyle="--", which="both", linewidth=0.7, alpha=0.8)
    plt.minorticks_on()  # Enable minor ticks
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.savefig("Images/covariances_plot.pdf", dpi=1200, bbox_inches="tight")


# Call the function to compute and plot standard deviations
compute_and_plot_covariance(cov_matrix, n_n, n_m)


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
        "A": COLOR_PALETTE[1],  # Red
        "B": COLOR_PALETTE[2],  # Orange
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
                edgecolor="black",
            )

    # Scatter coefficients for A, B
    scatter_coefficients(A, r"$A_{mn}$", colors["A"])
    scatter_coefficients(B, r"$B_{mn}$", colors["B"])

    # Set y-axis to logarithmic scale
    plt.yscale("log")
    plt.xlabel("Order m (-)", labelpad=10)
    plt.ylabel("Coefficient Magnitude (-)", labelpad=10)
    plt.grid(True, linestyle="--", which="both", linewidth=0.7, alpha=0.8)
    plt.minorticks_on()  # Enable minor ticks
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    # Add legend with color information
    plt.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        edgecolor="black",
        fontsize=14,
    )
    plt.savefig("Images/coefficients_plot.pdf", dpi=1200, bbox_inches="tight")


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
        c = ax.imshow(np.abs(matrix), aspect="auto", cmap="plasma")
        fig.colorbar(c, ax=ax)
        ax.set_title(
            f"Matrix of {label} Coefficients",
        )
        ax.set_xlabel("Order n (-)", labelpad=10)
        ax.set_ylabel("Order m (-)", labelpad=10)
        ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])


# Call the functions
A, B = print_fitted_parameters(fitted_params, n_n, n_m)
plot_coefficients_semilogy(A, B, n_n, n_m)
plot_coefficient_matrices(A, B)

## Plot percentage error


def plot_histogram_with_lognormal(percentage_error, title="Percentage Error"):
    plt.figure(figsize=(12, 8))

    # Plot histogram
    n, bins, patches = plt.hist(
        percentage_error,
        bins=50,
        color=COLOR_PALETTE[-1],
        alpha=0.7,
        edgecolor="black",
        density=True,
        label="Histogram (PDF Estimate)",
    )

    # Fit and plot log-normal curve
    shape, loc, scale = lognorm.fit(percentage_error)
    x = np.linspace(min(percentage_error), max(percentage_error), 1000)
    p = lognorm.pdf(x, shape, loc=loc, scale=scale)

    # Compute mean and std in linear space
    mu = lognorm.mean(shape, loc, scale)
    std = lognorm.std(shape, loc, scale)

    plt.plot(
        x,
        p,
        color=COLOR_PALETTE[0],
        linestyle="--",
        linewidth=2,
        label=rf"Lognormal Fit: $\mu={mu:.6f},\ \sigma={std:.6f}$",
    )

    # Add labels and title
    plt.xlabel("Percentage Error (-)", labelpad=10)
    plt.ylabel("Probability Density (-)", labelpad=10)
    # plt.title(title, pad=15)
    plt.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        edgecolor="black",
        fontsize=14,
    )
    plt.grid(True, linestyle="--", which="both", linewidth=0.7, alpha=0.8)
    plt.minorticks_on()  # Enable minor ticks
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.savefig(
        f"Images/histo_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        dpi=1200,
        bbox_inches="tight",
    )


## Plot percentage error on the cylinder
def plot_error_on_cylinder(cylinder_points, percentage_error, title="Percentage Error"):
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
        cmap="plasma",
        s=5,
        label="Percentage Error",
    )

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.11)
    cbar.set_label("Percentage Error (-)")

    # Set labels and title
    ax.set_xlabel("$X$ (m)", labelpad=10)
    ax.set_ylabel("$Y$ (m)", labelpad=10)
    ax.set_zlabel("$Z$ (m)", labelpad=10)
    ax.set_aspect("equal")
    plt.savefig(
        f"Images/cylerror_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        dpi=1200,
        bbox_inches="tight",
    )


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
    R_alpha = ALPHA * CYLINDER_RADIUS

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
plot_histogram_with_lognormal(percentage_acceleration_error, title="Acceleration Error")


# Plot error distribution in the cylinder
plot_error_on_cylinder(
    cylinder_points, percentage_acceleration_error, title="Acceleration Error"
)

## Study on potential

# Calculate percentage error
fitted_potentials = A_pot @ fitted_params
percentage_error = 100 * np.abs((fitted_potentials - b_pot) / b_pot)
print("Percentage Error:", percentage_error)

# Call the function to plot the histogram with Chi Squared fit
plot_histogram_with_lognormal(percentage_error, title="Potential Error")


# Call the function to plot the percentage error
plot_error_on_cylinder(cylinder_points, percentage_error, title="Potential Error")

plt.show()

## Compare Trajectories


def propagate_trajectory(
    initial_position,
    initial_velocity,
    acceleration_func,
    t_span,
    method="RK45",
):
    """
    Propagate a trajectory using SciPy's ODE solver.

    Args:
        initial_position: Initial position in Cartesian coordinates (3,).
        initial_velocity: Initial velocity in Cartesian coordinates (3,).
        acceleration_func: Function to compute acceleration (takes position as input).
        max_time: Maximum propagation time.
        time_step: Time step for integration output.
        method: Integration method to use (default: DOP853).

    Returns:
        t: Array of time values.
        y: Array of propagated states (N, 6) with [x, y, z, vx, vy, vz].
    """

    def dynamics(t, state):
        position = state[:3]
        velocity = state[3:]
        acceleration = acceleration_func(position)
        return np.hstack((velocity, acceleration))

    # Initial state: [x, y, z, vx, vy, vz]
    initial_state = np.hstack((initial_position, initial_velocity))

    # Solve the system
    sol = solve_ivp(
        dynamics,
        t_span=(t_span[0], t_span[-1]),
        y0=initial_state,
        method=method,
        t_eval=t_span,
        rtol=1e-12,
        atol=1e-12,
    )

    return sol.t, sol.y


def plot_trajectories(t_poly, y_poly, t_fitted, y_fitted):
    """
    Plots the trajectories of polyhedron and fitted models in 3D space.

    Parameters:
    t_poly (array-like): Time points for the constant-density polyhedron model.
    y_poly (array-like): Trajectory data for the constant-density polyhedron model, shape (3, N).
    t_fitted (array-like): Time points for the fitted model.
    y_fitted (array-like): Trajectory data for the fitted model, shape (3, N).

    Returns:
    None
    """

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot constant-density polyhedron trajectory
    ax.plot(
        y_poly[0, :],
        y_poly[1, :],
        y_poly[2, :],
        label="Constant Density Polyhedron",
        linewidth=2,
        color=COLOR_PALETTE[0],
    )

    # Plot fitted trajectory
    ax.plot(
        y_fitted[0, :],
        y_fitted[1, :],
        y_fitted[2, :],
        label="Interior Bessel Cylindrical Harmonics",
        linewidth=2,
        color=COLOR_PALETTE[1],
    )

    # Plot asteroid mesh
    mesh = Poly3DCollection(
        vertices[faces],
        alpha=0.8,
        edgecolor="k",
        linewidths=0.3,
        facecolor=COLOR_PALETTE[-1],
    )
    ax.add_collection3d(mesh)

    # Plot the cylinder
    theta = np.linspace(0, 2 * np.pi, 50)  # Angular discretization
    z = np.linspace(0, CYLINDER_HEIGHT, 50)  # Height discretization
    theta, z = np.meshgrid(theta, z)
    x = CYLINDER_RADIUS * np.cos(theta)
    y = CYLINDER_RADIUS * np.sin(theta)

    # Apply cylinder rotation and translation
    cylinder_points = np.column_stack(
        [x.flatten(), y.flatten(), z.flatten()]
    )  # Flatten and stack as Nx3
    transformed_cylinder = cylinder_points @ CYLINDER_ROTATION.T + CYLINDER_CENTER
    x_cylinder = transformed_cylinder[:, 0].reshape(x.shape)
    y_cylinder = transformed_cylinder[:, 1].reshape(y.shape)
    z_cylinder = transformed_cylinder[:, 2].reshape(z.shape)

    ax.plot_surface(
        x_cylinder,
        y_cylinder,
        z_cylinder,
        color="lightyellow",
        alpha=0.3,  # Low opacity for a subtle appearance
        edgecolor="none",
    )

    # Set labels and title
    ax.set_xlabel("$X$ (m)", labelpad=10)
    ax.set_ylabel("$Y$ (m)", labelpad=10)
    ax.set_zlabel("$Z$ (m)", labelpad=10)
    plt.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        edgecolor="black",
        fontsize=14,
    )
    ax.set_aspect("equal")
    plt.savefig(f"Images/trajectory_plot.pdf", dpi=1200, bbox_inches="tight")


def cylindrical_to_cartesian_acceleration(points, cylindrical_accelerations):
    """
    Convert cylindrical acceleration components to Cartesian coordinates.

    Args:
        points: Cartesian points (N, 3).
        cylindrical_accelerations: Cylindrical accelerations (N, 3): [a_rho, a_phi, a_z].
        cylinder_center: Center of the cylinder in Cartesian coordinates (3,).
        cylinder_rotation: Rotation matrix of the cylinder (3, 3).

    Returns:
        Cartesian accelerations (N, 3).
    """
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])

    a_rho, a_phi, a_z = (
        cylindrical_accelerations[:, 0],
        cylindrical_accelerations[:, 1],
        cylindrical_accelerations[:, 2],
    )
    a_x = a_rho * np.cos(phi) - a_phi * np.sin(phi)
    a_y = a_rho * np.sin(phi) + a_phi * np.cos(phi)

    return np.column_stack((a_x, a_y, a_z))


def acceleration_fitted(position):
    """
    Compute the acceleration at a given position using fitted cylindrical harmonics.

    Parameters:
    position (np.ndarray): A 3-element array representing the Cartesian coordinates of the position.

    Returns:
    np.ndarray: A 3-element array representing the acceleration in Cartesian coordinates.
    """
    # Compute acceleration components in cylindrical coordinates
    a_rho, a_phi, a_z = compute_fitted_cylindrical_acceleration(
        np.array([position]), fitted_params, n_n, n_m
    )[0]

    # Convert acceleration back to Cartesian coordinates
    return cylindrical_to_cartesian_acceleration(
        np.array([position]), np.array([[a_rho, a_phi, a_z]])
    )[0]


def acceleration_poly(position):
    """
    Evaluate the gravitational acceleration at a given position using the polyhedron gravity model.

    Parameters:
    position (array-like): The position at which to evaluate the gravitational acceleration.
                           It should be an array-like object representing the coordinates.

    Returns:
    numpy.ndarray: The gravitational acceleration at the given position.
    """
    # Evaluate gravity at position using the polyhedron gravity model
    potential, acceleration, tensor = evaluable_eros(
        computation_points=position, parallel=False
    )
    return acceleration


# Initial conditions
initial_position = np.array(
    [
        -5.45118663e-02,
        -6.08104828e-02,
        7.29726385e-01,
    ]
)  # Adjust as needed
initial_velocity = np.array(
    [9.74202292e-07, 1.09203903e-06, -7.28180036e-06]
)  # Adjust as needed
t_span = np.linspace(0, 55000, 100)

# Add this to your script after defining `initial_position`, `initial_velocity`, `t_span`, etc.


def fit_and_propagate_model(n_n, n_m, initial_position, initial_velocity, t_span):
    A_acc, b_acc = prepare_linear_system_for_cylindrical_acceleration(
        points, cylindrical_accelerations, n_n, n_m
    )
    A_pot, b_pot = prepare_linear_system_for_cylindrical_potential(
        points, potentials, n_n, n_m
    )

    aug_A = np.vstack([A_acc, A_pot])
    aug_b = np.hstack([b_acc, b_pot])

    result = np.linalg.lstsq(aug_A, aug_b, rcond=None)
    fitted_params = result[0]

    # Force B_0n = 0
    for n in range(n_n):
        fitted_params[2 * n + 1] = 0.0

    def acc_fitted(position):
        a_rho, a_phi, a_z = compute_fitted_cylindrical_acceleration(
            np.array([position]), fitted_params, n_n, n_m
        )[0]
        return cylindrical_to_cartesian_acceleration(
            np.array([position]), np.array([[a_rho, a_phi, a_z]])
        )[0]

    t, y = propagate_trajectory(initial_position, initial_velocity, acc_fitted, t_span)
    return t, y


def plot_trajectories_multiple(t_poly, y_poly, traj_dict):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        y_poly[0],
        y_poly[1],
        y_poly[2],
        label="Constant Density Polyhedron",
        linewidth=2,
        color="black",
    )

    for label, traj in traj_dict.items():
        ax.plot(traj[0], traj[1], traj[2], label=label, linewidth=2)

    mesh = Poly3DCollection(
        vertices[faces],
        alpha=0.8,
        edgecolor="k",
        linewidths=0.3,
        facecolor=COLOR_PALETTE[-1],
    )
    ax.add_collection3d(mesh)

    # Cylinder surface (reused from previous plot_trajectories function)
    theta = np.linspace(0, 2 * np.pi, 50)
    z = np.linspace(0, CYLINDER_HEIGHT, 50)
    theta, z = np.meshgrid(theta, z)
    x = CYLINDER_RADIUS * np.cos(theta)
    y = CYLINDER_RADIUS * np.sin(theta)
    cylinder_points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    transformed_cylinder = cylinder_points @ CYLINDER_ROTATION.T + CYLINDER_CENTER
    x_cyl = transformed_cylinder[:, 0].reshape(x.shape)
    y_cyl = transformed_cylinder[:, 1].reshape(y.shape)
    z_cyl = transformed_cylinder[:, 2].reshape(z.shape)
    ax.plot_surface(
        x_cyl, y_cyl, z_cyl, color="lightyellow", alpha=0.3, edgecolor="none"
    )

    ax.set_xlabel("$X$ (m)", labelpad=10)
    ax.set_ylabel("$Y$ (m)", labelpad=10)
    ax.set_zlabel("$Z$ (m)", labelpad=10)
    ax.set_aspect("equal")
    plt.legend(frameon=True, fancybox=True, edgecolor="black", fontsize=14)
    plt.savefig("Images/trajectory_comparison.pdf", dpi=1200, bbox_inches="tight")
    plt.show()


def plot_trajectory_differences(t, y_poly, traj_dict):
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axs = axs.ravel()

    labels = [
        "$|\\delta x|$ (km)",
        "$|\\delta y|$ (km)",
        "$|\\delta z|$ (km)",
        "$|\\delta v_x|$ (km/s)",
        "$|\\delta v_y|$ (km/s)",
        "$|\\delta v_z|$ (km/s)",
    ]

    markers = ["o", "<", "^", "D", "v", "s"]
    line_styles = ["-", "--"]
    t_hours = t / 3600.0

    for j, (label, y_fit) in enumerate(traj_dict.items()):
        diff = np.abs(y_poly - y_fit)
        for i in range(6):
            axs[i].semilogy(
                t_hours[1:],
                diff[i, 1:],
                color="black",
                marker=markers[j % len(markers)],
                linestyle=line_styles[j % len(line_styles)],
                markersize=4,
                label=label if i == 0 else None,
            )
            axs[i].set_ylabel(labels[i], fontsize=12)
            axs[i].grid(True, linestyle="--", which="both", linewidth=0.7, alpha=0.8)
            axs[i].minorticks_on()  # Enable minor ticks
            axs[i].grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    for ax in axs:
        ax.set_xlabel("Time (hours)", fontsize=12)
        ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    plt.savefig(
        "Images/trajectory_differences_semilogy.pdf", dpi=1200, bbox_inches="tight"
    )
    plt.show()


# === RUN BOTH MODELS ===
print("Propagating trajectories...")
print("  - Fitted 25x25 Model")
t_fitted_25, y_fitted_25 = fit_and_propagate_model(
    25, 25, initial_position, initial_velocity, t_span
)
print("  - Fitted 5x5 Model")
t_fitted_5, y_fitted_5 = fit_and_propagate_model(
    5, 5, initial_position, initial_velocity, t_span
)
print("  - Poly Model")
t_poly, y_poly = propagate_trajectory(
    initial_position, initial_velocity, acceleration_poly, t_span
)

plot_trajectories_multiple(
    t_poly,
    y_poly,
    {
        "Interior Cylindrical Harmonics (25x25)": y_fitted_25,
        "Interior Cylindrical Harmonics (5x5)": y_fitted_5,
    },
)

plot_trajectory_differences(t_poly, y_poly, {"25x25": y_fitted_25, "5x5": y_fitted_5})
