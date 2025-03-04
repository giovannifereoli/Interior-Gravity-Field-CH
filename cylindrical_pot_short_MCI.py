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
from scipy.special import jv as BesselJ, jn_zeros
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.integrate as integrate
import matplotlib as mpl

# TODO: integral should be within o and alpha * R
# TODO: you're using monte carlo integration ....

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
CYLINDER_CENTER = np.array([0.0, 0.0, 0.28])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
ALPHA = 100  # scaling parameter
CYLINDER_ROTATION = np.eye(3)  # Rotation matrix (identity matrix by default)


def random_rotation_matrix():
    # Generate a random 3x3 matrix.
    A = np.random.randn(3, 3)
    # Perform QR decomposition on A.
    Q, R = np.linalg.qr(A)
    # Ensure that Q is a proper rotation matrix with det(Q)=1.
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# CYLINDER_ROTATION = random_rotation_matrix()

# CYLINDER_ROTATION = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

NUM_POINTS = 1000  # Number of points to generate


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


# Modified random point generator for the extended_lid domain:
def generate_points_in_cylinder_extended_lid(
    center, radius, height, rotation, num_points
):
    np.random.seed(1)  # For reproducibility
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    # Sample radius from 0 to ALPHA * radius
    r = np.sqrt(np.random.uniform(0, (ALPHA * radius) ** 2, num_points))
    z = CYLINDER_HEIGHT + 0 * np.random.uniform(0, 2 * np.pi, num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    local_points = np.column_stack((x, y, z))
    rotated_points = local_points @ rotation.T
    translated_points = rotated_points + center
    return translated_points


# Use the extended_lid generator for Monte Carlo integration:
extended_lid_cylinder_points = generate_points_in_cylinder_extended_lid(
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

# Evaluate gravity at each point
results_extended_lid = []
for point in tqdm(extended_lid_cylinder_points, desc="Evaluating gravity at points"):
    potential, acceleration, tensor = evaluable_eros(
        computation_points=point, parallel=False
    )
    results_extended_lid.append(
        {
            "point": point,
            "potential": potential,
            "acceleration": acceleration,
            "tensor": tensor,
        }
    )

# Convert results to a structured numpy array for easier processing
structured_extended_lid_results = {
    "points": np.array([res["point"] for res in results_extended_lid]),
    "potential": np.array([res["potential"] for res in results_extended_lid]),
    "acceleration": np.array([res["acceleration"] for res in results_extended_lid]),
    "tensor": np.array([res["tensor"] for res in results_extended_lid]),
}


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
n_n, n_m = 20, 20  # Truncation parameters
num_params = 2 * n_n * n_m  # Updated for two coefficients per term (A and B)
points = structured_results["points"]
potentials = structured_results["potential"]
A_mat, b = prepare_linear_system_new(
    points, potentials, n_n, n_m
)  # Using the updated function


# Monte Carlo Integration for Coefficients
def monte_carlo_integration(n_n, n_m, points, potentials):
    A_mn, B_mn = np.zeros((n_m, n_n)), np.zeros((n_m, n_n))
    volume = np.pi * CYLINDER_RADIUS**2 * CYLINDER_HEIGHT
    k = lambda m, n: jn_zeros(m, n)[-1]

    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho, phi, z = (
        np.linalg.norm(transformed_points[:, :2], axis=1),
        np.arctan2(transformed_points[:, 1], transformed_points[:, 0]),
        transformed_points[:, 2],
    )

    for m in tqdm(range(n_m), desc="Monte Carlo Integration: m-loop"):
        for n in range(1, n_n + 1):
            k_mn = k(m, n)
            J_mn_squared = BesselJ(m + 1, k_mn) ** 2
            R_alpha = ALPHA * CYLINDER_RADIUS

            J_values = BesselJ(m, k_mn * rho / (R_alpha))
            exp_values = np.exp(k_mn * z / (R_alpha))
            cos_values, sin_values = np.cos(m * phi), np.sin(m * phi)

            A_mn_integral = (
                volume
                * np.sum(rho * potentials * J_values * cos_values * exp_values)
                / len(points)
            )
            B_mn_integral = (
                volume
                * np.sum(rho * potentials * J_values * sin_values * exp_values)
                / len(points)
            )

            # Angular normalization: 2π for m = 0, π for m >= 1.
            THETA_COS = 2 * np.pi if m == 0 else np.pi
            THETA_SIN = np.inf if m == 0 else np.pi

            A_mn[m, n - 1] = (
                2
                / (
                    THETA_COS
                    * CYLINDER_HEIGHT
                    * (ALPHA * CYLINDER_RADIUS) ** 2
                    * J_mn_squared
                )
            ) * A_mn_integral
            B_mn[m, n - 1] = (
                2
                / (
                    THETA_SIN
                    * CYLINDER_HEIGHT
                    * (ALPHA * CYLINDER_RADIUS) ** 2
                    * J_mn_squared
                )
            ) * B_mn_integral

    return A_mn, B_mn


# Efficient nquad-based coefficient computation
def compute_coefficients_nquad(n_n, n_m, unused_points, unused_potentials):
    A_mn, B_mn = np.zeros((n_m, n_n)), np.zeros((n_m, n_n))
    total_evals = n_m * n_n
    progress_bar = tqdm(total=total_evals, desc="nquad Integration", unit="evals")

    def Phi_alpha(rho, phi, z):
        """Evaluates the gravitational potential at cylindrical coordinates (rho, phi, z)."""
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        cartesian_point = np.array([x, y, z]) @ CYLINDER_ROTATION + CYLINDER_CENTER
        potential, _, _ = evaluable_eros(
            computation_points=cartesian_point, parallel=False
        )
        return potential

    def integrand_A(rho, phi, z, m, n):
        k_mn = jn_zeros(m, n)[-1]
        R_alpha = ALPHA * CYLINDER_RADIUS
        return (
            rho
            * Phi_alpha(rho, phi, z)
            * BesselJ(m, k_mn * rho / R_alpha)
            * np.cos(m * phi)
            * np.exp(k_mn * z / R_alpha)
        )

    def integrand_B(rho, phi, z, m, n):
        k_mn = jn_zeros(m, n)[-1]
        R_alpha = ALPHA * CYLINDER_RADIUS
        return (
            rho
            * Phi_alpha(rho, phi, z)
            * BesselJ(m, k_mn * rho / R_alpha)
            * np.sin(m * phi)
            * np.exp(k_mn * z / R_alpha)
        )

    for m in range(n_m):
        for n in range(1, n_n + 1):
            k_mn = jn_zeros(m, n)[-1]
            J_mn_squared = BesselJ(m + 1, k_mn) ** 2
            R_alpha = ALPHA * CYLINDER_RADIUS
            A_mn_integral, _ = integrate.nquad(
                integrand_A,
                [[0, CYLINDER_RADIUS], [0, 2 * np.pi], [0, CYLINDER_HEIGHT]],
                args=(m, n),
            )
            B_mn_integral, _ = integrate.nquad(
                integrand_B,
                [[0, CYLINDER_RADIUS], [0, 2 * np.pi], [0, CYLINDER_HEIGHT]],
                args=(m, n),
            )

            # Angular normalization: 2π for m = 0, π for m >= 1.
            THETA_COS = 2 * np.pi if m == 0 else np.pi
            THETA_SIN = np.inf if m == 0 else np.pi

            A_mn[m, n - 1] = (
                2 / (THETA_COS * CYLINDER_HEIGHT * (R_alpha) ** 2 * J_mn_squared)
            ) * A_mn_integral
            B_mn[m, n - 1] = (
                2 / (THETA_SIN * CYLINDER_HEIGHT * (R_alpha) ** 2 * J_mn_squared)
            ) * B_mn_integral
            progress_bar.update(1)

    progress_bar.close()
    return A_mn, B_mn


def monte_carlo_integration_surface(n_n, n_m, points, potentials):
    # Allocate coefficient matrices.
    A_mn, B_mn = np.zeros((n_m, n_n)), np.zeros((n_m, n_n))

    # Integration over the top surface (area of a circle)
    area = np.pi * (ALPHA * CYLINDER_RADIUS) ** 2
    k = lambda m, n: jn_zeros(m, n)[-1]

    # Transform points to cylindrical coordinates.
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.linalg.norm(transformed_points[:, :2], axis=1)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])

    # Loop over the (m,n) modes.
    for m in tqdm(range(n_m), desc="Monte Carlo Integration: m-loop"):
        for n in range(1, n_n + 1):
            k_mn = k(m, n)
            R_alpha = ALPHA * CYLINDER_RADIUS
            J_mn_squared = BesselJ(m + 1, k_mn) ** 2

            # Evaluate the radial part of the basis function.
            J_values = BesselJ(m, k_mn * rho / R_alpha)
            # Compensate for the forward model’s exp(k_mn*L/R_alpha)
            comp_factor = np.exp(k_mn * CYLINDER_HEIGHT / R_alpha)
            cos_values = np.cos(m * phi)
            sin_values = np.sin(m * phi)

            # Monte Carlo estimate of the projection integrals over the disk.
            A_mn_integral = (
                area * np.sum(rho * potentials * J_values * cos_values) / len(points)
            )
            B_mn_integral = (
                area * np.sum(rho * potentials * J_values * sin_values) / len(points)
            )

            # Angular normalization: 2π for m = 0, π for m >= 1.
            THETA_COS = 2 * np.pi if m == 0 else np.pi
            THETA_SIN = np.inf if m == 0 else np.pi

            # Normalize the coefficients using the Bessel orthogonality relation.
            A_mn[m, n - 1] = (
                2
                * comp_factor
                / (THETA_COS * (ALPHA * CYLINDER_RADIUS) ** 2 * J_mn_squared)
            ) * A_mn_integral
            B_mn[m, n - 1] = (
                2
                * comp_factor
                / (THETA_SIN * (ALPHA * CYLINDER_RADIUS) ** 2 * J_mn_squared)
            ) * B_mn_integral

    return A_mn, B_mn


A_mn, B_mn = compute_coefficients_nquad(
    n_m,
    n_n,
    structured_results["points"],
    structured_results["potential"],
)


"""
A_mn, B_mn = monte_carlo_integration_surface(
    n_m,
    n_n,
    structured_extended_lid_results["points"],
    structured_extended_lid_results["potential"],
)"""


def reconstruct_fitted_params(A, B):
    """
    Reconstruct the flattened fitted parameters array from A and B coefficient matrices,
    ensuring the correct order consistent with the least squares system.

    Args:
        A: 2D array of A_mn coefficients (shape: n_m x n_n).
        B: 2D array of B_mn coefficients (shape: n_m x n_n).

    Returns:
        fitted_params: Flattened 1D array of fitted coefficients in correct order.
    """
    n_m, n_n = A.shape
    fitted_params = np.zeros(2 * n_m * n_n)

    idx = 0
    for m in range(n_m):
        for n in range(
            1, n_n + 1
        ):  # Maintain consistency with indexing in least squares
            fitted_params[idx] = A[m, n - 1]  # A_mn coefficient
            fitted_params[idx + 1] = B[m, n - 1]  # B_mn coefficient
            idx += 2

    return fitted_params


# Calculate percentage error
fitted_params = reconstruct_fitted_params(A_mn, B_mn)
fitted_potentials = 100 * A_mat @ fitted_params
percentage_error = 100 * np.abs((fitted_potentials - b) / b)
print("Percentage Error:", percentage_error)


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
plot_coefficients_semilogy(A_mn, B_mn, n_n, n_m)
plot_coefficient_matrices(A_mn, B_mn)

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
