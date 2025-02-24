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
CYLINDER_ROTATION = np.eye(3)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate
ALPHA = 100  # Scaling parameter for the cylinder

# Define cylinder parameters (2)
"""
CYLINDER_CENTER = np.array([0.0, -0.3, 0])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.08  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate
ALPHA = 100  # Scaling parameter for the cylinder
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

np.save("fitted_params_both.npy", fitted_params)
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


def plot_histogram_with_lognormal(data, title="Log-Normal Fit"):
    plt.figure(figsize=(12, 8))

    # Plot histogram
    n, bins, patches = plt.hist(
        data,
        bins=50,
        color="#abd9e9",  # Example color
        alpha=0.7,
        edgecolor="black",
        density=True,
        label="Data Histogram",
    )

    # Fit a Log-Normal distribution
    shape, loc, scale = lognorm.fit(data, floc=0)  # Fix loc=0 for positive data

    # Generate Log-Normal PDF
    x = np.linspace(min(data), max(data), 1000)
    pdf = lognorm.pdf(x, shape, loc, scale)

    # Plot the Log-Normal curve
    plt.plot(
        x,
        pdf,
        color="#2c7bb6",
        linestyle="--",
        linewidth=2,
        label=rf"Log-Normal Fit: $\mu={np.log(scale):.6f}, \sigma={shape:.6f}$",
    )

    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel("Value", labelpad=10)
    plt.ylabel("Frequency Density", labelpad=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Compute equivalent parameters in log space
    mu = np.log(scale)
    sigma = shape

    # Compute and display parameters
    original_mean = np.exp(mu + (sigma**2) / 2)
    original_variance = np.exp(2 * mu + sigma**2) * (np.exp(sigma**2) - 1)
    original_std = np.sqrt(original_variance)

    print(f"Log-Normal Parameters (Log Space): μ = {mu:.6f}, σ = {sigma:.6f}")
    print(
        f"Original Space Parameters: Mean = {original_mean:.6f}, Std Dev = {original_std:.6f}"
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
        cmap="viridis",
        s=5,
        label="Percentage Error",
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
plot_histogram_with_gaussian(percentage_acceleration_error, title="Acceleration Error")
plot_histogram_with_lognormal(
    percentage_acceleration_error, title="Acceleration Error - LogNormal"
)


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
        rtol=1e-6,
        atol=1e-6,
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
        color="#d7191c",
    )

    # Plot fitted trajectory
    ax.plot(
        y_fitted[0, :],
        y_fitted[1, :],
        y_fitted[2, :],
        label="Interior Bessel Cylindrical Harmonics",
        linewidth=2,
        color="#2c7bb6",
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

    # Add labels and legend
    ax.set_xlabel("$X$ (km)")
    ax.set_ylabel("$Y$ (km)")
    ax.set_zlabel("$Z$ (km)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")


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
""" Final State:
        -5.45118663e-02,
        -6.08104828e-02,
        7.29726385e-01,
        9.74202292e-07,
        1.09203903e-06,
        -7.28180036e-06,"""
initial_position = np.array([0.0, 0.0, 0.27])  # Adjust as needed
initial_velocity = np.array([1e-6, 1e-6, -0.00001])  # Adjust as needed
t_span = np.linspace(0, -55000, 100)

# Propagate trajectories using SciPy's ODE solver
print("Propagating trajectories...")
print("  - Fitted Model")
t_fitted, y_fitted = propagate_trajectory(
    initial_position, initial_velocity, acceleration_fitted, t_span
)
print(" Done with fitted model.")
print("  - Poly Model")
t_poly, y_poly = propagate_trajectory(
    initial_position, initial_velocity, acceleration_poly, t_span
)
print(" Done with constant-density polyhedron model.")

# Plot trajectories
plot_trajectories(t_poly, y_poly, t_fitted, y_fitted)


# Compute differences between constant-density polyhedron and fitted models
def plot_differences_semilogy(t, y_poly, y_fitted):
    """
    Plot semi-logarithmic differences between y_poly and y_fitted for each state variable.

    Args:
        t: Time array.
        y_poly: State array for the constant-density polyhedron model (N, 6).
        y_fitted: State array for the fitted model (N, 6).
    """
    # Mirror the x-axis if time starts from negative
    if t[0] > t[-1]:
        t = -t[::-1]
        y_poly = y_poly[:, ::-1]
        y_fitted = y_fitted[:, ::-1]

    differences = np.abs(y_poly - y_fitted)

    labels = ["$x$", "$y$", "$z$", "$v_x$", "$v_y$", "$v_z$"]

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    axs = axs.ravel()  # Flatten the axes array for easier indexing

    for i in range(6):
        axs[i].semilogy(t[1:], differences[i, 1:], label=f"Difference in {labels[i]}")
        axs[i].set_xlabel("Time (s)", fontsize=12)
        axs[i].set_ylabel("Absolute Difference (-)", fontsize=12)
        axs[i].grid(True, which="both", linestyle="--", alpha=0.7)
        axs[i].legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])


# Plot differences in trajectories
plot_differences_semilogy(t_poly, y_poly, y_fitted)


# Show plots
plt.show()


## Analayze Contributions


def compute_potential_by_harmonic_type(points, fitted_params, n_n, n_m, harmonic_type):
    """
    Compute the potential using only specified harmonic types.

    Args:
        points: Cartesian points (N, 3).
        fitted_params: Fitted coefficients (1D array).
        n_n, n_m: Truncation parameters.
        harmonic_type: 'Axial', 'AzimuthalSym', or 'AzimuthalASym'.

    Returns:
        Potential at each point (N,).
    """
    R_alpha = ALPHA * CYLINDER_RADIUS
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
    z = transformed_points[:, 2]

    num_points = len(points)
    potential = np.zeros(num_points)

    k = lambda m, n: jn_zeros(m, n)[-1]

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            k_mn = k(m, n)

            # Cylindrical basis function components
            exp_term = np.exp(-k_mn * z / R_alpha)
            bessel_j = BesselJ(m, k_mn * rho / R_alpha)
            cos_m_phi = np.cos(m * phi)
            sin_m_phi = np.sin(m * phi)

            # Filter contributions based on harmonic type
            if harmonic_type == "Axial" and m == 0:
                potential += exp_term * bessel_j * fitted_params[idx]
            elif harmonic_type == "AzimuthalSym" and m == n:
                potential += (
                    exp_term
                    * bessel_j
                    * (
                        fitted_params[idx] * cos_m_phi
                        + fitted_params[idx + 1] * sin_m_phi
                    )
                )
            elif harmonic_type == "AzimuthalASym" and m != 0 and m != n:
                potential += (
                    exp_term
                    * bessel_j
                    * (
                        fitted_params[idx] * cos_m_phi
                        + fitted_params[idx + 1] * sin_m_phi
                    )
                )

            idx += 2

    return potential


def compute_potential_error_contributions(
    points, fitted_params, n_n, n_m, ground_truth_potentials
):
    """
    Compute percentage error for Axial, AzimuthalSym, and AzimuthalASym contributions.

    Args:
        points: Cartesian points (N, 3).
        fitted_params: Fitted coefficients (1D array).
        n_n, n_m: Truncation parameters.
        ground_truth_potentials: Ground-truth gravitational potentials (N,).

    Returns:
        errors: Dictionary with percentage errors for 'Axial', 'AzimuthalSym', and 'AzimuthalASym'.
    """
    errors = {}

    for harmonic_type in ["Axial", "AzimuthalSym", "AzimuthalASym"]:
        # Compute potential for the harmonic type
        potential_harmonic = compute_potential_by_harmonic_type(
            points, fitted_params, n_n, n_m, harmonic_type
        )

        # Calculate percentage error
        error = np.abs(potential_harmonic - ground_truth_potentials) / np.abs(
            ground_truth_potentials
        )
        errors[harmonic_type] = 100 * error  # Convert to percentage

    return errors


def plot_potential_error_cylinder(points, errors):
    """
    Plot potential error cylinders for Axial, AzimuthalSym, and AzimuthalASym contributions.

    Args:
        points: Cartesian points (N, 3).
        errors: Dictionary with percentage errors for each harmonic type.
    """
    fig = plt.figure(figsize=(18, 6))

    for i, (harmonic_type, error) in enumerate(errors.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        mesh = Poly3DCollection(
            vertices[faces],
            alpha=0.8,
            edgecolor="k",
            linewidths=0.3,
            facecolor=COLOR_PALETTE[-1],
        )
        ax.add_collection3d(mesh)
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=error,
            cmap="viridis",
            s=5,
            alpha=0.8,
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Percentage Error (-)")
        ax.set_title(f"{harmonic_type.capitalize()} Harmonics", fontsize=14)
        ax.set_xlabel("$X$ (m)")
        ax.set_ylabel("$Y$ (m)")
        ax.set_zlabel("$Z$ (m)")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


# Compute potential errors for each harmonic type
potential_errors = compute_potential_error_contributions(
    cylinder_points, fitted_params, n_n, n_m, structured_results["potential"]
)

# Plot the potential error cylinders
plot_potential_error_cylinder(cylinder_points, potential_errors)

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
