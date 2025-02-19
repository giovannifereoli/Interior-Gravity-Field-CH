#########################################################################################################################
# Small Body Characterization - Cylinder Gravity LS Fitting
# Author: Giovanni Fereoli (The University of Coloradto at Boulder)
# Advisor: Dr. McMahon (The University of Colorado at Boulder)
# Acknowledgement: None
# Date: 2021-09-30
#
# TODO: 1) Fix units 2) Fix how surrogate takes in cartesian
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
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.special import jv as BesselJ, iv as BesselI, jn_zeros
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl

# Use a colorblind-friendly color palette
color_palette = ["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=color_palette)

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


# Define surrogate model in cylindrical coordinates
def surrogate_model(params, points, n_n, n_m):
    R, L = CYLINDER_RADIUS, CYLINDER_HEIGHT
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
    z = transformed_points[:, 2]

    # Unpack parameters
    A, B, C, D = (
        params[: n_m * n_n],
        params[n_m * n_n : 2 * n_m * n_n],
        params[2 * n_m * n_n : 3 * n_m * n_n],
        params[3 * n_m * n_n :],
    )

    surrogate_potential = np.zeros_like(rho)
    k = lambda m, n: jn_zeros(m, n)[
        -1
    ]  # k_mn are the zeros of the Bessel function of the m-th order

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            k_mn = k(m, n)
            surrogate_potential += np.exp(-k_mn / R * z) * BesselJ(
                m, k_mn / R * rho
            ) * (A[idx] * np.cos(m * phi) + B[idx] * np.sin(m * phi)) + np.sin(
                n * np.pi / L * z
            ) * BesselI(
                m, n * np.pi / L * rho
            ) * (
                C[idx] * np.cos(m * phi) + D[idx] * np.sin(m * phi)
            )
            idx += 1
    return surrogate_potential


# Define residuals for least squares fitting
def residuals(params, points, potentials, n_n, n_m):
    return surrogate_model(params, points, n_n, n_m) - potentials


# Fit surrogate model
n_n, n_m = 5, 6  # Truncation parameters
num_params = 4 * n_n * n_m
initial_guess = np.random.rand(num_params)
points = structured_results["points"]
potentials = structured_results["potential"]
result = least_squares(
    residuals,  # Function to minimize
    initial_guess,  # Initial parameter guess
    args=(points, potentials, n_n, n_m),  # Additional arguments to `residuals`
    verbose=2,  # Level of verbosity (0: None, 1: Summary, 2: Detailed)
    method="trf",  # Optimization algorithm ('trf', 'dogbox', 'lm')
    ftol=1e-14,  # Tolerance for termination by change in cost
    xtol=1e-14,  # Tolerance for termination by change in parameters
    gtol=1e-14,  # Tolerance for termination by gradient norm
    max_nfev=10000,  # Maximum number of function evaluations
    loss="linear",  # Loss function for robust fitting ('linear', 'soft_l1', 'huber', etc.)
    f_scale=1.0,  # Parameter for robust loss functions
    jac="3-point",  # Method for computing Jacobian ('2-point', '3-point', 'cs', callable)
    bounds=(-10, 10),  # Lower and upper bounds for parameters
)

# Extract fitted parameters
fitted_params = result.x
print("Fitted parameters:", fitted_params)

# Calculate percentage error
fitted_potentials = surrogate_model(fitted_params, points, n_n, n_m)
percentage_error = 100 * np.abs((fitted_potentials - potentials) / potentials)
print("Percentage Error:", percentage_error)


##  Print coefficients with m and n indexing
def print_fitted_parameters(fitted_params, n_n, n_m):
    A = fitted_params[: n_m * n_n]
    B = fitted_params[n_m * n_n : 2 * n_m * n_n]
    C = fitted_params[2 * n_m * n_n : 3 * n_m * n_n]
    D = fitted_params[3 * n_m * n_n :]

    print("Fitted Parameters:")
    print("Coefficients A:")
    for m in range(n_m):
        for n in range(1, n_n + 1):
            idx = m * n_n + (n - 1)
            print(f"  A[m={m}, n={n}] = {A[idx]:.6f}")

    print("Coefficients B:")
    for m in range(n_m):
        for n in range(1, n_n + 1):
            idx = m * n_n + (n - 1)
            print(f"  B[m={m}, n={n}] = {B[idx]:.6f}")

    print("Coefficients C:")
    for m in range(n_m):
        for n in range(1, n_n + 1):
            idx = m * n_n + (n - 1)
            print(f"  C[m={m}, n={n}] = {C[idx]:.6f}")

    print("Coefficients D:")
    for m in range(n_m):
        for n in range(1, n_n + 1):
            idx = m * n_n + (n - 1)
            print(f"  D[m={m}, n={n}] = {D[idx]:.6f}")


# Call the function to print coefficients
print_fitted_parameters(fitted_params, n_n, n_m)

## Plot percentage error


def plot_histogram_with_gaussian(percentage_error):
    plt.figure(figsize=(10, 6))

    # Plot histogram
    n, bins, patches = plt.hist(
        percentage_error,
        bins=50,
        color=color_palette[-1],
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
        color=color_palette[0],
        linestyle="--",
        linewidth=2,
        label=rf"Gaussian Fit: $\mu={mu:.2f}, \sigma={std:.2f}$",
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
        facecolor=color_palette[-1],
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
    plt.title("Percentage Error on Cylinder Points")
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
        facecolor=color_palette[-1],
    )
    ax.add_collection3d(mesh)

    # Plot cylinder points
    ax.scatter(
        cylinder_points[:, 0],
        cylinder_points[:, 1],
        cylinder_points[:, 2],
        c=color_palette[0],
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


## Extra code for Jacobian, gradient, and Hessian computation

"""

# Define Jacobian of the surrogate model
def surrogate_jacobian(params, points, n_n, n_m):
    R, L = CYLINDER_RADIUS, CYLINDER_HEIGHT
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
    z = transformed_points[:, 2]

    # Initialize Jacobian matrix
    jacobian = np.zeros((len(points), len(params)))

    k = lambda m, n: jn_zeros(m, n)[
        -1
    ]  # k_mn are the zeros of the Bessel function of the m-th order

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            k_mn = k(m, n)

            # Derivatives with respect to A, B, C, D
            exp_term = np.exp(-k_mn / R * z)
            sin_term = np.sin(n * np.pi / L * z)

            jacobian[:, idx] = exp_term * BesselJ(m, k_mn / R * rho) * np.cos(m * phi)
            jacobian[:, idx + 1] = (
                exp_term * BesselJ(m, k_mn / R * rho) * np.sin(m * phi)
            )
            jacobian[:, idx + 2] = (
                sin_term * BesselI(m, n * np.pi / L * rho) * np.cos(m * phi)
            )
            jacobian[:, idx + 3] = (
                sin_term * BesselI(m, n * np.pi / L * rho) * np.sin(m * phi)
            )

            idx += 4

    return jacobian

# Compute gradient acceleration of the surrogate model
def surrogate_gradient(params, points, n_n, n_m):
    R, L = CYLINDER_RADIUS, CYLINDER_HEIGHT
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0]**2 + transformed_points[:, 1]**2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
    z = transformed_points[:, 2]

    # Unpack parameters
    A, B, C, D = params[:n_m * n_n], params[n_m * n_n:2 * n_m * n_n], params[2 * n_m * n_n:3 * n_m * n_n], params[3 * n_m * n_n:]

    grad_rho = np.zeros_like(rho)
    grad_phi = np.zeros_like(phi)
    grad_z = np.zeros_like(z)

    k = lambda m, n: jn_zeros(m, n)[-1]

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            k_mn = k(m, n)
            common_exp = np.exp(-k_mn / R * z)
            common_sin = np.sin(n * np.pi / L * z)

            grad_rho += (
                common_exp * (k_mn / R) * BesselJ(m, k_mn / R * rho, derivative=True) * (A[idx] * np.cos(m * phi) + B[idx] * np.sin(m * phi)) +
                common_sin * (n * np.pi / L) * BesselI(m, n * np.pi / L * rho, derivative=True) * (C[idx] * np.cos(m * phi) + D[idx] * np.sin(m * phi))
            )

            grad_phi += (
                common_exp * BesselJ(m, k_mn / R * rho) * (-m * A[idx] * np.sin(m * phi) + m * B[idx] * np.cos(m * phi)) +
                common_sin * BesselI(m, n * np.pi / L * rho) * (-m * C[idx] * np.sin(m * phi) + m * D[idx] * np.cos(m * phi))
            )

            grad_z += (
                -common_exp * (k_mn / R) * BesselJ(m, k_mn / R * rho) * (A[idx] * np.cos(m * phi) + B[idx] * np.sin(m * phi)) +
                n * np.pi / L * np.cos(n * np.pi / L * z) * BesselI(m, n * np.pi / L * rho) * (C[idx] * np.cos(m * phi) + D[idx] * np.sin(m * phi))
            )

            idx += 1

    grad_x = grad_rho * np.cos(phi) - grad_phi * np.sin(phi) / rho
    grad_y = grad_rho * np.sin(phi) + grad_phi * np.cos(phi) / rho

    return np.column_stack((grad_x, grad_y, grad_z))
        
    # Compute Hessian of the surrogate model
    def surrogate_hessian(params, points, n_n, n_m):
        R, L = CYLINDER_RADIUS, CYLINDER_HEIGHT
        transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
        rho = np.sqrt(transformed_points[:, 0]**2 + transformed_points[:, 1]**2)
        phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
        z = transformed_points[:, 2]

        # Unpack parameters
        A, B, C, D = params[:n_m * n_n], params[n_m * n_n:2 * n_m * n_n], params[2 * n_m * n_n:3 * n_m * n_n], params[3 * n_m * n_n:]

        hessian_rho_rho = np.zeros_like(rho)
        hessian_phi_phi = np.zeros_like(phi)
        hessian_z_z = np.zeros_like(z)
        hessian_rho_phi = np.zeros_like(rho)
        hessian_rho_z = np.zeros_like(rho)
        hessian_phi_z = np.zeros_like(rho)

        k = lambda m, n: jn_zeros(m, n)[-1]

        idx = 0
        for m in range(n_m):
            for n in range(1, n_n + 1):
                k_mn = k(m, n)
                common_exp = np.exp(-k_mn / R * z)
                common_sin = np.sin(n * np.pi / L * z)

                # Hessians with respect to rho
                hessian_rho_rho += (
                    common_exp * (k_mn / R)**2 * BesselJ(m, k_mn / R * rho, derivative=2) * 
                    (A[idx] * np.cos(m * phi) + B[idx] * np.sin(m * phi)) +
                    common_sin * (n * np.pi / L)**2 * BesselI(m, n * np.pi / L * rho, derivative=2) * 
                    (C[idx] * np.cos(m * phi) + D[idx] * np.sin(m * phi))
                )

                # Hessians with respect to phi
                hessian_phi_phi += (
                    common_exp * BesselJ(m, k_mn / R * rho) * (-m**2 * A[idx] * np.cos(m * phi) - m**2 * B[idx] * np.sin(m * phi)) +
                    common_sin * BesselI(m, n * np.pi / L * rho) * (-m**2 * C[idx] * np.cos(m * phi) - m**2 * D[idx] * np.sin(m * phi))
                )

                # Hessians with respect to z
                hessian_z_z += (
                    common_exp * (-k_mn / R)**2 * BesselJ(m, k_mn / R * rho) * 
                    (A[idx] * np.cos(m * phi) + B[idx] * np.sin(m * phi)) +
                    -(n * np.pi / L)**2 * np.sin(n * np.pi / L * z) * BesselI(m, n * np.pi / L * rho) * 
                    (C[idx] * np.cos(m * phi) + D[idx] * np.sin(m * phi))
                )

                idx += 1

        return {
            "rho_rho": hessian_rho_rho,
            "phi_phi": hessian_phi_phi,
            "z_z": hessian_z_z,
            "rho_phi": hessian_rho_phi,
            "rho_z": hessian_rho_z,
            "phi_z": hessian_phi_z,
        }
    """
