import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
import mesh_utility

# OSS: integration in R gives decreasing behaviour!

# Load mesh for the polyhedral gravity model
vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
vertices, faces = np.array(vertices), np.array(faces)

# Define asteroid density
DENSITY = 1.0

# Initialize polyhedron object and gravity evaluator
eros = Polyhedron(
    polyhedral_source=(vertices, faces),
    density=DENSITY,
    integrity_check=PolyhedronIntegrity.DISABLE,
)
evaluable_eros = GravityEvaluable(eros)

# Define cylindrical coordinates of evaluation points
CYLINDER_CENTER = np.array([0.0, 0.0, 0.28])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.eye(3)  # Rotation matrix (identity matrix by default)

# Define Bessel-related parameters
ALPHA = 100  # Scaling parameter
L = CYLINDER_HEIGHT
R_star = CYLINDER_RADIUS

# Number of Monte Carlo samples
N_samples = 10**2  # Adjust for accuracy

# Define ranges for n_n and n_m
n_n = 10
n_m = 10
n_n_range = range(1, n_n + 1)
n_m_range = range(0, n_m)

# Storage for results
A_mn_values = np.zeros((len(n_m_range), len(n_n_range)))
B_mn_values = np.zeros((len(n_m_range), len(n_n_range)))

# Iterate over ranges of n_n and n_m
for i, m in enumerate(n_m_range):
    for j, n in enumerate(n_n_range):
        j_mn = sp.jn_zeros(m, n)[-1]
        J_mn_squared = sp.jv(m + 1, j_mn) ** 2

        # Initialize sums for the integral
        A_mn_sum = 0.0
        B_mn_sum = 0.0

        # Monte Carlo integration (point-by-point with tqdm)
        for _ in tqdm(range(N_samples), desc=f"m={m}, n={n}"):
            # Generate a single random sample in cylindrical coordinates
            rho = np.sqrt(np.random.uniform(0, R_star**2))  # Radial coordinate
            phi = np.random.uniform(0, 2 * np.pi)  # Angular coordinate
            z = np.random.uniform(0, L)  # Height coordinate

            # Convert to Cartesian coordinates
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)

            # Transform to global frame
            cartesian_point = (
                np.array([x, y, z]) @ CYLINDER_ROTATION.T + CYLINDER_CENTER
            )

            # Evaluate gravity potential at the single point
            potential_value, _, _ = evaluable_eros(
                computation_points=cartesian_point, parallel=False
            )

            # Compute integrand values
            J_value = sp.jv(m, j_mn * rho / (ALPHA * R_star))
            cos_value = np.cos(m * phi)
            sin_value = np.sin(m * phi)
            exp_value = np.exp(j_mn * z / (ALPHA * R_star))

            A_mn_sum += rho * potential_value * J_value * cos_value * exp_value
            B_mn_sum += rho * potential_value * J_value * sin_value * exp_value

        # Compute Monte Carlo estimate of the integral
        volume = np.pi * (R_star) ** 2 * L  # integration volume
        A_mn_integral = volume * A_mn_sum / N_samples
        B_mn_integral = volume * B_mn_sum / N_samples

        A_mn = (
            2 / (np.pi * L * ((ALPHA * R_star) ** 2) * J_mn_squared)
        ) * A_mn_integral
        B_mn = (
            2 / (np.pi * L * ((ALPHA * R_star) ** 2) * J_mn_squared)
        ) * B_mn_integral

        A_mn_values[i, j] = A_mn
        B_mn_values[i, j] = B_mn
        print(f"A_mn: {A_mn_integral}, B_mn: {B_mn_integral}")

# Save results
np.save("A_coefficients.npy", A_mn_values)
np.save("B_coefficients.npy", B_mn_values)


# Function to plot coefficients in semilogarithmic scale with distinct colors
def plot_coefficients_semilogy(A, B, n_n, n_m):
    plt.figure(figsize=(12, 8))
    colors = {"A": "red", "B": "orange"}

    def scatter_coefficients(coefficients, label, color):
        for m in range(n_m):
            if m >= coefficients.shape[0]:
                continue
            coeffs_m = coefficients[m, :]
            x = np.full_like(coeffs_m, m, dtype=int)
            plt.scatter(
                x,
                np.abs(coeffs_m),
                color=color,
                label=f"{label}" if m == 0 else None,
                alpha=0.7,
                edgecolor="black",
            )

    scatter_coefficients(A, "A", colors["A"])
    scatter_coefficients(B, "B", colors["B"])

    plt.yscale("log")
    plt.xlabel("Order m (-)", labelpad=10)
    plt.ylabel("Coefficient Magnitude (-)", labelpad=10)
    plt.grid(True, linestyle="--", which="both", linewidth=0.7, alpha=0.8)
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.legend(loc="best", frameon=True, fancybox=True, edgecolor="black", fontsize=14)
    plt.show()


plot_coefficients_semilogy(A_mn_values, B_mn_values, len(n_n_range), len(n_m_range))


# Function to sample the potential at given points
def sample_potential(N_samples):
    sampled_points = []
    sampled_potentials = []

    for _ in range(N_samples):
        rho = np.sqrt(np.random.uniform(0, R_star) ** 2)
        phi = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(0, L)

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        cartesian_point = np.array([x, y, z]) @ CYLINDER_ROTATION.T + CYLINDER_CENTER
        potential_value, _, _ = evaluable_eros(
            computation_points=cartesian_point, parallel=False
        )

        sampled_points.append([rho, phi, z])
        sampled_potentials.append(potential_value)

    return np.array(sampled_points), np.array(sampled_potentials)


# Function to reconstruct potential from computed coefficients
def reconstruct_potential(points, A_mn_values, B_mn_values, n_n_range, n_m_range):
    reconstructed_potentials = []

    for rho, phi, z in points:
        potential_reconstructed = 0.0

        for i, m in enumerate(n_m_range):
            for j, n in enumerate(n_n_range):
                if i >= A_mn_values.shape[0] or j >= A_mn_values.shape[1]:
                    continue

                j_mn = sp.jn_zeros(m, n)[-1]
                J_value = sp.jv(m, j_mn * rho / (ALPHA * R_star))
                cos_value = np.cos(m * phi)
                sin_value = np.sin(m * phi)
                exp_value = np.exp(-j_mn * z / (ALPHA * R_star))

                potential_reconstructed += (
                    A_mn_values[i, j] * J_value * cos_value * exp_value
                    + B_mn_values[i, j] * J_value * sin_value * exp_value
                )

        reconstructed_potentials.append(potential_reconstructed)

    return np.array(reconstructed_potentials)


# Compute error between sampled and reconstructed potential
sampled_points, sampled_potentials = sample_potential(N_samples)
reconstructed_potentials = reconstruct_potential(
    sampled_points, A_mn_values, B_mn_values, n_n_range, n_m_range
)

error = np.abs(sampled_potentials - reconstructed_potentials)
relative_error_percentage = error / np.abs(sampled_potentials) * 100

# Compute statistics
mean_absolute_error = np.mean(error)
std_absolute_error = np.std(error)

mean_relative_error_percentage = np.mean(relative_error_percentage)
std_relative_error_percentage = np.std(relative_error_percentage)

# Print error statistics
print(f"Mean Absolute Error: {mean_absolute_error}")
print(f"Standard Deviation of Absolute Error: {std_absolute_error}")
print(f"Mean Relative Percentage Error: {mean_relative_error_percentage} %")
print(
    f"Standard Deviation of Relative Percentage Error: {std_relative_error_percentage} %"
)


# Plot histogram of absolute errors
plt.figure(figsize=(10, 6))
plt.hist(error, bins=30, color="skyblue", edgecolor="black")
plt.title("Histogram of Absolute Errors in Reconstructed Potential")
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# Optionally, you can also plot a histogram of the relative percentage errors
plt.figure(figsize=(10, 6))
plt.hist(relative_error_percentage, bins=30, color="salmon", edgecolor="black")
plt.title("Histogram of Relative Percentage Errors")
plt.xlabel("Relative Error (%)")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
