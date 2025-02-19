import numpy as np
from scipy.special import jv, jvp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


def f_prime(l, z, rho, m):
    """Compute the first derivative of f(l) = exp(-lz) J_m(l * rho)."""
    exp_term = np.exp(-l * z)
    bessel = jv(m, l * rho)
    bessel_prime = jvp(m, l * rho)
    return -z * exp_term * bessel + 0.5 * exp_term * rho * bessel_prime


def find_max_z_rho_f_prime(l, z_rho_values, m):
    """Find the z/rho value that maximizes f'(l)."""
    f_prime_vals = []
    for z_rho in z_rho_values:
        z = z_rho
        rho = 1.0  # Normalize rho to 1 for simplicity
        f_prime_vals.append(f_prime(l, z, rho, m))
    max_f_prime = max(f_prime_vals)
    optimal_z_rho = z_rho_values[np.argmax(f_prime_vals)]
    return optimal_z_rho, max_f_prime


def find_zeros_f_prime(l_range, z, rho, m):
    """Find the zeros of f'(l) for a given z and rho."""
    zeros = []
    l_vals = np.linspace(*l_range, 500)
    for i in range(len(l_vals) - 1):
        if f_prime(l_vals[i], z, rho, m) * f_prime(l_vals[i + 1], z, rho, m) < 0:
            sol = root_scalar(
                f_prime,
                args=(z, rho, m),
                bracket=[l_vals[i], l_vals[i + 1]],
                method="brentq",
            )
            if sol.converged:
                zeros.append(sol.root)
    return zeros


def compute_zero_values(reference_zeros, z_rho_values, m):
    """Compute the value of f'(l) at the zeros for different z/rho values."""
    zero_values = {}
    for z_rho in z_rho_values:
        z = z_rho
        rho = 1.0  # Normalize rho
        zero_values[z_rho] = [f_prime(zero, z, rho, m) for zero in reference_zeros]
    return zero_values


# Parameters
l = 1.0  # Fixed value of l to analyze f'(l) at
m = 1  # Order of the Bessel function
z_rho_values = np.linspace(0.1, 10, 500)  # Range of z/rho
l_range = (0.1, 20)  # Range of l to search for zeros

# Step 1: Find the z/rho that maximizes f'(l)
optimal_z_rho, max_f_prime = find_max_z_rho_f_prime(l, z_rho_values, m)
print(f"Optimal z/rho for max f'(l): {optimal_z_rho}, Max f'(l): {max_f_prime}")

# Step 2: Find the zeros of f'(l) for the optimal z/rho
reference_zeros = find_zeros_f_prime(l_range, optimal_z_rho, 1.0, m)
print(f"Zeros of f'(l) for optimal z/rho: {reference_zeros}")

# Step 3: Compute the values of f'(l) at the reference zeros for different z/rho values
zero_values = compute_zero_values(reference_zeros, z_rho_values, m)

# Visualization of zero values
plt.figure(figsize=(10, 6))
for z_rho, values in zero_values.items():
    plt.plot([z_rho] * len(values), values, "o", label=f"z/rho={z_rho:.2f}", alpha=0.6)

plt.title("Values of f'(l) at Zeros for Different z/rho Values", fontsize=14)
plt.xlabel("z/rho", fontsize=12)
plt.ylabel("f'(l) at Zeros", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
