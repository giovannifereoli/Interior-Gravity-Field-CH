import numpy as np
from scipy.special import jv, jvp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


def f_prime(l, z, rho, m):
    """Compute the first derivative of f(l) = exp(-lz) J_m(l * rho)."""
    exp_term = np.exp(-l * z)
    bessel = jv(m, l * rho)
    bessel_prime = jvp(m, l * rho)
    return -z * exp_term * bessel + exp_term * rho * bessel_prime


def find_zeros_f_prime_general(l_range, z_rho_values, m):
    """Find the zeros of f'(l) for all z/rho values."""
    zeros_by_z_rho = {}
    for z_rho in z_rho_values:
        z = z_rho
        rho = 1.0  # Normalize rho to 1 for simplicity
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
        zeros_by_z_rho[z_rho] = zeros
    return zeros_by_z_rho


# Parameters
m = 1  # Order of the Bessel function
z_rho_values = np.linspace(0.1, 10, 100)  # Range of z/rho
l_range = (1, 50)  # Range of l to search for zeros

# Find zeros for all z/rho values
zeros_by_z_rho = find_zeros_f_prime_general(l_range, z_rho_values, m)

# Visualization of zeros
plt.figure(figsize=(10, 6))
for z_rho, zeros in zeros_by_z_rho.items():
    plt.plot([z_rho] * len(zeros), zeros, "o", label=f"z/rho={z_rho:.2f}", alpha=0.6)

plt.title("Zeros of f'(l) for Different z/rho Values", fontsize=14)
plt.xlabel("z/rho", fontsize=12)
plt.ylabel("Zeros of f'(l)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
