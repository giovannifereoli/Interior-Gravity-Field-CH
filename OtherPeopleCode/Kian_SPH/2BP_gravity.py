import numpy as np
import pandas as pd
import spiceypy as spice
import os

# Constants
ALPHA_E = 3000  # m
L_MAX = 5  # Maximum degree for spherical harmonics


# Compute acceleration due to two-body gravity
def acc_2BP(y, mu, r):
    return -mu / r**3 * y[:3]


# Load and preprocess spherical harmonics coefficients
def load_coefficients(file_path, l_max):
    data = pd.read_csv(file_path, delimiter="   ", header=None, engine="python")
    coeff_arr = np.zeros((l_max + 1, 2 * (l_max + 1)))
    idx = 0

    for n in range(l_max + 1):
        for m in range(n + 1):
            coeff_arr[n, 2 * m] = data.iloc[idx, 2]
            coeff_arr[n, 2 * m + 1] = data.iloc[idx, 3]
            idx += 1

    # Denormalize coefficients
    for n in range(l_max + 1):
        for m in range(n + 1):
            delta = 1 if m == 0 else 0
            norm_factor = np.sqrt(
                (2 - delta)
                * (2 * n + 1)
                * np.math.factorial(n - m)
                / np.math.factorial(n + m)
            )
            coeff_arr[n, 2 * m] *= norm_factor
            coeff_arr[n, 2 * m + 1] *= norm_factor

    return coeff_arr


# Compute acceleration due to non-spherical gravity
def fast_gravity(mu, X, r, et, coeff_file):
    # Transform to body-fixed frame
    rotation_to_body = spice.pxform("J2000", "IAU_CG", et)
    X_body = np.dot(rotation_to_body, X)
    rho = np.sqrt(X_body[0] ** 2 + X_body[1] ** 2)
    epsilon = X_body[2] / r
    lambda_fg = np.arctan2(X_body[1], X_body[0])

    # Load coefficients
    coeff = load_coefficients(coeff_file, L_MAX)

    # Compute Legendre polynomials
    P = np.zeros((L_MAX + 2, L_MAX + 2))
    P[0, 0] = 1
    P[1, 0] = epsilon
    P[1, 1] = 1
    for n in range(2, L_MAX + 2):
        for m in range(n + 1):
            if m < n:
                P[n, m] = (
                    (2 * n - 1) * epsilon * P[n - 1, m] - (n + m - 1) * P[n - 2, m]
                ) / (n - m)
            elif m == n:
                P[n, m] = (2 * n - 1) * P[n - 1, m - 1]

    # Compute acceleration
    ag = np.zeros(3)
    for n in range(2, L_MAX + 1):
        for m in range(n + 1):
            Cm = rho**m * np.cos(m * lambda_fg)
            Sm = rho**m * np.sin(m * lambda_fg)
            Bnm = coeff[n, 2 * m] * Cm + coeff[n, 2 * m + 1] * Sm

            ag += (
                (mu / r**2)
                * ((ALPHA_E / r) ** n * (n + m + 1) * P[n, m] * Bnm / r**m)
                * (X_body / r)
            )
            ag += (
                (mu / r**2)
                * ((ALPHA_E / r) ** n * P[n, m + 1] * Bnm / r**m)
                * np.array([0, 0, 1])
            )

    # Transform back to inertial frame
    rotation_to_inertial = spice.pxform("IAU_CG", "J2000", et)
    return np.dot(rotation_to_inertial, ag)


# Combined gravity computation (spherical + non-spherical)
def compute_gravity(y, mu, et, coeff_file):
    r = np.linalg.norm(y[:3])
    a_2bp = acc_2BP(y[:3], mu, r)
    a_fast = fast_gravity(mu, y[:3], r, et, coeff_file)
    return a_2bp + a_fast


# Example usage
if __name__ == "__main__":
    # Test parameters
    y = np.array([7000, 0, 0, 0, 7.5, 0])  # Position and velocity
    mu = 3.986e14  # Gravitational parameter (m^3/s^2)
    et = spice.str2et("2024-11-15T00:00:00")  # Ephemeris time
    coeff_file = os.path.join(os.path.dirname(__file__), "67Pgravity.csv.txt")

    acceleration = compute_gravity(y, mu, et, coeff_file)
    print("Combined Acceleration:", acceleration)
