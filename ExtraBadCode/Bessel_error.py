import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jvp

# HP: only Jm(x) is used in the error function
# HP: for a fixed rho

# Parameters
x = np.linspace(0, 1000, 1000)  # Range of x from 0 to 50

# Prepare visualization
plt.figure(figsize=(12, 8))

# Loop over nu values
for nu in range(2, 101):
    jv_derivative = jvp(nu, x)  # Derivative of Bessel function
    E_i = 0.5 * jv_derivative
    plt.plot(x, E_i, label=f"$\nu={nu}$", alpha=0.6)

plt.xlabel("$x$", fontsize=12)
plt.ylabel("$E_i$", fontsize=12)
plt.show()

# Prepare cumulative sum visualization
plt.figure(figsize=(12, 8))

# Loop over nu values for cumulative sum
for nu in range(2, 101):
    jv_derivative = jvp(nu, x)  # Derivative of Bessel function
    E_i = 0.5 * jv_derivative
    cumulative_sum = np.cumsum(E_i) * (
        x[1] - x[0]
    )  # Approximate cumulative sum as integral
    plt.plot(x, cumulative_sum, label=f"$\nu={nu}$", alpha=0.6)

plt.xlabel("$x$", fontsize=12)
plt.ylabel("Cumulative Sum of $E_i$", fontsize=12)
plt.show()
