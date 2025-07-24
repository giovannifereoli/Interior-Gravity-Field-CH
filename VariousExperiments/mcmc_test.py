import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Define the true density function inside the sphere
def true_density(x, y, z):
    return np.exp(-5 * (x**2 + y**2 + z**2)) + 0.3 * np.sin(3 * x) * np.cos(
        3 * y
    ) * np.sin(3 * z)


# Generate synthetic observations inside the unit sphere
np.random.seed(42)
num_samples = 100
samples = []
densities = []

while len(samples) < num_samples:
    x, y, z = np.random.uniform(-1, 1, 3)
    if x**2 + y**2 + z**2 <= 1:  # Ensure points are within the sphere
        samples.append([x, y, z])
        densities.append(true_density(x, y, z) + np.random.normal(0, 0.1))  # Add noise

samples = np.array(samples)
densities = np.array(densities)


# Define basis functions (e.g., radial Gaussian functions)
def basis_functions(x, y, z, centers):
    return np.array(
        [
            np.exp(-10 * ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2))
            for cx, cy, cz in centers
        ]
    ).T


# Initialize basis function centers
num_basis = 100
basis_centers = np.random.uniform(-1, 1, (num_basis, 3))


# Define likelihood function
def log_likelihood(params, X, y):
    basis_vals = basis_functions(X[:, 0], X[:, 1], X[:, 2], basis_centers)
    predicted = np.dot(basis_vals, params)
    return -0.5 * np.sum((y - predicted) ** 2)  # Gaussian likelihood


# MCMC with Metropolis-Hastings
num_iters = 50000
param_dim = num_basis  # Number of basis functions
params = np.random.randn(param_dim)
accepted = []
likelihoods = []

for i in range(num_iters):
    proposal = params + np.random.normal(0, 0.05, param_dim)
    log_lik_current = log_likelihood(params, samples, densities)
    log_lik_proposal = log_likelihood(proposal, samples, densities)

    accept_prob = np.exp(log_lik_proposal - log_lik_current)
    if np.random.rand() < accept_prob:
        params = proposal
        print(f"Iteration {i+1}: Accepted new parameters")
    else:
        print(f"Iteration {i+1}: Rejected new parameters")

    accepted.append(params.copy())
    likelihoods.append(log_lik_current)

accepted = np.array(accepted)
likelihoods = np.array(likelihoods)

# Compute estimated density on a grid for visualization
grid_size = 30
xg, yg, zg = (
    np.linspace(-1, 1, grid_size),
    np.linspace(-1, 1, grid_size),
    np.linspace(-1, 1, grid_size),
)
Xg, Yg, Zg = np.meshgrid(xg, yg, zg)
X_flat = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

final_params = np.mean(accepted[-1000:], axis=0)  # Take the mean of last samples
basis_vals = basis_functions(X_flat[:, 0], X_flat[:, 1], X_flat[:, 2], basis_centers)
estimated_density = np.dot(basis_vals, final_params).reshape(Xg.shape)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Prior distribution (initial parameter distribution)
axes[0, 0].hist(accepted[0], bins=20, alpha=0.7, label="Prior")
axes[0, 0].set_title("Prior Distribution")
axes[0, 0].legend()

# Posterior distribution (final parameter distribution)
axes[0, 1].hist(accepted[-1], bins=20, alpha=0.7, label="Posterior")
axes[0, 1].set_title("Posterior Distribution")
axes[0, 1].legend()

# Likelihood progression
t = np.arange(len(likelihoods))
axes[0, 2].plot(t, likelihoods, label="Log-Likelihood")
axes[0, 2].set_title("Likelihood Over Iterations")
axes[0, 2].set_xlabel("Iteration")
axes[0, 2].set_ylabel("Log-Likelihood")
axes[0, 2].legend()

# Trace plots for parameters
for i in range(min(3, param_dim)):
    axes[1, 0].plot(accepted[:, i], label=f"Param {i+1}")
axes[1, 0].set_title("Trace Plots of Parameters")
axes[1, 0].legend()

# Estimated density slice at z=0
img = axes[1, 1].imshow(
    estimated_density[:, :, grid_size // 2],
    extent=[-1, 1, -1, 1],
    origin="lower",
    cmap="viridis",
)
axes[1, 1].set_title("Estimated Density Slice (z=0)")
fig.colorbar(img, ax=axes[1, 1])

# True density slice at z=0
true_density_slice = np.array([[true_density(x, y, 0) for x in xg] for y in yg])
img2 = axes[1, 2].imshow(
    true_density_slice, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis"
)
axes[1, 2].set_title("True Density Slice (z=0)")
fig.colorbar(img2, ax=axes[1, 2])

plt.tight_layout()
plt.show()
