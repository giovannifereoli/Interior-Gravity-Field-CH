import numpy as np
import scipy.special as sp
from tqdm import tqdm
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
import mesh_utility
from joblib import Parallel, delayed
from numba import njit
from scipy.integrate import dblquad

# TODO: why they go up????

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
NUM_POINTS = 25  # Number of points to generate

# Define Bessel-related parameters
M_max = 25  # Max value of m
N_max = 25  # Max value of n

# Define other parameters
ALPHA = 100  # Scaling parameter
L = CYLINDER_HEIGHT
R_star = CYLINDER_RADIUS

# Initialize coefficient matrices
A = np.zeros((M_max, N_max))
B = np.zeros((M_max, N_max))

# Precompute Bessel function zeros and values
bessel_zeros = {m: sp.jn_zeros(m, N_max) for m in range(M_max)}
precomputed_bessel = {
    (m, n): (
        lambda rho: sp.jv(m, bessel_zeros[m][n - 1] * rho / (ALPHA * R_star)),
        (sp.jv(m + 1, bessel_zeros[m][n - 1])) ** 2,
    )
    for m in range(M_max)
    for n in range(1, N_max + 1)
}


# Function to compute gravity potential at a given cylindrical coordinate
def Phi_alpha(rho, phi, z):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    cartesian_point = np.array([x, y, z]) @ CYLINDER_ROTATION + CYLINDER_CENTER
    potential, _, _ = evaluable_eros(computation_points=cartesian_point, parallel=False)
    return potential


# Define volume integrals using numba for speed
def compute_integrand_A(rho, phi, phi_alpha, J_mn_func, j_mn):
    return rho * phi_alpha * J_mn_func(rho) * np.cos(phi)


def compute_integrand_B(rho, phi, phi_alpha, J_mn_func, j_mn):
    return rho * phi_alpha * J_mn_func(rho) * np.sin(phi)


# Function to compute A_mn and B_mn in parallel
def compute_coefficients(m, n):
    j_mn = bessel_zeros[m][n - 1]
    J_mn_func, J_mn_squared = precomputed_bessel[(m, n)]

    def integrand_A(phi, rho):  # OSS: contrary!
        phi_alpha = Phi_alpha(rho, phi, L)
        return compute_integrand_A(rho, phi, phi_alpha, J_mn_func, j_mn)

    def integrand_B(phi, rho):
        phi_alpha = Phi_alpha(rho, phi, L)
        return compute_integrand_B(rho, phi, phi_alpha, J_mn_func, j_mn)

    A_mn_integral, _ = dblquad(
        integrand_A,
        0,
        ALPHA * R_star,
        0,
        2 * np.pi,
    )
    B_mn_integral, _ = dblquad(
        integrand_B,
        0,
        ALPHA * R_star,
        0,
        2 * np.pi,
    )

    A_mn = (
        2
        / (np.pi * (ALPHA * R_star) ** 2 * J_mn_squared)
        * np.exp(j_mn * L / (ALPHA * R_star))
    ) * A_mn_integral
    B_mn = (
        2
        / (np.pi * (ALPHA * R_star) ** 2 * J_mn_squared)
        * np.exp(j_mn * L / (ALPHA * R_star))
    ) * B_mn_integral

    return m, n - 1, A_mn, B_mn


# Parallel processing with progress bar
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(compute_coefficients)(m, n)
    for m in tqdm(range(M_max), desc="Computing coefficients for m")
    for n in range(1, N_max + 1)
)

# Store results in matrices
for m, n, A_mn, B_mn in results:
    A[m, n] = A_mn
    B[m, n] = B_mn

# Save results
np.save("A_coefficients.npy", A)
np.save("B_coefficients.npy", B)

print(f"A coefficients shape: {A.shape}")
print(f"B coefficients shape: {B.shape}")
