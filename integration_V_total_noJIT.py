import numpy as np
import scipy.integrate as integrate
import scipy.special as sp
from tqdm import tqdm
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
import mesh_utility
from joblib import Parallel, delayed

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
M_max = 25  # Max value of m
N_max = 25  # Max value of n

# Define other parameters
ALPHA = 100  # Scaling parameter
L = CYLINDER_HEIGHT
R_star = CYLINDER_RADIUS

# Initialize coefficient matrices
A = np.zeros((M_max, N_max))
B = np.zeros((M_max, N_max))


# Function to compute gravity potential at a given cylindrical coordinate
def Phi_alpha(rho, phi, z):
    """Evaluates the gravitational potential at cylindrical coordinates (rho, phi, z)."""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    cartesian_point = np.array([x, y, z]) @ CYLINDER_ROTATION + CYLINDER_CENTER
    potential, _, _ = evaluable_eros(computation_points=cartesian_point, parallel=False)
    return potential


# Define volume integrals
def compute_coefficients(m, n):
    j_mn = sp.jn_zeros(m, n)[-1]  # nth zero of J_m
    J_mn_squared = sp.jv(m + 1, j_mn) ** 2

    def integrand_A(rho, phi, z):
        return (
            rho
            * Phi_alpha(rho, phi, z)
            * sp.jv(m, j_mn * rho / (ALPHA * R_star))
            * np.cos(m * phi)
            * np.exp(j_mn * z / (ALPHA * R_star))
        )

    def integrand_B(rho, phi, z):
        return (
            rho
            * Phi_alpha(rho, phi, z)
            * sp.jv(m, j_mn * rho / (ALPHA * R_star))
            * np.sin(m * phi)
            * np.exp(j_mn * z / (ALPHA * R_star))
        )

    A_mn_integral, _ = integrate.nquad(
        integrand_A, [[0, ALPHA * R_star], [0, 2 * np.pi], [0, L]]
    )
    B_mn_integral, _ = integrate.nquad(
        integrand_B, [[0, ALPHA * R_star], [0, 2 * np.pi], [0, L]]
    )

    A_mn = (2 / (np.pi * L * (ALPHA * R_star) ** 2 * J_mn_squared)) * A_mn_integral
    B_mn = (2 / (np.pi * L * (ALPHA * R_star) ** 2 * J_mn_squared)) * B_mn_integral

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
