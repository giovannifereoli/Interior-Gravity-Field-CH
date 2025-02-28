import numpy as np
import scipy.special as sp
from tqdm import tqdm
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
import mesh_utility

# Load mesh for the polyhedral gravity model
vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
vertices, faces = np.array(vertices), np.array(faces)

# Define asteroid density
DENSITY = 1.0

# Initialize Constant to Help with Integration
CONST = 1e11

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
m = 0  # Order of Bessel function
n = 3  # Index of Bessel function
j_mn = sp.jn_zeros(m, n)[-1]
ALPHA = 100  # Scaling parameter
L = CYLINDER_HEIGHT
R_star = CYLINDER_RADIUS

# Compute the Bessel function normalization term
J_mn_squared = sp.jv(m + 1, j_mn) ** 2

# Number of Monte Carlo samples
N_samples = 10**2  # Adjust for accuracy

# Initialize sums for the integral
A_mn_sum = 0.0
B_mn_sum = 0.0

# Monte Carlo integration (point-by-point with tqdm)
for _ in tqdm(range(N_samples), desc="Monte Carlo Integration"):
    # Generate a single random sample in cylindrical coordinates
    rho = np.random.uniform(0, R_star)  # Radial coordinate
    phi = np.random.uniform(0, 2 * np.pi)  # Angular coordinate

    # Convert to Cartesian coordinates
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = L  # Fixed z plane

    # Transform to global frame
    cartesian_point = np.array([x, y, z]) @ CYLINDER_ROTATION + CYLINDER_CENTER

    # Evaluate gravity potential at the single point
    potential_value, _, _ = evaluable_eros(
        computation_points=cartesian_point, parallel=False
    )

    # Compute integrand values
    J_value = sp.jv(m, j_mn * rho / (ALPHA * R_star))
    cos_value = np.cos(m * phi)
    sin_value = np.sin(m * phi)

    A_mn_sum += CONST * rho * potential_value * J_value * cos_value
    B_mn_sum += CONST * rho * potential_value * J_value * sin_value

# Compute Monte Carlo estimate of the integral
area = R_star * 2 * np.pi  # Integral domain area in (rho, phi)
A_mn_integral = area * A_mn_sum / N_samples
B_mn_integral = area * B_mn_sum / N_samples

print(f"A_mn (Monte Carlo, point-by-point): {A_mn_integral}")
print(f"B_mn (Monte Carlo, point-by-point): {B_mn_integral}")
