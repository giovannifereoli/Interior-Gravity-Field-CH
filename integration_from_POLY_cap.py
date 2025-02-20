import numpy as np
import scipy.integrate as integrate
import scipy.special as sp
from tqdm import tqdm
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
import mesh_utility

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
m = 0  # Order of Bessel function
n = 3  # Index of Bessel function
j_mn = sp.jn_zeros(m, n)[-1]
ALPHA = 100  # Scaling parameter
L = CYLINDER_HEIGHT
R_star = CYLINDER_RADIUS

# Compute the Bessel function normalization term
J_mn_squared = sp.jv(m + 1, j_mn) ** 2


# Function to compute gravity potential at a given cylindrical coordinate
def Phi_alpha(rho, phi, z):
    """Evaluates the gravitational potential at cylindrical coordinates (rho, phi, z)."""
    # Transform cylindrical to Cartesian for gravity evaluation
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    cartesian_point = np.array([x, y, z]) @ CYLINDER_ROTATION + CYLINDER_CENTER

    # Evaluate gravity potential
    potential, _, _ = evaluable_eros(computation_points=cartesian_point, parallel=False)

    return potential


# Define a wrapper to track progress
progress_bar = tqdm(total=100000, desc="Integrating with nquad", unit="evals")


def track_progress(func):
    """Wraps the integrand function to track calls."""

    def wrapper(*args, **kwargs):
        progress_bar.update(1)  # Update progress bar
        return func(*args, **kwargs)

    return wrapper


# Define the integrand for A_mn
@track_progress
def integrand_A(rho, phi):
    return (
        rho
        * Phi_alpha(rho, phi, L)
        * sp.jv(m, j_mn * rho / (ALPHA * R_star))
        * np.cos(m * phi)
    )


# Define the integrand for B_mn
@track_progress
def integrand_B(rho, phi):
    return (
        rho
        * Phi_alpha(rho, phi, L)
        * sp.jv(m, j_mn * rho / (ALPHA * R_star))
        * np.sin(m * phi)
    )


# Perform numerical integration using scipy.integrate.nquad
A_mn_integral, _ = integrate.nquad(
    integrand_A,
    [[0, ALPHA * R_star], [0, 2 * np.pi]],
    opts={"epsabs": 1e-9, "epsrel": 1e-9},
)
B_mn_integral, _ = integrate.nquad(
    integrand_B,
    [[0, ALPHA * R_star], [0, 2 * np.pi]],
    opts={"epsabs": 1e-9, "epsrel": 1e-9},
)

# Compute final A_mn and B_mn values
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

# Close progress bar
progress_bar.close()

# Display results
print(f"A_mn = {A_mn}")
print(f"B_mn = {B_mn}")
