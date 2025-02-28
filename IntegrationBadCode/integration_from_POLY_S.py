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
CYLINDER_CENTER = np.array([-0.1, -0.28, 0])  # Center of the cylinder
CYLINDER_HEIGHT = 0.5  # Cylinder height
CYLINDER_RADIUS = 0.09  # Cylinder radius
CYLINDER_ROTATION = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # Rotation matrix

# Define Bessel-related parameters
m = 2  # Order of Bessel function
n = 3  # Index for the zero of the Bessel function

# Compute the first zero of J_m using scipy.special.jn_zeros
j_mn = sp.jn_zeros(m, n)[-1]  # First zero of J_m

# Define other parameters
alpha = 100  # Scaling parameter
L = CYLINDER_HEIGHT
R_star = CYLINDER_RADIUS

# Compute the Bessel function normalization term
J_mn_squared = sp.jv(m + 1, j_mn / (alpha * R_star)) ** 2


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


# Define F_z functions
@track_progress
def F_A_z(rho, phi, z):
    return (
        rho
        * (alpha * R_star / j_mn)
        * Phi_alpha(rho, phi, z)
        * sp.jv(m, j_mn * rho / (alpha * R_star))
        * np.cos(m * phi)
        * np.exp(j_mn * z / (alpha * R_star))
    )


@track_progress
def F_B_z(rho, phi, z):
    return (
        rho
        * (alpha * R_star / j_mn)
        * Phi_alpha(rho, phi, z)
        * sp.jv(m, j_mn * rho / (alpha * R_star))
        * np.sin(m * phi)
        * np.exp(j_mn * z / (alpha * R_star))
    )


# Compute surface integrals over the top and bottom caps
def integrate_surface(F_z, z_value):
    """Integrates over the circular cap at z=z_value."""

    def integrand(rho, phi):
        return F_z(rho, phi, z_value) * rho  # Surface element: rho d_rho d_phi

    result, _ = integrate.nquad(integrand, [[0, R_star], [0, 2 * np.pi]])
    return result


# Compute top and bottom surface integrals
A_mn_top = integrate_surface(F_A_z, L)
A_mn_bottom = integrate_surface(F_A_z, 0)
B_mn_top = integrate_surface(F_B_z, L)
B_mn_bottom = integrate_surface(F_B_z, 0)

# Compute final A_mn and B_mn values
A_mn = (4 / (np.pi * L * (R_star**2) * J_mn_squared)) * (A_mn_top - A_mn_bottom)
B_mn = (4 / (np.pi * L * (R_star**2) * J_mn_squared)) * (B_mn_top - B_mn_bottom)

# Display results
print(f"A_mn = {A_mn}")
print(f"B_mn = {B_mn}")
