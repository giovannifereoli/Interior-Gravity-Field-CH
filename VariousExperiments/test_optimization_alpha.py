#########################################################################################################################
# Small Body Characterization - Cylinder Gravity LS Fitting
# Author: Giovanni Fereoli (The University of Coloradto at Boulder)
# Advisor: Dr. McMahon (The University of Colorado at Boulder)
# Acknowledgement: None
# Date: 2024-09-30
#########################################################################################################################

## Initialization

# Import necessary libraries
import numpy as np
from polyhedral_gravity import (
    Polyhedron,
    PolyhedronIntegrity,
    GravityEvaluable,
)
import mesh_utility
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import (
    jv as BesselJ,
    jn_zeros,
)
import matplotlib as mpl
from scipy.optimize import minimize_scalar


# Use a colorblind-friendly color palette

COLOR_PALETTE = ["#E6001A", "#F08C00", "#0077BB", "#2c7bb6"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_PALETTE)

# Set LaTeX formatting
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 16  # Increase font size for better readability in papers
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["text.latex.preamble"] = r"\usepackage{mathrsfs}"

# Meshes from https://github.com/darioizzo/geodesyNets/tree/master/3dmeshes
vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
vertices_lp, faces_lp = mesh_utility.read_pk_file("3dmeshes/eros_lp.pk")
vertices, faces = np.array(vertices), np.array(faces)

# Define asteroid density
DENSITY = 1.0

# 1) No rotation
CYLINDER_CENTER = np.array([0.0, 0.0, 0.28])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.eye(3)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate

# Initialize the polyhedron object
eros = Polyhedron(
    polyhedral_source=(vertices, faces),
    density=DENSITY,
    integrity_check=PolyhedronIntegrity.DISABLE,
)

# Create an evaluable object for gravity calculations
evaluable_eros = GravityEvaluable(eros)


# Function to generate random points within a cylinder
def generate_points_in_cylinder(center, radius, height, rotation, num_points):
    np.random.seed(1)  # Set seed for reproducibility
    theta = np.random.uniform(0, 2 * np.pi, num_points)  # Random angles
    r = np.sqrt(np.random.uniform(0, radius**2, num_points))  # Random radii
    z = np.random.uniform(0, height, num_points)  # Random heights
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    local_points = np.column_stack((x, y, z))  # Points in local cylinder coordinates
    rotated_points = local_points @ rotation.T  # Apply rotation matrix
    translated_points = rotated_points + center  # Translate to the specified center
    return translated_points


# Generate dataset points within the cylindrical volume
cylinder_points = generate_points_in_cylinder(
    CYLINDER_CENTER, CYLINDER_RADIUS, CYLINDER_HEIGHT, CYLINDER_ROTATION, NUM_POINTS
)

# Evaluate gravity at each point
results = []
for point in tqdm(cylinder_points, desc="Evaluating gravity at points"):
    potential, acceleration, tensor = evaluable_eros(
        computation_points=point, parallel=False
    )
    results.append(
        {
            "point": point,
            "potential": potential,
            "acceleration": acceleration,
            "tensor": tensor,
        }
    )

# Convert results to a structured numpy array for easier processing
structured_results = {
    "points": np.array([res["point"] for res in results]),
    "potential": np.array([res["potential"] for res in results]),
    "acceleration": np.array([res["acceleration"] for res in results]),
    "tensor": np.array([res["tensor"] for res in results]),
}


def compute_error_for_alpha(alpha):
    # Hyperparameters
    ALPHA = alpha  # Scaling parameter for the cylinder

    # Fit cylindrical potential using least squares
    def prepare_linear_system_for_cylindrical_potential(points, potentials, n_n, n_m):
        R, L = CYLINDER_RADIUS, CYLINDER_HEIGHT
        transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
        rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
        phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
        z = transformed_points[:, 2]

        num_points = len(points)
        num_params = 2 * n_n * n_m  # Updated for A and B coefficients
        A = np.zeros((num_points, num_params))
        b = potentials

        k = lambda m, n: jn_zeros(m, n)[-1]
        R_alpha = ALPHA * CYLINDER_RADIUS

        idx = 0
        for m in range(n_m):
            for n in range(1, n_n + 1):
                # Compute terms
                k_mn = k(m, n)
                exp_term = np.exp(-k_mn * z / R_alpha)
                bessel_j = BesselJ(m, k_mn * rho / R_alpha)

                # Populate matrix A with coefficients for A_mn and B_mn
                A[:, idx] = exp_term * bessel_j * np.cos(m * phi)
                A[:, idx + 1] = exp_term * bessel_j * np.sin(m * phi)

                idx += 2

        return A, b

    # Generate the matrix A and vector b
    n_n, n_m = 5, 5  # Truncation parameters
    points = structured_results["points"]
    potentials = structured_results["potential"]

    # Prepare the system for cylindrical potential fitting
    A_pot, b_pot = prepare_linear_system_for_cylindrical_potential(
        points, potentials, n_n, n_m
    )

    # Define LSQ fitting
    aug_A = A_pot
    aug_b = b_pot

    # Numpy's lstsq
    result = np.linalg.lstsq(aug_A, aug_b, rcond=None)
    fitted_params = result[0]

    # Calculate percentage error
    fitted_potentials = A_pot @ fitted_params
    percentage_error = 100 * np.abs((fitted_potentials - b_pot) / b_pot)
    return np.max(percentage_error)


error_function = lambda alpha: compute_error_for_alpha(alpha)

res = minimize_scalar(
    error_function,
    bounds=(1, 1e6),
    method="bounded",
    options={"xatol": 1e-12, "disp": True},
)
optimal_alpha = res.x
minimum_error = res.fun
print(f"Optimal alpha: {optimal_alpha}")
print(f"Minimum error: {minimum_error}")
