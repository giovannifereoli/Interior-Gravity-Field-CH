#########################################################################################################################
# Small Body Characterization - Cylinder Gravity LS Fitting
# Author: Giovanni Fereoli (The University of Coloradto at Boulder)
# Advisor: Dr. McMahon (The University of Colorado at Boulder)
# Acknowledgement: None
# Date: 2024-09-30
#########################################################################################################################

# Import necessary libraries
import numpy as np
from polyhedral_gravity import (
    Polyhedron,
    PolyhedronIntegrity,
    GravityEvaluable,
)
import mesh_utility
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.special import (
    jv as BesselJ,
    jvp as BesselJp,
    jn_zeros,
)
from scipy.optimize import minimize_scalar
import matplotlib as mpl


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

# 2) as before,Rotation x-axis
"""'
CYLINDER_CENTER = np.array([-0.1, -0.28, 0])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate
"""

# 3) as before,Rotation x-axis
"""
CYLINDER_CENTER = np.array([-1.26, 0, 0])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.array(
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate
"""

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

    # Convert Cartesian acceleration components to cylindrical coordinates
    def cartesian_to_cylindrical_acceleration(points, accelerations):
        """
        Convert Cartesian acceleration components to cylindrical coordinates.

        Args:
            points: Cartesian points (N, 3).
            accelerations: Cartesian accelerations (N, 3).

        Returns:
            Cylindrical accelerations (N, 3): [a_rho, a_phi, a_z].
        """
        transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
        rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
        phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])

        transformed_accelerations = accelerations @ CYLINDER_ROTATION.T

        a_x, a_y, a_z = (
            transformed_accelerations[:, 0],
            transformed_accelerations[:, 1],
            transformed_accelerations[:, 2],
        )
        a_rho = a_x * np.cos(phi) + a_y * np.sin(phi)
        a_phi = -a_x * np.sin(phi) + a_y * np.cos(phi)
        a_z = a_z  # Axial acceleration remains the same

        return np.column_stack((a_rho, a_phi, a_z))

    # Fit cylindrical acceleration components using least squares
    def prepare_linear_system_for_cylindrical_acceleration(
        points, accelerations, n_n, n_m
    ):
        """
        Prepare the matrix A and vector b for fitting cylindrical acceleration components.

        Args:
            points: Cartesian points (N, 3).
            accelerations: Cylindrical accelerations (N, 3).
            n_n, n_m: Truncation parameters.

        Returns:
            A: Design matrix for least squares fitting.
            b: Flattened vector of acceleration components.
        """
        R, L = CYLINDER_RADIUS, CYLINDER_HEIGHT
        transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
        rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
        phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
        z = transformed_points[:, 2]

        num_points = len(points)
        num_params = 2 * n_n * n_m  # Updated for A and B coefficients
        A = np.zeros((3 * num_points, num_params))
        b = accelerations.flatten(
            order="C"
        )  # Ensure row-major flattening for consistency

        k = lambda m, n: jn_zeros(m, n)[-1]
        R_alpha = ALPHA * CYLINDER_RADIUS

        idx = 0
        for m in range(n_m):
            for n in range(1, n_n + 1):
                # Useful coefficients
                k_mn = k(m, n)
                bessel_j = BesselJ(m, k_mn / R_alpha * rho)
                bessel_j_derivative = BesselJp(m, k_mn / R_alpha * rho)
                exp_term = np.exp(-k_mn / R_alpha * z)
                cos_m_phi = np.cos(m * phi)
                sin_m_phi = np.sin(m * phi)

                # Radial (rho) component
                dV_drho = (k_mn / R_alpha) * exp_term * bessel_j_derivative

                # Angular (phi) component
                dV_dphi = exp_term * bessel_j * (m / (rho + 1e-14))

                # Axial (z) component
                dV_dz = (-k_mn / R_alpha) * exp_term * bessel_j

                for i in range(num_points):
                    # Radial (rho)
                    A[3 * i, idx] = dV_drho[i] * cos_m_phi[i]
                    A[3 * i, idx + 1] = dV_drho[i] * sin_m_phi[i]

                    # Angular (phi)
                    A[3 * i + 1, idx] = dV_dphi[i] * -sin_m_phi[i]
                    A[3 * i + 1, idx + 1] = dV_dphi[i] * cos_m_phi[i]

                    # Axial (z)
                    A[3 * i + 2, idx] = dV_dz[i] * cos_m_phi[i]
                    A[3 * i + 2, idx + 1] = dV_dz[i] * sin_m_phi[i]

                idx += 2

        return A, b

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
    n_n, n_m = 25, 25  # Truncation parameters
    num_params = 2 * n_n * n_m  # Updated for two coefficients per term (A and B)
    points = structured_results["points"]
    accelerations = structured_results["acceleration"]
    potentials = structured_results["potential"]

    # Transform Cartesian accelerations to cylindrical
    cylindrical_accelerations = cartesian_to_cylindrical_acceleration(
        points, accelerations
    )

    # Prepare the system for cylindrical acceleration fitting
    A_acc, b_acc = prepare_linear_system_for_cylindrical_acceleration(
        points, cylindrical_accelerations, n_n, n_m
    )

    # Prepare the system for cylindrical potential fitting
    A_pot, b_pot = prepare_linear_system_for_cylindrical_potential(
        points, potentials, n_n, n_m
    )

    # Define LSQ fitting
    aug_A = np.vstack([A_acc, A_pot])
    aug_b = np.hstack([b_acc, b_pot])

    # Numpy's lstsq
    result = np.linalg.lstsq(aug_A, aug_b, rcond=None)
    fitted_params = result[0]

    # Compute LSQ error for the potential part only
    fitted_potentials = aug_A @ fitted_params
    lsq_error = np.sum((fitted_potentials - aug_b) ** 2)

    return lsq_error


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
