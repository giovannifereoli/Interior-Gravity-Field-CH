# V&V Results

# Import necessary libraries
import numpy as np
from polyhedral_gravity import (
    Polyhedron,
    PolyhedronIntegrity,
    GravityEvaluable,
)
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.special import (
    jv as BesselJ,
    jvp as BesselJp,
    jn_zeros,
    factorial,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
import trimesh


# Use a colorblind-friendly color palette
COLOR_PALETTE = ["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_PALETTE)

# Set LaTeX formatting
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

# Load the OBJ file
mesh_pre = trimesh.load("3dmeshes/BENNU_preTag.obj")
mesh_after = trimesh.load("3dmeshes/BENNU_afterTag.obj")

# Define a 90-degree rotation matrix around the y-axis
angle = np.radians(-90)
rotation_matrix = np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1],
    ]
)

# Normalize the mesh (scale to fit in unit sphere)
mesh_pre.apply_translation(-mesh_pre.centroid)  # Center the mesh at the origin
scale_factor = 1.0 / np.max(
    mesh_pre.bounding_box.extents
)  # Scale to fit in unit sphere
mesh_pre.apply_scale(scale_factor)
mesh_after.apply_translation(-mesh_after.centroid)  # Center the mesh at the origin
scale_factor = 1.0 / np.max(
    mesh_after.bounding_box.extents
)  # Scale to fit in unit sphere
mesh_after.apply_scale(scale_factor)

# Apply the rotation
mesh_pre.apply_transform(rotation_matrix.T)
mesh_after.apply_transform(rotation_matrix.T)

# Extract vertices and faces
vertices_pre = 10 * mesh_pre.vertices  # (N, 3) array of vertex coordinates
faces_pre = mesh_pre.faces  # (M, 3) array of triangle indices
vertices_after = 10 * mesh_after.vertices  # (N, 3) array of vertex coordinates
faces_after = mesh_after.faces  # (M, 3) array of triangle indices


# Define asteroid density
DENSITY = 1.0

# 1) No rotation
CYLINDER_CENTER = np.array([0.0, 0.0, 1])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 5  # Height of the cylinder in meters
CYLINDER_RADIUS = 1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.eye(3)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate

# Hyperparameters
ALPHA = 100  # Scaling parameter for the cylinder

# Initialize the polyhedron object
bennu_pre = Polyhedron(
    polyhedral_source=(vertices_pre, faces_pre),
    density=DENSITY,
    integrity_check=PolyhedronIntegrity.DISABLE,
)
bennu_after = Polyhedron(
    polyhedral_source=(vertices_after, faces_after),
    density=DENSITY,
    integrity_check=PolyhedronIntegrity.DISABLE,
)

# Create an evaluable object for gravity calculations
evaluable_bennu_pre = GravityEvaluable(bennu_pre)
evaluable_bennu_after = GravityEvaluable(bennu_after)


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
results_pre = []
for point in tqdm(cylinder_points, desc="Evaluating gravity at points"):
    potential_pre, acceleration_pre, tensor_pre = evaluable_bennu_pre(
        computation_points=point, parallel=False
    )
    results_pre.append(
        {
            "point": point,
            "potential": potential_pre,
            "acceleration": acceleration_pre,
            "tensor": tensor_pre,
        }
    )

results_after = []
for point in tqdm(cylinder_points, desc="Evaluating gravity at points"):
    potential_after, acceleration_after, tensor_after = evaluable_bennu_after(
        computation_points=point, parallel=False
    )
    results_after.append(
        {
            "point": point,
            "potential": potential_after,
            "acceleration": acceleration_after,
            "tensor": tensor_after,
        }
    )

# Convert results to a structured numpy array for easier processing
structured_results_pre = {
    "points": np.array([res["point"] for res in results_pre]),
    "potential": np.array([res["potential"] for res in results_pre]),
    "acceleration": np.array([res["acceleration"] for res in results_pre]),
    "tensor": np.array([res["tensor"] for res in results_pre]),
}

structured_results_after = {
    "points": np.array([res["point"] for res in results_after]),
    "potential": np.array([res["potential"] for res in results_after]),
    "acceleration": np.array([res["acceleration"] for res in results_after]),
    "tensor": np.array([res["tensor"] for res in results_after]),
}

# Save the dataset to a file
np.savez("cylindrical_gravity_dataset_BENNU_pre.npz", **structured_results_pre)
np.savez("cylindrical_gravity_dataset_BENNU_after.npz", **structured_results_after)


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


# Generate the matrix A and vector b
points = structured_results_pre["points"]
potentials_pre = structured_results_pre["potential"]
potentials_after = structured_results_after["potential"]
accelerations_pre = structured_results_pre["acceleration"]
accelerations_after = structured_results_after["acceleration"]


## Plot percentage error on the cylinder
def plot_error_on_cylinder(cylinder_points, percentage_error, title="Percentage Error"):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot asteroid mesh
    mesh = Poly3DCollection(
        vertices_pre[faces_pre],
        alpha=0.3,
        edgecolor="k",
        linewidths=0.3,
        facecolor=COLOR_PALETTE[-1],
    )
    ax.add_collection3d(mesh)

    # Scatter plot of the percentage error
    scatter = ax.scatter(
        cylinder_points[:, 0],
        cylinder_points[:, 1],
        cylinder_points[:, 2],
        c=percentage_error,
        cmap="viridis",
        s=5,
        label="Percentage Error",
        depthshade=False,
        zorder=10,
    )

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Percentage Error (-)")

    # Set labels and title
    plt.title(title, fontsize=16)
    ax.set_xlabel("$X$ (m)")
    ax.set_ylabel("$Y$ (m)")
    ax.set_zlabel("$Z$ (m)")
    ax.set_aspect("equal")


# Convert Cartesian acceleration to cylindrical coordinates
acceleration_pre = cartesian_to_cylindrical_acceleration(points, accelerations_pre)
acceleration_after = cartesian_to_cylindrical_acceleration(points, accelerations_after)

# Calculate error in acceleration
acceleration_error = np.linalg.norm(acceleration_pre - acceleration_after, axis=1)
percentage_acceleration_error = (
    100 * acceleration_error / np.linalg.norm(acceleration_pre, axis=1)
)

# Calculate error on potential
percentage_potential_error = 100 * np.abs(
    (potentials_pre - potentials_after) / potential_pre
)


# Plot error distribution in the cylinder
plot_error_on_cylinder(
    cylinder_points,
    percentage_acceleration_error,
    title="Acceleration Difference Pre- and After-TAG",
)
# Call the function to plot the percentage error
plot_error_on_cylinder(
    cylinder_points,
    percentage_potential_error,
    title="Potential Difference Pre- and After-TAG",
)

from scipy.spatial import cKDTree
from scipy.interpolate import griddata

# Match vertices using nearest neighbors
tree_pre = cKDTree(vertices_pre)
distances, indices = tree_pre.query(vertices_after)

# Compute height (Z) differences
z_diff = vertices_after[:, 2] - vertices_pre[indices, 2]

# Extract corresponding XY coordinates
xy_points = vertices_pre[indices, :2]

# Define grid for interpolation
grid_x, grid_y = np.meshgrid(
    np.linspace(np.min(xy_points[:, 0]), np.max(xy_points[:, 0]), 200),
    np.linspace(np.min(xy_points[:, 1]), np.max(xy_points[:, 1]), 200),
)

# Interpolate height differences onto the grid
grid_z = griddata(xy_points, z_diff, (grid_x, grid_y), method="cubic")

# Define color limits (modify these values as needed)
vmin, vmax = -0.8, 0.4  # Adjust to your desired range
grid_z = np.clip(grid_z, vmin, vmax)

# Plot the heatmap
plt.figure(figsize=(10, 8))
contour = plt.contourf(
    grid_x, grid_y, grid_z, levels=100, cmap="coolwarm", vmin=vmin, vmax=vmax
)
cbar = plt.colorbar(contour)
cbar.set_label("Height Difference (m)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Height Difference Map (XY Plane) - Pre-TAG vs. Post-TAG")
plt.grid(True)

# Show plot
plt.show()
