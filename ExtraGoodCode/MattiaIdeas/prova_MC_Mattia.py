import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
from polyhedral_gravity import Polyhedron, GravityEvaluable, PolyhedronIntegrity

# Define directories
OBJ_DIR = "MattiaObj/"  # Directory containing .obj files
obj_files = [f for f in os.listdir(OBJ_DIR) if f.endswith(".obj")][:10]  # Limit to 50

# Define cylinder properties
CYLINDER_POSITIONS = [
    [1, 0, 0],
    [-1, 0, 0],  # ±X
    [0, 1, 0],
    [0, -1, 0],  # ±Y
    [0, 0, 1],
    [0, 0, -1],  # ±Z
]
CYLINDER_HEIGHT = 0.5
CYLINDER_RADIUS = 0.1
NUM_POINTS = 1000  # Points in each cylinder
DENSITY = 1.0  # Asteroid density


# Function to generate random points in a cylinder
def generate_points_in_cylinder(center, radius, height, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    r = np.sqrt(np.random.uniform(0, radius**2, num_points))
    z = np.random.uniform(-height / 2, height / 2, num_points)
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    z = z + center[2]
    return np.column_stack((x, y, z))


# Initialize plot for all results
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Acceleration Errors Across All OBJ Files")
ax.set_xlabel("Cylinder Position")
ax.set_ylabel("Mean Acceleration Error")
colors = plt.cm.viridis(np.linspace(0, 1, len(obj_files)))

# Iterate over each .obj file
for idx, obj_file in enumerate(tqdm(obj_files, desc="Processing OBJ files")):
    mesh = trimesh.load(os.path.join(OBJ_DIR, obj_file))
    mesh.apply_translation(-mesh.centroid)
    scale_factor = 1.0 / np.max(mesh.bounding_box.extents)
    mesh.apply_scale(scale_factor)

    # Extract vertices and faces
    vertices, faces = mesh.vertices, mesh.faces

    # Initialize polyhedron gravity model
    polyhedron = Polyhedron(
        (vertices, faces),
        density=DENSITY,
        integrity_check=PolyhedronIntegrity.DISABLE,
    )
    gravity_eval = GravityEvaluable(polyhedron)

    # Compute errors in six cylinders
    all_errors = []
    cylinder_points_collection = []
    for cylinder_center in CYLINDER_POSITIONS:
        cylinder_points = generate_points_in_cylinder(
            cylinder_center, CYLINDER_RADIUS, CYLINDER_HEIGHT, NUM_POINTS
        )
        errors = []
        cylinder_points_collection.append(cylinder_points)
        for point in cylinder_points:
            _, acc, _ = gravity_eval(computation_points=point)
            errors.append(np.linalg.norm(acc))  # Store acceleration magnitude
        all_errors.append(np.array(errors))

    # Convert to 2D error representation
    mean_errors = np.array([np.mean(errors) for errors in all_errors])

    # Plot results on a single figure
    ax.scatter(
        range(len(CYLINDER_POSITIONS)),
        mean_errors,
        label=f"{obj_file}",
        color=colors[idx],
        alpha=0.7,
    )

ax.legend(loc="upper right", fontsize=8)
plt.xticks(range(len(CYLINDER_POSITIONS)), ["+X", "-X", "+Y", "-Y", "+Z", "-Z"])
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# Plot 3D visualization of cylinders around the mesh
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot asteroid mesh
ax.plot_trisurf(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    triangles=faces,
    alpha=0.5,
    color="gray",
)

# Plot cylinder point
selected_obj_file = np.random.choice(obj_files)
for i, cylinder_points in enumerate(cylinder_points_collection):
    ax.scatter(
        cylinder_points[:, 0],
        cylinder_points[:, 1],
        cylinder_points[:, 2],
        c=all_errors[i],
        cmap="coolwarm",
        s=5,
        label=f"Cylinder {i+1}",
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Acceleration Errors around {selected_obj_file}")
ax.legend()
plt.show()
