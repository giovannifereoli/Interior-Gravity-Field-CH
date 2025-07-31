import numpy as np
import trimesh
import pickle
import os
from scipy.spatial import KDTree

# Global cache
MESH_CACHE = None
MESH_VERTICES = None
MESH_FACES = None
MESH_NORMALS = None
ASTEROID_CENTER = np.array([0.0, 0.0, 0.28])

def load_and_cache_mesh():
    """Load and cache asteroid mesh with normals."""
    global MESH_CACHE, MESH_VERTICES, MESH_FACES, MESH_NORMALS
    cache_file = "asteroid_mesh_cache.pkl"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            MESH_CACHE, MESH_VERTICES, MESH_FACES, MESH_NORMALS = pickle.load(f)
    else:
        try:
            MESH_CACHE = trimesh.load("asteroid_model.obj", force='mesh')
            MESH_VERTICES = MESH_CACHE.vertices + ASTEROID_CENTER
            MESH_FACES = MESH_CACHE.faces
            MESH_NORMALS = MESH_CACHE.vertex_normals
        except FileNotFoundError:
            print("Warning: asteroid_model.obj not found. Using fallback perturbed icosahedron.")
            MESH_CACHE = trimesh.creation.icosahedron()
            scale = 0.1
            MESH_VERTICES = MESH_CACHE.vertices * scale
            np.random.seed(42)
            MESH_VERTICES += np.random.normal(0, 0.01, MESH_VERTICES.shape)  # Perturb for realism
            MESH_VERTICES += ASTEROID_CENTER
            MESH_FACES = MESH_CACHE.faces
            MESH_NORMALS = MESH_CACHE.vertex_normals
        with open(cache_file, 'wb') as f:
            pickle.dump((MESH_CACHE, MESH_VERTICES, MESH_FACES, MESH_NORMALS), f)

def compute_measurement_partials(position, velocity, quaternion, n_state, camera_params, n_landmarks, noise_params):
    """
    Compute the measurement model Jacobian H and noise covariance R for LIDAR range,
    range rate, and camera pixel measurements, using optimized ray tracing against an
    asteroid mesh (.obj) for LIDAR and landmark projections for the camera. Assumes
    constant illumination, no albedo effects, no camera distortion, and no landmark occlusion.

    Args:
        position: Spacecraft position in asteroid body frame [x, y, z] (km).
        velocity: Spacecraft velocity in asteroid body frame [v_x, v_y, v_z] (km/s).
        quaternion: Spacecraft attitude quaternion [q0, q1, q2, q3] (scalar-first, body to asteroid frame).
        n_state: State dimension (10 + 2*n_n*n_m).
        camera_params: Dict with 'fx', 'fy' (focal lengths, pixels), 'cx', 'cy' (principal point, pixels),
                       'image_size' (pixels).
        n_landmarks: Number of landmarks to select (integer).
        noise_params: Dict with 'sigma_range' (m), 'sigma_range_rate' (m/s), 'sigma_pixel' (pixels).

    Returns:
        H: (2 + 2*N)xN Jacobian matrix [drange/dstate, drangerate/dstate, du_i/dstate, dv_i/dstate].
        R: (2 + 2*N)x(2 + 2*N) measurement noise covariance matrix.
        r_intersect: LIDAR intersection point with asteroid (km) or None.
        landmarks_visible: List of (u, v, idx) for visible landmarks.
    """
    # Load mesh if not cached
    if MESH_CACHE is None:
        load_and_cache_mesh()

    # Camera parameters
    fx = camera_params.get('fx', 1000)
    fy = camera_params.get('fy', 1000)
    cx = camera_params.get('cx', 0)
    cy = camera_params.get('cy', 0)
    image_size = camera_params.get('image_size', 1000)

    # Noise parameters
    sigma_range = noise_params.get('sigma_range', 0.1)  # m
    sigma_range_rate = noise_params.get('sigma_range_rate', 1e-3)  # m/s
    sigma_pixel = noise_params.get('sigma_pixel', 0.2)  # pixels

    # Normalize quaternion
    q = quaternion / np.linalg.norm(quaternion)
    q0, q1, q2, q3 = q

    # Rotation matrix
    R_body_to_asteroid = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    R_asteroid_to_body = R_body_to_asteroid.T

    # LIDAR beam
    beam_body = np.array([0.0, 0.0, 1.0])
    beam_asteroid = R_body_to_asteroid @ beam_body
    beam_divergence = 0.1e-3  # 0.1 mrad

    # Ray tracing with Embree
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(MESH_CACHE)
    origins = np.array([position])
    directions = np.array([beam_asteroid])
    locations, _, face_indices = intersector.intersects_location(origins, directions, multiple_hits=False)
    if len(locations) > 0:
        r_intersect = locations[0] + ASTEROID_CENTER
        min_t = np.linalg.norm(r_intersect - position)
        range_noise = sigma_range * (1 + min_t / 10)  # Range-dependent
        range_rate_noise = sigma_range_rate
        range_bias = 0.01e-3  # 0.01 m timing error
        min_t += range_bias
    else:
        r_intersect = None
        min_t = np.inf
        range_noise = sigma_range
        range_rate_noise = sigma_range_rate

    # Preselect landmarks with k-d tree
    kdtree = KDTree(MESH_VERTICES)
    _, candidate_indices = kdtree.query(position, k=min(100, len(MESH_VERTICES)))

    # Vectorized landmark projection
    lm_rel = MESH_VERTICES[candidate_indices] - position
    lm_cam = np.dot(lm_rel, R_asteroid_to_body.T)
    z_cam = lm_cam[:, 2]
    valid = z_cam > 0
    u = np.full(len(candidate_indices), np.inf)
    v = np.full(len(candidate_indices), np.inf)
    u[valid] = fx * lm_cam[valid, 0] / z_cam[valid] + cx
    v[valid] = fy * lm_cam[valid, 1] / z_cam[valid] + cy

    # Landmark selection (no occlusion, no distortion)
    landmarks_visible = []
    for idx in candidate_indices[np.where(valid)[0]]:
        if abs(u[idx - candidate_indices[0]]) > image_size or abs(v[idx - candidate_indices[0]]) > image_size:
            continue
        lm = MESH_VERTICES[idx]
        normal = MESH_NORMALS[idx]
        cam_dir = -(R_asteroid_to_body @ (lm - position))
        cam_dir /= np.linalg.norm(cam_dir)
        weight = max(np.dot(normal, cam_dir), 0.1)
        dist_to_center = np.sqrt((u[idx - candidate_indices[0]] - cx)**2 + (v[idx - candidate_indices[0]] - cy)**2)
        landmarks_visible.append((u[idx - candidate_indices[0]], v[idx - candidate_indices[0]], idx, dist_to_center / weight))

    # Select up to n_landmarks
    landmarks_visible.sort(key=lambda x: x[3])
    landmarks_visible = [(u, v, idx) for u, v, idx, _ in landmarks_visible[:n_landmarks]]

    # Initialize measurements
    n_meas = 2 + 2 * len(landmarks_visible)
    H = np.zeros((n_meas, n_state))
    R = np.zeros((n_meas, n_meas))

    # LIDAR measurements
    if r_intersect is not None:
        range_meas = min_t
        range_rate = np.dot(velocity, beam_asteroid / np.linalg.norm(beam_asteroid))
        H[0, 0:3] = beam_asteroid  # drange/dx
        H[1, 3:6] = beam_asteroid / np.linalg.norm(beam_asteroid)  # drangerate/dv
        R[0, 0] = range_noise**2
        R[1, 1] = range_rate_noise**2
    else:
        R[0, 0] = sigma_range**2
        R[1, 1] = sigma_range_rate**2

    # Camera measurements
    for i, (u, v, idx) in enumerate(landmarks_visible):
        lm = MESH_VERTICES[idx]
        lm_cam = R_asteroid_to_body @ (lm - position)
        z_cam = lm_cam[2]
        u_undist = fx * lm_cam[0] / z_cam + cx
        v_undist = fy * lm_cam[1] / z_cam + cy
        x_n = (u_undist - cx) / fx
        y_n = (v_undist - cy) / fy
        r2 = x_n**2 + y_n**2
        H[2 + 2*i, 0:3] = -R_asteroid_to_body @ np.array([
            [fx/z_cam, 0, -fx*lm_cam[0]/z_cam**2],
            [0, fy/z_cam, -fy*lm_cam[1]/z_cam**2]
        ])[:, :3]  # du/dx, dv/dx
        pixel_noise = sigma_pixel * (1 + 0.1 * r2)
        R[2 + 2*i, 2 + 2*i] = pixel_noise**2
        R[3 + 2*i, 3 + 2*i] = pixel_noise**2

    return H, R, r_intersect, landmarks_visible