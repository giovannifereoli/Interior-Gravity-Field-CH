import numpy as np
import trimesh


class Spacecraft:
    def __init__(
        self, position: np.ndarray, orientation: np.ndarray, velocity: np.ndarray = None
    ):
        """
        Spacecraft representation.

        position:  (3,) km in body-fixed frame
        orientation: (3,3) rotation matrix from body frame to camera frame
        velocity:  (3,) km/s in body-fixed frame (optional)
        """
        self.position = position
        self.orientation = orientation
        self.velocity = velocity if velocity is not None else np.zeros(3)


def measurement_and_analytic(
    spacecraft: Spacecraft,
    mesh_file: str,
    camera_intrinsics: dict,
    lon0_deg: float,
    lat0_deg: float,
    angular_radius_deg: float,
    num_landmarks: int = 50,
):
    """
    Analytic measurement model + Jacobian for camera landmarks and nadir altimeter.

    Inputs
    ------
    spacecraft: Spacecraft
    mesh_file: path to asteroid .obj
    camera_intrinsics: dict with fx, fy, cx, cy
    lon0_deg, lat0_deg: center of landmark region in degrees
    angular_radius_deg: angular radius in degrees
    num_landmarks: max number of landmarks in region

    Outputs
    -------
    h: (2*M + 1,) measurement vector [u1,v1,...,uM,vM, alt]'
    H: (2*M + 1, 6) analytic Jacobian w.r.t [px,py,pz,vx,vy,vz]
    """
    # Load mesh
    mesh = trimesh.load(mesh_file, process=False)
    verts = mesh.vertices  # (N,3)

    # Compute unit vectors of vertices
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    uv = verts / norms

    # Compute input direction unit vector
    lon0 = np.deg2rad(lon0_deg)
    lat0 = np.deg2rad(lat0_deg)
    uv0 = np.array(
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)]
    )

    # Angular selection
    cos_r = np.cos(np.deg2rad(angular_radius_deg))
    mask_ang = (uv @ uv0) >= cos_r
    idxs = np.where(mask_ang)[0]
    if idxs.size == 0:
        raise ValueError("No landmarks in specified angular region.")

    # Subsample landmarks
    chosen = np.random.choice(idxs, size=min(num_landmarks, idxs.size), replace=False)
    landmarks_body = verts[chosen]  # (M,3)

    # Prep intrinsics and pose
    fx, fy = camera_intrinsics["fx"], camera_intrinsics["fy"]
    cx, cy = camera_intrinsics["cx"], camera_intrinsics["cy"]
    R = spacecraft.orientation  # body->cam
    p = spacecraft.position

    # Compute relative and camera-frame coords
    rel = landmarks_body - p[np.newaxis, :]  # (M,3)
    y = (R @ rel.T).T  # (M,3)

    # Visibility mask (in front of camera)
    vis = y[:, 2] > 0
    y_vis = y[vis]
    M = y_vis.shape[0]

    # Build measurements
    u = fx * y_vis[:, 0] / y_vis[:, 2] + cx
    v = fy * y_vis[:, 1] / y_vis[:, 2] + cy
    # Altimeter range
    r_norm = np.linalg.norm(p)
    nadir_dir = -p / r_norm
    locations, _, _ = mesh.ray.intersects_location(origins=[p], directions=[nadir_dir])
    if len(locations) > 0:
        alt = np.min(np.linalg.norm(locations - p, axis=1))
    else:
        alt = np.nan

    h = np.hstack([u, v, alt])  # (2*M+1,)

    # Build analytic Jacobian H
    H = np.zeros((2 * M + 1, 6))

    # Pixel partials
    for i in range(M):
        yi = y_vis[i]
        yi_x, yi_y, yi_z = yi
        # projection Jacobian w.r.t yi
        J_proj = np.array(
            [
                [fx / yi_z, 0, -fx * yi_x / (yi_z**2)],
                [0, fy / yi_z, -fy * yi_y / (yi_z**2)],
            ]
        )  # (2,3)
        # ∂yi/∂p = -R -> ∂pix/∂p = J_proj @ (-R)
        H[2 * i : 2 * i + 2, 0:3] = -J_proj.dot(R)
        # ∂/∂v = 0 (already zero)

    # Altimeter partial: ∂alt/∂p ≈ p/||p||
    H[-1, 0:3] = p / r_norm
    # ∂alt/∂v = 0

    return h, H


# Example usage
if __name__ == "__main__":
    sc = Spacecraft(position=np.array([10.0, 5.0, 2.0]), orientation=np.eye(3))
    intrinsics = {"fx": 800, "fy": 800, "cx": 512, "cy": 512}
    h, H = measurement_and_analytic(
        sc,
        "path/to/asteroid.obj",
        intrinsics,
        lon0_deg=45.0,
        lat0_deg=10.0,
        angular_radius_deg=30.0,
        num_landmarks=30,
    )
    print("h:", h)
    print("H shape:", H.shape)
