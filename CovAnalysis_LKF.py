import numpy as np
import filterpy.kalman
import filterpy.common
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
import mesh_utility
from scipy.special import (
    jv as BesselJ,
    jvp as BesselJp,
    jn_zeros,
)
from tqdm import tqdm
import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl

# Constants
MU_ITOKAWA_TRUE = 2.36e-9  # km^3/s^2
MU_ITOKAWA_EST = 1.5e-9  # Initial estimate
R_ITOKAWA = 0.167  # km (approx radius)
DENSITY = 1.0  # Asteroid density

# Use a colorblind-friendly color palette
COLOR_PALETTE = ["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_PALETTE)

# Set LaTeX formatting
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

# Parameters
n_n, n_m = 25, 25  # Truncation parameters
fitted_params = np.load("fitted_params_both.npy")  # Fitted coefficients
dt, t_max = 120, 55000  # seconds
times = np.arange(0, t_max, dt)

# Meshes from https://github.com/darioizzo/geodesyNets/tree/master/3dmeshes
vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
vertices_lp, faces_lp = mesh_utility.read_pk_file("3dmeshes/eros_lp.pk")
vertices, faces = np.array(vertices), np.array(faces)

# Initialize the polyhedron object
eros = Polyhedron(
    polyhedral_source=(vertices, faces),
    density=DENSITY,
    integrity_check=PolyhedronIntegrity.DISABLE,
)

# Create an evaluable object for gravity calculations
evaluable_eros = GravityEvaluable(eros)

# Random Seed
np.random.seed(42)

# Initial spacecraft state
sc_state_0 = np.array(
    [
        -5.45118663e-02,
        -6.08104828e-02,
        7.29726385e-01,
        9.74202292e-07,
        1.09203903e-06,
        -7.28180036e-06,
        MU_ITOKAWA_EST,
    ]
)

# Process and measurement noise
Q = np.eye(7) * 1e-6
R = np.eye(2) * 1e-1

# Define cylinder parameters
CYLINDER_CENTER = np.array([0.0, 0.0, 0.28])  # Center of the cylinder base in XYZ
CYLINDER_HEIGHT = 0.5  # Height of the cylinder in meters
CYLINDER_RADIUS = 0.1  # Radius of the cylinder in meters
CYLINDER_ROTATION = np.eye(3)  # Rotation matrix (identity matrix by default)
NUM_POINTS = 1000  # Number of points to generate
ALPHA = 100  # Scaling parameter for the cylinder


def cylindrical_to_cartesian_acceleration(points, cylindrical_accelerations):
    """
    Convert cylindrical acceleration components to Cartesian coordinates.

    Args:
        points: Cartesian points (N, 3).
        cylindrical_accelerations: Cylindrical accelerations (N, 3): [a_rho, a_phi, a_z].
        cylinder_center: Center of the cylinder in Cartesian coordinates (3,).
        cylinder_rotation: Rotation matrix of the cylinder (3, 3).

    Returns:
        Cartesian accelerations (N, 3).
    """
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])

    a_rho, a_phi, a_z = (
        cylindrical_accelerations[:, 0],
        cylindrical_accelerations[:, 1],
        cylindrical_accelerations[:, 2],
    )
    a_x = a_rho * np.cos(phi) - a_phi * np.sin(phi)
    a_y = a_rho * np.sin(phi) + a_phi * np.cos(phi)

    return np.column_stack((a_x, a_y, a_z))


def compute_fitted_cylindrical_acceleration(points, fitted_params, n_n, n_m):
    """
    Compute the fitted acceleration in cylindrical coordinates using the fitted parameters.

    Args:
        points: Cartesian points (N, 3).
        fitted_params: Fitted coefficients (1D array).
        n_n, n_m: Truncation parameters.

    Returns:
        Fitted accelerations in cylindrical coordinates (N, 3): [a_rho, a_phi, a_z].
    """
    R, L = CYLINDER_RADIUS, CYLINDER_HEIGHT
    transformed_points = (points - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.sqrt(transformed_points[:, 0] ** 2 + transformed_points[:, 1] ** 2)
    phi = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
    z = transformed_points[:, 2]

    num_points = len(points)
    fitted_acceleration = np.zeros((num_points, 3))  # Cylindrical: [a_rho, a_phi, a_z]

    k = lambda m, n: jn_zeros(m, n)[-1]
    R_alpha = ALPHA * CYLINDER_RADIUS

    idx = 0
    for m in range(n_m):
        for n in range(1, n_n + 1):
            k_mn = k(m, n)

            # Cylindrical basis function components
            exp_term = np.exp(-k_mn * z / R_alpha)
            bessel_j = BesselJ(m, k_mn * rho / R_alpha)
            bessel_j_derivative = BesselJp(m, k_mn * rho / R_alpha)
            cos_m_phi = np.cos(m * phi)
            sin_m_phi = np.sin(m * phi)

            # Compute contributions to acceleration components
            dV_drho = (k_mn / R_alpha) * exp_term * bessel_j_derivative
            dV_dphi = (m / (rho + 1e-14)) * exp_term * bessel_j
            dV_dz = (-k_mn / R_alpha) * exp_term * bessel_j

            # Add contributions from coefficients
            fitted_acceleration[:, 0] += (
                dV_drho * fitted_params[idx] * cos_m_phi
                + dV_drho * fitted_params[idx + 1] * sin_m_phi
            )
            fitted_acceleration[:, 1] += (
                dV_dphi * fitted_params[idx] * -sin_m_phi
                + dV_dphi * fitted_params[idx + 1] * cos_m_phi
            )
            fitted_acceleration[:, 2] += (
                dV_dz * fitted_params[idx] * cos_m_phi
                + dV_dz * fitted_params[idx + 1] * sin_m_phi
            )

            idx += 2

    return fitted_acceleration


def acceleration_fitted(position):
    """
    Compute the acceleration at a given position using fitted cylindrical harmonics.

    Parameters:
    position (np.ndarray): A 3-element array representing the Cartesian coordinates of the position.

    Returns:
    np.ndarray: A 3-element array representing the acceleration in Cartesian coordinates.
    """
    # Compute acceleration components in cylindrical coordinates
    a_rho, a_phi, a_z = compute_fitted_cylindrical_acceleration(
        np.array([position]), fitted_params, n_n, n_m
    )[0]

    # Convert acceleration back to Cartesian coordinates
    return cylindrical_to_cartesian_acceleration(
        np.array([position]), np.array([[a_rho, a_phi, a_z]])
    )[0]


### DYNAMICS MODEL
def dynamics(t, state):
    """Spacecraft dynamics with polyhedral gravity model."""
    r, v, mu = state[:3], state[3:6], state[6]
    accel = acceleration_fitted(r)
    return np.hstack((v, accel, 0))


### NUMERICAL STATE TRANSITION MATRIX (STM) COMPUTATION
def compute_state_transition_matrix(dynamics, x0, dt, eps=1e-12):
    """
    Compute the state transition matrix (STM) by propagating small perturbations in state.

    :param dynamics: Function defining state dynamics.
    :param x0: Initial state vector.
    :param dt: Time step for propagation.
    :param eps: Small perturbation for finite differences.
    :return: State transition matrix (STM) of size (n, n).
    """
    n = len(x0)
    stm = np.zeros((n, n))

    # Propagate the nominal state
    sol_nominal = solve_ivp(
        dynamics, t_span=(0, dt), y0=x0, method="LSODA", t_eval=[dt]
    )
    x_nominal = sol_nominal.y[:, -1]  # Extract final state

    # Compute STM column by column
    for i in range(n):
        x_perturbed = np.copy(x0)
        x_perturbed[i] += eps  # Perturb one state variable

        sol_perturbed = solve_ivp(
            dynamics, (0, dt), x_perturbed, method="LSODA", t_eval=[dt]
        )
        x_perturbed_final = sol_perturbed.y[:, -1]

        # Compute finite difference column
        stm[:, i] = (x_perturbed_final - x_nominal) / eps  # Column of STM

    return stm


### MEASUREMENT MODELS
def measure_altitude(state):
    """Measurement function: altitude (distance from asteroid center)."""
    altitude = np.linalg.norm(state[:3])
    return np.array([altitude])  # Return altitude as a measurement


def measure_los(state_s):
    """
    Compute the line-of-sight (LOS) measurement in the camera frame.

    :param state_s: Spacecraft state (6x1).
    :param state_p: Pod state (6x1).
    :param T_BCI_CAM: Direction cosine matrix (3x3).
    :return: LOS measurement [eta, zeta] (2x1).
    """
    # Compute relative position
    delta_r = state_s[:3]
    rho = np.linalg.norm(delta_r)  # Magnitude of relative position

    if rho < 1e-12:
        raise ValueError("Relative position vector norm is too small.")

    # Compute unit vector r_hat
    r_hat = delta_r / rho

    # Rotate to camera frame
    los_cam = np.eye(3) @ r_hat

    # Extract η (x-axis) and ζ (z-axis) components
    eta = los_cam[0]
    zeta = los_cam[2]

    return np.array([eta, zeta])


### JACOBIAN OF MEASUREMENT FUNCTION (H)
def compute_H_altitude(state):
    """
    Compute the exact Jacobian (H) for the altitude measurement model.
    :param state: Current state vector (7x1)
    :return: Measurement Jacobian H (1x7)
    """
    x, y, z = state[:3]
    r_norm = np.sqrt(x**2 + y**2 + z**2)

    # Prevent division by zero
    if r_norm < 1e-10:
        return np.zeros((1, 7))

    H = np.zeros((1, 7))
    H[0, 0] = x / r_norm  # dh/dx
    H[0, 1] = y / r_norm  # dh/dy
    H[0, 2] = z / r_norm  # dh/dz
    # Velocity (vx, vy, vz) and mu derivatives are zero
    return H


def compute_H_los(state_s):
    """
    Compute the Jacobian matrix for the LOS measurement model.

    :param state_s: Spacecraft state (6x1).
    :param state_p: Pod state (6x1).
    :param T_BCI_CAM: Direction cosine matrix (3x3).
    :return: Measurement Jacobian H (2x6).
    """
    # Compute relative position
    delta_r = state_s[:3]
    rho = np.linalg.norm(delta_r)  # Distance magnitude

    if rho < 1e-12:
        raise ValueError("Relative position vector norm is too small.")

    # Compute H_r
    H_r = np.array(
        [
            [
                1 / rho - (delta_r[0] ** 2 / rho**3),
                -delta_r[0] * delta_r[1] / rho**3,
                -delta_r[0] * delta_r[2] / rho**3,
                0,
                0,
                0,
                0,
            ],
            [
                -delta_r[0] * delta_r[1] / rho**3,
                1 / rho - (delta_r[1] ** 2 / rho**3),
                -delta_r[1] * delta_r[2] / rho**3,
                0,
                0,
                0,
                0,
            ],
            [
                -delta_r[0] * delta_r[2] / rho**3,
                -delta_r[1] * delta_r[2] / rho**3,
                1 / rho - (delta_r[2] ** 2 / rho**3),
                0,
                0,
                0,
                0,
            ],
        ]
    )

    # Compute full measurement Jacobian
    H = np.eye(3) @ H_r

    # Extract the first and third rows (η and ζ components)
    return H[[0, 2], :]


### COMPUTE FULL TRAJECTORY BEFORE ENTERING THE LOOP
sol = solve_ivp(dynamics, (0, t_max), sc_state_0, method="LSODA", t_eval=times)
trajectory = sol.y.T  # Store full trajectory (shape: (N, state_dim))

### PROPAGATION LOOP WITH STM AND COVARIANCE UPDATE
states = [sc_state_0]
covariances = [1e-6 * np.eye(7)]  # Initial covariance

for i, t in tqdm(
    enumerate(times[1:]),
    total=len(times[1:]),
    desc="Propagating Trajectory and Covariance",
):
    x_current = trajectory[i]  # Use precomputed trajectory
    P_current = covariances[-1]

    # Compute STM from x0 at t0 up to the current time t
    F = compute_state_transition_matrix(dynamics, sc_state_0, t)

    # Covariance propagation
    P_next = F @ P_current @ F.T + Q

    # Measurement update step
    H = compute_H_los(x_current)  # Use LOS measurement model
    R_k = R  # Measurement noise

    # Kalman Gain
    S = H @ P_next @ H.T + R_k
    K = P_next @ H.T @ np.linalg.inv(S)

    # Update covariance
    P_next = (np.eye(len(x_current)) - K @ H) @ P_next

    # Store results
    states.append(x_current)
    covariances.append(P_next)

mpl.rcParams["text.usetex"] = False  # Disable LaTeX if causing issues

# Convert covariance results to standard deviations (extracting diagonal elements)
covariances_diag = np.array([np.diag(P) for P in covariances])  # Shape: (N, 7)

# Create DataFrame
df_cov = pd.DataFrame(
    np.sqrt(covariances_diag),  # Convert variances to standard deviations
    columns=["σ_x", "σ_y", "σ_z", "σ_vx", "σ_vy", "σ_vz", "σ_mu"],
)
df_cov["Time (s)"] = times  # Add time column

# Plot covariance evolution
plt.figure(figsize=(10, 6))
for col in df_cov.columns[:-1]:
    plt.semilogy(df_cov["Time (s)"], df_cov[col], label=col)  # Already sqrt-ed

plt.xlabel("Time (s)")
plt.ylabel("Standard Deviation (sigma)")  # Corrected LaTeX syntax
plt.legend()
plt.title("Covariance Evolution - Standard Deviations")
plt.grid(True, which="both")
plt.show()


def plot_trajectories(times, states):
    """
    Plots the spacecraft trajectory in 3D space.

    Parameters:
    times (array-like): Time points for the trajectory.
    states (array-like): Trajectory data with shape (N, 7).

    Returns:
    None
    """

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    states = np.array(states)  # Ensure it's a NumPy array
    if states.shape[1] < 3:
        raise ValueError(
            "States array does not have enough dimensions for 3D plotting."
        )

    # Extract position components
    x, y, z = states[:, 0], states[:, 1], states[:, 2]

    # Plot trajectory
    ax.plot(x, y, z, label="Spacecraft Trajectory", linewidth=2, color="r")

    # Add labels and legend
    ax.set_xlabel("$X$ (km)")
    ax.set_ylabel("$Y$ (km)")
    ax.set_zlabel("$Z$ (km)")
    ax.legend()
    ax.set_title("Spacecraft Trajectory")

    plt.show()


# Call the function
plot_trajectories(times, states)
