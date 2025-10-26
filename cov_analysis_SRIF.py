import numpy as np
from scipy.special import jv as BesselJ, jvp as BesselJp, jn_zeros
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import cholesky, qr
import matplotlib as mpl
from datetime import datetime

# Set plotting style
COLOR_PALETTE = [
    "#d7191c",  # red
    "#fdae61",  # orange
    "#2c7bb6",  # dark blue
    "#abd9e9",  # light blue
    "#66c2a5",  # teal green
    "#3288bd",  # ocean blue
    "#9e0142",  # dark red
    "#fee08b",  # pale yellow
    "#5e4fa2",  # purple
    "#a6d96a",  # green
    "#1b7837",  # deep forest green
]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_PALETTE)
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"


# Define constants
CYLINDER_CENTER = np.array([0.0, 0.0, 0.28])
CYLINDER_RADIUS = 0.1
CYLINDER_ROTATION = np.eye(3)
ALPHA = 100
rotation_inv = np.linalg.inv(CYLINDER_ROTATION)

# NOTE: For J1, perform pure covariance propagation using Monte Carlo.
# It is going to be a paper on dynamics.  Perhaps CMDA.  Then you will
# write a paper on JGCD or something related to engineering applications,
# complete with cases.  There, you can run multiple cases while also
# implementing the actual variable density (i.e., increase priori P for sure).
# Perhaps the first is about pinpoint landing with OpNav and altimeter,
# and the second is about using a transponder and stationkeeping to better
# fit those coefficients in order to estimate a mascon beneath the surface.
# You can also collaborate and use MCMC or optical flow to complete the circle.
# I really like this!

# NOTE: Your trajectory may not break azimuthal symmetry enough — e.g., it may favor odd
# m harmonics if your path spirals or moves more in rho than phi.

# TODO: Why m=2,4 doesnt improve? Is the trajectory?
# TODO: pick realistic and good sensor suite, implement realistic models, cadence and noise, minimum height, etc.
# TODO: Covariance realism isn't the aim here (consider convariance or SNC), it's just to see how those parameters estimation evolve. Does Jay agree?
# TODO: Check A and rotations, math in general, pipeline, etc.
# TODO: optical flow exploitation?


def compute_A_and_sensitivities(
    rho, phi, z, m_vals, j_mn, A_coeff, B_coeff, alpha, Rstar
):
    """
    Compute cylindrical Jacobian A_cyl and per-coefficient sensitivities for acceleration components.
    """
    fac = alpha * Rstar
    x = (j_mn * rho) / fac
    exp_term = np.exp(-j_mn * z / fac)
    Jm = BesselJ(m_vals, x)
    Jm_p = BesselJp(m_vals, x, 1)
    Jm_pp = BesselJp(m_vals, x, 2)

    cos_mphi = np.cos(m_vals * phi)
    sin_mphi = np.sin(m_vals * phi)

    d2_drho2 = np.sum(
        exp_term * (j_mn / fac) ** 2 * Jm_pp * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )
    d2_dphi_drho = np.sum(
        exp_term
        * (j_mn / fac)
        * Jm_p
        * m_vals
        * (-A_coeff * sin_mphi + B_coeff * cos_mphi)
    )
    d2_dz_drho = -np.sum(
        (j_mn / fac) ** 2 * exp_term * Jm_p * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )

    dPhi_dphi = np.sum(
        exp_term * Jm * m_vals * (-A_coeff * sin_mphi + B_coeff * cos_mphi)
    )
    d2_dphi2 = -np.sum(
        exp_term * Jm * (m_vals**2) * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )
    d2_dz_dphi = np.sum(
        (j_mn / fac)
        * exp_term
        * Jm
        * m_vals
        * (A_coeff * sin_mphi - B_coeff * cos_mphi)
    )

    d2_dz2 = np.sum(
        (j_mn / fac) ** 2 * exp_term * Jm * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )

    A_cyl = np.array(
        [
            [d2_drho2, d2_dphi_drho, d2_dz_drho],
            [
                -dPhi_dphi / (rho**2) + d2_dphi_drho / (rho),
                d2_dphi2 / (rho),
                d2_dz_dphi / (rho),
            ],
            [d2_dz_drho, d2_dz_dphi, d2_dz2],
        ]
    )

    sens = {
        "a_rho_A": exp_term * (j_mn / fac) * Jm_p * cos_mphi,
        "a_rho_B": exp_term * (j_mn / fac) * Jm_p * sin_mphi,
        "a_phi_A": -(1 / (rho)) * exp_term * Jm * m_vals * sin_mphi,
        "a_phi_B": (1 / (rho)) * exp_term * Jm * m_vals * cos_mphi,
        "a_z_A": -(j_mn / fac) * exp_term * Jm * cos_mphi,
        "a_z_B": -(j_mn / fac) * exp_term * Jm * sin_mphi,
    }

    return A_cyl, sens


def cylindrical_to_cartesian_jacobian(A_cyl, a_cyl, phi):
    """
    Transform 3x3 cylindrical Jacobian and acceleration into Cartesian Jacobian.
    """
    T = np.array(
        [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )
    T_T = T.T
    a_phi = a_cyl[1]
    T_tensor = a_phi * np.array(
        [[-np.sin(phi), np.cos(phi), 0], [-np.cos(phi), -np.sin(phi), 0], [0, 0, 0]]
    )
    return (T_tensor + T_T @ A_cyl) @ T


def rotate_sensitivity_cylindrical(J_theta_cyl, phi):
    """
    Rotate a (3xK) cylindrical sensitivity matrix into Cartesian coordinates.
    """
    T = np.array(
        [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )
    return T.T @ J_theta_cyl


def compute_acceleration(position, fitted_params, n_n, n_m, j_mn_cache):
    """
    Compute acceleration in Cartesian coordinates at a given position.
    """
    pt = np.array(position)
    transformed_point = (pt - CYLINDER_CENTER) @ rotation_inv
    rho = np.linalg.norm(transformed_point[:2])
    phi = np.arctan2(transformed_point[1], transformed_point[0])
    z = transformed_point[2]

    m_vals = np.repeat(np.arange(n_m), n_n)
    j_mn = j_mn_cache
    A_coeff = fitted_params[0::2]
    B_coeff = fitted_params[1::2]

    a_cyl = np.zeros(3)
    fac = ALPHA * CYLINDER_RADIUS
    x = (j_mn * rho) / fac
    exp_term = np.exp(-j_mn * z / fac)
    Jm = BesselJ(m_vals, x)
    Jm_p = BesselJp(m_vals, x, 1)
    cos_mphi = np.cos(m_vals * phi)
    sin_mphi = np.sin(m_vals * phi)
    a_cyl[0] = np.sum(
        exp_term * (j_mn / fac) * Jm_p * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )
    a_cyl[1] = np.sum(
        (1 / (rho))
        * exp_term
        * Jm
        * m_vals
        * (-A_coeff * sin_mphi + B_coeff * cos_mphi)
    )
    a_cyl[2] = np.sum(
        -(j_mn / fac) * exp_term * Jm * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )

    a_rho, a_phi, a_z = a_cyl
    a_x = a_rho * np.cos(phi) - a_phi * np.sin(phi)
    a_y = a_rho * np.sin(phi) + a_phi * np.cos(phi)
    a_z = a_z
    accel_cart = np.array([a_x, a_y, a_z]) @ CYLINDER_ROTATION
    return accel_cart


def compute_dynamical_matrix(position, fitted_params, n_n, n_m, j_mn_cache):
    """
    Compute the dynamical matrix A for the state [position, velocity, coefficients].
    """
    pt = np.array(position)
    transformed_point = (pt - CYLINDER_CENTER) @ rotation_inv
    rho = np.linalg.norm(transformed_point[:2])
    phi = np.arctan2(transformed_point[1], transformed_point[0])
    z = transformed_point[2]

    m_vals = np.repeat(np.arange(n_m), n_n)
    j_mn = j_mn_cache
    A_coeff = fitted_params[0::2]
    B_coeff = fitted_params[1::2]

    a_cyl = np.zeros(3)
    fac = ALPHA * CYLINDER_RADIUS
    x = (j_mn * rho) / fac
    exp_term = np.exp(-j_mn * z / fac)
    Jm = BesselJ(m_vals, x)
    Jm_p = BesselJp(m_vals, x, 1)
    cos_mphi = np.cos(m_vals * phi)
    sin_mphi = np.sin(m_vals * phi)
    a_cyl[0] = np.sum(
        exp_term * (j_mn / fac) * Jm_p * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )
    a_cyl[1] = np.sum(
        (1 / (rho))
        * exp_term
        * Jm
        * m_vals
        * (-A_coeff * sin_mphi + B_coeff * cos_mphi)
    )
    a_cyl[2] = np.sum(
        -(j_mn / fac) * exp_term * Jm * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )

    A_cyl, sens_cyl = compute_A_and_sensitivities(
        rho, phi, z, m_vals, j_mn, A_coeff, B_coeff, ALPHA, CYLINDER_RADIUS
    )

    J_cart = cylindrical_to_cartesian_jacobian(A_cyl, a_cyl, phi)

    J_theta_cyl = np.vstack(
        [
            np.hstack([sens_cyl["a_rho_A"], sens_cyl["a_rho_B"]]),
            np.hstack([sens_cyl["a_phi_A"], sens_cyl["a_phi_B"]]),
            np.hstack([sens_cyl["a_z_A"], sens_cyl["a_z_B"]]),
        ]
    )
    J_theta_cart = rotate_sensitivity_cylindrical(J_theta_cyl, phi)

    K = 2 * n_n * n_m
    A = np.zeros((6 + K, 6 + K))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = J_cart
    A[3:6, 6:] = J_theta_cart

    return A


def compute_measurement_partials(position, n_state, fx=1000, fy=1000):
    """
    Compute the measurement model Jacobian H and noise covariance R
    for range + optical navigation (pixel, line) measurements of CYLINDER_CENTER.

    Args:
        position: Camera position in inertial frame [x, y, z].
        n_state: State dimension (6 + params).
        fx, fy: Focal lengths in pixels.

    Returns:
        H: 3xN Jacobian matrix [drange/dstate, dpixel/dstate, dline/dstate]
        R: 3x3 measurement noise covariance matrix.
    """
    # Vector from camera to landmark (center of cylinder)
    dx, dy, dz = CYLINDER_CENTER - position
    rho_sq = dx**2 + dy**2 + dz**2
    rho = np.sqrt(rho_sq)

    # Projected pixel and line in image plane
    xp = fx * dx / dz
    yp = fy * dy / dz

    # Initialize measurement Jacobian
    H = np.zeros((3, n_state))

    # Range partials
    H[0, 0] = -dx / rho
    H[0, 1] = -dy / rho
    H[0, 2] = -dz / rho

    # Pixel partials (dxp/dx)
    H[1, 0] = -fx / dz
    H[1, 2] = fx * dx / dz**2

    # Line partials (dyp/dx)
    H[2, 1] = -fy / dz
    H[2, 2] = fy * dy / dz**2

    # Measurement noise covariance
    R = np.diag(
        [
            (1e-4) ** 2,  # Range [LU^2]
            (0.2) ** 2,  # Pixel noise [pixels^2]
            (0.2) ** 2,  # Line noise [pixels^2]
        ]
    )

    return H, R


def compute_measurement_partials2(position, n_state):
    """
    Compute the measurement model Jacobian H and noise covariance R for range and angular measurements.

    Args:
        position: Cartesian position [x, y, z].
        n_state: Total state dimension (6 + 2*n_n*n_m).

    Returns:
        H: 3x1256 Jacobian matrix [drange/dstate, dtheta/dstate, dphi/dstate].
        R: 3x3 measurement noise covariance matrix.
    """
    x, y, z = position
    dx, dy, dz = x - CYLINDER_CENTER[0], y - CYLINDER_CENTER[1], z - CYLINDER_CENTER[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r_xy = np.sqrt(dx**2 + dy**2)

    # Measurement partials
    H = np.zeros((3, n_state))

    # Range partials: r = sqrt(x^2 + y^2 + (z - z_c)^2)
    H[0, 0] = dx / r  # dr/dx
    H[0, 1] = dy / r  # dr/dy
    H[0, 2] = dz / r  # dr/dz

    # Azimuth partials: theta = atan2(y, x)
    H[1, 0] = -dy / (r_xy**2)  # dtheta/dx
    H[1, 1] = dx / (r_xy**2)  # dtheta/dy
    H[1, 2] = 0  # dtheta/dz = 0

    # Elevation partials: phi = arcsin((z - z_c) / r)
    cos_phi = np.sqrt(r**2 - dz**2) / r if r > dz else 0
    H[2, 0] = -dx * dz / (r**2 * cos_phi)  # dphi/dx
    H[2, 1] = -dy * dz / (r**2 * cos_phi)  # dphi/dy
    H[2, 2] = (r**2 - dz**2) / (r**3 * cos_phi)  # dphi/dz

    # Measurement noise covariance
    R = np.diag([(1e-4) ** 2, 0.001**2, 0.001**2])  # [LU^2, rad^2, rad^2]

    return H, R


def propagate_state_and_stm(initial_state, fitted_params, n_n, n_m, t_span):
    """
    Propagate the state and STM using solve_ivp, with a tqdm progress bar.
    """
    n_state = 6 + 2 * n_n * n_m
    stm0 = np.eye(n_state).ravel()
    y0 = np.hstack((initial_state, stm0))

    Ts = [t_span[0]]
    states = [initial_state.copy()]
    stms = [np.eye(n_state)]

    t0, tf = float(t_span[0]), float(t_span[-1])
    total_dt = tf - t0

    with tqdm(
        total=total_dt,
        desc="Propagating",
        unit="s",
        ncols=80,
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}{unit} [{elapsed}<{remaining}]",
    ) as pbar:
        t_current = t0
        y_current = y0

        for t_next in t_span[1:]:
            sol = solve_ivp(
                fun=lambda t, y: _dynamics_full(
                    t, y, n_state, fitted_params, n_n, n_m, j_mn_cache
                ),
                t_span=(t_current, t_next),
                y0=y_current,
                method="RK45",
                rtol=1e-10,
                atol=1e-10,
            )

            y_end = sol.y[:, -1]
            state_end = y_end[:n_state]
            stm_end = y_end[n_state:].reshape((n_state, n_state))

            Ts.append(t_next)
            states.append(state_end)
            stms.append(stm_end)

            dt = t_next - t_current
            pbar.update(dt)
            t_current = t_next
            y_current = y_end

    t = np.array(Ts)
    state = np.stack(states, axis=1)
    stm = np.stack(stms, axis=2)

    return t, state, stm


def _dynamics_full(t, y, n_state, fitted_params, n_n, n_m, j_mn_cache):
    """
    Returns the time-derivative of [state; STM] for solve_ivp.
    """
    state = y[:n_state]
    stm_mat = y[n_state:].reshape((n_state, n_state))
    pos = state[0:3]
    vel = state[3:6]
    a = compute_acceleration(pos, fitted_params, n_n, n_m, j_mn_cache)
    A = compute_dynamical_matrix(pos, fitted_params, n_n, n_m, j_mn_cache)
    state_dot = np.hstack((vel, a, np.zeros(2 * n_n * n_m)))
    stm_dot = (A @ stm_mat).ravel()
    return np.hstack((state_dot, stm_dot))


if __name__ == "__main__":
    # Load fitted parameters
    fitted_params = np.load("fitted_params_both.npy")
    full_cov_params = np.load("covariance_matrix.npy")
    full_cov_params[full_cov_params < 1e-30] = 1e-128  # For B_0n coefficients
    print("Loaded fitted parameters from 'fitted_params_both.npy'")

    n_n, n_m = 5, 5
    j_mn_cache = np.array(
        [jn_zeros(m, n + 1)[-1] for m in range(n_m) for n in range(n_n)]
    )
    initial_position = np.array([-0.0545118663, -0.0608104828, 0.729726385])
    initial_velocity = np.array([9.74202292e-07, 1.09203903e-06, -7.28180036e-06])
    initial_coeffs = fitted_params
    initial_state = np.hstack((initial_position, initial_velocity, initial_coeffs))

    # Initialize covariance matrix
    n_state = 6 + 2 * n_n * n_m
    P0 = np.zeros((n_state, n_state))
    P0[:3, :3] = np.eye(3) * (1e-3) ** 2  # Position variance: 1e-3 LU
    P0[3:6, 3:6] = np.eye(3) * (1e-3) ** 2  # Velocity variance: 1e-3 LU/s
    P0[6:, 6:] = np.diag(np.diag(full_cov_params))

    stop_at_percent = 0.985
    t_span = np.linspace(0, stop_at_percent * 55000, 100)  # 1 Hz sampling

    t, state, stm = propagate_state_and_stm(
        initial_state, fitted_params, n_n, n_m, t_span
    )

    # Initialize square root information matrix using solve, not inv
    print("Propagating and updating SRIF covariance matrix...")

    # Initialize SRIF variables
    n_state = P0.shape[0]
    n_steps = len(t)

    # Initialize square root information matrix
    R_sqrt = cholesky(np.linalg.inv(P0), lower=False)

    STM_tm = np.eye(n_state)
    P = np.empty((n_state, n_state, n_steps))
    P[:, :, 0] = P0

    # Precompute identity matrix
    I_n = np.eye(n_state)

    # Precompute measurement whitening matrix
    _, R_meas = compute_measurement_partials(state[:3, 1], n_state)
    SRI = cholesky(np.linalg.inv(R_meas), lower=False)

    for i in tqdm(range(1, n_steps), desc="SRIF", ncols=80):
        # Compute the STM for the current step
        Phi = stm[:, :, i] @ np.linalg.inv(STM_tm)
        STM_tm = stm[:, :, i]

        # Prediction step
        pred_matrix = R_sqrt @ np.linalg.inv(Phi)
        _, RQ = qr(pred_matrix, mode="economic")
        R_sqrt = RQ[:n_state, :n_state]

        # Measurement update
        H, _ = compute_measurement_partials(state[:3, i], n_state)
        H_w = SRI @ H
        update_matrix = np.vstack([R_sqrt, H_w])
        Q, R_combined = qr(update_matrix, mode="economic")
        R_sqrt = R_combined[:n_state, :n_state]

        # Covariance reconstruction
        RtR = R_sqrt.T @ R_sqrt
        P[:, :, i] = np.linalg.inv(RtR)

    print("Propagation and update completed.")

    # Plotting results
    fig1 = plt.figure(figsize=(14, 5))

    # Trajectory
    ax1 = fig1.add_subplot(131, projection="3d")
    ax1.plot(
        state[0, :],
        state[1, :],
        state[2, :],
        color=COLOR_PALETTE[0],
        label="Trajectory",
    )
    ax1.scatter(
        [0],
        [0],
        [CYLINDER_CENTER[2]],
        color=COLOR_PALETTE[1],
        s=100,
        label="Cylinder Center",
    )
    theta = np.linspace(0, 2 * np.pi, 50)
    z_cyl = np.linspace(CYLINDER_CENTER[2], initial_position[2], 10)
    theta, z_cyl = np.meshgrid(theta, z_cyl)
    x_cyl = CYLINDER_RADIUS * np.cos(theta)
    y_cyl = CYLINDER_RADIUS * np.sin(theta)
    ax1.plot_wireframe(x_cyl, y_cyl, z_cyl, color=COLOR_PALETTE[1], alpha=0.3)
    ax1.set_xlabel(r"$X$ (LU)", fontsize=12)
    ax1.set_ylabel(r"$Y$ (LU)", fontsize=12)
    ax1.set_zlabel(r"$Z$ (LU)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    ax1.set_aspect("equal")

    # Position vs Time
    ax2 = fig1.add_subplot(132)
    ax2.plot(t / 3600, state[0, :], label=r"$x$", color=COLOR_PALETTE[0])
    ax2.plot(t / 3600, state[1, :], label=r"$y$", color=COLOR_PALETTE[1])
    ax2.plot(t / 3600, state[2, :], label=r"$z$", color=COLOR_PALETTE[2])
    ax2.set_xlabel(r"Time (hr)", fontsize=12)
    ax2.set_ylabel(r"Position (LU)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)

    # Velocity vs Time
    ax3 = fig1.add_subplot(133)
    ax3.plot(t / 3600, state[3, :], label=r"$v_x$", color=COLOR_PALETTE[0])
    ax3.plot(t / 3600, state[4, :], label=r"$v_y$", color=COLOR_PALETTE[1])
    ax3.plot(t / 3600, state[5, :], label=r"$v_z$", color=COLOR_PALETTE[2])
    ax3.set_xlabel(r"Time (hr)", fontsize=12)
    ax3.set_ylabel(r"Velocity (LU/s)", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True)
    plt.tight_layout()
    fig1.savefig(
        f"Images/fig_trajectory_position_velocity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        dpi=1200,
        bbox_inches="tight",
    )

    # Plot covariance results
    fig2 = plt.figure(figsize=(14, 5))

    # Position Uncertainty
    ax4 = fig2.add_subplot(131)
    ax4.scatter(
        t / 3600, np.sqrt(P[0, 0, :]), label=r"$\sigma_x$", s=8, color=COLOR_PALETTE[0]
    )
    ax4.scatter(
        t / 3600, np.sqrt(P[1, 1, :]), label=r"$\sigma_y$", s=8, color=COLOR_PALETTE[1]
    )
    ax4.scatter(
        t / 3600, np.sqrt(P[2, 2, :]), label=r"$\sigma_z$", s=8, color=COLOR_PALETTE[2]
    )
    ax4.set_yscale("log")
    ax4.set_xlabel(r"Time (hr)", fontsize=12)
    ax4.set_ylabel(r"Std. Dev. (LU)", fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, which="both")

    # Velocity Uncertainty
    ax5 = fig2.add_subplot(132)
    ax5.scatter(
        t / 3600,
        np.sqrt(P[3, 3, :]),
        label=r"$\sigma_{v_x}$",
        s=8,
        marker="x",
        color=COLOR_PALETTE[0],
    )
    ax5.scatter(
        t / 3600,
        np.sqrt(P[4, 4, :]),
        label=r"$\sigma_{v_y}$",
        s=8,
        marker="x",
        color=COLOR_PALETTE[1],
    )
    ax5.scatter(
        t / 3600,
        np.sqrt(P[5, 5, :]),
        label=r"$\sigma_{v_z}$",
        s=8,
        marker="x",
        color=COLOR_PALETTE[2],
    )
    ax5.set_yscale("log")
    ax5.set_xlabel(r"Time (hr)", fontsize=12)
    ax5.set_ylabel(r"Std. Dev. (LU/s)", fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, which="both")

    # Coefficient RMS by m
    ax6 = fig2.add_subplot(133)
    coeff_std = np.zeros((2 * n_n * n_m, len(t)))
    for i in range(2 * n_n * n_m):
        coeff_std[i, :] = np.sqrt(P[6 + i, 6 + i, :])
    rms_by_m = []
    for m in range(n_m):
        idx_start = m * 2 * n_n
        idx_end = idx_start + 2 * n_n
        coeff_std_m = coeff_std[idx_start:idx_end, :]  # shape: (2*n_n, len(t))
        rms_m = np.sqrt(np.mean(coeff_std_m**2, axis=0))  # RMS over A/B for fixed m
        rms_by_m.append(rms_m)
    rms_by_m = np.array(rms_by_m)  # shape: (n_m, len(t))
    for m in range(n_m):
        ax6.plot(
            t / 3600,  # Convert time to hours
            rms_by_m[m],
            marker="o",
            markersize=2,
            linestyle="None",
            label=rf"$m={m}$",
        )
    ax6.set_yscale("log")
    ax6.set_xlabel(r"Time (hr)", fontsize=12)
    ax6.set_ylabel(r"RMS Std. Dev. (-)", fontsize=12)
    ax6.legend(fontsize=10, ncol=3)
    ax6.grid(True, which="both")
    plt.tight_layout()
    fig2.savefig(
        f"Images/fig_uncertainty_rmsr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    plt.show()

    # Compute the RMS of signal and noise by spectral order m and plot signal-to-noise ratio
    final_P = P[:, :, -1]
    coeff_std = np.sqrt(np.diag(final_P)[6:])  # 6+ onward are cylindrical coeffs
    fitted_params = fitted_params[: 2 * n_n * n_m]
    signal_rms_by_m = []
    noise_rms_by_m = []
    snr_by_m = []

    for m in range(n_m):
        idx_start = m * 2 * n_n
        idx_end = idx_start + 2 * n_n
        coeffs_m = fitted_params[idx_start:idx_end]
        stds_m = coeff_std[idx_start:idx_end]

        signal_rms = np.sqrt(np.sum(coeffs_m**2))
        noise_rms = np.sqrt(np.sum(stds_m**2))
        snr = signal_rms / noise_rms if noise_rms > 0 else np.nan

        signal_rms_by_m.append(signal_rms)
        noise_rms_by_m.append(noise_rms)
        snr_by_m.append(snr)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(n_m),
        signal_rms_by_m,
        marker="o",
        label="Signal RMS",
        color=COLOR_PALETTE[0],
    )
    plt.plot(
        range(n_m),
        noise_rms_by_m,
        marker="x",
        label="Noise RMS",
        color=COLOR_PALETTE[1],
    )
    plt.plot(range(n_m), snr_by_m, marker="s", label="SNR", color=COLOR_PALETTE[2])
    plt.yscale("log")
    plt.xlabel(r"Spectral Order $m$ (-)", fontsize=12)
    plt.ylabel(r"RMS / SNR (-)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(
        f"Images/fig_signal_noise_snr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    plt.show()
