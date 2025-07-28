import numpy as np
from scipy.special import jv as BesselJ, jvp as BesselJp, jn_zeros
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib.pyplot as plt


# Define constants
CYLINDER_CENTER = np.array([0.0, 0.0, 0.28])
CYLINDER_RADIUS = 0.1
CYLINDER_ROTATION = np.eye(3)
ALPHA = 100
rotation_inv = np.linalg.inv(CYLINDER_ROTATION)

def compute_A_and_sensitivities(rho, phi, z, m_vals, j_mn, A_coeff, B_coeff, alpha, Rstar):
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
                -dPhi_dphi / (rho**2 + 1e-14) + d2_dphi_drho / (rho + 1e-14),
                d2_dphi2 / (rho + 1e-14),
                d2_dz_dphi / (rho + 1e-14),
            ],
            [d2_dz_drho, d2_dz_dphi, d2_dz2],
        ]
    )

    sens = {
        "a_rho_A": exp_term * (j_mn / fac) * Jm_p * cos_mphi,
        "a_rho_B": exp_term * (j_mn / fac) * Jm_p * sin_mphi,
        "a_phi_A": -(1 / (rho + 1e-14)) * exp_term * Jm * m_vals * sin_mphi,
        "a_phi_B": (1 / (rho + 1e-14)) * exp_term * Jm * m_vals * cos_mphi,
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
    a_cyl[0] = np.sum(exp_term * (j_mn / fac) * Jm_p * (A_coeff * cos_mphi + B_coeff * sin_mphi))
    a_cyl[1] = np.sum((1 / (rho + 1e-14)) * exp_term * Jm * m_vals * (-A_coeff * sin_mphi + B_coeff * cos_mphi))
    a_cyl[2] = np.sum(-(j_mn / fac) * exp_term * Jm * (A_coeff * cos_mphi + B_coeff * sin_mphi))

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
    a_cyl[0] = np.sum(exp_term * (j_mn / fac) * Jm_p * (A_coeff * cos_mphi + B_coeff * sin_mphi))
    a_cyl[1] = np.sum((1 / (rho + 1e-14)) * exp_term * Jm * m_vals * (-A_coeff * sin_mphi + B_coeff * cos_mphi))
    a_cyl[2] = np.sum(-(j_mn / fac) * exp_term * Jm * (A_coeff * cos_mphi + B_coeff * sin_mphi))

    A_cyl, sens_cyl = compute_A_and_sensitivities(
        rho, phi, z, m_vals, j_mn, A_coeff, B_coeff, ALPHA, CYLINDER_RADIUS
    )

    J_cart = cylindrical_to_cartesian_jacobian(A_cyl, a_cyl, phi)

    J_theta_cyl = np.vstack([
        np.hstack([sens_cyl['a_rho_A'], sens_cyl['a_rho_B']]),
        np.hstack([sens_cyl['a_phi_A'], sens_cyl['a_phi_B']]),
        np.hstack([sens_cyl['a_z_A'], sens_cyl['a_z_B']])
    ])
    J_theta_cart = rotate_sensitivity_cylindrical(J_theta_cyl, phi)

    K = 2 * n_n * n_m
    A = np.zeros((6 + K, 6 + K))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = J_cart
    A[3:6, 6:] = J_theta_cart

    return A

def compute_measurement_partials(position, n_state):
    """
    Compute the measurement model Jacobian H and noise covariance R for range and angular measurements.

    Args:
        position: Cartesian position [x, y, z].
        n_state: Total state dimension (6 + 2*n_n*n_m).

    Returns:
        H: 3x1256 Jacobian matrix [∂range/∂state, ∂θ/∂state, ∂φ/∂state].
        R: 3x3 measurement noise covariance matrix.
    """
    x, y, z = position
    z_c = CYLINDER_CENTER[2]  # 0.28
    dx, dy, dz = x, y, z - z_c
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r_xy = np.sqrt(dx**2 + dy**2)

    # Measurement partials
    H = np.zeros((3, n_state))
    if r > 1e-10:  # Avoid division by zero
        # Range partials: r = sqrt(x² + y² + (z - z_c)²)
        H[0, 0] = dx / r  # ∂r/∂x
        H[0, 1] = dy / r  # ∂r/∂y
        H[0, 2] = dz / r  # ∂r/∂z

        # Azimuth partials: θ = atan2(y, x)
        if r_xy > 1e-10:
            H[1, 0] = -dy / (r_xy**2)  # ∂θ/∂x
            H[1, 1] = dx / (r_xy**2)   # ∂θ/∂y
            H[1, 2] = 0               # ∂θ/∂z = 0

        # Elevation partials: φ = arcsin((z - z_c) / r)
        cos_phi = np.sqrt(r**2 - dz**2) / r if r > dz else 0
        if cos_phi > 1e-10:
            H[2, 0] = -dx * dz / (r**2 * cos_phi)  # ∂φ/∂x
            H[2, 1] = -dy * dz / (r**2 * cos_phi)  # ∂φ/∂y
            H[2, 2] = (r**2 - dz**2) / (r**3 * cos_phi)  # ∂φ/∂z

    # Measurement noise covariance
    R = np.diag([0.01**2, 0.001**2, 0.001**2])  # [m², rad², rad²]

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
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}{unit} [{elapsed}<{remaining}]"
    ) as pbar:
        t_current = t0
        y_current = y0

        for t_next in t_span[1:]:
            sol = solve_ivp(
                fun=lambda t, y: _dynamics_full(t, y, n_state, fitted_params, n_n, n_m, j_mn_cache),
                t_span=(t_current, t_next),
                y0=y_current,
                method='RK45',
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
    print("Loaded fitted parameters from 'fitted_params_both.npy'")
    assert len(fitted_params) == 2 * 25 * 25, f"Expected 1250 coefficients, got {len(fitted_params)}"

    n_n, n_m = 25, 25
    j_mn_cache = np.array([jn_zeros(m, n+1)[-1] for m in range(n_m) for n in range(n_n)])
    initial_position = np.array([-0.0545118663, -0.0608104828, 0.729726385])
    initial_velocity = np.array([9.74202292e-07, 1.09203903e-06, -7.28180036e-06])
    initial_coeffs = fitted_params
    initial_state = np.hstack((initial_position, initial_velocity, initial_coeffs))

    # Initialize covariance matrix
    n_state = 6 + 2 * n_n * n_m
    P0 = np.zeros((n_state, n_state))
    P0[:3, :3] = np.eye(3) * 1e-4  # Position variance: 0.0001 m^2
    P0[3:6, 3:6] = np.eye(3) * 1e-8  # Velocity variance: 1e-8 m^2/s^2
    P0[6:, 6:] = np.eye(2 * n_n * n_m) * 1e-6  # Coefficient variance: 1e-6

    t_span = np.linspace(0, 55000, 1000)

    t, state, stm = propagate_state_and_stm(initial_state, fitted_params, n_n, n_m, t_span)

    # Precompute measurement Jacobians with dimension check
    def precompute_jacobians(state, n_state, t):
        N = len(t)
        H_all = np.zeros((3, n_state, N))
        for i in range(N):
            H, R = compute_measurement_partials(state[:3, i], n_state)
            if H.shape != (3, n_state):
                raise ValueError(f"compute_measurement_partials must return (3, {n_state}) Jacobian, got {H.shape} at i={i}")
            H_all[:, :, i] = H
        return H_all, R

    # Optimized propagate-and-update routine
    def propagate_update(stm, P0, H_all, R):
        """
        stm: (n_state, n_state, N) - state transition matrices
        P0:  (n_state, n_state) - initial covariance
        H_all: (3, n_state, N) - measurement Jacobians
        R:   (3, 3) - measurement noise covariance
        returns P: (n_state, n_state, N) - covariances
        """
        # Check input dimensions
        n, m, N = stm.shape
        if n != m:
            raise ValueError(f"stm must be square in first two dimensions, got {stm.shape}")
        if P0.shape != (n, n):
            raise ValueError(f"P0 must be {n}x{n}, got {P0.shape}")
        if H_all.shape != (3, n, N):
            raise ValueError(f"H_all must be (3, {n}, {N}), got {H_all.shape}")
        if R.shape != (3, 3):
            raise ValueError(f"R must be (3, 3), got {R.shape}")

        # Precompute phi @ P0 for all time steps
        phi_P0 = np.zeros((n, n, N))
        for i in range(N):
            phi_P0[:, :, i] = stm[:, :, i] @ P0

        P = np.zeros((n, n, N))
        
        # Pre-allocate temporary arrays
        P_pred = np.zeros((n, n))    # Predicted covariance
        H_trans = np.zeros((n, 3))   # Transposed measurement Jacobian
        S = np.zeros((3, 3))         # Innovation covariance
        K = np.zeros((n, 3))         # Kalman gain
        temp = np.zeros((n, n))      # Temporary matrix for intermediates
        temp2 = np.zeros((n, 3))     # Temporary for P_pred @ H_trans
        
        # Progress interval (every 10%)
        step = max(1, N // 10)
        
        for i in range(N):
            # Print progress
            if i % step == 0:
                print(f"Progress: {100 * i / N:.1f}%")
            
            # Prediction: P_pred = phi @ P0 @ phi.T
            P_pred = phi_P0[:, :, i] @ stm[:, :, i].T
            
            # Measurement update
            H = H_all[:, :, i]    # 3 × n_state
            H_trans = H.T         # n_state × 3
            
            # S = H @ P_pred @ H_trans + R
            S = H @ P_pred @ H_trans + R
            if S.shape != (3, 3):
                raise ValueError(f"S must be (3, 3) at i={i}, got {S.shape}")
            
            # K = (P_pred @ H_trans) @ inv(S)
            temp2 = P_pred @ H_trans
            if temp2.shape != (n, 3):
                raise ValueError(f"P_pred @ H_trans must be ({n}, 3) at i={i}, got {temp2.shape}")
            K = np.linalg.solve(S, temp2).T
            
            # Posterior: P = (I - K @ H) @ P_pred
            temp = np.eye(n) - K @ H
            P[:, :, i] = temp @ P_pred

        print("Progress: 100.0%")
        return P

    # Run the routine with error handling
    print("Starting propagation and update...")
    try:
        H_all, R = precompute_jacobians(state, n_state, t)
        print(f"Input shapes: stm={stm.shape}, P0={P0.shape}, H_all={H_all.shape}, R={R.shape}")
        P = propagate_update(stm, P0, H_all, R)
        print("Completed.")
    except ValueError as e:
        print(f"Error: {e}")

    print("Time points shape:", t.shape)
    print("State shape:", state.shape)
    print("STM shape:", stm.shape)
    print("Covariance shape:", P.shape)
    print("Final state (first 6 components):", state[:6, -1])
    print("Final STM diagonal (first 6):", np.diag(stm[:6, :6, -1]))

    # Plotting
    fig = plt.figure(figsize=(15, 12))

    # 3D Trajectory Plot
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(state[0, :], state[1, :], state[2, :], label='Trajectory', color='b')
    ax1.scatter([0], [0], [CYLINDER_CENTER[2]], color='r', s=100, label='Cylinder Center')
    theta = np.linspace(0, 2*np.pi, 50)
    z_cyl = np.linspace(CYLINDER_CENTER[2] - 0.1, CYLINDER_CENTER[2] + 0.1, 10)
    theta, z_cyl = np.meshgrid(theta, z_cyl)
    x_cyl = CYLINDER_RADIUS * np.cos(theta)
    y_cyl = CYLINDER_RADIUS * np.sin(theta)
    ax1.plot_wireframe(x_cyl, y_cyl, z_cyl, color='r', alpha=0.3, label='Cylinder')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)

    # Position vs. Time
    ax2 = fig.add_subplot(232)
    ax2.plot(t, state[0, :], label='X', color='r')
    ax2.plot(t, state[1, :], label='Y', color='g')
    ax2.plot(t, state[2, :], label='Z', color='b')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs. Time')
    ax2.legend()
    ax2.grid(True)

    # Velocity vs. Time
    ax3 = fig.add_subplot(233)
    ax3.plot(t, state[3, :], label='Vx', color='r')
    ax3.plot(t, state[4, :], label='Vy', color='g')
    ax3.plot(t, state[5, :], label='Vz', color='b')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity vs. Time')
    ax3.legend()
    ax3.grid(True)

    # STM Diagonal Elements (first 6)
    ax4 = fig.add_subplot(234)
    labels = ['∂x/∂x0', '∂y/∂y0', '∂z/∂z0', '∂vx/∂vx0', '∂vy/∂vy0', '∂vz/∂vz0']
    for i in range(6):
        ax4.plot(t, stm[i, i, :], label=labels[i])
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('STM Diagonal Elements')
    ax4.set_title('STM Diagonal Elements vs. Time')
    ax4.legend()
    ax4.grid(True)

    # Position and Velocity Uncertainty
    ax5 = fig.add_subplot(235)
    # position std: shape (3, N)
    pos_std = np.vstack([
        np.sqrt(np.maximum(P[0, 0, :], 0)),
        np.sqrt(np.maximum(P[1, 1, :], 0)),
        np.sqrt(np.maximum(P[2, 2, :], 0)),
    ])

    # velocity std: shape (3, N)
    vel_std = np.vstack([
        np.sqrt(np.maximum(P[3, 3, :], 0)),
        np.sqrt(np.maximum(P[4, 4, :], 0)),
        np.sqrt(np.maximum(P[5, 5, :], 0)),
    ])
    ax5.plot(t, pos_std[0, :], label='σ_x', color='r')
    ax5.plot(t, pos_std[1, :], label='σ_y', color='g')
    ax5.plot(t, pos_std[2, :], label='σ_z', color='b')
    ax5.plot(t, vel_std[0, :], label='σ_vx', linestyle='--', color='r')
    ax5.plot(t, vel_std[1, :], label='σ_vy', linestyle='--', color='g')
    ax5.plot(t, vel_std[2, :], label='σ_vz', linestyle='--', color='b')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Standard Deviation (m, m/s)')
    ax5.set_title('Position and Velocity Uncertainty')
    ax5.legend()
    ax5.grid(True)

    # Coefficient Uncertainty (subset) and Variance Sum
    ax6 = fig.add_subplot(236)
    coeff_indices = [6, 7, 8, 9, 10]
    for idx in coeff_indices:
        coeff_std = np.sqrt(np.maximum(P[idx, idx, :], 0))
        ax6.plot(t, coeff_std, label=f'σ_coeff_{idx-6}')
    variance_sum = np.sum(np.diagonal(P[6:, 6:, :], axis1=1, axis2=2), axis=0)
    ax6.plot(t, variance_sum, label='Sum of Coefficient Variances', color='k', linestyle='--')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Std. Dev. or Variance Sum')
    ax6.set_title('Coefficient Uncertainty')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.show()