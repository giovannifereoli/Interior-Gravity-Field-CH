import numpy as np
from scipy.special import jv as BesselJ, jvp as BesselJp, jn_zeros
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import cholesky, qr
import matplotlib as mpl
from datetime import datetime
import corner
import matplotlib.patches as patches


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


def plot_cov_ellipses(fig, mean, cov, color, nsig=1.0):
    ndim = len(mean)
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        for j in range(i):
            ax = axes[i, j]
            sub_mean = [mean[j], mean[i]]
            sub_cov = cov[np.ix_([j, i], [j, i])]

            vals, vecs = np.linalg.eigh(sub_cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            width, height = 2 * nsig * np.sqrt(vals)

            ell = patches.Ellipse(
                xy=sub_mean,
                width=width,
                height=height,
                angle=angle,
                edgecolor=color,
                fc="none",
                lw=2.0,
                ls="--",
                zorder=10,  # Draw on top
            )
            ax.add_patch(ell)


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
    P0[:3, :3] = np.eye(3) * (1e-6) ** 2  # Position variance: 1e-6 km
    P0[3:6, 3:6] = np.eye(3) * (1e-9) ** 2  # Velocity variance: 1e-9 km/s
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
    R_sqrt = cholesky(np.linalg.inv(P0[:6, :6]), lower=False)

    STM_tm = np.eye(6)
    P = np.empty((6, 6, n_steps))
    P[:, :, 0] = P0[:6, :6]

    # Precompute identity matrix
    I_n = np.eye(6)

    for i in tqdm(range(1, n_steps), desc="SRIF", ncols=80):
        # Compute the STM for the current step
        Phi = stm[:6, :6, i] @ np.linalg.inv(STM_tm)
        STM_tm = stm[:6, :6, i]

        # Prediction step
        pred_matrix = R_sqrt @ np.linalg.inv(Phi)
        _, RQ = qr(pred_matrix, mode="economic")
        R_sqrt = RQ[:n_state, :n_state]

        # Covariance reconstruction
        RtR = R_sqrt.T @ R_sqrt
        P[:, :, i] = np.linalg.inv(RtR)

    print("Propagation and update completed.")

    # Parameters
    N_samples = 1000  # Monte Carlo sample count
    rng = np.random.default_rng(42)
    n_state = initial_state.shape[0]

    # Sample initial conditions from P0
    initial_samples = rng.multivariate_normal(initial_state, P0, N_samples)

    # Allocate array for final states
    final_states = np.zeros((N_samples, n_state))

    # Propagate each sample independently
    print("Running Monte Carlo propagation...")
    for i in tqdm(range(N_samples), desc="Monte Carlo"):
        sample_state = initial_samples[i]
        t_mc, state_mc, _ = propagate_state_and_stm(
            sample_state, fitted_params, n_n, n_m, t_span
        )
        final_states[i] = state_mc[:, -1]
    print()

    # Compute empirical mean and covariance
    final_mean_mc = np.mean(final_states, axis=0)
    final_cov_mc = np.cov(final_states.T)

    # Extract SRIF final state and covariance
    final_mean_srif = state[:, -1]
    final_cov_srif = P[:, :, -1]

    # Plotting results
    compare_idx = np.arange(6)
    labels = [r"$x$", r"$y$", r"$z$", r"$v_x$", r"$v_y$", r"$v_z$"]
    print("\n===== Final State Statistics Comparison =====")
    print(f"{'State':6}  |  {'Mean MC':>15}  |  {'Mean SRIF':>15}")
    print("-" * 45)
    for i, label in enumerate(labels):
        print(
            f"{label:6}  |  {final_mean_mc[i]:+15.3e}  |  {final_mean_srif[i]:+15.3e}"
        )

    print("\nStandard deviations (1σ):")
    print(f"{'State':6}  |  {'Sigma MC':>15}  |  {'Sigma SRIF':>15}")
    print("-" * 45)
    for i, label in enumerate(labels):
        sigma_mc = np.sqrt(final_cov_mc[i, i])
        sigma_srif = np.sqrt(final_cov_srif[i, i])
        print(f"{label:6}  |  {sigma_mc:15.2e}  |  {sigma_srif:15.2e}")

    final_mc_samples = final_states[:, compare_idx]
    srif_mean_plot = final_mean_srif[compare_idx]
    srif_cov_plot = final_cov_srif[np.ix_(compare_idx, compare_idx)]

    fig = corner.corner(
        final_mc_samples,
        labels=labels,
        show_titles=True,
        color=COLOR_PALETTE[0],
        title_fmt=".2e",
        label_kwargs={"fontsize": 12},
    )

    corner.overplot_lines(fig, srif_mean_plot, color=COLOR_PALETTE[2])
    plot_cov_ellipses(fig, srif_mean_plot, srif_cov_plot, COLOR_PALETTE[2])

    plt.suptitle("Monte Carlo vs SRIF Final State Covariance", fontsize=14)
    plt.show()
