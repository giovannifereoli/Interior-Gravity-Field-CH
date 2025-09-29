import numpy as np
from scipy.special import jv as BesselJ, jvp as BesselJp, jn_zeros
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, qr
import matplotlib as mpl
from datetime import datetime
import corner
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
import mesh_utility
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy.stats import chi2
from datetime import datetime

# Set LaTeX formatting
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
mpl.rcParams["font.size"] = 16  # Increase font size for better readability in papers
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["text.latex.preamble"] = r"\usepackage{mathrsfs}"


# Define constants and cylinder parameters
CYLINDER_CENTER = np.array([0.0, 0.0, 0.28])
CYLINDER_RADIUS = 0.1
CYLINDER_ROTATION = np.eye(3)
ALPHA = 100

# TODO: When propagating with rotation different from I, check all the transformations


def compute_A_and_sensitivities(
    rho, phi, z, m_vals, j_mn, A_coeff, B_coeff, alpha, Rstar
):
    """
    Compute cylindrical Jacobian A_cyl and per-coefficient sensitivities for acceleration components.
    """
    # Compute misc terms
    R_alpha = alpha * Rstar
    arg_Bessel = (j_mn * rho) / R_alpha
    exp_term = np.exp(-j_mn * z / R_alpha)
    Jm = BesselJ(m_vals, arg_Bessel)
    Jm_p = BesselJp(m_vals, arg_Bessel, 1)
    Jm_pp = BesselJp(m_vals, arg_Bessel, 2)
    cos_mphi = np.cos(m_vals * phi)
    sin_mphi = np.sin(m_vals * phi)

    # Compute misc terms (2)
    dPhi_dphi = np.sum(
        exp_term * Jm * m_vals * (-A_coeff * sin_mphi + B_coeff * cos_mphi)
    )

    # Compute second derivatives wrt state
    # Jr first row
    d2_drho2 = np.sum(
        exp_term
        * (j_mn / R_alpha) ** 2
        * Jm_pp
        * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )
    d2_dphi_drho = np.sum(
        exp_term
        * (j_mn / R_alpha)
        * Jm_p
        * m_vals
        * (-A_coeff * sin_mphi + B_coeff * cos_mphi)
    )
    d2_dz_drho = -np.sum(
        (j_mn / R_alpha) ** 2
        * exp_term
        * Jm_p
        * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )

    # Jr second row
    d2_dphi2 = -np.sum(
        exp_term * Jm * (m_vals**2) * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )

    # Jr third row
    d2_dz_dphi = np.sum(
        (j_mn / R_alpha)
        * exp_term
        * Jm
        * m_vals
        * (A_coeff * sin_mphi - B_coeff * cos_mphi)
    )

    d2_dz2 = np.sum(
        (j_mn / R_alpha) ** 2
        * exp_term
        * Jm
        * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )

    A_cyl = np.array(
        [
            [d2_drho2, d2_dphi_drho, d2_dz_drho],
            [
                d2_dphi_drho / (rho) - dPhi_dphi / (rho**2),
                d2_dphi2 / (rho),
                d2_dz_dphi / (rho),
            ],
            [d2_dz_drho, d2_dz_dphi, d2_dz2],
        ]
    )

    sens = {
        "a_rho_A": exp_term * (j_mn / R_alpha) * Jm_p * cos_mphi,
        "a_rho_B": exp_term * (j_mn / R_alpha) * Jm_p * sin_mphi,
        "a_phi_A": -(1 / (rho)) * exp_term * Jm * m_vals * sin_mphi,
        "a_phi_B": (1 / (rho)) * exp_term * Jm * m_vals * cos_mphi,
        "a_z_A": -(j_mn / R_alpha) * exp_term * Jm * cos_mphi,
        "a_z_B": -(j_mn / R_alpha) * exp_term * Jm * sin_mphi,
    }

    return A_cyl, sens


def cylindrical_to_cartesian_jacobian(A_cyl, a_cyl, phi):
    """
    Transform 3x3 cylindrical Jacobian and acceleration into Cartesian Jacobian.
    """
    T = np.array(
        [[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )
    T_T = T.T
    a_phi = a_cyl[1]
    T_tensor = a_phi * np.array(
        [[-np.sin(phi), np.cos(phi), 0], [-np.cos(phi), -np.sin(phi), 0], [0, 0, 0]]
    )
    return CYLINDER_ROTATION @ (T_tensor + T_T @ A_cyl) @ T @ CYLINDER_ROTATION.T


def rotate_sensitivity_cylindrical(J_theta_cyl, phi):
    """
    Rotate a (3xK) cylindrical sensitivity matrix into Cartesian coordinates.
    """
    T = np.array(
        [[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )
    return CYLINDER_ROTATION @ T.T @ J_theta_cyl


def compute_acceleration(position, fitted_params, n_n, n_m, j_mn_cache):
    """
    Compute acceleration in Cartesian coordinates at a given position.
    """
    # Transform to cylindrical coordinates
    pt = np.array(position)
    transformed_point = (pt - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.linalg.norm(transformed_point[:2])
    phi = np.arctan2(transformed_point[1], transformed_point[0])
    z = transformed_point[2]

    # Expand m and n values
    m_vals = np.repeat(np.arange(n_m), n_n)  # Each m value repeated n_n times
    j_mn = j_mn_cache  # np.array([jn_zeros(m, n + 1)[-1] for m in range(n_m) for n in range(n_n)])
    A_coeff = fitted_params[
        0::2
    ]  # NOTE: taking every second element starting from index 0
    B_coeff = fitted_params[
        1::2
    ]  # NOTE: taking every second element starting from index 1

    # Compute misc terms
    a_cyl = np.zeros(3)
    R_alpha = ALPHA * CYLINDER_RADIUS
    arg_Bessel = (j_mn * rho) / R_alpha
    exp_term = np.exp(-j_mn * z / R_alpha)
    Jm = BesselJ(m_vals, arg_Bessel)
    Jm_p = BesselJp(m_vals, arg_Bessel, 1)
    cos_mphi = np.cos(m_vals * phi)
    sin_mphi = np.sin(m_vals * phi)

    # Cylindrical acceleration components
    a_cyl[0] = np.sum(
        exp_term * (j_mn / R_alpha) * Jm_p * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )
    a_cyl[1] = np.sum(
        (1 / (rho))
        * exp_term
        * Jm
        * m_vals
        * (-A_coeff * sin_mphi + B_coeff * cos_mphi)
    )
    a_cyl[2] = np.sum(
        -(j_mn / R_alpha) * exp_term * Jm * (A_coeff * cos_mphi + B_coeff * sin_mphi)
    )

    # Transform back to Cartesian coordinates
    # NOTE: @ R because is a row vector!
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
    # Transform to cylindrical coordinates
    pt = np.array(position)
    transformed_point = (pt - CYLINDER_CENTER) @ CYLINDER_ROTATION.T
    rho = np.linalg.norm(transformed_point[:2])
    phi = np.arctan2(transformed_point[1], transformed_point[0])
    z = transformed_point[2]

    # Expand m and n values
    m_vals = np.repeat(np.arange(n_m), n_n)
    j_mn = j_mn_cache
    A_coeff = fitted_params[0::2]
    B_coeff = fitted_params[1::2]

    # Compute misc terms
    a_cyl = np.zeros(3)
    fac = ALPHA * CYLINDER_RADIUS
    x = (j_mn * rho) / fac
    exp_term = np.exp(-j_mn * z / fac)
    Jm = BesselJ(m_vals, x)
    Jm_p = BesselJp(m_vals, x, 1)
    cos_mphi = np.cos(m_vals * phi)
    sin_mphi = np.sin(m_vals * phi)

    # Cylindrical acceleration components
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

    # Compute cylindrical Jacobian and rotate
    A_cyl, sens_cyl = compute_A_and_sensitivities(
        rho, phi, z, m_vals, j_mn, A_coeff, B_coeff, ALPHA, CYLINDER_RADIUS
    )
    J_cart = cylindrical_to_cartesian_jacobian(A_cyl, a_cyl, phi)

    # Compute coefficients sensitivity matrix and rotate
    J_theta_cyl = np.vstack(
        [
            np.hstack([sens_cyl["a_rho_A"], sens_cyl["a_rho_B"]]),
            np.hstack([sens_cyl["a_phi_A"], sens_cyl["a_phi_B"]]),
            np.hstack([sens_cyl["a_z_A"], sens_cyl["a_z_B"]]),
        ]
    )
    J_theta_cart = rotate_sensitivity_cylindrical(J_theta_cyl, phi)

    # Assemble full dynamical matrix
    K = 2 * n_n * n_m
    A = np.zeros((6 + K, 6 + K))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = J_cart
    A[3:6, 6:] = J_theta_cart

    return A


def propagate_state_and_stm(initial_state, fitted_params, n_n, n_m, t_span):
    """
    Propagate the state and STM using a single call to solve_ivp.
    """
    # Initial state and STM
    n_state = 6 + 2 * n_n * n_m
    stm0 = np.eye(n_state).ravel()
    y0 = np.hstack((initial_state, stm0))

    # Integrate the combined state and STM
    sol = solve_ivp(
        fun=lambda t, y: _dynamics_full(
            t, y, n_state, fitted_params, n_n, n_m, j_mn_cache
        ),
        t_span=(t_span[0], t_span[-1]),
        y0=y0,
        t_eval=t_span,
        method="RK45",
        rtol=1e-10,
        atol=1e-10,
    )

    # Extract results
    Y = sol.y
    states = Y[:n_state, :]
    stms = Y[n_state:, :].reshape((n_state, n_state, len(t_span)))

    return sol.t, states, stms


def plot_cov_ellipses(fig, mean, cov, color, nsig=1.0):
    # Plot covariance ellipses on corner plot
    ndim = len(mean)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # Loop over each subplot
    for i in range(ndim):
        for j in range(i):
            ax = axes[i, j]
            sub_mean = [mean[j], mean[i]]
            sub_cov = cov[np.ix_([j, i], [j, i])]

            vals, vecs = np.linalg.eigh(sub_cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            width, height = 4 * nsig * np.sqrt(vals)

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
    # Unpack state and STM
    state = y[:n_state]
    stm_mat = y[n_state:].reshape((n_state, n_state))
    pos = state[0:3]
    vel = state[3:6]

    # Compute acceleration and dynamical matrix
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

    n_n, n_m = 25, 25
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
    P0[:3, :3] = np.eye(3) * (1e-3) ** 2  # Position variance: 1e-3 km
    P0[3:6, 3:6] = np.eye(3) * (1e-6) ** 2  # Velocity variance: 1e-6 km/s
    P0[6:, 6:] = np.diag(np.diag(full_cov_params))

    stop_at_percent = 0.5  # NOTE: to remain in linear regime!
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

    print("Propagation covariance completed.")

    # Parameters
    N_samples = 1000  # Monte Carlo sample count
    MC_w_polyhedral = True
    rng = np.random.default_rng(42)
    n_state = initial_state.shape[0]

    # Sample initial conditions from P0
    initial_samples = rng.multivariate_normal(initial_state, P0, N_samples)

    # Allocate array for final states
    final_states = np.zeros((N_samples, n_state))

    if MC_w_polyhedral == True:
        # Polyhedral Model Initialization
        # Meshes from https://github.com/darioizzo/geodesyNets/tree/master/3dmeshes
        vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
        vertices, faces = np.array(vertices), np.array(faces)

        # Define asteroid density
        DENSITY = 1.0

        # Initialize the polyhedron object
        eros = Polyhedron(
            polyhedral_source=(vertices, faces),
            density=DENSITY,
            integrity_check=PolyhedronIntegrity.DISABLE,
        )

        # Create an evaluable object for gravity calculations
        evaluable_eros = GravityEvaluable(eros)

        # Polyhedral Acceleration Function
        def acceleration_poly(position):
            _, acceleration, _ = evaluable_eros(
                computation_points=position, parallel=False
            )
            return acceleration

        # Trajectory Propagation Function
        def propagate_trajectory(
            initial_position, initial_velocity, acceleration_func, t_span, method="RK45"
        ):
            def dynamics(t, state):
                position = state[:3]
                velocity = state[3:]
                acceleration = acceleration_func(position)
                if np.any(np.isnan(acceleration)):
                    return np.full(6, np.nan)  # Return NaN for invalid acceleration
                return np.hstack((velocity, acceleration))

            initial_state = np.hstack((initial_position, initial_velocity))

            sol = solve_ivp(
                dynamics,
                t_span=(t_span[0], t_span[-1]),
                y0=initial_state,
                method=method,
                t_eval=t_span,
                rtol=1e-10,
                atol=1e-10,
            )
            return sol.t, sol.y

        # Monte Carlo Propagation
        def propagate_single_sample(i, sample_state, t_span):
            # Extract position and velocity (first 6 components)
            initial_state = sample_state[:6]
            initial_position = initial_state[:3]
            initial_velocity = initial_state[3:6]

            # Propagate trajectory
            _, state_mc = propagate_trajectory(
                initial_position=initial_position,
                initial_velocity=initial_velocity,
                acceleration_func=acceleration_poly,
                t_span=t_span,
                method="RK45",
            )

            return state_mc[:, -1]  # Return final state (6D)

        # Propagate each sample in parallel
        print("Running Monte Carlo propagation with polyhedral gravity model...")
        n_state_mc = 6  # State size for polyhedral model (position and velocity only)
        final_states = np.zeros((N_samples, n_state_mc))
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(propagate_single_sample)(i, initial_samples[i], t_span)
            for i in tqdm(range(N_samples), desc="Monte Carlo (Polyhedral)")
        )
        for i, result in enumerate(results):
            final_states[i] = result

    else:
        # Function to propagate a single Monte Carlo sample
        def propagate_single_sample(i, sample_state, fitted_params, n_n, n_m, t_span):
            t_mc, state_mc, _ = propagate_state_and_stm(
                sample_state, fitted_params, n_n, n_m, t_span
            )
            return state_mc[:, -1]

        # Propagate each sample in parallel
        print("Running Monte Carlo propagation with cylindrical harmonics...")
        final_states = np.zeros((N_samples, n_state))
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(propagate_single_sample)(
                i, initial_samples[i], fitted_params, n_n, n_m, t_span
            )
            for i in tqdm(range(N_samples), desc="Monte Carlo (Cylindrical Harmonics)")
        )
        for i, result in enumerate(results):
            final_states[i] = result
    print("Monte Carlo propagation completed.")

    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from tqdm import tqdm

    # Function to plot cylinder
    def plot_cylinder(
        ax, center, radius, height=1.0, rotation=np.eye(3), color="gray", alpha=0.3
    ):
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(-height / 2, height / 2, 2)
        U, V = np.meshgrid(u, v)
        X = radius * np.cos(U)
        Y = radius * np.sin(U)
        Z = V

        # Apply rotation
        xyz = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
        xyz_rot = rotation @ xyz
        X_rot = xyz_rot[0].reshape(X.shape) + center[0]
        Y_rot = xyz_rot[1].reshape(Y.shape) + center[1]
        Z_rot = xyz_rot[2].reshape(Z.shape) + center[2]

        ax.plot_surface(X_rot, Y_rot, Z_rot, color=color, alpha=alpha)

    # Initialize figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")

    # Plot cylinder
    plot_cylinder(
        ax,
        CYLINDER_CENTER,
        CYLINDER_RADIUS,
        height=0.56,
        rotation=CYLINDER_ROTATION,
        alpha=0.2,
    )

    # Real-time Monte Carlo propagation
    N_plot = min(100, N_samples)  # Limit number of trajectories for plotting speed
    for i in tqdm(range(N_plot), desc="Monte Carlo Real-Time"):
        sample_state = initial_samples[i]
        t_mc, state_mc = propagate_trajectory(
            initial_position=sample_state[:3],
            initial_velocity=sample_state[3:6],
            acceleration_func=acceleration_poly,
            t_span=t_span,
        )
        ax.plot(state_mc[0], state_mc[1], state_mc[2], lw=1, alpha=0.6, color="blue")
        plt.pause(0.01)

    plt.show()
    """

    # Compute empirical mean and covariance
    final_mean_mc = np.mean(final_states, axis=0)
    final_cov_mc = np.cov(final_states.T)

    # Extract SRIF final state and covariance
    final_mean_srif = state[:, -1]
    final_cov_srif = P[:, :, -1]

    # Plotting results
    compare_idx = np.arange(6)
    labels = [
        r"$x (km)$",
        r"$y (km)$",
        r"$z (km)$",
        r"$v_x (km/s)$",
        r"$v_y (km/s)$",
        r"$v_z (km/s)$",
    ]
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

    # Plot the Monte Carlo samples
    fig = corner.corner(
        final_mc_samples,
        labels=labels,
        show_titles=False,
        color=COLOR_PALETTE[0],
        title_fmt=".4e",
        label_kwargs={"fontsize": 12},
        fig=plt.figure(figsize=(14, 14)),
        bins=30,
        hist_kwargs={"linewidth": 1.5},
        data_kwargs={"alpha": 0.6},
        plot_contours=True,
        fill_contours=True,
        max_n_ticks=5,
        space=0.15,
    )

    # Add MC mean and ellipse (black)
    mc_mean_plot = final_mean_mc[compare_idx]
    mc_cov_plot = final_cov_mc[np.ix_(compare_idx, compare_idx)]
    corner.overplot_lines(fig, mc_mean_plot, color="k", lw=1.5)
    plot_cov_ellipses(fig, mc_mean_plot, mc_cov_plot, "k")

    # Add SRIF mean and ellipse (colored)
    corner.overplot_lines(fig, srif_mean_plot, color=COLOR_PALETTE[2], lw=1.5)
    plot_cov_ellipses(fig, srif_mean_plot, srif_cov_plot, COLOR_PALETTE[2])

    # Add title and legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=COLOR_PALETTE[0],
            lw=2,
            label="Polyhedral (Const. Density) MC Samples",
        ),
        Line2D([0], [0], color="k", lw=2, label="Polyhedral MC Mean"),
        Line2D(
            [0], [0], color="k", lw=2, linestyle="--", label=r"Polyhedral MC $2\sigma$"
        ),
        Line2D(
            [0], [0], color=COLOR_PALETTE[2], lw=2, label="Cylindrical Harmonics Mean"
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_PALETTE[2],
            lw=2,
            linestyle="--",
            label=r"Cylindrical Harmonics $2\sigma$",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        fontsize=12,
        frameon=True,
    )
    # for ax in fig.get_axes():
    #    for tick in ax.get_xticklabels():
    #        tick.set_rotation(30)
    #    for tick in ax.get_yticklabels():
    #        tick.set_rotation(0)

    # from matplotlib.ticker import ScalarFormatter

    # for ax in fig.get_axes():
    #    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(
        left=0.12, bottom=0.12, right=0.95, top=0.93, hspace=0.15, wspace=0.15
    )
    fig.savefig(
        f"Images/fig_corner_MC_vs_LinCov_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    plt.show()

    # Extract 6D state (position + velocity) at final time
    mc_final_6d = final_states[:, :6]  # shape: (N_samples, 6)
    srif_mean_6d = state[:6, -1]  # shape: (6,)
    srif_cov_6d = P[:6, :6, -1]  # shape: (6,6)

    # Compute NEES
    inv_cov = np.linalg.inv(srif_cov_6d)
    diffs = mc_final_6d - srif_mean_6d[None, :]
    nees_values = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)

    # Print NEES statistics
    print("\n===== NEES Statistics =====")
    print(f"Mean NEES: {np.mean(nees_values):.3f}")
    print(f"Median NEES: {np.median(nees_values):.3f}")
    print(f"Min NEES: {np.min(nees_values):.3f}")
    print(f"Max NEES: {np.max(nees_values):.3f}")
    print(f"Standard Deviation: {np.std(nees_values):.3f}")

    # Chi-square 95% confidence bounds for 6 DOF
    alpha = 0.05
    dof = 6
    chi2_lower = chi2.ppf(alpha / 2, dof)
    chi2_upper = chi2.ppf(1 - alpha / 2, dof)
    print(
        f"95% confidence bounds (chi2, DOF={dof}): {chi2_lower:.3f} - {chi2_upper:.3f}"
    )

    # NEES test
    mean_nees = np.mean(nees_values)
    if chi2_lower <= mean_nees <= chi2_upper:
        print(
            f"NEES test PASSED: mean NEES = {mean_nees:.3f} within 95% chi2 bounds ({chi2_lower:.3f}, {chi2_upper:.3f})"
        )
    else:
        print(
            f"NEES test FAILED: mean NEES = {mean_nees:.3f} outside 95% chi2 bounds ({chi2_lower:.3f}, {chi2_upper:.3f})"
        )

    # Plot NEES histogram in the improved style
    plt.figure(figsize=(12, 8))

    # Histogram
    n, bins, patches = plt.hist(
        nees_values,
        bins=50,
        color=COLOR_PALETTE[2],
        alpha=0.7,
        edgecolor="black",
        density=True,
        label="NEES samples",
    )

    # 95% chi-square bounds
    plt.axvline(
        chi2_lower,
        color="k",
        linestyle="-.",
        linewidth=2,
        label=rf"Lower {alpha/2*100:.1f}\% $\chi^2$ = {chi2_lower:.3f}",
    )
    plt.axvline(
        chi2_upper,
        color="k",
        linestyle="--",
        linewidth=2,
        label=rf"Upper {(1-alpha/2)*100:.1f}\% $\chi^2$ = {chi2_upper:.3f}",
    )

    # Plot mean NEES
    mean_nees = np.mean(nees_values)
    plt.axvline(
        mean_nees,
        color=COLOR_PALETTE[0],
        linestyle="-",
        linewidth=2,
        label=f"Mean NEES={mean_nees:.2f}",
    )

    # Plot median NEES
    median_nees = np.median(nees_values)
    plt.axvline(
        median_nees,
        color=COLOR_PALETTE[-1],
        linestyle="-",
        linewidth=2,
        label=f"Median NEES={median_nees:.2f}",
    )

    # Labels and title
    plt.xlabel("NEES (-)", labelpad=10)
    plt.ylabel("Probability Density (-)", labelpad=10)
    plt.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        edgecolor="black",
        fontsize=14,
    )
    plt.grid(True, linestyle="--", which="both", linewidth=0.7, alpha=0.8)
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    # Save figure
    plt.savefig(
        f"Images/NEES_histogram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    plt.show()
