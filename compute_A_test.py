# Insert the following block immediately after your existing imports in the top of your script:

import numpy as np
from scipy.special import jv as BesselJ, jvp as BesselJp


# -----------------------------------------------------------------------------
# Analytical Jacobian in Cylindrical + Cyl→Cart Transform & Sensitivities
# -----------------------------------------------------------------------------
def compute_A_and_sensitivities(
    rho, phi, z, m_vals, j_mn, A_coeff, B_coeff, alpha, Rstar
):
    """
    Compute cylindrical Jacobian A_cyl and per-coefficient sensitivities
    for acceleration components.
    """
    fac = alpha * Rstar
    x = (j_mn * rho) / fac
    exp_term = np.exp(-j_mn * z / fac)
    Jm = BesselJ(m_vals, x)
    Jm_p = BesselJp(m_vals, x, 1)
    Jm_pp = BesselJp(m_vals, x, 2)

    cos_mphi = np.cos(m_vals * phi)
    sin_mphi = np.sin(m_vals * phi)

    # Second derivatives for A_cyl
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

    # Assemble cylindrical Jacobian
    A_cyl = np.array(
        [
            [d2_drho2, d2_dphi_drho, d2_dz_drho],
            [
                -dPhi_dphi / rho**2 + d2_dphi_drho / rho,
                d2_dphi2 / rho,
                d2_dz_dphi / rho,
            ],
            [d2_dz_drho, d2_dz_dphi, d2_dz2],
        ]
    )

    # Sensitivities per coefficient (arrays length K)
    sens = {
        "a_rho_A": exp_term * (j_mn / fac) * Jm_p * cos_mphi,
        "a_rho_B": exp_term * (j_mn / fac) * Jm_p * sin_mphi,
        "a_phi_A": -(1 / rho) * exp_term * Jm * m_vals * sin_mphi,
        "a_phi_B": (1 / rho) * exp_term * BesselJ(m_vals, x) * m_vals * cos_mphi,
        "a_z_A": -(j_mn / fac) * exp_term * Jm * cos_mphi,
        "a_z_B": -(j_mn / fac) * exp_term * Jm * sin_mphi,
    }

    return A_cyl, sens


def cylindrical_to_cartesian_jacobian(A_cyl, a_cyl, phi):
    """
    Transform 3x3 A_cyl and acceleration a_cyl into Cartesian J_cart.
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
    Rotate a (3xK) cylindrical sensitivity into Cartesian.
    """
    T = np.array(
        [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )
    return T.T @ J_theta_cyl


# -----------------------------------------------------------------------------
# Example of plugging in right after your LS-fit, before trajectory propagation:
# -----------------------------------------------------------------------------
"""'
if __name__ == "__main__":
    # ... (your LS fitting code has produced fitted_params, and you have
    #      a sample point in cylindrical coords: rho_i, phi_i, z_i)
    # e.g. choose the first data point:
    pt = structured_results['points'][0]
    acc_cart = structured_results['acceleration'][0]    # in Cartesian
    # Convert to cylindrical:
    cyl_acc = cartesian_to_cylindrical_acceleration(pt[None,:], acc_cart[None,:])[0]
    rho_i = np.linalg.norm((pt - CYLINDER_CENTER)[:2])
    phi_i = np.arctan2((pt - CYLINDER_CENTER)[1], (pt - CYLINDER_CENTER)[0])
    z_i   = (rotation_inv @ (pt - CYLINDER_CENTER))[2]  # use your cylinder rotation

    # Prepare (m, n) arrays matching your LS fit:
    m_vals = np.repeat(np.arange(n_m), n_n)
    j_mn   = np.array([ jn_zeros(m, n+1)[-1] for m in range(n_m) for n in range(n_n) ])
    A_coef = fitted_params[0::2]
    B_coef = fitted_params[1::2]

    # Compute cylindrical Jacobian + sensitivities:
    A_cyl, sens_cyl = compute_A_and_sensitivities(rho_i, phi_i, z_i,
                                                  m_vals, j_mn,
                                                  A_coef, B_coef,
                                                  ALPHA, CYLINDER_RADIUS)
    # Rotate to Cartesian:
    J_cart = cylindrical_to_cartesian_jacobian(A_cyl, cyl_acc, phi_i)

    # For coefficient sensitivities, build J_theta_cyl as 3xK matrix:
    J_theta_cyl = np.vstack([
        sens_cyl['a_rho_A'], sens_cyl['a_phi_A'], sens_cyl['a_z_A']
    ])
    # Rotate them:
    J_theta_cart = rotate_sensitivity_cylindrical(J_theta_cyl, phi_i)

    print("Cartesian Jacobian at sample point:\n", J_cart)
    print("Cartesian sensitivity shape:", J_theta_cart.shape)
"""
