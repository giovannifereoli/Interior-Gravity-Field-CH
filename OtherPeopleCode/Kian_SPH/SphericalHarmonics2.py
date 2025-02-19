import numpy as np


class SphericalHarmonics:
    def __init__(self):
        pass

    @staticmethod
    def compute_spherical_harmonics(
        rvec: np.array, R_ref: float, GM: float, C: np.matrix, S: np.matrix, degacc=3
    ):
        """
        Computes spherical harmonics accelerations, partials, and their derivatives.

        Parameters:
            rvec (np.array): Position vector [x, y, z].
            R_ref (float): Reference radius.
            GM (float): Gravitational constant times mass of the central body.
            C (np.matrix): Coefficients for the cosine terms.
            S (np.matrix): Coefficients for the sine terms.
            degacc (int): Degree of accuracy for spherical harmonics.

        Returns:
            tuple: Accelerations, partial derivatives of accelerations, partials with respect to C and S coefficients.
        """
        n_max = 100
        x, y, z = rvec
        r_sat = np.linalg.norm(rvec)

        # Initialize Bnm exterior terms
        bnm_ext_real = np.zeros((n_max + 3, n_max + 3))
        bnm_ext_imag = np.zeros((n_max + 3, n_max + 3))

        # Compute unnormalized Bnm terms
        SphericalHarmonics._compute_bnm(
            bnm_ext_real, bnm_ext_imag, x, y, z, r_sat, R_ref, degacc
        )

        # Compute accelerations and partials
        accelerations, partials_of_accels = (
            SphericalHarmonics._compute_acc_and_partials(
                bnm_ext_real, bnm_ext_imag, C, S, GM, R_ref, degacc
            )
        )

        # Compute partial derivatives with respect to C and S
        partials_by_C, partials_by_S = SphericalHarmonics._compute_partials_by_CS(
            bnm_ext_real, bnm_ext_imag, C, S, GM, R_ref, degacc
        )

        return accelerations, partials_of_accels, partials_by_C, partials_by_S

    @staticmethod
    def _compute_bnm(bnm_ext_real, bnm_ext_imag, x, y, z, r_sat, R_ref, degacc):
        """
        Computes unnormalized Bnm exterior terms.

        Parameters:
            bnm_ext_real (np.array): Real part of Bnm terms.
            bnm_ext_imag (np.array): Imaginary part of Bnm terms.
            x (float): x-coordinate of the position.
            y (float): y-coordinate of the position.
            z (float): z-coordinate of the position.
            r_sat (float): Distance to the satellite.
            R_ref (float): Reference radius.
            degacc (int): Degree of accuracy for spherical harmonics.
        """
        for mm in range(degacc + 3):
            for nn in range(mm, degacc + 3):
                if mm == nn:  # Diagonal terms
                    if mm == 0:
                        bnm_ext_real[0, 0] = R_ref / r_sat
                    else:
                        factor = (2 * nn - 1) * (R_ref / r_sat)
                        bnm_ext_real[nn, nn] = factor * (
                            (x / r_sat) * bnm_ext_real[nn - 1, nn - 1]
                            - (y / r_sat) * bnm_ext_imag[nn - 1, nn - 1]
                        )
                        bnm_ext_imag[nn, nn] = factor * (
                            (y / r_sat) * bnm_ext_real[nn - 1, nn - 1]
                            + (x / r_sat) * bnm_ext_imag[nn - 1, nn - 1]
                        )
                else:  # Non-diagonal terms
                    factor1 = (2 * nn - 1) / (nn - mm) * (R_ref / r_sat)
                    factor2 = (R_ref / r_sat) ** 2
                    if nn >= 2:
                        bnm_ext_real[nn, mm] = (
                            factor1 * (z / r_sat) * bnm_ext_real[nn - 1, mm]
                            - ((nn + mm - 1) / (nn - mm))
                            * factor2
                            * bnm_ext_real[nn - 2, mm]
                        )
                        bnm_ext_imag[nn, mm] = (
                            factor1 * (z / r_sat) * bnm_ext_imag[nn - 1, mm]
                            - ((nn + mm - 1) / (nn - mm))
                            * factor2
                            * bnm_ext_imag[nn - 2, mm]
                        )
                    else:
                        bnm_ext_real[nn, mm] = (
                            factor1 * (z / r_sat) * bnm_ext_real[nn - 1, mm]
                        )
                        bnm_ext_imag[nn, mm] = (
                            factor1 * (z / r_sat) * bnm_ext_imag[nn - 1, mm]
                        )

    @staticmethod
    def _compute_acc_and_partials(bnm_ext_real, bnm_ext_imag, C, S, GM, R_ref, degacc):
        """
        Computes accelerations and partial derivatives of accelerations.

        Parameters:
            bnm_ext_real (np.array): Real part of Bnm terms.
            bnm_ext_imag (np.array): Imaginary part of Bnm terms.
            C (np.matrix): Coefficients for cosine terms.
            S (np.matrix): Coefficients for sine terms.
            GM (float): Gravitational constant times mass.
            R_ref (float): Reference radius.
            degacc (int): Degree of accuracy for spherical harmonics.

        Returns:
            tuple: Accelerations and partial derivatives of accelerations.
        """
        # Initialize acceleration components
        x_ddot, y_ddot, z_ddot = 0, 0, 0

        # Initialize partial derivatives of accelerations
        dxddot_dx, dxddot_dy, dxddot_dz = 0, 0, 0
        dyddot_dx, dyddot_dy, dyddot_dz = 0, 0, 0
        dzddot_dx, dzddot_dy, dzddot_dz = 0, 0, 0

        for nn in range(degacc + 1):
            for mm in range(nn + 1):
                delta_m = 1 if mm == 0 else 0
                g_ext_1 = 0.5 * (1 + delta_m)
                g_ext_2 = 0.5 * (nn - mm + 2) * (nn - mm + 1)
                g_ext_3 = -(nn - mm + 1)

                # Compute acceleration components
                x_ddot += (
                    GM
                    / (R_ref**2)
                    * (
                        g_ext_1
                        * (
                            -C[nn, mm] * bnm_ext_real[nn + 1, mm + 1]
                            - S[nn, mm] * bnm_ext_imag[nn + 1, mm + 1]
                        )
                        + g_ext_2
                        * (
                            C[nn, mm] * bnm_ext_real[nn + 1, mm - 1]
                            + S[nn, mm] * bnm_ext_imag[nn + 1, mm - 1]
                        )
                    )
                )
                y_ddot += (
                    GM
                    / (R_ref**2)
                    * (
                        g_ext_1
                        * (
                            S[nn, mm] * bnm_ext_real[nn + 1, mm + 1]
                            - C[nn, mm] * bnm_ext_imag[nn + 1, mm + 1]
                        )
                        + g_ext_2
                        * (
                            S[nn, mm] * bnm_ext_real[nn + 1, mm - 1]
                            - C[nn, mm] * bnm_ext_imag[nn + 1, mm - 1]
                        )
                    )
                )
                z_ddot += (
                    GM
                    / (R_ref**2)
                    * g_ext_3
                    * (
                        C[nn, mm] * bnm_ext_real[nn + 1, mm]
                        + S[nn, mm] * bnm_ext_imag[nn + 1, mm]
                    )
                )

                # Compute partial derivatives of x_ddot
                dxddot_dx += (
                    GM
                    / (R_ref**3)
                    * (
                        g_ext_1
                        * (
                            C[nn, mm] * bnm_ext_real[nn + 2, mm + 2]
                            - (nn + 2) * (nn + 1) * C[nn, mm] * bnm_ext_real[nn + 2, mm]
                        )
                    )
                )
                dxddot_dy += (
                    GM
                    / (R_ref**3)
                    * (g_ext_1 * S[nn, mm] * bnm_ext_imag[nn + 2, mm + 2])
                )
                dxddot_dz += (
                    GM
                    / (R_ref**3)
                    * (g_ext_3 * (nn + 1) * C[nn, mm] * bnm_ext_real[nn + 2, mm + 1])
                )

                # Compute partial derivatives of y_ddot
                dyddot_dx += (
                    GM
                    / (R_ref**3)
                    * (g_ext_1 * S[nn, mm] * bnm_ext_real[nn + 2, mm + 2])
                )
                dyddot_dy += (
                    GM
                    / (R_ref**3)
                    * (
                        g_ext_1
                        * (
                            -C[nn, mm] * bnm_ext_imag[nn + 2, mm + 2]
                            - (nn + 2) * (nn + 1) * S[nn, mm] * bnm_ext_imag[nn + 2, mm]
                        )
                    )
                )
                dyddot_dz += (
                    GM
                    / (R_ref**3)
                    * (g_ext_3 * (nn + 1) * S[nn, mm] * bnm_ext_imag[nn + 2, mm + 1])
                )

                # Compute partial derivatives of z_ddot
                dzddot_dx += (
                    GM / (R_ref**3) * (g_ext_3 * C[nn, mm] * bnm_ext_real[nn + 2, mm])
                )
                dzddot_dy += (
                    GM / (R_ref**3) * (g_ext_3 * S[nn, mm] * bnm_ext_imag[nn + 2, mm])
                )
                dzddot_dz += (
                    GM
                    / (R_ref**3)
                    * (
                        g_ext_3
                        * (nn - mm + 2)
                        * (nn - mm + 1)
                        * (
                            C[nn, mm] * bnm_ext_real[nn + 2, mm]
                            + S[nn, mm] * bnm_ext_imag[nn + 2, mm]
                        )
                    )
                )

        # Combine accelerations and partial derivatives into matrices
        accelerations = np.array([x_ddot, y_ddot, z_ddot])
        partials_of_accels = np.array(
            [
                [dxddot_dx, dxddot_dy, dxddot_dz],
                [dyddot_dx, dyddot_dy, dyddot_dz],
                [dzddot_dx, dzddot_dy, dzddot_dz],
            ]
        )

        return accelerations, partials_of_accels

    @staticmethod
    def _compute_partials_by_CS(bnm_ext_real, bnm_ext_imag, C, S, GM, R_ref, degacc):
        """
        Computes partial derivatives with respect to coefficients C and S.

        Parameters:
            bnm_ext_real (np.array): Real part of Bnm terms.
            bnm_ext_imag (np.array): Imaginary part of Bnm terms.
            C (np.matrix): Coefficients for cosine terms.
            S (np.matrix): Coefficients for sine terms.
            GM (float): Gravitational constant times mass.
            R_ref (float): Reference radius.
            degacc (int): Degree of accuracy for spherical harmonics.

        Returns:
            tuple: Partial derivatives with respect to C and S coefficients.
        """
        dxddot_dC = np.zeros((degacc + 1, degacc + 1))
        dyddot_dC = np.zeros((degacc + 1, degacc + 1))
        dzddot_dC = np.zeros((degacc + 1, degacc + 1))

        dxddot_dS = np.zeros((degacc + 1, degacc + 1))
        dyddot_dS = np.zeros((degacc + 1, degacc + 1))
        dzddot_dS = np.zeros((degacc + 1, degacc + 1))

        for nn in range(degacc + 1):
            for mm in range(nn + 1):
                delta_m = 1 if mm == 0 else 0
                g_ext_1 = 0.5 * (1 + delta_m)
                g_ext_3 = -(nn - mm + 1)

                dxddot_dC[nn, mm] = (
                    GM / (R_ref**2) * g_ext_1 * (-bnm_ext_real[nn + 1, mm + 1])
                )
                dyddot_dC[nn, mm] = (
                    GM / (R_ref**2) * g_ext_1 * (-bnm_ext_imag[nn + 1, mm + 1])
                )
                dzddot_dC[nn, mm] = GM / (R_ref**2) * g_ext_3 * bnm_ext_real[nn + 1, mm]

                dxddot_dS[nn, mm] = (
                    GM / (R_ref**2) * g_ext_1 * (-bnm_ext_imag[nn + 1, mm + 1])
                )
                dyddot_dS[nn, mm] = (
                    GM / (R_ref**2) * g_ext_1 * (bnm_ext_real[nn + 1, mm + 1])
                )
                dzddot_dS[nn, mm] = GM / (R_ref**2) * g_ext_3 * bnm_ext_imag[nn + 1, mm]

        return [dxddot_dC, dyddot_dC, dzddot_dC], [dxddot_dS, dyddot_dS, dzddot_dS]
