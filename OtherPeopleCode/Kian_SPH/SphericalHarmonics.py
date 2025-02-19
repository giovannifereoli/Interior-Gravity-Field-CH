#===========================================
# ASEN 6080: Spherical Harmonics
# Author: Kian Shakerin
#===========================================
import numpy as np
import math as mt
import pandas as pd
import research as rsc

# from SphericalHarmonicConstants import *

class SphericalHarmonics:

    def __init__(self) -> None:
        pass

    #----------------------------
    # Compute Spherical Harmonics
    #----------------------------
    @staticmethod
    def compute_spherical_harmonics(rvec: np.array, R_ref: float, GM: float, C: np.matrix, S: np.matrix, degacc=3):    
        n_degree_max = 100
            
        #bnm_ext_real, bnm_ext_imag = SH_be_terms(rvec, R_ref, degacc)
        x = rvec[0]
        y = rvec[1]
        z = rvec[2]
        r_sat = np.sqrt(x*x + y*y + z*z)
        
        x_ddot = 0
        y_ddot = 0
        z_ddot = 0
        
        dxddot_dx = 0
        dxddot_dy = 0
        dyddot_dy = 0
        dxddot_dz = 0
        dyddot_dz = 0
        dzddot_dz = 0  
        
        # Calculate unnormalized Bnm exterior
        bnm_ext_real = np.zeros((n_degree_max+3, n_degree_max+3))
        bnm_ext_imag = np.zeros((n_degree_max+3, n_degree_max+3))

        for mm in range(0, degacc+3):
            m = mm
            for nn in range(mm, degacc+3):
                n = nn
                if mm == nn:
                    
                    if mm == 0:
                        bnm_ext_real[0, 0] = R_ref/r_sat
                        bnm_ext_imag[0, 0] = 0.0
                        
                    else:
                        bnm_ext_real[nn, nn] = (2.0*n - 1.0) * (R_ref/r_sat) * ( (x/r_sat)*bnm_ext_real[nn-1, nn-1] - (y/r_sat)*bnm_ext_imag[nn-1, nn-1] )
                        bnm_ext_imag[nn, nn] = (2.0*n - 1.0) * (R_ref/r_sat) * ( (y/r_sat)*bnm_ext_real[nn-1, nn-1] + (x/r_sat)*bnm_ext_imag[nn-1, nn-1] )
                
                else:
                    
                    if nn >= 2:
                        bnm_ext_real[nn, mm] = (2.0*n - 1.0)/(n - m)*(R_ref/r_sat)*(z/r_sat)*bnm_ext_real[nn-1, mm] - (n + m - 1.0)/(n - m)*(R_ref/r_sat)*(R_ref/r_sat)*bnm_ext_real[nn-2, mm]
                        bnm_ext_imag[nn, mm] = (2.0*n - 1.0)/(n - m)*(R_ref/r_sat)*(z/r_sat)*bnm_ext_imag[nn-1, mm] - (n + m - 1.0)/(n - m)*(R_ref/r_sat)*(R_ref/r_sat)*bnm_ext_imag[nn-2, mm]

                    else:
                        bnm_ext_real[nn, mm] = (2.0*n - 1.0)/(n - m)*(R_ref/r_sat)*(z/r_sat)*bnm_ext_real[nn-1, mm]
                        bnm_ext_imag[nn, mm] = (2.0*n - 1.0)/(n - m)*(R_ref/r_sat)*(z/r_sat)*bnm_ext_imag[nn-1, mm]                   

        # Calculate accelerations and partials
        for nn in range(0, degacc+1):
            n = nn
            for mm in range(0, nn+1):
                m = mm
        
                if mm == 0:
                    delta_0_m = 1.0
                else:
                    delta_0_m = 0.0

                g_ext_1 = 0.5*(1.0 + delta_0_m)
                g_ext_2 = 0.5*(n - m + 2.0)*(n - m + 1.0)
                g_ext_3 = - (n - m + 1.0)
                
                if mm == 0:
                    x_ddot += (GM/(R_ref*R_ref))*g_ext_1*( - C[nn, mm]*bnm_ext_real[nn+1, mm+1] - S[nn, mm]*bnm_ext_imag[nn+1, mm+1] )
                    y_ddot += (GM/(R_ref*R_ref))*g_ext_1*(   S[nn, mm]*bnm_ext_real[nn+1, mm+1] - C[nn, mm]*bnm_ext_imag[nn+1, mm+1] )
                
                else:
                    x_ddot += (GM/(R_ref*R_ref))*( g_ext_1 * ( - C[nn, mm]*bnm_ext_real[nn+1, mm+1] - S[nn, mm]*bnm_ext_imag[nn+1, mm+1] ) + g_ext_2 * ( C[nn, mm]*bnm_ext_real[nn+1, mm-1] + S[nn, mm]*bnm_ext_imag[nn+1, mm-1] ) )
                    y_ddot += (GM/(R_ref*R_ref))*( g_ext_1 * (   S[nn, mm]*bnm_ext_real[nn+1, mm+1] - C[nn, mm]*bnm_ext_imag[nn+1, mm+1] ) + g_ext_2 * ( S[nn, mm]*bnm_ext_real[nn+1, mm-1] - C[nn, mm]*bnm_ext_imag[nn+1, mm-1] ) )
                        
                z_ddot += (GM/(R_ref*R_ref))*g_ext_3*( C[nn, mm]*bnm_ext_real[nn+1, mm] + S[nn, mm]*bnm_ext_imag[nn+1, mm])
                
                # Partials of xddot by dx and dy
                if mm == 0:
                    dxddot_dx += GM/(R_ref*R_ref*R_ref) * 0.5 * (  C[nn, 0] * ( bnm_ext_real[nn + 2, 2] - (n + 2.0)*(n + 1.0) * bnm_ext_real[nn + 2, 0] ) )
                    dxddot_dy += GM/(R_ref*R_ref*R_ref) * 0.5 * (  C[nn, 0] * bnm_ext_imag[nn + 2, 2] )
                
                elif mm == 1:
                    dxddot_dx += GM/(R_ref*R_ref*R_ref) * 0.25 * ( (   C[nn, 1] * ( bnm_ext_real[nn + 2, 3] - 3.0 * (n + 1.0)* n * bnm_ext_real[nn + 2, 1] ) ) + ( S[nn, 1] * ( bnm_ext_imag[nn + 2, 3] - (n + 1.0)* n * bnm_ext_imag[nn + 2, 1] ) ) )
                    dxddot_dy += GM/(R_ref*R_ref*R_ref) * 0.25 * ( ( - S[nn, 1] * ( bnm_ext_real[nn + 2, 3] + (n + 1.0)* n * bnm_ext_real[nn + 2, 1] ) ) + ( C[nn, 1] * ( bnm_ext_imag[nn + 2, 3] - (n + 1.0)* n * bnm_ext_imag[nn + 2, 1] ) ) )
                    
                elif mm > 1:
                    dxddot_dx += GM/(R_ref*R_ref*R_ref) * 0.25 * ( (   C[nn, mm] * ( bnm_ext_real[nn+2, mm+2] - 2.0*(n - m + 2.0)*(n - m + 1.0)*bnm_ext_real[nn+2, mm] + (n - m + 4.0) * (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0)*bnm_ext_real[nn+2, mm-2] ) ) + ( S[nn, mm] * ( bnm_ext_imag[nn+2, mm+2] - 2.0*(n - m + 2.0)*(n - m + 1.0)*bnm_ext_imag[nn+2, mm] + (n - m + 4.0) * (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0)*bnm_ext_imag[nn+2, mm-2] ) ) )
                    dxddot_dy += GM/(R_ref*R_ref*R_ref) * 0.25 * ( ( - S[nn, mm] * ( bnm_ext_real[nn+2, mm+2] - (n - m + 4.0) * (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0)*bnm_ext_real[nn+2, mm-2] ) ) + ( C[nn, mm] * ( bnm_ext_imag[nn+2, mm+2] - (n - m + 4.0) * (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0)*bnm_ext_imag[nn+2, mm-2] ) ) )

                if ( mm == 0):                
                    dxddot_dz += GM/(R_ref*R_ref*R_ref) * (n + 1.0) * C[nn, 0] * bnm_ext_real[nn+2, 1]
                    dyddot_dz += GM/(R_ref*R_ref*R_ref) * (n + 1.0) * C[nn, 0] * bnm_ext_imag[nn+2, 1]
                    
                else:
                    dxddot_dz += GM/(R_ref*R_ref*R_ref) * 0.5 * ( (   C[nn, mm] * ( (n - m + 1.0) *  bnm_ext_real[nn+2, mm+1] - (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0) * bnm_ext_real[nn+2, mm-1] ) ) + ( S[nn, mm] * ( (n - m + 1.0) *  bnm_ext_imag[nn+2, mm+1] - (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0) * bnm_ext_imag[nn+2, mm-1] ) ) )
                    dyddot_dz += GM/(R_ref*R_ref*R_ref) * 0.5 * ( ( - S[nn, mm] * ( (n - m + 1.0) *  bnm_ext_real[nn+2, mm+1] + (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0) * bnm_ext_real[nn+2, mm-1] ) ) + ( C[nn, mm] * ( (n - m + 1.0) *  bnm_ext_imag[nn+2, mm+1] + (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0) * bnm_ext_imag[nn+2, mm-1] ) ) )
                
                dzddot_dz += GM/(R_ref*R_ref*R_ref) * ( (n - m + 2.0) * (n - m + 1.0) * ( C[nn, mm] * bnm_ext_real[nn+2, mm] + S[nn, mm] * bnm_ext_imag[nn+2, mm] ) )
                
        dyddot_dy = - dxddot_dx - dzddot_dz
        
        accelerations = np.array([x_ddot, y_ddot, z_ddot])
        
        partials_of_accels = np.array([
            [dxddot_dx, dxddot_dy, dxddot_dz],
            [dxddot_dy, dyddot_dy, dyddot_dz],
            [dxddot_dz, dyddot_dz, dzddot_dz]
        ])
        
        dxddot_dC = np.zeros((degacc+1, degacc+1))
        dyddot_dC = np.zeros((degacc+1, degacc+1))
        dzddot_dC = np.zeros((degacc+1, degacc+1))

        dxddot_dS = np.zeros((degacc+1, degacc+1))
        dyddot_dS = np.zeros((degacc+1, degacc+1))
        dzddot_dS = np.zeros((degacc+1, degacc+1))
        
        for nn in range(0, degacc+1):
            n = nn
            for mm in range(0, nn+1):
                m = mm
        
                if mm == 0:
                    delta_0_m = 1.0
                else:
                    delta_0_m = 0.0
                    
                g_ext_1 = 0.5*(1.0 + delta_0_m)
                g_ext_2 = 0.5*(n - m + 2.0)*(n - m + 1.0)
                g_ext_3 = - (n - m + 1.0)
            
                if mm == 0:
                    dxddot_dC[nn, mm] = GM/(R_ref*R_ref)*g_ext_1*( - bnm_ext_real[nn+1, mm+1] )
                    dyddot_dC[nn, mm] = GM/(R_ref*R_ref)*g_ext_1*( - bnm_ext_imag[nn+1, mm+1] )
                    
                else:
                    dxddot_dC[nn, mm] = GM/(R_ref*R_ref)*( g_ext_1 * ( - bnm_ext_real[nn+1, mm+1] ) + g_ext_2 * (   bnm_ext_real[nn+1, mm-1] ) )
                    dyddot_dC[nn, mm] = GM/(R_ref*R_ref)*( g_ext_1 * ( - bnm_ext_imag[nn+1, mm+1] ) + g_ext_2 * ( - bnm_ext_imag[nn+1, mm-1] ) )
                    dxddot_dS[nn, mm] = GM/(R_ref*R_ref)*( g_ext_1 * ( - bnm_ext_imag[nn+1, mm+1] ) + g_ext_2 * (   bnm_ext_imag[nn+1, mm-1] ) )
                    dyddot_dS[nn, mm] = GM/(R_ref*R_ref)*( g_ext_1 * (   bnm_ext_real[nn+1, mm+1] ) + g_ext_2 * (   bnm_ext_real[nn+1, mm-1] ) )
                
                dzddot_dC[nn, mm] = GM/(R_ref*R_ref) * g_ext_3 * bnm_ext_real[nn+1, mm]
                dzddot_dS[nn, mm] = GM/(R_ref*R_ref) * g_ext_3 * bnm_ext_imag[nn+1, mm]
                
        
        partials_by_C = [dxddot_dC, dyddot_dC, dzddot_dC]
        partials_by_S = [dxddot_dS, dyddot_dS, dzddot_dS]
        
        return accelerations, partials_of_accels, partials_by_C, partials_by_S