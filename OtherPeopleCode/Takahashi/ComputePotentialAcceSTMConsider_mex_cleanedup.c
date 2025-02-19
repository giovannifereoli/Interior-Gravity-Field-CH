///////////////////////////////////////////////////////////////////////////
// Small Body Characterization
// Author: Yu Takahashi (The University of Coloradto at Boulder)
// Advisor: Dr. Scheeres (The University of Colorado at Boulder)
// Acknowledgement: Theodore Sweetser and JPL for their support and funding
///////////////////////////////////////////////////////////////////////////
//
////// Description:
//
//  This function computes the spacecraft potential, acceleration, 
//  and the acceleration for the STM and consider STM.
//
////// Inputs:
//
//     deg_acce                  : [n.d.]   Degree of the spherical harmonics for acceleration
//
//     deg_estimate              : [n.d.]   Degree of the spherical harmonics to be estimated
//
//     deg_consider              : [n.d.]   Degree of the spherical harmonics to be considered
//
//     R_ref                     : [km]     Reference radius ( = Planet Radius )
//
//     GM_ref                    : [km^3/sec^2]     Reference GM ( = G* Planet mass )
//
//     SatPos_Ptr                : [km]     Spacecraft position
//
//     C_Ptr                     : [n.d.]   C spherical harmonics
//
//     S_Ptr                     : [n.d.]   S spherical harmonics
//
////// Outputs:
//
//     U_Ptr          : [km^2/s^2] Spacecraft potential
//
//     Acce_Ptr       : [km/s^2]   Spacecraft acceleration
//
//     A_Acce_Pos_Ptr : [1/s^2]    A-matrix for the state transition matrix in the body frame
//
//     A_Acce_C_Ptr   : [km/s^2]   Partials of the spacecraft acceleration w.r.t. the C spherical harmonics (for STM)
//
//     A_Acce_S_Ptr   : [km/s^2]   Partials of the spacecraft acceleration w.r.t. the S spherical harmonics (for STM)
//
//     B_Acce_C_Ptr   : [km/s^2]   Partials of the spacecraft acceleration w.r.t. the C spherical harmonics (for consider STM)
//
//     B_Acce_S_Ptr   : [km/s^2]   Partials of the spacecraft acceleration w.r.t. the S spherical harmonics (for consider STM)
//
////// Assumptions/References:
//
//  - None
//
////// Note:
//
//  - None
//
////// Dependencies:
//
//  - None
//
////// Call
//
//  - None
//
////// Called by
//
//  - flyby_acce.m/landing_acce.m
//  - flyby_acce_target.m/landing_acce_target.m
//  - flyby_acce_maneuver.m/landing_acce_maneuver.m
//
// Modification History:
//
//  15Jul10   Yu Takahashi   original version
//  04Mar11   Yu Takahashi   1st revision
//
///////////////////////////////////////////////////////////////////////////

#include    "mex.h"
#include	<math.h>
#include	<stdarg.h>
#include	<stdio.h>
#include	<stdlib.h>
#include    <string.h>

#define ABS(x) ((x) < 0) ? -(x) : (x)

#define G 6.67384E-20

#define n_degree_max    100

#define num_C_max       1000
#define num_S_max       1000
#define num_consd_C_max 1000
#define num_consd_S_max 1000

///////////////////
// -- Outputs -- //
///////////////////

double *U_Ptr, *Acce_Ptr, *A_Acce_Pos_Ptr;
double *A_Acce_C_Ptr, *A_Acce_S_Ptr, *B_Acce_C_Ptr, *B_Acce_S_Ptr;

//////////////////
// -- Inputs -- //
//////////////////

int    deg_acce, deg_estimate, deg_consider;
double R_ref, M_ref, GM_ref;
double *SatPos_Ptr, SatPos [3];
double *C_Ptr, *S_Ptr;
int    options_exterior_interior, options_harmonics_normalization;

////////////////////
// -- Indexing -- //
////////////////////

int ee, ff, gg, ii, mm, nn, xx, yy, zz;
int Index_C, Index_S, Index_consd_C, Index_consd_S;

//////////////////////////////
// -- Number of Elements -- //
//////////////////////////////

int num_C, num_S, num_consd_C, num_consd_S;

//////////////////////////////////////
// -- Degree of Acceleration/STM -- //
//////////////////////////////////////

double n, m;
double delta_0_m, delta_1_n, delta_1_m, delta_2_m;
double g_ext_1, g_ext_2, g_ext_3;
double g_bar_ext_1, g_bar_ext_2, g_bar_ext_3;
double g_int_1, g_int_2, g_int_3;
double g_bar_int_1, g_bar_int_2, g_bar_int_3;
double s_bar_ext_1, s_bar_ext_2, s_bar_ext_3, s_bar_ext_4, s_bar_ext_5, s_bar_ext_6;
double s_bar_int_1, s_bar_int_2, s_bar_int_3, s_bar_int_4, s_bar_int_5, s_bar_int_6;

////////////////////////////////
// -- Satellite Parameters -- //
////////////////////////////////

double x, y, z, U;
double x_ddot, y_ddot, z_ddot;
double dxddot_dx, dxddot_dy, dxddot_dz, dyddot_dz, dzddot_dz;
double Acce [3], A_Acce_Pos [3][3];
double dxddot_dC    [n_degree_max + 1][n_degree_max + 1], dxddot_dS    [n_degree_max + 1][n_degree_max + 1];
double dyddot_dC    [n_degree_max + 1][n_degree_max + 1], dyddot_dS    [n_degree_max + 1][n_degree_max + 1];
double dzddot_dC    [n_degree_max + 1][n_degree_max + 1], dzddot_dS    [n_degree_max + 1][n_degree_max + 1];
double dxddot_dCbar [n_degree_max + 1][n_degree_max + 1], dxddot_dSbar [n_degree_max + 1][n_degree_max + 1];
double dyddot_dCbar [n_degree_max + 1][n_degree_max + 1], dyddot_dSbar [n_degree_max + 1][n_degree_max + 1];
double dzddot_dCbar [n_degree_max + 1][n_degree_max + 1], dzddot_dSbar [n_degree_max + 1][n_degree_max + 1];
double A_Acce_C [3][num_C_max], A_Acce_S [3][num_S_max];
double B_Acce_C[3][num_consd_C_max], B_Acce_S [3][num_consd_S_max];
    
/////////////////////////
// -- Gravity Field -- //
/////////////////////////

double r_sat;
double C [n_degree_max+1][n_degree_max+1], S [n_degree_max+1][n_degree_max+1];
double Cbar [n_degree_max+1][n_degree_max+1], Sbar [n_degree_max+1][n_degree_max+1];

/////////////////
// -- Basis -- //
/////////////////

// - b_{n,m}^e

double bnm_ext_real[n_degree_max + 3][n_degree_max + 3];
double bnm_ext_imag[n_degree_max + 3][n_degree_max + 3];

// - \bar{b}_{n,m}^e

double bnm_bar_ext_real[n_degree_max + 3][n_degree_max + 3];
double bnm_bar_ext_imag[n_degree_max + 3][n_degree_max + 3];

// - b_{n,m}^i

double bnm_int_real[n_degree_max + 1][n_degree_max + 1];
double bnm_int_imag[n_degree_max + 1][n_degree_max + 1];

// - \bar{b}_{n,m}^i

double bnm_bar_int_real[n_degree_max + 1][n_degree_max + 1];
double bnm_bar_int_imag[n_degree_max + 1][n_degree_max + 1];

/////////////////////
// -- Functions -- //
/////////////////////

void GetBnmUnnormalizedExterior(void);
void GetExteriorPotentialAccelerationSTMConsider(void);

/***************************************************************
 * main program
 ***************************************************************/
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) 
{
    
    /////////////////////////////
    // -- Assign the Inputs -- //
    /////////////////////////////
    
    deg_acce                         = mxGetScalar(prhs[0]);
    deg_estimate                     = mxGetScalar(prhs[1]);
    deg_consider                     = mxGetScalar(prhs[2]);
    R_ref                            = mxGetScalar(prhs[3]);
    GM_ref                           = mxGetScalar(prhs[4]);
    SatPos_Ptr                       = mxGetPr(prhs[5]);
    C_Ptr                            = mxGetPr(prhs[6]);
    S_Ptr                            = mxGetPr(prhs[7]);
    
    //////////////////////////////
    // -- Number of Elements -- //
    //////////////////////////////
    
    if (deg_estimate == 0) {
        
        num_C  = 1;
        num_S  = 1;
        
    } else {
        
        num_C  = (int) ( (deg_estimate + 2)*(deg_estimate + 1)/2 - 1 );
        num_S  = (int) ( deg_estimate*(deg_estimate + 1)/2 );
        
    } // For if
    
    if (deg_estimate == deg_consider) {
        
        num_consd_C = 1;
        num_consd_S = 1;
    
    } else {
        
        num_consd_C = (int) ( (deg_consider + 1)*(deg_consider + 2)/2 - (deg_estimate + 1)*(deg_estimate + 2)/2 );
        num_consd_S = (int) ( deg_consider*(deg_consider + 1)/2 - deg_estimate*(deg_estimate + 1)/2 );
        
    } // % For if
    
    //////////////////////////////
    // -- Assign the Outputs -- //
    //////////////////////////////
        
    plhs[0]   = mxCreateDoubleMatrix(1,1, mxREAL);
    plhs[1]   = mxCreateDoubleMatrix(3,1, mxREAL);
    plhs[2]   = mxCreateDoubleMatrix(3,3, mxREAL);
    plhs[3]   = mxCreateDoubleMatrix(3,num_C, mxREAL);
    plhs[4]   = mxCreateDoubleMatrix(3,num_S, mxREAL);
    
    U_Ptr           =  mxGetPr(plhs[0]);
    Acce_Ptr        =  mxGetPr(plhs[1]);
    A_Acce_Pos_Ptr  =  mxGetPr(plhs[2]);
    A_Acce_C_Ptr    =  mxGetPr(plhs[3]);
    A_Acce_S_Ptr    =  mxGetPr(plhs[4]);
    
    ///////////////////////////////
    // -- Spherical Harmonics -- //
    ///////////////////////////////
            
    for (mm = 0; mm <= deg_acce; mm++) {
        
        for (nn = mm; nn <= deg_acce; nn++) {
                            
            C[nn][mm]    = C_Ptr[(deg_acce+1)*mm+nn];
            S[nn][mm]    = S_Ptr[(deg_acce+1)*mm+nn];
                        
        } // for mm
        
    } // for nn
    
    //////////////////////////////////////////////////
    // -- Compute the Potential/Acceleration/STM -- //
    //////////////////////////////////////////////////
       
    x         = SatPos_Ptr[0]; y         = SatPos_Ptr[1]; z         = SatPos_Ptr[2];
    x_ddot    = 0.0;           y_ddot    = 0.0;           z_ddot    = 0.0;
    dxddot_dx = 0.0;           dxddot_dy = 0.0;           dxddot_dz = 0.0; 
    dyddot_dz = 0.0;           dzddot_dz = 0.0;
    
    r_sat = sqrt(x*x + y*y + z*z);
    
    GetExteriorPotentialAccelerationSTMConsider();
    
    /////////////////////////////////////////////
    // -- Assign Potential/Acceleration/STM -- //
    /////////////////////////////////////////////
    
    U_Ptr[0]         = U;
    
    Acce[0]          = x_ddot;
    Acce[1]          = y_ddot;
    Acce[2]          = z_ddot;
    
    Acce_Ptr[0]      = x_ddot;
    Acce_Ptr[1]      = y_ddot;
    Acce_Ptr[2]      = z_ddot;
    
    A_Acce_Pos[0][0] = dxddot_dx;
    A_Acce_Pos[0][1] = dxddot_dy;
    A_Acce_Pos[0][2] = dxddot_dz;
    A_Acce_Pos[1][2] = dyddot_dz;
    A_Acce_Pos[2][2] = dzddot_dz;
    A_Acce_Pos[1][0] = A_Acce_Pos[0][1];
    A_Acce_Pos[1][1] = - A_Acce_Pos[0][0] - A_Acce_Pos[2][2];
    A_Acce_Pos[2][0] = A_Acce_Pos[0][2];
    A_Acce_Pos[2][1] = A_Acce_Pos[1][2];
        
    for (mm = 0; mm < 3; mm++) {
        
        for (nn = 0; nn < 3; nn++) {
  
            A_Acce_Pos_Ptr[mm*3+nn] = A_Acce_Pos[nn][mm];
                        
        } // For nn
        
    } // For mm
    
    ////////////////////////////////////////////
    // -- Assign the STM components for CS -- //
    ////////////////////////////////////////////
    
    for (nn = 1; nn <= deg_estimate; nn++) {
        
        n = (double) nn;
        
        for (mm = 0; mm <= nn; mm++) {
            
            m = (double) mm;
            
            // For C
            
            Index_C = (int) (n*(n+1)/2 - 1 + m);
                           
            A_Acce_C[0][Index_C]  = dxddot_dC[nn][mm];
            A_Acce_C[1][Index_C]  = dyddot_dC[nn][mm];
            A_Acce_C[2][Index_C]  = dzddot_dC[nn][mm];
            
            A_Acce_C_Ptr[3*Index_C]   = A_Acce_C[0][Index_C];
            A_Acce_C_Ptr[3*Index_C+1] = A_Acce_C[1][Index_C];
            A_Acce_C_Ptr[3*Index_C+2] = A_Acce_C[2][Index_C];
            
            // For S
            
            if (m != 0) { // S_{n0} is not in the state vector
                
                Index_S = (int) (n*(n-1)/2 + m - 1);
                                 
                A_Acce_S[0][Index_S] = dxddot_dS[nn][mm];
                A_Acce_S[1][Index_S] = dyddot_dS[nn][mm];
                A_Acce_S[2][Index_S] = dzddot_dS[nn][mm];
                
                // For CS
                
                A_Acce_S_Ptr[3*Index_S]   = A_Acce_S[0][Index_S];
                A_Acce_S_Ptr[3*Index_S+1] = A_Acce_S[1][Index_S];
                A_Acce_S_Ptr[3*Index_S+2] = A_Acce_S[2][Index_S];
                
            } // For if
            
        } // For mm
        
    } // For nn
            
    return;
    
} // For main

void GetBnmUnnormalizedExterior(void)

{
    
    ////////////////////////////////
    // -- Vertical Recurrences -- //
    ////////////////////////////////
    
    for (mm = 0; mm <= deg_acce + 2; mm++) {
        
        m = (double) mm;
        
        for (nn = mm; nn <= deg_acce + 2; nn++) {
            
            n = (double) nn;
            
            // Recursive Formulae
            
            if (mm == nn) {
                
                if (mm == 0) {
                    
                    bnm_ext_real[0][0] = R_ref/r_sat;
                    bnm_ext_imag[0][0] = 0.0;
                    
                } else {
                    
                    bnm_ext_real[nn][nn] = (2.0*n - 1.0) * (R_ref/r_sat) * ( x/r_sat*bnm_ext_real[nn-1][nn-1] - y/r_sat*bnm_ext_imag[nn-1][nn-1] );
                    bnm_ext_imag[nn][nn] = (2.0*n - 1.0) * (R_ref/r_sat) * ( y/r_sat*bnm_ext_real[nn-1][nn-1] + x/r_sat*bnm_ext_imag[nn-1][nn-1] );
                    
                } // For if
                
            } // For the Diagonals
            
            else {
                
                if ( nn >= 2 ) {
                    
                    bnm_ext_real[nn][mm] = (2.0*n - 1.0)/(n - m)*(R_ref/r_sat)*(z/r_sat)*bnm_ext_real[nn-1][mm] - (n + m - 1.0)/(n - m)*(R_ref/r_sat)*(R_ref/r_sat)*bnm_ext_real[nn-2][mm];
                    bnm_ext_imag[nn][mm] = (2.0*n - 1.0)/(n - m)*(R_ref/r_sat)*(z/r_sat)*bnm_ext_imag[nn-1][mm] - (n + m - 1.0)/(n - m)*(R_ref/r_sat)*(R_ref/r_sat)*bnm_ext_imag[nn-2][mm];
                    
                } else {
                    
                    bnm_ext_real[nn][mm] = (2.0*n - 1.0)/(n - m)*(R_ref/r_sat)*(z/r_sat)*bnm_ext_real[nn-1][mm];
                    bnm_ext_imag[nn][mm] = (2.0*n - 1.0)/(n - m)*(R_ref/r_sat)*(z/r_sat)*bnm_ext_imag[nn-1][mm];
                    
                } // For if
                
            } // For the Verticals
            
        } // For nn
        
    } // For mm
    
    return;
    
} // For GetBnmUnnormalizedExterior

void GetExteriorPotentialAccelerationSTMConsider(void)

{
    
    ////////////////////////////////////////////
    // -- Potential, Acceleration, and STM -- //
    ////////////////////////////////////////////
    
    U = 0.0;
            
    GetBnmUnnormalizedExterior();
    
    for (nn = 0; nn <= deg_acce; nn++) {
        
        n = (double) nn;
        
        for (mm = 0; mm <= nn; mm++) {
            
            m = (double) mm;
            
            /////////////////////
            // -- Potential -- //
            /////////////////////
            
            U += GM_ref/R_ref*(bnm_ext_real[nn][mm]*C[nn][mm] + bnm_ext_imag[nn][mm]*S[nn][mm]);
            
            //////////////////////////
            // -- Delta Function -- //
            //////////////////////////
            
            if (mm == 0) {
                
                delta_0_m = 1.0;
                
            } else {
                
                delta_0_m = 0.0;
                
            } // For if
            
            if (mm == 1) {
                
                delta_1_m = 1.0;
                
            } else {
                
                delta_1_m = 0.0;
                
            } // For if
            
            if (mm == 2) {
                
                delta_2_m = 1.0;
                
            } else {
                
                delta_2_m = 0.0;
                
            } // For if
            
            ////////////////////////
            // -- Acceleration -- //
            ////////////////////////
            
            g_ext_1 = 0.5*(1.0 + delta_0_m);
            g_ext_2 = 0.5*(n - m + 2.0)*(n - m + 1.0);
            g_ext_3 = - (n - m + 1.0);
            
            if (mm == 0) {
                
                // -- Unnormalized
                
                x_ddot += GM_ref/(R_ref*R_ref)*g_ext_1*( - C[nn][mm]*bnm_ext_real[nn+1][mm+1] - S[nn][mm]*bnm_ext_imag[nn+1][mm+1] );
                y_ddot += GM_ref/(R_ref*R_ref)*g_ext_1*(   S[nn][mm]*bnm_ext_real[nn+1][mm+1] - C[nn][mm]*bnm_ext_imag[nn+1][mm+1] );
                
            } else {
                
                // -- Unnormalized
                
                x_ddot += GM_ref/(R_ref*R_ref)*( g_ext_1 * ( - C[nn][mm]*bnm_ext_real[nn+1][mm+1] - S[nn][mm]*bnm_ext_imag[nn+1][mm+1] ) + g_ext_2 * ( C[nn][mm]*bnm_ext_real[nn+1][mm-1] + S[nn][mm]*bnm_ext_imag[nn+1][mm-1] ) );
                y_ddot += GM_ref/(R_ref*R_ref)*( g_ext_1 * (   S[nn][mm]*bnm_ext_real[nn+1][mm+1] - C[nn][mm]*bnm_ext_imag[nn+1][mm+1] ) + g_ext_2 * ( S[nn][mm]*bnm_ext_real[nn+1][mm-1] - C[nn][mm]*bnm_ext_imag[nn+1][mm-1] ) );
                
            } // For if
            
            // -- Unnormalized
            
            z_ddot += GM_ref/(R_ref*R_ref)*g_ext_3*( C[nn][mm]*bnm_ext_real[nn+1][mm] + S[nn][mm]*bnm_ext_imag[nn+1][mm]);
            
            ///////////////
            // -- STM -- //
            ///////////////
            
            // -- Partial of xddot/x and xddot/y
            
            if ( mm == 0) {
                
                // -- Unnormalized
                
                dxddot_dx += GM_ref/(R_ref*R_ref*R_ref) * 0.5 * (  C[nn][0] * ( bnm_ext_real[nn + 2][2] - (n + 2.0)*(n + 1.0) * bnm_ext_real[nn + 2][0] ) );
                dxddot_dy += GM_ref/(R_ref*R_ref*R_ref) * 0.5 * (  C[nn][0] * bnm_ext_imag[nn + 2][2] );
                
            } else if ( mm == 1) {
                
                // -- Unnormalized
                
                dxddot_dx += GM_ref/(R_ref*R_ref*R_ref) * 0.25 * ( (   C[nn][1] * ( bnm_ext_real[nn + 2][3] - 3.0 * (n + 1.0)* n * bnm_ext_real[nn + 2][1] ) ) + ( S[nn][1] * ( bnm_ext_imag[nn + 2][3] - (n + 1.0)* n * bnm_ext_imag[nn + 2][1] ) ) );
                dxddot_dy += GM_ref/(R_ref*R_ref*R_ref) * 0.25 * ( ( - S[nn][1] * ( bnm_ext_real[nn + 2][3] + (n + 1.0)* n * bnm_ext_real[nn + 2][1] ) ) + ( C[nn][1] * ( bnm_ext_imag[nn + 2][3] - (n + 1.0)* n * bnm_ext_imag[nn + 2][1] ) ) );
                
            } else if ( mm > 1) {
                
                // -- Unnormalized
                
                dxddot_dx += GM_ref/(R_ref*R_ref*R_ref) * 0.25 * ( (   C[nn][mm] * ( bnm_ext_real[nn+2][mm+2] - 2.0*(n - m + 2.0)*(n - m + 1.0)*bnm_ext_real[nn+2][mm] + (n - m + 4.0) * (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0)*bnm_ext_real[nn+2][mm-2] ) ) + ( S[nn][mm] * ( bnm_ext_imag[nn+2][mm+2] - 2.0*(n - m + 2.0)*(n - m + 1.0)*bnm_ext_imag[nn+2][mm] + (n - m + 4.0) * (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0)*bnm_ext_imag[nn+2][mm-2] ) ) );
                dxddot_dy += GM_ref/(R_ref*R_ref*R_ref) * 0.25 * ( ( - S[nn][mm] * ( bnm_ext_real[nn+2][mm+2] - (n - m + 4.0) * (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0)*bnm_ext_real[nn+2][mm-2] ) ) + ( C[nn][mm] * ( bnm_ext_imag[nn+2][mm+2] - (n - m + 4.0) * (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0)*bnm_ext_imag[nn+2][mm-2] ) ) );
                
            } // For mm
            
            // -- Partial of xddot/z and dyddot/dz
            
            if ( mm == 0) {
                
                // -- Unnormalized
                
                dxddot_dz += GM_ref/(R_ref*R_ref*R_ref) * (n + 1.0) * C[nn][0] * bnm_ext_real[nn+2][1];
                dyddot_dz += GM_ref/(R_ref*R_ref*R_ref) * (n + 1.0) * C[nn][0] * bnm_ext_imag[nn+2][1];
                
            } else {
                
                // -- Unnormalized
                
                dxddot_dz += GM_ref/(R_ref*R_ref*R_ref) * 0.5 * ( (   C[nn][mm] * ( (n - m + 1.0) *  bnm_ext_real[nn+2][mm+1] - (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0) * bnm_ext_real[nn+2][mm-1] ) ) + ( S[nn][mm] * ( (n - m + 1.0) *  bnm_ext_imag[nn+2][mm+1] - (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0) * bnm_ext_imag[nn+2][mm-1] ) ) );
                dyddot_dz += GM_ref/(R_ref*R_ref*R_ref) * 0.5 * ( ( - S[nn][mm] * ( (n - m + 1.0) *  bnm_ext_real[nn+2][mm+1] + (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0) * bnm_ext_real[nn+2][mm-1] ) ) + ( C[nn][mm] * ( (n - m + 1.0) *  bnm_ext_imag[nn+2][mm+1] + (n - m + 3.0) * (n - m + 2.0) * (n - m + 1.0) * bnm_ext_imag[nn+2][mm-1] ) ) );
                
            } // For mm
            
            // -- Partial of zddot/z
            
            // -- Unnormalized
            
            dzddot_dz += GM_ref/(R_ref*R_ref*R_ref) * ( (n - m + 2.0) * (n - m + 1.0) * ( C[nn][mm] * bnm_ext_real[nn+2][mm] + S[nn][mm] * bnm_ext_imag[nn+2][mm] ) );
            
        } // For mm
        
    } // For nn
        
    ///////////////////////////////////
    // -- Spherical Harmonics STM -- //
    ///////////////////////////////////
    
    for (nn = 0; nn <= deg_consider; nn++) {
        
        n = (double) nn;
        
        for (mm = 0; mm <= nn; mm++) {
            
            m = (double) mm;
            
            if (mm == 0) {
                
                delta_0_m = 1.0;
                
            } else {
                
                delta_0_m = 0.0;
                
            } // For if
            
            g_ext_1 = 0.5*(1.0 + delta_0_m);
            g_ext_2 = 0.5*(n - m + 2.0)*(n - m + 1.0);
            g_ext_3 = - (n - m + 1.0);
            
            if (mm == 0) {
                
                // -- Unnormalized
                
                dxddot_dC[nn][mm] = GM_ref/(R_ref*R_ref)*g_ext_1*( - bnm_ext_real[nn+1][mm+1] );
                dyddot_dC[nn][mm] = GM_ref/(R_ref*R_ref)*g_ext_1*( - bnm_ext_imag[nn+1][mm+1] );
                
            } else {
                
                // -- Unnormalized
                
                dxddot_dC[nn][mm] = GM_ref/(R_ref*R_ref)*( g_ext_1 * ( - bnm_ext_real[nn+1][mm+1] ) + g_ext_2 * (   bnm_ext_real[nn+1][mm-1] ) );
                dyddot_dC[nn][mm] = GM_ref/(R_ref*R_ref)*( g_ext_1 * ( - bnm_ext_imag[nn+1][mm+1] ) + g_ext_2 * ( - bnm_ext_imag[nn+1][mm-1] ) );
                dxddot_dS[nn][mm] = GM_ref/(R_ref*R_ref)*( g_ext_1 * ( - bnm_ext_imag[nn+1][mm+1] ) + g_ext_2 * (   bnm_ext_imag[nn+1][mm-1] ) );
                dyddot_dS[nn][mm] = GM_ref/(R_ref*R_ref)*( g_ext_1 * (   bnm_ext_real[nn+1][mm+1] ) + g_ext_2 * (   bnm_ext_real[nn+1][mm-1] ) );
                
            } // For m == 0
            
            // -- Unnormalized
            
            dzddot_dC[nn][mm] = GM_ref/(R_ref*R_ref) * g_ext_3 * bnm_ext_real[nn+1][mm];
            dzddot_dS[nn][mm] = GM_ref/(R_ref*R_ref) * g_ext_3 * bnm_ext_imag[nn+1][mm];
            
        } // For mm
        
    } // For nn
    
} // For GetExteriorPotentialAccelerationSTMConsider


///////////////////////////////////////////////////////////////////////////
