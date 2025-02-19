%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Small Body Characterization
% Author: Yu Takahashi (The University of Coloradto at Boulder)
% Advisor: Dr. Scheeres (The University of Colorado at Boulder)
% Acknowledgement: Theodore Sweetser and JPL for their support and funding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%% Description:
%
%  This function computes the acceleration of the state vector for the
%  numerical covariance analysis.
%
%%% Inputs:
%
%   - t (s) : time
%
%   - state : state vector + state transition matrix + asteroid/Earth's
%             dynamics
%
%   - init  : initial settings
%
%%% Outputs:
%
%   - deri: acceleration of the state vector
%
%%% Assumptions/References:
%
%   - None
%
%%% Dependencies:
%
%   - None
%
%%% Note:
%
%   - Regular_state : spacecraft position/velocity, asteroid
%                     position/velocity, \mu, and SRP scaling factor
%
%%% Call
%
%   - GetRotationMatrices.m
%   - GetRotFromQuaternion.m
%   - ComputePotentialAcceSTMConsider_mex.c
%   - flyby_acce_mex.c
%
%%% Called by
%
%   - LSB_Covariance_Flybys.m
%
%%% Modification History:
%
%   06Jul10   Yu Takahashi   original version
%   17Dec10   Yu Takahashi   1st revision
%   05Mar11   Yu Takahashi   2nd revision
%   25Aug11   Yu Takahashi   3rd revision
%   24Oct11   Yu Takahashi   4th revision
%   23Apr12   Yu Takahashi   5th revision
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function deri = flyby_acce(t, state, init)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if init.options.ode_mex == 0 % For Regular MATLAB Compuation
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Satellite Position and Velocity -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    x_sat         =  state(1);
    y_sat         =  state(2);
    z_sat         =  state(3);
    xdot_sat      =  state(4);
    ydot_sat      =  state(5);
    zdot_sat      =  state(6);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Asteroid Position and Velocity -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    x_ast         =  state(7);
    y_ast         =  state(8);
    z_ast         =  state(9);
    xdot_ast      =  state(10);
    ydot_ast      =  state(11);
    zdot_ast      =  state(12);
    
    %%%%%%%%%%%%%%%%%%%%%%
    %% -- Mu and SRP -- %%
    %%%%%%%%%%%%%%%%%%%%%%
    
    mu            =  init.Ast.mu;
    SRP_scale     =  init.SRP.scale;
          
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Attitude Coordinates -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if init.options.attitude_dynamics == 1       % 3-1-3 Euler angles
        
        % 3-1-3 Euler angles at t = 0
        
        alpha     = init.Ast.alpha_0; % [rad.]   alpha
        beta      = init.Ast.beta_0;  % [rad.]   beta
        gamma_0   = init.Ast.gamma_0; % [rad.]   gamma
        gamma_dot = init.Ast.ws;      % [rad./s] gamma_dot
        
        % Rotation about the body z-axis
        
        OmegaZBody    =  gamma_dot*t;
        
        ZRotation     =  [cos(OmegaZBody),  sin(OmegaZBody), 0;
            -sin(OmegaZBody), cos(OmegaZBody), 0;
            0                 0                1];   % Rotation about the body z-axis due the Asteroid rotation
        
        % -- Rotation matrix
        
        Rot = GetRotationMatrices(alpha, beta, gamma_0);
        
        % --  For the rotation between the Inertial and the Body-frame
        
        RotTotal   =  ZRotation * Rot.ThreeRot;   % Total Rotation from the inertial frame to the body frame
        
        BN   =  RotTotal;   % [n.d.] Inertial frame to the body frame
        NB   =  BN';        % [n.d.] Body frame to the inertial frame
        
        if ( init.options.estimate.attitude == 1 ) || ( init.options.consider.attitude == 1 )
            
            dRot_dalpha       =  Rot.dRot_dalpha;
            dRot_dbeta        =  Rot.dRot_dbeta;
            dRot_dgamma_0     =  Rot.dRot_dgamma_0;
            dRot_dgamma_dot   =  [- t * sin(OmegaZBody), - t * cos(OmegaZBody), 0;
                t * cos(OmegaZBody), - t * sin(OmegaZBody), 0;
                0,                     0, 0];
            
            dT_dalpha         =  dRot_dalpha * Rot.InvSecondX * Rot.InvThirdZ * ZRotation';
            dT_dbeta          =  Rot.InvFirstZ * dRot_dbeta * Rot.InvThirdZ * ZRotation';
            dT_dgamma_0       =  Rot.InvFirstZ * Rot.InvSecondX * dRot_dgamma_0 * ZRotation';
            dT_dgamma_dot     =  Rot.InvFirstZ * Rot.InvSecondX * Rot.InvThirdZ * dRot_dgamma_dot;
            
        end % For if
        
    elseif init.options.attitude_dynamics == 2     % Quaternion
        
        % -- Inertia tensor
        
        if init.options.normalize_inertia == 0
            
            I      =  init.Ast.I_tensor;     % [kg*km^2]     3 x 3 Inertia tensor
            I_inv  =  init.Ast.I_inv;        % [1/(kg*km^2)] 3 x 3 Inverse of the Inertia tensor
            
        elseif init.options.normalize_inertia == 1
            
            I      =  init.Ast.I_tensor_bar; % [n.d.] 3 x 3 Inertia tensor
            I_inv  =  init.Ast.I_inv_bar;    % [n.d.] 3 x 3 Inverse of the Inertia tensor
            
        end % For if
        
        % -- Quaternion
        
        q_0 = state(init.index.state.attitude_first + 0); % [n.d.] q_0
        q_1 = state(init.index.state.attitude_first + 1); % [n.d.] q_1
        q_2 = state(init.index.state.attitude_first + 2); % [n.d.] q_2
        q_3 = state(init.index.state.attitude_first + 3); % [n.d.] q_3
        
        q   = [q_0; q_1; q_2; q_3]; % [n.d.] Quaternion
        q   = q/norm(q);            % [n.d.] Normalization
        
        % -- Angular velocity
        
        omega_1 = state(init.index.state.angular_velocity_first + 0); % [rad./s] omega_1
        omega_2 = state(init.index.state.angular_velocity_first + 1); % [rad./s] omega_2
        omega_3 = state(init.index.state.angular_velocity_first + 2); % [rad./s] omega_3
        
        omega_vec   = [omega_1; omega_2; omega_3]; % [rad./s] Angular velocity
        
        omega_tilde = [0, -omega_3, omega_2;
            omega_3, 0, -omega_1;
            -omega_2, omega_1, 0];  % [rad./s] Angular velocity
        
        I_omega   = I*omega_vec;  % [kg*km^2/s] Angular momentum
        I_omega_1 = I_omega(1);   % [kg*km^2/s] Angular momentum
        I_omega_2 = I_omega(2);   % [kg*km^2/s] Angular momentum
        I_omega_3 = I_omega(3);   % [kg*km^2/s] Angular momentum
        
        I_omega_tilde = [0, -I_omega_3, I_omega_2;
            I_omega_3, 0, -I_omega_1;
            -I_omega_2, I_omega_1, 0]; % [kg*km^2/s] Angular momentum
        
        % -- F_omega
        
        F_omega = [0, -omega_1, -omega_2, -omega_3;
            omega_1, 0, omega_3, -omega_2;
            omega_2, -omega_3, 0, omega_1;
            omega_3, omega_2, -omega_1, 0];
        
        % -- B_q
        
        B_q = [-q_1, -q_2, -q_3;
            q_0, -q_3, q_2;
            q_3, q_0, -q_1;
            -q_2, q_1, q_0];
        
        % -- External Torque
        
        ExternalTorque = zeros(3,1); % [kg*km^2/s^2]
        dL_domega      = zeros(3,3); % [kg*km^2/s]
        dL_dI_ij       = zeros(3,6); % [1/s^2]
        
        % -- q_dot
        
        q_dot     = 0.5*F_omega*q; % [n.d.] acceleration of quaternion
        
        % -- omega_dot
        
        omega_dot = I_inv*(- omega_tilde*I*omega_vec + ExternalTorque); % [rad./s^2] angular acceleration
        
        % -- Rotation matrix
        
        RotQ      = GetRotFromQuaternion(q);
        
        BN = RotQ.BN; % [n.d.] Inertial frame to the body frame
        NB = RotQ.NB; % [n.d.] Body frame to the inertial frame
        
        if ( init.options.estimate.attitude == 1 ) || ( init.options.consider.attitude == 1 )
            
            dT_dq_0  =  RotQ.dNB_dq_0;
            dT_dq_1  =  RotQ.dNB_dq_1;
            dT_dq_2  =  RotQ.dNB_dq_2;
            dT_dq_3  =  RotQ.dNB_dq_3;
            
        end % For if
        
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Spherical Harmonics -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    if init.options.harmonics_normalization == 0 % Unnormalized
        
        C_input = init.C(1:init.deg.acce+1,1:init.deg.acce+1);    % [n.d.] C spherical harmonics
        S_input = init.S(1:init.deg.acce+1,1:init.deg.acce+1);    % [n.d.] S spherical harmonics
        
    elseif init.options.harmonics_normalization == 1 % Normalized
        
        C_input = init.Cbar(1:init.deg.acce+1,1:init.deg.acce+1); % [n.d.] C spherical harmonics
        S_input = init.Sbar(1:init.deg.acce+1,1:init.deg.acce+1); % [n.d.] S spherical harmonics
        
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Asteroid/Earth Dynamics -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % -- Asteroid's position and velocity
    
    x_Ast_SCI      = state(init.num.deri + 1);
    y_Ast_SCI      = state(init.num.deri + 2);
    z_Ast_SCI      = state(init.num.deri + 3);
    xdot_Ast_SCI   = state(init.num.deri + 4);
    ydot_Ast_SCI   = state(init.num.deri + 5);
    zdot_Ast_SCI   = state(init.num.deri + 6);
    
    d_pos          = [x_Ast_SCI; y_Ast_SCI; z_Ast_SCI];  % [km] Asteroid's position in the SCI frame
    
    if init.options.propagate_asteroid == 1
       
        % -- Asteroid's velocity and acceleration
        
        Pos_Ast_SCI    = d_pos;
        Vel_Ast_SCI    = [xdot_Ast_SCI; ydot_Ast_SCI; zdot_Ast_SCI];
        Acce_Ast_SCI   = - init.mu_Sun/norm(Pos_Ast_SCI)^3*Pos_Ast_SCI;
        VelAcceAst_SCI = [Vel_Ast_SCI; Acce_Ast_SCI];
        
        partial_ast_acce_partial_ast_pos = init.mu_Sun*( -1/norm(d_pos)^3*eye(3) + 3/norm(d_pos)^5*(d_pos*d_pos') );
        
    else
        
        VelAcceAst_SCI                   = zeros(6,1); % Asteroid dynamics are not propagated around the Sun
        partial_ast_acce_partial_ast_pos = zeros(3,3);
        
    end % For if
    
    % -- Earth's position and velocity
    
    x_Earth_SCI    = state(init.num.deri + 7);
    y_Earth_SCI    = state(init.num.deri + 8);
    z_Earth_SCI    = state(init.num.deri + 9);
    xdot_Earth_SCI = state(init.num.deri + 10);
    ydot_Earth_SCI = state(init.num.deri + 11);
    zdot_Earth_SCI = state(init.num.deri + 12);
        
    if init.options.propagate_Earth == 1
        
        % -- Earth's velocity and acceleration
        
        Pos_Earth_SCI    =  [x_Earth_SCI; y_Earth_SCI; z_Earth_SCI];
        Vel_Earth_SCI    =  [xdot_Earth_SCI; ydot_Earth_SCI; zdot_Earth_SCI];
        Acce_Earth_SCI   =  - init.mu_Sun/norm(Pos_Earth_SCI)^3*Pos_Earth_SCI;
        
        VelAcceEarth_SCI = [Vel_Earth_SCI; Acce_Earth_SCI];
        
    else
        
        VelAcceEarth_SCI = zeros(6,1); % Earth's dynamics are not propagated around the Sun
        
    end % For if
    
    VelAcceAstEarth_SCI  = [VelAcceAst_SCI; VelAcceEarth_SCI];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Satellite Position/Velocity -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Pos_Sat_Inertial  =  [x_sat; y_sat; z_sat];     % [km] Position of the Satellite in the inertial frame
    Pos_Sat_Body      =  BN * Pos_Sat_Inertial;     % [km] Position of the Satellite in the body frame
        
    Vel_Sat_Inertial  =  [xdot_sat; ydot_sat; zdot_sat];  % Velocity of the Satellite in the inertial frame
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Asteroid Position/Velocity -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Pos_Ast_Inertial  =  [x_ast; y_ast; z_ast];     % Position of the asteroid in the inertial frame
    Pos_Ast_Body      =  BN * Pos_Ast_Inertial;     % Position of the Asteroid in the body frame
        
    Vel_Ast_Inertial  =  [xdot_ast; ydot_ast; zdot_ast];  % Velocity of the Asteroid in the inertial frame
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Relative Position -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Pos_Sat_Rel_Ast_Inertial  =  Pos_Sat_Inertial - Pos_Ast_Inertial; % Relative Position of Satellite and Asteroid in the inertial frame (Ast --> Sat)
    Pos_Sat_Rel_Ast_Body      =  Pos_Sat_Body - Pos_Ast_Body;         % Relative Position of Satellite and Asteroid in the body frame (Ast --> Sat)
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Acceleration == %%
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    % -- Gravity field
    
    [~, Acce_dUdq_body, A_Acce_Pos, A_Acce_C, A_Acce_S, B_Acce_C, B_Acce_S] = ComputePotentialAcceSTMConsider_mex(init.deg.acce, init.deg.estimate, init.deg.consider, init.R_ref, init.M_ref, Pos_Sat_Rel_Ast_Body, C_input, S_input, init.options.exterior_interior, init.options.harmonics_normalization);
    Acce_dUdq_inertial = NB * Acce_dUdq_body;  % Acceleration in the inertial frame
    
    % -- SRP
    
    S_vec    = d_pos + Pos_Sat_Rel_Ast_Inertial;
    Acce_SRP = SRP_scale*init.SRP.Constant * S_vec/norm(S_vec)^3; % Acceleration by solar radiation pressure = constant
    
    % -- Tidal effect
    
    if init.options.tidal_effect == 1
        
        Acce_tidal = init.mu_Sun * (-1/norm(S_vec)^3*S_vec + 1/norm(d_pos)^3*d_pos);
        
    else
        
        Acce_tidal = zeros(3,1);
        
    end % For if
    
    % -- Total acceleration
    
    Acce_sat  =  Acce_dUdq_inertial + Acce_SRP + Acce_tidal; % Acceleration of the Satellite
    Acce_ast  =  [0; 0; 0];                                  % Acceleration of the Asteroid
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%
    %% -- Compute A -- %%
    %%%%%%%%%%%%%%%%%%%%%
    
    A   =  zeros(init.num.estimate.all,init.num.estimate.all);
    
    %%%%%%%%%%%%%%%%%%%%
    %% -- Velocity -- %%
    %%%%%%%%%%%%%%%%%%%%
    
    A(1:3, 4:6)   = eye(3);
    A(7:9, 10:12) = eye(3);
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Tidal effect -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    if init.options.tidal_effect == 1
        
        partial_tidal_partial_sat_pos = init.mu_Sun * ( -1/norm(S_vec)^3*eye(3) + 3/norm(S_vec)^5*(S_vec*S_vec') );
        
    else
        
        partial_tidal_partial_sat_pos = zeros(3,3);
        
    end % For if
    
    %%%%%%%%%%%%%%
    %% -- Mu -- %%
    %%%%%%%%%%%%%%
    
    if init.options.estimate.mu == 1
        
        A(4:6,init.index.estimate.mu)  = Acce_dUdq_inertial/mu;
        
    end % For if
    
    %%%%%%%%%%%%%%%
    %% -- SRP -- %%
    %%%%%%%%%%%%%%%
    
    if init.options.estimate.SRP == 1
        
        A(4:6,init.index.estimate.SRP) = Acce_SRP/SRP_scale;
        
    end % For if
        
    partial_SRP_partial_sat_pos = SRP_scale*init.SRP.Constant * ( 1/norm(S_vec)^3*eye(3) - 3/norm(S_vec)^5*(S_vec*S_vec') );
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Spherical Harmonics -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if init.deg.estimate > 0
    
        A(4:6,init.index.estimate.C_first:init.index.estimate.S_end) =  NB * [A_Acce_C, A_Acce_S];
    
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Body frame to Inertial frame & Summing it all together -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    A(4:6,1:3)   = NB*A_Acce_Pos*NB' + partial_tidal_partial_sat_pos + partial_SRP_partial_sat_pos;
    A(4:6,7:9)   = - NB*A_Acce_Pos*NB' - partial_ast_acce_partial_ast_pos;
    A(10:12,7:9) = partial_ast_acce_partial_ast_pos;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Attitude State -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if init.options.estimate.attitude == 1
        
        if init.options.attitude_dynamics == 1     % 3-1-3 Euler angles
            
            % d(a_inertial_sat)/d(spin state)
            
            A(4:6,init.index.estimate.attitude_first + 0)  =  dT_dalpha     * Acce_dUdq_body;
            A(4:6,init.index.estimate.attitude_first + 1)  =  dT_dbeta      * Acce_dUdq_body;
            A(4:6,init.index.estimate.attitude_first + 2)  =  dT_dgamma_0   * Acce_dUdq_body;
            A(4:6,init.index.estimate.attitude_first + 3)  =  dT_dgamma_dot * Acce_dUdq_body;
            
        elseif init.options.attitude_dynamics == 2 % Quaternion
            
            % d(a_inertial_sat)/d(q)
            
            A(4:6,init.index.estimate.attitude_first + 0)  =  dT_dq_0 * Acce_dUdq_body;
            A(4:6,init.index.estimate.attitude_first + 1)  =  dT_dq_1 * Acce_dUdq_body;
            A(4:6,init.index.estimate.attitude_first + 2)  =  dT_dq_2 * Acce_dUdq_body;
            A(4:6,init.index.estimate.attitude_first + 3)  =  dT_dq_3 * Acce_dUdq_body;
            
            % d(q_dot)/d(q)
            
            A(init.index.estimate.attitude_first:init.index.estimate.attitude_end,init.index.estimate.attitude_first:init.index.estimate.attitude_end) = ...
                0.5*F_omega;
            
        end % For if
        
    end % For if
    
    % d(omega_dot)/d(I_ij)
    
    if init.options.compute_d_omega_dot_d_I_ij == 1
        
        d_omega_dot_d_I_ij_all = zeros(3,6);
        
        for ii = 1:3
            
            for jj = 1:ii
                
                I_index = ii*(ii-1)/2 + jj;
                
                dI_dI_ij        = zeros(3,3);
                dI_dI_ij(ii,jj) = 1;
                dI_dI_ij(jj,ii) = 1;
                                
                d_omega_dot_d_I_ij_all(:,I_index) = - I_inv*dI_dI_ij*omega_dot + I_inv*(- omega_tilde*dI_dI_ij*omega_vec + dL_dI_ij(:,I_index));
                                                
            end % For jj
            
        end % For ii
        
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Angular velocity -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if init.options.estimate.angular_velocity == 1
        
        % d(omega_dot)/d(mu)
        
        if init.options.estimate.mu == 1
            
            d_omega_dot_d_mu = -ExternalTorque;
            
            for ii = 1:3
                
                for jj = 1:ii
                    
                    I_index = ii*(ii-1)/2 + jj;
                    
                    d_omega_dot_d_mu = d_omega_dot_d_mu + dL_dI_ij(:,I_index)*I(ii,jj);
                    
                end % For jj
                
            end % For ii
            
            d_omega_dot_d_mu = (1/mu)*I_inv*d_omega_dot_d_mu;
            A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.mu) = d_omega_dot_d_mu;
            
        end % For if
        
        % d(omega_dot)/d(omega)
        
        A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end) = I_inv*(I_omega_tilde - omega_tilde*I + dL_domega);
        
        % d(q_dot)/d(omega)
        
        if (init.options.estimate.attitude == 1) && (init.options.attitude_dynamics == 2)
            
            A(init.index.estimate.attitude_first:init.index.estimate.attitude_end,init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end) = 0.5*B_q;
            
        end % For if
        
        % d(omega_dot)/d(C,S)
        
        if ( init.options.link_inertia_second_degree == 1 ) && ( init.deg.estimate >= 2 )
            
            if init.options.harmonics_normalization == 0 % Unnormalized
            
                C_10 = C_input(2,1); C_11 = C_input(2,2); S_11 = S_input(2,2);
            
            elseif init.options.harmonics_normalization == 1 % Normalized
                
                Cbar_10 = C_input(2,1); Cbar_11 = C_input(2,2); Sbar_11 = S_input(2,2);
                
            end % For if
            
            N_10 = init.N(2,1); N_11 = init.N(2,2);
            N_20 = init.N(3,1); N_21 = init.N(3,2); N_22 = init.N(3,3);
            
            if ( init.options.normalize_inertia == 0 ) && ( init.options.harmonics_normalization == 0 )
                
                % d(omega_dot)/d(C_10)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(1))     = init.M_ref*init.R_ref^2*(-2*C_10*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3)) + C_11*d_omega_dot_d_I_ij_all(:,4) + S_11*d_omega_dot_d_I_ij_all(:,5));
                
                % d(omega_dot)/d(C_11)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(1) + 1) = init.M_ref*init.R_ref^2*(-2*C_11*(d_omega_dot_d_I_ij_all(:,3) + d_omega_dot_d_I_ij_all(:,6)) + S_11*d_omega_dot_d_I_ij_all(:,2) + C_10*d_omega_dot_d_I_ij_all(:,4));
                
                % d(omega_dot)/d(S_11)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.S_eachdegree_first(1))     = init.M_ref*init.R_ref^2*(-2*S_11*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,6)) + C_11*d_omega_dot_d_I_ij_all(:,2) + C_10*d_omega_dot_d_I_ij_all(:,5));
                
                % d(omega_dot)/d(C_20)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(2))     = init.M_ref*init.R_ref^2/3*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) - 2*d_omega_dot_d_I_ij_all(:,6));
                
                % d(omega_dot)/d(C_21)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(2) + 1) = - init.M_ref*init.R_ref^2*d_omega_dot_d_I_ij_all(:,4);
                
                % d(omega_dot)/d(C_22)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(2) + 2) = 2*init.M_ref*init.R_ref^2*( - d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) );
                
                % d(omega_dot)/d(S_21)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.S_eachdegree_first(2))     = - init.M_ref*init.R_ref^2*d_omega_dot_d_I_ij_all(:,5);
                
                % d(omega_dot)/d(S_22)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.S_eachdegree_first(2) + 1) = - 2*init.M_ref*init.R_ref^2*d_omega_dot_d_I_ij_all(:,2);
                                
            elseif ( init.options.normalize_inertia == 1 ) && ( init.options.harmonics_normalization == 0 )
                 
                % d(omega_dot)/d(C_10)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(1))     = -2*C_10*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3)) + C_11*d_omega_dot_d_I_ij_all(:,4) + S_11*d_omega_dot_d_I_ij_all(:,5);
                
                % d(omega_dot)/d(C_11)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(1) + 1) = -2*C_11*(d_omega_dot_d_I_ij_all(:,3) + d_omega_dot_d_I_ij_all(:,6)) + S_11*d_omega_dot_d_I_ij_all(:,2) + C_10*d_omega_dot_d_I_ij_all(:,4);

                % d(omega_dot)/d(S_11)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.S_eachdegree_first(1))     = -2*S_11*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,6)) + C_11*d_omega_dot_d_I_ij_all(:,2) + C_10*d_omega_dot_d_I_ij_all(:,5);
                
                % d(omega_dot)/d(C_20)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(2))     = 1/3*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) - 2*d_omega_dot_d_I_ij_all(:,6));
                
                % d(omega_dot)/d(C_21)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(2) + 1) = - d_omega_dot_d_I_ij_all(:,4);
                
                % d(omega_dot)/d(C_22)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(2) + 2) = 2*( - d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) );
                
                % d(omega_dot)/d(S_21)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.S_eachdegree_first(2))     = - d_omega_dot_d_I_ij_all(:,5);
                
                % d(omega_dot)/d(S_22)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.S_eachdegree_first(2) + 1) = - 2*d_omega_dot_d_I_ij_all(:,2);
                
            elseif ( init.options.normalize_inertia == 0 ) && ( init.options.harmonics_normalization == 1 )
              
                % d(omega_dot)/d(C_10)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(1))     = init.M_ref*init.R_ref^2*(-2*N_10^2*Cbar_10*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3)) + N_10*N_11*( Cbar_11*d_omega_dot_d_I_ij_all(:,4) + Sbar_11*d_omega_dot_d_I_ij_all(:,5) ));
                
                % d(omega_dot)/d(C_11)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(1) + 1) = init.M_ref*init.R_ref^2*(-2*N_11^2*Cbar_11*(d_omega_dot_d_I_ij_all(:,3) + d_omega_dot_d_I_ij_all(:,6)) + N_11*(N_11*Sbar_11*d_omega_dot_d_I_ij_all(:,2) + N_10*Cbar_10*d_omega_dot_d_I_ij_all(:,4)) );
                
                % d(omega_dot)/d(S_11)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.S_eachdegree_first(1))     = init.M_ref*init.R_ref^2*(-2*N_11^2*Sbar_11*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,6)) + N_11*(N_11*Cbar_11*d_omega_dot_d_I_ij_all(:,2) + N_10*Cbar_10*d_omega_dot_d_I_ij_all(:,5)) );
                
                % d(omega_dot)/d(C_20)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(2))     = init.M_ref*init.R_ref^2*N_20/3*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) - 2*d_omega_dot_d_I_ij_all(:,6));
                
                % d(omega_dot)/d(C_21)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(2) + 1) = - init.M_ref*init.R_ref^2*N_21*d_omega_dot_d_I_ij_all(:,4);
                
                % d(omega_dot)/d(C_22)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(2) + 2) = 2*init.M_ref*init.R_ref^2*N_22*( - d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) );
                
                % d(omega_dot)/d(S_21)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.S_eachdegree_first(2))     = - init.M_ref*init.R_ref^2*N_21*d_omega_dot_d_I_ij_all(:,5);
                
                % d(omega_dot)/d(S_22)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.S_eachdegree_first(2) + 1) = - 2*init.M_ref*init.R_ref^2*N_22*d_omega_dot_d_I_ij_all(:,2);
                
            elseif ( init.options.normalize_inertia == 1 ) && ( init.options.harmonics_normalization == 1 )
                
                % d(omega_dot)/d(C_10)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(1))     = -2*N_10^2*Cbar_10*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3)) + N_10*N_11*( Cbar_11*d_omega_dot_d_I_ij_all(:,4) + Sbar_11*d_omega_dot_d_I_ij_all(:,5) );
                
                % d(omega_dot)/d(C_11)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(1) + 1) = -2*N_11^2*Cbar_11*(d_omega_dot_d_I_ij_all(:,3) + d_omega_dot_d_I_ij_all(:,6)) + N_11*( N_11*Sbar_11*d_omega_dot_d_I_ij_all(:,2) + N_10*Cbar_10*d_omega_dot_d_I_ij_all(:,4) );
                
                % d(omega_dot)/d(S_11)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.S_eachdegree_first(1))     = -2*N_11^2*Sbar_11*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,6)) + N_11*( N_11*Cbar_11*d_omega_dot_d_I_ij_all(:,2) + N_10*Cbar_10*d_omega_dot_d_I_ij_all(:,5) );
                
                % d(omega_dot)/d(C_20)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(2))     = N_20/3*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) - 2*d_omega_dot_d_I_ij_all(:,6));
                
                % d(omega_dot)/d(C_21)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(2) + 1) = - N_21*d_omega_dot_d_I_ij_all(:,4);
                
                % d(omega_dot)/d(C_22)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.C_eachdegree_first(2) + 2) = 2*N_22*( - d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) );
                
                % d(omega_dot)/d(S_21)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.S_eachdegree_first(2))     = - N_21*d_omega_dot_d_I_ij_all(:,5);
                
                % d(omega_dot)/d(S_22)
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.S_eachdegree_first(2) + 1) = - 2*N_22*d_omega_dot_d_I_ij_all(:,2);
                
            end % For if
            
        end % For if
        
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Inertia Tensor -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % d(omega_dot)/d(\bar{I}_T)
    % or 
    % [ d(omega_dot)/d(I_ij) ]
    
    if (init.options.estimate.inertia_tensor == 1) && (init.options.estimate.angular_velocity == 1)
        
        if init.options.link_inertia_second_degree == 1
            
            % d(omega_dot)/d(\bar{I}_T)
            
            if init.options.normalize_inertia == 1
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.inertia_tensor_first) = (1/3)*I_inv*(- omega_dot + dL_dI_ij(:,1) + dL_dI_ij(:,3) + dL_dI_ij(:,6));
                
            elseif init.options.normalize_inertia == 0
                
                A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.inertia_tensor_first) = (1/3)*init.M_ref*init.R_ref^2*I_inv*(- omega_dot + dL_dI_ij(:,1) + dL_dI_ij(:,3) + dL_dI_ij(:,6));
                
            end % For if            
            
        elseif init.options.link_inertia_second_degree == 0
            
            % d(omega_dot)/d(I_ij)
            
            A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.inertia_tensor_first + 0) = d_omega_dot_d_I_ij_all(:,1);
            A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.inertia_tensor_first + 1) = d_omega_dot_d_I_ij_all(:,2);
            A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.inertia_tensor_first + 2) = d_omega_dot_d_I_ij_all(:,3);
            A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.inertia_tensor_first + 3) = d_omega_dot_d_I_ij_all(:,4);
            A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.inertia_tensor_first + 4) = d_omega_dot_d_I_ij_all(:,5);
            A(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.estimate.inertia_tensor_first + 5) = d_omega_dot_d_I_ij_all(:,6);
            
        end % For if
        
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Compute Phi_dot -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Phi     = state(init.num.state.all + 1:init.num.state.all + init.num.estimate.all^2,1);
    Phi     = reshape(Phi, init.num.estimate.all, init.num.estimate.all);
    
    Phi_dot = A*Phi;
    
    Phi_dot = reshape(Phi_dot, init.num.estimate.all*init.num.estimate.all, 1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Compute Theta_dot -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    THETA    =  state(init.num.state.all + init.num.estimate.all^2 + 1:init.num.state.all + init.num.estimate.all^2 + init.num.estimate.all*init.num.consider.all,1);
    THETA    =  reshape(THETA, init.num.estimate.all, init.num.consider.all);
    
    B        =  zeros(init.num.estimate.all,init.num.consider.all);
    
    %%%%%%%%%%%%%%
    %% -- Mu -- %%
    %%%%%%%%%%%%%%
    
    if init.options.consider.mu == 1
        
        % d(acce)/d(mu)
        
        B(4:6,init.index.consd.mu) = Acce_dUdq_inertial/mu;
        
        % d(omega_dot)/d(mu)
        
        if init.options.estimate.angular_velocity == 1
                        
            d_omega_dot_d_mu = -ExternalTorque;
            
            for ii = 1:3
                
                for jj = 1:ii
                    
                    I_index = ii*(ii-1)/2 + jj;
                    
                    d_omega_dot_d_mu = d_omega_dot_d_mu + dL_dI_ij(:,I_index)*I(ii,jj);
                    
                end % For jj
                
            end % For ii
            
            d_omega_dot_d_mu = (1/mu)*I_inv*d_omega_dot_d_mu;
            B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.mu) = d_omega_dot_d_mu;
            
        end % For if
        
    end % For if
            
    %%%%%%%%%%%%%%%
    %% -- SRP -- %%
    %%%%%%%%%%%%%%%
        
    if init.options.consider.SRP == 1
        
        B(4:6,init.index.consd.SRP) = Acce_SRP/SRP_scale;
        
    end % For if
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Attitude State -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if init.options.consider.attitude == 1
        
        if init.options.attitude_dynamics == 1
            
            % d(a_inertial_sat)/d(Euler angles + rotation rate)
            
            B(4:6,init.index.consd.attitude_first + 0)  =  dT_dalpha     * Acce_dUdq_body;  % alpha
            B(4:6,init.index.consd.attitude_first + 1)  =  dT_dbeta      * Acce_dUdq_body;  % beta
            B(4:6,init.index.consd.attitude_first + 2)  =  dT_dgamma_0   * Acce_dUdq_body;  % gamma
            B(4:6,init.index.consd.attitude_first + 3)  =  dT_dgamma_dot * Acce_dUdq_body;  % gamma_dot
            
        elseif init.options.attitude_dynamics == 2
            
            % d(a_inertial_sat)/d(q)
            
            B(4:6,init.index.consd.attitude_first + 0)  =  dT_dq_0 * Acce_dUdq_body;  % q_0
            B(4:6,init.index.consd.attitude_first + 1)  =  dT_dq_1 * Acce_dUdq_body;  % q_1
            B(4:6,init.index.consd.attitude_first + 2)  =  dT_dq_2 * Acce_dUdq_body;  % q_2
            B(4:6,init.index.consd.attitude_first + 3)  =  dT_dq_3 * Acce_dUdq_body;  % q_3
            
        end % For if
        
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Angular velocity -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if init.options.consider.angular_velocity == 1
        
        if init.options.estimate.attitude == 1
            
            % d(q_dot)/d(omega)
            
            B(init.index.estimate.attitude_first:init.index.estimate.attitude_end,init.index.consd.omega_first:init.index.consd.omega_end) = 0.5*B_q;
            
        end % For if
        
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Inertia Tensor -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if  (init.options.estimate.angular_velocity == 1) && (init.options.consider.inertia_tensor == 1)
        
        if init.options.link_inertia_second_degree == 1
            
            if init.options.normalize_inertia == 1
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.inertia_tensor_first) = (1/3)*I_inv*(- omega_dot + dL_dI_ij(:,1) + dL_dI_ij(:,3) + dL_dI_ij(:,6));
                
            elseif init.options.normalize_inertia == 0
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.inertia_tensor_first) = (1/3)*init.M_ref*init.R_ref^2*I_inv*(- omega_dot + dL_dI_ij(:,1) + dL_dI_ij(:,3) + dL_dI_ij(:,6));
                
            end % For if     
        
        elseif init.options.link_inertia_second_degree == 0
            
            % d(omega_dot)/d(I_ij)
            
            B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.inertia_first + 0) = d_omega_dot_d_I_ij_all(:,1);
            B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.inertia_first + 1) = d_omega_dot_d_I_ij_all(:,2);
            B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.inertia_first + 2) = d_omega_dot_d_I_ij_all(:,3);
            B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.inertia_first + 3) = d_omega_dot_d_I_ij_all(:,4);
            B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.inertia_first + 4) = d_omega_dot_d_I_ij_all(:,5);
            B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.inertia_first + 5) = d_omega_dot_d_I_ij_all(:,6);
            
        end % For if
        
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Spherical Harmonics -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if init.deg.estimate < init.deg.consider
        
        % d(a_sat_inertial)/d(C,S)
        
        B(4:6,init.index.consd.C_first:init.index.consd.S_end) = NB * [B_Acce_C, B_Acce_S];
        
        % d(omega_dot)/d(C,S) - 1st and 2nd degree
        
        if (init.options.estimate.angular_velocity == 1) && ( init.options.link_inertia_second_degree == 1 ) && ( init.deg.estimate < 1 ) && ( init.deg.consider >= 2 )
            
            % Assign spherical harmonics
            
            if init.options.harmonics_normalization == 0 % Unnormalized
                
                C_10 = C_input(2,1); C_11 = C_input(2,2); S_11 = S_input(2,2);
                
            elseif init.options.harmonics_normalization == 1 % Normalized
                
                Cbar_10 = C_input(2,1); Cbar_11 = C_input(2,2); Sbar_11 = S_input(2,2);
                
            end % For if
            
            N_10 = init.N(2,1); N_11 = init.N(2,2);
            N_20 = init.N(3,1); N_21 = init.N(3,2); N_22 = init.N(3,3);
            
            if ( init.options.normalize_inertia == 0 ) && ( init.options.harmonics_normalization == 0 )
                
                % d(omega_dot)/d(C_10)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(1))     = init.M_ref*init.R_ref^2*(-2*C_10*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3)) + C_11*d_omega_dot_d_I_ij_all(:,4) + S_11*d_omega_dot_d_I_ij_all(:,5));
                
                % d(omega_dot)/d(C_11)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(1) + 1) = init.M_ref*init.R_ref^2*(-2*C_11*(d_omega_dot_d_I_ij_all(:,3) + d_omega_dot_d_I_ij_all(:,6)) + S_11*d_omega_dot_d_I_ij_all(:,2) + C_10*d_omega_dot_d_I_ij_all(:,4));
                
                % d(omega_dot)/d(S_11)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.S_eachdegree_first(1))     = init.M_ref*init.R_ref^2*(-2*S_11*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,6)) + C_11*d_omega_dot_d_I_ij_all(:,2) + C_10*d_omega_dot_d_I_ij_all(:,5));
                
                % d(omega_dot)/d(C_20)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(2))     = init.M_ref*init.R_ref^2/3*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) - 2*d_omega_dot_d_I_ij_all(:,6));
                
                % d(omega_dot)/d(C_21)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(2) + 1) = - init.M_ref*init.R_ref^2*d_omega_dot_d_I_ij_all(:,4);
                
                % d(omega_dot)/d(C_22)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(2) + 2) = 2*init.M_ref*init.R_ref^2*( - d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) );
                
                % d(omega_dot)/d(S_21)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.S_eachdegree_first(2))     = - init.M_ref*init.R_ref^2*d_omega_dot_d_I_ij_all(:,5);
                
                % d(omega_dot)/d(S_22)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.S_eachdegree_first(2) + 1) = - 2*init.M_ref*init.R_ref^2*d_omega_dot_d_I_ij_all(:,2);
                
            elseif ( init.options.normalize_inertia == 1 ) && ( init.options.harmonics_normalization == 0 )
                
                % d(omega_dot)/d(C_10)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(1))     = -2*C_10*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3)) + C_11*d_omega_dot_d_I_ij_all(:,4) + S_11*d_omega_dot_d_I_ij_all(:,5);
                
                % d(omega_dot)/d(C_11)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(1) + 1) = -2*C_11*(d_omega_dot_d_I_ij_all(:,3) + d_omega_dot_d_I_ij_all(:,6)) + S_11*d_omega_dot_d_I_ij_all(:,2) + C_10*d_omega_dot_d_I_ij_all(:,4);
                
                % d(omega_dot)/d(S_11)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.S_eachdegree_first(1))     = -2*S_11*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,6)) + C_11*d_omega_dot_d_I_ij_all(:,2) + C_10*d_omega_dot_d_I_ij_all(:,5);
                
                % d(omega_dot)/d(C_20)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(2))     = 1/3*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) - 2*d_omega_dot_d_I_ij_all(:,6));
                
                % d(omega_dot)/d(C_21)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(2) + 1) = - d_omega_dot_d_I_ij_all(:,4);
                
                % d(omega_dot)/d(C_22)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(2) + 2) = 2*( - d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) );
                
                % d(omega_dot)/d(S_21)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.S_eachdegree_first(2))     = - d_omega_dot_d_I_ij_all(:,5);
                
                % d(omega_dot)/d(S_22)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.S_eachdegree_first(2) + 1) = - 2*d_omega_dot_d_I_ij_all(:,2);
                
                
            elseif ( init.options.normalize_inertia == 0 ) && ( init.options.harmonics_normalization == 1 )
                
                % d(omega_dot)/d(C_10)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(1))     = init.M_ref*init.R_ref^2*(-2*N_10^2*Cbar_10*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3)) + N_10*N_11*( Cbar_11*d_omega_dot_d_I_ij_all(:,4) + Sbar_11*d_omega_dot_d_I_ij_all(:,5) ));
                
                % d(omega_dot)/d(C_11)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(1) + 1) = init.M_ref*init.R_ref^2*(-2*N_11^2*Cbar_11*(d_omega_dot_d_I_ij_all(:,3) + d_omega_dot_d_I_ij_all(:,6)) + N_11*(N_11*Sbar_11*d_omega_dot_d_I_ij_all(:,2) + N_10*Cbar_10*d_omega_dot_d_I_ij_all(:,4)) );
                
                % d(omega_dot)/d(S_11)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.S_eachdegree_first(1))     = init.M_ref*init.R_ref^2*(-2*N_11^2*Sbar_11*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,6)) + N_11*(N_11*Cbar_11*d_omega_dot_d_I_ij_all(:,2) + N_10*Cbar_10*d_omega_dot_d_I_ij_all(:,5)) );
                
                % d(omega_dot)/d(C_20)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(2))     = init.M_ref*init.R_ref^2*N_20/3*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) - 2*d_omega_dot_d_I_ij_all(:,6));
                
                % d(omega_dot)/d(C_21)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(2) + 1) = - init.M_ref*init.R_ref^2*N_21*d_omega_dot_d_I_ij_all(:,4);
                
                % d(omega_dot)/d(C_22)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(2) + 2) = 2*init.M_ref*init.R_ref^2*N_22*( - d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) );
                
                % d(omega_dot)/d(S_21)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.S_eachdegree_first(2))     = - init.M_ref*init.R_ref^2*N_21*d_omega_dot_d_I_ij_all(:,5);
                
                % d(omega_dot)/d(S_22)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.S_eachdegree_first(2) + 1) = - 2*init.M_ref*init.R_ref^2*N_22*d_omega_dot_d_I_ij_all(:,2);
                
            elseif ( init.options.normalize_inertia == 1 ) && ( init.options.harmonics_normalization == 1 )
                
                % d(omega_dot)/d(C_10)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(1))     = -2*N_10^2*Cbar_10*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3)) + N_10*N_11*( Cbar_11*d_omega_dot_d_I_ij_all(:,4) + Sbar_11*d_omega_dot_d_I_ij_all(:,5) );
                
                % d(omega_dot)/d(C_11)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(1) + 1) = -2*N_11^2*Cbar_11*(d_omega_dot_d_I_ij_all(:,3) + d_omega_dot_d_I_ij_all(:,6)) + N_11*( N_11*Sbar_11*d_omega_dot_d_I_ij_all(:,2) + N_10*Cbar_10*d_omega_dot_d_I_ij_all(:,4) );
                
                % d(omega_dot)/d(S_11)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.S_eachdegree_first(1))     = -2*N_11^2*Sbar_11*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,6)) + N_11*( N_11*Cbar_11*d_omega_dot_d_I_ij_all(:,2) + N_10*Cbar_10*d_omega_dot_d_I_ij_all(:,5) );
                
                % d(omega_dot)/d(C_20)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(2))     = N_20/3*(d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) - 2*d_omega_dot_d_I_ij_all(:,6));
                
                % d(omega_dot)/d(C_21)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(2) + 1) = - N_21*d_omega_dot_d_I_ij_all(:,4);
                
                % d(omega_dot)/d(C_22)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.C_eachdegree_first(2) + 2) = 2*N_22*( - d_omega_dot_d_I_ij_all(:,1) + d_omega_dot_d_I_ij_all(:,3) );
                
                % d(omega_dot)/d(S_21)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.S_eachdegree_first(2))     = - N_21*d_omega_dot_d_I_ij_all(:,5);
                
                % d(omega_dot)/d(S_22)
                
                B(init.index.estimate.angular_velocity_first:init.index.estimate.angular_velocity_end,init.index.consd.S_eachdegree_first(2) + 1) = - 2*N_22*d_omega_dot_d_I_ij_all(:,2);
                
            end % For if
            
        end % For if
        
    end % For if
    
    THETA_DOT = A * THETA + B;
    THETA_DOT = reshape(THETA_DOT, init.num.estimate.all*init.num.consider.all,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%
    %% -- Output -- %%
    %%%%%%%%%%%%%%%%%%
    
    if init.options.attitude_dynamics == 1 % 3-1-3 Euler angles
        
        deri  =  [Vel_Sat_Inertial;
                  Acce_sat;
                  Vel_Ast_Inertial;
                  Acce_ast;
                  zeros(init.num.state.mu,1);
                  zeros(init.num.state.SRP,1);
                  zeros(init.num.state.attitude,1);
                  zeros(init.num.state.CS,1);
                  zeros(init.num.estimate.maneuver_total,1);
                  Phi_dot;
                  THETA_DOT;
                  VelAcceAstEarth_SCI];
        
    elseif init.options.attitude_dynamics == 2   % Quaternion
        
        deri  =  [Vel_Sat_Inertial;
                  Acce_sat;
                  Vel_Ast_Inertial;
                  Acce_ast;
                  zeros(init.num.state.mu,1);
                  zeros(init.num.state.SRP,1);
                  q_dot;
                  omega_dot;
                  zeros(init.num.state.inertia,1);
                  zeros(init.num.state.CS,1);
                  zeros(init.num.estimate.maneuver_total,1);
                  Phi_dot;
                  THETA_DOT;
                  VelAcceAstEarth_SCI];
        
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
elseif init.options.ode_mex == 1 % Mex function
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Degree of Gravity Field -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Deg     =  [init.deg.acce;
                init.deg.estimate;
                init.deg.consider];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Number of Elements -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Num     =  [init.num.state.all;
                init.num.estimate.all;
                init.num.consider.all];
    
    %% -- Integrator Options
    
    Options =  [init.options.exterior_interior;
                init.options.harmonics_normalization;
                init.options.estimate.mu;
                init.options.consider.mu;
                init.options.estimate.SRP;
                init.options.consider.SRP;
                init.options.attitude_dynamics;
                init.options.estimate.attitude;
                init.options.consider.attitude;
                init.options.estimate.angular_velocity;
                init.options.consider.angular_velocity;
                init.options.estimate.inertia_tensor;
                init.options.consider.inertia_tensor;
                init.options.normalize_inertia;
                init.options.tidal_effect;
                init.options.propagate_asteroid;
                init.options.propagate_Earth;
                init.options.link_inertia_second_degree;
                init.options.compute_d_omega_dot_d_I_ij];
    
    %% -- Indices
    
    Index = [init.index.state.mu;
             init.index.state.SRP;
             init.index.state.attitude_first;
             init.index.state.angular_velocity_first;
             init.index.estimate.mu;
             init.index.estimate.SRP;
             init.index.estimate.attitude_first;
             init.index.estimate.angular_velocity_first;
             init.index.estimate.inertia_tensor_first;
             init.index.estimate.C_first;
             init.index.estimate.C_eachdegree_first(1);
             init.index.estimate.S_eachdegree_first(1);
             init.index.estimate.C_eachdegree_first(2);
             init.index.estimate.S_eachdegree_first(2);
             init.index.consd.mu;
             init.index.consd.SRP;
             init.index.consd.attitude_first;
             init.index.consd.omega_first;
             init.index.consd.inertia_first;
             init.index.consd.C_first;
             init.index.consd.C_eachdegree_first(1);
             init.index.consd.S_eachdegree_first(1);
             init.index.consd.C_eachdegree_first(2);
             init.index.consd.S_eachdegree_first(2)];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
    %%%%%%%%%%%%%%%%%%%%%%%%
    %% - Regular state -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    Regular_state = state(1:init.num.state.regular);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Attitude State -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if init.options.attitude_dynamics == 1
        
        AttitudeState = [init.Ast.alpha_0; init.Ast.beta_0; init.Ast.gamma_0; init.Ast.ws];
        Omega         = zeros(3,1);
        
    elseif init.options.attitude_dynamics == 2
        
        % Quaternion
        
        q_0 = state(init.num.state.to_SRP + 1);
        q_1 = state(init.num.state.to_SRP + 2);
        q_2 = state(init.num.state.to_SRP + 3);
        q_3 = state(init.num.state.to_SRP + 4);
        
        q   = [q_0; q_1; q_2; q_3]; % [n.d.] Quaternion
        q   = q/norm(q);            % [n.d.] Normalization
        
        AttitudeState = q;
        
        % -- Angular velocity
        
        omega_1 = state(init.num.state.to_attitude + 1);
        omega_2 = state(init.num.state.to_attitude + 2);
        omega_3 = state(init.num.state.to_attitude + 3);
        
        Omega   = [omega_1; omega_2; omega_3]; % [rad/s] Angular velocity
        
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Inertia Tensor -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if init.options.normalize_inertia == 0
        
        I      =  init.Ast.I_tensor;     % [kg*km^2]     3 x 3 Inertia tensor
        I_inv  =  init.Ast.I_inv;        % [1/(kg*km^2)] 3 x 3 Inverse of the Inertia tensor
        
    elseif init.options.normalize_inertia == 1
        
        I      =  init.Ast.I_tensor_bar; % [n.d.] 3 x 3 Inertia tensor
        I_inv  =  init.Ast.I_inv_bar;    % [n.d.] 3 x 3 Inverse of the Inertia tensor
        
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% - Spherical Harmonics and Normalization -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    N_input = init.N(1:init.deg.acce+1,1:init.deg.acce+1); % [n.d.] Spherical harmonics normalization
    
    if init.options.harmonics_normalization == 0 % Unnormalized
        
        C_input = init.C(1:init.deg.acce+1,1:init.deg.acce+1);    % [n.d.] C spherical harmonics
        S_input = init.S(1:init.deg.acce+1,1:init.deg.acce+1);    % [n.d.] S spherical harmonics
        
    elseif init.options.harmonics_normalization == 1 % Normalized
        
        C_input = init.Cbar(1:init.deg.acce+1,1:init.deg.acce+1); % [n.d.] C spherical harmonics
        S_input = init.Sbar(1:init.deg.acce+1,1:init.deg.acce+1); % [n.d.] S spherical harmonics
        
    end % For if
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Asteroid/Earth Dynamics -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % -- Asteroid
    
    x_Ast_SCI        = state(init.num.deri + 1);
    y_Ast_SCI        = state(init.num.deri + 2);
    z_Ast_SCI        = state(init.num.deri + 3);
    xdot_Ast_SCI     = state(init.num.deri + 4);
    ydot_Ast_SCI     = state(init.num.deri + 5);
    zdot_Ast_SCI     = state(init.num.deri + 6);
    
    Pos_Ast_SCI      = [x_Ast_SCI; y_Ast_SCI; z_Ast_SCI];
    Vel_Ast_SCI      = [xdot_Ast_SCI; ydot_Ast_SCI; zdot_Ast_SCI];
    PosVel_Ast_SCI   = [Pos_Ast_SCI; Vel_Ast_SCI];
    
    % -- Earth
    
    x_Earth_SCI      = state(init.num.deri + 7);
    y_Earth_SCI      = state(init.num.deri + 8);
    z_Earth_SCI      = state(init.num.deri + 9);
    xdot_Earth_SCI   = state(init.num.deri + 10);
    ydot_Earth_SCI   = state(init.num.deri + 11);
    zdot_Earth_SCI   = state(init.num.deri + 12);

    Pos_Earth_SCI    = [x_Earth_SCI; y_Earth_SCI; z_Earth_SCI];
    Vel_Earth_SCI    = [xdot_Earth_SCI; ydot_Earth_SCI; zdot_Earth_SCI];
    PosVel_Earth_SCI = [Pos_Earth_SCI; Vel_Earth_SCI];
    
    %%%%%%%%%%%%%%
    %% - STM -- %%
    %%%%%%%%%%%%%%
    
    Phi     =  state(init.num.state.all + 1:init.num.state.all + init.num.estimate.all^2,1);
    Phi     =  reshape(Phi, init.num.estimate.all, init.num.estimate.all);
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Consider STM -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    THETA   =  state(init.num.state.all + init.num.estimate.all^2 + 1:init.num.state.all + init.num.estimate.all^2 + init.num.estimate.all*init.num.consider.all,1);
    THETA   =  reshape(THETA, init.num.estimate.all, init.num.consider.all);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% -- Compute Derivative -- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    deri = flyby_acce_mex(t,...
                          init.mu_Sun,...
                          init.R_ref, init.M_ref, init.Ast.mu,...
                          init.SRP.scale, init.SRP.Constant,...
                          Regular_state,...
                          AttitudeState, Omega, I, I_inv,...
                          N_input, C_input, S_input,...
                          Phi, THETA,...
                          PosVel_Ast_SCI, PosVel_Earth_SCI,...
                          Deg, Num, Index, Options);
    
end % For if

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%