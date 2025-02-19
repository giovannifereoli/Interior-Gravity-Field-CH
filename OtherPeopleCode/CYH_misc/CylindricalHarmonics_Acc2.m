% Parameters (define outside the function)
rho_max = 1;   % Maximum radius (boundary in rho)
L = 2;         % Length of the cylinder in z
m_max = 3;     % Max order in phi (azimuthal)
n_max = 3;     % Max mode in z direction
k_max = 3;     % Max root for radial modes

% Precompute Bessel roots and mode components
bessel_roots = zeros(m_max + 1, k_max);  % Precompute and store roots for each (m, k)
for m = 0:m_max
    for k = 1:k_max
        bessel_roots(m + 1, k) = approximate_besselj_root(m, k);
    end
end

% Generate random coefficients
A_mnk = rand(m_max + 1, n_max, k_max);  % Random coefficients for each (m,n,k)

% Define function to compute gravitational acceleration and partials
function [accel, partials] = compute_grav_accel_and_partials_optimized(x, y, z, A_mnk, bessel_roots, rho_max, L, m_max, n_max, k_max)
    % Convert Cartesian to cylindrical
    R = sqrt(x^2 + y^2);   % Radial distance
    phi = atan2(y, x);     % Azimuthal angle

    V_total = 0;           % Initialize potential
    accel = [0; 0; 0];     % Initialize acceleration
    partials = zeros(3, m_max + 1, n_max, k_max); % Initialize partial derivatives storage
    
    for m = 0:m_max
        for n = 1:n_max
            for k = 1:k_max
                % Retrieve precomputed Bessel root and coefficient
                alpha_mk = bessel_roots(m + 1, k);
                current_A_mnk = A_mnk(m + 1, n, k);

                % Compute cylindrical components once per mode
                R_rho = besselj(m, alpha_mk * R / rho_max);              % Radial component
                Z_z = sin(n * pi * z / L);                               % Axial component
                Phi_cos_sin = cos(m * phi) + sin(m * phi);               % Azimuthal component sum

                % Compute potential term
                V_component = current_A_mnk * R_rho * Phi_cos_sin * Z_z;
                V_total = V_total + V_component;

                % Calculate cylindrical gradients
                dV_dR = current_A_mnk * alpha_mk / rho_max * ...
                        besselj_derivative(m, alpha_mk * R / rho_max) * Phi_cos_sin * Z_z;
                dV_dphi = current_A_mnk * R_rho * (-m * sin(m * phi) + m * cos(m * phi)) * Z_z;
                dV_dz = current_A_mnk * R_rho * Phi_cos_sin * n * pi / L * cos(n * pi * z / L);

                % Convert cylindrical gradients to Cartesian and accumulate
                accel_x = -(dV_dR * cos(phi) - dV_dphi * sin(phi) / R);
                accel_y = -(dV_dR * sin(phi) + dV_dphi * cos(phi) / R);
                accel_z = -dV_dz;
                accel = accel + [accel_x; accel_y; accel_z];
                
                % Store partial derivatives of acceleration w.r.t. current A_mnk
                partials(1, m + 1, n, k) = -accel_x / current_A_mnk;
                partials(2, m + 1, n, k) = -accel_y / current_A_mnk;
                partials(3, m + 1, n, k) = -accel_z / current_A_mnk;
            end
        end
    end
end

% Generate a random trajectory and evaluate with optimized function
num_points = 50;
trajectory = [linspace(-rho_max, rho_max, num_points);  % x-coordinates
              linspace(-rho_max, rho_max, num_points);  % y-coordinates
              linspace(0, L, num_points)];              % z-coordinates

% Preallocate arrays for storing results
accel_values = zeros(3, num_points);
partial_values = zeros(3, m_max + 1, n_max, k_max, num_points);

% Compute values for each point in trajectory
for i = 1:num_points
    x = trajectory(1, i);
    y = trajectory(2, i);
    z = trajectory(3, i);
    [accel_values(:, i), partial_values(:, :, :, :, i)] = ...
        compute_grav_accel_and_partials_optimized(x, y, z, A_mnk, ...
        bessel_roots, rho_max, L, m_max, n_max, k_max);
end

% Plot acceleration along the trajectory
figure;
plot3(trajectory(1, :), trajectory(2, :), trajectory(3, :), 'b', 'LineWidth', 1.5);
hold on;
quiver3(trajectory(1, :), trajectory(2, :), trajectory(3, :), ...
        accel_values(1, :), accel_values(2, :), accel_values(3, :), 0.5, 'r');
title('Random Trajectory with Gravitational Acceleration Vectors');
xlabel('X');
ylabel('Y');
zlabel('Z');
grid on;
axis equal;
view(3);

% Auxiliary function for Bessel root approximation and Bessel derivative
function root = approximate_besselj_root(m, k)
    x_start = (k - 0.5) * pi;
    dx = 0.001;
    tolerance = 1e-6;

    while abs(besselj(m, x_start)) > tolerance
        x_start = x_start + dx;
    end
    
    root = x_start;
end

function dJ = besselj_derivative(m, x)
    dJ = (besselj(m - 1, x) - besselj(m + 1, x)) / 2;
end
