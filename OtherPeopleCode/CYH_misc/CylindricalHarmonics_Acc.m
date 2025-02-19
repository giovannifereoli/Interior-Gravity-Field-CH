% Parameters (same as in potential calculation)
rho_max = 1;   % Maximum radius (boundary in rho)
L = 2;         % Length of the cylinder in z
m_max = 3;     % Max order in phi (azimuthal)
n_max = 3;     % Max mode in z direction
k_max = 3;     % Max root for radial modes

% Discretization for computing acceleration
x_vals = linspace(-rho_max, rho_max, 50);
y_vals = linspace(-rho_max, rho_max, 50);
z_vals = linspace(0, L, 50);

% Create Cartesian grids using meshgrid
[X, Y, Z] = meshgrid(x_vals, y_vals, z_vals);
R = sqrt(X.^2 + Y.^2);                   % Radial distance
PHI = atan2(Y, X);                       % Azimuthal angle

% Initialize potential, acceleration, and coefficient storage
V_total = zeros(size(R));                % Potential matrix
accel_x = zeros(size(R));                % Acceleration components
accel_y = zeros(size(R));
accel_z = zeros(size(R));
A_mnk = rand(m_max + 1, n_max, k_max);   % Random coefficients for each (m,n,k)

% Storage for partial derivatives of acceleration w.r.t each A_mnk
partial_accel_x_wrt_A = zeros([size(R), m_max + 1, n_max, k_max]);
partial_accel_y_wrt_A = zeros([size(R), m_max + 1, n_max, k_max]);
partial_accel_z_wrt_A = zeros([size(R), m_max + 1, n_max, k_max]);

% Function to approximate the k-th root of Bessel function J_m(x)
find_bessel_approx_root = @(m, k) approximate_besselj_root(m, k);

% Loop through series terms to calculate potential and its derivatives
for m = 0:m_max
    for n = 1:n_max
        for k = 1:k_max
            % Calculate the k-th root of J_m
            alpha_mk = find_bessel_approx_root(m, k);
            
            % Components of the potential
            R_rho = besselj(m, alpha_mk * R / rho_max);      % Radial
            Phi_cos = cos(m * PHI);                           % Azimuthal cosine
            Phi_sin = sin(m * PHI);                           % Azimuthal sine
            Z_z = sin(n * pi * Z / L);                        % Axial (Dirichlet)

            % Retrieve stored coefficient A_mnk
            current_A_mnk = A_mnk(m + 1, n, k);

            % Accumulate the potential V
            V_component = current_A_mnk * R_rho .* (Phi_cos + Phi_sin) .* Z_z;
            V_total = V_total + V_component;

            % Compute the gradient of each component to get acceleration
            [dV_dx, dV_dy, dV_dz] = gradient(V_component, x_vals(2)-x_vals(1), y_vals(2)-y_vals(1), z_vals(2)-z_vals(1));

            % Sum the acceleration components
            accel_x = accel_x - dV_dx;
            accel_y = accel_y - dV_dy;
            accel_z = accel_z - dV_dz;
            
            % Store partial derivatives of acceleration with respect to current A_mnk
            partial_accel_x_wrt_A(:, :, :, m + 1, n, k) = -dV_dx;
            partial_accel_y_wrt_A(:, :, :, m + 1, n, k) = -dV_dy;
            partial_accel_z_wrt_A(:, :, :, m + 1, n, k) = -dV_dz;
        end
    end
end

% Plotting the computed acceleration field (magnitude)
figure;
quiver3(X, Y, Z, accel_x, accel_y, accel_z);
title('Acceleration Field from Gravitational Potential');
xlabel('X'); ylabel('Y'); zlabel('Z');
axis equal;
grid on;
view(3);

% Auxiliary function to approximate the k-th root of the Bessel function J_m(x)
function root = approximate_besselj_root(m, k)
    x_start = (k - 0.5) * pi;  % Starting guess
    dx = 0.001;                % Step size
    tolerance = 1e-6;          % Root tolerance

    while abs(besselj(m, x_start)) > tolerance
        x_start = x_start + dx;
    end
    
    root = x_start;
end
