% Parameters
rho_max = 1;   % Maximum radius (boundary in rho)
L = 2;         % Length of the cylinder in z
m_max = 3;     % Max order in phi (azimuthal)
n_max = 3;     % Max mode in z direction
k_max = 3;     % Max root for radial modes

% Discretization for plotting
x_vals = linspace(-rho_max, rho_max, 50); % Cartesian x range
y_vals = linspace(-rho_max, rho_max, 50); % Cartesian y range
z_vals = linspace(0, L, 50);              % z range

% Create Cartesian grids using meshgrid
[X, Y, Z] = meshgrid(x_vals, y_vals, z_vals);
R = sqrt(X.^2 + Y.^2);                   % Convert to radial distance
PHI = atan2(Y, X);                       % Convert to azimuthal angle
V_total = zeros(size(R));                % Initialize potential matrix

% Function to approximate the k-th root of Bessel function J_m(x)
find_bessel_approx_root = @(m, k) approximate_besselj_root(m, k);

% Loop through series terms
for m = 0:m_max
    for n = 1:n_max
        for k = 1:k_max
            % Calculate the k-th root of J_m using the approximation method
            alpha_mk = find_bessel_approx_root(m, k);
            R_rho = besselj(m, alpha_mk * R / rho_max);  % Radial component
            
            % Azimuthal component (sine and cosine terms)
            Phi_cos = cos(m * PHI);
            Phi_sin = sin(m * PHI);
            
            % Axial component in z (using sine for Dirichlet conditions)
            Z_z = sin(n * pi * Z / L);
            
            % Combine terms into solution component with random amplitude
            A_mnk = rand() * 0.1;  % Random coefficient for visualization
            V_total = V_total + A_mnk * R_rho .* (Phi_cos + Phi_sin) .* Z_z;
        end
    end
end

% Plot results using slice
figure;
slice(X, Y, Z, V_total, 0, 0, linspace(0, L, 5)); % Slice plot at x=0, y=0, and multiple z planes
shading interp;
title('Solution of Laplace Equation in Cylindrical Coordinates (Slices)');
xlabel('X');
ylabel('Y');
zlabel('Z');
colorbar;
axis equal;
view(3);

% Auxiliary function to approximate the k-th root of the Bessel function J_m(x)
function root = approximate_besselj_root(m, k)
    % Starting guess based on approximate location of roots
    x_start = (k - 0.5) * pi;  % Approximate starting point for the k-th root
    dx = 0.001;  % Step size for locating root
    tolerance = 1e-6;  % Tolerance for considering a value close to zero

    % Iterate through values to find where Bessel function is close to zero
    while abs(besselj(m, x_start)) > tolerance
        x_start = x_start + dx;
    end
    
    % Return the approximate root
    root = x_start;
end
