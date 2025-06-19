import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv
from mpl_toolkits.mplot3d import Axes3D

# --- Load your data (adjust path if needed) ---
data = np.load("spherical_gravity_dataset.npz")
points, potentials = data["points"], data["potential"]
center, R = data["center"], data["bounding_radius"]  # true support

# --- Fit params are assumed already in memory: ---
# l_max, n_max, R_ref, labels, fitted_params
# If not, rebuild them exactly as before.

l_max, n_max = 6, 6
R_ref = 2 * R
G = 6.67430e-11

# --- Build radial basis B_{ℓn}(r) on [0,R] ---
Nr = 200
r_grid = np.linspace(1e-8, R, Nr)
dr = r_grid[1] - r_grid[0]
w = r_grid**2 * dr

B = np.zeros((l_max + 1, n_max + 1, Nr))
for ell in range(l_max + 1):
    for n in range(n_max + 1):
        k = n - ell + 2
        if n == ell - 2:
            B[ell, n] = np.log(R / np.maximum(r_grid, 1e-12)) / (2 * ell + 1)
        else:
            B[ell, n] = (R**k - r_grid**k) / (k * (2 * ell + 1))

# --- Precompute Gram inverses Minv[ℓ] ---
Minv = {}
for ell in range(l_max + 1):
    M = np.zeros((n_max + 1, n_max + 1))
    for i in range(n_max + 1):
        for j in range(n_max + 1):
            M[i, j] = np.sum(B[ell, i] * B[ell, j] * w)
    Minv[ell] = np.linalg.inv(M)

# --- Invert for all ρ_{ℓ m n} ---
rho_a = {}  # cosine-part for m≥0
rho_b = {}  # sine-part for m>0

# rebuild design matrix A & solve A·γ = potentials if needed...
# assume `labels` and `fitted_params` are available here

for ell in range(l_max + 1):
    for m in range(ell + 1):
        # cosine γₙ = γ[a_{ℓ}_{m}_{n}]
        γa = np.array(
            [fitted_params[labels.index(f"a_{ell}_{m}_{n}")] for n in range(n_max + 1)]
        )
        φ_lm = γa @ B[ell, :, :]  # φ_{ℓm}(r)
        dvec = np.array([np.sum(φ_lm * B[ell, n] * w) for n in range(n_max + 1)])
        rho_a[(ell, m)] = Minv[ell] @ dvec / (4 * np.pi * G)

        # sine part if m>0
        if m > 0:
            γb = np.array(
                [
                    fitted_params[labels.index(f"b_{ell}_{m}_{n}")]
                    for n in range(n_max + 1)
                ]
            )
            φ_lm_s = γb @ B[ell, :, :]
            dvec_s = np.array(
                [np.sum(φ_lm_s * B[ell, n] * w) for n in range(n_max + 1)]
            )
            rho_b[(ell, m)] = Minv[ell] @ dvec_s / (4 * np.pi * G)

# Optional: print a few sample coefficients
print("ρ_{0,0,n} =", rho_a[(0, 0)])
print("ρ_{1,0,n} =", rho_a[(1, 0)])


# --- Reconstruction function ---
def reconstruct_rho(pts):
    v = pts - center
    r = np.linalg.norm(v, axis=1)
    θ = np.arccos(np.clip(v[:, 2] / r, -1, 1))
    φ = np.arctan2(v[:, 1], v[:, 0])
    ρ = np.full_like(r, np.nan)
    mask = r <= R
    rm, θm, φm = r[mask], θ[mask], φ[mask]

    ρm = np.zeros_like(rm)
    for ell in range(l_max + 1):
        for m in range(ell + 1):
            Plm = lpmv(m, ell, np.cos(θm))
            # cosine term
            ρm += np.sum(
                [
                    rho_a[(ell, m)][n] * rm**n * Plm * np.cos(m * φm)
                    for n in range(n_max + 1)
                ],
                axis=0,
            )
            # sine term
            if m > 0:
                ρm += np.sum(
                    [
                        rho_b[(ell, m)][n] * rm**n * Plm * np.sin(m * φm)
                        for n in range(n_max + 1)
                    ],
                    axis=0,
                )
    ρ[mask] = ρm
    return ρ


# --- Plot XY, XZ, YZ slices ---
def plot_slice(plane, N=300):
    if plane == "xy":
        i1, i2 = 0, 1
    elif plane == "xz":
        i1, i2 = 0, 2
    else:
        i1, i2 = 1, 2
    coords = np.linspace(-R, R, N)
    X, Y = np.meshgrid(coords, coords)
    pts = np.zeros((N * N, 3))
    for i, u in enumerate(coords):
        for j, v in enumerate(coords):
            idx = i * N + j
            pts[idx] = center
            pts[idx, i1] += u
            pts[idx, i2] += v

    Z = reconstruct_rho(pts).reshape(N, N)
    plt.figure(figsize=(5, 4))
    plt.contourf(X, Y, Z, levels=60, cmap="viridis")
    plt.colorbar(label=r"$\rho$")
    plt.xlabel(f"{['X','Y','Z'][i1]} (m)")
    plt.ylabel(f"{['X','Y','Z'][i2]} (m)")
    plt.title(f"{plane.upper()} slice")
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()


for pl in ("xy", "xz", "yz"):
    plot_slice(pl)

# --- 3D spherical surface at r=R ---
nθ, nφ = 80, 160
θ = np.linspace(0, np.pi, nθ)
φ = np.linspace(0, 2 * np.pi, nφ)
Θ, Φ = np.meshgrid(θ, φ)
xs = center[0] + R * np.sin(Θ) * np.cos(Φ)
ys = center[1] + R * np.sin(Θ) * np.sin(Φ)
zs = center[2] + R * np.cos(Θ)
pts_s = np.vstack([xs.ravel(), ys.ravel(), zs.ravel()]).T
ρs = reconstruct_rho(pts_s).reshape(xs.shape)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
norm = plt.Normalize(np.nanmin(ρs), np.nanmax(ρs))
colors = plt.cm.viridis(norm(ρs))
ax.plot_surface(xs, ys, zs, facecolors=colors, rstride=1, cstride=1, linewidth=0)
m = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
m.set_array([])
fig.colorbar(m, ax=ax, shrink=0.5, label=r"$\rho$")
ax.set_title("Density on sphere $r=R$")
plt.tight_layout()
plt.show()
