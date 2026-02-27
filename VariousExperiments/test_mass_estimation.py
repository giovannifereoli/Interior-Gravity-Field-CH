"""
Validate the thin-layer (sheet) mapping for CYLINDRICAL harmonics:

Derived math (sheet at z=0 on the cylinder local plane):
  Δm(Rp) = (Rα / G) * Σ_{n=1..N} ΔA_{0n} * J1( k_{0n} * Rp/Rα )

where:
  - Rα = alpha * radius
  - k_{0n} are zeros of J0
  - ΔA_{0n} are the m=0 cosine coefficients from your fitted params
  - Rp is the disk radius where you want net mass (typically cylinder radius)

This script validates that scaling + sign by:
  1) taking a base polyhedron mesh
  2) applying a controlled "piston" displacement u(ρ) on the patch (net mass change in the patch)
  3) computing dm_true ≈ ρ * ∑ (A_vertex * u_vertex)   (1st-order thin-layer volume)
  4) fitting cylindrical-harmonic coefficients p0, p1 from gravity in the cylinder
  5) computing dm_hat_raw = Rα * Σ ΔA_{0n} J1(...)
     and estimating the effective 1/G scale + sign from many samples

Requirements:
  - mesh_utility.read_pk_file
  - polyhedral_gravity: Polyhedron, PolyhedronIntegrity, GravityEvaluable
  - numpy, scipy, matplotlib

Notes:
  - We do NOT assume what "G" is inside polyhedral_gravity. Instead we estimate an effective scale:
        dm_true ≈ S * dm_hat_raw
    then S ≈ 1/G_eff (and sign).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

from scipy.special import jv as BesselJ, jvp as BesselJp, jn_zeros, j1

import mesh_utility
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable


# ======================================================================================
# 0) Spec
# ======================================================================================


@dataclass
class CylinderSpec:
    center: np.ndarray
    radius: float
    height: float
    rotation: np.ndarray  # local->global
    alpha: float


# ======================================================================================
# 1) Mesh helpers
# ======================================================================================


def vertex_normals_from_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, float)
    f = np.asarray(faces, int)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # area-weighted face normals
    vn = np.zeros_like(v)
    np.add.at(vn, f[:, 0], fn)
    np.add.at(vn, f[:, 1], fn)
    np.add.at(vn, f[:, 2], fn)
    nrm = np.linalg.norm(vn, axis=1)
    return vn / (nrm[:, None] + 1e-30)


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def vertex_area_weights(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    Af = triangle_areas(vertices, faces)
    w = np.zeros(len(vertices), float)
    for j in range(3):
        np.add.at(w, faces[:, j], Af / 3.0)
    return w


# ======================================================================================
# 2) Patch + piston (thin layer)
# ======================================================================================


def patch_mask_vertices(
    vertices: np.ndarray, spec: CylinderSpec, r_patch: float, z_top_local: float
) -> np.ndarray:
    v = np.asarray(vertices, float)
    pl = (v - spec.center) @ spec.rotation.T
    rho = np.sqrt(pl[:, 0] ** 2 + pl[:, 1] ** 2)
    z = pl[:, 2]
    return (rho <= r_patch) & (z <= z_top_local)


def smooth_disk_taper(
    vertices: np.ndarray, spec: CylinderSpec, r_patch: float, power: float = 1.0
) -> np.ndarray:
    v = np.asarray(vertices, float)
    pl = (v - spec.center) @ spec.rotation.T
    rho = np.sqrt(pl[:, 0] ** 2 + pl[:, 1] ** 2)
    rr = np.clip(rho / (r_patch + 1e-30), 0.0, 1.0)
    w = 0.5 * (1.0 + np.cos(np.pi * rr))
    w[rho > r_patch] = 0.0
    if power != 1.0:
        w = w**power
    return w


def apply_patch_piston(vertices0, normals0, mask, taper, amp):
    """
    u = amp * taper on patch, 0 outside; displace along outward vertex normals.
    """
    u = np.zeros(len(vertices0), float)
    u[mask] = amp * taper[mask]
    v_new = vertices0 + normals0 * u[:, None]
    return v_new, u


# ======================================================================================
# 3) Cylinder sampling + basis matrices (fixed)
# ======================================================================================


def generate_points_in_cylinder(
    spec: CylinderSpec, num_points: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, num_points)
    r = np.sqrt(rng.uniform(0, spec.radius**2, num_points))
    z = rng.uniform(0, spec.height, num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    local = np.column_stack((x, y, z))
    return local @ spec.rotation.T + spec.center


def cartesian_to_cylindrical_acceleration(
    spec: CylinderSpec, points: np.ndarray, acc: np.ndarray
) -> np.ndarray:
    pl = (points - spec.center) @ spec.rotation.T
    rho = np.sqrt(pl[:, 0] ** 2 + pl[:, 1] ** 2)
    phi = np.arctan2(pl[:, 1], pl[:, 0])

    acc_l = acc @ spec.rotation.T
    ax, ay, az = acc_l[:, 0], acc_l[:, 1], acc_l[:, 2]

    a_rho = ax * np.cos(phi) + ay * np.sin(phi)
    a_phi = -ax * np.sin(phi) + ay * np.cos(phi)
    return np.column_stack((a_rho, a_phi, az))


def prepare_A_acc(
    spec: CylinderSpec, points: np.ndarray, n_n: int, n_m: int
) -> np.ndarray:
    pl = (points - spec.center) @ spec.rotation.T
    rho = np.sqrt(pl[:, 0] ** 2 + pl[:, 1] ** 2)
    phi = np.arctan2(pl[:, 1], pl[:, 0])
    z = pl[:, 2]

    N = points.shape[0]
    P = 2 * n_n * n_m
    A = np.zeros((3 * N, P), float)

    R_alpha = spec.alpha * spec.radius

    k_mn = lambda m, n: jn_zeros(m, n)[-1]

    idx = 0
    r0 = np.arange(N) * 3
    r1 = r0 + 1
    r2 = r0 + 2

    for m in range(n_m):
        cos_m = np.cos(m * phi)
        sin_m = np.sin(m * phi)
        for n in range(1, n_n + 1):
            k = k_mn(m, n)
            exp_term = np.exp(-k * z / R_alpha)
            J = BesselJ(m, k * rho / R_alpha)
            Jp = BesselJp(m, k * rho / R_alpha)

            dV_drho = (k / R_alpha) * exp_term * Jp
            dV_dphi = exp_term * J * (m / (rho + 1e-14))
            dV_dz = (-k / R_alpha) * exp_term * J

            A[r0, idx] = dV_drho * cos_m
            A[r0, idx + 1] = dV_drho * sin_m

            A[r1, idx] = dV_dphi * (-sin_m)
            A[r1, idx + 1] = dV_dphi * (cos_m)

            A[r2, idx] = dV_dz * cos_m
            A[r2, idx + 1] = dV_dz * sin_m

            idx += 2

    return A


def prepare_A_pot(
    spec: CylinderSpec, points: np.ndarray, n_n: int, n_m: int
) -> np.ndarray:
    pl = (points - spec.center) @ spec.rotation.T
    rho = np.sqrt(pl[:, 0] ** 2 + pl[:, 1] ** 2)
    phi = np.arctan2(pl[:, 1], pl[:, 0])
    z = pl[:, 2]

    N = points.shape[0]
    P = 2 * n_n * n_m
    A = np.zeros((N, P), float)

    R_alpha = spec.alpha * spec.radius

    k_mn = lambda m, n: jn_zeros(m, n)[-1]

    idx = 0
    for m in range(n_m):
        cos_m = np.cos(m * phi)
        sin_m = np.sin(m * phi)
        for n in range(1, n_n + 1):
            k = k_mn(m, n)
            exp_term = np.exp(-k * z / R_alpha)
            J = BesselJ(m, k * rho / R_alpha)
            A[:, idx] = exp_term * J * cos_m
            A[:, idx + 1] = exp_term * J * sin_m
            idx += 2

    return A


def enforce_B0n(params: np.ndarray, n_n: int):
    # keep consistent with your fits if you normally enforce this
    for n in range(n_n):
        params[2 * n + 1] = 0.0


# ======================================================================================
# 4) Deterministic coefficient fitter (fixed points + fixed A matrices)
# ======================================================================================


class FixedCylinderFitter:
    def __init__(
        self,
        vertices0,
        faces,
        density,
        spec,
        points,
        A_acc,
        A_pot,
        n_n,
        n_m,
        enforce_b0n=True,
    ):
        self.faces = np.asarray(faces, int)
        self.density = float(density)
        self.spec = spec
        self.points = np.asarray(points, float)
        self.A = np.vstack([np.asarray(A_acc, float), np.asarray(A_pot, float)])
        self.n_n = int(n_n)
        self.n_m = int(n_m)
        self.enforce_b0n = bool(enforce_b0n)

    def fit(self, vertices: np.ndarray) -> np.ndarray:
        poly = Polyhedron(
            polyhedral_source=(np.asarray(vertices, float), self.faces),
            density=self.density,
            integrity_check=PolyhedronIntegrity.DISABLE,
        )
        eval_poly = GravityEvaluable(poly)

        N = self.points.shape[0]
        pot = np.zeros(N, float)
        acc = np.zeros((N, 3), float)

        for i, p in enumerate(self.points):
            V, a, _ = eval_poly(computation_points=p, parallel=False)
            pot[i] = float(np.squeeze(V))
            acc[i] = np.squeeze(a)

        cyl_acc = cartesian_to_cylindrical_acceleration(self.spec, self.points, acc)
        b_acc = cyl_acc.reshape(-1, order="C")
        b_pot = pot.astype(float)
        b = np.hstack([b_acc, b_pot])

        params, *_ = np.linalg.lstsq(self.A, b, rcond=None)
        if self.enforce_b0n:
            enforce_B0n(params, n_n=self.n_n)
        return params


# ======================================================================================
# 5) Sheet-math estimator: dm_hat_raw = Rα * Σ ΔA_0n J1(...)
#    True dm_true (thin-layer proxy): dm_true = ρ * Σ (A_vertex * u_vertex) on patch
# ======================================================================================


def dm_hat_raw_from_dp(dp, n_n, R_alpha, R_patch):
    from scipy.special import jn_zeros, j1 as J1
    import numpy as np

    # 1. Identify your internal Gravity Constant.
    # If polyhedral_gravity uses G=1 internally, set this to 1.0
    G_internal = 6.64996410e-11

    dp = np.asarray(dp).ravel()
    k0n = jn_zeros(0, n_n)
    dA0n = dp[0 : 2 * n_n : 2]

    # 2. The Argument
    arg = k0n * R_patch / R_alpha

    # 3. The Integral logic
    # Since dA0n comes from a fit where A_acc = (k/Ra) * J0,
    # the dA0n already has the 'units' of (potential / length).
    # The integral of J0(k*rho/Ra) * rho is (Ra*Rp/k) * J1(arg).

    # This is the scaling that SHOULD be alpha-independent:
    dm_list = dA0n * (R_alpha * R_patch / k0n) * J1(arg)

    # 4. Final Mass (check for the 2*pi factor depending on your gravity engine)
    # If it's still off by ~3, change (1/G) to (2*pi/G)
    dm = (2 * np.pi / G_internal) * np.sum(dm_list)

    return float(dm)


# ======================================================================================
# 6) Main validation
# ======================================================================================


def main():
    os.makedirs("Images", exist_ok=True)

    # ---------- USER SETTINGS ----------
    mesh_path = "3dmeshes/eros.pk"
    DENSITY = 1.0

    spec = CylinderSpec(
        center=np.array([0.0, 0.0, 0.28], float),
        radius=0.10,
        height=0.50,
        rotation=np.eye(3),
        alpha=1000.0,
    )
    R_alpha = spec.alpha * spec.radius
    R_patch = spec.radius  # validate net mass in the cylinder footprint

    z_top_local = 0.05
    taper_power = 1.0

    n_n, n_m = 25, 25
    NUM_POINTS = 1400
    seed_points = 1

    # piston amplitude range for trials (LU)
    amp_min, amp_max = -3e-4, 3e-4
    N_trials = 14
    rng = np.random.default_rng(42)

    # ---------- LOAD MESH ----------
    vertices0, faces = mesh_utility.read_pk_file(mesh_path)
    vertices0 = np.asarray(vertices0, float)
    faces = np.asarray(faces, int)

    normals0 = vertex_normals_from_faces(vertices0, faces)
    wA = vertex_area_weights(vertices0, faces)

    mask = patch_mask_vertices(
        vertices0, spec, r_patch=R_patch, z_top_local=z_top_local
    )
    if not np.any(mask):
        raise RuntimeError(
            "Patch mask empty. Check cylinder center/rotation/z_top_local."
        )

    taper = smooth_disk_taper(vertices0, spec, r_patch=R_patch, power=taper_power)
    taper[~mask] = 0.0
    wAp = wA * mask.astype(float)

    # ---------- FIXED CYLINDER POINTS + FIXED A MATRICES ----------
    points = generate_points_in_cylinder(spec, num_points=NUM_POINTS, seed=seed_points)
    A_acc = prepare_A_acc(spec, points, n_n=n_n, n_m=n_m)
    A_pot = prepare_A_pot(spec, points, n_n=n_n, n_m=n_m)

    fitter = FixedCylinderFitter(
        vertices0=vertices0,
        faces=faces,
        density=DENSITY,
        spec=spec,
        points=points,
        A_acc=A_acc,
        A_pot=A_pot,
        n_n=n_n,
        n_m=n_m,
        enforce_b0n=True,
    )

    print("Computing baseline coefficients p0...")
    p0 = fitter.fit(vertices0)

    # ---------- TRIALS ----------
    amps = rng.uniform(amp_min, amp_max, size=N_trials)

    dm_true_list = []
    dm_hat_raw_list = []

    for k, amp in enumerate(amps):
        v1, u = apply_patch_piston(vertices0, normals0, mask, taper, amp=amp)

        # thin-layer proxy "true" net mass change in the patch
        dV_patch = float(np.sum(wAp * u))
        dm_true = DENSITY * dV_patch

        p1 = fitter.fit(v1)
        dp = p1 - p0

        dm_hat_raw = dm_hat_raw_from_dp(dp, n_n=n_n, R_alpha=R_alpha, R_patch=R_patch)

        dm_true_list.append(dm_true)
        dm_hat_raw_list.append(dm_hat_raw)

        print(
            f"trial {k+1:02d}/{N_trials}: amp={amp:+.3e}  dm_true={dm_true:+.3e}  dm_hat_raw={dm_hat_raw:+.3e}"
        )

    dm_true = np.array(dm_true_list)
    dm_hat_raw = np.array(dm_hat_raw_list)

    # ---------- FIT SCALE: dm_true ≈ s * dm_hat_raw ----------
    # s ≈ 1/G_eff (with sign)
    # least squares slope through origin:
    denom = float(np.dot(dm_hat_raw, dm_hat_raw)) + 1e-30
    s = float(np.dot(dm_hat_raw, dm_true) / denom)
    # LU = 20.143
    # G = 6.64996410e-11  # for reference; not used in fitting
    # s = 1 / (LU * G)

    dm_pred = s * dm_hat_raw
    rmse = float(np.sqrt(np.mean((dm_pred - dm_true) ** 2)))
    rel = rmse / (np.sqrt(np.mean(dm_true**2)) + 1e-30)

    print("\n=== Sheet-math validation ===")
    print(f"Best-fit scale s (≈ 1/G_eff): {s:+.6e}")
    if abs(s) > 0:
        print(f"Implied G_eff: {1.0/s:+.6e}")
    print(f"RMSE(dm_pred vs dm_true): {rmse:.3e}")
    print(f"Relative RMSE (RMS-normalized): {rel:.3%}")
    print("============================\n")

    # ---------- PLOTS ----------
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(dm_true, dm_pred, s=50)
    lim = 1.1 * max(np.max(np.abs(dm_true)), np.max(np.abs(dm_pred)), 1e-30)
    plt.plot([-lim, lim], [-lim, lim], linestyle="--")
    plt.xlabel("dm_true  (ρ * ∑ A_vertex u)  [density*LU^3]")
    plt.ylabel("dm_pred  (s * Rα Σ ΔA0n J1(...))  [density*LU^3]")
    plt.title("Validate Δm from cylindrical coeffs (thin-layer sheet math)")
    plt.grid(True, linestyle="--", alpha=0.6)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(dm_true, marker="o", label="dm_true")
    plt.plot(dm_pred, marker="s", label="dm_pred")
    plt.xlabel("trial")
    plt.ylabel("dm")
    plt.title("Per-trial dm_true vs dm_pred")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)

    plt.show()


if __name__ == "__main__":
    main()
