"""
TAG-like local shape perturbation + cylinder gravity fitting + 3 plots only:

PLOT 1: displacement magnitude map on the surface (ONLY in the TAG/cylinder "below" region)
PLOT 2: cylinder + sampled points (single plot)
PLOT 3: power spectra (RMS coefficients by order m) for base vs pert vs delta

Drop-in: this file assumes you already have:
  - mesh_utility.read_pk_file
  - polyhedral_gravity Polyhedron / GravityEvaluable
  - your fit_cylinder_coeffs_from_polyhedron(), prepare_*(), etc. (copied below only if needed)

Key idea:
  - Define TAG region = vertices inside cylinder footprint (rho <= Rtag) AND below cylinder top plane (z_local <= z_top)
  - Perturb ONLY those vertices along local -axis ("down") or along vertex normals (pick one)

Author: (you)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm, colors

import mesh_utility
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
from scipy.special import jv as BesselJ, jvp as BesselJp, jn_zeros


# ======================================================================================
# 0) Spec
# ======================================================================================


@dataclass
class CylinderSpec:
    center: np.ndarray  # (3,)
    radius: float
    height: float
    rotation: np.ndarray  # (3,3) local->global
    alpha: float  # ALPHA scaling used in basis (R_alpha = alpha*radius)


# ======================================================================================
# 1) Geometry helpers
# ======================================================================================


def _vertex_normals_from_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices)
    f = np.asarray(faces, dtype=int)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # area-weighted face normals

    vn = np.zeros_like(v)
    np.add.at(vn, f[:, 0], fn)
    np.add.at(vn, f[:, 1], fn)
    np.add.at(vn, f[:, 2], fn)

    nrm = np.linalg.norm(vn, axis=1)
    return vn / (nrm[:, None] + 1e-30)


def _set_axes_equal(ax):
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xmid, ymid, zmid = (
        (xlim[0] + xlim[1]) / 2,
        (ylim[0] + ylim[1]) / 2,
        (zlim[0] + zlim[1]) / 2,
    )
    r = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    ax.set_xlim3d(xmid - r, xmid + r)
    ax.set_ylim3d(ymid - r, ymid + r)
    ax.set_zlim3d(zmid - r, zmid + r)


def _cylinder_wireframe(spec: CylinderSpec, n_theta=72, n_z=18):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z = np.linspace(0, spec.height, n_z)
    th, zz = np.meshgrid(theta, z)

    x = spec.radius * np.cos(th)
    y = spec.radius * np.sin(th)

    local = np.stack([x, y, zz], axis=-1).reshape(-1, 3)
    glob = local @ spec.rotation.T + spec.center
    X = glob[:, 0].reshape(n_z, n_theta)
    Y = glob[:, 1].reshape(n_z, n_theta)
    Z = glob[:, 2].reshape(n_z, n_theta)
    return X, Y, Z


# ======================================================================================
# 2) TAG-like local perturbation (ONLY below cylinder)
# ======================================================================================


def tag_region_mask_vertices(
    vertices: np.ndarray,
    spec: CylinderSpec,
    r_tag: float,
    z_top_local: float,
) -> np.ndarray:
    """
    Define vertices impacted by TAG:

    Transform vertices into cylinder local frame:
      p_local = (p - center) @ rotation.T

    Then select:
      rho <= r_tag  AND  z_local <= z_top_local

    For your typical cylinder sampling where local z goes 0..height:
      - the "base plane" is z_local = 0
      - "below it" means z_local <= z_top_local, where z_top_local is small-ish (e.g. 0.05)
    """
    v = np.asarray(vertices)
    pl = (v - spec.center) @ spec.rotation.T
    rho = np.sqrt(pl[:, 0] ** 2 + pl[:, 1] ** 2)
    z = pl[:, 2]
    return (rho <= r_tag) & (z <= z_top_local)


def perturb_mesh_locally_TAG(
    vertices: np.ndarray,
    faces: np.ndarray,
    spec: CylinderSpec,
    r_tag: Optional[float] = None,
    z_top_local: float = 0.05,
    sigma: float = 1e-4,
    seed: int = 11,
    clip: float = 3.0,
    mode: str = "down_axis",
    smooth_radial: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      v_pert, mask (boolean per-vertex)

    mode:
      - "down_axis": displace along -cylinder axis (local -z), i.e. a "crater/excavation" direction
      - "normal": displace along vertex normals (less TAG-like but ok)
      - "up_axis": along +cylinder axis

    smooth_radial:
      apply a taper w(rho) = 0.5*(1+cos(pi*rho/r_tag)) so displacement goes to 0 at the edge.
      This prevents a sharp step at rho=r_tag.
    """
    v = np.asarray(vertices)
    f = np.asarray(faces, dtype=int)
    rng = np.random.default_rng(seed)

    if r_tag is None:
        r_tag = spec.radius  # default: exactly your cylinder footprint

    mask = tag_region_mask_vertices(v, spec, r_tag=r_tag, z_top_local=z_top_local)
    if not np.any(mask):
        # Nothing selected -> return original
        return v.copy(), mask

    # radial taper in local frame
    pl = (v - spec.center) @ spec.rotation.T
    rho = np.sqrt(pl[:, 0] ** 2 + pl[:, 1] ** 2)

    w = np.ones(v.shape[0])
    if smooth_radial:
        # cosine taper inside r_tag, zero outside
        rr = np.clip(rho / (r_tag + 1e-30), 0.0, 1.0)
        w = 0.5 * (1.0 + np.cos(np.pi * rr))
        w[rho > r_tag] = 0.0

    # random scalar displacement for selected vertices
    d = rng.normal(0.0, sigma, size=v.shape[0])
    d = np.clip(d, -clip * sigma, clip * sigma)
    d = d * w
    d[~mask] = 0.0

    # direction field
    if mode == "normal":
        n = _vertex_normals_from_faces(v, f)
        v_pert = v + n * d[:, None]
        return v_pert, mask

    # cylinder local axis in GLOBAL coordinates is rotation.T @ [0,0,1]
    axis_global = spec.rotation.T @ np.array([0.0, 0.0, 1.0])
    axis_global = axis_global / (np.linalg.norm(axis_global) + 1e-30)

    if mode == "down_axis":
        v_pert = v - axis_global[None, :] * d[:, None]
        return v_pert, mask

    if mode == "up_axis":
        v_pert = v + axis_global[None, :] * d[:, None]
        return v_pert, mask

    raise ValueError(f"Unknown mode={mode}")


# ======================================================================================
# 3) Cylinder sampling + fit (your functions, lightly cleaned)
# ======================================================================================


def generate_points_in_cylinder(
    spec: CylinderSpec, num_points: int, seed: int = 1
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


def prepare_linear_system_for_cylindrical_acceleration(
    spec: CylinderSpec, points: np.ndarray, cyl_acc: np.ndarray, n_n: int, n_m: int
) -> Tuple[np.ndarray, np.ndarray]:
    pl = (points - spec.center) @ spec.rotation.T
    rho = np.sqrt(pl[:, 0] ** 2 + pl[:, 1] ** 2)
    phi = np.arctan2(pl[:, 1], pl[:, 0])
    z = pl[:, 2]

    N = points.shape[0]
    P = 2 * n_n * n_m
    A = np.zeros((3 * N, P), float)
    b = cyl_acc.reshape(-1, order="C")

    def k_mn(m, n):
        return jn_zeros(m, n)[n - 1]

    R_alpha = spec.alpha * spec.radius

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

    return A, b


def prepare_linear_system_for_cylindrical_potential(
    spec: CylinderSpec, points: np.ndarray, pot: np.ndarray, n_n: int, n_m: int
) -> Tuple[np.ndarray, np.ndarray]:
    pl = (points - spec.center) @ spec.rotation.T
    rho = np.sqrt(pl[:, 0] ** 2 + pl[:, 1] ** 2)
    phi = np.arctan2(pl[:, 1], pl[:, 0])
    z = pl[:, 2]

    N = points.shape[0]
    P = 2 * n_n * n_m
    A = np.zeros((N, P), float)
    b = pot.astype(float)

    def k_mn(m, n):
        return jn_zeros(m, n)[n - 1]

    R_alpha = spec.alpha * spec.radius

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

    return A, b


@dataclass
class FitResult:
    params: np.ndarray
    points: np.ndarray
    meta: Dict


def _zero_B0n(params: np.ndarray, n_n: int):
    for n in range(n_n):
        params[2 * n + 1] = 0.0


def fit_cylinder_coeffs_from_polyhedron(
    vertices: np.ndarray,
    faces: np.ndarray,
    density: float,
    spec: CylinderSpec,
    num_points: int,
    n_n: int,
    n_m: int,
    seed_points: int = 1,
    parallel: bool = False,
    enforce_B0n: bool = True,
) -> FitResult:
    poly = Polyhedron(
        polyhedral_source=(np.asarray(vertices), np.asarray(faces)),
        density=density,
        integrity_check=PolyhedronIntegrity.DISABLE,
    )
    eval_poly = GravityEvaluable(poly)

    pts = generate_points_in_cylinder(spec, num_points=num_points, seed=seed_points)

    pot = np.zeros(num_points, float)
    acc = np.zeros((num_points, 3), float)
    for i, p in enumerate(pts):
        V, a, _ = eval_poly(computation_points=p, parallel=parallel)
        pot[i] = float(np.squeeze(V))
        acc[i] = np.squeeze(a)

    cyl_acc = cartesian_to_cylindrical_acceleration(spec, pts, acc)

    A_acc, b_acc = prepare_linear_system_for_cylindrical_acceleration(
        spec, pts, cyl_acc, n_n, n_m
    )
    A_pot, b_pot = prepare_linear_system_for_cylindrical_potential(
        spec, pts, pot, n_n, n_m
    )

    aug_A = np.vstack([A_acc, A_pot])
    aug_b = np.hstack([b_acc, b_pot])

    params, *_ = np.linalg.lstsq(aug_A, aug_b, rcond=None)

    if enforce_B0n:
        _zero_B0n(params, n_n=n_n)

    return FitResult(
        params=params,
        points=pts,
        meta=dict(
            n_n=n_n,
            n_m=n_m,
            num_points=num_points,
            seed_points=seed_points,
            density=density,
        ),
    )


# ======================================================================================
# 4) 3 Plots
# ======================================================================================


def plot1_displacement_map_TAG(
    vertices_base, vertices_pert, faces, mask, spec, outpath=None
):
    """
    Single displacement plot:
      - face-colored by |Δv| (per-face avg)
      - highlight TAG region boundary by overlaying the cylinder rim circle at z=0
    """
    v0 = np.asarray(vertices_base)
    v1 = np.asarray(vertices_pert)
    f = np.asarray(faces, dtype=int)

    dv = v1 - v0
    mag_v = np.linalg.norm(dv, axis=1)
    mag_f = mag_v[f].mean(axis=1)

    # robust scaling
    vmax = np.percentile(mag_f[mag_f > 0], 99.5) if np.any(mag_f > 0) else 1.0
    norm = colors.Normalize(vmin=0.0, vmax=max(vmax, 1e-30))
    cmap = cm.get_cmap("viridis")
    facecolors = cmap(norm(mag_f))

    tris = v0[f]
    coll = Poly3DCollection(tris, facecolors=facecolors, edgecolor="none", alpha=1.0)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(coll)

    # cylinder rim at z_local=0 (TAG plane)
    th = np.linspace(0, 2 * np.pi, 200)
    rim_local = np.column_stack(
        [spec.radius * np.cos(th), spec.radius * np.sin(th), np.zeros_like(th)]
    )
    rim_glob = rim_local @ spec.rotation.T + spec.center
    ax.plot(rim_glob[:, 0], rim_glob[:, 1], rim_glob[:, 2], linewidth=2.0)

    ax.auto_scale_xyz(v0[:, 0], v0[:, 1], v0[:, 2])
    _set_axes_equal(ax)
    ax.set_title(r"TAG-like local perturbation: surface $||\Delta v||$")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label(r"$||\Delta v||$ (LU)")

    if outpath:
        plt.savefig(outpath, dpi=400, bbox_inches="tight")
    return fig, ax


def plot2_cylinder_and_points(vertices, faces, spec, points, outpath=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    tris = vertices[np.asarray(faces, dtype=int)]
    ax.add_collection3d(
        Poly3DCollection(
            tris, facecolor=(0.75, 0.75, 0.75), edgecolor="none", alpha=0.30
        )
    )

    X, Y, Z = _cylinder_wireframe(spec)
    ax.plot_wireframe(
        X, Y, Z, rcount=Z.shape[0], ccount=Z.shape[1], linewidth=0.6, alpha=0.9
    )

    # axis
    p0 = spec.center
    p1 = spec.center + spec.rotation.T @ np.array([0.0, 0.0, spec.height])
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=2.0)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=6, alpha=0.7)

    ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    _set_axes_equal(ax)
    ax.set_title("Cylinder + gravity sample points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if outpath:
        plt.savefig(outpath, dpi=400, bbox_inches="tight")
    return fig, ax


def plot3_power_spectrum(base_params, pert_params, n_n, n_m, outpath=None):
    """
    Power spectrum: RMS per order m for base / pert / delta.
    """

    def unpack(params):
        A = np.zeros((n_m, n_n))
        B = np.zeros((n_m, n_n))
        idx = 0
        for m in range(n_m):
            for n in range(n_n):
                A[m, n] = params[idx]
                B[m, n] = params[idx + 1]
                idx += 2
        return A, B

    A0, B0 = unpack(base_params)
    A1, B1 = unpack(pert_params)
    Ad, Bd = unpack(pert_params - base_params)

    rms0 = np.sqrt(np.sum(A0**2 + B0**2, axis=1))
    rms1 = np.sqrt(np.sum(A1**2 + B1**2, axis=1))
    rmsd = np.sqrt(np.sum(Ad**2 + Bd**2, axis=1))

    m = np.arange(n_m)
    fig = plt.figure(figsize=(12, 8))
    plt.semilogy(m, rms0, marker="o", linestyle="-", label="RMS(base)")
    plt.semilogy(m, rms1, marker="s", linestyle="--", label="RMS(pert)")
    plt.semilogy(m, rmsd, marker="d", linestyle="-.", label="RMS(delta)")
    plt.xlabel("Order $m$ (-)")
    plt.ylabel(r"$\sqrt{\sum_n (A_{mn}^2 + B_{mn}^2)}$ (-)")
    plt.title("Power spectrum by order (cylindrical coeffs)")
    plt.grid(True, linestyle="--", which="both", alpha=0.6)
    plt.minorticks_on()
    plt.legend(frameon=True)

    if outpath:
        plt.savefig(outpath, dpi=400, bbox_inches="tight")
    return fig


# ======================================================================================
# 5) Main (minimal)
# ======================================================================================

if __name__ == "__main__":
    os.makedirs("Images", exist_ok=True)

    # --- load mesh ---
    vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
    vertices, faces = np.asarray(vertices), np.asarray(faces)

    DENSITY = 1.0

    # --- cylinder spec (your choice) ---
    CYLINDER_CENTER = np.array([0.0, 0.0, 0.28])
    CYLINDER_HEIGHT = 0.5
    CYLINDER_RADIUS = 0.1
    CYLINDER_ROTATION = np.eye(3)
    ALPHA = 100.0

    spec = CylinderSpec(
        center=CYLINDER_CENTER,
        radius=CYLINDER_RADIUS,
        height=CYLINDER_HEIGHT,
        rotation=CYLINDER_ROTATION,
        alpha=ALPHA,
    )

    # --- TAG-like local perturbation parameters ---
    # perturb ONLY within footprint r_tag and shallow depth z_top_local in local cylinder frame.
    r_tag = CYLINDER_RADIUS  # same as cylinder footprint (you can set smaller)
    z_top_local = 0.05  # only vertices with z_local <= 0.05 get perturbed
    sigma_shape = 1e-4  # LU (pick physical scale)
    seed_shape = 11

    v_pert, mask = perturb_mesh_locally_TAG(
        vertices=vertices,
        faces=faces,
        spec=spec,
        r_tag=r_tag,
        z_top_local=z_top_local,
        sigma=sigma_shape,
        seed=seed_shape,
        clip=3.0,
        mode="down_axis",  # TAG excavation direction
        smooth_radial=True,
    )

    # --- Fit base and pert on the SAME sampled points seed ---
    n_n, n_m = 25, 25
    NUM_POINTS = 1000
    seed_points = 1

    base_fit = fit_cylinder_coeffs_from_polyhedron(
        vertices,
        faces,
        DENSITY,
        spec,
        num_points=NUM_POINTS,
        n_n=n_n,
        n_m=n_m,
        seed_points=seed_points,
        parallel=False,
        enforce_B0n=True,
    )
    pert_fit = fit_cylinder_coeffs_from_polyhedron(
        v_pert,
        faces,
        DENSITY,
        spec,
        num_points=NUM_POINTS,
        n_n=n_n,
        n_m=n_m,
        seed_points=seed_points,
        parallel=False,
        enforce_B0n=True,
    )

    # --- 3 plots ONLY ---
    plot1_displacement_map_TAG(
        vertices_base=vertices,
        vertices_pert=v_pert,
        faces=faces,
        mask=mask,
        spec=spec,
        outpath="Images/plot1_TAG_displacements.png",
    )

    plot2_cylinder_and_points(
        vertices=vertices,
        faces=faces,
        spec=spec,
        points=base_fit.points,
        outpath="Images/plot2_cylinder_points.png",
    )

    plot3_power_spectrum(
        base_params=base_fit.params,
        pert_params=pert_fit.params,
        n_n=n_n,
        n_m=n_m,
        outpath="Images/plot3_power_spectrum.png",
    )

    plt.show()
