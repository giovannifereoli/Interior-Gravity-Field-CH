"""
Bennu Pre-TAG / Post-TAG — Cylindrical Harmonic Gravity Fitting (SI units)
==========================================================================
Estimates the mass moved by the OSIRIS-REx TAG event from the *difference*
of cylindrical-harmonic gravity coefficients fitted before and after TAG,
and validates the gravimetric estimate against the geometric ground truth
(the DTM height change itself).

Input data
----------
`Bennu_preTag.obj` / `Bennu_afterTag.obj` are local DTM patches of the
Nightingale TAG site (~44 m x 44 m, heights 0..9.5 m, y-up, metres).
They are scan products: full of holes, with a detached bottom plate, and
with slightly different extents — NOT directly usable as closed polyhedra.

Pipeline (all quantities SI: m, kg, s)
--------------------------------------
1.  Extract the terrain sheet from each OBJ, rotate to z-up.
2.  Rasterise both onto ONE common (x, y) grid -> height maps h_pre, h_post.
    A common grid guarantees a common frame: walls/bottom cancel exactly in
    the pre/post difference.  (Per-mesh centring/scaling would corrupt Δg.)
3.  Rebuild each state as a watertight "slab" polyhedron (top = terrain,
    bottom z=0, side walls) -> valid input for the Tsoulis/Werner method.
4.  Locate the TAG crater from Δh = h_post − h_pre; centre the analysis
    cylinder there.  The harmonic expansion plane (the "sheet") is placed
    at the mean pre-TAG surface height inside the footprint.  Field points
    descend all the way to the LOCAL terrain surface (small clearance,
    point-by-point) — this maximises the observability of the high-k
    modes; inside the crater bowl points may sit slightly below the sheet
    plane, which is fine because the local Δ sources are still below them.
5.  Evaluate polyhedral gravity (U [m²/s²], g [m/s²]) at identical field
    points for both states  (polyhedral_gravity: U > 0, g = +∇U, verified).
6.  Weighted least-squares fit of the cylindrical-harmonic coefficients
    for each state (identical design matrix & weights) -> ΔA = A_post−A_pre.
    Since LS is linear, this equals fitting the difference field directly:
    every static source cancels exactly in Δc.  This is WHY only the local
    DTM patch needs to be meshed even though the gravity inside the
    cylinder is really dominated by the whole ~490 m asteroid: the TAG
    event changed only the local site, so the entire unchanged bulk of
    Bennu — inside the cylinder, outside it, or far away — contributes
    IDENTICALLY to U_pre and U_post and vanishes in the difference.  The
    weights are therefore built from the DIFFERENCE field (post − pre),
    not the absolute pre field, because the absolute field is set by the
    unchanged background (Bennu's gravity gradient across the site dwarfs
    the local TAG signal) and would otherwise leak a few-% background-
    dependent bias into ΔM (verified: absolute-field weights drift ΔM by
    ~7 % as the modelled bulk deepens 0→500 m; difference-field weights
    give an identical ΔM at every depth).  The fit is regularised by
    truncated SVD (`cond`).  Caveat: this cancellation assumes the local
    Δh map captures ALL the mass that moved — ejecta that landed beyond
    the meshed patch, or mass redistributed to ρ > R*, is not counted.

    Validated against the geometric ground truth (seeds 1–3, cond
    1e-3…1e-5, surface-hugging points with 0.5 m clearance):
        (m,n) = (5,6)  → ΔM ratio ≈ 0.89–0.91
        (m,n) = (6,8)  → ΔM ratio ≈ 0.93–0.97   (default)
        (m,n) = (8,10) → degrades to ≈ 0.78–0.88 (unobservable modes)
    The residual few-% deficit is bandlimit truncation plus the thin-sheet
    approximation (real sources spread ±1 m about the sheet plane).
7.  Wahr-like inversion of ΔA -> ΔM [kg] and Δσ(ρ,φ) [kg/m²].
8.  Ground truth from geometry: ΔM_true = ρ_bulk ∫∫ Δh dA (footprint),
    Δσ_true = ρ_bulk Δh — direct validation of the inversion.

Formulae
--------
Basis (solves Laplace's eq. for sources below the sheet plane z = z0):
    U(ρ,φ,z) = Σ_{m,n} J_m(k_mn ρ) exp(−k_mn (z−z0)) [A_mn cos mφ + B_mn sin mφ]
    k_mn = j_{m,n} / (α R*),   α > 1  (Dirichlet zeros pushed to α R*)

Gradients (g = +∇U, attraction convention, matches polyhedral_gravity):
    g_ρ = ∂U/∂ρ = Σ k_mn J'_m(k_mn ρ) e^{−k_mn(z−z0)} [A cos + B sin]
    g_φ = (1/ρ) ∂U/∂φ = Σ (m/ρ) J_m e^{−k_mn(z−z0)} [−A sin + B cos]
    g_z = ∂U/∂z = Σ (−k_mn) J_m e^{−k_mn(z−z0)} [A cos + B sin]

Thin-sheet (Wahr-like) inversion — a surface-density mode
σ_mn J_m(k ρ) e^{imφ} on the plane z = z0 generates, for z > z0,
    U = (2πG σ_mn / k) J_m(kρ) e^{imφ} e^{−k(z−z0)}
        ⇒  σ_mn = k_mn A_mn / (2πG)
so
    Δσ(ρ,φ) = 1/(2πG α R*) Σ_{m,n} j_{mn} J_m(k_mn ρ)[ΔA cos mφ + ΔB sin mφ]   [kg/m²]
    ΔM(ρ<R*) = ∫Δσ dA = (R*/G) Σ_n J_1(j_{0n}/α) ΔA_{0n}                        [kg]
(using ∫_0^{R*} J_0(k ρ) ρ dρ = (R*/k) J_1(k R*);  only m=0 survives ∫dφ).
Both formulae are dimensionally consistent in SI.
"""

import numpy as np
import trimesh
from scipy.special import jv as BesselJ, jn_zeros
from scipy.linalg import lstsq
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
import time, os

# ── physical constants (SI) ────────────────────────────────────────────────
G_SI = 6.67430e-11  # m³ kg⁻¹ s⁻²
RHO_BULK = 1190.0  # kg/m³  — Bennu bulk density (Lauretta et al. 2019)
UGAL = 1.0e-8  # 1 µGal = 1e-8 m/s²

COLOR_PALETTE = ["#d7191c", "#fdae61", "#2c7bb6", "#1a9641", "#762a83", "#e66101"]
mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=COLOR_PALETTE),
    }
)

SEP = "=" * 65
DASH = "─" * 65


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — DTM UTILITIES  (terrain extraction, common grid, slab rebuild)
# ═══════════════════════════════════════════════════════════════════════════


def load_terrain_points(path: str) -> np.ndarray:
    """
    Load a TAG-site OBJ and return the terrain-sheet vertices in a z-up
    working frame [m].

    The raw OBJs contain the terrain sheet, a detached flat bottom plate
    (area ~1920 m² at y=0) and thousands of small scan fragments; only the
    largest-area component (the terrain sheet) is kept.

    Frame change  (x, y, z)_obj -> (x, −z, y):  y-up  ->  z-up
    (equivalent to the +90° rotation about X used previously, but applied
    identically to both meshes — NO per-mesh centring or scaling, so pre
    and post stay in the SAME metric frame).
    """
    mesh = trimesh.load(path, force="mesh")
    comps = mesh.split(only_watertight=False)
    terrain = comps[np.argmax([c.area for c in comps])]
    V = np.asarray(terrain.vertices, dtype=float)
    return np.column_stack([V[:, 0], -V[:, 2], V[:, 1]])


def common_grid(P_pre, P_post, grid_res=0.30, edge_margin=0.5):
    """
    Regular (x, y) grid over the intersection of both footprints [m].

    Returns gx (nx,), gy (ny,), GX, GY (nx, ny).
    """
    lo = np.maximum(P_pre[:, :2].min(axis=0), P_post[:, :2].min(axis=0)) + edge_margin
    hi = np.minimum(P_pre[:, :2].max(axis=0), P_post[:, :2].max(axis=0)) - edge_margin
    nx = int(round((hi[0] - lo[0]) / grid_res)) + 1
    ny = int(round((hi[1] - lo[1]) / grid_res)) + 1
    gx = np.linspace(lo[0], hi[0], nx)
    gy = np.linspace(lo[1], hi[1], ny)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    return gx, gy, GX, GY


def height_map(P, GX, GY, h_min=0.05):
    """
    Rasterise scattered terrain vertices to a height map h(x, y) [m].
    Linear interpolation on the Delaunay triangulation fills the scan
    holes; nearest-neighbour fills the (rare) hull gaps at the edges.
    Heights are clipped to h_min > 0 so the slab never degenerates.
    """
    h = LinearNDInterpolator(P[:, :2], P[:, 2])(GX, GY)
    bad = np.isnan(h)
    if bad.any():
        h[bad] = NearestNDInterpolator(P[:, :2], P[:, 2])(GX[bad], GY[bad])
    return np.maximum(h, h_min)


def build_slab_mesh(h, gx, gy) -> trimesh.Trimesh:
    """
    Watertight 'slab' solid from a height map: top surface z = h(x,y),
    flat bottom z = 0, vertical side walls.  Both states are built on the
    SAME grid, so bottom/walls cancel exactly in the pre/post difference.
    """
    nx, ny = h.shape
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    top = np.column_stack([GX.ravel(), GY.ravel(), h.ravel()])
    bot = np.column_stack([GX.ravel(), GY.ravel(), np.zeros(nx * ny)])
    V = np.vstack([top, bot])
    idx = np.arange(nx * ny).reshape(nx, ny)
    N = nx * ny

    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    top_f = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    bot_f = top_f[:, ::-1] + N

    def wall(strip):
        t0, t1 = strip[:-1], strip[1:]
        b0, b1 = t0 + N, t1 + N
        return np.vstack([np.column_stack([t0, b0, b1]), np.column_stack([t0, b1, t1])])

    walls = np.vstack(
        [wall(idx[:, 0]), wall(idx[:, -1]), wall(idx[0, :]), wall(idx[-1, :])]
    )
    mesh = trimesh.Trimesh(V, np.vstack([top_f, bot_f, walls]), process=True)
    trimesh.repair.fix_normals(mesh)  # consistent outward winding, volume > 0
    assert mesh.is_watertight, "slab construction failed to close"
    assert mesh.volume > 0, "slab has negative volume (winding)"
    return mesh


def locate_tag_site(dh, GX, GY, margin=3.0):
    """
    TAG-site centre = |Δh|-weighted centroid of the excavated (Δh < 0)
    region, excluding a border strip of `margin` [m] (edge noise).
    """
    inner = (
        (GX > GX.min() + margin)
        & (GX < GX.max() - margin)
        & (GY > GY.min() + margin)
        & (GY < GY.max() - margin)
    )
    w = np.where(inner, np.clip(-dh, 0.0, None), 0.0)
    cx = (GX * w).sum() / w.sum()
    cy = (GY * w).sum() / w.sum()
    return cx, cy


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — POLYHEDRAL GRAVITY  (Tsoulis / Werner analytic method)
# ═══════════════════════════════════════════════════════════════════════════


def make_evaluable(mesh: trimesh.Trimesh, density: float) -> GravityEvaluable:
    """
    GravityEvaluable from a watertight z-up slab mesh [m] + density [kg/m³].
    Integrity check is disabled because `build_slab_mesh` already
    guarantees a closed, consistently outward-wound polyhedron.

    Sign convention (verified against a unit cube): U > 0 and g = +∇U
    (attraction, g_z < 0 above the body) — the same convention as the
    harmonic basis and the thin-sheet inversion below.
    """
    poly = Polyhedron(
        polyhedral_source=(
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.faces, dtype=int),
        ),
        density=density,
        integrity_check=PolyhedronIntegrity.DISABLE,
    )
    return GravityEvaluable(poly)


def eval_gravity(evaluable, field_pts):
    """
    Potential and acceleration at each field point (batch, threaded).

    Returns U [m²/s²], gx, gy, gz [m/s²]  — each (N,).
    """
    results = evaluable(computation_points=np.asarray(field_pts), parallel=True)
    U = np.array([r[0] for r in results])
    g = np.array([r[1] for r in results])
    return U, g[:, 0], g[:, 1], g[:, 2]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — FIELD POINT GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def make_cylinder_field_points(
    center_xy,
    z_sheet,
    R_star,
    H,
    h_itp,
    clearance=0.5,
    z_bias=2.0,
    N=2000,
    seed=1,
):
    """
    N random points inside the cylinder ρ < R_star, axis through
    `center_xy`.  Each point descends all the way to the LOCAL terrain
    surface (`clearance` above it) — the local Δ sources stay below every
    point, and the high-k modes remain observable.  Heights are strongly
    biased toward low altitude (u^z_bias) where the high-k signal lives.

    Returns
    -------
    rp, pp, zp : cylindrical coords about the axis; zp is height ABOVE the
                 sheet plane (this is the z that enters exp(−k z); inside
                 the crater bowl zp may be slightly negative)         [m]
    pts_cart   : (N, 3) absolute Cartesian coordinates                [m]
    """
    rng = np.random.default_rng(seed)
    rp = np.sqrt(rng.uniform(0.0, R_star**2, N))  # uniform over disk area
    pp = rng.uniform(0.0, 2.0 * np.pi, N)
    X = rp * np.cos(pp) + center_xy[0]
    Y = rp * np.sin(pp) + center_xy[1]
    z_floor = h_itp(np.column_stack([X, Y])) + clearance - z_sheet
    zp = z_floor + (H - z_floor) * rng.uniform(0.0, 1.0, N) ** z_bias
    pts_cart = np.column_stack([X, Y, zp + z_sheet])
    return rp, pp, zp, pts_cart


def cart_to_cyl_g(gx, gy, phi_pts):
    """Cartesian (gx, gy) → cylindrical (gρ, gφ) about the cylinder axis."""
    gr = gx * np.cos(phi_pts) + gy * np.sin(phi_pts)
    gphi = -gx * np.sin(phi_pts) + gy * np.cos(phi_pts)
    return gr, gphi


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — DESIGN MATRIX & WEIGHTED LEAST SQUARES
# ═══════════════════════════════════════════════════════════════════════════

# Validated bandlimit target: k_max·R* for the (α=2, m_max=6, n_max=8) config
# that was checked against geometric ground truth (ΔM ratio ≈ 0.96, module
# docstring).  k_mn = j_{m,n}/(α R*) — α only sets WHERE the fictitious
# Dirichlet boundary sits; n_max sets the highest wavenumber (resolution)
# reachable at that α.  Raising α without raising n_max shrinks k_max
# proportionally and the basis loses the ability to represent the crater
# at all (empirically: α=100 with n_max=8 gives k_max·R* ≈ 0.3, i.e. a
# shortest representable wavelength of ~340 m against a ~16 m crater —
# the fit aliases noise into a wrong answer, ΔM ratio ≈ +1.25, not merely
# "worse"). This is NOT a bug: it is the Bessel-series analogue of a
# Nyquist limit. To use a large α "because physics wants the boundary far
# away", n_max must grow ~linearly with α to keep the same resolution.
KR_TARGET_DEFAULT = 12.0


def required_n_max(alpha, R_star, m_max, k_target_R=KR_TARGET_DEFAULT, margin=2):
    """
    Smallest n_max such that the highest retained mode (m = m_max−1) still
    reaches k_max·R* ≥ k_target_R at the given α — i.e. the n_max needed
    to preserve spatial resolution when α is increased.  Uses the McMahon
    large-n asymptotic j_{m,n} ≈ (n + m/2 − 1/4)π to guess, then verifies
    exactly with `jn_zeros` (cheap: a handful of extra evaluations).
    """
    m = m_max - 1
    target_j = k_target_R * alpha
    n = max(1, int(np.ceil(target_j / np.pi - m / 2 + 0.25)) + margin)
    while jn_zeros(m, n)[-1] < target_j:
        n += max(5, n // 10)
    return n


def build_design_matrix(rho_pts, phi_pts, z_pts, R_alpha, m_max, n_max):
    """
    (4N × 2·m_max·n_max) design matrix for the simultaneous fit of
    U, gρ, gφ, gz at each field point.  z_pts are heights above the
    sheet plane.  Column order: col 2*(m*n_max + n−1) → A_mn,  +1 → B_mn.
    Row order per point i: 4i → U, 4i+1 → gρ, 4i+2 → gφ, 4i+3 → gz.

    Note: for m = 0 the B (sine) columns are identically zero; `lstsq`
    (SVD, minimum-norm) returns B_0n = 0 for them.
    """
    zeros_dict = {m: jn_zeros(m, n_max) for m in range(m_max)}
    N = len(rho_pts)
    A = np.zeros((4 * N, 2 * m_max * n_max))
    rs = np.maximum(rho_pts, 1e-9)  # guard ρ = 0 in the gφ row

    for m in range(m_max):
        cp, sp = np.cos(m * phi_pts), np.sin(m * phi_pts)
        for n in range(1, n_max + 1):
            kmn = zeros_dict[m][n - 1] / R_alpha
            x = kmn * rho_pts
            Ez = np.exp(-kmn * z_pts)
            Jm = BesselJ(m, x)
            dJm = 0.5 * (BesselJ(m - 1, x) - BesselJ(m + 1, x))
            col = 2 * (m * n_max + (n - 1))

            A[0::4, col] = Jm * Ez * cp  # U
            A[0::4, col + 1] = Jm * Ez * sp
            A[1::4, col] = kmn * dJm * Ez * cp  # gρ
            A[1::4, col + 1] = kmn * dJm * Ez * sp
            A[2::4, col] = -m / rs * Jm * Ez * sp  # gφ
            A[2::4, col + 1] = m / rs * Jm * Ez * cp
            A[3::4, col] = -kmn * Jm * Ez * cp  # gz
            A[3::4, col + 1] = -kmn * Jm * Ez * sp

    return A, zeros_dict


def assemble_obs_vector(U, gr, gphi, gz):
    """Interleave [U, gρ, gφ, gz] per point into a (4N,) vector."""
    b = np.zeros(4 * len(U))
    b[0::4], b[1::4], b[2::4], b[3::4] = U, gr, gphi, gz
    return b


def make_weights(U, gr, gphi, gz, U2=None, gr2=None, gphi2=None, gz2=None):
    """
    Per-observable weights w = 1/RMS for the mixed-unit LS problem (U is
    m²/s², g is m/s² — unweighted mixing would make the solution depend on
    the unit system).  The SAME weight vector is used for the pre and post
    fits so that Δc = c_post − c_pre stays meaningful.

    If the post-TAG fields (U2, gr2, ...) are supplied, the weights are
    built from the DIFFERENCE field (post − pre) rather than the absolute
    pre-TAG field.  This matters physically: the field inside the cylinder
    is dominated by the WHOLE asteroid (Bennu is ~490 m across, not 16 m),
    whose gravity gradient over the TAG site is far larger than the local
    TAG signal — so absolute-field weights are set by unchanged background
    mass, not by the change we are trying to measure, and the recovered ΔM
    then drifts by a few % depending on how much of Bennu is meshed.  The
    difference field cancels every static source exactly (verified: ΔU is
    bit-identical whether the slab bottom sits at 0 m or −500 m), so
    difference-field weights make ΔM INVARIANT to all mass outside that
    did not move — which is the correct behaviour for a change detector.
    """
    rms = lambda v: np.sqrt(np.mean(v**2)) + 1e-30
    if U2 is not None:  # difference-field weights (background-invariant)
        U, gr, gphi, gz = U2 - U, gr2 - gr, gphi2 - gphi, gz2 - gz
    W = np.zeros(4 * len(U))
    W[0::4] = 1.0 / rms(U)
    W[1::4] = 1.0 / rms(gr)
    W[2::4] = 1.0 / rms(gphi)
    W[3::4] = 1.0 / rms(gz)
    return W


def fit_coefficients(A_des, U, gr, gphi, gz, W, cond=1e-3):
    """
    Weighted, truncated-SVD-regularised LS:  (W A) c ≈ (W b).
    Singular values below cond·s_max are discarded (they correspond to
    mode combinations the field-point geometry cannot observe; without
    the cutoff their coefficients are noise amplified by 1/s).
    Returns coeffs, weighted RMS, weighted relative RMS.
    """
    b = assemble_obs_vector(U, gr, gphi, gz)
    coeffs, _, _, _ = lstsq(A_des * W[:, None], b * W, cond=cond)
    resid = (A_des @ coeffs - b) * W
    rms = np.sqrt(np.mean(resid**2))
    rel = rms / (np.sqrt(np.mean((b * W) ** 2)) + 1e-30)
    return coeffs, rms, rel


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — WAHR-LIKE THIN-SHEET INVERSION  (ΔM [kg], Δσ [kg/m²])
# ═══════════════════════════════════════════════════════════════════════════


def wahr_invert(
    delta_coeffs, R_star, alpha, n_max, m_max, zeros_dict, n_rho=80, n_phi=120
):
    """
    Invert ΔA = A_post − A_pre for the mass change and the surface-density
    change map, assuming the mass change is a thin sheet on the expansion
    plane z = z_sheet (see module docstring for the derivation):

        Δσ(ρ,φ)  = 1/(2πG αR*) Σ_{m,n} j_mn J_m(k_mn ρ)[ΔA cos mφ + ΔB sin mφ]
        ΔM(ρ<R*) = (R*/G) Σ_n J_1(j_0n/α) ΔA_0n

    ΔM is the integral of Δσ over the cylinder FOOTPRINT (ρ < R*); mass
    change in the buffer annulus R* < ρ < αR* is representable by the
    basis but not counted here.

    Returns delta_M [kg], sigma_map [kg/m²] (n_rho, n_phi), RHO, PHI [m, rad].
    """
    R_alpha = alpha * R_star

    delta_M = 0.0
    for n in range(1, n_max + 1):
        j0n = zeros_dict[0][n - 1]
        col = 2 * (0 * n_max + (n - 1))
        delta_M += (R_star / G_SI) * BesselJ(1, j0n / alpha) * delta_coeffs[col]

    rho_1d = np.linspace(0.02 * R_star, 0.98 * R_star, n_rho)
    phi_1d = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    RHO, PHI = np.meshgrid(rho_1d, phi_1d, indexing="ij")
    sigma_map = np.zeros_like(RHO)
    pref = 1.0 / (2.0 * np.pi * G_SI * R_alpha)

    for m in range(m_max):
        for n in range(1, n_max + 1):
            jmn = zeros_dict[m][n - 1]
            kmn = jmn / R_alpha
            col = 2 * (m * n_max + (n - 1))
            sigma_map += (
                pref
                * jmn
                * BesselJ(m, kmn * RHO)
                * (
                    delta_coeffs[col] * np.cos(m * PHI)
                    + delta_coeffs[col + 1] * np.sin(m * PHI)
                )
            )

    return delta_M, sigma_map, RHO, PHI


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def run_bennu_tag(
    path_pre: str = "3dmeshes/Bennu_preTag.obj",
    path_post: str = "3dmeshes/Bennu_afterTag.obj",
    density: float = RHO_BULK,  # bulk density [kg/m³]
    grid_res: float = 0.30,  # DTM raster resolution [m]
    # Cylinder / basis parameters (SI metres)
    site_center=None,  # (x, y) [m]; None → auto-detect from Δh
    R_star: float = 8.0,  # cylinder radius [m]
    H: float = 16.0,  # cylinder height above sheet [m]
    clearance: float = 0.5,  # local terrain clearance of field points [m]
    alpha: float = 2.0,  # Bessel extension — boundary-placement only (α > 1)
    m_max: int = 6,  # azimuthal orders 0..m_max−1
    n_max=8,  # radial modes 1..n_max; int, or "auto" to derive from α
    N_field: int = 2000,
    seed: int = 1,
    cond: float = 1e-4,  # truncated-SVD cutoff of the weighted LS
    n_ensemble: int = 1,  # independent field-point draws averaged for ΔM
    k_target_R: float = KR_TARGET_DEFAULT,  # bandlimit target for n_max="auto"
    verbose: bool = True,
):
    """
    Full pre/post TAG pipeline in SI units.  Returns a dict with all
    intermediate and final results (see bottom of function).

    `n_ensemble` > 1 repeats field-point generation + gravity + LS fit for
    `n_ensemble` independent seeds (seed, seed+1, ...) and reports ΔM as a
    mean ± standard deviation over the ensemble, instead of a single
    Monte-Carlo draw.  This does NOT remove the systematic bandlimit /
    thin-sheet bias (validated ≈ 4-5 % low against geometric truth at
    m_max=6, n_max=8) — it only replaces a single, possibly lucky/unlucky
    draw with an honest estimate of the fit's Monte-Carlo scatter (± 1-2 %
    at n_ensemble=5).  The returned `sigma_map`/plots use the ensemble-
    averaged coefficients; all other diagnostics come from the first draw.

    On α and n_max: k_mn = j_{m,n}/(α R*) — α only sets where the
    fictitious Dirichlet boundary sits, n_max sets the highest wavenumber
    (spatial resolution) reachable at that α.  Raising α without raising
    n_max shrinks k_max proportionally and silently destroys resolution
    (validated: α=100 with n_max=8 gives a shortest representable
    wavelength of ~340 m against a ~16 m crater and a wrong ΔM, not just a
    noisier one).  Pass `n_max="auto"` to have n_max computed from α so
    that k_max·R* ≥ `k_target_R` is preserved automatically (see
    `required_n_max`); with an explicit int n_max, an under-resolved
    combination raises instead of silently returning garbage.
    """
    if verbose:
        print(SEP)
        print("  BENNU TAG site — Cylindrical Harmonic Mass-Change Estimation")
        print("  (all quantities SI: m, kg, s)")
        print(SEP)

    # ── 1. TERRAIN → COMMON GRID → WATERTIGHT SLABS ────────────────────
    if verbose:
        print("\n[1] Rebuilding watertight slabs from DTM patches …")
    P_pre = load_terrain_points(path_pre)
    P_post = load_terrain_points(path_post)
    gx, gy, GX, GY = common_grid(P_pre, P_post, grid_res=grid_res)
    h_pre = height_map(P_pre, GX, GY)
    h_post = height_map(P_post, GX, GY)
    dh = h_post - h_pre
    dA = (gx[1] - gx[0]) * (gy[1] - gy[0])

    mesh_pre = build_slab_mesh(h_pre, gx, gy)
    mesh_post = build_slab_mesh(h_post, gx, gy)
    if verbose:
        print(f"    grid: {len(gx)} × {len(gy)} @ {grid_res} m")
        print(
            f"    pre : {len(mesh_pre.faces):6d} faces, V = {mesh_pre.volume:9.2f} m³"
        )
        print(
            f"    post: {len(mesh_post.faces):6d} faces, V = {mesh_post.volume:9.2f} m³"
        )
        print(f"    total ΔV (patch)  = {mesh_post.volume - mesh_pre.volume:+8.2f} m³")

    # ── 2. TAG SITE & CYLINDER GEOMETRY ────────────────────────────────
    if site_center is None:
        site_center = locate_tag_site(dh, GX, GY)
    cx, cy = site_center
    r_grid = np.hypot(GX - cx, GY - cy)
    foot = r_grid < R_star
    z_sheet = float(h_pre[foot].mean())  # expansion plane [m]
    R_alpha = alpha * R_star
    h_env = np.maximum(h_pre, h_post)

    # ── α / n_max resolution guard ──────────────────────────────────────
    # k_mn = j_{m,n}/(α R*): α only places the fictitious Dirichlet
    # boundary; n_max sets the highest wavenumber reachable at that α.
    # See `required_n_max` docstring for why this must be checked.
    if isinstance(n_max, str) and n_max.lower() == "auto":
        n_max = required_n_max(alpha, R_star, m_max, k_target_R)
        if verbose:
            print(
                f"\n[*] n_max='auto': α={alpha} → n_max={n_max} "
                f"(targeting k_max·R* ≥ {k_target_R})"
            )
    else:
        n_max = int(n_max)
        k_max_R = jn_zeros(m_max - 1, n_max)[-1] / alpha
        if k_max_R < k_target_R:
            n_needed = required_n_max(alpha, R_star, m_max, k_target_R)
            raise ValueError(
                f"α={alpha}, m_max={m_max}, n_max={n_max} gives k_max·R* = "
                f"{k_max_R:.2f}, below the {k_target_R} needed to resolve "
                f"this crater (~{R_star:.0f} m radius) — the fit would "
                f"alias, not just get noisier (validated failure mode: "
                f"ΔM comes out with the wrong sign/magnitude, not merely "
                f"attenuated). Raising α increases the required n_max "
                f"roughly linearly: use n_max='auto', or set n_max ≥ "
                f"{n_needed} explicitly (cost grows with m_max·n_max)."
            )

    # geometric ground truth (what the inversion should recover)
    dV_foot = float(dh[foot].sum() * dA)
    dM_true = density * dV_foot
    dV_total = float(dh.sum() * dA)
    if verbose:
        print(f"\n[2] TAG site (auto): ({cx:+.2f}, {cy:+.2f}) m")
        print(f"    cylinder R* = {R_star} m, H = {H} m, α = {alpha}, n_max = {n_max}")
        print(f"    sheet plane z0 = {z_sheet:.2f} m")
        print(f"    GROUND TRUTH  ΔV(ρ<R*) = {dV_foot:+.2f} m³")
        print(f"                  ΔM(ρ<R*) = {dM_true:+.4e} kg  (ρ={density} kg/m³)")
        print(f"                  ΔV(patch) = {dV_total:+.2f} m³")

    # ── 3./4./5. FIELD POINTS, GRAVITY, LS FIT — repeated per ensemble ─
    # GravityEvaluable depends only on the mesh, not the field points, so
    # it's built once and reused across the ensemble.
    if verbose:
        print(f"\n[3] Building GravityEvaluable objects …", end=" ", flush=True)
    t0 = time.time()
    ev_pre = make_evaluable(mesh_pre, density)
    ev_post = make_evaluable(mesh_post, density)
    if verbose:
        print(f"done ({time.time()-t0:.1f}s)")

    h_itp = RegularGridInterpolator((gx, gy), h_env)
    n_ens = max(1, n_ensemble)
    dcoeffs_draws, rel_delta_draws = [], []

    if verbose:
        print(
            f"\n[4] Field points + gravity + LS fit "
            f"({n_ens} draw{'s' if n_ens > 1 else ''} × {N_field} pts) …"
        )
    t0 = time.time()
    for i in range(n_ens):
        rp_i, pp_i, zp_i, pts_i = make_cylinder_field_points(
            (cx, cy),
            z_sheet,
            R_star,
            H,
            h_itp,
            clearance=clearance,
            N=N_field,
            seed=seed + i,
        )
        h_under = h_itp(pts_i[:, :2])
        assert (pts_i[:, 2] > h_under).all(), "field points intersect terrain"

        U0, gx0, gy0, gz0 = eval_gravity(ev_pre, pts_i)
        U1, gx1, gy1, gz1 = eval_gravity(ev_post, pts_i)
        gr0, gp0 = cart_to_cyl_g(gx0, gy0, pp_i)
        gr1, gp1 = cart_to_cyl_g(gx1, gy1, pp_i)

        A_i, zd_i = build_design_matrix(rp_i, pp_i, zp_i, R_alpha, m_max, n_max)
        # difference-field weights → ΔM independent of unmeshed rest-of-Bennu
        W_i = make_weights(U0, gr0, gp0, gz0, U1, gr1, gp1, gz1)
        c0, rms0, rel0 = fit_coefficients(A_i, U0, gr0, gp0, gz0, W_i, cond=cond)
        c1, rms1, rel1 = fit_coefficients(A_i, U1, gr1, gp1, gz1, W_i, cond=cond)
        dc_i = c1 - c0
        db_i = assemble_obs_vector(U1 - U0, gr1 - gr0, gp1 - gp0, gz1 - gz0) * W_i
        rel_delta_draws.append(
            np.linalg.norm((A_i * W_i[:, None]) @ dc_i - db_i)
            / (np.linalg.norm(db_i) + 1e-30)
        )
        dcoeffs_draws.append(dc_i)

        if i == 0:  # keep first draw for diagnostics / plotting
            rp, pp, zp, pts_cart = rp_i, pp_i, zp_i, pts_i
            U_pre, gz_pre, U_post, gz_post = U0, gz0, U1, gz1
            gr_pre, gphi_pre, gr_post, gphi_post = gr0, gp0, gr1, gp1
            c_pre, c_post = c0, c1
            rms_pre, rms_post, rel_pre, rel_post = rms0, rms1, rel0, rel1
            A_des, zeros_dict = A_i, zd_i

    if verbose:
        print(f"    done ({time.time()-t0:.1f}s)")
        print(f"    field z above sheet ∈ [{zp.min():.2f}, {zp.max():.2f}] m")
        z_q10 = np.percentile(zp, 10)
        k_max = jn_zeros(m_max - 1, n_max)[-1] / R_alpha
        print(
            f"    observability k_max·z_q10 = {k_max*z_q10:.2f} "
            f"(keep ≲ 4, else lower n_max)"
        )
        print(f"    U_pre ∈ [{U_pre.min():.3e}, {U_pre.max():.3e}] m²/s²")
        print(f"    gz_pre ∈ [{gz_pre.min():.3e}, {gz_pre.max():.3e}] m/s²")
        print(f"    rel RMS  pre = {rel_pre:.3e},  post = {rel_post:.3e}")
        rel_delta = float(np.mean(rel_delta_draws))
        print(f"    rel RMS  Δ-field fit = {rel_delta:.3e}   <-- quality metric")
    else:
        rel_delta = float(np.mean(rel_delta_draws))

    # ΔM is a LINEAR functional of the m=0 coefficients, so averaging the
    # coefficient vectors first and inverting once is exactly equivalent
    # to averaging ΔM over the ensemble — but also gives one clean Δσ map.
    d_coeffs = np.mean(dcoeffs_draws, axis=0)
    dM_draws = np.array(
        [
            wahr_invert(dc, R_star, alpha, n_max, m_max, zeros_dict, n_rho=2, n_phi=2)[
                0
            ]
            for dc in dcoeffs_draws
        ]
    )
    dM_ens_std = float(dM_draws.std()) if n_ens > 1 else 0.0
    if verbose and n_ens > 1:
        print(
            f"    ensemble ΔM: mean={dM_draws.mean():+.4e} kg, "
            f"std={dM_ens_std:.2e} kg ({100*dM_ens_std/abs(dM_draws.mean()):.1f}%)"
        )

    # ── 6. WAHR INVERSION ──────────────────────────────────────────────
    dM_est, sigma_map, RHO, PHI = wahr_invert(
        d_coeffs, R_star, alpha, n_max, m_max, zeros_dict
    )

    # ── 7. DERIVED QUANTITIES & TRUTH COMPARISON ───────────────────────
    # true surface-density change on the same polar grid
    dh_itp = RegularGridInterpolator((gx, gy), dh, bounds_error=False, fill_value=0.0)
    sigma_true = density * dh_itp(
        np.column_stack(
            [(cx + RHO * np.cos(PHI)).ravel(), (cy + RHO * np.sin(PHI)).ravel()]
        )
    ).reshape(RHO.shape)

    # ── central-peak recovery diagnostic ──────────────────────────────
    # The recovered Δσ is BANDLIMITED: gravity measured at height z above
    # the surface is a low-pass filter (upward continuation), so a sharp
    # central spike is smoothed and its PEAK amplitude is underestimated,
    # even though the INTEGRAL (ΔM) is preserved.  Quantify this at the
    # centre so it is transparent rather than mistaken for a bug.  The
    # effective resolution ≈ the field-point altitude above the sources.
    core = RHO[:, 0] < 1.0
    sigma_peak_rec = float(sigma_map[core].mean())
    sigma_peak_true = float(sigma_true[core].mean())
    sigma_peak_true_pix = density * float(dh[np.hypot(GX - cx, GY - cy) < 1.0].min())
    z_resolution = float(zp.min())  # field-point altitude above expansion plane
    if verbose:
        print(
            f"    Δσ central peak (ρ<1 m): recovered {sigma_peak_rec:+.0f} vs "
            f"true {sigma_peak_true:+.0f} kg/m² "
            f"({sigma_peak_rec/sigma_peak_true:.2f}×; deepest pixel "
            f"{sigma_peak_true_pix:+.0f})"
        )
        print(
            f"    → the {1-sigma_peak_rec/sigma_peak_true:.0%} peak deficit is the "
            f"gravity low-pass at z≳{max(z_resolution,0.3):.1f} m, not an error; "
            f"ΔM (the integral) is unaffected"
        )

    V_cyl = np.pi * R_star**2 * H
    delta_rho = dM_est / V_cyl  # effective density change in cylinder [kg/m³]
    dh_equiv = dM_est / (density * np.pi * R_star**2)  # mean elevation change [m]

    dU = U_post - U_pre
    dgz = gz_post - gz_pre
    sig_ratio = np.std(dU) / (np.sqrt(np.mean(U_pre**2)) + 1e-30)
    coeff_ratio = np.linalg.norm(d_coeffs) / (np.linalg.norm(c_pre) + 1e-30)

    if verbose:
        print(f"\n{DASH}\n  RESULTS (SI)\n{DASH}")
        if n_ens > 1:
            print(
                f"  ΔM  gravimetric       = {dM_est:+.4e} ± {dM_ens_std:.1e} kg  "
                f"(n_ensemble={n_ens})"
            )
        else:
            print(f"  ΔM  gravimetric       = {dM_est:+.4e} kg")
        print(f"  ΔM  geometric truth   = {dM_true:+.4e} kg   (ρ·∫Δh dA, ρ<R*)")
        print(f"  recovery ratio        = {dM_est / dM_true:8.3f}")
        print(f"  ΔV  equivalent        = {dM_est/density:+.2f} m³")
        print(f"  mean Δh over footprint= {dh_equiv:+.4f} m")
        print(
            f"  Δρ_eff (ΔM/V_cyl)     = {delta_rho:+.4f} kg/m³  (V_cyl={V_cyl:.0f} m³)"
        )
        print(f"  Δgz RMS               = {np.std(dgz)/UGAL:.2f} µGal")
        print(f"  ΔU/U                  = {sig_ratio:.3e}")
        print(f"  ||Δc||/||c||          = {coeff_ratio:.3e}")
        print(DASH)

    return dict(
        # geometry / DTM
        mesh_pre=mesh_pre,
        mesh_post=mesh_post,
        gx=gx,
        gy=gy,
        h_pre=h_pre,
        h_post=h_post,
        dh=dh,
        density=density,
        # cylinder
        cx=cx,
        cy=cy,
        z_sheet=z_sheet,
        R_star=R_star,
        H=H,
        alpha=alpha,
        m_max=m_max,
        n_max=n_max,
        R_alpha=R_alpha,
        # field points
        rp=rp,
        pp=pp,
        zp=zp,
        pts_cart=pts_cart,
        # gravity
        U_pre=U_pre,
        U_post=U_post,
        gz_pre=gz_pre,
        gz_post=gz_post,
        gr_pre=gr_pre,
        gphi_pre=gphi_pre,
        gr_post=gr_post,
        gphi_post=gphi_post,
        dU=dU,
        dgz=dgz,
        # fit
        design_matrix=A_des,  # first-draw design matrix (introspection only)
        zeros_dict=zeros_dict,
        c_pre=c_pre,
        c_post=c_post,
        d_coeffs=d_coeffs,
        rms_pre=rms_pre,
        rms_post=rms_post,
        rel_pre=rel_pre,
        rel_post=rel_post,
        rel_delta=rel_delta,
        # ensemble (n_ensemble=1 → dM_ens_std=0, dM_draws is a 1-element array)
        n_ensemble=n_ens,
        dM_draws=dM_draws,
        dM_ens_std=dM_ens_std,
        # inversion + truth
        dM_est=dM_est,
        dM_true=dM_true,
        dV_foot=dV_foot,
        dV_total=dV_total,
        sigma_map=sigma_map,
        sigma_true=sigma_true,
        sigma_peak_rec=sigma_peak_rec,
        sigma_peak_true=sigma_peak_true,
        sigma_peak_true_pix=sigma_peak_true_pix,
        z_resolution=z_resolution,
        RHO=RHO,
        PHI=PHI,
        # derived
        V_cyl=V_cyl,
        delta_rho=delta_rho,
        dh_equiv=dh_equiv,
        sig_ratio=sig_ratio,
        coeff_ratio=coeff_ratio,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — DIAGNOSTIC PLOTS  (SI units)
# ═══════════════════════════════════════════════════════════════════════════


def plot_results(res, outdir=None):
    """Three figures: geometry, gravity change, recovered vs true Δσ."""
    R, H = res["R_star"], res["H"]
    cx, cy, z0 = res["cx"], res["cy"], res["z_sheet"]
    gx, gy = res["gx"], res["gy"]
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    rp, pp, zp = res["rp"], res["pp"], res["zp"]
    RHO, PHI = res["RHO"], res["PHI"]
    Xp, Yp = RHO * np.cos(PHI), RHO * np.sin(PHI)
    sm, st = res["sigma_map"], res["sigma_true"]
    dU, dgz = res["dU"], res["dgz"]

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    def _nice_3d_axes(ax):
        ax.set_facecolor("white")
        ax.grid(False)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor("white")
        ax.tick_params(labelsize=8)
        ax.set_xlabel("$x$ [m]", fontsize=8)
        ax.set_ylabel("$y$ [m]", fontsize=8)
        ax.set_zlabel("$z$ [m]", fontsize=8)
        ax.view_init(elev=28, azim=-60)

    def _draw_cylinder(ax, color="tab:cyan"):
        th = np.linspace(0, 2 * np.pi, 120)
        for z in [z0, z0 + H]:
            ax.plot(cx + R * np.cos(th), cy + R * np.sin(th), z, color=color, lw=1.4)
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            ax.plot(
                [cx + R * np.cos(a)] * 2,
                [cy + R * np.sin(a)] * 2,
                [z0, z0 + H],
                color=color,
                lw=0.7,
                alpha=0.6,
            )

    # ── FIGURE 1 — geometry ────────────────────────────────────────────
    fig1 = plt.figure(figsize=(18, 6))
    gs1 = GridSpec(1, 3, figure=fig1, wspace=0.15)
    hmin = min(res["h_pre"].min(), res["h_post"].min())
    hmax = max(res["h_pre"].max(), res["h_post"].max())

    for col, (ttl, h) in enumerate(
        [("PRE-tag terrain", res["h_pre"]), ("POST-tag terrain", res["h_post"])]
    ):
        ax = fig1.add_subplot(gs1[col], projection="3d")
        _nice_3d_axes(ax)
        ax.plot_surface(
            GX,
            GY,
            h,
            cmap="viridis",
            vmin=hmin,
            vmax=hmax,
            rstride=2,
            cstride=2,
            linewidth=0,
            antialiased=False,
            alpha=0.95,
        )
        _draw_cylinder(ax)
        ax.set_zlim(0, z0 + H)
        ax.set_title(ttl, fontsize=11, fontweight="bold")

    ax = fig1.add_subplot(gs1[2], projection="3d")
    _nice_3d_axes(ax)
    ax.plot_surface(
        GX,
        GY,
        res["h_pre"],
        color="0.75",
        rstride=3,
        cstride=3,
        linewidth=0,
        antialiased=False,
        alpha=0.45,
    )
    _draw_cylinder(ax)
    mag = np.abs(res["gz_pre"]) / UGAL / 1e3  # mGal
    lo, hi = np.percentile(mag, [2, 98])
    sc = ax.scatter(
        rp * np.cos(pp) + cx,
        rp * np.sin(pp) + cy,
        zp + z0,
        c=mag,
        cmap="plasma",
        s=8,
        vmin=lo,
        vmax=hi,
        edgecolors="none",
    )
    fig1.colorbar(sc, ax=ax, pad=0.08, shrink=0.7, label=r"$|g_z|$ pre-tag [mGal]")
    ax.set_zlim(0, z0 + H)
    ax.set_title("Field points in cylinder", fontsize=11, fontweight="bold")

    fig1.suptitle(
        f"Bennu TAG-site measurement geometry — cylinder R* = {R:.1f} m, "
        f"H = {H:.1f} m, sheet z₀ = {z0:.2f} m, N = {len(rp)}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    if outdir:
        fig1.savefig(
            os.path.join(outdir, "fig1_geometry.png"), dpi=200, bbox_inches="tight"
        )

    # ── FIGURE 2 — gravity change ──────────────────────────────────────
    fig2, axes2 = plt.subplots(
        2, 3, figsize=(16, 10), gridspec_kw={"hspace": 0.42, "wspace": 0.35}
    )
    fig2.suptitle(
        "Gravity at field points — pre vs post TAG (SI)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    def _scatter_cyl(ax, vals, title, cmap, label, symmetric=False):
        if symmetric:
            vmax = np.percentile(np.abs(vals), 98)
            vmin = -vmax
        else:
            vmin, vmax = np.percentile(vals, [2, 98])
        sc = ax.scatter(
            rp, zp, c=vals, cmap=cmap, s=13, alpha=0.9, lw=0, vmin=vmin, vmax=vmax
        )
        fig2.colorbar(sc, ax=ax, pad=0.02).set_label(label, fontsize=9)
        ax.set_xlabel(r"$\rho$ [m]")
        ax.set_ylabel("z above sheet [m]")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_facecolor("#fafafa")
        ax.grid(True, alpha=0.25)

    _scatter_cyl(axes2[0, 0], res["U_pre"], "U — PRE", "viridis", "U [m²/s²]")
    _scatter_cyl(axes2[0, 1], res["U_post"], "U — POST", "viridis", "U [m²/s²]")
    _scatter_cyl(axes2[0, 2], dU, r"$\Delta U$", "RdBu_r", "ΔU [m²/s²]", symmetric=True)
    _scatter_cyl(
        axes2[1, 0],
        res["gz_pre"] / UGAL / 1e3,
        r"$g_z$ — PRE",
        "viridis",
        "gz [mGal]",
    )
    _scatter_cyl(
        axes2[1, 1],
        res["gz_post"] / UGAL / 1e3,
        r"$g_z$ — POST",
        "viridis",
        "gz [mGal]",
    )
    _scatter_cyl(
        axes2[1, 2],
        dgz / UGAL,
        r"$\Delta g_z$",
        "RdBu_r",
        "Δgz [µGal]",
        symmetric=True,
    )

    txt = (
        f"RMS ΔU  = {np.std(dU):.2e} m²/s²\n"
        f"RMS Δgz = {np.std(dgz)/UGAL:.1f} µGal\n"
        f"ΔU/U    = {res['sig_ratio']:.2e}"
    )
    axes2[1, 2].text(
        0.03,
        0.97,
        txt,
        transform=axes2[1, 2].transAxes,
        fontsize=8.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, ec="0.7"),
    )
    if outdir:
        fig2.savefig(
            os.path.join(outdir, "fig2_gravity_change.png"),
            dpi=200,
            bbox_inches="tight",
        )

    # ── FIGURE 3 — recovered vs true mass change ───────────────────────
    fig3 = plt.figure(figsize=(17, 11))
    gs3 = GridSpec(2, 3, figure=fig3, hspace=0.48, wspace=0.42)
    dM_unc = f" ± {res['dM_ens_std']:.1e}" if res.get("n_ensemble", 1) > 1 else ""
    fig3.suptitle(
        f"Mass-change recovery — ΔM = {res['dM_est']:+.3e}{dM_unc} kg "
        f"(truth {res['dM_true']:+.3e} kg, ratio {res['dM_est']/res['dM_true']:.2f})"
        f"  |  Δρ_eff = {res['delta_rho']:+.3f} kg/m³"
        f"  |  α={res['alpha']}, m_max={res['m_max']}, n_max={res['n_max']}",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    vmax = max(np.percentile(np.abs(sm), 98), np.percentile(np.abs(st), 98))
    tc = np.linspace(0, 2 * np.pi, 200)

    # close the azimuthal seam for pcolormesh (φ wraps at 2π)
    def _wrap(a):
        return np.column_stack([a, a[:, :1]])

    Xw, Yw = _wrap(Xp), _wrap(Yp)
    for k, (mp, ttl) in enumerate(
        [
            (sm, r"Recovered $\Delta\sigma$ (bandlimited)"),
            (st, r"True $\rho\,\Delta h$"),
        ]
    ):
        ax = fig3.add_subplot(gs3[0, k])
        c = ax.pcolormesh(
            Xw,
            Yw,
            _wrap(mp)[:-1, :-1],
            cmap="RdBu_r",
            shading="flat",
            vmin=-vmax,
            vmax=vmax,
        )
        fig3.colorbar(c, ax=ax, label="Δσ [kg/m²]")
        ax.plot(R * np.cos(tc), R * np.sin(tc), "k--", lw=1.2, alpha=0.65)
        ax.set_aspect("equal")
        ax.set_xlabel("x − x₀ [m]")
        ax.set_ylabel("y − y₀ [m]")
        ax.set_title(ttl, fontweight="bold")

    ax3C = fig3.add_subplot(gs3[0, 2])
    rho_1d = RHO[:, 0]
    ax3C.plot(rho_1d, sm.mean(axis=1), lw=2.2, label="recovered (azim. mean)")
    ax3C.plot(rho_1d, st.mean(axis=1), lw=2.2, ls="--", label="true ρΔh (azim. mean)")
    ax3C.fill_between(
        rho_1d, sm.min(axis=1), sm.max(axis=1), alpha=0.18, label="recovered min–max"
    )
    ax3C.axhline(0, color="k", lw=0.8, ls="--")
    ax3C.set_xlabel(r"$\rho$ [m]")
    ax3C.set_ylabel("Δσ [kg/m²]")
    ax3C.set_title("Radial profile", fontweight="bold")
    ax3C.grid(True, alpha=0.25)
    ax3C.legend(fontsize=8, loc="upper right")
    # make the bandlimit peak-smoothing explicit at the centre
    pk_r, pk_t = res.get("sigma_peak_rec"), res.get("sigma_peak_true")
    if pk_r is not None:
        ax3C.text(
            0.03,
            0.04,
            f"peak ρ<1 m: rec {pk_r:+.0f} vs true {pk_t:+.0f} ({pk_r/pk_t:.0%})\n"
            f"deficit is the gravity low-pass, not an error",
            transform=ax3C.transAxes,
            fontsize=7.5,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff6e6", ec="0.6"),
        )

    ax3D = fig3.add_subplot(gs3[1, 0])
    phi_deg = np.degrees(PHI[0, :])
    for frac in [0.25, 0.50, 0.75]:
        i_r = np.argmin(np.abs(rho_1d - frac * R))
        ax3D.plot(phi_deg, sm[i_r, :], lw=2, label=f"ρ = {rho_1d[i_r]:.1f} m")
    ax3D.axhline(0, color="k", lw=0.8, ls="--")
    ax3D.set_xlabel("φ [deg]")
    ax3D.set_ylabel("Δσ [kg/m²]")
    ax3D.set_title("Azimuthal profiles (recovered)", fontweight="bold")
    ax3D.grid(True, alpha=0.25)
    ax3D.legend(fontsize=8)

    ax3E = fig3.add_subplot(gs3[1, 1])
    dc = res["d_coeffs"]
    n_max, m_max = res["n_max"], res["m_max"]
    mode_amp = np.sqrt(dc[0::2] ** 2 + dc[1::2] ** 2)
    x_pos = 0
    tick_pos, tick_lbl = [], []
    for m in range(m_max):
        amps = mode_amp[m * n_max : (m + 1) * n_max]
        xs = np.arange(x_pos, x_pos + n_max)
        ax3E.bar(xs, amps, color=plt.cm.tab10(m / max(m_max - 1, 1)), alpha=0.85)
        tick_pos.append(x_pos + n_max / 2 - 0.5)
        tick_lbl.append(f"m={m}")
        x_pos += n_max + 1
    ax3E.set_xticks(tick_pos)
    ax3E.set_xticklabels(tick_lbl, fontsize=9)
    ax3E.set_ylabel(r"$\sqrt{\Delta A^2+\Delta B^2}$  [m²/s²]")
    ax3E.set_title("ΔA coefficient spectrum", fontweight="bold")
    ax3E.grid(True, axis="y", alpha=0.25)

    ax3F = fig3.add_subplot(gs3[1, 2])
    ax3F.axis("off")
    summary = (
        f"CYLINDER (SI)\n"
        f"  centre  = ({cx:+.2f}, {cy:+.2f}) m\n"
        f"  R*      = {R:.1f} m   H = {H:.1f} m\n"
        f"  sheet z0= {z0:.2f} m\n"
        f"  V_cyl   = {res['V_cyl']:.0f} m³\n\n"
        f"FIT QUALITY (weighted)\n"
        f"  rel RMS Δ-field = {res['rel_delta']:.2e}\n"
        f"  rel RMS pre/post= {res['rel_pre']:.1e}/{res['rel_post']:.1e}\n"
        f"  ||Δc||/||c||    = {res['coeff_ratio']:.2e}\n\n"
        f"MASS CHANGE\n"
        f"  ΔM grav  = {res['dM_est']:+.4e} kg\n"
        f"  ΔM truth = {res['dM_true']:+.4e} kg\n"
        f"  ratio    = {res['dM_est']/res['dM_true']:.3f}\n"
        f"  ΔV grav  = {res['dM_est']/res['density']:+.1f} m³\n"
        f"  mean Δh  = {res['dh_equiv']:+.3f} m\n"
        f"  Δρ_eff   = {res['delta_rho']:+.3f} kg/m³\n\n"
        f"CENTRAL PEAK Δσ (ρ<1 m)\n"
        f"  recovered= {res.get('sigma_peak_rec', float('nan')):+.0f} kg/m²\n"
        f"  true     = {res.get('sigma_peak_true', float('nan')):+.0f} kg/m²  "
        f"({res.get('sigma_peak_rec',1)/res.get('sigma_peak_true',1):.0%})\n"
        f"  (deficit = bandlimit, not error)\n\n"
        f"SIGNAL\n"
        f"  RMS Δgz  = {np.std(dgz)/UGAL:.1f} µGal\n"
        f"  ΔU/U     = {res['sig_ratio']:.2e}"
    )
    ax3F.text(
        0.02,
        0.98,
        summary,
        transform=ax3F.transAxes,
        va="top",
        fontsize=9.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", fc="#f8f8f8", ec="0.75"),
    )
    ax3F.set_title("Summary", fontsize=10, fontweight="bold")

    if outdir:
        fig3.savefig(
            os.path.join(outdir, "fig3_mass_change.png"), dpi=200, bbox_inches="tight"
        )

    plt.show()
    return fig1, fig2, fig3


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    result = run_bennu_tag(
        path_pre="3dmeshes/Bennu_preTag.obj",
        path_post="3dmeshes/Bennu_afterTag.obj",
        density=RHO_BULK,  # [kg/m³]
        grid_res=0.30,  # [m]
        site_center=None,  # auto-detect TAG crater from Δh
        # R_star controls PEAK resolution (k ∝ 1/R*): a smaller cylinder
        # concentrates the basis on the crater and recovers the sharp
        # central Δσ, at a small cost in mass completeness.  Validated
        # peak recovery of the true ρΔh centre (−682 kg/m² over ρ<1 m):
        #   R*=16 → 0.58×  (over-smoothed — the "−400" you saw)
        #   R*=8  → 0.82×  (ΔM ratio ≈ 1.0 — best mass/peak balance)
        #   R*=6  → 0.98×  (ΔM ratio ≈ 0.90 — best peak, ~10% mass cost)
        R_star=8.0,  # [m]  (use 6 to prioritise the peak, 12+ only for mass)
        H=16.0,  # [m]
        clearance=0.5,  # points hug the surface: local terrain + 0.5 m
        alpha=2.0,  # boundary placement only; n_max scales with α ("auto")
        m_max=6,
        n_max="auto",  # → 8 at α=2 (see required_n_max)
        N_field=2000,
        cond=1e-4,  # truncated-SVD regularisation
        verbose=True,
    )

    fig1, fig2, fig3 = plot_results(result, outdir="Images")

    print("\nDone.")
