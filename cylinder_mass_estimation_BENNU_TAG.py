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
    are drawn UNIFORMLY over the vacuum volume of the cylinder — constant
    density per unit volume, with no coordinate deliberately over-sampled
    (see `make_cylinder_field_points`).  Their lower bound follows the
    LOCAL terrain (small clearance, point-by-point); inside the crater
    bowl points may sit slightly below the sheet plane, which is fine
    because the local Δ sources are still below them.
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

    Validated against the geometric ground truth (3 draws, cond 1e-4,
    uniform volume sampling, 0.5 m clearance):
        (m,n) = (5,6)  → ΔM ratio 1.018 ± 0.021
        (m,n) = (6,8)  → ΔM ratio 1.012 ± 0.012   (default)
        (m,n) = (8,10) → collapses to 0.464 ± 0.060 (unobservable modes)
    The (8,10) collapse is the price of an unbiased sample: only ~2 % of a
    uniform draw sits within one e-folding 1/k_max of the sheet, so the
    shortest modes are not observed at all and the SVD cutoff discards
    them.  The earlier altitude-biased sampler (z ∝ u², ~8 % of points
    that low) held (8,10) at ≈ 0.78–0.88, but carried a slightly larger
    ΔM bias at the default truncation: 1.024 ± 0.009 against 1.008 ± 0.019
    over 5 draws, with the per-draw spread halved (0.9 % vs 1.9 %) and the
    formal √Σ_ΔM 0.17 % against 0.29 %.  Unbiased sampling therefore trades
    precision, and the headroom to raise the truncation, for a little less
    bias.  The residual deficit at the default is bandlimit truncation plus
    the thin-sheet approximation (sources spread ±1 m about the sheet).
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

# Okabe-Ito, the colour-vision-deficiency-safe palette used by the GLOBAL
# scripts, in the same role order.  The previous set opened with #d7191c and
# carried #1a9641 at index 3 — red against green, the pair deuteranopes and
# protanopes cannot separate.
COLOR_PALETTE = ["#D55E00", "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9"]
ACCENT = "#882255"  # structural elements, kept clear of the data colours

# Truncated-SVD cutoff of the weighted least squares, the same value the
# GLOBAL scripts use as CH_RCOND.  It was previously a bare 1e-4 at the
# pipeline entry with THREE inner functions defaulting to 1e-3 — calling any of
# them directly (as the covariance path nearly did) silently truncated ten times
# more loosely than the fit it was meant to describe.
CH_COND = 1e-4

USE_TEX = False  # os.environ.get("GLOBAL_NO_TEX", "") == ""

mpl.rcParams.update(
    {
        "axes.prop_cycle": mpl.cycler(color=COLOR_PALETTE),
        # Same switch as the GLOBAL scripts.  Every label here is written to be
        # valid in BOTH modes — maths in $...$, no bare unicode, no % — so
        # flipping it changes only the typeface and the speed.
        "text.usetex": USE_TEX,
        "font.family": "serif" if USE_TEX else "STIXGeneral",
        "mathtext.fontset": "stix",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        # ── journal styling ────────────────────────────────────────────
        # Ticks inward on all four sides with minors, hairline spines, frameless
        # legends, faint grids and 300 dpi output: the conventions AAS/Icarus
        # figures follow.  Titles are NOT bold — a bold title inside a
        # single-column figure competes with the caption below it.
        "axes.linewidth": 0.8,
        "axes.titleweight": "normal",
        "axes.titlepad": 8.0,
        "axes.labelpad": 3.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.6,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.8,
        "lines.markeredgewidth": 0.7,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # editable text in the PDF, not outlines
        "figure.dpi": 110,
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
    N=2000,
    seed=1,
):
    """
    N points drawn UNIFORMLY over the vacuum volume of the cylinder ρ < R_star,
    between the local terrain (+ `clearance`) and the top z = z_sheet + H.

    "Uniform" here means constant density per unit VOLUME — no coordinate is
    deliberately over-sampled:

        ρ = R*·√u     constant density per unit AREA.  (ρ = R*·u would instead
                      pile points onto the axis: it is uniform per unit RADIUS,
                      which is a density gradient, not an unbiased sample.)
        φ ~ U(0, 2π)
        z ~ U(z_lo, H), keeping only points above the LOCAL terrain.

    The rejection step is what makes this uniform rather than merely
    uniform-per-column: the terrain under the footprint varies by several metres
    (~22 % of H for the TAG site), so giving every column the same number of
    points would leave the deep parts of the crater ~1.3× less densely sampled
    than the high ground.  Drawing z over one common range and rejecting what
    falls below the terrain removes that gradient.

    NOTE on observability: the CH modes decay as e^{−k_mn (z−z0)}, so a uniform
    sample spends most of its points at altitudes where the short modes are
    already negligible.  That is the price of an unbiased sample and it is
    deliberate here; `clearance` still sets how close to the surface the lowest
    points may come.

    Returns
    -------
    rp, pp, zp : cylindrical coords about the axis; zp is height ABOVE the
                 sheet plane (this is the z that enters exp(−k z); inside
                 the crater bowl zp may be slightly negative)         [m]
    pts_cart   : (N, 3) absolute Cartesian coordinates                [m]
    """
    rng = np.random.default_rng(seed)

    # Lowest terrain under the footprint: the bottom of the z range that has to
    # be offered so that no part of the vacuum volume is unreachable.
    g_r = np.sqrt(np.linspace(0.0, 1.0, 60)) * R_star
    g_p = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    GR, GP = np.meshgrid(g_r, g_p, indexing="ij")
    z_lo = (
        float(
            h_itp(
                np.column_stack(
                    [
                        (GR * np.cos(GP)).ravel() + center_xy[0],
                        (GR * np.sin(GP)).ravel() + center_xy[1],
                    ]
                )
            ).min()
        )
        + clearance
        - z_sheet
    )

    r_a, p_a, z_a = [], [], []
    n_have = 0
    while n_have < N:
        M = int(max(256, 1.3 * (N - n_have)))  # acceptance is ~90 % here
        r = np.sqrt(rng.uniform(0.0, R_star**2, M))
        ph = rng.uniform(0.0, 2.0 * np.pi, M)
        X = r * np.cos(ph) + center_xy[0]
        Y = r * np.sin(ph) + center_xy[1]
        z = rng.uniform(z_lo, H, M)
        ok = z >= h_itp(np.column_stack([X, Y])) + clearance - z_sheet
        r_a.append(r[ok])
        p_a.append(ph[ok])
        z_a.append(z[ok])
        n_have += int(ok.sum())

    rp = np.concatenate(r_a)[:N]
    pp = np.concatenate(p_a)[:N]
    zp = np.concatenate(z_a)[:N]
    X = rp * np.cos(pp) + center_xy[0]
    Y = rp * np.sin(pp) + center_xy[1]
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


def fit_coefficients(A_des, U, gr, gphi, gz, W, cond=CH_COND):
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
# SECTION 4b — UNCERTAINTY PROPAGATION   Σ_ΔCS  →  Σ_Δσ(ρ,φ),  Σ_ΔM
# ═══════════════════════════════════════════════════════════════════════════
# Both products of the inversion are LINEAR functionals of the differenced
# coefficients ΔCS ∈ R^{N_k},  N_k = 2·m_max·n_max:
#
#     q = f_qᵀ ΔCS          ⇒        Σ_q = f_qᵀ Σ_ΔCS f_q ,
#
# for q ∈ {Δσ(ρ,φ), ΔM}.  Nothing is linearized: unlike the global estimator,
# which has to be expanded about a reference state before its covariance can be
# written, these covariances are EXACT consequences of Σ_ΔCS.  They describe
# dispersion only — the truncation bias, the mass that left the footprint, and
# the thin-sheet idealization are systematic and invisible to them, so a quoted
# √Σ_ΔM is a LOWER BOUND on the total error.  The whole analysis therefore
# reduces to writing down f_Δσ and f_ΔM, which is what this section does.


def projection_matrix(A_des, W, cond=CH_COND):
    """
    The map M with c = M y that `fit_coefficients` actually applies:

        (W Ψ) = U S Vᵀ  →   M = V S⁺ Uᵀ diag(W)     (S⁺ truncated at cond·s_max)

    M is identical for the two epochs (same Ψ, same W), which is exactly what
    lets the epoch difference be taken in coefficient space:
    ΔCS = M y_post − M y_pre = M Δy.  Returns (M, n_kept, s_ratio).
    """
    Aw = A_des * W[:, None]
    Us, S, Vt = np.linalg.svd(Aw, full_matrices=False)
    keep = S > cond * S[0]
    Sp = np.zeros_like(S)
    Sp[keep] = 1.0 / S[keep]
    M = (Vt.T * Sp) @ Us.T * W[None, :]
    return M, int(keep.sum()), float(S[keep][-1] / S[0])


def diff_field_variance(
    W, meas_rel=0.02, y_pre=None, y_post=None, epoch_rel=None, rho_epoch=0.0
):
    """
    Diagonal of Σ_Δy, the covariance of the DIFFERENCED field samples.

    Route A (default — `meas_rel`).  Book the precision on the difference
    directly:  σ_i = meas_rel / W_i.  W is 1/RMS of the Δ-field for each
    observable type, so this reads "the epoch-to-epoch INDEPENDENT error of each
    sample is meas_rel of the RMS of the signal being measured".  It also makes
    the fit's own weights the whitening matrix, W = Σ_Δy^{-1/2}/meas_rel, so the
    information form  Σ_ΔCS = 2(Ψᵀ Σ_y⁻¹ Ψ)⁻¹  applies exactly.

    Route B (`epoch_rel` with `y_pre`, `y_post`).  Book it per epoch on the
    ABSOLUTE field, σ_pre = epoch_rel·RMS(y_pre) per observable type, and remove
    the common-mode part through the cross-covariance:

        Σ_Δy = σ_pre² + σ_post² − 2 ρ σ_pre σ_post ,    ρ = `rho_epoch` ∈ [0,1)

    ρ = 0 is the conservative choice (it can only inflate Σ_Δy); ρ → 1 is the
    statistical counterpart of the background cancellation that makes the local
    method work at all — a shape or frame mismodelling shared by both epochs
    subtracts out of ΔCS exactly as the static field does.
    """
    if epoch_rel is None:
        return (meas_rel / W) ** 2  # σ_Δy booked directly on the difference
    sig = np.zeros_like(W)
    for k in range(4):  # per observable type: U, gρ, gφ, gz
        r_pre = np.sqrt(np.mean(y_pre[k::4] ** 2))
        r_post = np.sqrt(np.mean(y_post[k::4] ** 2))
        sig[k::4] = epoch_rel**2 * (
            r_pre**2 + r_post**2 - 2.0 * rho_epoch * r_pre * r_post
        )
    return sig


def coeff_covariance(M, var_dy):
    """Σ_ΔCS = M Σ_Δy Mᵀ, with Σ_Δy diagonal (vector `var_dy`)."""
    return (M * var_dy[None, :]) @ M.T


def sigma_functional(RHO, PHI, R_alpha, m_max, n_max, zeros_dict):
    """
    F_Δσ ∈ R^{N_g × N_k}: row g is f_Δσ(ρ_g, φ_g)ᵀ with entries

        f^c_mn = k_mn/(2πG) J_m(k_mn ρ) cos mφ ,   f^s_mn = … sin mφ ,

    in the column order of the coefficient vector.  Δσ = F_Δσ ΔCS reproduces
    `wahr_invert` exactly (asserted in `_selftest_covariance`).
    """
    g = RHO.size
    F = np.zeros((g, 2 * m_max * n_max))
    r, ph = RHO.ravel(), PHI.ravel()
    for m in range(m_max):
        cm, sm = np.cos(m * ph), np.sin(m * ph)
        for n in range(1, n_max + 1):
            kmn = zeros_dict[m][n - 1] / R_alpha
            base = kmn / (2.0 * np.pi * G_SI) * BesselJ(m, kmn * r)
            col = 2 * (m * n_max + (n - 1))
            F[:, col] = base * cm
            F[:, col + 1] = base * sm
    return F


def mass_functional(R_star, alpha, m_max, n_max, zeros_dict):
    """
    f_ΔM ∈ R^{N_k}:  f_0n = (R*/G) J₁(j_{0,n}/α)  on the ZONAL COSINE entries,
    zero everywhere else.

    Two properties of this vector explain why ΔM survives what destroys the
    pointwise map.  (i) The azimuthal integration annihilates every m ≥ 1 mode
    exactly, so the N_k − N_c coefficients that carry the localized structure —
    and that are amplified hardest in Σ_Δσ — do not enter Σ_ΔM at all.  (ii) The
    entries carry NO factor k_0n, where those of f_Δσ carry one apiece: the k
    from the differentiation is cancelled term by term by the 1/k from the radial
    integration.  The weights are therefore bounded by 0.5819·R*/G (the maximum
    of J₁) and decay only as n^{-1/2}, so coefficient errors enter the mass
    essentially unamplified.
    """
    f = np.zeros(2 * m_max * n_max)
    for n in range(1, n_max + 1):
        f[2 * (0 * n_max + (n - 1))] = (R_star / G_SI) * BesselJ(
            1, zeros_dict[0][n - 1] / alpha
        )
    return f


def propagate_covariance(
    A_des,
    W,
    R_star,
    alpha,
    m_max,
    n_max,
    zeros_dict,
    RHO,
    PHI,
    cond=CH_COND,
    meas_rel=0.02,
    y_pre=None,
    y_post=None,
    epoch_rel=None,
    rho_epoch=0.0,
):
    """
    Full covariance analysis of one TAG inversion.  Returns a dict with

      sigma_dM        √Σ_ΔM   [kg]        formal 1σ of the moved mass
      sigma_map_1sig  √Σ_Δσ   [kg/m²]     pointwise 1σ of the density map
      Sigma_cs        Σ_ΔCS               coefficient covariance
      f_dM, F_sigma   the two functional vectors / matrix
      naive_dM        the WRONG route ∫√Σ_Δσ dA, for comparison
      modal           per-mode diagnostics (k, σ of ΔC_0n, share of Σ_ΔM)
    """
    M, n_kept, s_ratio = projection_matrix(A_des, W, cond=cond)
    var_dy = diff_field_variance(
        W,
        meas_rel=meas_rel,
        y_pre=y_pre,
        y_post=y_post,
        epoch_rel=epoch_rel,
        rho_epoch=rho_epoch,
    )
    S_cs = coeff_covariance(M, var_dy)

    f_dM = mass_functional(R_star, alpha, m_max, n_max, zeros_dict)
    var_dM = float(f_dM @ S_cs @ f_dM)

    F_sig = sigma_functional(RHO, PHI, alpha * R_star, m_max, n_max, zeros_dict)
    var_map = np.einsum("gk,kl,gl->g", F_sig, S_cs, F_sig).reshape(RHO.shape)
    sig_map = np.sqrt(np.maximum(var_map, 0.0))

    # the incorrect route the paper warns about: integrating the 1σ map.  It
    # sums standard deviations that partly cancel AND credits the m ≥ 1 modes
    # with a contribution that integrates to exactly zero.
    rho_1d, phi_1d = RHO[:, 0], PHI[0, :]
    dphi = phi_1d[1] - phi_1d[0]
    drho = rho_1d[1] - rho_1d[0]
    naive_dM = float(np.sum(sig_map * RHO) * drho * dphi)

    # modal breakdown of Σ_ΔM (zonal cosine block only) and of the coefficients
    k0 = np.array(
        [zeros_dict[0][n - 1] / (alpha * R_star) for n in range(1, n_max + 1)]
    )
    zc = [2 * (0 * n_max + (n - 1)) for n in range(1, n_max + 1)]
    sig_c0n = np.sqrt(np.diag(S_cs)[zc])
    share = np.array([f_dM[c] ** 2 * S_cs[c, c] for c in zc])
    share = share / share.sum() if share.sum() > 0 else share
    k_all = np.array(
        [
            zeros_dict[m][n - 1] / (alpha * R_star)
            for m in range(m_max)
            for n in range(1, n_max + 1)
        ]
    )
    sig_all = np.sqrt(np.diag(S_cs)[0::2])

    # Spatial coherence of the map error.  Σ_Δσ = F Σ_ΔCS Fᵀ is synthesized from
    # N_k coefficients however finely the map is gridded, so it has rank ≤ N_k
    # and neighbouring points do NOT carry independent errors.  Measure it: the
    # correlation between the innermost point and the rest of its radial line,
    # against the shortest retained wavelength 2π/k_max.
    n_phi = RHO.shape[1]
    row = (F_sig[0] @ S_cs) @ F_sig.T
    denom = np.sqrt(var_map.ravel()[0] * np.maximum(var_map.ravel(), 1e-300))
    corr_rad = (row / denom)[::n_phi]  # along φ = φ_0
    d_rad = rho_1d - rho_1d[0]
    below = np.where(corr_rad < np.exp(-1.0))[0]
    corr_len = float(d_rad[below[0]]) if below.size else float(d_rad[-1])
    lam_min = float(2.0 * np.pi / k_all.max())

    return dict(
        M=M,
        n_kept=n_kept,
        s_ratio=s_ratio,
        var_dy=var_dy,
        Sigma_cs=S_cs,
        f_dM=f_dM,
        F_sigma=F_sig,
        var_dM=var_dM,
        sigma_dM=float(np.sqrt(var_dM)),
        sigma_map_1sig=sig_map,
        naive_dM=naive_dM,
        modal=dict(
            k0n=k0,
            sigma_C0n=sig_c0n,
            share_dM=share,
            f0n=f_dM[zc],
            k_all=k_all,
            sigma_all=sig_all,
        ),
        corr_rad=corr_rad,
        d_rad=d_rad,
        corr_len=corr_len,
        lam_min=lam_min,
        rank_max=S_cs.shape[0],
        n_grid=RHO.size,
        meas_rel=meas_rel,
        epoch_rel=epoch_rel,
        rho_epoch=rho_epoch,
        cond=cond,
    )


def covariance_report(cov, res, verbose=True):
    """Print the covariance analysis, including the checks the derivation implies."""
    if not verbose:
        return
    dM, R_star, alpha = res["dM_est"], res["R_star"], res["alpha"]
    n_max, m_max = res["n_max"], res["m_max"]
    sd = cov["sigma_dM"]
    print(
        f"\n{DASH}\n  COVARIANCE ANALYSIS  (formal 1σ — dispersion, not accuracy)\n{DASH}"
    )
    src = (
        f"Δ-samples known to {cov['meas_rel']:.1%} of the Δ-field RMS"
        if cov["epoch_rel"] is None
        else f"each epoch to {cov['epoch_rel']:.1%} of its absolute field, "
        f"epoch correlation ρ={cov['rho_epoch']:.2f}"
    )
    print(f"    noise model     : {src}")
    print(
        f"    projection M    : {cov['n_kept']}/{2*m_max*n_max} SVD modes kept "
        f"(cond={cov['cond']:.0e}, smallest kept s/s_max = {cov['s_ratio']:.1e})"
    )
    print(
        f"    √Σ_ΔM           = {sd:.3e} kg   "
        f"({100*sd/abs(dM):.2f} % of ΔM = {dM:+.3e} kg)"
    )
    print(f"    ΔM = {dM:+.3e} ± {sd:.2e} kg  (1σ, formal)")
    scale = cov["meas_rel"] if cov["epoch_rel"] is None else cov["epoch_rel"]
    print(
        f"      Σ_ΔM is quadratic in the assumed precision, so √Σ_ΔM is LINEAR in it:"
        f"\n      {sd/(100*scale):.2e} kg per 1% — rescale rather than re-running."
    )
    sm = cov["sigma_map_1sig"]
    print(
        f"    √Σ_Δσ pointwise : centre {sm[0].mean():.1f}, median "
        f"{np.median(sm):.1f}, max {sm.max():.1f} kg/m²  "
        f"(map peak |Δσ| = {np.abs(res['sigma_map']).max():.0f} kg/m²)"
    )
    print(
        f"    WRONG route ∫√Σ_Δσ dA = {cov['naive_dM']:.3e} kg — "
        f"{cov['naive_dM']/sd:.0f}× the correct √Σ_ΔM: it sums standard"
    )
    print(
        f"      deviations that partly cancel and credits the m≥1 modes, which "
        f"integrate to zero."
    )

    md = cov["modal"]
    bound = 0.5819 * R_star / G_SI
    print(
        f"\n    zonal weights f_0n (the only ones ΔM sees): "
        f"|f| ≤ {np.abs(md['f0n']).max():.3e} vs bound 0.5819·R*/G = {bound:.3e}"
    )
    js = np.array([res["zeros_dict"][0][n - 1] for n in range(1, n_max + 1)])
    n_signflip = int((np.diff(np.sign(md["f0n"])) != 0).sum())
    print(
        f"    j_0,Nc = {js[-1]:.2f} vs α·j_1,1 = {3.8317*alpha:.2f} → "
        f"{n_signflip} sign change(s) among the {n_max} retained zonal weights"
    )
    print(
        f"    {'n':>3} {'k_0n [1/m]':>11} {'f_0n':>11} {'σ(ΔC_0n)':>11} "
        f"{'share of Σ_ΔM':>14}"
    )
    for i in range(n_max):
        print(
            f"    {i+1:3d} {md['k0n'][i]:11.4f} {md['f0n'][i]:+11.3e} "
            f"{md['sigma_C0n'][i]:11.3e} {100*md['share_dM'][i]:13.1f} %"
        )
    ka, sa = md["k_all"], md["sigma_all"]
    o = np.argsort(ka)
    i_pk = int(np.argmax(sa))
    print(f"\n    coefficient σ vs wavenumber k (downward continuation, e^{{+2k h̄}}):")
    print(
        f"      lowest k={ka[o][0]:.3f} → σ={sa[o][0]:.2e};  "
        f"worst k={ka[i_pk]:.3f} → σ={sa[i_pk]:.2e}  "
        f"({sa[i_pk]/sa[o][0]:.0f}× amplification)"
    )
    print(
        f"      beyond that the σ FALL again — not because those modes are well "
        f"determined\n      but because the SVD cutoff has removed them "
        f"({cov['n_kept']}/{2*m_max*n_max} kept).  Truncation is\n      what "
        f"regularizes the downward continuation; it trades resolution for stability."
    )
    print(
        "    → Σ_Δσ carries k² on top of that growth (the inversion is a "
        "differentiation);\n      Σ_ΔM carries none — the k from the derivative is "
        "cancelled by the 1/k from the\n      radial integral — which is why the mass "
        "is the robust product of the two."
    )
    print(
        f"\n    map-error coherence: Σ_Δσ is {cov['n_grid']} × {cov['n_grid']} but has "
        f"rank ≤ {cov['rank_max']},"
    )
    print(
        f"      so the errors are correlated: 1/e correlation length "
        f"{cov['corr_len']:.2f} m vs shortest\n      retained wavelength "
        f"2π/k_max = {cov['lam_min']:.2f} m.  Refining the grid does not buy "
        f"independent points."
    )
    st = cov.get("selftest")
    if st:
        print(
            f"\n    checks: F_Δσ·ΔCS reproduces wahr_invert to {st['e_map']:.1e}, "
            f"f_ΔMᵀ·ΔCS to {st['e_dM']:.1e};\n      equal-variance modes give an "
            f"azimuth-independent σ map to {st['aniso']:.1e} (isotropy test)."
        )
    print(
        "    NOTE: formal covariance only.  Bandlimit truncation, mass moved past "
        "ρ>R*,\n      and the thin-sheet idealization are systematic — √Σ_ΔM is a "
        "LOWER BOUND\n      on the total error; the geometric ground truth measures "
        "the rest."
    )


def _tex(v, nd=2):
    """A number as LaTeX scientific notation, or plain if it is O(1)."""
    if v == 0 or not np.isfinite(v):
        return "0"
    if float(v).is_integer() and abs(v) < 1e5:
        return f"{int(v)}"  # mode counts and the like, not 8.00
    if 1e-2 <= abs(v) < 1e4:
        return f"{v:.{nd}f}"
    e = int(np.floor(np.log10(abs(v))))
    return rf"{v / 10 ** e:.{nd}f}\times10^{{{e}}}"


def latex_tables(res, cov=None):
    """
    The paper's tables, ready to paste.  The figures carry no numbers — as in
    the GLOBAL scripts — so everything quotable is emitted here instead.
    """
    R, a = res["R_star"], res["alpha"]
    cov = cov if cov is not None else res.get("cov")
    print(f"\n{SEP}\n  LaTeX tabular bodies (figures carry no numbers)\n{SEP}")

    print("\n  % Table — TAG analysis geometry")
    for lab, v, u in [
        ("Cylinder radius $R^*$", R, "m"),
        ("Cylinder height $H$", res["H"], "m"),
        ("Bessel extension $\\alpha$", a, "--"),
        ("Sheet plane $z_0$", res["z_sheet"], "m"),
        ("Azimuthal orders $M_c$", res["m_max"], "--"),
        ("Radial modes $N_c$", res["n_max"], "--"),
        ("SVD cutoff", cov["cond"] if cov else float("nan"), "--"),
    ]:
        print(rf"  {lab} & ${_tex(v)}$ & {u} \\")

    print("\n  % Table — mass recovered by the differenced CH inversion")
    ratio = res["dM_est"] / res["dM_true"]
    for lab, v, u in [
        ("Gravimetric $\\Delta M$", res["dM_est"], "kg"),
        ("Geometric truth $\\Delta M$", res["dM_true"], "kg"),
        ("Recovery ratio", ratio, "--"),
        ("Equivalent $\\Delta V$", res["dM_est"] / res["density"], "m$^3$"),
        ("Mean $\\Delta h$", res["dh_equiv"], "m"),
        ("Effective $\\Delta\\rho$", res["delta_rho"], "kg\\,m$^{-3}$"),
        ("Relative signal $\\Delta U/U$", res["sig_ratio"], "--"),
    ]:
        print(rf"  {lab} & ${_tex(v, 3)}$ & {u} \\")

    if cov is None:
        return
    sd, dM = cov["sigma_dM"], res["dM_est"]
    rel = cov["meas_rel"] if cov["epoch_rel"] is None else cov["epoch_rel"]
    print("\n  % Table — formal uncertainty of the moved mass (dispersion only)")
    for lab, v, u in [
        ("Assumed field precision", 100 * rel, "\\%"),
        ("$\\sqrt{\\Sigma_{\\Delta M}}$", sd, "kg"),
        ("as a fraction of $\\Delta M$", 100 * sd / abs(dM), "\\%"),
        ("scaling, per 1\\% precision", sd / (100 * rel), "kg"),
        ("SVD modes retained", cov["n_kept"], "--"),
        ("Map-error correlation length", cov["corr_len"], "m"),
        ("Shortest retained wavelength", cov["lam_min"], "m"),
        (
            "Incorrect route $\\int\\!\\sqrt{\\Sigma_{\\Delta\\sigma}}\\,dA$",
            cov["naive_dM"],
            "kg",
        ),
    ]:
        print(rf"  {lab} & ${_tex(v, 3)}$ & {u} \\")

    if "mc" in cov:
        mc = cov["mc"]
        print("\n  % Table — Monte-Carlo verification of the analytic covariance")
        r_dM = mc["mc_sigma_dM"] / mc["an_sigma_dM"]
        rat = mc["mc_sigma_map"] / np.maximum(mc["an_sigma_map"], 1e-300)
        tol = 100.0 / np.sqrt(2.0 * mc["n_mc"])
        for lab, v, u in [
            ("Noise realizations", mc["n_mc"], "--"),
            ("MC $\\sqrt{\\Sigma_{\\Delta M}}$", mc["mc_sigma_dM"], "kg"),
            ("Analytic $\\sqrt{\\Sigma_{\\Delta M}}$", mc["an_sigma_dM"], "kg"),
            ("Ratio MC/analytic", r_dM, "--"),
            ("Map ratio, median", float(np.median(rat)), "--"),
            ("Map ratio, worst point", float(np.max(np.abs(rat - 1.0)) + 1.0), "--"),
            ("MC precision on a std", tol, "\\%"),
        ]:
            print(rf"  {lab} & ${_tex(v, 3)}$ & {u} \\")

    print(
        "\n  % Table — zonal weight budget: only these modes enter $\\Sigma_{\\Delta M}$"
    )
    print(r"  % n & k_{0n} [1/m] & f_{0n} & sigma(dC_{0n}) & share of Sigma_dM [%]")
    md = cov["modal"]
    for n in range(len(md["f0n"])):
        print(
            rf"  {n + 1} & ${_tex(md['k0n'][n], 3)}$ & ${_tex(md['f0n'][n])}$ & "
            rf"${_tex(md['sigma_C0n'][n])}$ & ${100 * md['share_dM'][n]:.1f}$ \\"
        )


def covariance_mc(res, cov, n_mc=4000, seed=3):
    """
    Monte-Carlo check of the analytic covariance chain.

    Draw noise on the DIFFERENCED field samples with the assumed
    Sigma_dy, push each realization through the SAME operators the analytic
    result uses --  M -> dCS,  then f_dM -> dM  and  F_sigma -> the map --
    and compare the realized spread against Sigma_dM and diag(Sigma_dsigma).

    Nothing here re-derives the covariance: it re-measures it.  The propagation
    is exact linear algebra, so agreement is expected to sampling precision;
    what the test can actually catch is an implementation error -- a wrong
    column ordering in F_sigma or f_dM, a stale M, a mis-scaled var_dy -- which
    is exactly the class of bug that would otherwise pass unnoticed because the
    analytic number looks perfectly reasonable on its own.

    Errors here are GAUSSIAN, unlike the positive RMS quantities of the GLOBAL
    scripts: dM is a fixed linear functional of Gaussian sample noise, so a
    normal overlay is the right reference, not a log-normal.
    """
    rng = np.random.default_rng(seed)
    sd_y = np.sqrt(cov["var_dy"])
    # noise-only coefficient realizations: dCS = M (dy + n) - M dy = M n
    E = cov["M"] @ (rng.normal(size=(sd_y.size, n_mc)) * sd_y[:, None])
    dM_err = cov["f_dM"] @ E  # (n_mc,)
    map_err = cov["F_sigma"] @ E  # (N_g, n_mc)
    return dict(
        dM_err=dM_err,
        mc_sigma_dM=float(dM_err.std(ddof=1)),
        an_sigma_dM=float(cov["sigma_dM"]),
        mc_sigma_map=map_err.std(axis=1, ddof=1),
        an_sigma_map=np.ravel(cov["sigma_map_1sig"]),
        n_mc=n_mc,
    )


def plot_covariance_mc(res, cov, outdir="Images", n_mc=4000, mc=None):
    """
    One panel per product of the inversion, each verifying its own covariance.

      LEFT   Delta sigma, the density MAP.  Every grid point's realized spread
             against the analytic sqrt(Sigma_dsigma) it was predicted to have,
             on the 1:1 line, with the ratio distribution inset.  This is the
             pointwise statement: does F Sigma_dCS F^T describe the map errors
             point by point?
      RIGHT  Delta M, the moved MASS.  The realized error distribution against
             the analytic N(0, Sigma_dM).  This is the scalar statement, and it
             is NOT implied by the left panel: DeltaM is a quadratic form over
             the whole coefficient covariance, off-diagonals included, and the
             map's pointwise variances alone would give the wrong answer by 8x
             (the "wrong route" line in the covariance report).

    `mc` lets the caller pass an already-computed `covariance_mc` result — the
    tables need those numbers before the figures are drawn, and the map-error
    array is (N_grid x n_mc), so computing it twice is neither free nor small.
    """
    mc = covariance_mc(res, cov, n_mc=n_mc) if mc is None else mc
    C = COLOR_PALETTE
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), gridspec_kw={"wspace": 0.28})

    # ── LEFT: the MAP covariance, point by point ──────────────────────────
    ax = axes[0]
    an, mcs = mc["an_sigma_map"], mc["mc_sigma_map"]
    ax.scatter(
        an, mcs, s=5, color=C[2], alpha=0.30, lw=0, rasterized=True, label="Grid points"
    )
    lim = [min(an.min(), mcs.min()) * 0.85, max(an.max(), mcs.max()) * 1.15]
    ax.plot(lim, lim, "k--", lw=1.6, zorder=4, label="1:1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Analytic $\sqrt{\Sigma_{\Delta\sigma}}$  [kg/m$^2$]")
    ax.set_ylabel(r"MC $\sqrt{\Sigma_{\Delta\sigma}}$  [kg/m$^2$]")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(fontsize=9, loc="upper left")

    axin = ax.inset_axes([0.60, 0.10, 0.36, 0.28], facecolor="white")
    axin.set_zorder(6)
    axin.patch.set_alpha(0.95)
    ratio = mcs / np.maximum(an, 1e-300)
    axin.hist(ratio, bins=45, color=C[2], alpha=0.8, edgecolor="k", lw=0.3)
    axin.axvline(1.0, color="k", lw=1.4)
    # the precision n_mc draws buy on a standard deviation
    tol = 1.0 / np.sqrt(2.0 * mc["n_mc"])
    axin.axvspan(1 - tol, 1 + tol, color="0.5", alpha=0.30, lw=0)
    axin.set_yticks([])
    axin.tick_params(labelsize=6.5)
    axin.set_xlabel("MC / analytic  [-]", fontsize=7)

    # ── RIGHT: the MASS covariance ────────────────────────────────────────
    ax = axes[1]
    e, sd = mc["dM_err"], mc["an_sigma_dM"]
    ax.hist(
        e,
        bins=48,
        density=True,
        color=C[0],
        alpha=0.78,
        edgecolor="k",
        lw=0.4,
        label="MC realizations",
    )
    x = np.linspace(e.min(), e.max(), 400)
    ax.plot(
        x,
        np.exp(-0.5 * (x / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
        color="k",
        lw=2.2,
        zorder=4,
        label=r"Analytic $N(0,\Sigma_{\Delta M})$",
    )
    for k in (-1, 1):
        ax.axvline(
            k * sd,
            color="0.30",
            ls="--",
            lw=1.5,
            zorder=3,
            label=r"Analytic $\pm\sqrt{\Sigma_{\Delta M}}$" if k == 1 else None,
        )
    ax.set_xlabel(r"$\widehat{\Delta M}-\Delta M$  [kg]")
    ax.set_ylabel("Density  [1/kg]")
    ax.grid(True, alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(
            os.path.join(outdir, "bennu_tag_fig4_covariance_mc.pdf"),
            bbox_inches="tight",
        )
    return fig, mc


def _selftest_covariance(res, cov):
    """
    Consistency checks the derivation implies.
      1. F_Δσ ΔCS reproduces `wahr_invert`'s map, f_ΔMᵀ ΔCS its ΔM.
      2. M applied to the differenced samples reproduces the fitted Δ coefficients.
      3. If Σ_ΔCS is diagonal with equal cos/sin variance per mode, the 1σ map is
         a function of ρ ALONE (cos²+sin² = 1) — isotropic even though the
         recovered feature is not.
    """
    dc = res["d_coeffs"]
    m1 = (cov["F_sigma"] @ dc).reshape(res["RHO"].shape)
    e_map = np.max(np.abs(m1 - res["sigma_map"])) / (
        np.abs(res["sigma_map"]).max() + 1e-30
    )
    e_dM = abs(float(cov["f_dM"] @ dc) - res["dM_est"]) / (abs(res["dM_est"]) + 1e-30)
    assert e_map < 1e-10 and e_dM < 1e-10, f"functional mismatch {e_map:.1e} {e_dM:.1e}"

    S_iso = np.diag(np.repeat(np.diag(cov["Sigma_cs"])[0::2], 2))  # equal cos/sin
    v = np.einsum("gk,kl,gl->g", cov["F_sigma"], S_iso, cov["F_sigma"]).reshape(
        res["RHO"].shape
    )
    aniso = float(np.max(np.ptp(v, axis=1) / (np.mean(v, axis=1) + 1e-300)))
    assert aniso < 1e-9, f"equal-variance modes gave an anisotropic map ({aniso:.1e})"
    return dict(e_map=e_map, e_dM=e_dM, aniso=aniso)


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
    cond: float = CH_COND,  # truncated-SVD cutoff of the weighted LS
    n_ensemble: int = 1,  # independent field-point draws averaged for ΔM
    k_target_R: float = KR_TARGET_DEFAULT,  # bandlimit target for n_max="auto"
    # covariance analysis (Section 4b)
    do_covariance: bool = True,
    meas_rel: float = 0.02,  # 1σ of a DIFFERENCED sample, as a fraction of the
    #                          Δ-field RMS of its observable type (route A)
    epoch_rel=None,  # instead: 1σ per epoch as a fraction of the ABSOLUTE field
    rho_epoch: float = 0.0,  # epoch-to-epoch error correlation (route B only)
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
            A_des, zeros_dict, W_des = A_i, zd_i, W_i

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

    # ── 6b. COVARIANCE PROPAGATION (Section 4b) ────────────────────────
    # Σ_ΔCS = M Σ_Δy Mᵀ with M the very projection the fit applied, then the two
    # quadratic forms.  Uses the first draw's geometry (Ψ, W): the covariance is
    # a property of the measurement design, not of a particular noise draw.
    cov = None
    if do_covariance:
        y_pre_v = assemble_obs_vector(U_pre, gr_pre, gphi_pre, gz_pre)
        y_post_v = assemble_obs_vector(U_post, gr_post, gphi_post, gz_post)
        cov = propagate_covariance(
            A_des,
            W_des,
            R_star,
            alpha,
            m_max,
            n_max,
            zeros_dict,
            RHO,
            PHI,
            cond=cond,
            meas_rel=meas_rel,
            y_pre=y_pre_v,
            y_post=y_post_v,
            epoch_rel=epoch_rel,
            rho_epoch=rho_epoch,
        )
        # M must reproduce the fit it stands for, on the actual pre-TAG samples
        e_fit = np.max(np.abs(cov["M"] @ y_pre_v - c_pre)) / (
            np.max(np.abs(c_pre)) + 1e-30
        )
        assert e_fit < 1e-8, f"projection matrix != fit_coefficients ({e_fit:.1e})"

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

    res = dict(
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
        weights=W_des,  # first-draw LS weights — re-propagate covariance with these
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
        # covariance analysis (None if do_covariance=False)
        cov=cov,
        sigma_dM=None if cov is None else cov["sigma_dM"],
        sigma_map_1sig=None if cov is None else cov["sigma_map_1sig"],
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

    if cov is not None:
        cov["selftest"] = _selftest_covariance(res, cov)
        covariance_report(cov, res, verbose=verbose)  # reads cov["selftest"]

    return res


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
        for z, lw in ((z0, 2.4), (z0 + H, 1.6)):  # base heavier: it marks the site
            ax.plot(
                cx + R * np.cos(th),
                cy + R * np.sin(th),
                z,
                color=color,
                lw=lw,
                zorder=6,
            )
        # Wireframe only: the two rings plus a few solid generatrices.  No
        # translucent wall — it hid the terrain it stands on.
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            ax.plot(
                [cx + R * np.cos(a)] * 2,
                [cy + R * np.sin(a)] * 2,
                [z0, z0 + H],
                color=color,
                lw=1.3,
                zorder=5,
            )

    # ── FIGURE 1 — geometry ────────────────────────────────────────────
    # One axes, not three.  The pre- and post-TAG terrains differ only by the
    # crater, so side by side they read as the same picture twice; colouring a
    # single surface by Delta h shows precisely what changed and nothing that
    # did not.  The field points and the analysis cylinder go on the same axes,
    # so what moved and the volume it is measured over are one frame.
    fig1 = plt.figure(figsize=(8.8, 7.2))
    ax = fig1.add_subplot(111, projection="3d")
    _nice_3d_axes(ax)

    dh = np.nan_to_num(res["h_post"] - res["h_pre"])
    # CROP to a window about the TAG site.  Over the full 44 m patch the DTM
    # edges carry scan artefacts as deep as the crater — those saturated red
    # and blue streaks are registration noise, not terrain — and the 16 m
    # cylinder is lost inside a patch three times its width.  Cropping removes
    # the artefacts and lets the site fill the frame.
    W = 1.9 * R
    ix, iy = np.abs(gx - cx) <= W, np.abs(gy - cy) <= W
    sel = np.ix_(ix, iy)
    GXc, GYc, hc, dhc = GX[sel], GY[sel], res["h_pre"][sel], dh[sel]
    # symmetric scale on the footprint, so the crater sets the colour range
    _inside = np.hypot(GXc - cx, GYc - cy) <= R
    v = float(np.percentile(np.abs(dhc[_inside]), 99)) or 1.0
    norm = mpl.colors.Normalize(-v, v)
    cmap = plt.get_cmap("RdBu_r")
    ax.plot_surface(
        GXc,
        GYc,
        hc,
        facecolors=cmap(norm(dhc)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.97,
    )
    _draw_cylinder(ax)
    # NOT `sm`: that name is the recovered sigma map further down this function
    cbar_src = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_src.set_array([])
    fig1.colorbar(cbar_src, ax=ax, pad=0.10, shrink=0.65).set_label(
        r"$\Delta h$ (post $-$ pre)  [m]"
    )
    ax.set_xlim(cx - W, cx + W)
    ax.set_ylim(cy - W, cy + W)
    ax.set_zlim(0, z0 + H)
    # the cylinder is 16 m tall over a 30 m window; without this it is drawn
    # into a tall thin box and reads as floating rather than sitting on the site
    ax.set_box_aspect((1.0, 1.0, 0.55))
    ax.legend(
        handles=[
            plt.Line2D(
                [], [], color="tab:cyan", lw=1.8, label=r"Analysis cylinder $R^*,\,H$"
            )
        ],
        loc="upper right",
        fontsize=9,
    )
    if outdir:
        fig1.savefig(
            os.path.join(outdir, "bennu_tag_fig1_geometry.pdf"),
            dpi=200,
            bbox_inches="tight",
        )

    # ── FIGURE 2 — what TAG changed ────────────────────────────────────
    # Everything static cancels in the difference, so the differences ARE the
    # measurement: the absolute field panel was showing the background the
    # estimator never sees.  Left and centre are the two differenced
    # observables the fit consumes; right is what they become after projection
    # onto the Bessel-Fourier basis, i.e. the vector the mass functional acts
    # on.  Three panels, one story, left to right.
    fig2, axes2 = plt.subplots(1, 3, figsize=(17.0, 4.9), gridspec_kw={"wspace": 0.42})

    def _scatter_cyl(ax, vals, title, label):
        vmax = np.percentile(np.abs(vals), 98)
        sc = ax.scatter(
            rp,
            zp,
            c=vals,
            cmap="RdBu_r",
            s=14,
            lw=0,
            vmin=-vmax,
            vmax=vmax,
            rasterized=True,
        )
        cb = fig2.colorbar(sc, ax=ax, pad=0.035, fraction=0.046)
        cb.set_label(label, fontsize=10)
        cb.ax.tick_params(labelsize=8)
        ax.axhline(0.0, color="0.35", ls="--", lw=1.1, zorder=1)  # the sheet plane
        ax.set_xlabel(r"$\rho$  [m]")
        ax.set_ylabel(r"$z-z_0$  [m]")
        ax.grid(True, alpha=0.2, lw=0.6)
        ax.set_axisbelow(True)
        for sd_ in ("top", "right"):
            ax.spines[sd_].set_visible(False)

    _scatter_cyl(
        axes2[0], dU, r"Differenced potential  $\Delta U$", r"$\Delta U$  [m$^2$/s$^2$]"
    )
    _scatter_cyl(
        axes2[1],
        dgz / UGAL,
        r"Differenced acceleration  $\Delta g_z$",
        r"$\Delta g_z$  [$\mu$Gal]",
    )

    # (c) the SPECTRUM of the differenced coefficients the two panels project
    # onto.  Same construction the GLOBAL scripts use for their CH panel: the
    # RMS over the radial modes at each azimuthal order, so one number per m
    # rather than 48 bars.
    ax = axes2[2]
    dc = res["d_coeffs"]
    n_max, m_max = res["n_max"], res["m_max"]
    amp = np.sqrt(dc[0::2] ** 2 + dc[1::2] ** 2).reshape(m_max, n_max)
    ms = np.arange(m_max)
    rms_m = np.sqrt((amp**2).mean(axis=1))
    # dispersion of the radial modes within each order, as a band.  It is
    # MULTIPLICATIVE: these are positive amplitudes on a log axis, so the
    # 1-sigma spread is exp(std of log amp) and the band is RMS x/÷ that
    # factor — an additive band would run through zero.
    lg = np.log(np.maximum(amp, 1e-300))
    gsd = np.exp(lg.std(axis=1, ddof=1))
    ax.fill_between(
        ms,
        rms_m / gsd,
        rms_m * gsd,
        color=COLOR_PALETTE[0],
        alpha=0.20,
        lw=0,
        zorder=2,
        label=r"Spread over $n$ ($\times/\div\,1\sigma$)",
    )
    ax.plot(
        ms,
        rms_m,
        "-o",
        color=COLOR_PALETTE[0],
        lw=2.2,
        ms=8,
        mec="k",
        mew=0.7,
        zorder=4,
        label=r"RMS over $n$",
    )
    ax.set_yscale("log")
    ax.set_ylim((rms_m / gsd).min() * 0.45, (rms_m * gsd).max() * 2.2)
    ax.set_xticks(ms)
    ax.set_xlabel(r"Azimuthal order $m$  [-]")
    ax.set_ylabel(r"$|\Delta\mathbf{CS}_{mn}|$  [m$^2$/s$^2$]")
    ax.grid(True, which="both", alpha=0.2, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
    for sd_ in ("top", "right"):
        ax.spines[sd_].set_visible(False)

    if outdir:
        fig2.savefig(
            os.path.join(outdir, "bennu_tag_fig2_gravity_change.pdf"),
            bbox_inches="tight",
        )

    # ── FIGURE 3 — recovered vs true mass change ───────────────────────
    # One row of three: the two maps and the radial profile that compares
    # them.  The coefficient spectrum moved to figure 2, beside the differenced
    # fields it is the projection of; the numeric summary became the terminal's
    # LaTeX tables; and the azimuthal profiles were dropped — the radial cut
    # already carries the bandlimit story, and the map itself shows the
    # azimuthal structure better than three slices through it.
    fig3 = plt.figure(figsize=(16.5, 5.2))
    gs3 = GridSpec(1, 3, figure=fig3, wspace=0.34)
    dM_unc = f" ± {res['dM_ens_std']:.1e}" if res.get("n_ensemble", 1) > 1 else ""
    vmax = max(np.percentile(np.abs(sm), 98), np.percentile(np.abs(st), 98))
    tc = np.linspace(0, 2 * np.pi, 200)

    # close the azimuthal seam for pcolormesh (φ wraps at 2π)
    def _wrap(a):
        return np.column_stack([a, a[:, :1]])

    Xw, Yw = _wrap(Xp), _wrap(Yp)
    # One colourbar per panel.  With the titles gone the bar labels are what
    # identify the panels, so each says explicitly which field it shows; and
    # all three use the same fraction/pad on equal-aspect axes, so the bars come
    # out the same size across the row.
    CBAR = dict(fraction=0.046, pad=0.03)
    for k, (mp, lab) in enumerate(
        [
            (sm, r"Estimated $\Delta\sigma$  [kg/m$^2$]"),
            (st, r"True $\rho\,\Delta h$  [kg/m$^2$]"),
        ]
    ):
        ax = fig3.add_subplot(gs3[k])
        c = ax.pcolormesh(
            Xw,
            Yw,
            _wrap(mp)[:-1, :-1],
            cmap="RdBu_r",
            shading="flat",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.plot(R * np.cos(tc), R * np.sin(tc), "k--", lw=1.2, alpha=0.65)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x-x_0$  [m]")
        ax.set_ylabel(r"$y-y_0$  [m]")
        fig3.colorbar(c, ax=ax, **CBAR).set_label(lab)

    # third panel, drawn exactly like the first two: the ERROR map, recovered
    # minus true.  Its range is set by its own percentile rather than the maps'
    # shared one — the residual is far smaller than either field, and reusing
    # their scale would render it uniformly white.
    ax3C = fig3.add_subplot(gs3[2])
    # Error as a PERCENTAGE OF THE PEAK truth, not pointwise (recovered-true)/true.
    # The truth is a signed field that crosses zero all over this map — 24% of
    # it lies below 5% of the peak and 39% below 10% — so a pointwise ratio is
    # undefined or explosive across a quarter of the disk.  Normalising by the
    # peak keeps every pixel finite and comparable, which is the usual choice
    # for a signed field.
    pk = float(np.abs(st).max()) or 1.0
    err = 100.0 * (sm - st) / pk
    verr = float(np.percentile(np.abs(err), 98)) or 1.0
    ce = ax3C.pcolormesh(
        Xw,
        Yw,
        _wrap(err)[:-1, :-1],
        cmap="RdBu_r",
        shading="flat",
        vmin=-verr,
        vmax=verr,
    )
    fig3.colorbar(ce, ax=ax3C, **CBAR).set_label(
        r"$(\widehat{\Delta\sigma}-\Delta\sigma)\,/\,\max|\Delta\sigma|$  "
        + (r"[\%]" if USE_TEX else "[%]")
    )
    ax3C.plot(R * np.cos(tc), R * np.sin(tc), "k--", lw=1.2, alpha=0.65)
    ax3C.set_aspect("equal")
    ax3C.set_xlabel(r"$x-x_0$  [m]")
    ax3C.set_ylabel(r"$y-y_0$  [m]")

    # (the old Summary text panel lived at gs3[1, 2]; every number in it
    #  is now in the terminal's LaTeX tables — figures carry no numbers)

    if outdir:
        fig3.savefig(
            os.path.join(outdir, "bennu_tag_fig3_mass_change.pdf"),
            dpi=200,
            bbox_inches="tight",
        )

    # NO plt.show() here: it blocks, so anything created after this call would
    # be built and never displayed.  The caller shows every figure at the end.
    return fig1, fig2, fig3


def plot_covariance(res, outdir=None):
    """
    Figure 4 — the covariance analysis of Section 4b.
      top    : the recovered Δσ map, its pointwise 1σ, and the mass summary;
      bottom : where Σ_ΔM comes from (zonal weights and their share), the
               coefficient σ against wavenumber (downward continuation, capped
               by the SVD cutoff), and the spatial coherence of the map error.
    Returns the figure, or None if the run had `do_covariance=False`.
    """
    cov = res.get("cov")
    if cov is None:
        return None
    RHO, PHI, R = res["RHO"], res["PHI"], res["R_star"]
    Xp, Yp = RHO * np.cos(PHI), RHO * np.sin(PHI)
    _wrap = lambda a: np.column_stack([a, a[:, :1]])
    tc = np.linspace(0, 2 * np.pi, 200)
    md = cov["modal"]
    n_max, m_max = res["n_max"], res["m_max"]

    fig = plt.figure(figsize=(17.5, 9.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # ── top: the map and its pointwise 1σ ────────────────────────────────
    for k, (mp, cm, lab) in enumerate(
        [
            (res["sigma_map"], "RdBu_r", r"$\Delta\sigma$  [kg/m$^2$]"),
            (cov["sigma_map_1sig"], "viridis", r"$\sigma$  [kg/m$^2$]"),
        ]
    ):
        ax = fig.add_subplot(gs[0, k])
        kw = dict(cmap=cm, shading="flat")
        if k == 0:
            v = np.percentile(np.abs(mp), 98)
            kw.update(vmin=-v, vmax=v)
        c = ax.pcolormesh(_wrap(Xp), _wrap(Yp), _wrap(mp)[:-1, :-1], **kw)
        fig.colorbar(c, ax=ax, label=lab)
        ax.plot(R * np.cos(tc), R * np.sin(tc), "k--", lw=1.2, alpha=0.65)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x-x_0$  [m]")
        ax.set_ylabel(r"$y-y_0$  [m]")

    # ── top right: the headline numbers ──────────────────────────────────
    # (a numbers panel lived at gs[0, 2]: DM, its 1-sigma, the noise model,
    #  the SVD modes kept and the "wrong route" figure.  All of it is in the
    #  terminal's LaTeX tables now — figures carry no numbers.)

    # ── bottom left: zonal weights + their share of the mass variance ────
    ax = fig.add_subplot(gs[1, 0])
    n = np.arange(1, n_max + 1)
    ax.bar(
        n, md["f0n"], color=COLOR_PALETTE[2], edgecolor="k", label=r"$f_{0n}$ (left)"
    )
    ax.axhline(0, color="k", lw=0.8)
    for sgn in (+1, -1):
        ax.axhline(sgn * 0.5819 * R / G_SI, color=COLOR_PALETTE[0], ls=":", lw=1.5)
    ax.set_xlabel(r"Zonal mode $n$  [-]")
    ax.set_ylabel(r"$f_{0n}$   [kg / (m$^2$ s$^{-2}$)]")
    ax.grid(alpha=0.3, axis="y")
    ax2 = ax.twinx()
    ax2.plot(
        n,
        100 * md["share_dM"],
        "o--",
        color=COLOR_PALETTE[3],
        lw=2,
        label=r"Share of $\Sigma_{\Delta M}$ (right)",
    )
    ax2.set_ylabel(r"Share of $\Sigma_{\Delta M}$  " + (r"[\%]" if USE_TEX else "[%]"))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower left")

    # ── bottom middle: coefficient sigma against wavenumber ──────────────
    ax = fig.add_subplot(gs[1, 1])
    m_of = np.repeat(np.arange(m_max), n_max)
    sc = ax.scatter(
        md["k_all"],
        md["sigma_all"],
        c=m_of,
        cmap="plasma",
        s=34,
        edgecolor="k",
        linewidths=0.4,
    )
    fig.colorbar(sc, ax=ax, label="Azimuthal order $m$")
    ax.set_yscale("log")
    ax.set_xlabel(r"Wavenumber $k_{mn}$  [1/m]")
    ax.set_ylabel(r"$\sigma(\Delta\mathcal{C}_{mn})$  [-]")
    ax.grid(alpha=0.3, which="both")

    # ── bottom right: spatial coherence of the map error ─────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(cov["d_rad"], cov["corr_rad"], lw=2.2, color=COLOR_PALETTE[2])
    ax.axhline(np.exp(-1), color="0.5", ls=":", lw=1.2)
    ax.axvline(
        cov["corr_len"],
        color=COLOR_PALETTE[0],
        ls="--",
        lw=1.5,
        label=f"1/e length = {cov['corr_len']:.2f} m",
    )
    ax.axvline(
        cov["lam_min"],
        color=COLOR_PALETTE[3],
        ls="-.",
        lw=1.5,
        label=rf"shortest $\lambda$ = {cov['lam_min']:.2f} m",
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Radial separation from the centre  [m]")
    ax.set_ylabel("Map-error correlation  [-]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(
            os.path.join(outdir, "bennu_tag_fig4_covariance.pdf"),
            dpi=170,
            bbox_inches="tight",
        )
    return fig


if __name__ == "__main__":

    result = run_bennu_tag(
        path_pre="3dmeshes/Bennu_preTag.obj",
        path_post="3dmeshes/Bennu_afterTag.obj",
        density=RHO_BULK,  # [kg/m³]
        grid_res=0.30,  # [m]
        site_center=None,  # auto-detect TAG crater from Δh
        # R_star controls PEAK resolution (k ∝ 1/R*): a smaller cylinder
        # concentrates the basis on the crater and recovers the sharp
        # central Δσ, at a cost in mass completeness.  Peak recovery of the
        # true ρΔh centre, re-measured with the uniform sampler (3 draws):
        #   R*=16 → 0.59×  (over-smoothed;  ΔM ratio 1.100 ± 0.065)
        #   R*=8  → 0.79×  (best mass/peak balance; ΔM 1.012 ± 0.012)
        #   R*=6  → 0.94×  (best peak, but ΔM 0.800 ± 0.012 — 20% low)
        R_star=8.0,  # [m]  (use 6 to prioritise the peak, 12+ only for mass)
        H=16.0,  # [m]
        clearance=0.25,  # points hug the surface: local terrain + 0.5 m
        alpha=2.0,  # boundary placement only; n_max scales with α ("auto")
        m_max=6,
        n_max="auto",  # → 8 at α=2 (see required_n_max)
        N_field=2000,
        cond=1e-4,  # truncated-SVD regularisation
        verbose=True,
    )

    # The MC numbers are needed by the tables, which print before any figure,
    # but the MC FIGURE closes the set — so compute once here and hand the
    # result to the plot, rather than drawing it early just to get the numbers.
    result["cov"]["mc"] = covariance_mc(result, result["cov"])

    latex_tables(result, result.get("cov"))

    fig1, fig2, fig3 = plot_results(result, outdir="Images")
    fig4, _ = plot_covariance_mc(
        result, result["cov"], outdir="Images", mc=result["cov"]["mc"]
    )

    plt.show()  # once, with all five figures built, so all five appear

    print("\nDone.")
