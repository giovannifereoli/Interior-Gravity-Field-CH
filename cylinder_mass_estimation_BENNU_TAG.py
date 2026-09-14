"""
Bennu Pre-TAG / Post-TAG — Cylindrical Harmonic Gravity Fitting
===============================================================
Estimates the mass moved by the OSIRIS-REx TAG event from the *difference* of
cylindrical-harmonic coefficients fitted before and after TAG, and validates it
against the geometric ground truth (the DTM height change itself).  Units are
selected by the module-level `MODE` switch: "SI" (m, kg, s) or "NORM" (L_REF,
RHO_REF, G = 1).  The raw unweighted potential/acceleration fit depends on the
unit system: the basis and cutoff calibrated in SI need rechecking in NORM.

Input data
----------
`Bennu_preTag.obj` / `Bennu_afterTag.obj` are local DTM patches of the
Nightingale site (~44 x 44 m, heights 0..9.5 m, y-up, metres).  They are scan
products — holes, a detached bottom plate, slightly different extents — so NOT
usable as closed polyhedra without rebuilding.

Pipeline
--------
1.  Extract the terrain sheet from each OBJ, rotate to z-up.
2.  Rasterise both onto ONE common (x, y) grid -> h_pre, h_post.  A common grid
    guarantees a common frame, so walls/bottom cancel exactly in the pre/post
    difference (per-mesh centring/scaling would corrupt Δg).
3.  Rebuild each state as a watertight "slab" polyhedron (top = terrain, bottom
    z = 0, side walls) -> valid input for the Tsoulis/Werner method.
4.  Locate the TAG crater from Δh = h_post − h_pre and centre the analysis
    cylinder there; the expansion plane (the "sheet") sits at the mean pre-TAG
    height inside the footprint.  Field points are drawn UNIFORMLY over the
    cylinder's vacuum volume — constant density per unit volume, no coordinate
    over-sampled (see `make_cylinder_field_points`) — with a lower bound that
    follows the LOCAL terrain (small clearance, point-by-point).  Inside the
    crater bowl points may sit slightly below the sheet plane; harmless, since
    the local Δ sources are still below them.
5.  Evaluate polyhedral gravity (U, g) at identical field points for both
    states (polyhedral_gravity: U > 0, g = +∇U, verified).
6.  Unweighted LS fit of the CH coefficients per state, identical design matrix
    and SVD cutoff -> ΔA = A_post − A_pre.  LS being linear, this equals fitting
    the difference field directly: every static source cancels exactly in Δc.
    That is WHY only the local patch needs meshing even though the field inside
    the cylinder is dominated by the whole ~490 m asteroid — TAG changed only
    the local site, so the unchanged bulk contributes IDENTICALLY to U_pre and
    U_post.  Potential and acceleration rows are stacked without weighting
    or rescaling, as in cylindrical_acc_pot_SHORT_fitting_both_MOV.py.
    Regularised by truncated SVD (`cond`).  Caveat: the cancellation assumes
    the local Δh map captures ALL
    the mass that moved — ejecta beyond the meshed patch, or mass moved to
    ρ > R*, is not counted.

    The cutoff is chosen WITHOUT the true ΔM by `calibration_sweep`: over a
    log grid of cutoffs, several field draws and two clearances, reject
    cutoffs whose held-out misfit (fit on one draw, predict the others)
    exceeds kappa x the best, then take the admissible cutoff where ΔM is
    most stable against draw, clearance and a small change of cutoff.  The
    true ΔM is only reported next to it, as validation (fig6_*).  These are
    sampling checks on the same terrain, not validation on new physical data.
    Uniform volume sampling does not imply an unbiased inverse.
7.  Wahr-like inversion of ΔA -> ΔM and Δσ(ρ,φ).  With n_ensemble > 1 the
    coefficients are averaged over that many field draws and the spread of
    ΔM over the draws is reported as the field-sampling scatter.
8.  Geometric ground truth: ΔM_true = ρ_bulk ∫∫ Δh dA over the footprint,
    Δσ_true = ρ_bulk Δh — a direct validation of the inversion.
9.  Assumed OD uncertainty via GLOBAL.od_sigma, by default on the recovered
    CHANGE ΔCS (route "delta"): a static background cancels in ΔCS, so it
    must not set the error bar.  Route "epoch" books od_sigma on each epoch's
    full CS with Σ_ΔCS = Σ_post + Σ_pre − Σ_post,pre − Σ_pre,post; it scales
    with that background unless rho_epoch ≈ 1 (covariance_report prints the
    sweep in rho and a point-mass background check).  Either Σ_ΔCS is
    propagated exactly to ΔM and Δσ.
10. Monte Carlo draws ΔCS from N(CS_post-CS_pre, Σ_ΔCS), with fixed geometry,
    and inverts those coefficient realizations.  It checks the propagated
    noise dispersion, not the terrain, truncation or thin-sheet model bias.

Formulae
--------
Basis (solves Laplace for sources below the sheet plane z = z0):
    U(ρ,φ,z) = Σ_{m,n} J_m(k_mn ρ) exp(−k_mn (z−z0)) [A_mn cos mφ + B_mn sin mφ]
    k_mn = j_{m,n} / (α R*),   α > 1  (Dirichlet zeros pushed out to α R*)

Gradients (g = +∇U, attraction convention, matches polyhedral_gravity):
    g_ρ = ∂U/∂ρ      = Σ k_mn J'_m(k_mn ρ) e^{−k_mn(z−z0)} [A cos + B sin]
    g_φ = (1/ρ)∂U/∂φ = Σ (m/ρ) J_m       e^{−k_mn(z−z0)} [−A sin + B cos]
    g_z = ∂U/∂z      = Σ (−k_mn) J_m     e^{−k_mn(z−z0)} [A cos + B sin]

Thin-sheet (Wahr-like) inversion — a surface-density mode σ_mn J_m(kρ) e^{imφ}
on z = z0 generates, for z > z0, U = (2πG σ_mn/k) J_m(kρ) e^{imφ} e^{−k(z−z0)},
hence σ_mn = k_mn A_mn / (2πG) and
    Δσ(ρ,φ) = 1/(2πG α R*) Σ_{m,n} j_{mn} J_m(k_mn ρ)[ΔA cos mφ + ΔB sin mφ]
    ΔM(ρ<R*) = ∫Δσ dA = (R*/G) Σ_n J_1(j_{0n}/α) ΔA_{0n}
(using ∫_0^{R*} J_0(kρ) ρ dρ = (R*/k) J_1(k R*); only m = 0 survives ∫dφ).
Both are dimensionally consistent in either unit system.
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
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
import time, os

# Reuse the GLOBAL experiment's assumed per-coefficient OD uncertainty rule.
# Import before this module's rcParams setup so its plotting defaults win here.
from cylinder_mass_estimation_GLOBAL import od_sigma

# TODO: LOCAL: how much SH would do here? Is CH needed?

# ── physical constants (SI) ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# UNIT SYSTEM — the one switch that decides how the whole script computes
# ═══════════════════════════════════════════════════════════════════════════
# "SI"   : metres, kilograms, seconds; G carries its physical value.
# "NORM" : dimensionless.  Three choices fix the system —
#             L_REF    the length unit, the analysis cylinder radius, so R* = 1
#                      and every length reads "in cylinder radii";
#             RHO_REF  the density unit, Bennu's bulk density, so rho_bulk = 1;
#             G = 1    the gravitational constant absorbed into the field scale.
#          Then  x~ = x/L,  M~ = M/(rho L^3),  sigma~ = sigma/(rho L),
#                U~ = U/(G rho L^2),  g~ = g/(G rho L),  and Poisson's equation
#          becomes lap U~ = -4 pi rho~.  That single substitution is what strips
#          G out of the inversion, the surface-density functional and the mass
#          functional.  The physical formulae transform consistently, but raw
#          unweighted LS changes the relative influence of U and g when length
#          units change.  The SI-calibrated basis/cutoff need rechecking in NORM.
MODE = "SI"  # "SI" or "NORM"

G_SI = 6.67430e-11  # [m³/kg/s²]  the physical constant, always
RHO_BULK_SI = 1190.0  # [kg/m³]  Bennu bulk density (Lauretta et al. 2019)
R_STAR_SI = 8.0  # [m]  the cylinder radius, and NORM's length unit
UGAL = 1.0e-8  # 1 µGal = 1e-8 m/s²  (SI only; see ACC_SCALE below)

if MODE == "NORM":
    L_REF, RHO_REF, G_W = R_STAR_SI, RHO_BULK_SI, 1.0
elif MODE == "SI":
    L_REF, RHO_REF, G_W = 1.0, 1.0, G_SI
else:
    raise ValueError(f"MODE must be 'SI' or 'NORM', got {MODE!r}")

RHO_BULK = RHO_BULK_SI / RHO_REF  # 1190 in SI, exactly 1 in NORM
PREFIX = "bennu_tag_" if MODE == "SI" else "bennu_tag_norm_"

# multiply a working-unit quantity by these to get SI back
TO_SI = dict(
    length=L_REF,
    area=L_REF**2,
    volume=L_REF**3,
    density=RHO_REF,
    mass=RHO_REF * L_REF**3,
    surface_density=RHO_REF * L_REF,
    # U~ = U/(G rho L^2) and g~ = g/(G rho L); in SI every factor is 1 already
    potential=1.0 if MODE == "SI" else G_SI * RHO_REF * L_REF**2,
    acceleration=1.0 if MODE == "SI" else G_SI * RHO_REF * L_REF,
    wavenumber=1.0 / L_REF,
)

# unit strings for labels and prints: empty in NORM, where nothing has units
ACC_SCALE = UGAL if MODE == "SI" else 1.0

# Unit strings.  _U is PLAIN TEXT, for the terminal; _UL is the same set in
# mathtext, for figure labels.  Both collapse to "-" in NORM, where nothing in
# the computation carries a unit.
if MODE == "SI":
    _U = dict(
        len="m",
        mass="kg",
        sd="kg/m²",
        dens="kg/m³",
        pot="m²/s²",
        accraw="m/s²",
        acc="µGal",
        k="1/m",
    )
    _UL = dict(
        len="m",
        mass="kg",
        sd=r"kg/m$^2$",
        dens=r"kg/m$^3$",
        pot=r"m$^2$/s$^2$",
        accraw=r"m/s$^2$",
        acc=r"$\mu$Gal",
        k="1/m",
    )
else:
    _U = _UL = dict(
        len="-", mass="-", sd="-", dens="-", pot="-", accraw="-", acc="-", k="-"
    )

# Okabe-Ito, the colour-vision-deficiency-safe palette used by the GLOBAL
# scripts, in the same role order.
COLOR = ["#D55E00", "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9"]
ACCENT = "#882255"  # structural elements (the analysis cylinder), as in GLOBAL

# Raw unweighted SI fit at R*=8 m, H=16 m, N_field=2000.  cond is relative to
# the largest singular value.  ΔM is a STAIRCASE in cond (one step per
# retained singular value), so it is chosen WITHOUT the true ΔM by
# `calibration_sweep` (Section 5b): among cutoffs whose held-out misfit is
# within kappa of the best, take the one where ΔM is most stable against
# field draws, clearance and the cutoff itself.  __main__ reruns that sweep
# and uses its c*; CH_COND is that c* (5.62e-3 on seeds 1..8, clearances
# 0.25/0.5 m), rounded, for callers that skip the sweep.  The earlier 3.1e-3
# was tuned against the true ΔM and sits on a one-step plateau next to a
# ~20% drop — see fig6_calibration_ratio.  Empirical for this geometry, not
# a universal resolution or accuracy guarantee.
CH_ALPHA = 3.0
CH_M_MAX = 5
CH_N_MAX = 10
CH_COND = 5.6e-3

USE_TEX = False  # os.environ.get("GLOBAL_NO_TEX", "") == ""


# ── font scale ──────────────────────────────────────────────────────────────
# ONE knob for every text size in this file: the rcParams below and every
# explicit `fontsize=` / `labelsize=` are written as (base * FONT_SCALE).
FONT_SCALE = 1.35

mpl.rcParams.update(
    {
        "axes.prop_cycle": mpl.cycler(color=COLOR),
        # Same switch as the GLOBAL scripts.  Every label here is written to be
        # valid in BOTH modes — maths in $...$, no bare unicode, no % — so
        # flipping it changes only the typeface and the speed.
        "text.usetex": USE_TEX,
        "font.family": "serif" if USE_TEX else "STIXGeneral",
        "mathtext.fontset": "stix",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": 12 * FONT_SCALE,
        "axes.labelsize": 13 * FONT_SCALE,
        "axes.titlesize": 13 * FONT_SCALE,
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

# Every panel is written to its OWN file: the paper places the figures
# individually, so nothing is composed into a multi-panel sheet here.
FS = (7.2, 5.4)  # default standalone panel (same as the GLOBAL scripts)
FS_MAP = (6.4, 5.4)  # equal-aspect map with its own colour bar


# see the GLOBAL scripts: 3-D axis labels need a pad that tracks the font scale
LPAD3D = 10 * FONT_SCALE


def _save(fig, outdir, name, pad=None):
    """Tight-crop one standalone panel to `outdir/PREFIX+name`; no-op if outdir is None.

    No dpi here: `savefig.dpi` (300) is set in the rcParams block above, and only
    rasterized content (the 3-D surfaces, the rasterized scatters) is affected.
    """
    if not outdir:
        return
    os.makedirs(outdir, exist_ok=True)
    kw = {"pad_inches": pad} if pad is not None else {}
    fig.savefig(os.path.join(outdir, PREFIX + name), bbox_inches="tight", **kw)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — DTM UTILITIES  (terrain extraction, common grid, slab rebuild)
# ═══════════════════════════════════════════════════════════════════════════


def load_terrain_points(path: str) -> np.ndarray:
    """
    Load a TAG-site OBJ and return the terrain-sheet vertices in a z-up
    working frame [m].

    The raw OBJs hold the terrain sheet, a detached flat bottom plate (~1920 m²
    at y=0) and thousands of scan fragments; only the largest-area component
    (the sheet) is kept.

    Frame change (x, y, z)_obj -> (x, −z, y): y-up -> z-up, a +90° rotation
    about X applied identically to both meshes — NO per-mesh centring or
    scaling, so pre and post stay in the SAME metric frame.
    """
    mesh = trimesh.load(path, force="mesh")
    comps = mesh.split(only_watertight=False)
    terrain = comps[np.argmax([c.area for c in comps])]
    # The one line in the script that touches a physical unit.  The OBJ is in
    # metres; everything after this is in working units.  It is a single COMMON
    # factor applied identically to both epochs, so unlike a per-mesh centring
    # it cannot corrupt the pre/post difference.  L_REF = 1 in SI mode.
    V = np.asarray(terrain.vertices, dtype=float) / L_REF
    return np.column_stack([V[:, 0], -V[:, 2], V[:, 1]])


def common_grid(P_pre, P_post, grid_res=0.30 / L_REF, edge_margin=0.5 / L_REF):
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


def height_map(P, GX, GY, h_min=0.05 / L_REF):
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


def locate_tag_site(dh, GX, GY, margin=3.0 / L_REF):
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
        # polyhedral_gravity applies the SI G internally.  Scaling the density
        # by G_W/G_SI makes what comes back already in working units — a no-op
        # in SI mode, and the 1/G_SI that defines the normalized field in NORM.
        density=density * (G_W / G_SI),
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
    clearance=0.5 / L_REF,
    N=2000,
    seed=1,
):
    """
    N points drawn UNIFORMLY over the vacuum volume of the cylinder ρ < R_star,
    between the local terrain (+ `clearance`) and the top z = z_sheet + H.

    "Uniform" means constant density per unit VOLUME — no coordinate is
    deliberately over-sampled:

        ρ = R*·√u     constant density per unit AREA.  (ρ = R*·u is uniform per
                      unit RADIUS, i.e. a density gradient piling points onto
                      the axis — not an unbiased sample.)
        φ ~ U(0, 2π)
        z ~ U(z_lo, H), keeping only points above the LOCAL terrain.

    The rejection step is what makes this uniform rather than merely
    uniform-per-column: the terrain under the footprint varies by several metres
    (~22 % of H here), so a fixed count per column would sample the deep crater
    ~1.3× less densely than the high ground.  Drawing z over one common range
    and rejecting below the terrain removes that gradient.

    NOTE on observability: CH modes decay as e^{−k_mn (z−z0)}, so a uniform
    sample spends most points at altitudes where the short modes are already
    negligible.  That is the deliberate price of an unbiased sample;
    `clearance` still sets how close to the surface the lowest points come.

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
# SECTION 3 — DESIGN MATRIX & UNWEIGHTED LEAST SQUARES
# ═══════════════════════════════════════════════════════════════════════════

# Checked: the scales below are correct, so there is no bug to fix, only a
# trade-off.  At fixed n_max, k_max ∝ 1/α: pushing the Dirichlet boundary αR*
# outward coarsens the basis.  To keep the bandlimit, grow n_max ∝ α, which
# `required_n_max` (n_max="auto") does.  Current defaults (R*=8 m, α=3,
# m_max=5, n_max=10): j_{4,10}=36.699, k_max*R*=12.23, λ_min=4.109 m; the
# zonal modes, the only ones in ΔM, stop at k*R*=10.21, λ=4.922 m.
# Basis wavenumbers and characteristic radial wavelengths:
#   k_mn = j_{m,n} / (α R*)             [1 / working length unit]
#   λ_mn = 2π / k_mn                   [working length unit]
# Included indices: m = 0..m_max-1, n = 1..n_max.  Hence:
#   k_max = j_{m_max-1,n_max} / (α R*)
#   λ_min = 2π / k_max = 2π α R* / j_{m_max-1,n_max}
#   k_min = j_{0,1} / (α R*)
#   λ_max = 2π / k_min = 2π α R* / j_{0,1}
# In Python (jn_zeros returns the first n_max positive zeros):
#   k_max = jn_zeros(m_max - 1, n_max)[-1] / (alpha * R_star)
#   lambda_min = 2.0 * np.pi / k_max
# Example: R*=8 m, α=2, m_max=6, n_max=8 gives j_{5,8}=31.8117,
#   k_max=1.9882 m^-1, k_max*R*=15.9059, λ_min=3.160 m.
# With α=100 and the same R*, m_max and n_max, λ_min becomes 158.009 m.
# Increasing α stretches the radial modes; increasing n_max approximately
# proportionally preserves k_max at large n.  α R* is the artificial boundary
# where the radial basis functions vanish.
# These are characteristic scales of the INCLUDED basis, before SVD filtering;
# they do not establish the spatial resolution supported by the observations.
# The heuristic target below requires k_max*R* >= 12, equivalently
# λ_min <= 2π R*/12 (4.189 m for R*=8 m); it does not guarantee mass accuracy.
# propagate_covariance() already computes this wavelength as `lam_min`.
KR_TARGET_DEFAULT = 12.0


def basis_spectral_scales(R_star, alpha, m_max, n_max, zeros_dict=None):
    """Characteristic scales of the INCLUDED basis, in working length units.

    lambda_min = 2π/k_max; lambda_max = 2π/k_min. Only m=0 contributes
    to the circular-footprint mass, so its shortest wavelength is also given.
    An individual mode decays by exp(-1) over a height 1/k. These scales
    precede SVD filtering; retained singular vectors mix Bessel modes and
    cannot be assigned a resolved wavelength from their rank alone.
    """
    if zeros_dict is None:
        zeros_dict = {m: jn_zeros(m, n_max) for m in range(m_max)}
    k = np.concatenate([zeros_dict[m][:n_max] for m in range(m_max)]) / (alpha * R_star)
    k_min, k_max = float(k.min()), float(k.max())
    return dict(
        k_min=k_min,
        k_max=k_max,
        k_max_R=k_max * R_star,
        lambda_min=2 * np.pi / k_max,
        lambda_max=2 * np.pi / k_min,
        lambda_min_zonal=2 * np.pi * alpha * R_star / zeros_dict[0][n_max - 1],
        efold_height_min=1 / k_max,
        efold_height_max=1 / k_min,
    )


# Checked: correct.  McMahon's large-n form j_{m,n} ≈ π(n + m/2 − 1/4), solved
# for n, only sets the starting guess; the while-loop then tests the EXACT zero,
# so the approximation cannot make the result wrong, only larger by the margin.
# m = m_max−1 is used because j_{m,n} grows with m, so that order sets k_max.
# The zonal (m = 0) modes that carry ΔM stay coarser — see lambda_min_zonal.
def required_n_max(alpha, R_star, m_max, k_target_R=KR_TARGET_DEFAULT, margin=2):
    """
    Estimate n_max with a margin, then verify k_max*R* >= k_target_R.
    The asymptotic zero j_mn ≈ π(n + m/2 - 1/4) gives the initial count.
    The margin and upward search can return more than the smallest count.
    R_star cancels from this dimensionless criterion.  This preserves a basis
    bandlimit when alpha grows; it does not optimize mass recovery or cutoff.
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


def fit_coefficients(A_des, U, gr, gphi, gz, cond=CH_COND):
    """
    Unweighted, truncated-SVD LS: A_des @ coeffs ≈ b, as in the MOV fit.
    No row scaling or RMS weights are applied to potential or acceleration.
    This objective depends on the working units selected by MODE; basis and
    cutoff calibration in SI does not automatically transfer to NORM.
    Singular values below cond*s_max are discarded (None uses SciPy's default).
    Returns coeffs, raw stacked RMS, raw stacked relative RMS.  These mixed-unit
    residual diagnostics do not supply the assumed coefficient uncertainties.
    """
    b = assemble_obs_vector(U, gr, gphi, gz)
    coeffs, _, _, _ = lstsq(A_des, b, cond=cond)
    resid = A_des @ coeffs - b
    rms = np.sqrt(np.mean(resid**2))
    rel = rms / (np.sqrt(np.mean(b**2)) + 1e-30)
    return coeffs, rms, rel


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — WAHR-LIKE THIN-SHEET INVERSION  (ΔM [kg], Δσ [kg/m²])
# ═══════════════════════════════════════════════════════════════════════════


def wahr_invert(
    delta_coeffs, R_star, alpha, m_max, n_max, zeros_dict, n_rho=80, n_phi=120
):
    """
    Invert ΔA = A_post − A_pre for the mass change and the surface-density
    change map, assuming the mass change is a thin sheet on the expansion
    plane z = z_sheet (see module docstring for the derivation):

        Δσ(ρ,φ)  = 1/(2πG αR*) Σ_{m,n} j_mn J_m(k_mn ρ)[ΔA cos mφ + ΔB sin mφ]
        ΔM(ρ<R*) = (R*/G) Σ_n J_1(j_0n/α) ΔA_0n

    ΔM integrates Δσ over the cylinder FOOTPRINT (ρ < R*); mass change in the
    buffer annulus R* < ρ < αR* is representable by the basis but not counted.

    Returns delta_M [kg], sigma_map [kg/m²] (n_rho, n_phi), RHO, PHI [m, rad].
    """
    R_alpha = alpha * R_star

    delta_M = 0.0
    for n in range(1, n_max + 1):
        j0n = zeros_dict[0][n - 1]
        col = 2 * (0 * n_max + (n - 1))
        delta_M += (R_star / G_W) * BesselJ(1, j0n / alpha) * delta_coeffs[col]

    rho_1d = np.linspace(0.02 * R_star, 0.98 * R_star, n_rho)
    phi_1d = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    RHO, PHI = np.meshgrid(rho_1d, phi_1d, indexing="ij")
    sigma_map = np.zeros_like(RHO)
    pref = 1.0 / (2.0 * np.pi * G_W * R_alpha)

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
# which must be expanded about a reference state first, these covariances are
# EXACT consequences of Σ_ΔCS.  They describe dispersion only — truncation bias,
# mass that left the footprint, and the thin-sheet idealization are systematic
# and invisible to them.  A quoted √Σ_ΔM is a conditional noise uncertainty,
# not a bound on the realized error against geometric truth.
# The analysis thus reduces to writing down f_Δσ and f_ΔM, as done below.


def coefficient_difference_covariance(
    c_pre, c_post, eps=0.02, floor_frac=0.1, od_alpha=0.10, rho_epoch=0.0,
    route="delta",
):
    """
    Assumed diagonal OD covariance of ΔCS = CS_post - CS_pre.  Two routes:

    route="delta" (default) books od_sigma on the recovered CHANGE itself,

        sigma_delta = od_sigma(c_post - c_pre, eps, floor_frac, alpha=od_alpha),

    i.e. eps is the precision of the differenced solution.  A static background
    (the rest of Bennu, any mismodelled field common to both epochs) cancels in
    ΔCS, so it cancels here too: σ_ΔM is a property of the TAG signal alone.

    route="epoch" books it on each epoch's FULL coefficients and subtracts,

        sigma_pre  = od_sigma(c_pre,  eps, floor_frac, alpha=od_alpha)
        sigma_post = od_sigma(c_post, eps, floor_frac, alpha=od_alpha)
        var(ΔCS_i) = sigma_pre_i² + sigma_post_i²
                     - 2*rho_epoch*sigma_pre_i*sigma_post_i

    Because od_sigma is RELATIVE, this σ scales with |CS|, i.e. with the static
    background, even though ΔM does not.  It is meaningful only for rho_epoch
    close to 1 (the common-mode error cancelling like the background does), or
    with an absolute eps taken from a real OD covariance.  covariance_report()
    prints the rho_epoch sweep and _selftest_covariance() the background check.

    Different coefficients are independent in both assumed models.  A supplied
    OD covariance can instead go straight into propagate_covariance(), including
    off-diagonal coefficient correlations.

    GLOBAL's od_sigma is reused unchanged: its degree labels are inferred from
    SH-style array packing (Lmin=2 here).  On CH coefficients this exponential
    factor is an index-based heuristic, not a function of k_mn.  od_alpha=0
    gives the relative-precision-plus-floor rule without that factor.
    od_alpha is distinct from the cylinder's Bessel extension alpha.
    """
    c_pre, c_post = np.asarray(c_pre, float), np.asarray(c_post, float)
    if c_pre.ndim != 1 or c_pre.size == 0 or c_post.shape != c_pre.shape:
        raise ValueError(
            "c_pre and c_post must be nonempty matching coefficient vectors"
        )
    if not (np.isfinite(c_pre).all() and np.isfinite(c_post).all()):
        raise ValueError("coefficient vectors must be finite")
    if not np.isfinite([eps, floor_frac, od_alpha, rho_epoch]).all():
        raise ValueError("OD uncertainty settings must be finite")
    if eps < 0 or floor_frac < 0 or od_alpha < 0 or not -1 <= rho_epoch <= 1:
        raise ValueError(
            "eps, floor_frac and od_alpha must be nonnegative; |rho_epoch| <= 1"
        )
    if route not in ("delta", "epoch"):
        raise ValueError("route must be 'delta' or 'epoch'")
    sig_pre = od_sigma(c_pre, eps, floor_frac=floor_frac, alpha=od_alpha)
    sig_post = od_sigma(c_post, eps, floor_frac=floor_frac, alpha=od_alpha)
    if route == "delta":
        var_delta = od_sigma(c_post - c_pre, eps, floor_frac=floor_frac,
                             alpha=od_alpha) ** 2
    else:
        # Equivalent to the subtraction formula, stable when rho_epoch is near 1.
        var_delta = (sig_post - sig_pre) ** 2 + 2 * (1 - rho_epoch) * sig_pre * sig_post
    return dict(
        Sigma_cs=np.diag(var_delta),
        sigma_pre=sig_pre,
        sigma_post=sig_post,
        sigma_delta=np.sqrt(var_delta),
        coeff_rel=eps,
        coeff_floor_frac=floor_frac,
        od_alpha=od_alpha,
        rho_epoch=rho_epoch,
        route=route,
    )


def _coefficient_covariance_factor(Sigma_cs, n_coeff):
    """Validate a coefficient covariance and factor it, allowing zero modes."""
    S = np.asarray(Sigma_cs, float)
    if S.shape != (n_coeff, n_coeff) or not np.isfinite(S).all():
        raise ValueError(f"Sigma_cs must be a finite ({n_coeff}, {n_coeff}) matrix")
    tol = 100 * np.finfo(float).eps * n_coeff * np.max(np.abs(S))
    if np.max(np.abs(S - S.T)) > tol:
        raise ValueError("Sigma_cs must be symmetric")
    S = (S + S.T) / 2
    eig, Q = np.linalg.eigh(S)
    if eig[0] < -tol:
        raise ValueError("Sigma_cs must be positive semidefinite")
    factor = Q * np.sqrt(np.maximum(eig, 0.0))
    # Clip only negative eigenvalues within roundoff of zero.
    if eig[0] < 0:
        S = factor @ factor.T
    return S, factor


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
            base = kmn / (2.0 * np.pi * G_W) * BesselJ(m, kmn * r)
            col = 2 * (m * n_max + (n - 1))
            F[:, col] = base * cm
            F[:, col + 1] = base * sm
    return F


def mass_functional(R_star, alpha, m_max, n_max, zeros_dict):
    """
    f_ΔM ∈ R^{N_k}:  f_0n = (R*/G) J₁(j_{0,n}/α)  on the ZONAL COSINE entries,
    zero everywhere else.

    Two properties explain why ΔM survives what destroys the pointwise map.
    (i) Azimuthal integration annihilates every m ≥ 1 mode exactly, so the
    N_k − N_c coefficients carrying the localized structure — the ones amplified
    hardest in Σ_Δσ — never enter Σ_ΔM.  (ii) The entries carry NO factor k_0n,
    where those of f_Δσ carry one apiece: the k from the differentiation is
    cancelled term by term by the 1/k from the radial integration.  The weights
    are therefore bounded by 0.5819·R*/G (max of J₁) and decay only as n^{-1/2},
    so coefficient errors enter the mass essentially unamplified.
    """
    f = np.zeros(2 * m_max * n_max)
    for n in range(1, n_max + 1):
        f[2 * (0 * n_max + (n - 1))] = (R_star / G_W) * BesselJ(
            1, zeros_dict[0][n - 1] / alpha
        )
    return f


def propagate_covariance(
    Sigma_cs,
    R_star,
    alpha,
    m_max,
    n_max,
    zeros_dict,
    RHO,
    PHI,
):
    """
    Propagate an INPUT covariance of ΔCS = CS_post - CS_pre through the
    thin-sheet inversion.  Sigma_cs has shape (N_k, N_k), N_k=2*m_max*n_max, in
    the same interleaved cosine/sine packing as wahr_invert.  Both diagonal
    and correlated covariances are supported.  No field samples, fitting
    weights or SVD projection enter this uncertainty calculation:

        var(ΔM) = f_dM.T @ Sigma_cs @ f_dM
        Cov(Δσ) = F_sigma @ Sigma_cs @ F_sigma.T

    Only the diagonal and one correlation row of the map covariance are
    evaluated, avoiding a dense (n_grid, n_grid) allocation.  Returns:

      sigma_dM        √Σ_ΔM   [kg]        formal 1σ of the moved mass
      sigma_map_1sig  √Σ_Δσ   [kg/m²]     pointwise 1σ of the density map
      Sigma_cs        Σ_ΔCS               coefficient covariance
      f_dM, F_sigma   the two functional vectors / matrix
      naive_dM        the WRONG route ∫√Σ_Δσ dA, for comparison
      modal           per-mode diagnostics (k, σ of ΔC_0n, share of Σ_ΔM)
    """
    S_cs, _ = _coefficient_covariance_factor(Sigma_cs, 2 * m_max * n_max)

    f_dM = mass_functional(R_star, alpha, m_max, n_max, zeros_dict)
    var_dM = max(0.0, float(f_dM @ S_cs @ f_dM))

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
    # Include cross-covariances in the budget.  Contributions sum to var_dM
    # and can be negative when coefficient correlations cancel mass error.
    contribution = (f_dM * (S_cs @ f_dM))[zc]
    share = contribution / var_dM if var_dM > 0 else np.zeros(n_max)
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
    # against the shortest included basis wavelength 2π/k_max.
    n_phi = RHO.shape[1]
    row = (F_sig[0] @ S_cs) @ F_sig.T
    denom = np.sqrt(
        np.maximum(var_map.ravel()[0], 0.0) * np.maximum(var_map.ravel(), 0.0)
    )
    corr_rad = np.divide(row, denom, out=np.zeros_like(row), where=denom > 0)[::n_phi]
    d_rad = rho_1d - rho_1d[0]
    below = np.where(corr_rad < np.exp(-1.0))[0]
    corr_len = float("nan")
    if var_map.ravel()[0] > 0:
        corr_len = float(d_rad[below[0]]) if below.size else float(d_rad[-1])
    lam_min = float(2.0 * np.pi / k_all.max())
    scales = basis_spectral_scales(R_star, alpha, m_max, n_max, zeros_dict)

    return dict(
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
        lam_max=scales["lambda_max"],
        basis_scales=scales,
        rank_max=S_cs.shape[0],
        n_grid=RHO.size,
    )


def covariance_report(cov, res, verbose=True):
    """Print the covariance analysis, including the checks the derivation implies."""
    if not verbose:
        return
    dM, R_star, alpha = res["dM_est"], res["R_star"], res["alpha"]
    m_max, n_max = res["m_max"], res["n_max"]
    sd = cov["sigma_dM"]
    print(
        f"\n{DASH}\n  COVARIANCE ANALYSIS  (formal 1σ — dispersion, not accuracy)\n{DASH}"
    )
    if "coeff_rel" in cov:
        if cov.get("route", "epoch") == "delta":
            where = "od_sigma on the recovered change ΔCS (background-free)"
        else:
            where = (f"od_sigma on each epoch's full CS, "
                     f"epoch correlation ρ={cov['rho_epoch']:.3f}")
        print(f"    noise model     : {where}")
        print(
            f"                      eps={cov['coeff_rel']:.1%}, "
            f"floor_frac={cov['coeff_floor_frac']:.2f}, od_alpha={cov['od_alpha']:.2f}"
        )
        print("    CH degree factor: GLOBAL's SH-packing index heuristic (not k_mn)")
    else:
        print("    noise model     : supplied covariance of ΔCS")
    print("    propagation     : Σ_ΔCS → fᵀ Σ_ΔCS f and diag(F Σ_ΔCS Fᵀ)")
    rel_sd = sd / abs(dM) if dM else float("nan")
    print(
        f"    √Σ_ΔM           = {sd:.3e} {_U['mass']}   "
        f"({100*rel_sd:.2f} % of ΔM = {dM:+.3e} {_U['mass']})"
    )
    print(f"    ΔM = {dM:+.3e} ± {sd:.2e} {_U['mass']}  (1σ, formal)")
    scale = cov.get("coeff_rel", 0.0)
    if scale > 0:
        print(
            f"      √Σ_ΔM scales linearly with eps: "
            f"{sd/(100*scale):.2e} {_U['mass']} per 1% coefficient precision."
        )
        # Why the default books eps on ΔCS: the per-epoch rule vs epoch correlation.
        kw = dict(eps=scale, floor_frac=cov["coeff_floor_frac"], od_alpha=cov["od_alpha"])
        f = cov["f_dM"]
        s_d = np.sqrt(f @ coefficient_difference_covariance(
            res["c_pre"], res["c_post"], route="delta", **kw)["Sigma_cs"] @ f)
        print(f"    per-epoch rule vs ρ (same eps; delta route gives "
              f"{100*s_d/abs(dM):.2f} % of ΔM):")
        for rho in (0.0, 0.9, 0.99, 0.999, 1.0):
            s_e = np.sqrt(f @ coefficient_difference_covariance(
                res["c_pre"], res["c_post"], route="epoch", rho_epoch=rho,
                **kw)["Sigma_cs"] @ f)
            print(f"      ρ = {rho:5.3f}: √Σ_ΔM = {s_e:.3e} {_U['mass']} "
                  f"({100*s_e/abs(dM):9.2f} % of ΔM)")
    sm = cov["sigma_map_1sig"]
    print(
        f"    √Σ_Δσ pointwise : centre {sm[0].mean():.1f}, median "
        f"{np.median(sm):.1f}, max {sm.max():.1f} {_U['sd']}  "
        f"(map peak |Δσ| = {np.abs(res['sigma_map']).max():.0f} {_U['sd']})"
    )
    print(
        f"    WRONG route ∫√Σ_Δσ dA = {cov['naive_dM']:.3e} {_U['mass']} — "
        f"{cov['naive_dM']/sd if sd else float('nan'):.0f}× the correct √Σ_ΔM: it sums standard"
    )
    print(
        f"      deviations that partly cancel and credits the m≥1 modes, which "
        f"integrate to zero."
    )

    md = cov["modal"]
    bound = 0.5819 * R_star / G_W
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
    print("\n    assumed coefficient σ, listed against basis wavenumber k:")
    print(
        f"      lowest k={ka[o][0]:.3f} → σ={sa[o][0]:.2e};  "
        f"largest σ at k={ka[i_pk]:.3f} → σ={sa[i_pk]:.2e}"
    )
    print(
        "      This covariance is assigned to CS after fitting.  It is not "
        "inferred from the field samples or their SVD cutoff."
    )
    print(
        "    → Σ_Δσ carries k² on top of that growth (the inversion is a "
        "differentiation);\n      Σ_ΔM carries none — the k from the derivative is "
        "cancelled by the 1/k from the\n      radial integral.  This does not "
        "guarantee small mass bias."
    )
    print(
        f"\n    map-error coherence: Σ_Δσ is {cov['n_grid']} × {cov['n_grid']} but has "
        f"rank ≤ {cov['rank_max']},"
    )
    print(
        f"      so the errors are correlated: 1/e correlation length "
        f"{cov['corr_len']:.2f} {_U['len']} vs shortest\n      included basis wavelength "
        f"2π/k_max = {cov['lam_min']:.2f} {_U['len']}.  Refining the grid does not buy "
        f"independent points."
    )
    st = cov.get("selftest")
    if st:
        print(
            f"\n    checks: F_Δσ·ΔCS reproduces wahr_invert to {st['e_map']:.1e}, "
            f"f_ΔMᵀ·ΔCS to {st['e_dM']:.1e};\n      equal-variance modes give an "
            f"azimuth-independent σ map to {st['aniso']:.1e} (isotropy test)."
        )
        if "bg_delta" in st:
            print(
                f"      static background (Bennu-mass point {st['bg_depth']:.0f} "
                f"{_U['len']} below the sheet) added to both epochs:\n"
                f"      ΔM changes by {st['bg_dM']:.1e} (rel.), delta-route √Σ_ΔM by "
                f"{st['bg_delta']:.1e}, per-epoch (ρ={cov['rho_epoch']:.3f}) √Σ_ΔM "
                f"×{st['bg_epoch']:.1e}."
            )
    print(
        "    NOTE: formal covariance only.  Bandlimit truncation, mass moved past "
        "ρ>R*,\n      and the thin-sheet idealization are excluded.  The Monte "
        "Carlo tests\n      propagation of the assumed CS errors, not physical accuracy."
    )


def _tex(v, nd=2):
    """A number as LaTeX scientific notation, or plain if it is O(1)."""
    if not np.isfinite(v):
        return r"\mathrm{n/a}"
    if v == 0:
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
        ("Coefficient-fit SVD cutoff", res["cond"], "--"),
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
    rel = cov.get("coeff_rel", float("nan"))
    epoch = cov.get("route") == "epoch"
    rows = [
        ("Assumed per-epoch coefficient precision" if epoch
         else "Assumed precision of the coefficient change", 100 * rel, "\\%"),
        ("Coefficient floor fraction", cov.get("coeff_floor_frac", float("nan")), "--"),
        ("OD index growth parameter", cov.get("od_alpha", float("nan")), "--"),
    ]
    if epoch:
        rows.append(("Pre/post coefficient correlation",
                     cov.get("rho_epoch", float("nan")), "--"))
    print("\n  % Table — formal uncertainty of the moved mass (dispersion only)")
    for lab, v, u in rows + [
        ("$\\sqrt{\\Sigma_{\\Delta M}}$", sd, "kg"),
        (
            "as a fraction of $\\Delta M$",
            100 * sd / abs(dM) if dM else float("nan"),
            "\\%",
        ),
        (
            "scaling, per 1\\% coefficient precision",
            sd / (100 * rel) if rel > 0 else float("nan"),
            "kg",
        ),
        ("Map-error correlation length", cov["corr_len"], "m"),
        ("Shortest included basis wavelength", cov["lam_min"], "m"),
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
        r_dM = (
            mc["mc_sigma_dM"] / mc["an_sigma_dM"]
            if mc["an_sigma_dM"] > 0
            else float("nan")
        )
        observed = mc["an_sigma_map"] > 0
        rat = mc["mc_sigma_map"][observed] / mc["an_sigma_map"][observed]
        tol = 100.0 / np.sqrt(2.0 * mc["n_map"])  # the map is the shallower one
        for lab, v, u in [
            ("Noise realizations, $\\Delta M$", mc["n_mc"], "--"),
            ("Noise realizations, map", mc["n_map"], "--"),
            ("MC $\\sqrt{\\Sigma_{\\Delta M}}$", mc["mc_sigma_dM"], "kg"),
            ("Analytic $\\sqrt{\\Sigma_{\\Delta M}}$", mc["an_sigma_dM"], "kg"),
            ("Ratio MC/analytic", r_dM, "--"),
            (
                "Map ratio, median",
                float(np.median(rat)) if rat.size else float("nan"),
                "--",
            ),
            (
                "Map ratio, worst point",
                float(np.max(np.abs(rat - 1.0)) + 1.0) if rat.size else float("nan"),
                "--",
            ),
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


def covariance_mc(res, cov, n_mc=200000, n_map=20000, seed=3, batch_size=256):
    """
    Draw ΔCS ~ N(res['d_coeffs'], cov['Sigma_cs']) and invert every draw.
    With route="delta" this samples the assumed error of the recovered change
    directly; with route="epoch" it is equivalent to drawing jointly Gaussian
    pre/post CS and subtracting them (coefficient_difference_covariance).
    No field samples, field noise or coefficient refits are used; the scatter
    over field-point draws is a separate number (res['dM_ens_std']).

    The inversion is linear: evaluating the functionals on coefficient errors
    and adding the nominal result equals inverting each perturbed coefficient
    vector.  Both independent and correlated coefficient errors are supported.
    Map moments are accumulated in batches to avoid an n_grid*n_map allocation.
    Only res['d_coeffs'] is needed from the nominal fit; all other inputs are
    the coefficient covariance and the propagation functionals in cov.
    """
    for name, value in (("n_mc", n_mc), ("n_map", n_map), ("batch_size", batch_size)):
        minimum = 1 if name == "batch_size" else 2
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    dc = np.asarray(res["d_coeffs"], float)
    if dc.ndim != 1 or not np.isfinite(dc).all():
        raise ValueError("d_coeffs must be a finite coefficient vector")
    _, factor = _coefficient_covariance_factor(cov["Sigma_cs"], dc.size)
    rng = np.random.default_rng(seed)
    n_map = min(n_map, n_mc)
    dM_err = np.empty(n_mc)
    map_mean = np.zeros(cov["F_sigma"].shape[0])
    map_m2 = np.zeros_like(map_mean)
    count = 0
    for start in range(0, n_mc, batch_size):
        stop = min(start + batch_size, n_mc)
        errors = factor @ rng.normal(size=(stop - start, dc.size)).T
        dM_err[start:stop] = cov["f_dM"] @ errors
        take = min(stop, n_map) - start
        if take > 0:
            maps = cov["F_sigma"] @ errors[:, :take]
            mean = maps.mean(axis=1)
            delta = mean - map_mean
            total = count + take
            map_m2 += np.sum((maps - mean[:, None]) ** 2, axis=1)
            map_m2 += delta**2 * count * take / total
            map_mean += delta * take / total
            count = total
    nominal_mass = float(cov["f_dM"] @ dc)
    nominal_map = cov["F_sigma"] @ dc
    an_map = np.ravel(cov["sigma_map_1sig"])
    return dict(
        dM_err=dM_err,
        dM_samples=nominal_mass + dM_err,
        mc_mean_dM=nominal_mass + float(dM_err.mean()),
        mc_sigma_dM=float(dM_err.std(ddof=1)),
        an_sigma_dM=float(cov["sigma_dM"]),
        mc_mean_map=nominal_map + map_mean,
        mc_sigma_map=np.sqrt(np.maximum(map_m2 / (count - 1), 0.0)),
        an_sigma_map=an_map,
        n_mc=n_mc,
        n_map=n_map,
        seed=seed,
    )


def plot_covariance_mc(res, cov, outdir="Images", n_mc=200000, n_map=20000, mc=None):
    """
    The predicted 1-sigma map of Delta sigma against the one the Monte-Carlo
    actually produces, their ratio, and the Delta M histogram — one file each.

      ANALYTIC   sqrt(diag(F Sigma_dCS F^T)), the pointwise 1-sigma the
                 covariance predicts, before any sampling.
      NUMERICAL  the spread of `n_map` noise realizations pushed through the
                 same inversion, measured point by point.
      RATIO      the two divided, on a scale centred at 1.  Agreement means they
                 differ only by Monte-Carlo scatter, of size 1/sqrt(2 n_map) at
                 this many draws; the colour range is four times that, so
                 anything structural would be unmistakable.
      MASS       Delta M is a scalar, so one histogram against its predicted
                 Gaussian.

    The two maps share one colour scale so they can be compared across files;
    the ratio keeps its own, living in a narrow band about unity.

    Returns (list of figures, the Monte-Carlo dict).
    """
    mc = covariance_mc(res, cov, n_mc=n_mc, n_map=n_map) if mc is None else mc
    RHO, PHI, R = res["RHO"], res["PHI"], res["R_star"]
    Xp, Yp = RHO * np.cos(PHI), RHO * np.sin(PHI)
    _wrap = lambda A: np.column_stack([A, A[:, :1]])
    Xw, Yw = _wrap(Xp), _wrap(Yp)
    tc = np.linspace(0, 2 * np.pi, 200)

    an = mc["an_sigma_map"].reshape(RHO.shape)
    nu = mc["mc_sigma_map"].reshape(RHO.shape)
    ratio = np.divide(nu, an, out=np.full_like(nu, np.nan), where=an > 0)
    tol = 1.0 / np.sqrt(2.0 * mc["n_map"])

    # The maps and the mass go to separate files.  Delta sigma is a field and
    # needs three maps; Delta M is a scalar and needs one histogram — forcing
    # them onto one row squeezed the maps and left the histogram in a cell of
    # the wrong shape.  One panel per file, and each gets the aspect it wants.
    CBAR = dict(fraction=0.046, pad=0.03)
    vmax = max(an.max(), nu.max())

    def _decor(ax):
        """Footprint circle, equal aspect and axis labels, on every map."""
        ax.plot(R * np.cos(tc), R * np.sin(tc), "k--", lw=1.2, alpha=0.65)
        ax.set_aspect("equal")
        ax.set_xlabel(rf"$x-x_0$  [{_UL['len']}]")
        ax.set_ylabel(rf"$y-y_0$  [{_UL['len']}]")

    figs = []
    for tag, mp, lab in (
        (
            "analytic",
            an,
            rf"Analytic $\sqrt{{\Sigma_{{\Delta\sigma}}}}$  [{_UL['sd']}]",
        ),
        (
            "numerical",
            nu,
            rf"Monte-Carlo $\sqrt{{\Sigma_{{\Delta\sigma}}}}$  [{_UL['sd']}]",
        ),
    ):
        fig, ax = plt.subplots(figsize=FS_MAP)
        c = ax.pcolormesh(
            Xw,
            Yw,
            _wrap(mp)[:-1, :-1],
            cmap="viridis",
            shading="flat",
            vmin=0.0,
            vmax=vmax,
        )
        fig.colorbar(c, ax=ax, **CBAR).set_label(lab)
        _decor(ax)
        _save(fig, outdir, f"fig4_covariance_map_{tag}.pdf")
        figs.append(fig)

    fig, ax = plt.subplots(figsize=FS_MAP)
    cr = ax.pcolormesh(
        Xw,
        Yw,
        _wrap(ratio)[:-1, :-1],
        cmap="RdBu_r",
        shading="flat",
        vmin=1 - 4 * tol,
        vmax=1 + 4 * tol,
    )
    fig.colorbar(cr, ax=ax, **CBAR).set_label("Monte-Carlo / analytic  [-]")
    _decor(ax)
    _save(fig, outdir, "fig4_covariance_map_ratio.pdf")
    figs.append(fig)

    # ── the mass, on its own: a scalar, so one histogram against its predicted
    # Gaussian.  Kept in kg rather than normalized — the kilograms make the size
    # of the uncertainty readable straight off the axis.
    fig_m, ax = plt.subplots(figsize=FS)
    e, sd = mc["dM_err"], mc["an_sigma_dM"]
    if sd > 0:
        ax.hist(
            e,
            bins=60,
            density=True,
            color=COLOR[0],
            alpha=0.78,
            edgecolor="k",
            lw=0.3,
            label="Monte-Carlo coefficient draws",
        )
        xg = np.linspace(e.min(), e.max(), 400)
        ax.plot(
            xg,
            np.exp(-0.5 * (xg / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
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
    else:
        ax.axvline(0, color="k", label="Zero assumed mass variance")
    ax.set_xlabel(
        rf"$\Delta M_{{\mathrm{{draw}}}}-\Delta M_{{\mathrm{{nominal}}}}$  [{_UL['mass']}]"
    )
    ax.set_ylabel(f"PDF  [1/{_U['mass']}]")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9 * FONT_SCALE)
    _save(fig_m, outdir, "fig5_covariance_mass.pdf")
    return figs + [fig_m], mc


def _selftest_covariance(res, cov):
    """
    Consistency checks the derivation implies.
      1. F_Δσ ΔCS reproduces `wahr_invert`'s map, f_ΔMᵀ ΔCS its ΔM.
      2. If Σ_ΔCS is diagonal with equal cos/sin variance per mode, the 1σ map is
         a function of ρ ALONE (cos²+sin² = 1) — isotropic even though the
         recovered feature is not.
      3. A static background common to both epochs (a Bennu-mass point 245 m
         below the sheet, fitted with the same design matrix and cutoff) leaves
         ΔM and the delta-route σ_ΔM unchanged; the per-epoch σ_ΔM inflates.
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
    out = dict(e_map=e_map, e_dM=e_dM, aniso=aniso)
    if "coeff_rel" not in cov:
        return out

    depth = 245.0 / TO_SI["length"]
    GM = G_W * 7.33e10 / TO_SI["mass"]  # Bennu's mass
    rp, dz = res["rp"], res["zp"] + depth
    r = np.hypot(rp, dz)
    sgn = np.sign(np.mean(res["U_pre"]))  # the potential sign convention in use
    b_bg = assemble_obs_vector(sgn * GM / r, -GM * rp / r**3, 0 * r, -GM * dz / r**3)
    c_bg = lstsq(res["design_matrix"], b_bg, cond=res["cond"])[0]
    kw = dict(eps=cov["coeff_rel"], floor_frac=cov["coeff_floor_frac"],
              od_alpha=cov["od_alpha"], rho_epoch=cov["rho_epoch"])
    f = cov["f_dM"]

    def sd(cp, cq, route):
        S = coefficient_difference_covariance(cp, cq, route=route, **kw)["Sigma_cs"]
        return np.sqrt(f @ S @ f)

    cp, cq = res["c_pre"], res["c_post"]
    bg_dM = abs(f @ ((cq + c_bg) - (cp + c_bg)) - f @ (cq - cp)) / abs(f @ (cq - cp))
    bg_delta = abs(sd(cp + c_bg, cq + c_bg, "delta") / sd(cp, cq, "delta") - 1)
    # round-off only: the background coefficients are ~1e6× the TAG change
    assert bg_dM < 1e-6 and bg_delta < 1e-6, (
        f"background leaked into ΔM ({bg_dM:.1e}) or delta σ ({bg_delta:.1e})")
    bg_epoch = sd(cp + c_bg, cq + c_bg, "epoch") / sd(cp, cq, "epoch")
    out.update(bg_dM=bg_dM, bg_delta=bg_delta, bg_epoch=bg_epoch,
               bg_depth=depth * TO_SI["length"])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def run_bennu_tag(
    path_pre: str = "3dmeshes/Bennu_preTag.obj",
    path_post: str = "3dmeshes/Bennu_afterTag.obj",
    density: float = RHO_BULK,  # bulk density (1 in NORM)
    grid_res: float = 0.30,  # DTM raster resolution [m]
    # Cylinder / basis parameters (SI metres)
    site_center=None,  # (x, y) [m]; None → auto-detect from Δh
    R_star: float = R_STAR_SI / L_REF,  # cylinder radius (1 in NORM)
    H: float = 16.0 / L_REF,  # cylinder height above the sheet
    clearance: float = 0.5,  # local terrain clearance of field points [m]
    alpha: float = CH_ALPHA,  # Bessel extension (α > 1)
    m_max: int = CH_M_MAX,  # azimuthal orders 0..m_max−1
    n_max=CH_N_MAX,  # radial modes 1..n_max; int, or "auto" for bandlimit only
    N_field: int = 2000,
    seed: int = 1,
    n_ensemble: int = 1,  # field-point draws (seeds seed..seed+n−1), averaged
    cond: float = CH_COND,  # truncated-SVD cutoff of the unweighted LS
    k_target_R: float = KR_TARGET_DEFAULT,  # bandlimit target for n_max="auto"
    # covariance analysis (Section 4b)
    do_covariance: bool = True,
    cov_route: str = "delta",  # od_sigma on ΔCS ("delta") or per epoch ("epoch")
    coeff_rel: float = 0.02,  # od_sigma eps (on ΔCS, or on each epoch's CS)
    coeff_floor_frac: float = 0.1,  # floor relative to the CS RMS it is applied to
    od_alpha: float = 0.10,  # GLOBAL's index-based OD growth; distinct from alpha
    rho_epoch: float = 0.0,  # pre/post error correlation (route="epoch" only)
    verbose: bool = True,
):
    """
    Full pre/post TAG pipeline in the unit system selected by `MODE`.  Returns
    a dict of all intermediate and final results (see bottom of function).

    Fit PRE/POST coefficients on the same field points; with n_ensemble > 1,
    repeat on independent field-point draws and average the coefficients.  The
    std of ΔM over those draws (res['dM_ens_std']) is FIELD-SAMPLING scatter.
    Separately, assign OD uncertainty with GLOBAL's od_sigma (on ΔCS by default,
    see coefficient_difference_covariance) and propagate Σ_ΔCS to mass and
    density.  Call covariance_mc(res, res['cov']) to draw coefficient
    realizations; field points and gravity are not redrawn in that Monte Carlo.
    Neither number includes terrain or thin-sheet model bias.

    On α and n_max: k_mn = j_{m,n}/(α R*) — α only sets where the fictitious
    Dirichlet boundary sits, n_max sets the highest wavenumber (resolution)
    reachable at that α.  Raising α without raising n_max shrinks k_max
    proportionally (α=100, m_max=6, n_max=8, R*=8 m gives λ_min≈158 m).
    `n_max="auto"` derives n_max from α so that k_max·R* ≥ `k_target_R` holds
    (see `required_n_max`); an explicit int below that heuristic target raises.
    Passing the bandlimit check does not guarantee mass accuracy.  The tuned
    explicit defaults and the bandlimit-only "auto" option serve different aims.
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
            f"    pre : {len(mesh_pre.faces):6d} faces, V = {mesh_pre.volume:9.2f} {_U['len']}³"
        )
        print(
            f"    post: {len(mesh_post.faces):6d} faces, V = {mesh_post.volume:9.2f} {_U['len']}³"
        )
        print(
            f"    total ΔV (patch)  = {mesh_post.volume - mesh_pre.volume:+8.2f} {_U['len']}³"
        )

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
                f"{k_max_R:.2f}, below the configured bandlimit target "
                f"{k_target_R}. Use n_max='auto', set n_max ≥ {n_needed}, or "
                f"explicitly lower k_target_R when testing a coarser basis. "
                f"This guard checks included wavelengths, not mass accuracy."
            )

    scales = basis_spectral_scales(R_star, alpha, m_max, n_max)

    # geometric ground truth (what the inversion should recover)
    dV_foot = float(dh[foot].sum() * dA)
    dM_true = density * dV_foot
    dV_total = float(dh.sum() * dA)
    if verbose:
        print(f"\n[2] TAG site (auto): ({cx:+.2f}, {cy:+.2f}) m")
        print(f"    cylinder R* = {R_star} m, H = {H} m, α = {alpha}, n_max = {n_max}")
        print(f"    fit: unweighted LS, m_max={m_max}, n_max={n_max}, cond={cond}")
        print(
            f"    included k range: {scales['k_min']:.4f}–{scales['k_max']:.4f} {_U['k']}"
        )
        print(
            f"    included λ range: {scales['lambda_min']:.3f}–{scales['lambda_max']:.3f} {_U['len']}"
        )
        print(
            f"    shortest m=0 wavelength (mass): {scales['lambda_min_zonal']:.3f} {_U['len']}"
        )
        print(f"    sheet plane z0 = {z_sheet:.2f} m")
        print(f"    GROUND TRUTH  ΔV(ρ<R*) = {dV_foot:+.2f} {_U['len']}³")
        print(
            f"                  ΔM(ρ<R*) = {dM_true:+.4e} {_U['mass']}  (ρ={density} {_U['dens']})"
        )
        print(f"                  ΔV(patch) = {dV_total:+.2f} {_U['len']}³")

    # ── 3./4./5. FIELD POINTS, GRAVITY, NOMINAL COEFFICIENT FIT ──────────
    if verbose:
        print(f"\n[3] Building GravityEvaluable objects …", end=" ", flush=True)
    t0 = time.time()
    ev_pre = make_evaluable(mesh_pre, density)
    ev_post = make_evaluable(mesh_post, density)
    if verbose:
        print(f"done ({time.time()-t0:.1f}s)")

    h_itp = RegularGridInterpolator((gx, gy), h_env)
    if verbose:
        print(f"\n[4] Nominal field points + gravity + CS fit ({N_field} pts) …")
    t0 = time.time()
    rp, pp, zp, pts_cart = make_cylinder_field_points(
        (cx, cy),
        z_sheet,
        R_star,
        H,
        h_itp,
        clearance=clearance,
        N=N_field,
        seed=seed,
    )
    h_under = h_itp(pts_cart[:, :2])
    assert (pts_cart[:, 2] > h_under).all(), "field points intersect terrain"

    U_pre, gx_pre, gy_pre, gz_pre = eval_gravity(ev_pre, pts_cart)
    U_post, gx_post, gy_post, gz_post = eval_gravity(ev_post, pts_cart)
    gr_pre, gphi_pre = cart_to_cyl_g(gx_pre, gy_pre, pp)
    gr_post, gphi_post = cart_to_cyl_g(gx_post, gy_post, pp)
    A_des, zeros_dict = build_design_matrix(rp, pp, zp, R_alpha, m_max, n_max)
    c_pre, rms_pre, rel_pre = fit_coefficients(
        A_des, U_pre, gr_pre, gphi_pre, gz_pre, cond=cond
    )
    c_post, rms_post, rel_post = fit_coefficients(
        A_des, U_post, gr_post, gphi_post, gz_post, cond=cond
    )
    d_coeffs = c_post - c_pre
    db = assemble_obs_vector(
        U_post - U_pre, gr_post - gr_pre, gphi_post - gphi_pre, gz_post - gz_pre
    )
    rel_delta = float(
        np.linalg.norm(A_des @ d_coeffs - db) / (np.linalg.norm(db) + 1e-30)
    )

    if verbose:
        print(f"    done ({time.time()-t0:.1f}s)")
        print(f"    field z above sheet ∈ [{zp.min():.2f}, {zp.max():.2f}] m")
        z_q10 = np.percentile(zp, 10)
        k_max = jn_zeros(m_max - 1, n_max)[-1] / R_alpha
        print(
            f"    observability k_max·z_q10 = {k_max*z_q10:.2f} "
            f"(keep ≲ 4, else lower n_max)"
        )
        print(f"    U_pre ∈ [{U_pre.min():.3e}, {U_pre.max():.3e}] {_U['pot']}")
        print(f"    gz_pre ∈ [{gz_pre.min():.3e}, {gz_pre.max():.3e}] {_U['accraw']}")
        print(f"    rel RMS  pre = {rel_pre:.3e},  post = {rel_post:.3e}")
        print(f"    rel RMS  Δ-field fit (raw stacked U/g) = {rel_delta:.3e}")

    # ── 5b. FIELD-SAMPLING ENSEMBLE ────────────────────────────────────
    # Refit on independent point draws (seeds seed+1 …) and average the
    # coefficients; draw 0 above stays the diagnostic draw (design matrix,
    # residuals, plots).  The ΔM std over draws is sampling scatter of the
    # field-point geometry — NOT the OD noise propagated in 6b.
    n_ens = max(1, int(n_ensemble))
    c_pre_draws, c_post_draws = [c_pre], [c_post]
    for i in range(1, n_ens):
        rp_i, pp_i, zp_i, pts_i = make_cylinder_field_points(
            (cx, cy), z_sheet, R_star, H, h_itp,
            clearance=clearance, N=N_field, seed=seed + i,
        )
        assert (pts_i[:, 2] > h_itp(pts_i[:, :2])).all(), "field points intersect terrain"
        A_i, _ = build_design_matrix(rp_i, pp_i, zp_i, R_alpha, m_max, n_max)
        for ev, draws in ((ev_pre, c_pre_draws), (ev_post, c_post_draws)):
            U_i, gx_i, gy_i, gz_i = eval_gravity(ev, pts_i)
            gr_i, gphi_i = cart_to_cyl_g(gx_i, gy_i, pp_i)
            draws.append(fit_coefficients(A_i, U_i, gr_i, gphi_i, gz_i, cond=cond)[0])
    c_pre, c_post = np.mean(c_pre_draws, axis=0), np.mean(c_post_draws, axis=0)
    d_coeffs = c_post - c_pre
    dM_draws = mass_functional(R_star, alpha, m_max, n_max, zeros_dict) @ (
        np.array(c_post_draws) - np.array(c_pre_draws)
    ).T
    dM_ens_std = float(dM_draws.std(ddof=1)) if n_ens > 1 else 0.0
    if verbose and n_ens > 1:
        print(
            f"    field-sampling ensemble: {n_ens} draws (seeds {seed}..{seed+n_ens-1}), "
            f"ΔM std = {dM_ens_std:.3e} {_U['mass']} "
            f"({100*dM_ens_std/abs(dM_draws.mean()):.2f} %)"
        )

    # ── 6. WAHR INVERSION ──────────────────────────────────────────────
    dM_est, sigma_map, RHO, PHI = wahr_invert(
        d_coeffs, R_star, alpha, m_max, n_max, zeros_dict
    )

    # ── 6b. COVARIANCE PROPAGATION (Section 4b) ────────────────────────
    # Assume the fitted CS and their OD uncertainties are the available data.
    # Form Σ_ΔCS (route: see coefficient_difference_covariance), then propagate.
    cov = None
    if do_covariance:
        od = coefficient_difference_covariance(
            c_pre,
            c_post,
            eps=coeff_rel,
            floor_frac=coeff_floor_frac,
            od_alpha=od_alpha,
            rho_epoch=rho_epoch,
            route=cov_route,
        )
        cov = propagate_covariance(
            od["Sigma_cs"],
            R_star,
            alpha,
            m_max,
            n_max,
            zeros_dict,
            RHO,
            PHI,
        )
        cov.update({k: v for k, v in od.items() if k != "Sigma_cs"})

    # ── 7. DERIVED QUANTITIES & TRUTH COMPARISON ───────────────────────
    # true surface-density change on the same polar grid
    dh_itp = RegularGridInterpolator((gx, gy), dh, bounds_error=False, fill_value=0.0)
    sigma_true = density * dh_itp(
        np.column_stack(
            [(cx + RHO * np.cos(PHI)).ravel(), (cy + RHO * np.sin(PHI)).ravel()]
        )
    ).reshape(RHO.shape)

    # ── central-peak recovery diagnostic ──────────────────────────────
    # The recovered Δσ is bandlimited: upward continuation suppresses short
    # modes and the inversion can smooth a sharp central peak.  Quantify it
    # separately from mass recovery; the finite-footprint integral is not
    # guaranteed to be preserved by truncation or the thin-sheet model.
    core = RHO[:, 0] < 1.0
    sigma_peak_rec = float(sigma_map[core].mean())
    sigma_peak_true = float(sigma_true[core].mean())
    sigma_peak_true_pix = density * float(dh[np.hypot(GX - cx, GY - cy) < 1.0].min())
    z_resolution = float(zp.min())  # field-point altitude above expansion plane
    if verbose:
        print(
            f"    Δσ central peak (ρ<1 m): recovered {sigma_peak_rec:+.0f} vs "
            f"true {sigma_peak_true:+.0f} {_U['sd']} "
            f"({sigma_peak_rec/sigma_peak_true:.2f}×; deepest pixel "
            f"{sigma_peak_true_pix:+.0f})"
        )
        print(
            f"    → peak deficit {1-sigma_peak_rec/sigma_peak_true:.0%}; "
            f"check ΔM separately below. Peak and integral accuracy can differ."
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
        print(f"  ΔM  gravimetric       = {dM_est:+.4e} {_U['mass']}")
        if n_ens > 1:
            print(
                f"      ± {dM_ens_std:.2e} {_U['mass']} field-sampling scatter "
                f"(1σ over n_ensemble={n_ens} draws; OD noise: see covariance)"
            )
        print(
            f"  ΔM  geometric truth   = {dM_true:+.4e} {_U['mass']}   (ρ·∫Δh dA, ρ<R*)"
        )
        print(f"  recovery ratio        = {dM_est / dM_true:8.3f}")
        print(f"  ΔV  equivalent        = {dM_est/density:+.2f} {_U['len']}³")
        print(f"  mean Δh over footprint= {dh_equiv:+.4f} m")
        print(
            f"  Δρ_eff (ΔM/V_cyl)     = {delta_rho:+.4f} {_U['dens']}  (V_cyl={V_cyl:.0f} {_U['len']}³)"
        )
        print(f"  Δgz RMS               = {np.std(dgz)/UGAL:.2f} {_U['acc']}")
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
        # field points (draw 0) and the sampling settings calibration_sweep reuses
        rp=rp,
        pp=pp,
        zp=zp,
        pts_cart=pts_cart,
        clearance=clearance,
        N_field=N_field,
        seed=seed,
        n_ensemble=n_ens,
        dM_draws=dM_draws,  # per-draw ΔM; dM_est is their mean
        dM_ens_std=dM_ens_std,  # field-sampling scatter, not OD noise
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
        design_matrix=A_des,  # nominal coefficient-fit diagnostics only
        fit_objective="unweighted",
        basis_scales=scales,
        cond=cond,
        zeros_dict=zeros_dict,
        c_pre=c_pre,
        c_post=c_post,
        d_coeffs=d_coeffs,
        rms_pre=rms_pre,
        rms_post=rms_post,
        rel_pre=rel_pre,
        rel_post=rel_post,
        rel_delta=rel_delta,
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
# SECTION 5b — SVD-CUTOFF CALIBRATION  (truth-free choice of cond)
# ═══════════════════════════════════════════════════════════════════════════


def calibration_sweep(
    res,
    seeds=range(1, 9),
    clearances=(0.25, 0.5),  # [m]
    conds=np.logspace(-6, -1, 61),
    kappa=2.0,
    half_window=0.15,
    band_tol=1.25,
    eps=0.02,
    floor_frac=0.1,
    od_alpha=0.10,
    cond_ref=None,
    verbose=True,
):
    """
    Sweep the truncated-SVD cutoff `cond` and choose it WITHOUT the true ΔM.

    The fit is linear, so one SVD per field-point draw gives the solution at
    every cutoff:  x(c) = Σ_{s_i > c·s_0} v_i (u_iᵀ Δb)/s_i.  Only Δb = b_post -
    b_pre is needed: the static background cancels, exactly as in the pipeline.
    Draws are `seeds` × `clearances` at res's site, basis and N_field.

    Truth-free rule (the geometric ΔM only VALIDATES it, printed separately):
      1. Admissible: CV(c) ≤ kappa·min_c CV(c) at every clearance.  CV is the
         held-out misfit: the draw-j solution predicts the Δ-field at the other
         draws' points (same clearance), relative misfit averaged over j.  It
         penalizes cutting real signal.  The data are noise-free, so CV has no
         minimum at small c — alone it would keep every mode.
      2. Instability I(c): relative std of ΔM pooled over all draws, all
         clearances, and cutoffs within ±half_window decades.  It penalizes a
         cutoff next to a rank step (one singular vector can move ΔM by ~20 %)
         and modes the sampling cannot pin down.
      3. c* = argmin I over the admissible cutoffs.  The stable band is the
         contiguous admissible run around c* with I ≤ band_tol·I(c*).
    ΔM is a staircase in c, so quote it over the band, not at a single cutoff.
    sig_delta is the delta-route OD σ_ΔM/|ΔM| (coefficient_difference_covariance)
    — not a selection input, shown for the resolution/noise trade-off.
    cond_ref (default res['cond']) is evaluated alongside for comparison.
    """
    conds = np.sort(np.asarray(conds, float))
    seeds, clearances = list(seeds), list(clearances)
    cond_ref = res["cond"] if cond_ref is None else float(cond_ref)
    c_all = np.append(conds, cond_ref)  # last column: the reference cutoff
    R, a, mm, nn = res["R_star"], res["alpha"], res["m_max"], res["n_max"]
    f = mass_functional(R, a, mm, nn, res["zeros_dict"])
    ev_pre = make_evaluable(res["mesh_pre"], res["density"])
    ev_post = make_evaluable(res["mesh_post"], res["density"])
    h_itp = RegularGridInterpolator(
        (res["gx"], res["gy"]), np.maximum(res["h_pre"], res["h_post"])
    )

    shape = (len(clearances), len(seeds), c_all.size)
    dM, rank, cv, sig, step = (np.empty(shape) for _ in range(5))
    q_rel = []  # s/s_0 of every draw, for the distance of c to a rank step
    t0 = time.time()
    for ic, cl in enumerate(clearances):
        draws = []
        for sd in seeds:
            rp, pp, zp, pts = make_cylinder_field_points(
                (res["cx"], res["cy"]), res["z_sheet"], R, res["H"], h_itp,
                clearance=cl / L_REF, N=res["N_field"], seed=sd,
            )
            assert (pts[:, 2] > h_itp(pts[:, :2])).all(), "field points intersect terrain"
            U0, gx0, gy0, gz0 = eval_gravity(ev_pre, pts)
            U1, gx1, gy1, gz1 = eval_gravity(ev_post, pts)
            gr0, gp0 = cart_to_cyl_g(gx0, gy0, pp)
            gr1, gp1 = cart_to_cyl_g(gx1, gy1, pp)
            A, _ = build_design_matrix(rp, pp, zp, res["R_alpha"], mm, nn)
            db = assemble_obs_vector(U1 - U0, gr1 - gr0, gp1 - gp0, gz1 - gz0)
            Us, s, Vt = np.linalg.svd(A, full_matrices=False)
            keep = s[None, :] > c_all[:, None] * s[0]  # lstsq's rule, per cutoff
            beta = np.divide(Us.T @ db, s, out=np.zeros_like(s), where=s > 0)
            X = Vt.T @ (keep * beta).T  # (n_coeff, n_cutoffs)
            draws.append((A, db, X))
            q_rel.append(s[s > 0] / s[0])
            j = len(draws) - 1
            dM[ic, j] = f @ X
            k = keep.sum(axis=1)
            rank[ic, j] = k
            # each singular vector's share of ΔM; a one-rank change at c adds
            # the first dropped share or removes the last kept one
            t = np.abs((Vt @ f) * beta)
            step[ic, j] = (np.maximum(t[np.minimum(k, t.size - 1)], t[np.maximum(k - 1, 0)])
                           / np.abs(dM[ic, j]))
            for i in range(c_all.size):
                sd_i = f * od_sigma(X[:, i], eps, floor_frac=floor_frac, alpha=od_alpha)
                sig[ic, j, i] = np.sqrt(np.sum(sd_i**2)) / abs(dM[ic, j, i])
        for j, (_, _, X) in enumerate(draws):
            cv[ic, j] = np.mean(
                [np.linalg.norm(Ao @ X - dbo[:, None], axis=0) / np.linalg.norm(dbo)
                 for o, (Ao, dbo, _) in enumerate(draws) if o != j],
                axis=0,
            )
        if verbose:
            print(f"    clearance {cl} {_U['len']}: {len(seeds)} draws "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # the reference cutoff rides along in the last column; the rule sees the grid
    dM_ref, rank_ref, sig_ref = dM[..., -1], rank[..., -1], sig[..., -1]
    step_ref = step[..., -1]
    dM, rank, cv, sig = dM[..., :-1], rank[..., :-1], cv[..., :-1], sig[..., :-1]
    step = step[..., :-1]
    ratio, ratio_ref = dM / res["dM_true"], dM_ref / res["dM_true"]  # validation only

    cv_med = np.median(cv, axis=1)  # (n_clearance, n_cond)
    cv_norm = cv_med / cv_med.min(axis=1, keepdims=True)
    admissible = cv_norm.max(axis=0) <= kappa
    if not admissible.any():
        raise RuntimeError("no cutoff passes the CV admissibility test; raise kappa")
    lc = np.log10(conds)
    instab = np.empty(conds.size)
    for i in range(conds.size):
        pool = dM[..., np.abs(lc - lc[i]) <= half_window + 1e-9].ravel()
        instab[i] = pool.std(ddof=1) / abs(pool.mean())
    idx = np.flatnonzero(admissible)
    i_star = int(idx[np.argmin(instab[idx])])
    ok = admissible & (instab <= band_tol * instab[i_star])
    lo = hi = i_star
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < conds.size - 1 and ok[hi + 1]:
        hi += 1
    band = np.zeros(conds.size, bool)
    band[lo : hi + 1] = True

    def step_gap(c):
        """Worst-draw distance (in ln s) from c·s_0 to the nearest singular value."""
        return min(np.min(np.abs(np.log(q / c))) for q in q_rel)

    sw = dict(
        conds=conds, clearances=clearances, seeds=seeds, dM=dM, ratio=ratio,
        rank=rank, cv=cv, cv_norm=cv_norm, sig_delta=sig, instab=instab,
        admissible=admissible, band=band, i_star=i_star, cond_star=float(conds[i_star]),
        band_range=(float(conds[lo]), float(conds[hi])), kappa=kappa,
        half_window=half_window, band_tol=band_tol, cond_ref=cond_ref,
        dM_ref=dM_ref, ratio_ref=ratio_ref, rank_ref=rank_ref, sig_ref=sig_ref,
        gap_star=step_gap(conds[i_star]), gap_ref=step_gap(cond_ref),
        step=step, step_star=float(step[..., i_star].max()),
        step_ref=float(step_ref.max()), dM_true=res["dM_true"],
    )
    if verbose:
        calibration_report(sw)
    return sw


def calibration_report(sw):
    """Print the sweep table, the truth-free choice, and its validation."""
    c, i_s, band = sw["conds"], sw["i_star"], sw["band"]
    dM, ratio, rank = sw["dM"], sw["ratio"], sw["rank"]
    print(f"\n{DASH}\n  SVD-CUTOFF CALIBRATION  (truth-free choice; truth only validates)\n{DASH}")
    print(
        f"    draws: seeds {sw['seeds'][0]}..{sw['seeds'][-1]} × clearances "
        f"{', '.join(f'{x:g}' for x in sw['clearances'])} {_U['len']}; "
        f"{c.size} cutoffs {c[0]:.0e}..{c[-1]:.0e}"
    )
    print(
        f"    rule : CV ≤ {sw['kappa']:g}·min CV at every clearance, then minimize ΔM "
        f"instability pooled over ±{sw['half_window']:g} decade"
    )
    print(
        f"    {'cond':>9} {'rank':>7} {'CV/min':>7} {'I(c) %':>7} "
        f"{'ΔM median':>11} {'σ_OD %':>7} | {'ratio':>6}  (ratio = validation)"
    )
    for i in range(c.size):
        tag = "c*" if i == i_s else ("band" if band[i] else
                                     ("" if sw["admissible"][i] else "rejected"))
        print(
            f"    {c[i]:9.2e} {int(rank[..., i].min()):3d}-{int(rank[..., i].max()):<3d} "
            f"{sw['cv_norm'][:, i].max():7.2f} {100*sw['instab'][i]:7.2f} "
            f"{np.median(dM[..., i]):+11.4e} {100*np.median(sw['sig_delta'][..., i]):7.2f} "
            f"| {np.median(ratio[..., i]):6.3f}  {tag}"
        )
    lo, hi = sw["band_range"]
    b_dM = dM[..., band].ravel()
    print(
        f"\n    c*   = {sw['cond_star']:.3e}  (rank {int(rank[..., i_s].min())}–"
        f"{int(rank[..., i_s].max())}; nearest singular value "
        f"{100*sw['gap_star']:.1f} % away, one rank step moves ΔM ≤ "
        f"{100*sw['step_star']:.1f} %, worst draw)"
    )
    print(f"    band = [{lo:.3e}, {hi:.3e}]  (admissible, I ≤ {sw['band_tol']:g}·I(c*))")
    print(
        f"    ΔM over the band (truth-free): median {np.median(b_dM):+.4e} "
        f"{_U['mass']}, spread {100*b_dM.std(ddof=1)/abs(b_dM.mean()):.2f} % "
        f"(all draws, clearances, cutoffs)"
    )
    print(
        f"    OD σ_ΔM at c* (delta route): median "
        f"{100*np.median(sw['sig_delta'][..., i_s]):.2f} % of ΔM"
    )

    def val(r):
        r = np.ravel(r)
        return (f"median {np.median(r):.3f}  [{r.min():.3f}, {r.max():.3f}]  "
                f"RMSE {100*np.sqrt(np.mean((r - 1) ** 2)):.1f} %")

    print("    validation against the geometric truth (NOT used above):")
    print(f"      at c*               : {val(ratio[..., i_s])}")
    print(f"      over the band       : {val(ratio[..., band])}")
    print(
        f"      at cond_ref {sw['cond_ref']:.2e}: {val(sw['ratio_ref'])}  (rank "
        f"{int(sw['rank_ref'].min())}–{int(sw['rank_ref'].max())}, nearest singular "
        f"value {100*sw['gap_ref']:.1f} % away, one rank step moves ΔM ≤ "
        f"{100*sw['step_ref']:.1f} %)"
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — DIAGNOSTIC PLOTS  (SI units)
# ═══════════════════════════════════════════════════════════════════════════


def plot_results(res, outdir=None):
    """Geometry, gravity change (3 panels) and recovered vs true Δσ (3 panels),
    each panel its own file.  Returns the list of figures."""
    R, H = res["R_star"], res["H"]
    cx, cy, z0 = res["cx"], res["cy"], res["z_sheet"]
    gx, gy = res["gx"], res["gy"]
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    rp, zp = res["rp"], res["zp"]
    RHO, PHI = res["RHO"], res["PHI"]
    Xp, Yp = RHO * np.cos(PHI), RHO * np.sin(PHI)
    sm, st = res["sigma_map"], res["sigma_true"]
    dU, dgz = res["dU"], res["dgz"]
    # |Delta g|, the magnitude of the acceleration change, rather than its
    # vertical component alone: the fit consumes all three components, so the
    # figure should show what changed, not one projection of it.
    dgvec = np.sqrt(
        (res["gr_post"] - res["gr_pre"]) ** 2
        + (res["gphi_post"] - res["gphi_pre"]) ** 2
        + dgz**2
    )

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    def _nice_3d_axes(ax):
        ax.set_facecolor("white")
        ax.grid(False)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor("white")
        ax.tick_params(labelsize=9 * FONT_SCALE)
        # 3-D labels sit beyond their tick labels; mplot3d's default 4 pt pad
        # puts them on the numbers once the text is scaled up for print
        ax.set_xlabel(f"$x$ [{_UL['len']}]", labelpad=LPAD3D)
        ax.set_ylabel(f"$y$ [{_UL['len']}]", labelpad=LPAD3D)
        # z needs a wider pad than x/y: its tick labels are the tallest numbers
        # on the plot ("17.5"), and at LPAD3D the label sits on top of them
        ax.set_zlabel(f"$z$ [{_UL['len']}]", labelpad=LPAD3D * 1.9)
        ax.view_init(elev=28, azim=-60)

    def _draw_cylinder(ax, color=ACCENT):
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
    # pad clears the z label, which mplot3d places outboard of the z ticks;
    # at 0.10 the bar sat on top of it
    fig1.colorbar(cbar_src, ax=ax, pad=0.17, shrink=0.65).set_label(
        rf"Surface Height Change $\Delta h$  [{_UL['len']}]"
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
                [], [], color=ACCENT, lw=1.8, label=r"Analysis cylinder $R^*,\,H$"
            )
        ],
        loc="upper right",
        fontsize=9 * FONT_SCALE,
    )
    _save(fig1, outdir, "fig1_geometry.pdf")

    # ── FIGURE 2 — what TAG changed ────────────────────────────────────
    # Everything static cancels in the difference, so the differences ARE the
    # measurement: the absolute field panel was showing the background the
    # estimator never sees.  The first two panels are the differenced
    # observables the fit consumes; the third is what they become after
    # projection onto the Bessel-Fourier basis, i.e. the vector the mass
    # functional acts on.  Three panels, one story, three files.
    def _scatter_cyl(ax, vals, title, label, diverging=True):
        if diverging:
            v = np.percentile(np.abs(vals), 98)
            kw = dict(cmap="RdBu_r", vmin=-v, vmax=v)
        else:  # a magnitude is one-signed, so a sequential map, not RdBu
            kw = dict(cmap="viridis", vmin=0.0, vmax=np.percentile(vals, 98))
        sc = ax.scatter(
            rp,
            zp,
            c=vals,
            s=14,
            lw=0,
            rasterized=True,
            **kw,
        )
        cb = ax.get_figure().colorbar(sc, ax=ax, pad=0.035, fraction=0.046)
        cb.set_label(label, fontsize=10 * FONT_SCALE)
        cb.ax.tick_params(labelsize=8 * FONT_SCALE)
        ax.axhline(0.0, color="0.35", ls="--", lw=1.1, zorder=1)  # the sheet plane
        # Equal-AREA radial axis.  The points are uniform in VOLUME, so their
        # number per unit rho grows as rho, and on a linear axis they look
        # bunched at the rim — an artefact of projecting a 3-D uniform cloud
        # onto (rho, z), not of the sampler.  Stretching x as rho^2 gives equal
        # areas equal widths, and a uniform cloud then looks uniform.  Checked:
        # equal-area rho bins hold 232-271 of 2000 points, and rho^2/R^2 passes
        # a KS test against uniform at D = 0.017.
        ax.set_xscale(
            "function",
            functions=(
                lambda r: np.maximum(r, 0.0) ** 2,
                lambda a: np.sqrt(np.maximum(a, 0.0)),
            ),
        )
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_xlabel(rf"$\rho$  [{_UL['len']}]  (equal-area scale)")
        ax.set_ylabel(rf"$z-z_0$  [{_UL['len']}]")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        for sd_ in ("top", "right"):
            ax.spines[sd_].set_visible(False)

    fig2a, ax = plt.subplots(figsize=FS)
    _scatter_cyl(
        ax, dU, r"Differenced potential  $\Delta U$", rf"$\Delta U$  [{_UL['pot']}]"
    )
    _save(fig2a, outdir, "fig2_gravity_change_potential.pdf")

    fig2b, ax = plt.subplots(figsize=FS)
    _scatter_cyl(
        ax,
        dgvec / ACC_SCALE,
        "",
        rf"$|\Delta \mathbf{{g}}|$  [{_UL['acc']}]",
        diverging=False,
    )
    _save(fig2b, outdir, "fig2_gravity_change_acceleration.pdf")

    # (c) the SPECTRUM of the differenced coefficients the two panels project
    # onto.  Same construction the GLOBAL scripts use for their CH panel: the
    # RMS over the radial modes at each azimuthal order, so one number per m
    # rather than 48 bars.
    fig2c, ax = plt.subplots(figsize=FS)
    dc = res["d_coeffs"]
    m_max, n_max = res["m_max"], res["n_max"]
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
        color=COLOR[0],
        alpha=0.20,
        lw=0,
        zorder=2,
        label=r"Spread over $n$ ($\times/\div\,1\sigma$)",
    )
    ax.plot(
        ms,
        rms_m,
        "-o",
        color=COLOR[0],
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
    ax.set_ylabel(rf"$|\Delta\mathbf{{CS}}_{{mn}}|$  [{_UL['pot']}]")
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5 * FONT_SCALE, loc="upper right", framealpha=0.92)
    for sd_ in ("top", "right"):
        ax.spines[sd_].set_visible(False)
    _save(fig2c, outdir, "fig2_gravity_change_spectrum.pdf")

    # ── FIGURE 3 — recovered vs true mass change ───────────────────────
    # Three panels, three files: the two maps and the error between them.  The
    # coefficient spectrum moved to figure 2, beside the differenced fields it
    # is the projection of; the numeric summary became the terminal's LaTeX
    # tables; and the azimuthal profiles were dropped — the radial cut already
    # carries the bandlimit story, and the map itself shows the azimuthal
    # structure better than three slices through it.
    vmax = max(np.percentile(np.abs(sm), 98), np.percentile(np.abs(st), 98))
    tc = np.linspace(0, 2 * np.pi, 200)

    # close the azimuthal seam for pcolormesh (φ wraps at 2π)
    def _wrap(a):
        return np.column_stack([a, a[:, :1]])

    Xw, Yw = _wrap(Xp), _wrap(Yp)
    # One colourbar per panel.  With the titles gone the bar labels are what
    # identify the panels, so each says explicitly which field it shows; and
    # all three use the same fraction/pad on equal-aspect axes, so the bars come
    # out the same size from file to file.
    CBAR = dict(fraction=0.046, pad=0.03)

    def _decor3(ax):
        """Footprint circle, equal aspect and axis labels, on every Δσ map."""
        ax.plot(R * np.cos(tc), R * np.sin(tc), "k--", lw=1.2, alpha=0.65)
        ax.set_aspect("equal")
        ax.set_xlabel(rf"$x-x_0$  [{_UL['len']}]")
        ax.set_ylabel(rf"$y-y_0$  [{_UL['len']}]")

    figs3 = []
    for tag, (mp, lab) in zip(
        ("estimated", "true"),
        [
            (sm, rf"Estimated $\Delta\sigma$  [{_UL['sd']}]"),
            (st, rf"True $\rho\,\Delta h$  [{_UL['sd']}]"),
        ],
    ):
        fig3, ax = plt.subplots(figsize=FS_MAP)
        c = ax.pcolormesh(
            Xw,
            Yw,
            _wrap(mp)[:-1, :-1],
            cmap="RdBu_r",
            shading="flat",
            vmin=-vmax,
            vmax=vmax,
        )
        _decor3(ax)
        fig3.colorbar(c, ax=ax, **CBAR).set_label(lab)
        _save(fig3, outdir, f"fig3_mass_change_{tag}.pdf")
        figs3.append(fig3)

    # third panel, drawn exactly like the first two: the ERROR map, recovered
    # minus true.  Its range is set by its own percentile rather than the maps'
    # shared one — the residual is far smaller than either field, and reusing
    # their scale would render it uniformly white.
    fig3C, ax3C = plt.subplots(figsize=FS_MAP)
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
    fig3C.colorbar(ce, ax=ax3C, **CBAR).set_label(
        r"$(\widehat{\Delta\sigma}-\Delta\sigma)\,/\,\max|\Delta\sigma|$  "
        + (r"[\%]" if USE_TEX else "[%]")
    )
    _decor3(ax3C)

    # (the old Summary text panel was a fourth cell of this figure; every number
    #  in it is now in the terminal's LaTeX tables — figures carry no numbers)
    _save(fig3C, outdir, "fig3_mass_change_error.pdf")

    # NO plt.show() here: it blocks, so anything created after this call would
    # be built and never displayed.  The caller shows every figure at the end.
    return [fig1, fig2a, fig2b, fig2c] + figs3 + [fig3C]


def plot_calibration_sweep(sw, outdir=None):
    """
    The cutoff sweep of `calibration_sweep`, as two standalone panels:
      fig6_calibration_ratio    — recovered/true ΔM per draw (validation view)
      fig6_calibration_criteria — the truth-free criteria and the retained rank
    Both mark the rejected cutoffs (grey), the stable band (shaded), c* (solid)
    and the reference cutoff (dotted).  Numbers go to calibration_report.
    """
    c = sw["conds"]
    n_cl = len(sw["clearances"])
    names = (["Lower clearance", "Higher clearance"] if n_cl == 2
             else [f"Clearance {k + 1}" for k in range(n_cl)])
    cols = [COLOR[0], COLOR[2], COLOR[3], COLOR[4]]
    edges = np.concatenate([[c[0]], np.sqrt(c[1:] * c[:-1]), [c[-1]]])
    rej = np.diff(np.concatenate([[0], (~sw["admissible"]).astype(int), [0]]))
    rej_runs = list(zip(np.flatnonzero(rej == 1), np.flatnonzero(rej == -1)))
    lo, hi = sw["band_range"]

    def frame(ax):
        for k, (i0, i1) in enumerate(rej_runs):
            ax.axvspan(edges[i0], edges[i1], color="0.87", lw=0, zorder=0,
                       label="Rejected: held-out misfit" if k == 0 else None)
        ax.axvspan(lo, hi, color=ACCENT, alpha=0.14, lw=0, zorder=0.5,
                   label="Stable band")
        ax.axvline(sw["cond_star"], color=ACCENT, lw=1.8, zorder=3,
                   label=r"Selected cutoff $c^\ast$")
        ax.axvline(sw["cond_ref"], color="k", ls=":", lw=1.3, zorder=3,
                   label="Reference cutoff")
        ax.set_xscale("log")
        ax.set_xlim(c[0], c[-1])
        ax.set_xlabel(r"SVD cutoff $c$, relative to $s_{\max}$  [-]")
        ax.set_axisbelow(True)

    # (a) recovered/true mass for every draw: the staircase in c
    fig_r, ax = plt.subplots(figsize=FS)
    frame(ax)
    for k in range(n_cl):
        for j in range(sw["ratio"].shape[1]):
            ax.plot(c, sw["ratio"][k, j], color=cols[k], lw=0.6, alpha=0.35, zorder=1.5)
        ax.plot(c, np.median(sw["ratio"][k], axis=0), color=cols[k], lw=2.4, zorder=4,
                label=f"{names[k]}, median of draws")
    ax.axhline(1.0, color="k", lw=0.9, zorder=2)
    ax.set_ylabel(r"$\Delta M / \Delta M_{\mathrm{true}}$  [-]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8 * FONT_SCALE, loc="lower left")
    _save(fig_r, outdir, "fig6_calibration_ratio.pdf")

    # (b) what the rule sees — no truth anywhere on this panel
    fig_c, ax = plt.subplots(figsize=FS)
    frame(ax)
    ax.plot(c, sw["instab"], color=COLOR[0], lw=2.2, zorder=4,
            label=r"Instability $I(c)$")
    ax.plot(c, sw["cv_norm"].max(axis=0), color=COLOR[2], lw=2.2, zorder=4,
            label=r"Held-out misfit $\mathrm{CV}/\mathrm{CV}_{\min}$")
    ax.axhline(sw["kappa"], color=COLOR[2], lw=1.0, ls="--", zorder=2,
               label=r"Admissibility threshold $\kappa$")
    ax.plot(c, np.median(sw["sig_delta"], axis=(0, 1)), color=COLOR[3], lw=1.6,
            ls="-.", zorder=4, label=r"OD $\sigma_{\Delta M}/|\Delta M|$")
    ax.set_yscale("log")
    ax.set_ylabel("Criterion  [-]")
    ax.tick_params(axis="y", which="both", right=False)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()  # drawn above ax, so the grey spans do not hide the rank
    ax2.step(c, np.median(sw["rank"], axis=(0, 1)), where="mid", color="0.35", lw=1.1)
    ax2.set_ylabel("Retained singular values  [-]", color="0.35")
    ax2.tick_params(axis="y", which="both", colors="0.35")
    ax.legend(fontsize=8 * FONT_SCALE, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=2)
    _save(fig_c, outdir, "fig6_calibration_criteria.pdf")
    return [fig_r, fig_c]


if __name__ == "__main__":

    setup = dict(
        path_pre="3dmeshes/Bennu_preTag.obj",
        path_post="3dmeshes/Bennu_afterTag.obj",
        density=RHO_BULK,  # [kg/m³]
        grid_res=0.30 / L_REF,
        site_center=None,  # auto-detect TAG crater from Δh
        # The cutoff sweep holds this footprint and cylinder height fixed.
        # Changing either alters the inversion; the sweep below re-runs anyway.
        R_star=R_STAR_SI / L_REF,
        H=16.0 / L_REF,
        clearance=0.25 / L_REF,  # points hug the surface: local terrain + this
        alpha=CH_ALPHA,  # 3.0: fictitious radial boundary at 24 m
        m_max=CH_M_MAX,  # 5: azimuthal orders 0..4
        n_max=CH_N_MAX,  # 10 radial modes per azimuthal order
        N_field=2000,
    )

    # 1. Choose the SVD cutoff without the true ΔM (Section 5b).  The quick
    #    run only supplies the site, meshes and basis; its cutoff is unused.
    site = run_bennu_tag(**setup, do_covariance=False, verbose=False)
    sweep = calibration_sweep(site, cond_ref=3.1e-3)  # earlier truth-tuned cutoff
    figs = plot_calibration_sweep(sweep, outdir="Images")

    # 2. Full run at the selected cutoff.
    result = run_bennu_tag(
        **setup,
        cond=sweep["cond_star"],
        n_ensemble=5,  # field draws averaged; their ΔM spread is reported
        cov_route="delta",  # OD σ on the recovered change ΔCS, not per epoch
        coeff_rel=0.02,  # od_sigma eps on ΔCS
        coeff_floor_frac=0.1,
        od_alpha=0.10,  # index-based growth from GLOBAL.od_sigma; 0 disables it
        verbose=True,
    )

    # Draw coefficients with the assumed OD covariance; reuse the same Monte
    # Carlo for the tables and figures.  No field samples are redrawn.
    result["cov"]["mc"] = covariance_mc(result, result["cov"])

    latex_tables(result, result.get("cov"))

    figs += plot_results(result, outdir="Images")
    figs += plot_covariance_mc(
        result, result["cov"], outdir="Images", mc=result["cov"]["mc"]
    )[0]

    plt.show()  # once, with every panel built, so all of them appear

    print("\nDone.")
