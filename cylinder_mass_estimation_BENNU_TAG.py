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
    cylinder there; the expansion plane (the "sheet") sits at the signed
    vertical centroid of the differential slab inside the footprint.  This
    puts the thin-sheet reference at the centre of the mass that moved rather
    than above it at the mean pre-TAG terrain height.  Field points are drawn
    UNIFORMLY over the
    cylinder's vacuum volume — constant density per unit volume, no coordinate
    over-sampled (see `make_cylinder_field_points`) — with a lower bound that
    follows the LOCAL terrain (small clearance, point-by-point).  The plane
    sits inside the terrain's height range, so inside the crater bowl a few
    points fall slightly BELOW it (zp ≈ −0.2 m at worst for the default draws);
    exp(−k zp) stays bounded there because k_max·|zp| is small.
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

    The cutoff can be chosen WITHOUT the true ΔM by `calibration_sweep`: over a
    log grid of cutoffs, several field draws and two clearances, reject
    cutoffs whose held-out misfit (fit on one draw, predict the others)
    exceeds kappa x the best, then take the admissible cutoff where ΔM is
    most stable against draw, clearance and a small change of cutoff.  The
    true ΔM is only reported next to it, as validation, in the terminal.  These
    are sampling checks on the same terrain, not validation on new physical
    data.
    Uniform volume sampling does not imply an unbiased inverse.
    `basis_calibration` applies the same rule with the basis (alpha, m_max,
    n_max) as extra axes and, among the equally good pairs, takes the basis
    with the fewest coefficients.  CH_MODE="calibrated" (default) runs it for
    the current setup (cached, recomputed when any setting changes); "fixed"
    uses CH_ALPHA, CH_M_MAX, CH_N_MAX, CH_COND, which were NOT chosen truth-free
    (see the note at CH_COND).  The basis ablation keeps the chosen cutoff.
7.  Wahr-like inversion of ΔA -> ΔM and Δσ(ρ,φ), from the ONE fit on the
    N_field points of step 5 — not an average of fits on smaller draws
    (measured equivalence and the N_field choice: note in Section 5b).
8.  Geometric ground truth: ΔM_true = ρ_bulk ∫∫ Δh dA over the footprint,
    Δσ_true = ρ_bulk Δh — a direct validation of the inversion.
9.  Assumed OD uncertainty via GLOBAL.od_sigma (CH mode: the growth factor
    rides the azimuthal order m), booked on EACH epoch's full CS and
    kept separate: Σ_ΔCS = diag(σ_pre² + σ_post² − 2k·σ_pre·σ_post), default
    k = 0 (independent epochs).  od_sigma is relative and anchored per block,
    so this σ responds to any field common to both epochs even though ΔM does
    not; covariance_report prints the sweep in k and a point-mass background
    check that measures the size of that response rather than assuming it.  Σ_ΔCS is propagated
    exactly to ΔM and Δσ.
10. Monte Carlo draws ΔCS from N(CS_post-CS_pre, Σ_ΔCS), with fixed geometry,
    and inverts those coefficient realizations.  It checks the propagated
    noise dispersion, not the terrain, truncation or thin-sheet model bias.
11. Basis ablation: sweep alpha, m_max and n_max with the selected cutoff and
    all other settings fixed.  Reuse the same field draws across every basis.
    fig7_basis_mass_ratio shows the signed relative mass error
    100*(estimated - true)/true, with included wavelength ranges in each
    panel.  Full values, field-sampling scatter and settings stay in memory;
    truth is not a basis-selection rule.

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
from scipy.linalg import lstsq, qr
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)
import matplotlib.pyplot as plt
import matplotlib as mpl
from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable
import time, os, io, warnings
from concurrent.futures import ThreadPoolExecutor

# Reuse the GLOBAL experiment's assumed per-coefficient OD uncertainty rule.
# Import before this module's rcParams setup so its plotting defaults win here.
from cylinder_mass_estimation_GLOBAL import od_sigma
from cylinder_mass_estimation_GLOBAL import OD_RISE_DEC as G_OD_RISE_DEC

# TODO: How much SH would do here? Is CH needed?
# TODO: Justify a for od_sigma

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

# ΔU over the TAG field is ~1e-7 m²/s².  Carrying that decade in the LABEL,
# the way µGal carries it for |Δg|, keeps matplotlib's floating "1e-7" offset
# text off the colourbar.
POT_SCALE = 1e-7 if MODE == "SI" else 1.0
_POT_LBL = rf"$10^{{-7}}$ {_UL['pot']}" if MODE == "SI" else _UL["pot"]

# Okabe-Ito, the colour-vision-deficiency-safe palette used by the GLOBAL
# scripts, in the same role order.
COLOR = ["#D55E00", "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9"]
ACCENT = "#882255"  # structural elements (the analysis cylinder), as in GLOBAL
CH_VIOLET = "#5D3A9B"  # the CH case colour of the GLOBAL scripts


# Raw unweighted SI fit at R*=8 m, H=16 m, N_field=2000.  cond is relative to
# the largest singular value.  ΔM is a STAIRCASE in cond (one step per
# retained singular value) and depends on the basis, so basis and cutoff are
# chosen together WITHOUT the true ΔM by `basis_calibration` (Section 5b):
# among (basis, cutoff) pairs whose held-out misfit is within kappa of the
# best, keep those where ΔM is nearly as stable (field draws, clearance, the
# cutoff itself) as the most stable pair, and take the fewest coefficients.
# Only bases meeting the Δσ-map bandlimit k_max·R* ≥ KR_TARGET_DEFAULT (which
# run_bennu_tag enforces) can be chosen.
#
# CH_MODE="calibrated" runs it for the CURRENT setup via `calibrated_basis`
# (cached per configuration, recomputed whenever any setting changes) and
# IGNORES the four CH_* values below.  CH_MODE="fixed" uses them as written.
# CH_COND is only meaningful for the configuration it was calibrated on.
CH_ALPHA = 4.0
CH_M_MAX = 8
CH_N_MAX = 13
CH_COND = 10.0 ** (-2.25)
CH_MODE = "calibrated"  # "calibrated" (truth-free basis + cutoff) or "fixed" (CH_*)

# ── TERRAIN REFINEMENT (Section 4a) ─────────────────────────────────────────
# `wahr_invert` assumes the moved mass is a sheet ON the plane z = z_sheet.
# It is not: the slab centroid z_c = (h_pre+h_post)/2 wanders ±1.25 m (1σ)
# about the plane, and each mode is amplified by e^{+k(z_c−z_sheet)}.  That
# height SPREAD — not a mean offset, the signed centroid already sits at
# z_sheet by construction — costs ~9 points of map error (measured: sources
# forced onto the plane invert to 6.4 %, the real 3-D slab to 16.5 %), and no
# choice of plane removes it (sweeping z_sheet only makes it worse).
# MAP_REFINE_ITERS > 0 runs damped Richardson passes that use `wahr_invert`
# itself as the preconditioner and the KNOWN pre-TAG DTM as the sheet graph.
# ΔM is NOT touched: it stays the plain `wahr_invert` value of Section 6.
# `terrain_refine_map` records the per-pass map error and data residual of the
# run itself; fig6 plots them (see `plot_refinement`).  On the calibrated basis
# (α=5, m=9, n=16, N_field=10000) both settle by pass ~7 and sit at their
# plateau minimum at 8; further passes only trade decimals.  The map's own
# ΔM is NOT the stopping criterion: it swings between −3 % and +8 % from
# pass to pass, as does the residual
# (0.03–0.08).  The damped iteration cycles about the solution rather than
# converging to it — one more reason the REPORTED ΔM stays the linear
# flat-sheet functional of Section 6 and is never taken from this map.
MAP_REFINE_ITERS = 8  # 0 disables the refinement (plain flat-sheet map)
MAP_REFINE_DAMPING = 0.5  # w = 1 oscillates; 0.5 converges monotonically

# Candidate grid of the calibration (basis_calibration).
# Keep n_max up to 24: the pick is only grid-independent from n_max ≥ 18
# (smaller grids cannot resolve alpha ≥ 5 at the bandlimit; basis_calibration
# refuses them).  This grid: (5,9,16) c=1.47e-3, validation −3.1 %.
CAL_ALPHAS = (3.0, 4.0, 5.0, 6.0, 7.0)
CAL_M_MAX = tuple(range(1, 13))
CAL_N_MAX = tuple(range(2, 25))

# Basis ablation grid.  The cutoff is calibrated ONCE on the main basis, then
# frozen along with geometry, clearance, N_field and the field-draw seeds.
# alpha=1 places the Dirichlet boundary at the footprint edge for this ablation.
BASIS_SWEEP_ALPHAS = tuple(range(1, 51))
BASIS_SWEEP_M_MAX = tuple(range(1, 21))
BASIS_SWEEP_N_MAX = tuple(range(1, 25))

# Three common field draws expose sampling sensitivity while avoiding the
# five-draw cost of the nominal result.  Alpha blocks are independent; use the
# available CPU workers without changing the fit or cutoff mathematics.
BASIS_SWEEP_N_DRAWS = 3
BASIS_SWEEP_WORKERS = min(12, os.cpu_count() or 1)

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

# Diagnostic panels are standalone.  The basis sweep uses small multiples,
# as in GLOBAL_pt2, so different alpha values share axes and a colour scale.
FS = (7.2, 5.4)  # default standalone panel (same as the GLOBAL scripts)
FS_MAP = (6.4, 5.4)  # equal-aspect map with its own colour bar


# see the GLOBAL scripts: 3-D axis labels need a pad that tracks the font scale
LPAD3D = 10 * FONT_SCALE


def _save(fig, outdir, name, pad=None):
    """Tight-layout and tight-crop one standalone panel to `outdir/PREFIX+name`;
    no-op if outdir is None.  As in GLOBAL, except that constrained-layout
    figures (the basis-sweep small multiples) keep their own layout engine.

    No dpi here: `savefig.dpi` (300) is set in the rcParams block above, and only
    rasterized content (the 3-D surfaces, the rasterized maps) is affected.
    """
    if not outdir:
        return
    os.makedirs(outdir, exist_ok=True)
    if fig.get_layout_engine() is None:
        with warnings.catch_warnings():  # 3-D axes: "not compatible" is harmless
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
    kw = {"pad_inches": pad} if pad is not None else {}
    buf = io.BytesIO()
    fmt = os.path.splitext(name)[1][1:] or "pdf"
    fig.savefig(buf, format=fmt, bbox_inches="tight", **kw)
    _write_bytes(os.path.join(outdir, PREFIX + name), buf.getvalue())


def _write_bytes(path, data, retries=3):
    """Write a rendered figure to disk in one call, via a temp file.

    Streaming savefig straight into this OneDrive (CloudStorage) folder can die
    with TimeoutError [Errno 60] while the file provider is busy, leaving a
    corrupt PDF and killing the run before plt.show().  Figures are rendered in
    memory, written to `path.part` and renamed; a busy provider is retried, and
    a final failure only warns so the remaining figures are still shown.
    """
    tmp = path + ".part"
    for attempt in range(retries):
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return
        except OSError as err:
            if attempt == retries - 1:
                warnings.warn(f"could not save {path}: {err}")
                return
            time.sleep(2.0 * (attempt + 1))


def _polar_closed(A):
    """A polar-grid field (n_rho, n_phi) ready for a gouraud map: the azimuthal
    seam closed (φ wraps at 2π) and a centre row added at ρ = 0 (the ring mean),
    so the map has no pinhole inside the first ring."""
    A = np.column_stack([A, A[:, :1]])
    with warnings.catch_warnings():  # all-NaN first ring (masked error map)
        warnings.simplefilter("ignore", RuntimeWarning)
        c = np.nanmean(A[0])
    return np.vstack([np.full((1, A.shape[1]), c), A])


# GLOBAL figure conventions, shared by every panel below
CBAR = dict(pad=0.02, fraction=0.030)  # colour bar of a standalone map


def _grid(ax, **kw):
    """GLOBAL's grid: dotted, major and minor, behind the data."""
    ax.grid(True, which="both", ls=":", alpha=0.45, **kw)
    ax.set_axisbelow(True)


def _sigma_shade(w):
    """Colour for the noise gradient under a coefficient spectrum, `w` running
    from 1 at the 3-sigma edge to 0 at the axis (GLOBAL `_sigma_shade`).

    A blend between two cool grey-blues rather than a matplotlib colormap: the
    ramp has to sit UNDER coloured data without competing with it, so both ends
    are desaturated and the span is narrow.
    """
    lo = np.array([0.760, 0.800, 0.870])  # near the axis, the darkest
    hi = np.array([0.945, 0.957, 0.976])  # at 3 sigma, almost white
    return tuple(lo + (hi - lo) * float(np.clip(w, 0.0, 1.0)))


def sigma_bands(ax, xs, y_sig, pad=0.4):
    """Shade the 1σ, 2σ and 3σ noise levels under a coefficient spectrum and
    set the x-limits to match (GLOBAL `sigma_bands`, same figure grammar).

    A CONTINUOUS ramp, not three flat steps: shading the noise as a gradient
    that darkens toward the axis says "more likely here" without drawing edges
    the eye reads as features.  Built from many thin fills because
    `fill_between` takes one colour — 64 of them is visually continuous at
    print resolution — with a hairline at 1, 2 and 3 sigma so the levels can
    still be READ off, each named once just outside the right spine.

    Carried FLAT to the frame, `pad` beyond the first and last group: stopped
    at the groups, the fill left a blank strip at either end of the axis.
    """
    xs = np.asarray(xs, float)
    y_sig = np.asarray(y_sig, float)
    xe = np.r_[xs[0] - pad, xs, xs[-1] + pad]
    ye = np.r_[y_sig[0], y_sig, y_sig[-1]]
    n_ramp = 64
    for t in np.linspace(3.0, 3.0 / n_ramp, n_ramp):
        ax.fill_between(
            xe,
            0.0,
            t * ye,
            facecolor=_sigma_shade(t / 3.0),  # 1 at the outer edge, ~0 at axis
            edgecolor="none",
            zorder=1,
        )
    for k in (3, 2, 1):
        ax.plot(xe, k * ye, color="#9aa6bd", lw=0.6, zorder=1)
        # outside the frame: inside, the label would sit on its own hairline
        ax.annotate(
            rf"{k}$\sigma$",
            xy=(1.0, k * y_sig[-1]),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8 * FONT_SCALE,
            color="0.45",
            ha="left",
            va="center",
            annotation_clip=False,
        )
    ax.set_xlim(xe[0], xe[-1])


def hist_legend(ax, ncol=2):
    """Legend ABOVE the axes, never on the data (GLOBAL `hist_legend`)."""
    ax.legend(
        fontsize=8 * FONT_SCALE,
        ncol=ncol,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01, 1.0, 0.2),
        mode="expand",
        borderaxespad=0.0,
        handletextpad=0.5,
        columnspacing=1.2,
    )


def _map(ax, X, Y, Z, cmap="RdBu_r", vmin=None, vmax=None, bold=0.0):
    """GLOBAL map: gouraud colour, thin black contours at ten equal steps, and
    one bold contour at `bold` (the zero of a signed field; None for none)."""
    c = ax.pcolormesh(
        X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud", rasterized=True
    )
    lv = np.linspace(vmin, vmax, 11)[1:-1]
    if bold is not None:
        lv = lv[~np.isclose(lv, bold, atol=1e-9 * (vmax - vmin))]
    Zc = np.ma.masked_invalid(Z)
    ax.contour(X, Y, Zc, levels=lv, colors="k", linewidths=0.45, alpha=0.35, zorder=2)
    if bold is not None and Zc.min() < bold < Zc.max():
        ax.contour(
            X, Y, Zc, levels=[bold], colors="k", linewidths=1.3, alpha=0.8, zorder=2
        )
    return c


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


def locate_tag_site(dh, GX, GY):
    """
    TAG-site centre = |Δh|-weighted centroid of the excavated (Δh < 0)
    region, over the whole common grid (no border strip is excluded).
    """
    w = np.clip(-dh, 0.0, None)
    cx = (GX * w).sum() / w.sum()
    cy = (GY * w).sum() / w.sum()
    return cx, cy


def differential_sheet_plane(h_pre, h_post, foot, fallback=None):
    """Height z0 of the thin sheet that best stands in for the mass that moved.

    The Wahr inversion collapses the change onto ONE horizontal plane, so the
    plane should sit where that mass actually is.  Per grid column (area dA)
    the moved material is the slab between h_pre and h_post:

        signed volume   dA·Δh              (Δh = h_post − h_pre; < 0 if dug)
        first moment    dA·∫ z dz = dA·(h_post² − h_pre²)/2

    Summing over the footprint and dividing gives the height of the centre of
    mass of the (net) removed/added material — the same as the centre of mass
    of a body: Σ(z·m)/Σm.  Example: a column dug from 3 m to 2 m contributes
    Δh = −1 and moment (4 − 9)/2 = −2.5, i.e. its material sat at 2.5 m, the
    middle of the removed slab.  The pre-TAG mean height (the old choice) is
    the TOP of an excavation, so it sits ~0.5 m above the moved mass here.

    Because Σ(z·m)/Σm uses SIGNED masses, excavation and deposition partly
    cancel in both sums; when they nearly balance the ratio can land anywhere.
    It then falls back to the |Δh|-weighted mid-height of the changed columns,
    which always lies inside the changed terrain.  The same fallback is used if
    the signed centroid leaves that terrain's height range.
    """
    hp = np.asarray(h_pre, float)[foot]
    hq = np.asarray(h_post, float)[foot]
    dh = hq - hp
    denom = float(np.sum(dh))
    scale = float(np.sum(np.abs(dh)))
    if not np.isfinite(denom) or scale == 0.0:
        if fallback is None:
            raise ValueError("differential slab has no finite volume change")
        return float(fallback)
    w = np.abs(dh)
    z_abs = float(np.average(0.5 * (hp + hq), weights=w))
    if abs(denom) <= 1.0e-3 * scale:
        return z_abs
    z0 = 0.5 * float(np.sum(hq * hq - hp * hp)) / denom
    moved = w > 0.0
    lo = float(np.minimum(hp, hq)[moved].min())
    hi = float(np.maximum(hp, hq)[moved].max())
    if not (np.isfinite(z0) and lo <= z0 <= hi):
        return z_abs
    return float(z0)


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
    rp, pp, zp : cylindrical coords about the axis; zp is height relative
                 to the sheet plane (this is the z that enters exp(−k z));
                 inside the crater bowl zp may be slightly negative   [m]
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

# At fixed n_max, k_max ∝ 1/α: pushing the Dirichlet boundary αR*
# outward coarsens the basis.  To keep the bandlimit, grow n_max ∝ α, which
# `required_n_max` (n_max="auto") does.
# Basis wavenumbers and characteristic radial wavelengths:
#   k_mn = j_{m,n} / (α R*)            [1 / working length unit]
#   λ_mn = 2π / k_mn                   [working length unit]
# Included indices: m = 0..m_max-1, n = 1..n_max.  Hence:
#   k_max = j_{m_max-1,n_max} / (α R*)
#   λ_min = 2π / k_max = 2π α R* / j_{m_max-1,n_max}
#   k_min = j_{0,1} / (α R*)
#   λ_max = 2π / k_min = 2π α R* / j_{0,1}
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
# NOTE: Slepian doesn’t provide much benefit here: it produces essentially the
# same solution, but in a reduced-dimensional basis by removing poorly
# concentrated/low-information modes.


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
# SECTION 4a — TERRAIN REFINEMENT OF THE Δσ MAP: the flat-sheet bias, removed
# ═══════════════════════════════════════════════════════════════════════════
# `wahr_invert` inverts the flat-sheet forward operator W: a sheet Δσ sitting
# exactly on z = z_sheet.  The truth is a sheet on the GRAPH
#
#       z = ζ(ρ,φ) = z_sheet + (h_pre − z_sheet) + Δh/2 ,      Δh = Δσ/ρ_bulk
#
# i.e. the mid-surface of the excavated/deposited slab.  Call that forward
# operator T.  Mode by mode T = W·e^{+k_mn (ζ−z_sheet)}, so T ≠ W and the
# flat-sheet map is biased wherever ζ ≠ z_sheet.  Deconvolving the gain
# directly (inverting T on the coefficients) is hopeless — the gain matrix has
# a condition number ~10³ here and blows the map up to 86-100 % error.
#
# Richardson iteration avoids that inverse entirely.  W is used as a FIXED
# preconditioner for T:
#
#       Δσ⁰      = W d                                    (plain wahr_invert)
#       Δσ^{i+1} = Δσ^i + w · W (d − T[Δσ^i])             (w = damping)
#
# Every step is one `wahr_invert` call on a residual data vector, so the
# analysis stays exactly the Wahr thin-sheet inverse; the terrain enters only
# through the FORWARD synthesis T, which needs h_pre (known before the event)
# plus the current Δσ estimate for the Δh/2 half-thickness.  Undamped (w = 1)
# the iteration oscillates; w = 0.5 converges monotonically in ~5 passes.
def sheet_forward(sigma, zeta, RHO, PHI, centre, z_sheet, rp, pp, zp, chunk=64):
    """
    Gravity of a thin sheet of surface density `sigma` lying on the graph
    z = z_sheet + zeta(ρ,φ), evaluated at the cylinder field points
    (rp, pp, zp) and packed in the same interleaved [U, gρ, gφ, gz] order as
    `assemble_obs_vector`.  Direct point-mass summation over the polar cells,
    dm = Δσ · ρ Δρ Δφ, so it is the operator T described above with no
    band limit of its own.
    """
    cx, cy = centre
    d_rho = float(RHO[1, 0] - RHO[0, 0])
    d_phi = float(2.0 * np.pi / RHO.shape[1])
    dm = (sigma * RHO * d_rho * d_phi).ravel()
    src = np.column_stack(
        [
            (cx + RHO * np.cos(PHI)).ravel(),
            (cy + RHO * np.sin(PHI)).ravel(),
            (z_sheet + zeta).ravel(),
        ]
    )
    P = np.column_stack([cx + rp * np.cos(pp), cy + rp * np.sin(pp), z_sheet + zp])
    out = np.zeros(4 * len(P))
    for i in range(0, len(P), chunk):
        sl = slice(i, i + chunk)
        d = P[sl, None, :] - src[None]
        r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        ir3 = G_W / r**3
        gx_c = -(ir3 * d[..., 0]) @ dm
        gy_c = -(ir3 * d[..., 1]) @ dm
        gz_c = -(ir3 * d[..., 2]) @ dm
        cos_p, sin_p = np.cos(pp[sl]), np.sin(pp[sl])
        out[4 * i + 0 : 4 * (i + len(r)) : 4] = (G_W / r) @ dm
        out[4 * i + 1 : 4 * (i + len(r)) : 4] = gx_c * cos_p + gy_c * sin_p
        out[4 * i + 2 : 4 * (i + len(r)) : 4] = -gx_c * sin_p + gy_c * cos_p
        out[4 * i + 3 : 4 * (i + len(r)) : 4] = gz_c
    return out


def terrain_refine_map(
    sigma_map,
    RHO,
    PHI,
    field_draws,
    centre,
    z_sheet,
    h_pre_itp,
    density,
    R_star,
    alpha,
    m_max,
    n_max,
    zeros_dict,
    cond=CH_COND,
    n_iter=MAP_REFINE_ITERS,
    damping=MAP_REFINE_DAMPING,
    sigma_true=None,  # VALIDATION ONLY: per-pass map error for the fig6 curve
    verbose=True,
):
    """
    Damped Richardson terrain refinement of the `wahr_invert` map (Section 4a).

    `field_draws` is the list [(rp, pp, zp, db), ...] of Section 5b — normally
    the run's single draw; db is its DIFFERENCED observation vector.  Each pass forward-
    models the current Δσ on the terrain graph, feeds the data residual back
    through the very same `fit_coefficients` + `wahr_invert` chain, and adds a
    damped share of the correction.

    Returns (sigma_refined, history, dM_corr, diag).  history[i] = residual
    ‖d−T Δσ‖/‖d‖ of the map after i passes, averaged over the draws — a
    truth-free convergence diagnostic, n_iter+1 long (pass 0 is the flat map,
    pass n_iter the returned one).  dM_corr is the matching correction to
    `wahr_invert`'s ΔM, accumulated from the same residual fits; it is
    reported as a DIAGNOSTIC only, since Section 4b propagates the LINEAR
    flat-sheet functional.  `diag` carries the same per-pass record in plot
    form — and, when `sigma_true` is given, the map error against it; that
    truth NEVER enters the iteration, it is only recorded alongside it.
    """
    if n_iter <= 0:
        return sigma_map, [], 0.0, {}

    R_alpha = alpha * R_star
    XY = np.column_stack(
        [
            (centre[0] + RHO * np.cos(PHI)).ravel(),
            (centre[1] + RHO * np.sin(PHI)).ravel(),
        ]
    )
    zeta_pre = h_pre_itp(XY).reshape(RHO.shape) - z_sheet  # known pre-TAG graph
    designs = [
        build_design_matrix(rp, pp, zp, R_alpha, m_max, n_max)[0]
        for rp, pp, zp, _ in field_draws
    ]
    norm0 = float(np.mean([np.linalg.norm(db) for *_, db in field_draws]))

    sigma = np.array(sigma_map, float)
    peak = float(np.max(np.abs(sigma_true))) if sigma_true is not None else np.nan

    def _err(sig):
        """|error| median/p90 and RMSE of one map, in % of max|Δσ_true|."""
        if sigma_true is None:
            return np.nan, np.nan, np.nan
        e = 100.0 * (sig - sigma_true) / peak
        return (
            float(np.median(np.abs(e))),
            float(np.percentile(np.abs(e), 90)),
            float(np.sqrt(np.mean(e**2))),
        )

    history, dM_corr = [], 0.0
    med, p90, rmse, dM_track = [], [], [], []
    for it in range(1, n_iter + 1):
        for lst, v in zip((med, p90, rmse), _err(sigma)):
            lst.append(v)
        dM_track.append(dM_corr)
        # mid-slab graph: pre-TAG surface plus half the current thickness
        zeta = zeta_pre + 0.5 * sigma / density
        corr, res_norm, dM_step = [], [], []
        for A_i, (rp, pp, zp, db) in zip(designs, field_draws):
            res = db - sheet_forward(sigma, zeta, RHO, PHI, centre, z_sheet, rp, pp, zp)
            res_norm.append(np.linalg.norm(res))
            c_res = fit_coefficients(
                A_i, res[0::4], res[1::4], res[2::4], res[3::4], cond=cond
            )[0]
            dM_i, sig_i, _, _ = wahr_invert(
                c_res, R_star, alpha, m_max, n_max, zeros_dict, *RHO.shape
            )
            corr.append(sig_i)
            dM_step.append(dM_i)
        sigma = sigma + damping * np.mean(corr, axis=0)
        dM_corr += damping * float(np.mean(dM_step))
        history.append(float(np.mean(res_norm) / (norm0 + 1e-30)))
        if verbose:
            print(f"      pass {it}: entering ‖d − T Δσ‖/‖d‖ = {history[-1]:.3f}")

    # Close the record on the map the run actually keeps: one more forward
    # model, no fit — otherwise every curve would stop one pass short.
    for lst, v in zip((med, p90, rmse), _err(sigma)):
        lst.append(v)
    dM_track.append(dM_corr)
    zeta = zeta_pre + 0.5 * sigma / density
    history.append(
        float(
            np.mean(
                [
                    np.linalg.norm(
                        db
                        - sheet_forward(
                            sigma, zeta, RHO, PHI, centre, z_sheet, rp, pp, zp
                        )
                    )
                    for rp, pp, zp, db in field_draws
                ]
            )
            / (norm0 + 1e-30)
        )
    )
    if verbose:
        print(f"      after {n_iter} passes: ‖d − T Δσ‖/‖d‖ = {history[-1]:.3f}")
    diag = {
        "pass": list(range(n_iter + 1)),
        "residual": list(history),
        "median": med,
        "p90": p90,
        "rmse": rmse,
        "dM_corr": dM_track,
        "damping": float(damping),
    }
    if sigma_true is not None:
        # The two maps as DISTRIBUTIONS: |error| of every grid cell, in % of
        # max|Δσ_true|, sorted.  fig6 plots the RMSE curve alone, so what these
        # add is the TAIL — the flat-sheet map is not uniformly mediocre, it is
        # wrong where the slab is thick — and that is reported as a P90 below.
        diag["abs_err_flat"] = np.sort(
            np.abs(100.0 * (np.asarray(sigma_map, float) - sigma_true) / peak).ravel()
        )
        diag["abs_err_refined"] = np.sort(
            np.abs(100.0 * (sigma - sigma_true) / peak).ravel()
        )
        if verbose:
            print(
                "      P90 |map error|: flat sheet "
                f"{np.percentile(diag['abs_err_flat'], 90):.1f} % "
                f"-> refined {np.percentile(diag['abs_err_refined'], 90):.1f} %"
                "  of max|Δσ|"
            )
    return sigma, history, dM_corr, diag


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
    c_pre,
    c_post,
    n_max,
    eps=0.02,
    floor_frac=0.1,
    od_alpha=None,
    k_epoch=1.0,
):
    """
    Assumed diagonal OD covariance of ΔCS = CS_post - CS_pre.

    The two epochs keep their OWN uncertainty.  od_sigma is booked on each
    epoch's full coefficients,

        sigma_pre  = od_sigma(c_pre,  eps, floor_frac, alpha=od_alpha)
        sigma_post = od_sigma(c_post, eps, floor_frac, alpha=od_alpha)

    and the difference of two correlated quantities gives

        var(ΔCS_i) = sigma_pre_i² + sigma_post_i² - 2·k·sigma_pre_i·sigma_post_i

    with k the pre/post error correlation, default k = 1: the two OD solutions
    share their error, a common-mode term that cancels in the difference
    exactly as a static background does, leaving var(ΔCS_i) = (sigma_post_i −
    sigma_pre_i)².  k = 0 is the opposite extreme (independent epochs, the two
    solutions share nothing), conservative but dominated by the absolute field
    because od_sigma is relative.

    Because od_sigma is RELATIVE, this σ depends on the static background (the
    rest of Bennu, any field common to both epochs) even though ΔM does not.
    That is the price of not assuming the differenced solution is delivered to
    eps.  Since od_sigma anchors a whole block on its m=0 power rather than on
    each coefficient, what k=1 books here is the difference of the two epochs'
    anchors, and a strong common field pushes those anchors together — so the
    background REDUCES this σ rather than inflating it.  The direction is not
    worth asserting from theory: covariance_report() prints the k sweep and
    _selftest_covariance() measures the background factor outright, while
    checking that ΔM itself stays background-free.

    Different coefficients are independent in this assumed model.  A supplied
    OD covariance can instead go straight into propagate_covariance(), including
    off-diagonal coefficient correlations.

    GLOBAL's od_sigma is called in CH mode (kind="ch", n_max), so the index that
    drives the growth factor is the AZIMUTHAL ORDER m of each coefficient —
    the index that sets how fast a mode decays away from the cylinder — and not
    a spherical-harmonic degree read off SH-style packing.  σ is flat in the
    radial index n within one order.  od_alpha=None selects od_sigma's shaped
    rule (σ climbs GLOBAL.OD_RISE_DEC decades from m=0 to m_max along a curve
    whose slope steepens and is jittered, rather than a straight line in log
    σ), which is what the study runs on; a float means a pure exponential
    exp(od_alpha·m), and od_alpha=0 makes σ flat in m.
    Either way it is distinct from the cylinder's Bessel extension alpha.
    """
    c_pre, c_post = np.asarray(c_pre, float), np.asarray(c_post, float)
    if c_pre.ndim != 1 or c_pre.size == 0 or c_post.shape != c_pre.shape:
        raise ValueError(
            "c_pre and c_post must be nonempty matching coefficient vectors"
        )
    if not (np.isfinite(c_pre).all() and np.isfinite(c_post).all()):
        raise ValueError("coefficient vectors must be finite")
    if not np.isfinite([eps, floor_frac, k_epoch]).all():
        raise ValueError("OD uncertainty settings must be finite")
    if eps < 0 or floor_frac < 0 or not -1 <= k_epoch <= 1:
        raise ValueError("eps and floor_frac must be nonnegative; |k_epoch| <= 1")
    # od_alpha stays None all the way into od_sigma: None is not shorthand for
    # some equivalent rate that could be resolved here, it selects od_sigma's
    # SHAPED rule, whose slope varies across the block.  Collapsing it to one
    # exponential rate first — which an earlier version did, so that the number
    # could be reported like any other — silently swapped the shaped spectrum
    # for a straight line.  A float still means a pure exponential in m.
    if od_alpha is not None:
        if not np.isfinite(od_alpha) or od_alpha < 0:
            raise ValueError("od_alpha must be a nonnegative float, or None")
    kw = dict(floor_frac=floor_frac, alpha=od_alpha, kind="ch", n_max=n_max)
    sig_pre = od_sigma(c_pre, eps, **kw)
    sig_post = od_sigma(c_post, eps, **kw)
    # Written as (post - pre)² + 2(1-k)·pre·post: same value as the plain
    # subtraction, but with no cancellation when k is close to 1.
    var_delta = (sig_post - sig_pre) ** 2 + 2 * (1 - k_epoch) * sig_pre * sig_post
    return dict(
        Sigma_cs=np.diag(var_delta),
        sigma_pre=sig_pre,
        sigma_post=sig_post,
        sigma_delta=np.sqrt(var_delta),
        coeff_rel=eps,
        coeff_floor_frac=floor_frac,
        od_alpha=od_alpha,
        k_epoch=k_epoch,
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
      modal           per-mode diagnostics (k, σ of ΔC_0n, share of Σ_ΔM)
    """
    S_cs, _ = _coefficient_covariance_factor(Sigma_cs, 2 * m_max * n_max)

    f_dM = mass_functional(R_star, alpha, m_max, n_max, zeros_dict)
    var_dM = max(0.0, float(f_dM @ S_cs @ f_dM))

    F_sig = sigma_functional(RHO, PHI, alpha * R_star, m_max, n_max, zeros_dict)
    var_map = np.einsum("gk,kl,gl->g", F_sig, S_cs, F_sig).reshape(RHO.shape)
    sig_map = np.sqrt(np.maximum(var_map, 0.0))

    rho_1d = RHO[:, 0]

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
        print(
            f"    noise model     : od_sigma on each epoch's full CS, "
            f"σ_pre² + σ_post² − 2k·σ_pre·σ_post, k={cov['k_epoch']:.3f}"
        )
        a = cov["od_alpha"]
        print(
            f"                      eps={cov['coeff_rel']:.1%}, "
            f"floor_frac={cov['coeff_floor_frac']:.2f}, "
            f"od_alpha={'shaped' if a is None else format(a, '.2f')}"
        )
        print(
            "    CH order factor : "
            + (
                f"{G_OD_RISE_DEC:g} decade(s) across m, shaped slope"
                if a is None
                else "exp(od_alpha·m)"
            )
            + ", azimuthal order m (not k_mn)"
        )
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
        # How much of √Σ_ΔM is the assumption that the epochs share no error:
        # the same eps, read across the whole range of the epoch correlation k.
        kw = dict(
            eps=scale,
            floor_frac=cov["coeff_floor_frac"],
            od_alpha=cov["od_alpha"],
        )
        f = cov["f_dM"]
        print(f"    per-epoch rule vs k (same eps; the run uses k={cov['k_epoch']:g}):")
        for k in (0.0, 0.9, 0.99, 0.999, 1.0):
            s_e = np.sqrt(
                f
                @ coefficient_difference_covariance(
                    res["c_pre"], res["c_post"], n_max, k_epoch=k, **kw
                )["Sigma_cs"]
                @ f
            )
            print(
                f"      k = {k:5.3f}: √Σ_ΔM = {s_e:.3e} {_U['mass']} "
                f"({100*s_e/abs(dM):9.2f} % of ΔM)"
            )
    sm = cov["sigma_map_1sig"]
    print(
        f"    √Σ_Δσ pointwise : centre {sm[0].mean():.1f}, median "
        f"{np.median(sm):.1f}, max {sm.max():.1f} {_U['sd']}  "
        f"(map peak |Δσ| = {np.abs(res['sigma_map']).max():.0f} {_U['sd']})"
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
        if "bg_epoch" in st:
            print(
                f"      static background (Bennu-mass point {st['bg_depth']:.0f} "
                f"{_U['len']} below the sheet) added to both epochs:\n"
                f"      ΔM changes by {st['bg_dM']:.1e} (rel.), √Σ_ΔM "
                f"×{st['bg_epoch']:.1e} at the run's k={st['bg_k']:.3f}, "
                f"×{st['bg_common']:.1e} at k=1."
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


def relative_error_percent(estimate, truth):
    """Signed percentage error, 100 * (estimate - truth) / truth.

    The sign is retained: positive means an overestimate relative to the
    signed reference value and negative means an underestimate.  This helper
    is used for mass-validation displays so ratios are not mistaken for
    errors.  The reference must be finite and nonzero.
    """
    estimate, truth = np.asarray(estimate, float), np.asarray(truth, float)
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(truth)):
        raise ValueError("estimate and truth must be finite")
    if np.any(truth == 0):
        raise ValueError("relative percentage error is undefined for truth = 0")
    return 100.0 * (estimate - truth) / truth


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
    mass_error = relative_error_percent(res["dM_est"], res["dM_true"])
    for lab, v, u in [
        ("Gravimetric $\\Delta M$", res["dM_est"], "kg"),
        ("Geometric truth $\\Delta M$", res["dM_true"], "kg"),
        ("Recovery ratio", ratio, "--"),
        ("Relative mass error", mass_error, "\\%"),
        ("Equivalent $\\Delta V$", res["dM_est"] / res["density"], "m$^3$"),
        ("Mean $\\Delta h$", res["dh_equiv"], "m"),
        ("Effective $\\Delta\\varrho$", res["delta_rho"], "kg\\,m$^{-3}$"),
        ("Relative signal $\\Delta U/U$", res["sig_ratio"], "--"),
    ]:
        print(rf"  {lab} & ${_tex(v, 3)}$ & {u} \\")

    if cov is None:
        return
    sd, dM = cov["sigma_dM"], res["dM_est"]
    rel = cov.get("coeff_rel", float("nan"))
    rows = [
        ("Assumed per-epoch coefficient precision", 100 * rel, "\\%"),
        ("Coefficient floor fraction", cov.get("coeff_floor_frac", float("nan")), "--"),
        (
            "OD order growth parameter",
            cov.get("od_alpha") if cov.get("od_alpha") is not None else "shaped",
            "--",
        ),
        ("Pre/post coefficient correlation", cov.get("k_epoch", float("nan")), "--"),
    ]
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
    ]:
        cell = rf"\text{{{v}}}" if isinstance(v, str) else _tex(v, 3)
        print(rf"  {lab} & ${cell}$ & {u} \\")

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
    This is equivalent to drawing jointly Gaussian pre/post CS, each with its
    own od_sigma and correlation k, and subtracting them
    (coefficient_difference_covariance).
    No field samples, field noise or coefficient refits are used; scatter of
    the field-point geometry is a separate effect (Section 5b).

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
    actually produces and the relative error between them (one row, one file),
    plus the Delta M histogram.

      ANALYTIC   sqrt(diag(F Sigma_dCS F^T)), the pointwise 1-sigma the
                 covariance predicts, before any sampling.
      NUMERICAL  the spread of `n_map` noise realizations pushed through the
                 same inversion, measured point by point.
      ERROR      100 (numerical - analytic) / analytic, in percent: the same
                 comparison the ratio panel used to make, but read straight off
                 the colour bar as a discrepancy instead of as a number near 1,
                 and on the same signed convention as the other error maps in
                 this file.  Agreement means the two differ only by Monte-Carlo
                 scatter, of size 1/sqrt(2 n_map) at this many draws; the
                 colour range is four times that, so anything structural would
                 be unmistakable.
      MASS       Delta M is a scalar, so one histogram against its predicted
                 Gaussian, in its own file.

    The two sigma maps share one colour scale; the error panel keeps its own.

    Returns (list of figures, the Monte-Carlo dict).
    """
    mc = covariance_mc(res, cov, n_mc=n_mc, n_map=n_map) if mc is None else mc
    RHO, PHI, R = res["RHO"], res["PHI"], res["R_star"]
    Xp, Yp = RHO * np.cos(PHI), RHO * np.sin(PHI)
    Xw, Yw = _polar_closed(Xp), _polar_closed(Yp)
    _wrap = _polar_closed
    tc = np.linspace(0, 2 * np.pi, 200)

    an = mc["an_sigma_map"].reshape(RHO.shape)
    nu = mc["mc_sigma_map"].reshape(RHO.shape)
    rel = 100.0 * np.divide(nu - an, an, out=np.full_like(nu, np.nan), where=an > 0)
    tol = 1.0 / np.sqrt(2.0 * mc["n_map"])

    # The three maps are one row of one file; the mass keeps its own.  Delta
    # sigma is a field and its three maps are read against each other; Delta M
    # is a scalar and needs a histogram, which in a cell of that row would have
    # the wrong shape.
    vmax = max(an.max(), nu.max())

    def _decor(ax):
        """Footprint circle, equal aspect and axis labels, on every map."""
        ax.plot(R * np.cos(tc), R * np.sin(tc), "k--", lw=1.2, alpha=0.65)
        ax.set_aspect("equal")
        ax.set_xlabel(rf"$x-x_0$  [{_UL['len']}]")
        ax.set_ylabel(rf"$y-y_0$  [{_UL['len']}]")

    fig4, axs4 = plt.subplots(
        1, 3, figsize=(3 * FS_MAP[0], FS_MAP[1]), layout="constrained"
    )
    for ax, mp, lab in (
        (
            axs4[0],
            an,
            rf"Analytic $\sigma_{{\Delta\varrho}}$  [{_UL['sd']}]",
        ),
        (
            axs4[1],
            nu,
            rf"Monte-Carlo $\sigma_{{\Delta\varrho}}$  [{_UL['sd']}]",
        ),
    ):
        # one-signed uncertainty map: GLOBAL's viridis_r, no zero contour
        c = _map(ax, Xw, Yw, _wrap(mp), "viridis_r", 0.0, vmax, bold=None)
        fig4.colorbar(c, ax=ax, **CBAR).set_label(lab)
        _decor(ax)

    ve = 400.0 * tol  # +-4 sigma of the Monte-Carlo scatter itself, in percent
    cr = _map(axs4[2], Xw, Yw, _wrap(rel), "RdBu_r", -ve, ve, bold=0.0)
    fig4.colorbar(cr, ax=axs4[2], **CBAR).set_label(
        r"$(\sigma_{\Delta\varrho,\mathrm{MC}}"
        r"-\sigma_{\Delta\varrho,\mathrm{Analytic}})"
        r"/\sigma_{\Delta\varrho,\mathrm{Analytic}}$  " + (r"[\%]" if USE_TEX else "[%]")
    )
    _decor(axs4[2])
    _save(fig4, outdir, "fig4_covariance_map.pdf")
    figs = [fig4]

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
            color=CH_VIOLET,  # GLOBAL case_hist: CH case colour
            alpha=0.75,
            edgecolor="k",
            lw=0.5,
            label="Monte-Carlo Draws",
        )
        xg = np.linspace(e.min(), e.max(), 400)
        ax.plot(
            xg,
            np.exp(-0.5 * (xg / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
            color="k",
            lw=2.2,
            zorder=4,
            label=r"Analytic $N(0,\sigma_{\Delta M}^{2})$",
        )
        for k in (-1, 1):
            ax.axvline(
                k * sd,
                color="0.30",
                ls="--",
                lw=1.5,
                zorder=3,
                label=r"Analytic $\pm\sigma_{\Delta M}$" if k == 1 else None,
            )
    else:
        ax.axvline(0, color="k", label="Zero Assumed Mass Variance")
    ax.set_xlabel(
        rf"$\Delta M_{{\mathrm{{draw}}}}-\Delta M_{{\mathrm{{nominal}}}}$  [{_UL['mass']}]"
    )
    ax.set_ylabel(f"PDF  [1/{_U['mass']}]")
    _grid(ax)
    hist_legend(ax, ncol=3)  # three entries, one row above the axes
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
         ΔM unchanged, because it cancels in ΔCS.  The per-epoch σ_ΔM does not:
         od_sigma is relative, so the background moves it.  The factor is
         reported, not asserted — with the block-anchored rule a common field
         pulls the two epochs' anchors together and the factor comes out below
         one, the opposite of what a per-coefficient relative rule would give.
    """
    dc = res["d_coeffs"]
    m1 = (cov["F_sigma"] @ dc).reshape(res["RHO"].shape)
    # F_sigma is the LINEAR flat-sheet functional, so it must be checked against
    # the plain wahr_invert map, not the terrain-refined one (Section 4a).
    sm_flat = res.get("sigma_map_flat", res["sigma_map"])
    e_map = np.max(np.abs(m1 - sm_flat)) / (np.abs(sm_flat).max() + 1e-30)
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
    kw = dict(
        eps=cov["coeff_rel"],
        floor_frac=cov["coeff_floor_frac"],
        od_alpha=cov["od_alpha"],
    )
    f = cov["f_dM"]

    def sd(cp, cq, k):
        S = coefficient_difference_covariance(
            cp, cq, res["n_max"], k_epoch=k, **kw
        )["Sigma_cs"]
        return np.sqrt(f @ S @ f)

    cp, cq = res["c_pre"], res["c_post"]
    bg_dM = abs(f @ ((cq + c_bg) - (cp + c_bg)) - f @ (cq - cp)) / abs(f @ (cq - cp))
    # round-off only: the background coefficients are ~1e6× the TAG change
    assert bg_dM < 1e-6, f"background leaked into ΔM ({bg_dM:.1e})"
    k_run = cov["k_epoch"]
    bg_epoch = sd(cp + c_bg, cq + c_bg, k_run) / sd(cp, cq, k_run)
    bg_common = sd(cp + c_bg, cq + c_bg, 1.0) / sd(cp, cq, 1.0)
    out.update(
        bg_dM=bg_dM,
        bg_epoch=bg_epoch,
        bg_common=bg_common,
        bg_k=k_run,
        bg_depth=depth * TO_SI["length"],
    )
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
    N_field: int = 10000,  # one draw, no ensemble — see Section 5b
    seed: int = 1,
    cond: float = CH_COND,  # truncated-SVD cutoff of the unweighted LS
    k_target_R: float = KR_TARGET_DEFAULT,  # bandlimit target for n_max="auto"
    # covariance analysis (Section 4b)
    do_covariance: bool = True,
    coeff_rel: float = 0.004,  # od_sigma eps on each epoch's CS
    coeff_floor_frac: float = 0.1,  # floor relative to the CS RMS it is applied to
    od_alpha: float | None = None,  # None: od_sigma's shaped rule; not Bessel alpha
    k_epoch: float = 1.0,  # pre/post coefficient error correlation
    refine_map: bool = True,  # terrain refinement of Δσ (Section 4a)
    verbose: bool = True,
):
    """
    Full pre/post TAG pipeline in the unit system selected by `MODE`.  Returns
    a dict of all intermediate and final results (see bottom of function).

    Fit PRE/POST coefficients once, on the same N_field points for both
    epochs; field-sampling scatter is reduced by raising N_field, not by
    averaging repeated fits (Section 5b measures both).
    Separately, assign OD uncertainty with GLOBAL's od_sigma on EACH epoch's
    coefficients and combine them as σ_pre² + σ_post² − 2k·σ_pre·σ_post (see
    coefficient_difference_covariance), then propagate Σ_ΔCS to mass and
    density.  Call covariance_mc(res, res['cov']) to draw coefficient
    realizations; field points and gravity are not redrawn in that Monte Carlo.
    Neither number includes terrain or thin-sheet model bias.

    The sheet plane is the signed vertical centroid of the pre/post
    differential slab.  This removes the systematic upward offset that occurs
    when the mean pre-TAG height is used for a predominantly excavated source.

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
    z_sheet_pre_mean = float(h_pre[foot].mean())
    z_sheet = differential_sheet_plane(
        h_pre, h_post, foot, fallback=z_sheet_pre_mean
    )  # expansion plane [m]
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
        print(
            f"    sheet plane z0 = {z_sheet:.2f} m "
            f"(signed Δ-slab centroid; pre-TAG mean = {z_sheet_pre_mean:.2f} m)"
        )
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

    # ── 5b. ONE FIT, NOT AN ENSEMBLE OF SMALLER ONES ───────────────────
    # An earlier version refit on independent point draws and averaged the
    # coefficients.  Dropped: an average of truncated-SVD solutions is not the
    # LS solution of the pooled points — each draw keeps its own retained
    # subspace — and it buys nothing measurable.  At a FIXED basis
    # (α=6, m_max=10, n_max=19, cond=1.47e-2, what the calibration picked back
    # when it was run at N_field = 2000), grid_res = 0.30 m, 12 seeds/N:
    #     N_field     250    500   1000   2000   4000   8000  16000
    #     mean error +5.28  +5.54  +5.33  +5.51  +5.79  +5.60  +5.39   [%]
    #     1σ scatter  1.21   1.38   0.76   1.04   0.67   0.30   0.35   [%]
    # The mean error is FLAT in N_field — it is the truncation + thin-sheet
    # bias, which no amount of sampling or averaging removes.  Only the
    # scatter falls (≈ N^-0.3), so K·N pooled points and K averaged fits of N
    # points buy the same reduction; head to head at equal cost (4 replicates)
    # the mean of five N=2000 fits gives +5.42 ± 0.18 % and one N=10000 fit
    # +5.48 ± 0.21 % — indistinguishable, and ≲ 1/25 of the bias either way.
    # Pooling additionally conditions the fit better (retained rank 46..52 at
    # N=250 → 56 at N=16000; the share of the ΔM functional the cutoff throws
    # away settles from 0.162 ± 0.011 to 0.137 ± 0.001), so the single fit is
    # both simpler and the better-defined estimator.  N_field = 10000 costs
    # exactly what the 5 × 2000 ensemble used to in gravity evaluations — but
    # note that everything which REFITS on those points (the calibration, the
    # basis ablation) does pay the 5×, and that the calibration, being run on
    # better-conditioned draws, no longer picks the same basis.
    field_draws = [(rp, pp, zp, db)]  # basis_sweep reuses this draw; memory only
    d_coeffs = c_post - c_pre

    # ── 6. WAHR INVERSION ──────────────────────────────────────────────
    dM_est, sigma_map, RHO, PHI = wahr_invert(
        d_coeffs, R_star, alpha, m_max, n_max, zeros_dict
    )
    sigma_map_flat = sigma_map.copy()  # plain thin-sheet-on-z_sheet inverse

    # True surface-density change on the same polar grid.  Computed here, ahead
    # of the refinement, only so the refinement can RECORD its error pass by
    # pass (fig6); the iteration itself never sees it.
    dh_itp = RegularGridInterpolator((gx, gy), dh, bounds_error=False, fill_value=0.0)
    sigma_true = density * dh_itp(
        np.column_stack(
            [(cx + RHO * np.cos(PHI)).ravel(), (cy + RHO * np.sin(PHI)).ravel()]
        )
    ).reshape(RHO.shape)

    # ── TERRAIN REFINEMENT OF THE MAP (Section 4a) ─────────────────────
    # Same inverse, better forward: the moved mass is a sheet on the terrain
    # graph, not on the plane.  ΔM and the Section 4b covariance stay on the
    # plain wahr path — only the MAP is refined.
    refine_history, dM_refine_corr, refine_diag = [], 0.0, {}
    if refine_map and MAP_REFINE_ITERS > 0:
        if verbose:
            print(
                f"    terrain refinement of Δσ: {MAP_REFINE_ITERS} damped passes "
                f"(w = {MAP_REFINE_DAMPING}) using the pre-TAG DTM"
            )
        h_pre_itp = RegularGridInterpolator(
            (gx, gy), h_pre, bounds_error=False, fill_value=float(h_pre[foot].mean())
        )
        sigma_map, refine_history, dM_refine_corr, refine_diag = terrain_refine_map(
            sigma_map,
            RHO,
            PHI,
            field_draws,
            (cx, cy),
            z_sheet,
            h_pre_itp,
            density,
            R_star,
            alpha,
            m_max,
            n_max,
            zeros_dict,
            cond=cond,
            sigma_true=sigma_true,
            verbose=verbose,
        )

    # ── 6b. COVARIANCE PROPAGATION (Section 4b) ────────────────────────
    # Assume the fitted CS and their OD uncertainties are the available data.
    # Form Σ_ΔCS (see coefficient_difference_covariance), then propagate.
    cov = None
    if do_covariance:
        od = coefficient_difference_covariance(
            c_pre,
            c_post,
            n_max,
            eps=coeff_rel,
            floor_frac=coeff_floor_frac,
            od_alpha=od_alpha,
            k_epoch=k_epoch,
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

    # Map error, normalised by the ONE scale the field has.  A POINTWISE
    # relative error 100(Δσ̂−Δσ)/Δσ is not usable here: Δσ_true is signed and
    # crosses zero over most of the disc, so the quotient blows up wherever
    # the terrain barely moved, and every summary built on it then depends on
    # a threshold chosen to hide that blow-up — the threshold, not the
    # inversion, sets the number.  Normalising by max|Δσ_true| instead is
    # defined at every point, needs no mask and no tuned constant, and keeps
    # one fixed denominator so cells, bases and refinement passes stay
    # comparable.  RMSE/peak (already the honest summary) is the same
    # quantity's quadratic mean.
    map_truth_peak = float(np.max(np.abs(sigma_true)))
    map_error_percent = (
        100.0 * (sigma_map - sigma_true) / map_truth_peak
        if map_truth_peak > 0
        else np.full_like(sigma_true, np.nan, dtype=float)
    )
    map_abs_error = np.abs(map_error_percent)
    map_error_bias = float(np.mean(map_error_percent))
    map_error_median = float(np.median(map_abs_error))
    map_error_p90 = float(np.percentile(map_abs_error, 90))
    map_error_rmse_peak = float(np.sqrt(np.mean(map_error_percent**2)))

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
        print(
            f"    map error / peak |Δσ| (whole disc, no threshold): "
            f"median |e| {map_error_median:.1f}%, p90 {map_error_p90:.1f}%, "
            f"RMSE {map_error_rmse_peak:.1f}%, mean (bias) {map_error_bias:+.1f}%"
        )
        if refine_history:
            # same statistic for both maps: median/p90/p98 of |error| / peak
            def _abs_stats(sm):
                e = np.abs(100.0 * (sm - sigma_true) / map_truth_peak)
                return np.median(e), np.percentile(e, 90), np.percentile(e, 98)

            f_med, f_p90, f_p98 = _abs_stats(sigma_map_flat)
            r_med, r_p90, r_p98 = _abs_stats(sigma_map)
            print(
                f"    |error|/peak  flat sheet : median {f_med:5.1f}%  "
                f"p90 {f_p90:5.1f}%  p98 {f_p98:5.1f}%"
            )
            print(
                f"                  terrain-ref: median {r_med:5.1f}%  "
                f"p90 {r_p90:5.1f}%  p98 {r_p98:5.1f}%   "
                f"(‖d−TΔσ‖/‖d‖ {refine_history[0]:.3f} → {refine_history[-1]:.3f})"
            )
            print(
                f"    ΔM diagnostic: terrain-refined ΔM = {dM_est + dM_refine_corr:.4e} "
                f"{_U['mass']} ({100*((dM_est+dM_refine_corr)/dM_true - 1):+.2f} % vs "
                f"{100*(dM_est/dM_true - 1):+.2f} % flat sheet); the reported ΔM and "
                f"its Section 4b covariance stay on the LINEAR flat-sheet functional"
            )

    V_cyl = np.pi * R_star**2 * H
    delta_rho = dM_est / V_cyl  # effective density change in cylinder [kg/m³]
    dh_equiv = dM_est / (density * np.pi * R_star**2)  # mean elevation change [m]

    dU = U_post - U_pre
    dgz = gz_post - gz_pre
    sig_ratio = np.std(dU) / (np.sqrt(np.mean(U_pre**2)) + 1e-30)
    coeff_ratio = np.linalg.norm(d_coeffs) / (np.linalg.norm(c_pre) + 1e-30)

    if verbose:
        mass_error_percent = relative_error_percent(dM_est, dM_true)
        print(f"\n{DASH}\n  RESULTS (SI)\n{DASH}")
        print(f"  ΔM  gravimetric       = {dM_est:+.4e} {_U['mass']}")
        print(
            f"  ΔM  geometric truth   = {dM_true:+.4e} {_U['mass']}   (ρ·∫Δh dA, ρ<R*)"
        )
        print(f"  recovery ratio        = {dM_est / dM_true:8.3f}")
        print(f"  relative mass error   = {mass_error_percent:+8.2f} %")
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
        z_sheet_pre_mean=z_sheet_pre_mean,
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
        field_draws=field_draws,  # the single draw's coordinates and Δ-field
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
        sigma_map_flat=sigma_map_flat,
        refine_history=refine_history,
        refine_diag=refine_diag,
        dM_refined=dM_est + dM_refine_corr,
        sigma_true=sigma_true,
        # covariance analysis (None if do_covariance=False)
        cov=cov,
        sigma_dM=None if cov is None else cov["sigma_dM"],
        sigma_map_1sig=None if cov is None else cov["sigma_map_1sig"],
        sigma_peak_rec=sigma_peak_rec,
        sigma_peak_true=sigma_peak_true,
        sigma_peak_true_pix=sigma_peak_true_pix,
        map_error_percent=map_error_percent,  # 100 (est - true) / max|true|
        map_truth_peak=map_truth_peak,
        map_error_bias=map_error_bias,
        map_error_median=map_error_median,
        map_error_p90=map_error_p90,
        map_error_rmse_peak=map_error_rmse_peak,
        z_resolution=z_resolution,
        RHO=RHO,
        PHI=PHI,
        # derived
        V_cyl=V_cyl,
        delta_rho=delta_rho,
        dh_equiv=dh_equiv,
        sig_ratio=sig_ratio,
        coeff_ratio=coeff_ratio,
        mass_error_percent=relative_error_percent(dM_est, dM_true),
    )

    if cov is not None:
        cov["selftest"] = _selftest_covariance(res, cov)
        covariance_report(cov, res, verbose=verbose)  # reads cov["selftest"]

    return res


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5b — SVD-CUTOFF CALIBRATION  (truth-free choice of cond)
# ═══════════════════════════════════════════════════════════════════════════


def calibration_draws(res, seeds=range(1, 9), clearances=(0.25, 0.5), verbose=True):
    """
    Difference-field draws for the calibrations: one list per clearance [m]
    of (rp, pp, zp, Δb) per seed, at res's site, footprint, H and N_field.
    They do not depend on the basis or the cutoff, so one set serves every
    candidate basis and cutoff (the polyhedral evaluations dominate the cost).
    """
    ev_pre = make_evaluable(res["mesh_pre"], res["density"])
    ev_post = make_evaluable(res["mesh_post"], res["density"])
    h_itp = RegularGridInterpolator(
        (res["gx"], res["gy"]), np.maximum(res["h_pre"], res["h_post"])
    )
    draws, t0 = [], time.time()
    for cl in clearances:
        row = []
        for sd in seeds:
            rp, pp, zp, pts = make_cylinder_field_points(
                (res["cx"], res["cy"]),
                res["z_sheet"],
                res["R_star"],
                res["H"],
                h_itp,
                clearance=cl / L_REF,
                N=res["N_field"],
                seed=int(sd),
            )
            assert (
                pts[:, 2] > h_itp(pts[:, :2])
            ).all(), "field points intersect terrain"
            U0, gx0, gy0, gz0 = eval_gravity(ev_pre, pts)
            U1, gx1, gy1, gz1 = eval_gravity(ev_post, pts)
            gr0, gp0 = cart_to_cyl_g(gx0, gy0, pp)
            gr1, gp1 = cart_to_cyl_g(gx1, gy1, pp)
            db = assemble_obs_vector(U1 - U0, gr1 - gr0, gp1 - gp0, gz1 - gz0)
            row.append((rp, pp, zp, db))
        draws.append(row)
        if verbose:
            print(
                f"    clearance {cl} {_U['len']}: {len(row)} draws "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    return draws


def calibration_report(sw):
    """Print the sweep table, the truth-free choice, and its validation."""
    c, i_s, band = sw["conds"], sw["i_star"], sw["band"]
    dM, ratio, error, rank = sw["dM"], sw["ratio"], sw["error_percent"], sw["rank"]
    print(
        f"\n{DASH}\n  SVD-CUTOFF CALIBRATION  (truth-free choice; truth only validates)\n{DASH}"
    )
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
        f"{'ΔM median':>11} {'σ_OD %':>7} | {'error %':>8}  (validation)"
    )
    for i in range(c.size):
        tag = (
            "c*"
            if i == i_s
            else ("band" if band[i] else ("" if sw["admissible"][i] else "rejected"))
        )
        print(
            f"    {c[i]:9.2e} {int(rank[..., i].min()):3d}-{int(rank[..., i].max()):<3d} "
            f"{sw['cv_norm'][:, i].max():7.2f} {100*sw['instab'][i]:7.2f} "
            f"{np.median(dM[..., i]):+11.4e} {100*np.median(sw['sig_delta'][..., i]):7.2f} "
            f"| {np.median(error[..., i]):+8.1f}  {tag}"
        )
    lo, hi = sw["band_range"]
    b_dM = dM[..., band].ravel()
    print(
        f"\n    c*   = {sw['cond_star']:.3e}  (rank {int(rank[..., i_s].min())}–"
        f"{int(rank[..., i_s].max())}; nearest singular value "
        f"{100*sw['gap_star']:.1f} % away, one rank step moves ΔM ≤ "
        f"{100*sw['step_star']:.1f} %, worst draw)"
    )
    print(
        f"    band = [{lo:.3e}, {hi:.3e}]  (admissible, I ≤ {sw['band_tol']:g}·I(c*))"
    )
    print(
        f"    ΔM over the band (truth-free): median {np.median(b_dM):+.4e} "
        f"{_U['mass']}, spread {100*b_dM.std(ddof=1)/abs(b_dM.mean()):.2f} % "
        f"(all draws, clearances, cutoffs)"
    )
    print(
        f"    OD σ_ΔM at c* (per-epoch rule): median "
        f"{100*np.median(sw['sig_delta'][..., i_s]):.2f} % of ΔM"
    )

    def val(e):
        e = np.ravel(e)
        return (
            f"median {np.median(e):+.1f} %  [{e.min():+.1f}, {e.max():+.1f}]  "
            f"RMSE {np.sqrt(np.mean(e ** 2)):.1f} %"
        )

    print("    validation against the geometric truth (NOT used above):")
    print(f"      at c*               : {val(error[..., i_s])}")
    print(f"      over the band       : {val(error[..., band])}")
    print(
        f"      at cond_ref {sw['cond_ref']:.2e}: {val(sw['error_ref_percent'])}  (rank "
        f"{int(sw['rank_ref'].min())}–{int(sw['rank_ref'].max())}, nearest singular "
        f"value {100*sw['gap_ref']:.1f} % away, one rank step moves ΔM ≤ "
        f"{100*sw['step_ref']:.1f} %)"
    )


def basis_calibration(
    res,
    alphas=CAL_ALPHAS,
    m_max_values=CAL_M_MAX,
    n_max_values=CAL_N_MAX,
    seeds=range(1, 9),
    clearances=(0.25, 0.5),  # [m]
    conds=np.logspace(-6, -1, 61),
    kappa=2.0,
    half_window=0.15,
    band_tol=1.25,
    k_target_R=KR_TARGET_DEFAULT,  # bandlimit run_bennu_tag enforces
    draws=None,  # calibration_draws(res, seeds, clearances); drawn if None
    verbose=True,
):
    """
    Choose the basis (alpha, m_max, n_max) AND the cutoff WITHOUT the true ΔM.

    `calibration_sweep`'s rule with the basis as extra axes, on one shared set
    of field draws:
      1. CV(b, c): held-out relative misfit of basis b at cutoff c (fit one
         draw, predict the other draws at the same clearance; median over the
         fitted draw).  Admissible: CV ≤ kappa·min CV at every clearance, the
         minimum taken over ALL bases and cutoffs — a basis that cannot
         represent the Δ-field is rejected like a cutoff that cuts signal.
      2. I(b, c): relative std of ΔM pooled over draws, clearances and cutoffs
         within ±half_window decades, as in calibration_sweep.
      3. Candidates: admissible (b, c) with I ≤ band_tol·min I.  Take the basis
         with the fewest coefficients among them (ties: lower I), then its
         admissible cutoff of least I.  Minimizing I alone over ~10⁵ noisy
         (b, c) pairs rewards chance; within the tolerance, the smaller basis
         has fewer modes the sampling must pin down.
    Only bases meeting run_bennu_tag's bandlimit (k_max·R* ≥ k_target_R, a
    resolution requirement for the Δσ map) can be chosen; coarser ones still
    count toward nothing but the tables.  The true ΔM only validates the
    choice (printed, never used).  The choice is one of several equally good
    candidates: the report gives their ΔM spread, the selection uncertainty.

    Speed as in basis_sweep: per alpha and draw, A = Q R for the largest basis,
    and any sub-basis S is fitted as R[:, S] against Qᵀ Δb (same singular values
    and solutions).  The held-out misfit needs no full-size product either:
    ||A_o[:, S] x − Δb_o||² = ||R_o[:, S] x − Q_oᵀ Δb_o||² + ||Δb_o||² − ||Q_oᵀ Δb_o||².
    """
    conds = np.sort(np.asarray(conds, float))
    seeds, clearances = list(seeds), list(clearances)
    alphas = np.unique(np.asarray(alphas, float))
    ms = np.unique(np.asarray(m_max_values, int))
    ns = np.unique(np.asarray(n_max_values, int))
    if alphas.min() < 1 or ms.min() < 1 or ns.min() < 1:
        raise ValueError("alphas must be ≥ 1 and mode counts ≥ 1")
    # Grid guard.  The rule compares every pair with the BEST pair on the grid,
    # so an alpha the grid cannot resolve (no basis reaches the bandlimit) is a
    # silent truncation of the search.  With CAL_N_MAX ≤ 11 this dropped
    # alpha ≥ 5 and picked (3,7,9) c=5.6e-6: +27 % ΔM error.
    blind = [
        a
        for a in alphas
        if jn_zeros(int(ms.max()) - 1, int(ns.max()))[-1] / a < k_target_R
    ]
    if blind:
        need = [
            next(
                n
                for n in range(1, 200)
                if jn_zeros(int(ms.max()) - 1, n)[-1] / a >= k_target_R
            )
            for a in blind
        ]
        raise ValueError(
            f"calibration grid too small: alpha={', '.join(f'{a:g}' for a in blind)} has no "
            f"basis with k_max·R* ≥ {k_target_R:g} at m_max ≤ {ms.max()}, n_max ≤ {ns.max()} "
            f"(needs n_max ≥ {', '.join(map(str, need))}).  Extend CAL_N_MAX or drop those alphas."
        )
    if draws is None:
        draws = calibration_draws(res, seeds, clearances, verbose=verbose)
    elif len(draws) != len(clearances) or any(len(d) != len(seeds) for d in draws):
        raise ValueError("draws must hold len(seeds) draws per clearance")
    R = res["R_star"]
    max_m, max_n = int(ms[-1]), int(ns[-1])
    packed = np.arange(2 * max_m * max_n)
    active = (packed >= 2 * max_n) | (packed % 2 == 0)  # m=0 sine columns are 0
    order = (packed // (2 * max_n))[active]
    radial = ((packed // 2) % max_n + 1)[active]
    shape = (len(alphas), len(ms), len(ns), len(clearances))
    dM = np.empty(shape + (len(seeds), conds.size))
    rank = np.empty(dM.shape, int)
    cv = np.empty(shape + (conds.size,))
    t0 = time.time()
    if verbose:
        print(
            f"\n{DASH}\n  BASIS + CUTOFF CALIBRATION  (truth-free; truth only validates)\n{DASH}\n"
            f"    {alphas.size}×{ms.size}×{ns.size} bases × {conds.size} cutoffs × "
            f"{len(seeds)} draws × {len(clearances)} clearances",
            flush=True,
        )

    def solve_alpha(item):
        ia, alpha = item
        zeros = {m: jn_zeros(m, max_n) for m in range(max_m)}
        fac = []  # per clearance, per draw: (R, Qᵀ Δb, ||Δb||² − ||Qᵀ Δb||², ||Δb||)
        for row in draws:
            fr = []
            for rp, pp, zp, db in row:
                A, _ = build_design_matrix(rp, pp, zp, alpha * R, max_m, max_n)
                Q, R_des = qr(A[:, active], mode="economic")
                qb = Q.T @ db
                fr.append((R_des, qb, max(db @ db - qb @ qb, 0.0), np.linalg.norm(db)))
            fac.append(fr)
        for im, mm in enumerate(ms):
            for jn, nn in enumerate(ns):
                cols = np.flatnonzero((order < mm) & (radial <= nn))
                f = mass_functional(R, alpha, mm, nn, zeros)
                local = np.arange(f.size)
                f = f[(local >= 2 * nn) | (local % 2 == 0)]
                for ic, fr in enumerate(fac):
                    X = []
                    for j, (R_des, qb, _, _) in enumerate(fr):
                        U, s, Vt = np.linalg.svd(R_des[:, cols], full_matrices=False)
                        keep = s[None, :] > conds[:, None] * s[0]
                        beta = np.divide(U.T @ qb, s, out=np.zeros_like(s), where=s > 0)
                        X.append(Vt.T @ (keep * beta).T)  # (n_coeff, n_cutoffs)
                        dM[ia, im, jn, ic, j] = f @ X[-1]
                        rank[ia, im, jn, ic, j] = keep.sum(axis=1)
                    miss = np.zeros((len(fr), conds.size))
                    for o, (R_o, qb_o, out_o, nb_o) in enumerate(fr):
                        R_oS = R_o[:, cols]
                        for j in range(len(fr)):
                            if j != o:
                                r = R_oS @ X[j] - qb_o[:, None]
                                miss[j] += np.sqrt(np.sum(r**2, axis=0) + out_o) / nb_o
                    cv[ia, im, jn, ic] = np.median(miss / (len(fr) - 1), axis=0)
        return alpha

    workers = max(1, min(int(BASIS_SWEEP_WORKERS), alphas.size))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for alpha in pool.map(solve_alpha, enumerate(alphas)):
            if verbose:
                print(f"    alpha={alpha:g}: done ({time.time()-t0:.0f}s)", flush=True)

    cal = dict(
        alphas=alphas,
        m_max_values=ms,
        n_max_values=ns,
        conds=conds,
        seeds=seeds,
        clearances=clearances,
        dM=dM,
        rank=rank,
        cv=cv,
        dM_true=res["dM_true"],
    )
    select_basis(
        cal,
        kappa=kappa,
        half_window=half_window,
        band_tol=band_tol,
        k_target_R=k_target_R,
    )
    if verbose:
        basis_calibration_report(cal)
    return cal


def select_basis(
    cal, kappa=2.0, half_window=0.15, band_tol=1.25, k_target_R=KR_TARGET_DEFAULT
):
    """Apply basis_calibration's truth-free rule to its arrays (in place)."""
    conds, dM, cv = cal["conds"], cal["dM"], cal["cv"]
    k_max_R = np.array(
        [
            [
                [jn_zeros(m - 1, n)[-1] / a for n in cal["n_max_values"]]
                for m in cal["m_max_values"]
            ]
            for a in cal["alphas"]
        ]
    )  # R* cancels: k_max·R* = j_max/alpha
    cv_norm = cv / cv.min(axis=(0, 1, 2, 4), keepdims=True)  # per clearance
    admissible = cv_norm.max(axis=3) <= kappa  # (alpha, m, n, cond)
    admissible &= (k_max_R >= k_target_R)[..., None]
    if not admissible.any():
        raise RuntimeError(
            "no basis/cutoff passes the CV admissibility test; raise kappa"
        )
    lc = np.log10(conds)
    instab = np.empty(admissible.shape)
    for i in range(conds.size):
        pool = dM[..., np.abs(lc - lc[i]) <= half_window + 1e-9]
        pool = pool.reshape(pool.shape[:3] + (-1,))
        instab[..., i] = pool.std(axis=-1, ddof=1) / np.abs(pool.mean(axis=-1))
    instab[~np.isfinite(instab)] = np.inf  # ΔM pooled to zero: not a candidate
    I_adm = np.where(admissible, instab, np.inf)
    ok = I_adm <= band_tol * I_adm.min()
    a, m, n = np.meshgrid(
        cal["alphas"], cal["m_max_values"], cal["n_max_values"], indexing="ij"
    )
    n_coeff = n * (2 * m - 1)  # m=0 has cosines only
    cand = ok.any(axis=-1)
    best_I = I_adm.min(axis=-1)
    flat = np.flatnonzero(cand)
    pick = flat[np.lexsort((best_I.ravel()[flat], n_coeff.ravel()[flat]))[0]]
    ia, im, jn = np.unravel_index(pick, cand.shape)
    i_star = int(np.argmin(I_adm[ia, im, jn]))
    cal.update(
        kappa=kappa,
        half_window=half_window,
        band_tol=band_tol,
        k_target_R=k_target_R,
        k_max_R=k_max_R,
        cv_norm=cv_norm,
        admissible=admissible,
        instab=instab,
        candidates=cand,
        n_coeff=n_coeff,
        index=(int(ia), int(im), int(jn), i_star),
        alpha=float(cal["alphas"][ia]),
        m_max=int(cal["m_max_values"][im]),
        n_max=int(cal["n_max_values"][jn]),
        cond=float(conds[i_star]),
    )
    return cal


def basis_calibration_report(cal):
    """Print the candidate bases, the truth-free choice, and its validation."""
    ia, im, jn, i_s = cal["index"]
    I, adm, cand = cal["instab"], cal["admissible"], cal["candidates"]
    err = relative_error_percent(cal["dM"], cal["dM_true"])
    I_min = np.where(adm, I, np.inf).min()
    print(
        f"    rule: CV ≤ {cal['kappa']:g}·min CV (all bases) at every clearance; "
        f"k_max·R* ≥ {cal['k_target_R']:g}; candidates I ≤ {cal['band_tol']:g}·min I = "
        f"{100*cal['band_tol']*I_min:.2f} %; fewest coefficients wins"
    )
    print(
        f"    {int(adm.any(axis=-1).sum())} of {adm[..., 0].size} bases have an admissible "
        f"cutoff; {int(cand.sum())} are candidates:"
    )
    print(
        f"    {'alpha':>5} {'m':>3} {'n':>3} {'coeffs':>6} {'c*':>9} {'rank':>7} "
        f"{'I %':>6} | {'error %':>8}  (validation)"
    )
    rows = np.argwhere(cand)
    rows = rows[
        np.lexsort(
            (
                np.where(adm, I, np.inf)[tuple(rows.T)].min(axis=-1),
                cal["n_coeff"][tuple(rows.T)],
            )
        )
    ]
    for a, m, n in rows[:25]:
        i = int(np.argmin(np.where(adm[a, m, n], I[a, m, n], np.inf)))
        tag = "  <- chosen" if (a, m, n) == (ia, im, jn) else ""
        rk = cal["rank"][a, m, n, ..., i]
        print(
            f"    {cal['alphas'][a]:5g} {cal['m_max_values'][m]:3d} {cal['n_max_values'][n]:3d} "
            f"{cal['n_coeff'][a, m, n]:6d} {cal['conds'][i]:9.2e} {rk.min():3d}-{rk.max():<3d} "
            f"{100*I[a, m, n, i]:6.2f} | {np.median(err[a, m, n, ..., i]):+8.1f}{tag}"
        )
    if len(rows) > 25:
        print(f"    ... {len(rows) - 25} more")
    # the candidates are equally good by the rule: their ΔM spread is the
    # model-selection uncertainty the single choice hides
    dM_c = np.array(
        [
            np.median(
                cal["dM"][
                    a, m, n, ..., np.argmin(np.where(adm[a, m, n], I[a, m, n], np.inf))
                ]
            )
            for a, m, n in rows
        ]
    )
    r = dM_c / np.median(dM_c)
    print(
        f"    ΔM over the {len(rows)} candidates (truth-free, each at its own c*): median "
        f"{np.median(dM_c):+.4e} {_U['mass']}, range {100*(r.min()-1):+.1f}.."
        f"{100*(r.max()-1):+.1f} % of it, std {100*dM_c.std(ddof=1)/abs(dM_c.mean()):.1f} %"
    )
    e = err[ia, im, jn, ..., i_s].ravel()
    print(
        f"\n    chosen: alpha={cal['alpha']:g}, m_max={cal['m_max']}, n_max={cal['n_max']}, "
        f"cond={cal['cond']:.3e}  (I = {100*I[ia, im, jn, i_s]:.2f} %)"
    )
    print(
        f"    validation (NOT used): median {np.median(e):+.1f} %  "
        f"[{e.min():+.1f}, {e.max():+.1f}]  RMSE {np.sqrt(np.mean(e**2)):.1f} %"
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5c — BASIS SWEEP  (alpha, m_max, n_max; everything else fixed)
# ═══════════════════════════════════════════════════════════════════════════


def basis_sweep(
    res,
    alphas=BASIS_SWEEP_ALPHAS,
    m_max_values=BASIS_SWEEP_M_MAX,
    n_max_values=BASIS_SWEEP_N_MAX,
    n_draws=BASIS_SWEEP_N_DRAWS,
    verbose=True,
):
    """Compare bases at the main run's FIXED cutoff, geometry and sampling.

    Array order is (alpha, m_max, n_max[, field draw]).  Each cell reports
    mean(Delta M over draws) / Delta M_true, equivalent to inverting the mean
    coefficients as run_bennu_tag does.  `error_percent` converts that ratio
    to the signed percentage error 100*(estimate - true)/true.  The std across
    draws is sampling scatter, not OD uncertainty or a standard error of the
    mean.  `map_rmse_percent` scores the same cells on the Delta-sigma MAP --
    RMS(sigma_est - sigma_true) as a percentage of peak |sigma_true|, on the
    main run's own (RHO, PHI) grid -- so one sweep feeds both fig7 panels.

    Only the basis changes.  Reuse exactly the same points and difference
    fields for all cells; no per-cell cutoff selection or truth-based tuning.
    Coarse bases are included deliberately, even below KR_TARGET_DEFAULT,
    so their failures remain visible.  Included wavelengths precede the SVD
    cutoff and are not a claim about resolved spatial scales.  alpha=1 is
    allowed here to test a Dirichlet boundary at the footprint edge.

    For speed, factor the largest design matrix per alpha/draw as Q @ R.
    Fitting a subset of R to Q.T @ db has the same singular values and LS
    solution as fitting those columns of the full matrix.  This orthogonal
    reduction applies no row weights or column scaling.  Identically zero
    m=0 sine columns are omitted; their coefficients and mass weights are zero.
    """
    alphas = np.asarray(alphas, dtype=float)
    if (
        alphas.ndim != 1
        or not alphas.size
        or not np.isfinite(alphas).all()
        or np.any(alphas < 1)
    ):
        raise ValueError("alphas must be a nonempty vector of finite values >= 1")
    alphas = np.unique(alphas)

    def counts(values, name):
        values = np.asarray(values, dtype=float)
        if (
            values.ndim != 1
            or not values.size
            or not np.isfinite(values).all()
            or np.any(values < 1)
            or np.any(values != np.floor(values))
        ):
            raise ValueError(f"{name} must contain positive integer mode counts")
        return np.unique(values.astype(int))

    ms, ns = counts(m_max_values, "m_max_values"), counts(n_max_values, "n_max_values")
    if not isinstance(n_draws, (int, np.integer)) or n_draws < 1:
        raise ValueError("n_draws must be a positive integer")
    cond = res["cond"]
    if cond is None or not np.isfinite(cond) or not 0 < cond < 1:
        raise ValueError("res['cond'] must be a fixed relative cutoff between 0 and 1")
    truth = float(res["dM_true"])
    if not np.isfinite(truth) or truth == 0:
        raise ValueError("a finite nonzero dM_true is needed for mass ratios")
    seeds = np.arange(res["seed"], res["seed"] + int(n_draws))
    R = res["R_star"]
    t0 = time.time()
    if verbose:
        print(f"\n{DASH}\n  BASIS SWEEP: alpha, m_max, n_max\n{DASH}", flush=True)
        print(
            f"    {len(alphas)*len(ms)*len(ns)} bases × {len(seeds)} shared field draws; "
            f"fixed cond={cond:.8g}, N_field={res['N_field']}, "
            f"clearance={res['clearance']:g} {_U['len']}",
            flush=True,
        )

    # run_bennu_tag keeps its single draw; any further draws the sweep wants
    # are regenerated here at the run's own sampling settings.
    draws = [
        (
            res["rp"],
            res["pp"],
            res["zp"],
            assemble_obs_vector(
                res["U_post"] - res["U_pre"],
                res["gr_post"] - res["gr_pre"],
                res["gphi_post"] - res["gphi_pre"],
                res["gz_post"] - res["gz_pre"],
            ),
        )
    ]
    cached_draws = res.get("field_draws", ())
    if len(cached_draws) >= len(seeds):
        draws = cached_draws[: len(seeds)]
    elif len(seeds) > 1:
        ev_pre = make_evaluable(res["mesh_pre"], res["density"])
        ev_post = make_evaluable(res["mesh_post"], res["density"])
        h_itp = RegularGridInterpolator(
            (res["gx"], res["gy"]), np.maximum(res["h_pre"], res["h_post"])
        )
        for seed in seeds[1:]:
            rp, pp, zp, pts = make_cylinder_field_points(
                (res["cx"], res["cy"]),
                res["z_sheet"],
                R,
                res["H"],
                h_itp,
                clearance=res["clearance"],
                N=res["N_field"],
                seed=int(seed),
            )
            U0, gx0, gy0, gz0 = eval_gravity(ev_pre, pts)
            U1, gx1, gy1, gz1 = eval_gravity(ev_post, pts)
            gr, gp = cart_to_cyl_g(gx1 - gx0, gy1 - gy0, pp)
            draws.append((rp, pp, zp, assemble_obs_vector(U1 - U0, gr, gp, gz1 - gz0)))

    shape = (len(alphas), len(ms), len(ns))
    masses = np.empty(shape + (len(seeds),))
    ranks = np.empty_like(masses, dtype=int)
    spectral = {
        key: np.empty(shape)
        for key in basis_spectral_scales(R, alphas[0], ms[0], ns[0])
    }
    max_m, max_n = int(ms[-1]), int(ns[-1])
    packed = np.arange(2 * max_m * max_n)
    active = (packed >= 2 * max_n) | (packed % 2 == 0)
    order = (packed // (2 * max_n))[active]
    radial = ((packed // 2) % max_n + 1)[active]
    is_cos = (packed % 2 == 0)[active]

    # Delta-sigma map error, the companion of the mass error.  Scored on the
    # SAME (RHO, PHI) grid and with the SAME normalisation run_bennu_tag uses
    # for its map diagnostic -- RMS of (sigma_est - sigma_true) as a percentage
    # of peak |sigma_true| -- so a sweep cell and the main-run header are the
    # same number.  sigma_true enters only as the yardstick AFTER each fit; no
    # cell sees it while solving, exactly as for the mass.  Unweighted over grid
    # nodes, matching run_bennu_tag; the grid is uniform in rho, so outer annuli
    # are not area-weighted up.  This is the FLAT thin-sheet map: terrain
    # refinement is a post-process on one basis, not part of the ablation.
    sigma_true = res.get("sigma_true")
    RHO_map, PHI_map = res.get("RHO"), res.get("PHI")
    map_ok = sigma_true is not None and RHO_map is not None and PHI_map is not None
    map_peak = np.nan
    if map_ok:
        sigma_true = np.asarray(sigma_true, float)
        map_peak = float(np.max(np.abs(sigma_true)))
        map_ok = bool(np.isfinite(map_peak) and map_peak > 0)
    if map_ok:
        st_flat = sigma_true.ravel()
        n_map = st_flat.size
        sts = float(st_flat @ st_flat)
        rho_flat = np.asarray(RHO_map, float).ravel()
        phi_flat = np.asarray(PHI_map, float).ravel()
    map_rmse = np.full(shape + (len(seeds),), np.nan)
    map_rmse_mean_coeff = np.full(shape, np.nan)

    # Independent alpha panels run concurrently. The worker cap keeps QR and
    # eigendecompositions concurrent without allowing BLAS to oversubscribe.
    def solve_alpha(item):
        ia, alpha = item
        zeros = {m: jn_zeros(m, max_n) for m in range(max_m)}

        # One map basis per alpha, in the packing of the LARGEST basis: the
        # modes do not depend on m_max or n_max, so every sub-basis is a column
        # subset and reuses this matrix.  Only the normal equations of the map
        # are kept -- G'G, G'sigma_true, sigma_true'sigma_true -- which turns
        # each cell's RMSE into a small quadratic form instead of a full map
        # reconstruction at every grid node.
        if map_ok:
            R_alpha = alpha * R
            pref = 1.0 / (2.0 * np.pi * G_W * R_alpha)
            Gmap = np.empty((n_map, order.size))
            rad_cache = {}
            for icol in range(order.size):
                m, n = int(order[icol]), int(radial[icol])
                rad = rad_cache.get((m, n))
                if rad is None:
                    jmn = zeros[m][n - 1]
                    rad = pref * jmn * BesselJ(m, (jmn / R_alpha) * rho_flat)
                    rad_cache[(m, n)] = rad
                ang = np.cos(m * phi_flat) if is_cos[icol] else np.sin(m * phi_flat)
                Gmap[:, icol] = rad * ang
            GtG = Gmap.T @ Gmap
            Gts = Gmap.T @ st_flat
            del Gmap, rad_cache

            def map_rmse_of(x):
                mse = (float(x @ (GtG @ x)) - 2.0 * float(x @ Gts) + sts) / n_map
                return 100.0 * np.sqrt(max(mse, 0.0)) / map_peak

        cases = []
        local_spectral = {key: np.empty((len(ms), len(ns))) for key in spectral}
        for im, mm in enumerate(ms):
            for jn, nn in enumerate(ns):
                cols = np.flatnonzero((order < mm) & (radial <= nn))
                f = mass_functional(R, alpha, mm, nn, zeros)
                local = np.arange(f.size)
                f = f[(local >= 2 * nn) | (local % 2 == 0)]
                cases.append((im, jn, cols, f))
                scales = basis_spectral_scales(R, alpha, mm, nn, zeros)
                for key in local_spectral:
                    local_spectral[key][im, jn] = scales[key]
        local_masses = np.empty((len(ms), len(ns), len(seeds)))
        local_ranks = np.empty_like(local_masses, dtype=int)
        local_map = np.full(local_masses.shape, np.nan)
        local_map_mean = np.full((len(ms), len(ns)), np.nan)
        # Running coefficient sum per cell: the mass panel inverts the mean over
        # draws, so the plotted map error is the map of the mean coefficients.
        coeff_sum = np.zeros((len(ms), len(ns), order.size)) if map_ok else None
        for idraw, (rp, pp, zp, db) in enumerate(draws):
            A, _ = build_design_matrix(rp, pp, zp, alpha * R, max_m, max_n)
            Q, R_des = qr(A[:, active], mode="economic")
            qb = Q.T @ db
            for im, jn, cols, f in cases:
                R_sub = R_des[:, cols]
                gram = R_sub.T @ R_sub
                rhs = R_sub.T @ qb
                eig, vec = np.linalg.eigh(gram)
                threshold = (cond * np.sqrt(max(eig[-1], 0.0))) ** 2
                keep = eig > threshold
                if keep.any():
                    vk = vec[:, keep]
                    local_masses[im, jn, idraw] = np.sum(
                        (vk.T @ f) * (vk.T @ rhs) / eig[keep]
                    )
                else:
                    local_masses[im, jn, idraw] = 0.0
                local_ranks[im, jn, idraw] = int(keep.sum())
                if map_ok:
                    x = np.zeros(order.size)
                    if keep.any():
                        x[cols] = vk @ ((vk.T @ rhs) / eig[keep])
                    coeff_sum[im, jn] += x
                    local_map[im, jn, idraw] = map_rmse_of(x)
        if map_ok:
            for im in range(len(ms)):
                for jn in range(len(ns)):
                    local_map_mean[im, jn] = map_rmse_of(coeff_sum[im, jn] / len(draws))
        return (
            ia,
            local_masses,
            local_ranks,
            local_spectral,
            local_map,
            local_map_mean,
        )

    worker_count = max(1, min(int(BASIS_SWEEP_WORKERS), len(alphas)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for (
            ia,
            local_masses,
            local_ranks,
            local_spectral,
            local_map,
            local_map_mean,
        ) in pool.map(solve_alpha, enumerate(alphas)):
            masses[ia] = local_masses
            ranks[ia] = local_ranks
            map_rmse[ia] = local_map
            map_rmse_mean_coeff[ia] = local_map_mean
            for key in spectral:
                spectral[key][ia] = local_spectral[key]
            if verbose:
                print(
                    f"    alpha={alphas[ia]:g}: completed ({time.time()-t0:.0f}s)",
                    flush=True,
                )

    mass_mean = masses.mean(axis=-1)
    mass_std = masses.std(axis=-1, ddof=1) if len(seeds) > 1 else np.full(shape, np.nan)
    sw = dict(
        alphas=alphas,
        m_max_values=ms,
        n_max_values=ns,
        cond=float(cond),
        seeds=seeds,
        dM_draws=masses,
        dM_mean=mass_mean,
        dM_std=mass_std,
        dM_true=truth,
        ratio=mass_mean / truth,
        ratio_std=mass_std / abs(truth),
        error_percent=relative_error_percent(mass_mean, truth),
        error_std_percent=100.0 * mass_std / abs(truth),
        # Delta-sigma map error, percent of peak |sigma_true|.  `map_rmse_percent`
        # is the plotted one (map of the mean coefficients, the mass panel's
        # convention); `map_rmse_draw_mean` is the typical SINGLE-epoch map error
        # and `map_rmse_std_percent` its sampling scatter, not an OD uncertainty.
        map_rmse_percent=map_rmse_mean_coeff,
        map_rmse_draws=map_rmse,
        map_rmse_draw_mean=map_rmse.mean(axis=-1),
        map_rmse_std_percent=(
            map_rmse.std(axis=-1, ddof=1) if len(seeds) > 1 else np.full(shape, np.nan)
        ),
        map_peak=map_peak,
        rank=ranks,
        reference=np.array([res["alpha"], res["m_max"], res["n_max"]]),
        N_field=res["N_field"],
        R_star=R,
        H=res["H"],
        clearance=res["clearance"],
        cx=res["cx"],
        cy=res["cy"],
        z_sheet=res["z_sheet"],
        density=res["density"],
        grid_dx=float(res["gx"][1] - res["gx"][0]),
        grid_dy=float(res["gy"][1] - res["gy"][0]),
        mode=MODE,
        **spectral,
    )
    if verbose:
        print(
            f"    ensemble mass-error range: {sw['error_percent'].min():+.1f} .. "
            f"{sw['error_percent'].max():+.1f} %"
        )
        if map_ok:
            mr = sw["map_rmse_percent"]
            print(
                f"    Δσ map RMSE range: {mr.min():.1f} .. {mr.max():.1f} % of "
                f"peak |Δσ| ({map_peak:.0f} {_U['sd']})"
            )
        print(
            "    Reference basis is marked in the plots; no defaults are selected from truth."
        )
        iqr = np.subtract(*np.percentile(sw["error_percent"], [75, 25], axis=1))
        print(
            f"    m_max collapse: {100 * np.mean(iqr <= 0.5):.0f} % of "
            f"(alpha, n_max) columns vary by ≤ 0.5 points over m_max"
        )
        for ia, alpha in enumerate(alphas):
            print(
                f"    alpha={alpha:g}: included lambda_min "
                f"{spectral['lambda_min'][ia].min():.2f}..{spectral['lambda_min'][ia].max():.2f}, "
                f"lambda_max={spectral['lambda_max'][ia, 0, 0]:.2f} {_U['len']}"
            )
    return sw


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
        """GLOBAL's 3-D look (its fig 1 `make_plots`): the default panes with
        their grid, default tick text, plain axis labels padded clear of the
        numbers.  The ONE thing changed here is the depth sort, and only
        because this panel draws a translucent body over a full-frame surface.
        """
        # mplot3d depth-sorts whole ARTISTS, and the terrain is ONE artist
        # spanning the frame: whatever its alpha, the analysis cylinder was
        # painted over and invisible at every view angle.  Turning the sort off
        # makes zorder mean what it means in 2-D, and the cylinder — drawn last
        # below — becomes a translucent overlay the site still reads through.
        ax.computed_zorder = False
        # 3-D labels sit beyond their tick labels; mplot3d's default 4 pt pad
        # puts them on the numbers once the text is scaled up for print
        ax.set_xlabel(f"x [{_UL['len']}]", labelpad=LPAD3D)
        ax.set_ylabel(f"y [{_UL['len']}]", labelpad=LPAD3D)
        # z needs a wider pad than x/y: its tick labels are the tallest numbers
        # on the plot ("17.5"), and at LPAD3D the label sits on top of them
        ax.set_zlabel(f"z [{_UL['len']}]", labelpad=LPAD3D * 1.9)
        ax.view_init(elev=28, azim=-60)

    def _draw_cylinder(ax, color=ACCENT, alpha=0.13, n_th=48, lw=1.5, label=None):
        """GLOBAL `draw_cylinder`: translucent lateral wall, the two end rings
        and eight generatrices, so the wall reads as a wall and not as a tint.

        The wall is deliberately faint (both halves are painted over the site,
        so the tint compounds) and the LINES carry the shape: they are what
        stays legible once the surface is 13 % opaque.
        """
        th = np.linspace(0.0, 2.0 * np.pi, n_th)
        TH, ZZ = np.meshgrid(th, [z0, z0 + H])
        ax.plot_surface(
            cx + R * np.cos(TH),
            cy + R * np.sin(TH),
            ZZ,
            color=color,
            alpha=alpha,
            linewidth=0,
            shade=False,
            zorder=6,
        )
        for t in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):  # generatrices
            ax.plot(
                [cx + R * np.cos(t)] * 2,
                [cy + R * np.sin(t)] * 2,
                [z0, z0 + H],
                color=color,
                lw=0.9,
                alpha=0.6,
                zorder=7,
            )
        for z in (z0, z0 + H):  # end rings, so the footprint reads at grazing angles
            ax.plot(
                cx + R * np.cos(th),
                cy + R * np.sin(th),
                np.full_like(th, z),
                color=color,
                lw=lw,
                alpha=0.95,
                zorder=8,
                label=label if z == z0 else None,
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
        zorder=1,  # under the cylinder; see `_nice_3d_axes`
    )
    _draw_cylinder(ax, label=r"Analysis Cylinder $R^*,\,H$")
    # NOT `sm`: that name is the recovered sigma map further down this function
    cbar_src = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_src.set_array([])
    # pad clears the z label, which mplot3d places outboard of the z ticks;
    # at 0.10 the bar sat on top of it
    fig1.colorbar(cbar_src, ax=ax, pad=0.17, shrink=0.65).set_label(
        rf"$\Delta h$  [{_UL['len']}]",
        fontsize=10.5 * FONT_SCALE,
    )
    ax.set_xlim(cx - W, cx + W)
    ax.set_ylim(cy - W, cy + W)
    ax.set_zlim(0, z0 + H)
    # GLOBAL's `set_axes_true_shape`: the box aspect IS the data extent, so the
    # 16 m cylinder over a 30 m window keeps its true proportions instead of
    # being drawn into a tall thin box, and each axis gets a tick count scaled
    # by its share of the largest extent — at the default locator the short z
    # axis carried ten labels against the long axes' six.
    _span = np.array([2.0 * W, 2.0 * W, z0 + H], float)
    ax.set_box_aspect(_span)
    for _axis, _f in zip((ax.xaxis, ax.yaxis, ax.zaxis), _span / _span.max()):
        _axis.set_major_locator(mpl.ticker.MaxNLocator(nbins=max(3, round(5 * _f))))
    ax.legend(loc="upper left", fontsize=8 * FONT_SCALE)
    _save(fig1, outdir, "fig1_geometry.pdf")

    # ── FIGURE 2 — what TAG changed ────────────────────────────────────
    # Everything static cancels in the difference, so the differences ARE the
    # measurement: the absolute field panel was showing the background the
    # estimator never sees.  The first two panels are the differenced
    # observables the fit consumes; the third is what they become after
    # projection onto the Bessel-Fourier basis, i.e. the vector the mass
    # functional acts on.  Three panels, one story, ONE file: they are read in
    # sequence, and three files made the reader assemble the sequence by hand.
    def _field_cyl(ax, vals, label, diverging=True):
        # CONTINUOUS field, not the cloud that samples it.  The draw is 3-D and
        # random, so one (rho, z) is visited at many phi and by different
        # numbers of points; plotted as dots, the panel showed the sampler's
        # density as much as the field.  Averaging the draw into equal-area
        # (rho, z) cells and Gouraud-shading the result draws the axisymmetric
        # part of the differenced field — the part the m = 0 modes that carry
        # ΔM act on — with no gaps and no stipple.
        n_r, n_z = 26, 30  # ~13 points per cell at N_field = 10000
        r_edges = np.sqrt(np.linspace(0.0, rp.max() ** 2, n_r + 1))  # equal area
        z_edges = np.linspace(zp.min(), zp.max(), n_z + 1)
        bins = [r_edges, z_edges]
        tot = np.histogram2d(rp, zp, bins=bins, weights=vals)[0]
        cnt = np.histogram2d(rp, zp, bins=bins)[0]
        M = np.divide(tot, cnt, out=np.full(tot.shape, np.nan), where=cnt > 0)
        rc = 0.5 * (r_edges[:-1] + r_edges[1:])
        zc = 0.5 * (z_edges[:-1] + z_edges[1:])
        hole = ~np.isfinite(M)
        if hole.any():  # an empty cell would punch a white hole in the shading
            Rc, Zc = np.meshgrid(rc, zc, indexing="ij")
            M[hole] = NearestNDInterpolator(
                np.column_stack([Rc[~hole], Zc[~hole]]), M[~hole]
            )(np.column_stack([Rc[hole], Zc[hole]]))
        if diverging:
            v = np.percentile(np.abs(M), 99)
            kw = dict(cmap="RdBu_r", vmin=-v, vmax=v)
        else:  # a magnitude is one-signed, so a sequential map, not RdBu
            kw = dict(cmap="viridis_r", vmin=0.0, vmax=np.percentile(M, 99))
        sc = ax.pcolormesh(rc, zc, M.T, shading="gouraud", rasterized=True, **kw)
        cb = ax.get_figure().colorbar(sc, ax=ax, **CBAR)
        cb.set_label(label)
        ax.axhline(0.0, color="0.3", ls="--", lw=1.1, zorder=1)  # the sheet plane
        # Equal-AREA radial axis, matching the equal-area cells above: the
        # draw is uniform in VOLUME, so the number of points per unit rho grows
        # as rho and a linear axis both bunches them at the rim and gives the
        # rim cells the width of the axis cells.  Stretching x as rho^2 gives
        # equal areas equal widths, so every cell is drawn at the weight it
        # carries.  Checked: on a 2000-point draw, equal-area rho bins hold
        # 232-271 points, and rho^2/R^2 passes a KS test against uniform at
        # D = 0.017.
        ax.set_xscale(
            "function",
            functions=(
                lambda r: np.maximum(r, 0.0) ** 2,
                lambda a: np.sqrt(np.maximum(a, 0.0)),
            ),
        )
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_xlabel(rf"$\rho$  [{_UL['len']}]")
        ax.set_ylabel(rf"$z-z_0$  [{_UL['len']}]")
        _grid(ax)

    fig2, axs2 = plt.subplots(1, 3, figsize=(3 * FS[0], FS[1]), layout="constrained")
    _field_cyl(axs2[0], dU / POT_SCALE, rf"$\Delta U$  [{_POT_LBL}]")
    _field_cyl(
        axs2[1],
        dgvec / ACC_SCALE,
        rf"$|\Delta \mathbf{{g}}|$  [{_UL['acc']}]",
        diverging=False,
    )

    # (c) the SPECTRUM of the differenced coefficients the two panels project
    # onto.  Same construction the GLOBAL scripts use for their CH panel: the
    # RMS over the radial modes at each azimuthal order, so one number per m
    # rather than one bar per (m, n).
    ax = axs2[2]
    dc = res["d_coeffs"]
    m_max, n_max = res["m_max"], res["n_max"]
    amp = np.sqrt(dc[0::2] ** 2 + dc[1::2] ** 2).reshape(m_max, n_max)
    ms = np.arange(m_max)
    rms_m = np.sqrt((amp**2).mean(axis=1))
    # PHYSICAL units, not a ratio: A_mn and B_mn carry the units of a potential
    # (the Bessel-exponential product is dimensionless), so the power of the
    # differenced signal at each azimuthal order is read off this axis in the
    # run's own units and sits on the same scale as the OD uncertainty under
    # it — which dividing both by the largest order hid.  Whitening by σ
    # instead would buy the dimensionless axis at the price of the one thing
    # this panel is for: the SIZE of the differenced signal.
    # No spread band over n: the radial modes at one m are not repeated
    # measurements of one quantity, so a 1-sigma envelope around their RMS
    # invited exactly the reading it cannot support, an error bar on the curve.
    lo, hi = float(rms_m.min()), float(rms_m.max())
    # The ASSUMED OD uncertainty of the same coefficients, in the SAME units,
    # drawn the way GLOBAL's fig 4 draws it: as the 1σ/2σ/3σ NOISE FLOOR the
    # signal stands on.  cov["sigma_delta"] combines the two epochs' od_sigma
    # (Section 4b), reduced over n exactly as the signal is.  The height of the
    # curve above the shading is the per-order signal-to-noise the mass
    # estimate runs on, now readable in sigmas rather than as a gap between two
    # lines; the ramp is not flat because od_sigma anchors on the epoch's own
    # m=0 power (eps|CS|, floor-guarded) before grading in m, so its overall
    # level follows the epochs while its SHAPE in m does not.
    cov = res.get("cov")
    if cov is not None:
        sd = np.asarray(cov["sigma_delta"], float)
        s_amp = np.sqrt(sd[0::2] ** 2 + sd[1::2] ** 2).reshape(m_max, n_max)
        rms_sig = np.sqrt((s_amp**2).mean(axis=1))
        lo = min(lo, float(rms_sig.min()))
        sigma_bands(ax, ms, rms_sig)
    ax.plot(
        ms,
        rms_m,
        "-o",
        color=COLOR[2],
        lw=1.8,
        ms=9,
        mec="k",
        mew=0.7,
        zorder=5,
        label=r"Signal  $\Delta\mathbf{CS}$",
    )
    if cov is not None:
        pct = r"\%" if USE_TEX else "%"
        # a proxy handle for the shading: the ramp has no legend entry of its
        # own, and the ε it was built from belongs on the panel
        ax.fill_between(
            [],
            [],
            [],
            facecolor=_sigma_shade(0.45),
            edgecolor="none",
            label=(r"OD $\sigma_{\Delta\mathbf{CS}}$"),
        )
        # Upper right: the signal curve falls by an order of magnitude across
        # the panel and the shading runs along the bottom, so the corner above
        # the tail of the signal is the only empty one.
        ax.legend(frameon=False, loc="upper right", fontsize=9.5 * FONT_SCALE)
    ax.set_yscale("log")
    # the ramp reaches down to zero, which a log axis cannot show, so the floor
    # still comes from the CURVE and the 1-sigma level (GLOBAL's fig 4)
    ax.set_ylim(lo * 0.45, hi * 2.5)
    ax.set_xticks(ms)
    ax.set_xlabel(r"Azimuthal Order $m$  [-]")
    # GLOBAL's fig 4 name for the same quantity — the RMS of a coefficient
    # group, here over the radial modes at one m.  Its axis reads [-] because
    # that script runs non-dimensional (G = 1, M* = 1); this one runs in SI, so
    # A_mn and B_mn carry the units of a potential and the axis says so.  Under
    # MODE = "ND" the unit string here becomes "-" and the two labels coincide.
    ax.set_ylabel(rf"RMS Coefficient Amplitude  [{_UL['pot']}]")
    # y only: the shading spans x continuously, and vertical grid lines drawn
    # over it read as structure in the noise (GLOBAL's fig 4 does the same)
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    _save(fig2, outdir, "fig2_gravity_change.pdf")

    # ── FIGURE 3 — recovered vs true mass change ───────────────────────
    # Three panels, one file and one row: the two maps and the error between
    # them, which are read against each other rather than one at a time.  The
    # coefficient spectrum moved to figure 2, beside the differenced fields it
    # is the projection of; the numeric summary became the terminal's LaTeX
    # tables; and the azimuthal profiles were dropped — the radial cut already
    # carries the bandlimit story, and the map itself shows the azimuthal
    # structure better than three slices through it.
    # The row is drawn TWICE, from the two maps the pipeline actually makes:
    # the plain linear flat-sheet inverse (`wahr_invert` alone — the first
    # iteration, terrain not yet accounted for) and the terrain-refined map at
    # the end of the damped-Richardson passes.  Both use the same colour
    # limits, so the pair reads as a before/after instead of two independently
    # scaled pictures, and the error panel of the first is the baseline the
    # second has to beat.
    sm_flat = np.asarray(res.get("sigma_map_flat", sm), float)
    two_stage = bool(res.get("refine_diag")) and not np.array_equal(sm_flat, sm)
    vmax = max(
        np.percentile(np.abs(sm), 99),
        np.percentile(np.abs(sm_flat), 99),
        np.percentile(np.abs(st), 99),
    )
    tc = np.linspace(0, 2 * np.pi, 200)

    _wrap = _polar_closed
    Xw, Yw = _wrap(Xp), _wrap(Yp)
    # One colourbar per panel.  With the titles gone the bar labels are what
    # identify the panels, so each says explicitly which field it shows — and
    # which stage of the inversion it came from; and all three use the same
    # GLOBAL fraction/pad on equal-aspect axes, so the bars come out the same
    # size across the row.

    def _decor3(ax):
        """Footprint circle, equal aspect and axis labels, on every Δσ map."""
        ax.plot(R * np.cos(tc), R * np.sin(tc), "k--", lw=1.2, alpha=0.65)
        ax.set_aspect("equal")
        ax.set_xlabel(rf"$x-x_0$  [{_UL['len']}]")
        ax.set_ylabel(rf"$y-y_0$  [{_UL['len']}]")

    # The third panel is drawn exactly like the first two: the ERROR map, on
    # the same colour convention as the two fields beside it.  A POINTWISE
    # relative error is not drawable here — Δσ_true is signed and passes
    # through zero over most of the disc, so 100(est−true)/true diverges on a
    # ring through the middle of the picture and any mask that hides it is a
    # free parameter deciding what the reader sees.  The error is therefore
    # normalised by the single scale the field has, max|Δσ_true|: every cell
    # is defined, none is greyed out, and the bar means the same thing
    # everywhere.  Its range is set by the WORSE of the two stages, so the
    # refinement shows up as colour draining out of the panel.
    pk = float(np.abs(st).max()) or 1.0
    err_ref = 100.0 * (sm - st) / pk
    err_flat = 100.0 * (sm_flat - st) / pk
    verr = max(
        float(np.percentile(np.abs(err_flat if two_stage else err_ref), 98)), 1.0
    )
    _epct = r"[\%]" if USE_TEX else "[%]"

    def _row3(mp, err, stage, fname):
        """One figure-3 row — estimate, truth, error — for ONE inversion stage."""
        fig, axs = plt.subplots(
            1, 3, figsize=(3 * FS_MAP[0], FS_MAP[1]), layout="constrained"
        )
        for ax, (fld, lab) in zip(
            axs[:2],
            [
                (mp, rf"Estimated $\widehat{{\Delta\varrho}}$, {stage}  [{_UL['sd']}]"),
                (st, rf"True $\Delta\varrho$  [{_UL['sd']}]"),
            ],
        ):
            c = _map(ax, Xw, Yw, _wrap(fld), "RdBu_r", -vmax, vmax)
            _decor3(ax)
            fig.colorbar(c, ax=ax, **CBAR).set_label(lab)
        ce = _map(axs[2], Xw, Yw, _wrap(err), "RdBu_r", -verr, verr)
        fig.colorbar(ce, ax=axs[2], **CBAR).set_label(
            r"$(\widehat{\Delta\varrho}-\Delta\varrho)/\max|\Delta\varrho|$  " + _epct
        )
        _decor3(axs[2])
        # (the old Summary text panel was a fourth cell of this figure; every
        #  number in it is now in the terminal's LaTeX tables)
        _save(fig, outdir, fname)
        return fig

    figs3 = []
    if two_stage:
        figs3.append(
            _row3(sm_flat, err_flat, "Flat Sheet", "fig3a_mass_change_flat.pdf")
        )
    figs3.append(
        _row3(
            sm,
            err_ref,
            "Terrain Refined" if two_stage else "Flat Sheet",
            "fig3_mass_change.pdf",
        )
    )

    # NO plt.show() here: it blocks, so anything created after this call would
    # be built and never displayed.  The caller shows every figure at the end.
    return [fig1, fig2, *figs3]


def plot_refinement(res, outdir=None):
    """What the terrain refinement buys, from the run's own `refine_diag`.

    ONE panel: RMSE of the Δσ map against the truth, pass by pass, in percent
    of max|Δσ_true| (the normalisation of fig3 and the tables).  Pass 0 IS the
    plain flat-sheet `wahr_invert` map — the dotted reference line — so the
    shaded area between it and the curve is the refinement's whole gain, and
    the comparison against the flat sheet lives inside the one curve instead
    of beside it.

    The truth-free residual ‖d−TΔσ‖/‖d‖ and the two error DISTRIBUTIONS stay
    in `refine_diag` and in the run report, but are not plotted: under w = 0.5
    the residual cycles (Section 4a) instead of tracking the map error, and the
    distributions are one line of numbers rather than a panel.  Returns []
    when the refinement was off (`basis_sweep` runs with refine_map=False).
    """
    d = res.get("refine_diag") or {}
    if not d.get("pass"):
        return []
    p = np.asarray(d["pass"], float)
    rmse = np.asarray(d["rmse"], float)
    pct = r"\%" if USE_TEX else "%"

    fig, ax = plt.subplots(figsize=FS)
    # the gain, as an area: everything between the flat sheet and the curve
    ax.fill_between(p, rmse, rmse[0], color=CH_VIOLET, alpha=0.13, lw=0, zorder=1)
    ax.axhline(
        rmse[0], color="0.35", ls=":", lw=1.5, zorder=2, label="Flat Sheet (Pass 0)"
    )
    ax.plot(
        p,
        rmse,
        color=CH_VIOLET,
        lw=2.4,
        marker="o",
        ms=7,
        mfc="w",
        mec=CH_VIOLET,
        mew=1.8,
        zorder=4,
        label="Terrain Refined",
    )
    # the kept map: the one filled marker on the curve
    ax.plot(p[-1], rmse[-1], marker="o", ms=8, color=CH_VIOLET, zorder=5)
    ax.legend(frameon=False, loc="upper right", fontsize=10 * FONT_SCALE)
    ax.set_xlabel("Iteration [-]")
    ax.set_ylabel(rf"Map RMSE  [{pct} of $\max|\Delta\varrho|$]")
    ax.set_xlim(p[0], p[-1])
    ax.set_ylim(0.0, rmse[0] * 1.22)
    ax.set_xticks(p[:: max(1, int(len(p) // 9))])
    _grid(ax)
    _save(fig, outdir, "fig6_terrain_refinement.pdf")
    return [fig]


def plot_basis_sweep(sw, outdir=None, error_limits=(-100.0, 100.0), band=5.0):
    """One page for the whole (alpha, m_max, n_max) cube of mass errors.

    Each cell is the signed percentage error 100*(estimate - true)/true at the
    main run's FIXED cutoff.  Only two of the three axes carry information: the
    TAG signal is axisymmetric enough that adding azimuthal orders beyond
    m_max=2 changes Delta M by a fraction of a point, so the left panel medians
    over m_max and keeps alpha and n_max, and the right panel shows one
    uncollapsed (m_max, n_max) plane at the main run's alpha -- including the
    low-m_max rows the median outvotes.  The collapse is faithful because that
    median summarises a nearly degenerate population: the share of
    (alpha, n_max) columns whose interquartile range over m_max stays within
    0.5 points is reported by `basis_sweep` rather than drawn here.
    Both panels share the diverging error scale centred on exact recovery; the
    default -100..100 % range keeps the contrast around zero and the colorbar
    extensions flag larger under/overestimates.
    `band` outlines |median error| = band % in panel 1.  Included wavelengths
    precede the SVD cutoff and are not a claim about resolved spatial scales.
    The input dict retains every value.  Returns a list holding the one figure.
    """
    from contextlib import nullcontext
    from matplotlib.backends.backend_pdf import PdfPages

    vmin, vmax = error_limits
    if not np.isfinite([vmin, vmax]).all() or not vmin < 0 < vmax:
        raise ValueError("error_limits must be finite and bracket 0")
    ms, ns, alphas = sw["m_max_values"], sw["n_max_values"], sw["alphas"]
    ref = sw["reference"]
    # The star marks the calibrated basis, so it needs all three of its
    # coordinates on the sweep grid.  The alpha plane the right panel shows
    # needs only the nearest swept alpha, so `ia` is NOT gated on that test:
    # an off-grid m_max or n_max used to silently fall back to alphas[0].
    has_reference = np.any(np.isclose(alphas, ref[0])) and ref[1] in ms and ref[2] in ns
    error = sw.get("error_percent")
    if error is None:
        # Compatibility with sweep dictionaries saved before the percentage
        # error field was added.
        error = relative_error_percent(sw["ratio"], 1.0)
    pct = r"\%" if USE_TEX else "%"

    # The collapse: the median over m_max of every (alpha, n_max) column.  Its
    # error bar -- the interquartile range of the same column -- is printed by
    # `basis_sweep`, not drawn: neither a third panel nor a header line said
    # anything the one number does not.
    collapsed = np.median(error, axis=1)
    ia = int(np.argmin(np.abs(alphas - ref[0])))

    def ticks(values):
        step = max(1, int(np.ceil(len(values) / 6)))
        positions = np.unique(np.r_[np.arange(0, len(values), step), len(values) - 1])
        return positions, values[positions]

    def frame(ax, xs, ys, xlabel, ylabel, rx, ry):
        # Uniformly sized categorical cells, including nonconsecutive requested
        # mode counts.  Tick labels are the actual counts.
        xt, xl = ticks(xs)
        yt, yl = ticks(ys)
        ax.set_xticks(xt, labels=[f"{v:g}" for v in xl])
        ax.set_yticks(yt, labels=[f"{v:g}" for v in yl])
        ax.minorticks_off()
        ax.tick_params(labelsize=8.5 * FONT_SCALE)
        ax.set_xlabel(xlabel, fontsize=10 * FONT_SCALE)
        ax.set_ylabel(ylabel, fontsize=10 * FONT_SCALE)
        if has_reference:
            ax.plot(
                np.flatnonzero(np.isclose(xs, rx))[0],
                np.flatnonzero(np.isclose(ys, ry))[0],
                marker="*",
                ms=14,
                color="w",
                mec="k",
                mew=0.8,
                ls="none",
                zorder=5,
            )

    def lambda_max_axis(ax):
        # alpha alone sets the largest included wavelength, 2*pi/k_min.
        lam = sw["lambda_max"][:, 0, 0]
        index = np.arange(len(alphas))
        sec = ax.secondary_yaxis(
            "right",
            functions=(
                lambda i: np.interp(i, index, lam),
                lambda v: np.interp(v, lam, index),
            ),
        )
        # Tick it only where alpha is sampled: the interpolation is exact
        # there, and clamps outside the swept range.
        at, _ = ticks(alphas)
        sec.set_yticks(lam[at], labels=[f"{v:.0f}" for v in lam[at]])
        sec.set_ylabel(rf"$\lambda_{{\max}}$  [{_UL['len']}]", fontsize=9 * FONT_SCALE)
        sec.tick_params(labelsize=8 * FONT_SCALE)

    norm = mpl.colors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    lower, upper = np.any(error < vmin), np.any(error > vmax)
    extend = (
        "both" if lower and upper else "min" if lower else "max" if upper else "neither"
    )
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.6), layout="constrained")

    # 1. Every alpha and n_max, m_max collapsed by the median.
    im = axs[0].imshow(
        collapsed,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=norm,
    )
    axs[0].contour(
        np.abs(collapsed), levels=[band], colors="k", linewidths=1.0, linestyles="--"
    )
    frame(
        axs[0],
        ns,
        alphas,
        r"Radial Count $n_{\max}$  [-]",
        r"Boundary Ratio $\alpha$  [-]",
        ref[2],
        ref[0],
    )
    lambda_max_axis(axs[0])

    # 2. One uncollapsed plane, at the alpha the main run uses.  The lambda_min
    #    contours are the included resolution: it is n_max that sets it.
    axs[1].imshow(
        error[ia],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=norm,
    )
    cs = axs[1].contour(
        sw["lambda_min"][ia], levels=[5, 10, 20, 50], colors="0.25", linewidths=0.9
    )
    axs[1].clabel(
        cs,
        fmt=lambda v: rf"$\lambda_{{\min}}={v:g}$ {_UL['len']}",
        fontsize=7.5 * FONT_SCALE,
        inline_spacing=2,
    )
    frame(
        axs[1],
        ns,
        ms,
        r"Radial Count $n_{\max}$  [-]",
        r"Azimuthal Count $m_{\max}$  [-]",
        ref[2],
        ref[1],
    )
    axs[1].set_title(rf"$\alpha={alphas[ia]:g}$", fontsize=10 * FONT_SCALE)
    cb = fig.colorbar(
        im,
        ax=list(axs),
        fraction=0.045,
        pad=0.02,
        extend=extend,
        location="bottom",
        aspect=45,
    )
    cb.set_label(
        r"$(\widehat{\Delta M}-\Delta M_{\rm true})/\Delta M_{\rm true}$  " f"[{pct}]",
        fontsize=10 * FONT_SCALE,
    )
    cb.ax.tick_params(labelsize=8.5 * FONT_SCALE)

    handles = [
        plt.Line2D(
            [],
            [],
            color="k",
            ls="--",
            lw=1.0,
            label=rf"Median Error $=\pm{band:g}$ {pct}",
        )
    ]
    if has_reference:
        handles.insert(
            0,
            plt.Line2D(
                [],
                [],
                marker="*",
                color="w",
                mec="k",
                mew=0.8,
                ms=14,
                ls="none",
                label="Main-Run Basis",
            ),
        )
    # Below the colorbar: the panel titles reach the top edge.
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncols=len(handles),
        fontsize=9 * FONT_SCALE,
        frameon=False,
    )
    # No header: the run settings (c, N, clearance, draws) are in the sweep
    # report and in the caption, and printed over the panels they only stole
    # the height the two maps are read at.

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    buf = io.BytesIO()  # rendered in memory, written once (see _write_bytes)
    destination = PdfPages(buf) if outdir else nullcontext()
    with destination as pdf:
        if pdf is not None:
            pdf.savefig(fig, bbox_inches="tight")
    if outdir:
        _write_bytes(
            os.path.join(outdir, PREFIX + "fig7_basis_mass_ratio.pdf"), buf.getvalue()
        )
    return [fig]


def plot_basis_sweep_map(sw, outdir=None, rmse_limits=None, bands=(15.0,)):
    """The fig7 cube again, scored on the Delta-sigma MAP instead of the mass.

    Companion to `plot_basis_sweep`, same layout and the same cells: only the
    quantity changes.  Each cell is

        100 * RMS(sigma_est - sigma_true) / max|sigma_true|   [%]

    over the whole (RHO, PHI) disc, for the map of the mean coefficients at the
    main run's FIXED cutoff -- the same statistic and the same normalisation
    run_bennu_tag prints as "RMSE" in its map-error line.  The reference cell
    reproduces that header number when the main run used refine_map=False;
    with refinement on, the header is the refined map and the cell is the flat
    thin-sheet one, because refinement is a post-process on a single basis and
    is deliberately outside the ablation.  sigma_true is never seen by a fit;
    it is only the yardstick each cell is scored against afterwards.

    Two things make this panel read differently from the mass one.  The
    quantity is non-negative, so the scale is sequential and logarithmic rather
    than diverging about zero -- there is no sign to cancel, and a basis that
    misses the map by ten peak-amplitudes is a decade away from one that misses
    it by one.  And 100 % is a real line, not just a round number: a map whose
    RMS error equals the true peak has no skill left.  So the default scale
    tops out there and the colorbar arrow carries everything beyond -- above
    no-skill there is nothing left to tell apart, and a handful of degenerate
    cells at the smallest alpha would otherwise compress the whole informative
    10..40 % range into one shade.  `bands` are the levels contoured on the
    collapsed panel, and they have to sit where the cube actually varies: 97 %
    of it already beats 50 %, so a no-skill contour only ever outlines the
    degenerate alpha=1 row and says nothing about the plateau one would pick a
    basis from.  The default is the single 15 % line around that plateau's best
    corner, which is the only threshold a reader chooses a basis against.
    Where the mass panel shows large cells of
    near-exact recovery, the same cells can sit far above 100 % here: the mass
    is one integral of the map, and cancellation in that integral is invisible
    to it.  That contrast is the point of the figure.

    Falls back to the per-draw mean when a sweep saved before the
    mean-coefficient map error was added is passed.  The input dict retains
    every value.  Returns a list holding the one figure.
    """
    from contextlib import nullcontext
    from matplotlib.backends.backend_pdf import PdfPages

    rmse = sw.get("map_rmse_percent")
    if rmse is None:
        rmse = sw.get("map_rmse_draw_mean")
    if rmse is None:
        raise ValueError(
            "this sweep carries no Delta-sigma map error; rerun basis_sweep on a "
            "result that includes sigma_true, RHO and PHI"
        )
    rmse = np.asarray(rmse, float)
    if not np.isfinite(rmse).any():
        raise ValueError("the Delta-sigma map error is all non-finite")

    ms, ns, alphas = sw["m_max_values"], sw["n_max_values"], sw["alphas"]
    ref = sw["reference"]
    # The star marks the calibrated basis, so it needs all three of its
    # coordinates on the sweep grid.  The alpha plane the right panel shows
    # needs only the nearest swept alpha, so `ia` is NOT gated on that test:
    # an off-grid m_max or n_max used to silently fall back to alphas[0].
    has_reference = np.any(np.isclose(alphas, ref[0])) and ref[1] in ms and ref[2] in ns
    pct = r"\%" if USE_TEX else "%"

    # Same collapse as the mass figure: the median over m_max of every
    # (alpha, n_max) column.  It is the same population of cells, so the
    # degeneracy `basis_sweep` reports for the mass carries over.
    collapsed = np.median(rmse, axis=1)
    ia = int(np.argmin(np.abs(alphas - ref[0])))

    finite = rmse[np.isfinite(rmse)]
    if rmse_limits is None:
        vmin = float(np.min(finite[finite > 0]))
        vmax = max(min(100.0, float(np.max(finite))), 2.0 * vmin)
    else:
        vmin, vmax = rmse_limits
    if not np.isfinite([vmin, vmax]).all() or not 0 < vmin < vmax:
        raise ValueError("rmse_limits must be finite and positive with vmin < vmax")
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    lower, upper = np.any(finite < vmin), np.any(finite > vmax)
    extend = (
        "both" if lower and upper else "min" if lower else "max" if upper else "neither"
    )

    def ticks(values):
        step = max(1, int(np.ceil(len(values) / 6)))
        positions = np.unique(np.r_[np.arange(0, len(values), step), len(values) - 1])
        return positions, values[positions]

    def frame(ax, xs, ys, xlabel, ylabel, rx, ry):
        xt, xl = ticks(xs)
        yt, yl = ticks(ys)
        ax.set_xticks(xt, labels=[f"{v:g}" for v in xl])
        ax.set_yticks(yt, labels=[f"{v:g}" for v in yl])
        ax.minorticks_off()
        ax.tick_params(labelsize=8.5 * FONT_SCALE)
        ax.set_xlabel(xlabel, fontsize=10 * FONT_SCALE)
        ax.set_ylabel(ylabel, fontsize=10 * FONT_SCALE)
        if has_reference:
            ax.plot(
                np.flatnonzero(np.isclose(xs, rx))[0],
                np.flatnonzero(np.isclose(ys, ry))[0],
                marker="*",
                ms=14,
                color="w",
                mec="k",
                mew=0.8,
                ls="none",
                zorder=5,
            )

    def lambda_max_axis(ax):
        # alpha alone sets the largest included wavelength, 2*pi/k_min.
        lam = sw["lambda_max"][:, 0, 0]
        index = np.arange(len(alphas))
        sec = ax.secondary_yaxis(
            "right",
            functions=(
                lambda i: np.interp(i, index, lam),
                lambda v: np.interp(v, lam, index),
            ),
        )
        at, _ = ticks(alphas)
        sec.set_yticks(lam[at], labels=[f"{v:.0f}" for v in lam[at]])
        sec.set_ylabel(rf"$\lambda_{{\max}}$  [{_UL['len']}]", fontsize=9 * FONT_SCALE)
        sec.tick_params(labelsize=8 * FONT_SCALE)

    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.6), layout="constrained")

    # 1. Every alpha and n_max, m_max collapsed by the median.
    im = axs[0].imshow(
        collapsed,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma_r",
        norm=norm,
    )
    lo, hi = np.nanmin(collapsed), np.nanmax(collapsed)
    levels = [float(b) for b in np.atleast_1d(bands) if lo < b < hi]
    styles = ["--", "-", ":"]
    for i, lev in enumerate(levels):
        axs[0].contour(
            collapsed,
            levels=[lev],
            colors="k",
            linewidths=1.0,
            linestyles=styles[i % len(styles)],
        )
    frame(
        axs[0],
        ns,
        alphas,
        r"Radial Count $n_{\max}$  [-]",
        r"Boundary Ratio $\alpha$  [-]",
        ref[2],
        ref[0],
    )
    lambda_max_axis(axs[0])

    # 2. One uncollapsed plane, at the alpha the main run uses.  The lambda_min
    #    contours are the included resolution: it is n_max that sets it.
    axs[1].imshow(
        rmse[ia],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma_r",
        norm=norm,
    )
    cs = axs[1].contour(
        sw["lambda_min"][ia], levels=[5, 10, 20, 50], colors="0.25", linewidths=0.9
    )
    axs[1].clabel(
        cs,
        fmt=lambda v: rf"$\lambda_{{\min}}={v:g}$ {_UL['len']}",
        fontsize=7.5 * FONT_SCALE,
        inline_spacing=2,
    )
    frame(
        axs[1],
        ns,
        ms,
        r"Radial Count $n_{\max}$  [-]",
        r"Azimuthal Count $m_{\max}$  [-]",
        ref[2],
        ref[1],
    )
    axs[1].set_title(rf"$\alpha={alphas[ia]:g}$", fontsize=10 * FONT_SCALE)
    cb = fig.colorbar(
        im,
        ax=list(axs),
        fraction=0.045,
        pad=0.02,
        extend=extend,
        location="bottom",
        aspect=45,
    )
    cb.set_label(
        r"RMS$(\widehat{\Delta\varrho}-\Delta\varrho_{\rm true})\,/\,"
        r"\max|\Delta\varrho_{\rm true}|$  " f"[{pct}]",
        fontsize=10 * FONT_SCALE,
    )
    # Plain numbers on a sub-decade log bar: 10^1/10^2 alone would label almost
    # nothing across the 10..100 % range the cells actually occupy.
    cb.ax.xaxis.set_major_locator(mpl.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    cb.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    cb.ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    cb.ax.tick_params(labelsize=8.5 * FONT_SCALE)

    handles = []
    if has_reference:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="*",
                color="w",
                mec="k",
                mew=0.8,
                ms=14,
                ls="none",
                label="Main-Run Basis",
            )
        )
    for i, lev in enumerate(levels):
        handles.append(
            plt.Line2D(
                [],
                [],
                color="k",
                ls=styles[i % len(styles)],
                lw=1.0,
                label=(
                    rf"Median RMSE $={lev:g}$ {pct}"
                    + (" (No Skill)" if lev == 100.0 else "")
                ),
            )
        )
    if handles:
        fig.legend(
            handles=handles,
            loc="outside upper center",
            ncols=len(handles),
            fontsize=9 * FONT_SCALE,
            frameon=False,
        )

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    buf = io.BytesIO()  # rendered in memory, written once (see _write_bytes)
    destination = PdfPages(buf) if outdir else nullcontext()
    with destination as pdf:
        if pdf is not None:
            pdf.savefig(fig, bbox_inches="tight")
    if outdir:
        _write_bytes(
            os.path.join(outdir, PREFIX + "fig7b_basis_map_rmse.pdf"), buf.getvalue()
        )
    return [fig]


# Everything the calibration depends on besides `setup` and the grid: the
# site-location rule, the sheet-plane rule and the selection rule.  Bump this
# string when any of them changes; every cached calibration is then recomputed.
CALIBRATION_RULES_VERSION = (
    "fullgrid-site/signed-centroid-plane/cv-instability-fewest-coeffs-bandlimit/v4"
)


def calibrated_basis(setup, cache_path=None, verbose=True, **cal_kw):
    """
    Truth-free (alpha, m_max, n_max, cond) for `setup`, via basis_calibration.

    The choice depends on everything that shapes the field draws (footprint,
    height, N_field, grid, units, meshes), on the candidate grid and on the
    rules in CALIBRATION_RULES_VERSION.  All of them form the cache key, so any
    change triggers a fresh calibration; an unchanged configuration reuses the
    stored choice instead of repeating the 16 gravity-field draws.  setup's own
    alpha, m_max, n_max and cond are NOT in the key: they are the outputs.
    """
    import hashlib, json

    if cache_path is None:
        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            PREFIX + "calibration_cache.json",
        )
    cal_kw = dict(
        dict(
            alphas=CAL_ALPHAS,
            m_max_values=CAL_M_MAX,
            n_max_values=CAL_N_MAX,
            k_target_R=KR_TARGET_DEFAULT,
        ),
        **cal_kw,
    )
    meshes = {
        p: os.path.getmtime(setup[p])
        for p in ("path_pre", "path_post")
        if os.path.exists(setup[p])
    }
    config = dict(
        setup={
            k: v
            for k, v in setup.items()
            if k not in ("alpha", "m_max", "n_max", "cond", "verbose")
        },
        mesh_mtime=meshes,
        MODE=MODE,
        L_REF=L_REF,
        rules=CALIBRATION_RULES_VERSION,
        calibration=cal_kw,
    )
    text = json.dumps(config, sort_keys=True, default=repr)
    key = hashlib.sha1(text.encode()).hexdigest()[:16]
    try:
        with open(cache_path) as fh:
            cache = json.load(fh)
    except (OSError, ValueError):
        cache = {}
    if key not in cache:
        site = run_bennu_tag(
            **setup, do_covariance=False, refine_map=False, verbose=False
        )
        cal = basis_calibration(site, verbose=verbose, **cal_kw)
        cache[key] = dict(
            {k: cal[k] for k in ("alpha", "m_max", "n_max", "cond")},
            config=json.loads(text),
        )
        with open(cache_path, "w") as fh:
            json.dump(cache, fh, indent=1)
    choice = {k: cache[key][k] for k in ("alpha", "m_max", "n_max", "cond")}
    if verbose:
        print(
            f"  calibrated basis: alpha={choice['alpha']:g}, m_max={choice['m_max']}, "
            f"n_max={choice['n_max']}, cond={choice['cond']:.3e}  (cache key {key})"
        )
    return choice


if __name__ == "__main__":

    setup = dict(
        path_pre="3dmeshes/Bennu_preTag.obj",
        path_post="3dmeshes/Bennu_afterTag.obj",
        density=RHO_BULK,  # [kg/m³]
        # grid_res: the raster BOTH epochs are rebuilt on.  0.30 m is the
        # DTMs' own resolution (median vertex spacing 0.250 m pre / 0.292 m
        # post; sqrt(area/N) = 0.47 / 0.54 m), so finer rasters only
        # interpolate.  Measured against a 0.10 m raster, basis held fixed:
        #  grid_res [m]  0.10  0.15  0.20  0.25  0.30  0.40  0.50  0.75  1.00
        #  ΔM_true dev  +0.00 +0.02 -0.07 -0.13 -0.26 -0.85 -0.94 +1.70 +5.50  [%]
        #  recovery err +4.68 +4.50 +4.66 +4.76 +4.82 +5.49 +5.17 +6.92 +2.77  [%]
        #  rebuild time   157    72    44    27    19    12     8     5     4   [s]
        # 0.10–0.30 m is one plateau: the truth moves < 0.3 % and the recovery
        # error < 0.35 points across a 3× refinement.  From 0.40 m the crater
        # is under-resolved; by 0.75–1.00 m the detected site and the sheet
        # plane move as well (the small error at 1.00 m is cancellation
        # between a wrong truth and a wrong estimate, not accuracy).  0.30 m
        # is the coarsest point on the plateau — 8× cheaper than 0.10 m.
        grid_res=0.30 / L_REF,
        site_center=None,  # auto-detect TAG crater from Δh
        R_star=R_STAR_SI / L_REF,
        H=16.0 / L_REF,
        clearance=0.25 / L_REF,  # points hug the surface: local terrain + this
        alpha=CH_ALPHA,  # basis: replaced by the calibration if CH_MODE="calibrated"
        m_max=CH_M_MAX,
        n_max=CH_N_MAX,
        N_field=10000,  # one draw; scatter ≈ N^-0.3, bias-dominated anyway
    )

    # 1. Basis and cutoff.  "calibrated" = truth-free (alpha, m_max, n_max, cond)
    #    for exactly this setup (basis_calibration over CAL_*; cached, and
    #    recomputed automatically when any setting changes).
    #    "fixed" = CH_ALPHA, CH_M_MAX, CH_N_MAX, CH_COND as written.
    if CH_MODE == "calibrated":
        ch = calibrated_basis(setup)
    elif CH_MODE == "fixed":
        ch = dict(alpha=CH_ALPHA, m_max=CH_M_MAX, n_max=CH_N_MAX, cond=CH_COND)
    else:
        raise ValueError(f"CH_MODE must be 'calibrated' or 'fixed', got {CH_MODE!r}")
    setup.update(alpha=ch["alpha"], m_max=ch["m_max"], n_max=ch["n_max"])
    selected_cond = ch["cond"]

    # 2. Full run at the selected cutoff.
    result = run_bennu_tag(
        **setup,
        cond=selected_cond,
        coeff_rel=0.004,  # od_sigma eps on each epoch's CS
        coeff_floor_frac=0.1,
        od_alpha=None,  # None: od_sigma's shaped rule; 0 disables growth in m
        k_epoch=1.0,  # pre/post errors fully correlated: the common field cancels
        verbose=True,
    )

    # 3. Basis-only ablation: freeze the selected cutoff and all other settings
    #    (the cutoff is chosen for the selected basis; other bases reuse it).
    # Edit BASIS_SWEEP_* above to change the three grid axes.  The ablation uses
    # BASIS_SWEEP_N_DRAWS draws; the main run supplies the first, the sweep
    # regenerates the rest, so its cell-to-cell spread is sampling scatter.
    basis_study = basis_sweep(result)
    figs = plot_basis_sweep(basis_study, outdir="Images")
    # Same cube, scored on the Delta-sigma map instead of the mass: the two
    # figures disagree wherever cancellation inside the mass integral hides a
    # map the basis cannot represent.
    figs += plot_basis_sweep_map(basis_study, outdir="Images")

    # Draw coefficients with the assumed OD covariance; reuse the same Monte
    # Carlo for the tables and figures.  No field samples are redrawn.
    result["cov"]["mc"] = covariance_mc(result, result["cov"])

    latex_tables(result, result.get("cov"))

    figs += plot_results(result, outdir="Images")
    figs += plot_refinement(result, outdir="Images")
    figs += plot_covariance_mc(
        result, result["cov"], outdir="Images", mc=result["cov"]["mc"]
    )[0]

    plt.show()  # once, with every panel built, so all of them appear

    print("\nDone.")
