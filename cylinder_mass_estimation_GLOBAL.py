"""
Interior-density recovery: Spherical Harmonics vs Spherical + Cylindrical Harmonics
===================================================================================
Author: Giovanni Fereoli / experiment build

Question
--------
Can SH + CH recover INTERIOR density better than SH alone, because cylindrical
harmonics sit close to the surface and carry localized information that global
spherical harmonics smear out?

Model — a SCALED CONSTANT-DENSITY BULK plus localized anomalies
---------------------------------------------------------------
The interior is not a cloud of mascons summing to the total mass; it is the
shape model's constant-density polyhedron, scaled by β̃, plus a few mascons
carrying the departures from homogeneity:

    U_T(r) = β̃ · U_CD(r) + Σ_j β_j · U_pt(r; p_j),     β_j = m_j / M* ,

with M* = GM known from tracking (= 1 in normalized units).  The mass budget
fixes the bulk scale outright, β̃ = 1 − Σ_j β_j, so β̃ is NOT an independent
unknown, total mass is exact by construction, and the old "Σβ = 1 known to σ_M"
pseudo-observation disappears.  The estimated vector is β = {β_j}: positive is
an over-dense concentration, negative a deficit.  Only m_j = Δρ_j v_j enters the
field, so the mass FRACTION, not the density contrast, is identifiable.

Substituting β̃ isolates the discrepancy the estimator actually fits,

    ΔU(r) = U_T(r) − U_CD(r) = Σ_j β_j [ U_pt(r; p_j) − U_CD(r) ] ,

so every design column is a CONTRAST against the homogeneous body.  Eros here
carries one shallow anomaly under the +z pole (the CH target, whose position is
recovered too) and two deep ones in the lobes.

Mechanism
---------
A deep anomaly is a low-degree feature that SH constrains well.  A small
near-surface anomaly writes mostly into HIGH-degree coefficients, truncated and
noisy from orbit, so SH is nearly blind to it and it stays degenerate with the
bulk.  Interior CH converge inside the Brillouin sphere down to the surface,
where exterior SH diverges; low-altitude data over a cylinder above the anomaly
injects exactly the local information SH lacks.

Both observables are LINEAR in β, so mass recovery is a linear Gaussian inverse
problem with an exact posterior covariance.  Position is nonlinear, handled by a
linearized (Fisher) covariance.

Truth / units
-------------
Eros shape (`3dmeshes/eros.pk`), normalized units (LU), M* = 1.  Bulk observables
are computed once to machine precision: Stokes coefficients by exact tetrahedral
quadrature of the solid harmonics (the integrand is a polynomial), near-surface
field by polyhedral_gravity.

Two observables (both fitted as DISCREPANCIES from the constant-density model)
------------------------------------------------------------------------------
  A (SH)    : fully-normalized C̄_nm, S̄_nm, degree 2..L_SH, from tracking.  No
              total-mass row — the budget is structural, β̃ = 1 − Σβ.
  B (SH+CH) : plus the CH coefficients of the near-surface field (potential +
              acceleration) in a cylinder above the anomaly, from an UNWEIGHTED
              fit of the Bessel–Fourier basis, c = Φ⁺ field.  Only what Φ can
              represent survives; the projector P_CH = Φ(ΦᵀΦ)⁻¹Φᵀ says how much.

Weights (see `od_sigma`)
------------------------
Estimation is in COEFFICIENT space, weighted per coefficient by
σ_i = eps·max(|coeff_i|, floor) — OD-like, "each coefficient known to a fixed
fraction of itself above a noise floor".  The Phi-to-field fit that manufactures
the CH coefficients from field samples is deliberately unweighted.  The analytic
covariance and the Monte-Carlo fits are fed the SAME (A, σ) blocks, so they
describe one estimator.

Experiments
-----------
  1. MASS FRACTION.  Posterior σ on each β_j and on the derived β̃, in all three
     observation models: the global field alone, the near-surface patch alone,
     and the joint fit.  The third model is what the pair is for; the SECOND is
     what makes the pair interpretable, because it separates "the patch adds
     information" from "the patch is simply better data" — see the reach map,
     where SH is a broad flat field and CH one sharp peak.
  2. POSITION.  The near-surface anomaly's location, linearized covariance.

Formulae
--------
SH (exterior) unit-mass Stokes basis, mass at (r,φ,λ), ref radius R*:
    {C̄,S̄}_nm = (1/(2n+1)) (r/R*)^n P̄_nm(sinφ) {cos,sin}(mλ)
CH (interior) basis in a cylinder (axis ẑ, radius R_cyl, extension α):
    U_mn = J_m(k_mn ρ) e^{-k_mn z} {cos,sin}(mφ),   k_mn = j_{m,n}/(α R_cyl)
Point mass at p:  U = Gm/|x-p|,  g = -Gm (x-p)/|x-p|³   (G = 1 in LU).
Constant-density bulk, unit mass:
    C̄_nm = (1/((2n+1) Vol)) ∫_body (r/R*)^n P̄_nm cos mλ dV.
"""

from __future__ import annotations
import os, math
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import jv as BesselJ, jvp as BesselJp, jn_zeros
from scipy.optimize import least_squares

import mesh_utility

try:
    import trimesh

    _HAVE_TRIMESH = True
except Exception:
    _HAVE_TRIMESH = False

try:
    from polyhedral_gravity import Polyhedron, PolyhedronIntegrity, GravityEvaluable

    _HAVE_PG = True
except Exception:
    _HAVE_PG = False


# ── plotting ────────────────────────────────────────────────────────────────
# Okabe-Ito, the standard colour-vision-deficiency-safe qualitative palette.
# The previous set paired #E6001A with #1a9641 — red against green, which is
# exactly the pair deuteranopes and protanopes cannot separate, and those two
# carried "SH+CH" and "truth draws" in adjacent plot elements.  Order here is
# by ROLE, not by hue, so every existing COLOR[i] keeps its meaning:
#   0 vermillion   the CH / "with cylinder" case
#   1 orange       truth markers, second configuration
#   2 blue         the SH-only baseline
#   3 bluish green third configuration, truth draws
#   4 purple       spare / accents
#   5 sky blue     sixth series (the CH patch curves in pt2 fig 5)
COLOR = ["#D55E00", "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9"]
# ── the observation models, named once ──────────────────────────────────────
# Every table column, every curve and every Monte-Carlo output in this file is
# keyed by one of these, and pt2 keys its own (longer) list the same way.  The
# THIRD case is the one that makes the other two legible: SH+CH beating SH says
# the patch adds something, but only CH ALONE says what the patch is — a
# local instrument that outperforms the entire global field where it looks and
# loses to it everywhere else.  Reading the pair without it invites the wrong
# conclusion, that CH is simply better data.
SH_ONLY, CH_ONLY, SH_CH = "SH", "CH", "SH + CH"
CASES = (SH_ONLY, CH_ONLY, SH_CH)
# SH keeps the blue it has always had (the "no cylinder" baseline) and SH+CH the
# vermillion ("with cylinder").  CH-only has been through the palette: the
# Okabe-Ito purple read as pink against the vermillion, and the green read as
# green.  It now takes a DEEP VIOLET, which is dark and saturated enough not to
# be confused with either the mid blue or the vermillion, and is separable under
# the common colour-vision deficiencies (it differs from both in lightness as
# well as in hue, so it survives greyscale too).
CH_VIOLET = "#5D3A9B"
CASE_COLOR = {SH_ONLY: COLOR[2], CH_ONLY: CH_VIOLET, SH_CH: COLOR[0]}
CASE_MARKER = {SH_ONLY: "o", CH_ONLY: "^", SH_CH: "s"}
CASE_LS = {SH_ONLY: "-", CH_ONLY: ":", SH_CH: "--"}
# Dash pattern per model for the PREDICTED covariance ellipses, which are all
# drawn black (colour there means the cloud a model produced, not the model).
LS_OF_CASE = {SH_ONLY: "-", CH_ONLY: "-.", SH_CH: ":"}


def case_blocks(A_sh, sig_sh, A_ch, sig_ch):
    """
    The three observation models, as the (design, sigma) block lists every
    estimator in this file consumes.  One definition, so the mass fit, the
    position fit, the detection sweep and the L_SH sweep cannot drift apart on
    what "CH only" means.
    """
    return {
        SH_ONLY: [(A_sh, sig_sh)],
        CH_ONLY: [(A_ch, sig_ch)],
        SH_CH: [(A_sh, sig_sh), (A_ch, sig_ch)],
    }


# structural elements (cylinder outlines and their labels), kept clear of the
# data colours above
ACCENT = "#882255"
# Filename stem for every figure this module writes.  A variant script can
# reassign it so its figures land beside these instead of overwriting them.
PREFIX = "global_"
# Render figure text with a real LaTeX engine (exact document fonts) or with
# matplotlib's own mathtext (much faster).  A full run is several times slower
# with USE_TEX on, because every label is a separate LaTeX compile.
# Override from the shell without editing:  GLOBAL_NO_TEX=1 python ...
USE_TEX = False  # os.environ.get("GLOBAL_NO_TEX", "") == ""


# ── font scale ──────────────────────────────────────────────────────────────
# ONE knob for every text size in this file: the rcParams below and every
# explicit `fontsize=` / `labelsize=` are written as (base * FONT_SCALE).
# Why it is needed: a 7.2 in wide figure dropped into a two-column paper at
# \linewidth (~3.4 in) is scaled by ~0.47, so 12 pt is drawn on the page at
# ~6 pt.  Raising this raises everything together and keeps the relative
# hierarchy (axis labels > ticks > legends > inset labels) intact.
#   1.00  on-screen sizes, correct if the figure is placed at its natural size
#   1.35  legible at ~0.7 x reduction (single-column, 6.5 in text width)
#   1.60  legible at ~0.5 x reduction (two-column journal)
FONT_SCALE = 1.35

mpl.rcParams.update(
    {
        "axes.prop_cycle": mpl.cycler(color=COLOR),
        # USE_TEX (above) picks the renderer.  Every label in these scripts is
        # written to be valid in BOTH modes — maths in $...$, no bare unicode,
        # no % — so flipping the switch changes only the typeface and the speed.
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
G = 1.0  # gravitational constant in normalized units
G_SI = 6.67430e-11  # polyhedral_gravity works in SI; divided out to get G = 1
SEP = "=" * 70


# NOTE: incosistency on how MC is used and results plotted. ls_fit_once used just in mc. avoid montecarlo come one over noise..
# also how analytical covafiance, fisher prior, MC inconsitnet..
# NOTE: avoid MC at all if it is on realization measurement....
# NOTE: maybe figure 2 emulatye figure 3 and put the sweep on its own...

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — SHAPE
# ═══════════════════════════════════════════════════════════════════════════


def load_eros(path="3dmeshes/eros.pk"):
    V, F = mesh_utility.read_pk_file(path)
    V, F = np.asarray(V, float), np.asarray(F, int)
    tm = trimesh.Trimesh(V, F, process=False) if _HAVE_TRIMESH else None
    R_brillouin = float(np.linalg.norm(V, axis=1).max())
    return V, F, tm, R_brillouin


def inside_body(tm, V, F, P):
    """Boolean mask: which points are inside the closed shape."""
    if tm is not None:
        return tm.contains(P)
    # crude fallback: inside the mean radius (only used if trimesh missing)
    return np.linalg.norm(P, axis=1) < np.linalg.norm(V, axis=1).mean()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — SPHERICAL-HARMONIC (Stokes) OBSERVABLE
# ═══════════════════════════════════════════════════════════════════════════


def fully_normalized_legendre_v(nmax: int, x) -> np.ndarray:
    """Vectorized `fully_normalized_legendre`: same recursions, (nmax+1, nmax+1, N)."""
    x = np.atleast_1d(np.asarray(x, float))
    P = np.zeros((nmax + 1, nmax + 1, x.size))
    P[0, 0] = 1.0
    if nmax == 0:
        return P
    sx = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    for m in range(1, nmax + 1):
        f = math.sqrt((2.0 * m + 1.0) / (2.0 * m))
        if m == 1:
            # the (2 - delta_0m) step: geodesy Pbar_11 = sqrt(3) sx, not
            # sqrt(3/2) sx.  Without it every m > 0 is short by sqrt(2) and the
            # addition theorem fails, so the coefficients are not 4pi-normalized.
            f *= math.sqrt(2.0)
        P[m, m] = f * sx * P[m - 1, m - 1]
    for m in range(0, nmax):
        P[m + 1, m] = math.sqrt(2.0 * m + 3.0) * x * P[m, m]
    for m in range(0, nmax + 1):
        for n in range(m + 2, nmax + 1):
            a = math.sqrt(((2 * n + 1) * (2 * n - 1)) / ((n - m) * (n + m)))
            b = math.sqrt(
                ((2 * n + 1) * (n + m - 1) * (n - m - 1))
                / ((2 * n - 3) * (n - m) * (n + m))
            )
            P[n, m] = a * x * P[n - 1, m] - b * P[n - 2, m]
    return P


def fully_normalized_legendre(nmax: int, x: float) -> np.ndarray:
    """
    Scalar case of `fully_normalized_legendre_v`, as (nmax+1, nmax+1).

    A wrapper, not a second implementation: the two carried the same three
    recursions line for line, differing only in `math` vs `np`.  Two copies of
    a normalization recursion is exactly where a silent convention bug hides —
    the sqrt(2) at m = 1 had to be patched in both, and a future edit to one
    would not have shown up until the coefficients were wrong.
    """
    return fully_normalized_legendre_v(nmax, x)[..., 0]


def sh_stokes_basis(pts, Lmin, Lmax, Rref):
    """
    Vectorized `sh_stokes_of_point`: row i is the unit-mass Stokes signature of a
    point mass at pts[i], in the SAME coefficient order.  (N, n_coeff)
    """
    pts = np.atleast_2d(np.asarray(pts, float))
    r = np.linalg.norm(pts, axis=1)
    lam = np.arctan2(pts[:, 1], pts[:, 0])
    Pb = fully_normalized_legendre_v(Lmax, pts[:, 2] / r)
    # cos/sin(m*lam) depend on m alone, but the loop below visits each m once
    # per degree n >= m, so the inner trig was being recomputed up to Lmax times
    # over.  Building the tables once is bit-identical (same arguments, same
    # function) and is ~30% of this routine, which the position fit calls once
    # per residual evaluation.  NOTE: (r/Rref)**n stays inside the loop with a
    # PYTHON int exponent — hoisting it as x ** arange(...) takes a different
    # numpy code path and shifts the last bits of the coefficients.
    ml = np.arange(Lmax + 1)[:, None] * lam[None, :]
    cosm, sinm = np.cos(ml), np.sin(ml)
    out = np.empty((r.size, sum(2 * (n + 1) for n in range(Lmin, Lmax + 1))))
    k = 0
    for n in range(Lmin, Lmax + 1):
        # 1/(2n+1) is the addition-theorem factor in the standard (4pi-
        # normalized, geodesy/OD) definition of a Stokes coefficient
        rr = (r / Rref) ** n / (2 * n + 1)
        for m in range(0, n + 1):
            base = rr * Pb[n, m]
            out[:, k] = base * cosm[m]
            k += 1
            out[:, k] = base * sinm[m]
            k += 1
    return out


def sh_stokes_of_point(p, Lmin, Lmax, Rref):
    """
    Unit-mass fully-normalized Stokes coefficients of a point mass at p.

    A wrapper on `sh_stokes_basis`, not a second implementation.  Both carried
    the same loop and the same 1/(2n+1) addition-theorem factor, and that factor
    is precisely what was once missing from this file — a convention living in
    two places is a convention that will eventually disagree with itself.
    """
    return sh_stokes_basis(np.asarray(p, float)[None, :], Lmin, Lmax, Rref)[0]


def A_stokes(positions, Lmin, Lmax, Rref):
    """Design matrix: mascon masses -> Stokes coefficients (n_coeff, n_mascon)."""
    return sh_stokes_basis(np.asarray(positions, float), Lmin, Lmax, Rref).T


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — CYLINDRICAL-HARMONIC (near-surface) OBSERVABLE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Cylinder:
    center: np.ndarray  # base center [x,y,z]
    radius: float
    height: float
    alpha: float = 100.0  # Bessel extension (interior CH)
    R: np.ndarray = None  # rotation (axis alignment); identity if None

    def rot(self):
        return np.eye(3) if self.R is None else self.R


def cylinder_points(cyl: Cylinder, n=300, seed=3):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    rr = np.sqrt(rng.uniform(0, cyl.radius**2, n))
    zz = rng.uniform(0, cyl.height, n)
    loc = np.column_stack([rr * np.cos(th), rr * np.sin(th), zz])
    return loc @ cyl.rot().T + cyl.center


def point_mass_field(p, obs):
    """Unit-mass potential and acceleration of a point at p, stacked [U; ax; ay; az]."""
    d = obs - p
    r = np.linalg.norm(d, axis=1)
    U = G / r
    acc = -G * d / r[:, None] ** 3
    return np.concatenate([U, acc[:, 0], acc[:, 1], acc[:, 2]])


def A_field(positions, obs):
    """Design matrix: mascon masses -> near-surface field [U;ax;ay;az] (4N, n_mascon)."""
    return np.column_stack([point_mass_field(p, obs) for p in positions])


def cyl_basis(cyl: Cylinder, obs, n_m, n_n):
    """
    Bessel–Fourier basis Φ evaluated at obs points, stacked as [U; a_rho; a_phi; a_z]
    then rotated to the SAME [U; ax; ay; az] layout as `point_mass_field`, so that
    Φ and the mascon field live in one common measurement space.
    """
    Rrot = cyl.rot()
    tp = (obs - cyl.center) @ Rrot
    rho = np.sqrt(tp[:, 0] ** 2 + tp[:, 1] ** 2)
    phi = np.arctan2(tp[:, 1], tp[:, 0])
    z = tp[:, 2]
    Ra = cyl.alpha * cyl.radius
    cphi, sphi = np.cos(phi), np.sin(phi)
    cols = []
    for m in range(n_m):
        cmp, smp = np.cos(m * phi), np.sin(m * phi)
        for n in range(1, n_n + 1):
            kmn = jn_zeros(m, n)[-1] / Ra
            ez = np.exp(-kmn * z)
            Jm = BesselJ(m, kmn * rho)
            dJm = BesselJp(m, kmn * rho)
            # potential and cylindrical gradient of each (cos, sin) mode
            for trig_c, trig_s in ((cmp, smp), (smp, -cmp)):  # A_mn then B_mn
                U = Jm * ez * trig_c
                a_rho = kmn * dJm * ez * trig_c
                a_phi = -(m / (rho + 1e-14)) * Jm * ez * trig_s
                a_z = -kmn * Jm * ez * trig_c
                # cyl accel -> local cartesian -> global cartesian
                ax = a_rho * cphi - a_phi * sphi
                ay = a_rho * sphi + a_phi * cphi
                az = a_z
                axyz = np.column_stack([ax, ay, az]) @ Rrot.T
                cols.append(np.concatenate([U, axyz[:, 0], axyz[:, 1], axyz[:, 2]]))
    return np.column_stack(cols)


# Truncated-SVD cutoff for every CH pseudo-inverse.  This matters more than it
# looks.  Over a patch the Bessel-Fourier columns are nearly linearly dependent
# (cond(Phi) ~ 1e16), and `np.linalg.pinv`'s DEFAULT cutoff is machine-epsilon
# based — max(M,N)*eps*s_max, ~9e-12 relative — so it keeps directions whose
# singular value is ~1e-13 of the largest.  Their coefficients come out as
# (projection)/sigma, i.e. enormous, and cancel again when multiplied back by
# Phi. Truncating is not cosmetic: the discarded directions carry noise, not signal,
# so counting them as information inflates what the CH block appears to know.
# `cylinder_mass_estimation_BENNU_TAG.fit_coefficients` has always passed an
# explicit cond for this reason.
CH_RCOND = 1e-4


def ch_pinv(Phi, rcond=None):
    """Truncated-SVD pseudo-inverse of the CH basis (see CH_RCOND)."""
    return np.linalg.pinv(Phi, rcond=CH_RCOND if rcond is None else rcond)


def ch_pinv_for(cyl, obs, ch_modes, rcond=None):
    """
    The truncated-SVD inverse of the CH basis for this cylinder and sampling.

    Callers want Phi only to invert it, so building it is an implementation
    detail and lives here.  It is returned rather than cached inside each
    consumer because Phi does not depend on the truth: `run_experiment` hands
    the SAME inverse to the coefficients, the position covariance and the
    spectra, and the SVD of a (4*n_obs x n_modes) matrix is not free.
    """
    return ch_pinv(cyl_basis(cyl, obs, *ch_modes), rcond)


def ch_projector(Phi, rcond=None):
    """
    P_CH : projection onto the span the CH model can actually represent, built
    from the SAME truncated pseudo-inverse the fits use, so the projector and
    the coefficient designs agree on what "representable" means.
    """
    return Phi @ ch_pinv(Phi, rcond=rcond)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2b — THE CONSTANT-DENSITY BULK (the polyhedron carries the mass)
# ═══════════════════════════════════════════════════════════════════════════
# The interior is NOT a cloud of mascons summing to the total mass.  It is the
# constant-density SHAPE MODEL scaled by β̃, plus N localized mascons carrying
# the departures from homogeneity,
#
#     U_T(r) = β̃ · U_CD(r) + Σ_j β_j · U_pt(r; p_j),      β_j = m_j / M* ,
#
# and the budget β̃ M* + Σ_j m_j = M* fixes the bulk scale outright, β̃ = 1 − Σβ.
# So β̃ is NOT an independent unknown, the total mass is M* by construction, and
# the old "Σ f = 1 known to σ_M" pseudo-observation is gone with it; what is left
# to estimate is β = {β_j}.  Substituting β̃ isolates the discrepancy between the
# measured field and the constant-density model,
#
#     ΔU(r) = U_T(r) − U_CD(r) = Σ_j β_j [ U_pt(r; p_j) − U_CD(r) ] ,
#
# which is what every design matrix below is: column j is a CONTRAST, the
# signature of taking β_j out of the homogeneous body and concentrating it at
# p_j.  β_j > 0 is a local excess, β_j < 0 a deficit; only the product
# m_j = Δρ_j v_j is identifiable, so the FRACTION, not the density contrast, is
# the estimated parameter.
#
# Both pieces of U_CD are computed once from the shape, to machine precision:
#   Stokes     — the integrand (r/R*)^n P̄_nm(sinφ){cos,sin}(mλ) is a solid
#                harmonic, a homogeneous POLYNOMIAL of degree n, so a tetrahedral
#                Gauss rule of degree ≥ n is exact (no model error).
#   near-field — polyhedral_gravity (Werner–Scheeres), which converges right down
#                to the surface where the SH series does not.


def _gauss01(n):
    """n-point Gauss–Legendre nodes/weights mapped to [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


@dataclass
class Bulk:
    """
    Constant-density polyhedron normalized to UNIT total mass (G = 1, M* = 1).

    `stokes()` and `field()` return the observables of the WHOLE bulk at unit
    mass; the model uses them scaled by β̃ = 1 − Σβ.  Both are cached, because
    the network experiments ask for the same cylinders over and over.
    """

    V: np.ndarray
    F: np.ndarray

    def __post_init__(self):
        self.V = np.asarray(self.V, float)
        self.F = np.asarray(self.F, int)
        a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        # signed volume of the tetrahedron (origin, a, b, c) — the sum is the
        # polyhedron volume for any closed, consistently wound mesh
        self._tet = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
        self.volume = float(self._tet.sum())
        self._sh, self._fd, self._ev = {}, {}, None

    # ── near-surface field ─────────────────────────────────────────────────
    def field(self, obs):
        """
        [U; ax; ay; az] of the unit-mass constant-density body at `obs` — the
        same stacking and sign convention (U > 0, a = +∇U) as `point_mass_field`
        and `cyl_basis`, so bulk, mascons and CH basis share one space.
        """
        obs = np.atleast_2d(np.asarray(obs, float))
        key = obs.tobytes()
        if key not in self._fd:
            if self._ev is None:
                if not _HAVE_PG:
                    raise RuntimeError(
                        "polyhedral_gravity is required for the constant-density "
                        "bulk field (pip install polyhedral-gravity)"
                    )
                self._ev = GravityEvaluable(
                    Polyhedron(
                        polyhedral_source=(self.V, self.F),
                        density=1.0 / self.volume,  # unit total mass
                        integrity_check=PolyhedronIntegrity.DISABLE,
                    )
                )
            res = self._ev(computation_points=obs, parallel=True)
            U = np.array([r[0] for r in res]) / G_SI  # SI G divided out → G = 1
            g = np.array([r[1] for r in res]) / G_SI
            self._fd[key] = np.concatenate([U, g[:, 0], g[:, 1], g[:, 2]])
        return self._fd[key]

    # ── Stokes coefficients ────────────────────────────────────────────────
    # The Stokes coefficients can be computed by a brute-force tetrahedral quadrature of the solid harmonics, but Werner (1997) gives a recurrence that is faster and more numerically stable.  Anyway, they are both exact!
    def stokes_werner(self, Lmin, Lmax, Rref, chunk=200_000):
        """
        Fully-normalized Stokes coefficients of the unit-mass constant-density
        polyhedron, following Werner (1997).

        The polyhedron is decomposed into signed tetrahedra
        (origin, v0, v1, v2). For each tetrahedron, the normalized solid-harmonic
        integrands are generated recursively as homogeneous trinomials in the
        local simplex coordinates, then integrated analytically:

            ∫_simplex X^i Y^j Z^k dX dY dZ
                = i! j! k! / (n + 3)!,
                i + j + k = n.

        The final coefficients are normalized by the total polyhedron volume,
        corresponding to a unit-mass constant-density body.

        Werner's recurrences are:

            c̄00 = 1
            s̄00 = 0

            [c̄11]   1/(Rref*sqrt(3)) [x]
            [s̄11] =                      [y]

            [c̄nn]                    2n-1
            [s̄nn] = -------------------------- [x  -y] [c̄n-1,n-1]
                        Rref*sqrt(2n(2n+1)) [y   x] [s̄n-1,n-1]

            [c̄n,n-1]   2n-1    z
            [s̄n,n-1] = -------- --- [c̄n-1,n-1]
                        sqrt(2n+1) Rref

        and, for m < n-1,

            [c̄nm] = A_nm (z/Rref) [c̄n-1,m]
                    - B_nm (r/Rref)^2 [c̄n-2,m]

            [s̄nm] = A_nm (z/Rref) [s̄n-1,m]
                    - B_nm (r/Rref)^2 [s̄n-2,m]

        with

            A_nm = (2n-1) sqrt(
                        (2n-1) /
                        ((2n+1)(n+m)(n-m))
                    )

            B_nm = sqrt(
                        (2n-3)(n+m-1)(n-m-1) /
                        ((2n+1)(n+m)(n-m))
                    ).

        The trinomial integration is the analytic Lien-Kajiya integral used by
        Werner (1997). Signed tetrahedral Jacobians preserve the orientation of
        the polyhedron, including concave bodies.
        """
        key = (Lmin, Lmax, round(float(Rref), 12))
        if key in self._sh:
            return self._sh[key]

        Rref = float(Rref)
        if Lmin < 0 or Lmax < Lmin:
            raise ValueError("require 0 <= Lmin <= Lmax")
        if not np.isfinite(Rref) or Rref <= 0.0:
            raise ValueError("Rref must be finite and positive")
        if chunk <= 0:
            raise ValueError("chunk must be positive")

        import math

        # ------------------------------------------------------------------
        # Homogeneous monomial lists.
        #
        # mons[n] contains all (i,j,k) with i+j+k=n.
        # ------------------------------------------------------------------
        mons = {
            n: [(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)]
            for n in range(Lmax + 1)
        }

        # ------------------------------------------------------------------
        # Index maps for multiplying a degree-n polynomial by X, Y, Z.
        # ------------------------------------------------------------------
        mul_xyz = {}
        for n in range(Lmax):
            dst = {ijk: q for q, ijk in enumerate(mons[n + 1])}

            ix = np.array(
                [dst[(i + 1, j, k)] for i, j, k in mons[n]],
                dtype=np.intp,
            )
            iy = np.array(
                [dst[(i, j + 1, k)] for i, j, k in mons[n]],
                dtype=np.intp,
            )
            iz = np.array(
                [dst[(i, j, k + 1)] for i, j, k in mons[n]],
                dtype=np.intp,
            )

            mul_xyz[n] = (ix, iy, iz)

        # ------------------------------------------------------------------
        # Index maps for multiplying by
        #
        #     r^2 = qXX X^2 + qYY Y^2 + qZZ Z^2
        #             + qXY XY + qXZ XZ + qYZ YZ.
        # ------------------------------------------------------------------
        qmons = (
            (2, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
        )

        mul_r2 = {}
        for n in range(max(0, Lmax - 1)):
            dst = {ijk: q for q, ijk in enumerate(mons[n + 2])}
            mul_r2[n] = [
                np.array(
                    [
                        dst[
                            (
                                i + dq[0],
                                j + dq[1],
                                k + dq[2],
                            )
                        ]
                        for i, j, k in mons[n]
                    ],
                    dtype=np.intp,
                )
                for dq in qmons
            ]

        # ------------------------------------------------------------------
        # Exact simplex moments:
        #
        #     ∫ X^i Y^j Z^k dX dY dZ = i!j!k!/(n+3)!
        # ------------------------------------------------------------------
        simplex_moments = {}
        for n in range(Lmax + 1):
            denom = float(math.factorial(n + 3))
            simplex_moments[n] = np.array(
                [
                    (math.factorial(i) * math.factorial(j) * math.factorial(k) / denom)
                    for i, j, k in mons[n]
                ],
                dtype=float,
            )

        # ------------------------------------------------------------------
        # Accumulated global coefficients.
        #
        # Assumed ordering:
        #   [C00, S00, C10, S10, C11, S11, C20, S20, ...]
        #
        # If sh_stokes_of_point uses a different packing, only this final
        # packing section needs to change.
        # ------------------------------------------------------------------
        accumulated = {}

        V = self.V
        F = self.F

        for s0 in range(0, len(F), chunk):
            sl = slice(s0, min(s0 + chunk, len(F)))

            # Tetrahedron vertices:
            #   origin, a, b, c
            a = V[F[sl, 0]]
            b = V[F[sl, 1]]
            c = V[F[sl, 2]]

            nt = len(a)

            # J maps local simplex coordinates (X,Y,Z) to physical (x,y,z):
            #
            # [x]   [x1 x2 x3] [X]
            # [y] = [y1 y2 y3] [Y]
            # [z]   [z1 z2 z3] [Z]
            #
            # Each row is therefore the coefficient vector of x,y,z.
            J = np.stack((a, b, c), axis=2)  # (nt, 3, 3)
            detJ = 6.0 * self._tet[sl]  # signed det(J)

            lin_x = J[:, 0, :]
            lin_y = J[:, 1, :]
            lin_z = J[:, 2, :]

            # r^2 = x^2 + y^2 + z^2 as a quadratic in X,Y,Z.
            q = np.empty((nt, 6), dtype=float)
            q[:, 0] = lin_x[:, 0] ** 2 + lin_y[:, 0] ** 2 + lin_z[:, 0] ** 2  # X^2
            q[:, 1] = lin_x[:, 1] ** 2 + lin_y[:, 1] ** 2 + lin_z[:, 1] ** 2  # Y^2
            q[:, 2] = lin_x[:, 2] ** 2 + lin_y[:, 2] ** 2 + lin_z[:, 2] ** 2  # Z^2
            q[:, 3] = 2.0 * (
                lin_x[:, 0] * lin_x[:, 1]
                + lin_y[:, 0] * lin_y[:, 1]
                + lin_z[:, 0] * lin_z[:, 1]
            )  # XY
            q[:, 4] = 2.0 * (
                lin_x[:, 0] * lin_x[:, 2]
                + lin_y[:, 0] * lin_y[:, 2]
                + lin_z[:, 0] * lin_z[:, 2]
            )  # XZ
            q[:, 5] = 2.0 * (
                lin_x[:, 1] * lin_x[:, 2]
                + lin_y[:, 1] * lin_y[:, 2]
                + lin_z[:, 1] * lin_z[:, 2]
            )  # YZ

            def mul_linear(poly, linear, degree):
                """Multiply degree-(n) polynomial by a linear form."""
                out = np.zeros((nt, len(mons[degree + 1])), dtype=float)
                ix, iy, iz = mul_xyz[degree]

                out[:, ix] += poly * linear[:, 0, None]
                out[:, iy] += poly * linear[:, 1, None]
                out[:, iz] += poly * linear[:, 2, None]
                return out

            def mul_r_squared(poly, degree):
                """Multiply a degree-n polynomial by r^2."""
                out = np.zeros((nt, len(mons[degree + 2])), dtype=float)

                for t, idx in enumerate(mul_r2[degree]):
                    out[:, idx] += poly * q[:, t, None]

                return out

            def integrate(poly, degree):
                """Exact integral of a degree-n trinomial over each tetrahedron."""
                return detJ * (poly @ simplex_moments[degree])

            # ==============================================================
            # Werner recursion
            #
            # prev2 = degree n-2
            # prev  = degree n-1
            # curr  = degree n
            # ==============================================================

            prev2 = {
                0: (
                    np.ones((nt, 1), dtype=float),
                    np.zeros((nt, 1), dtype=float),
                )
            }

            # Degree 0
            if Lmin <= 0:
                c0 = integrate(prev2[0][0], 0).sum()
                s0 = integrate(prev2[0][1], 0).sum()
                accumulated[(0, 0)] = (
                    accumulated.get((0, 0), (0.0, 0.0))[0] + c0,
                    accumulated.get((0, 0), (0.0, 0.0))[1] + s0,
                )

            if Lmax >= 1:
                # ----------------------------------------------------------
                # n = 1
                #
                # c10 = z / (sqrt(3) Rref)
                # s10 = 0
                #
                # c11 = x / (sqrt(3) Rref)
                # s11 = y / (sqrt(3) Rref)
                #
                # The c10 relation is the n=1 sub-diagonal case.
                # ----------------------------------------------------------
                c10 = mul_linear(
                    prev2[0][0],
                    lin_z / (Rref * np.sqrt(3.0)),
                    0,
                )
                s10 = np.zeros_like(c10)

                c11 = mul_linear(
                    prev2[0][0],
                    lin_x / (Rref * np.sqrt(3.0)),
                    0,
                )
                s11 = mul_linear(
                    prev2[0][0],
                    lin_y / (Rref * np.sqrt(3.0)),
                    0,
                )

                prev = {
                    0: (c10, s10),
                    1: (c11, s11),
                }

                if Lmin <= 1:
                    accumulated[(1, 0)] = (
                        accumulated.get((1, 0), (0.0, 0.0))[0]
                        + integrate(c10, 1).sum(),
                        accumulated.get((1, 0), (0.0, 0.0))[1]
                        + integrate(s10, 1).sum(),
                    )
                    accumulated[(1, 1)] = (
                        accumulated.get((1, 1), (0.0, 0.0))[0]
                        + integrate(c11, 1).sum(),
                        accumulated.get((1, 1), (0.0, 0.0))[1]
                        + integrate(s11, 1).sum(),
                    )

                # ----------------------------------------------------------
                # n >= 2
                # ----------------------------------------------------------
                for n in range(2, Lmax + 1):
                    curr = {}

                    cdiag_prev, sdiag_prev = prev[n - 1]

                    # ------------------------------------------------------
                    # Diagonal: m = n
                    #
                    # [c_nn]   (2n-1)/(R sqrt(2n(2n+1))) [ x -y ] [c]
                    # [s_nn]                                      [ y  x ] [s]
                    # ------------------------------------------------------
                    fdiag = (2.0 * n - 1.0) / (
                        Rref * np.sqrt(2.0 * n * (2.0 * n + 1.0))
                    )

                    cdiag = mul_linear(
                        cdiag_prev,
                        lin_x * fdiag,
                        n - 1,
                    ) - mul_linear(
                        sdiag_prev,
                        lin_y * fdiag,
                        n - 1,
                    )

                    sdiag = mul_linear(
                        sdiag_prev,
                        lin_x * fdiag,
                        n - 1,
                    ) + mul_linear(
                        cdiag_prev,
                        lin_y * fdiag,
                        n - 1,
                    )

                    curr[n] = (cdiag, sdiag)

                    # ------------------------------------------------------
                    # Sub-diagonal: m = n-1
                    # ------------------------------------------------------
                    fsub = (2.0 * n - 1.0) / (Rref * np.sqrt(2.0 * n + 1.0))

                    csub = mul_linear(
                        cdiag_prev,
                        lin_z * fsub,
                        n - 1,
                    )
                    ssub = mul_linear(
                        sdiag_prev,
                        lin_z * fsub,
                        n - 1,
                    )

                    curr[n - 1] = (csub, ssub)

                    # ------------------------------------------------------
                    # Vertical coefficients: m < n-1
                    # ------------------------------------------------------
                    for m in range(n - 1):
                        c_prev, s_prev = prev[m]
                        c_prev2, s_prev2 = prev2[m]

                        A_nm = (2.0 * n - 1.0) * np.sqrt(
                            (2.0 * n - 1.0) / ((2.0 * n + 1.0) * (n + m) * (n - m))
                        )

                        B_nm = np.sqrt(
                            ((2.0 * n - 3.0) * (n + m - 1.0) * (n - m - 1.0))
                            / ((2.0 * n + 1.0) * (n + m) * (n - m))
                        )

                        z_term_c = mul_linear(
                            c_prev,
                            lin_z * (A_nm / Rref),
                            n - 1,
                        )
                        z_term_s = mul_linear(
                            s_prev,
                            lin_z * (A_nm / Rref),
                            n - 1,
                        )

                        r2_term_c = mul_r_squared(c_prev2, n - 2)
                        r2_term_s = mul_r_squared(s_prev2, n - 2)

                        ccur = z_term_c - (B_nm / Rref**2) * r2_term_c
                        scur = z_term_s - (B_nm / Rref**2) * r2_term_s

                        curr[m] = (ccur, scur)

                    # ------------------------------------------------------
                    # Integrate this degree immediately; no need to retain
                    # older levels beyond the two required by the recurrence.
                    # ------------------------------------------------------
                    if Lmin <= n <= Lmax:
                        for m, (c_poly, s_poly) in curr.items():
                            c_val = integrate(c_poly, n).sum()
                            s_val = integrate(s_poly, n).sum()

                            old_c, old_s = accumulated.get(
                                (n, m),
                                (0.0, 0.0),
                            )

                            accumulated[(n, m)] = (
                                old_c + c_val,
                                old_s + s_val,
                            )

                    prev2, prev = prev, curr

        # ------------------------------------------------------------------
        # Normalize by total volume.
        # Werner's equation has rho/M = 1/V for constant density.
        # ------------------------------------------------------------------
        if self.volume == 0.0:
            raise ValueError("polyhedron volume is zero")

        inv_volume = 1.0 / self.volume

        # ------------------------------------------------------------------
        # Pack in the assumed sh_stokes_of_point ordering.
        # ------------------------------------------------------------------
        result = []
        for n in range(Lmin, Lmax + 1):
            for m in range(n + 1):
                c_val, s_val = accumulated[(n, m)]
                result.extend(
                    (
                        c_val * inv_volume,
                        s_val * inv_volume,
                    )
                )

        result = np.asarray(result, dtype=float)
        self._sh[key] = result
        return result

    def stokes(self, Lmin, Lmax, Rref, chunk=200_000):
        """
        Fully-normalized Stokes coefficients of the unit-mass constant-density
        polyhedron, in the ordering of `sh_stokes_of_point`:

            C̄_nm = (1/((2n+1) Vol)) ∫_body (r/R*)^n P̄_nm(sinφ) cos mλ dV

        Each tetrahedron (origin, v0, v1, v2) uses a Duffy-mapped tensor Gauss
        rule; the integrand is a polynomial of degree ≤ Lmax and the map
        contributes (1−u)²(1−v), so `ng` points per direction integrate degree
        2·ng−1 ≥ Lmax+2 EXACTLY.  Signed volumes keep the decomposition valid
        for a concave body whatever the origin.
        """
        key = (Lmin, Lmax, round(float(Rref), 12))
        if key in self._sh:
            return self._sh[key]
        ng = max(3, (Lmax + 4) // 2)
        u, wu = _gauss01(ng)
        UU, VV, WW = np.meshgrid(u, u, u, indexing="ij")
        Wq = (
            wu[:, None, None]
            * wu[None, :, None]
            * wu[None, None, :]
            * (1.0 - UU) ** 2
            * (1.0 - VV)
        ).ravel()
        l1 = UU.ravel()
        l2 = (VV * (1.0 - UU)).ravel()
        l3 = (WW * (1.0 - UU) * (1.0 - VV)).ravel()
        a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        acc, step = None, max(1, chunk // len(Wq))
        for s0 in range(0, len(self.F), step):
            sl = slice(s0, s0 + step)
            pts = (
                a[sl][:, None, :] * l1[None, :, None]
                + b[sl][:, None, :] * l2[None, :, None]
                + c[sl][:, None, :] * l3[None, :, None]
            )  # (n_tet, n_quad, 3)
            B = sh_stokes_basis(pts.reshape(-1, 3), Lmin, Lmax, Rref)
            w = (6.0 * self._tet[sl][:, None] * Wq[None, :]).ravel()
            acc = w @ B if acc is None else acc + w @ B
        self._sh[key] = acc / self.volume
        return self._sh[key]


def bulk_fraction(beta):
    """β̃ = 1 − Σβ : the fraction of M* left in the homogeneous polyhedron."""
    return 1.0 - float(np.sum(beta))


def A_stokes_contrast(positions, bulk, Lmin, Lmax, Rref):
    """
    Design matrix of the SH DISCREPANCY, ΔCS = A β.  Column j is
    [ Stokes of a unit mass at p_j ] − [ Stokes of the same mass spread through
    the body ] — the signature of a density CONTRAST, not of mass in vacuum.
    """
    return (
        A_stokes(positions, Lmin, Lmax, Rref) - bulk.stokes(Lmin, Lmax, Rref)[:, None]
    )


def A_field_contrast(positions, bulk, obs):
    """Same contrast, in the near-surface field [U; ax; ay; az]:  Δfield = A β."""
    return A_field(positions, obs) - bulk.field(obs)[:, None]


def sh_coefficients_total(beta, positions, bulk, Lmin, Lmax, Rref):
    """
    Stokes coefficients of the FULL truth  β̃·CD + Σ β_j pt_j.

    "Full" is the point: this is the MEASURED quantity, bulk included, and it is
    what `od_sigma` turns into a per-coefficient sigma — an instrument's
    precision follows the size of what it measures.  Its counterpart
    `A_sh_contrast` carries only the discrepancy, which is what gets FITTED.
    """
    return (
        bulk.stokes(Lmin, Lmax, Rref)
        + A_stokes_contrast(positions, bulk, Lmin, Lmax, Rref) @ beta
    )


def field_samples_total(beta, positions, bulk, obs):
    """
    Near-surface field samples [U; ax; ay; az] of the FULL truth
    β̃·CD + Σ β_j pt_j — samples, not coefficients; see `ch_coefficients_total`.
    """
    return bulk.field(obs) + A_field_contrast(positions, bulk, obs) @ beta


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — OBSERVATION WEIGHTS, INFORMATION / COVARIANCE
# ═══════════════════════════════════════════════════════════════════════════
# TWO least-squares problems live in this code and they are weighted DIFFERENTLY
# on purpose:
#
#   1. Building the CH coefficients from sampled field values (Φ c = field).
#      UNWEIGHTED ordinary least squares, c = Φ⁺ field: the samples are one
#      instrument's synthetic product over one small patch, with no per-sample
#      error model to impose — imposing one would just be a knob.
#
#   2. Estimating the mass fractions β from the COEFFICIENT discrepancies.
#      WEIGHTED, with a per-coefficient σ imitating an OD solution: each
#      coefficient delivered to a fixed FRACTION of its own magnitude, above an
#      absolute noise floor below which nothing is resolved.  This is the V that
#      whitens the cost — one σ per COEFFICIENT, not per block, so a strong
#      low-degree term and a weak high-degree one do not share a weight.


# NOTE: Use the OD uncertainty of the absolute coefficients for
# the residual coefficients because subtracting the noise-free
# constant-density model does not change their covariance.
def od_sigma2(cs, eps, floor_frac=0.1):
    """
    OD-like 1σ for a measured coefficient vector `cs`:

        σ_i = eps · max( |cs_i| , floor_frac · RMS(cs) )

    First branch: "every coefficient is known to eps of itself", the relative
    precision an OD solution quotes.  Second: the absolute noise floor — OD
    cannot resolve a coefficient far below the scale of the field it fits, and
    without it the entries identically zero by construction (the S̄_n0 sine
    terms) would carry infinite weight.  `cs` must be the FULL measured vector
    (bulk + anomalies): that is what the instrument delivers before the
    constant-density model is removed.

    Paper caveat: constant relative precision across degrees is optimistic at
    high degree, where a real OD solution degrades faster than the signal does.
    A degree-dependent rule slots in here.
    """
    cs = np.asarray(cs, float)
    floor = floor_frac * float(np.sqrt(np.mean(cs**2)))
    return eps * np.maximum(np.abs(cs), floor)


def od_sigma(cs, eps, floor_frac=0.1, alpha=0.10):
    """
    Degree-dependent OD-like 1σ uncertainties.

    Keeps the original interface, but lets uncertainty grow with spherical-
    harmonic degree to mimic the loss of sensitivity to shorter-wavelength
    gravity structure at high degree.
    """
    cs = np.asarray(cs, float)

    # Infer degree from packing:
    # [C_n0,S_n0,C_n1,S_n1,...], starting at n=2.
    degrees = []
    n = 2
    while len(degrees) < len(cs):
        degrees.extend([n] * (2 * (n + 1)))
        n += 1
    degrees = np.asarray(degrees[: len(cs)])

    # Global absolute noise floor.
    scale = float(np.sqrt(np.mean(cs**2)))
    floor = floor_frac * scale

    # Baseline coefficient uncertainty.
    sigma = eps * np.maximum(np.abs(cs), floor)

    # OD sensitivity degrades with degree.
    sigma *= np.exp(alpha * (degrees - degrees.min()))

    return sigma


def _col(sig):
    """σ as a column so `A / _col(σ)` whitens rows for scalar OR per-row σ."""
    s = np.asarray(sig, float)
    return s if s.ndim == 0 else s[:, None]


def fisher_mass_fractions(blocks, prior_sigma=1.0):
    """
    Fisher information for the anomaly mass-fraction vector β, built from the
    SAME (A, σ) coefficient blocks the Monte-Carlo fit uses:

        F = Σ_blocks  A_wᵀ A_w ,     A_w = A / σ   (row-wise; σ may be a vector)

    so the analytic covariance and the actual fits describe one and the same
    estimator.  Every design matrix must be a contrast one
    (`A_stokes_contrast`, `A_ch_contrast`).  There is no total-mass block: the
    budget β̃ = 1 − Σβ is structural, not a pseudo-observation.  A weak Gaussian
    prior keeps everything finite.
    """
    n = blocks[0][0].shape[1]
    Fi = np.eye(n) / prior_sigma**2
    for A, sig in blocks:
        Aw = A / _col(sig)
        Fi = Fi + Aw.T @ Aw
    return Fi


def posterior_sigma(C):
    """Per-parameter 1σ from a covariance: sqrt of its diagonal."""
    return np.sqrt(np.diag(C))


# `fisher_mass_fractions` seeds its information with I/prior_sigma², so an
# under-determined case still inverts.  This names that seed so the L_SH sweep
# can ask how far a posterior has moved off it: R = sigma / PRIOR_SIGMA.
PRIOR_SIGMA = 1.0

# The position twin of PRIOR_SIGMA — a weak Gaussian seed on every coordinate,
# about the body's half-length — and the R at or above which a sigma counts as
# prior-dominated.  Same values as pt2, so the two sweeps flag on one rule.
# Only the sweep seeds the position covariance (see `position_covariance`).
POS_PRIOR_SIGMA = 1.0  # LU
PRIOR_RATIO_THRESHOLD = 0.95  # flag less than or equal to 5% sigma reduction


def posterior_rms(C):
    """
    One isotropic 1σ from a covariance: sqrt(trace(C)/n).

    The scalar summary for a VECTOR parameter — the position, where three
    components share one physical meaning and a single number is what gets
    quoted.  `posterior_sigma` is the counterpart for the mass fractions, which
    are separate physical quantities and are reported one by one.
    """
    return float(np.sqrt(np.trace(C) / len(C)))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — THE ANOMALIES (contrasts on top of the constant-density bulk)
# ═══════════════════════════════════════════════════════════════════════════

# name, position [LU], truth mass fraction β_j = m_j / M*.  The bulk of the mass
# is NOT here — it is in the constant-density polyhedron, which keeps
# β̃ = 1 − Σβ = 0.96 of M*.  These are the departures from homogeneity:
# β_j > 0 an over-dense concentration, β_j < 0 a mass deficit.  Index 0 is the
# shallow anomaly (the CH target); 1 and 2 sit deep in the two lobes.
MASCONS = [
    ("Near-surface Anomaly", np.array([0.00, 0.00, 0.22]), 0.03),
    ("+x Lobe Excess", np.array([0.42, 0.00, 0.00]), 0.05),
    ("-x Lobe Deficit", np.array([-0.45, 0.00, 0.00]), -0.04),
]

# The reach maps (fig 3b).  REACH_BETA is the test anomaly's mass fraction for
# the POSITION map — the near-surface target's, so the map under the target
# reproduces its `position_covariance` 1σ (to <1%, grid interpolation), and
# inside pt2's |β| range (0.015-0.05).  The levels are
# the maps' contours and the reach tables' thresholds, so figure and table agree.
REACH_BETA = abs(MASCONS[0][2])
REACH_LEVELS_MASS = (1e-3, 3e-3, 1e-2)  # σ_β  [-]
REACH_LEVELS_POS = (1e-3, 1e-2, 1e-1)  # position 1σ  [LU]


# ── vocabulary ──────────────────────────────────────────────────────────────
# Three tiers of mass, and every name below says which one it means:
#   MASCONS  the localized anomalies.  `beta_true` / `betas` / `beta` are their
#            fractions beta_j = m_j/M*, `P` their positions.  Signed: + is an
#            excess, - a deficit.
#   BULK     the constant-density polyhedron.  The object is `bulk`; the share
#            of the mass left in it is `beta_bulk` = 1 - sum(beta).
#   TOTAL    bulk + mascons, i.e. what an instrument actually measures.  The
#            `*_total` functions return it; `A_*_contrast` return the mascon
#            part alone (the discrepancy against the bulk), which is what is
#            FITTED.  sigma is built from the TOTAL, because precision follows
#            the size of what is measured, not of what is being solved for.


def mascon_arrays():
    """names, positions, truth mass FRACTIONS β (not ratios summing to one)."""
    names = [m[0] for m in MASCONS]
    P = np.array([m[1] for m in MASCONS])
    beta = np.array([m[2] for m in MASCONS])
    return names, P, beta


# Largest density contrast an EXCESS can plausibly carry, as a fraction of the
# bulk density.  Against Eros' 2.67 g/cc it is bracketed by what the material
# can be: a solid ordinary-chondrite block (3.4 g/cc, pores squeezed out) gives
# only Δρ/ρ ≈ 0.27, while solid FeNi (7.8 g/cc) gives Δρ/ρ ≈ 1.92 — so ~2 is the
# hard ceiling, not a free choice.  Deficits have no such ceiling: their floor is
# total vacuum, Δρ/ρ = −1 exactly, which is why they are handled separately.
# NOTE: EXCESS_CONTRAST is a free parameter of the admissibility tests,
# here I'm allowing 3 times the bulk density, which is arbitrary.

EXCESS_CONTRAST = 2.00


def admissible_radius(beta, bulk_frac, volume, contrast=EXCESS_CONTRAST):
    """
    Smallest equivalent sphere [LU] that makes a point anomaly β physical.

    Nothing constrains the SIGN of β: it is a CONTRAST against the constant-
    density bulk, so a void or porous patch is negative BY CONSTRUCTION — a mass
    deficit, not an unphysical mass.  What must stay physical is the TOTAL local
    density.  Smearing the anomaly over a sphere of volume V_a,

        ρ = β̃ M*/V_body  +  β M*/V_a  =  ρ_bulk (1 + (β/β̃)(V_body/V_a)),

    so ρ ≥ 0 (a deficit can at worst empty the region) requires

        V_a ≥ |β| V_body / β̃                      for β < 0,

    while an excess is capped not by ρ ≥ 0 but by the densest inclusion nature
    supplies, Δρ ≤ contrast·ρ_bulk:

        V_a ≥  β  V_body / (contrast · β̃)         for β > 0.

    By Newton's theorem a uniform sphere's exterior field is EXACTLY that of the
    point mass at its centre, so the swap costs the model nothing — provided
    every observation point stays outside the sphere.
    """
    scale = 1.0 if beta < 0 else contrast
    V_a = abs(beta) * volume / (bulk_frac * scale)
    return (3.0 * V_a / (4.0 * math.pi)) ** (1.0 / 3.0)


def admissible_beta(radius, beta_sign, bulk_frac, volume, contrast=EXCESS_CONTRAST):
    """Inverse of `admissible_radius`: the largest |β| a uniform sphere of the
    given radius can carry without going unphysical.  This is what a SITE can
    hold — put `radius` = distance to the nearest surface or datum."""
    scale = 1.0 if beta_sign < 0 else contrast
    return scale * bulk_frac * (4.0 / 3.0 * math.pi * radius**3) / volume


def admissibility(P, betas, bulk_frac, volume, obs, tm):
    """
    Per-anomaly: minimum admissible radius, distance to the shape surface, and
    distance to the nearest field point.  a_min < both ⇒ the point-mass truth is
    an exact stand-in for a physical, buried, uniform density contrast.
    """
    a = np.array([admissible_radius(b, bulk_frac, volume) for b in betas])
    if tm is not None:
        d_surf = np.abs(trimesh.proximity.signed_distance(tm, P))
    else:
        d_surf = np.full(len(P), np.nan)
    d_obs = np.min(np.linalg.norm(obs[None, :, :] - P[:, None, :], axis=2), axis=1)
    # largest |β| that stays buried AND clear of every field point
    r_ok = np.minimum(np.nan_to_num(d_surf, nan=np.inf), d_obs)
    b_max = np.array(
        [
            math.copysign(admissible_beta(r, b, bulk_frac, volume), b)
            for r, b in zip(r_ok, betas)
        ]
    )
    return a, d_surf, d_obs, b_max


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════


def mass_fraction_covariance(blocks, prior_sigma=1.0):
    """
    Posterior covariance of the MASS FRACTIONS beta, C = (A^T W A)^-1.

    The twin of `position_covariance`, and the reason both exist as named
    functions: the mass problem is LINEAR, so this C is exact — no expansion
    point, no derivatives — while the position problem is nonlinear and its
    covariance is only a linearization about the truth.  Same (A, sigma) blocks
    the fits consume, so the predicted sigma and the realized scatter describe
    one estimator.
    """
    return np.linalg.inv(fisher_mass_fractions(blocks, prior_sigma))


def position_covariance(
    idx,
    P,
    obs,
    cyl,
    ch_modes,
    sig_sh,
    sig_ch,
    Lmax,
    Rref,
    pinvPhi,
    prior_sigma=None,
):
    """
    Linearized position covariance of anomaly `idx`, with its truth mass
    fraction, in each of the three observation models.  Position partials by
    central differences,
    in the same COEFFICIENT space and with the same per-coefficient weights as
    the fits: the CH partial is Φ⁺ ∂(field)/∂p, the position sensitivity of the
    coefficients the unweighted Phi-to-field fit would return.  (Other masses/positions
    fixed — the near-surface anomaly is the target.)  The bulk term β̃·U_CD does
    not move with p and drops out of ∂y/∂p entirely; only the σ's carry its
    (large) presence, through the relative-precision rule.
    """
    _, _, f = mascon_arrays()
    p0, mass = P[idx].copy(), f[idx]
    eps = 1e-4

    def jac(func, dim):
        J = np.empty((dim, 3))
        for k in range(3):
            d = np.zeros(3)
            d[k] = eps
            J[:, k] = mass * (func(p0 + d) - func(p0 - d)) / (2 * eps)
        return J

    Jsh = jac(
        lambda p: sh_stokes_of_point(p, 2, Lmax, Rref),
        A_stokes(P, 2, Lmax, Rref).shape[0],
    )
    Jsh_w = Jsh / _col(sig_sh)
    Fi_sh = Jsh_w.T @ Jsh_w
    Jch = pinvPhi @ jac(lambda p: point_mass_field(p, obs), 4 * len(obs))
    Jch_w = Jch / _col(sig_ch)
    Fi_ch = Jch_w.T @ Jch_w

    # covariances, not summaries: the caller reduces them with `posterior_rms`,
    # exactly as it reduces `mass_fraction_covariance` with `posterior_sigma`.
    # No prior by default — unlike the mass Fisher, a 3x3 built from a full-rank
    # Jacobian inverts on its own, and CH-only is full rank for any target the
    # patch actually overlooks.  The L_SH sweep passes `prior_sigma` (a seed
    # I/prior_sigma² on the three coordinates) so that its posterior/prior
    # ratio R is defined for every anomaly, overlooked or not.
    F0 = np.zeros((3, 3)) if prior_sigma is None else np.eye(3) / prior_sigma**2
    return {
        SH_ONLY: np.linalg.inv(Fi_sh + F0),
        CH_ONLY: np.linalg.inv(Fi_ch + F0),
        SH_CH: np.linalg.inv(Fi_sh + Fi_ch + F0),
    }


def reach_map(
    bulk,
    obs_list,
    pinv_list,
    sig_ch_list,
    sig_sh,
    Lmax,
    Rref,
    V,
    F,
    tm,
    n=61,
    beta_test=REACH_BETA,
):
    """
    1-sigma on the mass fraction — and on the POSITION — of a SINGLE test
    anomaly placed at each point of an x-z slice through the body, in each
    observation model.

    The tables quote three anomalies at three fixed places, which is enough to
    notice that the patch is local and not enough to show its SHAPE.  This shows
    the shape, and it is the panel that makes CH-only worth running: SH is a
    broad smooth field, CH is a sharp peak under the cylinder, and the two facts
    together are the argument for combining them.

    MASS: one free parameter, so the Fisher is a scalar and the sigma is closed
    form:

        sigma(p) = 1 / sqrt( I(p) + 1/prior^2 ),   I(p) = sum_i (A_i(p)/sigma_i)^2

    and because information ADDS, the three cases need only two maps —
    I_sh, I_ch, and their sum.  A(p) is the same CONTRAST column the fits use,
    a point mass at p minus the same mass spread through the body.  The weights
    are frozen at the nominal truth's sigma so the map varies with GEOMETRY
    alone; letting sigma follow each hypothetical truth would fold the anomaly's
    own signal strength back in and blur the thing being drawn.

    POSITION: the same construction one dimension up.  With the test mass held
    at `beta_test` — known, as `position_covariance` holds it — the Fisher on
    the three coordinates is  F(p) = sum_i J_i^T J_i / sigma_i^2  with
    J = beta_test dA(p)/dp, by the same central differences, and it adds across
    data sets exactly as the scalar does.  It is reduced with `posterior_rms`,
    the isotropic 1-sigma of `rms_pos`, so with beta_test at the target's value
    the map under the target reproduces `rms_pos` (checked: within 1% in all
    three cases, the residual being grid interpolation).  sigma scales as
    1/beta_test: `beta_test` sets the map's
    scale, not its shape.  A weak seed, 1/R^2 with R the body's circumscribing
    radius, says only "the anomaly is inside the body": without it CH-only is
    singular far from the patch, and wherever the data inform the position at
    all it changes nothing.

    `obs_list`/`pinv_list`/`sig_ch_list` are LISTS so one cylinder and a whole
    network are the same call — pt2 passes its network here unchanged.
    """
    xs = np.linspace(V[:, 0].min(), V[:, 0].max(), n)
    zs = np.linspace(V[:, 2].min(), V[:, 2].max(), n)
    XX, ZZ = np.meshgrid(xs, zs, indexing="xy")
    pts = np.column_stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()])
    inside = inside_body(tm, V, F, pts)
    ins = np.flatnonzero(inside)
    I_sh = np.full(XX.size, np.nan)
    I_ch = np.full(XX.size, np.nan)
    Fp_sh = np.zeros((XX.size, 3, 3))
    Fp_ch = np.zeros((XX.size, 3, 3))
    cs_bulk = bulk.stokes(2, Lmax, Rref)
    f_bulk = [bulk.field(o) for o in obs_list]

    def grad(func, p, h=1e-4):
        """beta_test * d(func)/dp, central differences as `position_covariance`."""
        cols = []
        for k in range(3):
            d = np.zeros(3)
            d[k] = h
            cols.append(func(p + d) - func(p - d))
        return beta_test * np.column_stack(cols) / (2.0 * h)

    for i in ins:
        p = pts[i]
        a = (sh_stokes_of_point(p, 2, Lmax, Rref) - cs_bulk) / sig_sh
        I_sh[i] = a @ a
        J = grad(lambda q: sh_stokes_of_point(q, 2, Lmax, Rref), p) / _col(sig_sh)
        Fp_sh[i] = J.T @ J
        tot = 0.0
        for o, pinv, s_ch, fb in zip(obs_list, pinv_list, sig_ch_list, f_bulk):
            c = (pinv @ (point_mass_field(p, o) - fb)) / s_ch
            tot += float(c @ c)
            J = (pinv @ grad(lambda q: point_mass_field(q, o), p)) / _col(s_ch)
            Fp_ch[i] += J.T @ J
        I_ch[i] = tot
    inv_prior2 = 1.0 / PRIOR_SIGMA**2
    info = {SH_ONLY: I_sh, CH_ONLY: I_ch, SH_CH: I_sh + I_ch}
    R_pos = float(np.linalg.norm(V, axis=1).max())

    def pos_sigma(Fp):
        out = np.full(XX.size, np.nan)
        C = np.linalg.inv(Fp[ins] + np.eye(3) / R_pos**2)
        out[ins] = np.sqrt(np.einsum("kii->k", C) / 3.0)  # posterior_rms, stacked
        return out.reshape(XX.shape)

    return dict(
        x=xs,
        z=zs,
        sigma={
            k: (1.0 / np.sqrt(v + inv_prior2)).reshape(XX.shape)
            for k, v in info.items()
        },
        sigma_pos={
            SH_ONLY: pos_sigma(Fp_sh),
            CH_ONLY: pos_sigma(Fp_ch),
            SH_CH: pos_sigma(Fp_sh + Fp_ch),
        },
        beta_test=float(beta_test),
        pos_prior=R_pos,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5b — ACTUAL LEAST-SQUARES FIT (Monte-Carlo, like the reference code)
# ═══════════════════════════════════════════════════════════════════════════


def ch_coefficients_total(beta, positions, bulk, obs, pinv):
    """
    CH coefficients of the FULL truth: `field_samples_total` projected onto Φ.

    `pinv` is passed in rather than rebuilt because Phi and its truncated-SVD
    inverse do not depend on the truth, and a caller looping over interiors
    would otherwise redo an (n_pts x n_modes) SVD every iteration.
    """
    return pinv @ field_samples_total(beta, positions, bulk, obs)


def A_ch_contrast(positions, bulk, obs, cyl, ch_modes):
    """
    Same contrast, in CYLINDRICAL-HARMONIC coefficients:  Δc = A β.

    Third member of the A_*_contrast family, and the one that is FITTED rather
    than written down: `A_stokes_contrast` and `A_field_contrast` evaluate their
    bases in closed form, while this one has to project a sampled field onto Φ.
    Per anomaly, evaluate the near-surface field of the CONTRAST (unit mass at
    p_j minus the same mass spread through the body) and fit the Bessel–Fourier
    basis Φ by ORDINARY (unweighted) least squares, as the reference script fits
    CH coefficients to a sampled field: A_ch = Φ⁺ A_field_contrast.  Weighting
    deliberately enters one level up, on the COEFFICIENTS (`od_sigma`), not on
    the field samples producing them.  Fitting the contrast is the near-surface
    form of ΔU — the constant-density field is known from the shape and
    subtracted before the anomalies are estimated.
    """
    # (n_ch, n_anom)
    return ch_pinv_for(cyl, obs, ch_modes) @ A_field_contrast(positions, bulk, obs)


def ls_fit_once(blocks, m_true, rng):
    """
    One weighted (whitened) linear least-squares fit of the mass fractions from
    noisy coefficient observables:
        minimize Σ_blocks || (A β − y)/σ ||² ,   y = A β_true + N(0, σ).
    `blocks` is a list of (A, σ), where σ is the PER-COEFFICIENT vector from
    `od_sigma` (a scalar still works and means an isotropic block).  Returns the
    recovered mass-fraction vector.
    """
    As, ys = [], []
    for A, sig in blocks:
        y = A @ m_true + rng.normal(0.0, sig, size=A.shape[0])
        As.append(A / _col(sig))
        ys.append(y / sig)
    Aw, yw = np.vstack(As), np.concatenate(ys)
    m_hat, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    return m_hat


def monte_carlo_fit(blocks, m_true, n_mc=2000, seed=7):
    """Monte-Carlo over noise: returns recovered masses (n_mc, n_mascon)."""
    rng = np.random.default_rng(seed)
    return np.array([ls_fit_once(blocks, m_true, rng) for _ in range(n_mc)])


def detection_sweep(A_sh, sig_sh, A_ch, sig_ch, f_base, mu_grid, n_mc=1000, seed=11):
    """
    Smallest detectable anomaly with vs without the cylinder.

    Sweeps the truth mass fraction β_0 of the shallow anomaly over `mu_grid` (deep anomalies held at
    nominal; β̃ = 1 − Σβ absorbs the change, so the body keeps unit mass) and at
    each value fits `n_mc` noisy realizations in each observation model.

    Three facts govern the figure this feeds:

    * THE SCATTER DOES NOT MOVE ALONG THE SWEEP — exactly, not approximately.
      `sig_sh`/`sig_ch` are fixed (built once from the nominal truth), and a
      linear model with fixed weights has covariance (A^T W A)^-1, which
      contains no truth.  Over the 16-point grid: 3.211e-3 everywhere for SH,
      1.652e-4 for SH+CH, varying by 1.00x.
    * THAT SIGMA IS THE ANALYTIC ONE: MC 3.211e-3 vs (A^T W A)^-1 3.169e-3 (SH),
      1.652e-4 vs 1.737e-4 (SH+CH).  The floor needs no fitting at all.
    * SO THE RMS ERROR IS REPORTED, NOT THE MEAN.  The estimator is unbiased —
      E[β̂_0] = β_0 for any β_0 — so the mean tracks the truth all the way down
      and says nothing about detectability, plateauing only at sigma/sqrt(n_mc),
      an MC artifact.  The RMS error is flat at sigma, so the RELATIVE error
      sigma/β_0 degrades as the anomaly shrinks: that is the performance curve.

    Returns `rms` (MC) beside `sd` (analytic) per case, measurement vs theory.
    """
    blocks = case_blocks(A_sh, sig_sh, A_ch, sig_ch)
    sd = {k: math.sqrt(mass_fraction_covariance(blocks[k])[0, 0]) for k in CASES}
    rms = {k: [] for k in CASES}
    mu1 = {k: [] for k in CASES}
    for j, mu in enumerate(mu_grid):
        m_true = f_base.copy()
        m_true[0] = mu
        for k in CASES:
            # a DIFFERENT seed per grid point: with one shared seed every point
            # reuses the same draws, so the RMS/sigma check would be one test
            # drawn 16 times rather than 16 independent ones
            v = monte_carlo_fit(blocks[k], m_true, n_mc, seed + j)[:, 0]
            rms[k].append(math.sqrt(np.mean((v - mu) ** 2)))
            mu1[k].append(v[0])  # one realization, kept for reference
    out = dict(
        rms={k: np.asarray(v) for k, v in rms.items()},
        mu1={k: np.asarray(v) for k, v in mu1.items()},
        sd=sd,  # analytic, exact, truth-independent
        n_mc=n_mc,
        mu_grid=np.asarray(mu_grid),
        # 3σ detection threshold: the smallest anomaly mass fraction whose
        # recovery stands 3 sigma clear of the noise.
        thr={k: 3.0 * sd[k] for k in CASES},
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5c — EXPERIMENT 2: fit anomaly POSITION with masses FIXED (MC)
# ═══════════════════════════════════════════════════════════════════════════
# (Experiment 1 — masses free, positions fixed — is the linear fit above.)


def _pos_forward(posj, j, P, masses, bulk, Lmax, Rref, obs, pinvPhi, case):
    """
    Forward observables when ONLY anomaly `j`'s position posj=(x,y,z) is
    unknown; all mass fractions and the other positions are known.  The model is
    the full field β̃·CD + Σ β_k pt_k — the bulk is an additive constant here
    (β̃ is fixed with the masses), present so the forward model is the one the
    paper writes down.

    `case` selects which blocks exist.  It replaced a `use_ch` flag, which could
    say "SH, and also CH" but had no way to say "CH, and NOT SH".

    `j` and the full `P` replaced a (pos0, lobe_pos) pair that could only ever
    free the SHALLOW anomaly — fig 3 now shows the covariance of every anomaly,
    which needs each of them free in turn.
    """
    positions = np.asarray(P, float).copy()
    positions[j] = posj
    blocks = []
    if case in (SH_ONLY, SH_CH):
        # ONE batched Stokes evaluation for all three anomalies, not one call each.
        S = sh_stokes_basis(np.asarray(positions, float), 2, Lmax, Rref)
        y_sh = bulk_fraction(masses) * bulk.stokes(2, Lmax, Rref)
        for mj, Sj in zip(masses, S):
            y_sh = y_sh + mj * Sj
        blocks.append(y_sh)
    if case in (CH_ONLY, SH_CH):
        field = bulk_fraction(masses) * bulk.field(obs)
        for mj, pj in zip(masses, positions):
            field = field + mj * point_mass_field(pj, obs)
        blocks.append(pinvPhi @ field)
    return blocks


def _pos_sigmas(case, s_sh, s_ch):
    """The sigma blocks matching `_pos_forward`'s output, in the same order."""
    return {SH_ONLY: [s_sh], CH_ONLY: [s_ch], SH_CH: [s_sh, s_ch]}[case]


def _pos_residual(
    posj,
    j,
    P,
    data_blocks,
    sig_blocks,
    masses,
    bulk,
    Lmax,
    Rref,
    obs,
    pinvPhi,
    case,
):
    model = _pos_forward(posj, j, P, masses, bulk, Lmax, Rref, obs, pinvPhi, case)
    return np.concatenate(
        [(mo - da) / s for mo, da, s in zip(model, data_blocks, sig_blocks)]
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5d — MONTE-CARLO OVER THE TRUTH, NOT JUST OVER THE NOISE
# ═══════════════════════════════════════════════════════════════════════════
# The experiments above fix one truth interior and resample the measurement
# noise, which answers "how precise is this estimate?".  It cannot answer "does
# the conclusion depend on the particular truth I hard-coded?".  These two
# routines redraw the TRUTH each iteration — the masses for the mass experiment,
# the anomaly position for the position experiment — and run the whole estimate
# inside, so the reported spread is over interiors, not over noise draws.


def draw_truth_masses(n, rng, mag=(0.01, 0.06)):
    """
    n random truth mass-fraction vectors: |β_j| uniform in `mag`, random signs
    (compaction or porosity).  Positions are untouched, so the design matrices
    are unchanged — anything that moves does so through σ, which is set from the
    full measured field and therefore depends on the truth.
    """
    m = rng.uniform(mag[0], mag[1], size=(n, 3))
    return m * rng.choice([-1.0, 1.0], size=(n, 3))


def draw_truth_positions(n, V, F, tm, rng, center=None, spread=None, n_try=4000):
    """
    n positions inside the body.

    With `center` and `spread`, drawn uniformly in a BALL of that radius about
    the nominal site — the anomaly is near where we thought, not anywhere in the
    asteroid.  That is the relevant population, since the cylinder was placed
    FOR this site; scattering the truth through the whole interior answers "what
    if we pointed the patch at nothing", a different question.  Without them,
    uniform over the whole body.
    """
    out = []
    while len(out) < n:
        if center is None:
            lo, hi = V.min(0), V.max(0)
            q = rng.uniform(lo, hi, (n_try, 3))
        else:
            d = rng.normal(size=(n_try, 3))
            d /= np.linalg.norm(d, axis=1)[:, None]
            r = spread * rng.uniform(0.0, 1.0, n_try) ** (1.0 / 3.0)
            q = np.asarray(center) + d * r[:, None]
        q = q[inside_body(tm, V, F, q)]
        out.extend(q)
    return np.asarray(out[:n])


def truth_mc_masses(
    P,
    bulk,
    obs,
    cyl,
    ch_modes,
    Lmax,
    Rref,
    eps,
    n_truth=400,
    seed=101,
    beta_ref=None,
    tgt=0,
    n_cloud=2000,
    mag=(0.01, 0.06),
):
    """
    Redraw the truth MASSES; for each, rebuild σ from that truth's own field and
    refit.  The loop is over INTERIORS, so this asks "would this survive a
    different body?", not "how precise is this one fit?".

    Per truth, both halves of the consistency test:
      PREDICTED  sig[case] / bulk_sig[case] — analytic (A^T W A)^-1, no sampling;
      REALIZED   dev[case] / dev_bulk[case] — the error made fitting ONE noisy
                 realization of that interior.

    m_err[case] is that error on the CH target, which figure 2a histograms.

    ONE draw per interior, deliberately: the noise is not averaged away, because
    an observer flies one spacecraft around one body and gets exactly one
    realization, and that is what the experiment is about.  The price is
    that |e| is half-normal about sigma_i and contributes a geometric spread of
    exp(pi/sqrt(8)) = x3.04 on its own, which is wider than the x1.12 the
    interiors themselves span — so this histogram is dominated by the noise draw,
    and it is the CLOUD at `i_rep` (n_cloud draws, one interior) that isolates the
    covariance.  The two answer different questions and both are plotted.

    Every case gets a fresh generator on the same seed, so they see the same
    noise realization wherever they share a block.  Measured, that pairing buys
    NOTHING: corr(e_SH, e_SH+CH) = +0.01 and the per-interior gain scatter is
    identical either way (sd of ln ratio 0.55 vs 0.55 over 300 interiors) —
    once the CH block is in, the SH+CH error is set by the CH data.  Kept as
    the tidier default, not as a help.
    """
    rng = np.random.default_rng(seed)
    betas = draw_truth_masses(n_truth, rng, mag=mag)
    pinvPhi = ch_pinv_for(cyl, obs, ch_modes)
    A_sh = A_stokes_contrast(P, bulk, 2, Lmax, Rref)
    A_ch = A_ch_contrast(P, bulk, obs, cyl, ch_modes)
    sig = {k: np.empty((n_truth, len(P))) for k in CASES}
    dev = {k: np.empty((n_truth, len(P))) for k in CASES}
    bulk_sig = {k: np.empty(n_truth) for k in CASES}
    dev_bulk = {k: np.empty(n_truth) for k in CASES}
    m_err = {k: np.empty(n_truth) for k in CASES}
    one = np.ones(len(P))
    # the interior closest to the nominal one gets the dense cloud, so the
    # covariance check in the figure is made on a representative body
    i_rep = (
        0
        if beta_ref is None
        else int(np.argmin(np.linalg.norm(betas - np.asarray(beta_ref), axis=1)))
    )
    rep = {}
    for i, b in enumerate(betas):
        s_sh = od_sigma(sh_coefficients_total(b, P, bulk, 2, Lmax, Rref), eps)
        s_ch = od_sigma(ch_coefficients_total(b, P, bulk, obs, pinvPhi), eps)
        blocks = case_blocks(A_sh, s_sh, A_ch, s_ch)
        for k in CASES:
            # PREDICTED: the mass fit is LINEAR, so its posterior covariance is
            # exactly (A^T W A)^-1 — no sampling needed for the sigma itself.
            C = mass_fraction_covariance(blocks[k])
            sig[k][i] = np.sqrt(np.diag(C))
            # beta_tilde = 1 - sum(beta), so its variance is 1^T C 1
            bulk_sig[k][i] = np.sqrt(one @ C @ one)
            # REALIZED: fit noisy data for this interior and keep the error made.
            # PAIRED: a fresh generator on the same seed per case, so a shared
            # block sees an identical noise realization across cases.
            e = ls_fit_once(blocks[k], b, np.random.default_rng(7 + i)) - b
            dev[k][i] = e
            # beta_tilde = 1 - sum(beta), so its error is minus the sum of theirs
            dev_bulk[k][i] = -e.sum()
            # what fig 2a histograms: this interior's error on the CH target
            m_err[k][i] = abs(e[tgt])
            if i == i_rep:
                # the ONE place a noise cloud is still needed: panel (c) checks
                # the predicted ellipse against the scatter it claims to describe
                rep.setdefault("m_cloud", {})[k] = monte_carlo_fit(
                    blocks[k], b, n_mc=n_cloud, seed=7 + i
                )
                rep.setdefault("m_cov", {})[k] = C
                rep["m_rep_beta"] = b.copy()
    return dict(
        sig=sig,
        bulk_sig=bulk_sig,
        betas=betas,
        dev=dev,
        dev_bulk=dev_bulk,
        m_err=m_err,
        m_tgt=tgt,
        **rep,
    )


def truth_mc_position(
    P,
    beta_true,
    bulk,
    obs,
    cyl,
    ch_modes,
    Lmax,
    Rref,
    eps,
    V,
    F,
    tm,
    n_truth=300,
    seed=202,
    start_offset=0.03,
    spread=0.12,
    n_cloud=750,
    pair_noise=True,
):
    """
    Redraw the truth POSITION of the shallow anomaly near its nominal site; for
    each, rebuild the data and refit in every observation model.  Returns a dict
    with per-truth RMS position error, where each truth sat relative to the
    cylinder, and — for one representative draw (`i_rep`, the truth closest to
    the nominal site) — the full cloud of recovered positions per case, so the
    estimator's own scatter and covariance can be shown.
    """
    rng = np.random.default_rng(seed)
    pts = draw_truth_positions(n_truth, V, F, tm, rng, center=P[0], spread=spread)
    pinvPhi = ch_pinv_for(cyl, obs, ch_modes)
    axis = cyl.rot() @ np.array([0.0, 0.0, 1.0])
    err = {k: np.empty(n_truth) for k in CASES}
    d_ax, dep = np.empty(n_truth), np.empty(n_truth)
    zmax = V[:, 2].max()
    i_rep = int(np.argmin(np.linalg.norm(pts - P[0], axis=1)))
    clouds, covs, rep_truths = {}, {}, P.copy()
    for i, p0 in enumerate(pts):
        Pi = P.copy()
        Pi[0] = p0
        s_sh = od_sigma(sh_coefficients_total(beta_true, Pi, bulk, 2, Lmax, Rref), eps)
        s_ch = od_sigma(ch_coefficients_total(beta_true, Pi, bulk, obs, pinvPhi), eps)
        v = p0 - cyl.center
        d_ax[i] = np.linalg.norm(v - np.dot(v, axis) * axis)
        dep[i] = zmax - p0[2]
        # one draw everywhere except the representative interior, which gets
        # the dense cloud the covariance-ellipse panel is drawn from
        n_draw = n_cloud if i == i_rep else 1
        for case in CASES:
            blocks = _pos_forward(
                p0, 0, Pi, beta_true, bulk, Lmax, Rref, obs, pinvPhi, case
            )
            sig_b = _pos_sigmas(case, s_sh, s_ch)
            # Same seed for both cases, so the SH block sees an identical
            # noise realization with and without CH.  Measured, this changes
            # nothing (sd of ln gain 0.60 paired vs 0.52 per-case, n = 30 —
            # indistinguishable): the SH+CH error is driven by the CH data, so
            # it is nearly independent of the SH noise regardless.  A matter of
            # tidiness, not of variance reduction.
            r = np.random.default_rng(
                seed + 1000 * i + (0 if pair_noise else CASES.index(case))
            )
            acc = []
            for _ in range(n_draw):
                data = [
                    tb + r.normal(0.0, sg, size=tb.shape)
                    for tb, sg in zip(blocks, sig_b)
                ]
                sol = least_squares(
                    _pos_residual,
                    p0 + start_offset,
                    args=(
                        0,
                        Pi,
                        data,
                        sig_b,
                        beta_true,
                        bulk,
                        Lmax,
                        Rref,
                        obs,
                        pinvPhi,
                        case,
                    ),
                    # Optimization method
                    method="trf",
                    # Better numerical Jacobian
                    jac="3-point",  # More accurate, about 2× the residual evaluations
                    # Automatically account for differently sensitive coordinates
                    x_scale="jac",
                    # Convergence criteria
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12,
                    # Allow difficult cases to converge
                    max_nfev=2000,
                )
                acc.append(sol.x)
            acc = np.asarray(acc)
            err[case][i] = np.sqrt(np.mean(np.sum((acc - p0) ** 2, axis=1)))
            if i == i_rep:
                clouds.setdefault(case, {})[0] = acc
        if i == i_rep:
            # J^T W J from the position partials — the covariance the estimator
            # HAS, rather than the one this particular set of draws happened to
            # produce.  Cheap, and it needs no Monte-Carlo at all.  Every anomaly
            # now, not just the shallow one: fig 3 shows the whole grid.
            for j in range(len(P)):
                cj = position_covariance(
                    j, Pi, obs, cyl, ch_modes, s_sh, s_ch, Lmax, Rref, pinvPhi
                )
                for case in CASES:
                    covs.setdefault(case, {})[j] = cj[case]
            # The MC above only ever moves the SHALLOW anomaly, so only it comes
            # out of that loop with a cloud.  The other two are fitted here —
            # same estimator, same draw count, each freed in turn with the rest
            # held — and only at `i_rep`, the one interior fig 3 is drawn from.
            for j in range(1, len(P)):
                for case in CASES:
                    blocks = _pos_forward(
                        Pi[j], j, Pi, beta_true, bulk, Lmax, Rref, obs, pinvPhi, case
                    )
                    sig_b = _pos_sigmas(case, s_sh, s_ch)
                    r = np.random.default_rng(seed + 5000 + 100 * j)
                    acc = []
                    for _ in range(n_cloud):
                        data = [
                            tb + r.normal(0.0, sg, size=tb.shape)
                            for tb, sg in zip(blocks, sig_b)
                        ]
                        sol = least_squares(
                            _pos_residual,
                            Pi[j] + start_offset,
                            args=(
                                j,
                                Pi,
                                data,
                                sig_b,
                                beta_true,
                                bulk,
                                Lmax,
                                Rref,
                                obs,
                                pinvPhi,
                                case,
                            ),
                            method="trf",
                            jac="3-point",
                            x_scale="jac",
                            xtol=1e-12,
                            ftol=1e-12,
                            gtol=1e-12,
                            max_nfev=2000,
                        )
                        acc.append(sol.x)
                    clouds[case][j] = np.asarray(acc)
            rep_truths = Pi.copy()
    return dict(
        err=err,
        pos=pts,
        d_axis=d_ax,
        depth=dep,
        i_rep=i_rep,
        rep_truth=pts[i_rep],
        # per case, per anomaly: {case: {j: (n_cloud, 3)}} and {case: {j: 3x3}}
        cloud=clouds,
        cov=covs,
        rep_truths=rep_truths,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — RESULTS REPORT  (terminal tables + LaTeX, ready to paste)
# ═══════════════════════════════════════════════════════════════════════════


def _tex_num(v, sig=2):
    """4.93e-03 -> $4.93\times10^{-3}$ ; plain decimal when it reads better."""
    if v == 0 or not np.isfinite(v):
        return "--"
    e = int(np.floor(np.log10(abs(v))))
    if -1 <= e <= 3:  # decimals only where they are genuinely shorter;
        #                    mixing 0.0114 with 6.16e-4 in one column reads badly
        return f"${v:.{max(0, sig - e)}f}$"
    m = v / 10.0**e
    return rf"${m:.{sig}f}\times10^{{{e}}}$"


def results_report(res, tex=True):
    """
    Every number worth quoting, as aligned tables and as LaTeX tabular bodies.
    The figures deliberately carry no values, so this is the single place the
    numbers live — copy a block straight into the paper.
    """
    names, tmc, det = res["names"], res["truth_mc"], res["det"]
    ft, bb = res["beta_true"], res["beta_bulk"]
    q = lambda v: np.percentile(v, [10, 50, 90])
    # one row per component, each carrying the 10/50/90 band in EVERY case
    rows_m = []
    for j, nm in enumerate(names):
        qs = {k: q(tmc["sig"][k][:, j]) for k in CASES}
        g = np.median(tmc["sig"][SH_ONLY][:, j] / tmc["sig"][SH_CH][:, j])
        rows_m.append((nm, ft[j], qs, g))
    rows_m.append(
        (
            r"body $\tilde\beta$",
            bb,
            {k: q(tmc["bulk_sig"][k]) for k in CASES},
            np.median(tmc["bulk_sig"][SH_ONLY] / tmc["bulk_sig"][SH_CH]),
        )
    )

    print(f"\n{SEP}\n  RESULTS  (figures carry no numbers; quote from here)\n{SEP}")

    # β < 0 is a DEFICIT, not a negative mass — the anomalies are contrasts on
    # top of the bulk, so the sign is the physics.  What has to hold is that the
    # implied density stays ≥ 0 (deficit) or below the densest realistic
    # inclusion (excess), which sets a minimum size for the equivalent sphere.
    adm = res["adm"]
    a_min, d_surf, d_obs, b_max = (
        adm["a_min"],
        adm["d_surf"],
        adm["d_obs"],
        adm["b_max"],
    )
    print(
        f"\n  TABLE 0 — physical admissibility of the truth anomalies "
        f"(excess ceiling Δρ/ρ = {EXCESS_CONTRAST:.2f})"
    )
    print(
        f"  {'component':22s} {'β':>8} {'Δρ/ρ':>7} {'a_min':>8} {'to surf':>8} "
        f"{'to obs':>8} {'β_max':>9}  verdict"
    )
    for k, nm in enumerate(names):
        ratio = -1.0 if ft[k] < 0 else EXCESS_CONTRAST
        if a_min[k] > d_surf[k]:
            v = "breaches the surface"
        elif a_min[k] > d_obs[k]:
            v = "field points inside it"
        else:
            v = "buried, clear of the data — exact"
        print(
            f"  {nm:22s} {ft[k]:+8.3f} {ratio:+7.2f} {a_min[k]:8.3f} "
            f"{d_surf[k]:8.3f} {d_obs[k]:8.3f} {b_max[k]:+9.4f}  {v}"
        )
    print(
        f"  body β̃ = {res['beta_bulk']:.3f} > 0, and Σβ + β̃ = 1 exactly, so the "
        "total mass is conserved and positive."
    )
    print(
        "\n  TABLE 1 — mass-fraction uncertainty, "
        f"{len(tmc['sig'][SH_ONLY])} truth interiors, 1σ, median [10–90%]"
    )
    print(
        f"  {'component':22s} {'truth β':>9} "
        + " ".join(f"{'σ ' + k:>10} {'[10–90%]':>21}" for k in CASES)
        + f" {'gain':>7}"
    )
    for nm, tr, qs, g in rows_m:
        nm_ = nm.replace(r"$\tilde\beta$", "β̃")
        print(
            f"  {nm_:22s} {tr:+9.4f} "
            + " ".join(
                f"{qs[k][1]:10.2e} [{qs[k][0]:8.2e},{qs[k][2]:8.2e}]" for k in CASES
            )
            + f" {g:6.1f}×"
        )

    # Does the analytic covariance actually predict the error made?  The
    # realized column comes from noisy fits, the predicted column from
    # (A^T W A)^-1 — nothing is shared, so the ratio is a real check.
    _rms = lambda M: np.sqrt(np.mean(np.asarray(M) ** 2, axis=0))
    real = {
        k: _rms(np.column_stack([tmc["dev"][k], tmc["dev_bulk"][k]])) for k in CASES
    }
    pred = {
        k: _rms(np.column_stack([tmc["sig"][k], tmc["bulk_sig"][k]])) for k in CASES
    }
    nms = [n.replace(r"$\tilde\beta$", "β̃") for n, *_ in rows_m]
    print(
        f"\n  TABLE 1b — covariance consistency, {len(tmc['dev'][SH_ONLY])} noisy "
        "fits: realized RMS(estimate − truth) vs predicted 1σ"
    )
    print(
        f"  {'component':22s} "
        + " ".join(f"{'real ' + k:>13} {'pred':>11} {'ratio':>6}" for k in CASES)
    )
    for j, nm in enumerate(nms):
        print(
            f"  {nm:22s} "
            + " ".join(
                f"{real[k][j]:13.2e} {pred[k][j]:11.2e} "
                f"{real[k][j]/pred[k][j]:6.2f}"
                for k in CASES
            )
        )

    pe, dax = tmc["err"], tmc["d_axis"]
    print(
        f"\n  TABLE 2 — anomaly position, {len(pe[SH_ONLY])} truth interiors, "
        f"RMS error [LU], median [10–90%]"
    )
    print(f"  {'case':22s} {'median':>10} {'[10–90%]':>21} {'gain over SH':>13}")
    for k in CASES:
        Q = q(pe[k])
        gs = f"{np.median(pe[SH_ONLY] / pe[k]):12.1f}×" if k != SH_ONLY else " " * 13
        print(f"  {k:22s} {Q[1]:10.2e} [{Q[0]:8.2e},{Q[2]:8.2e}] {gs}")
    ed = np.quantile(dax, [0, 0.25, 0.5, 0.75, 1.0])
    ed[-1] += 1e-9
    print(f"\n  TABLE 3 — position error vs distance from the cylinder axis")
    print(
        f"  {'range [LU]':>16} {'n':>4} "
        + " ".join(f"{'median ' + k:>15}" for k in CASES)
        + f" {'gain':>7}"
    )
    for lo, hi in zip(ed[:-1], ed[1:]):
        m = (dax >= lo) & (dax < hi)
        if m.sum():
            print(
                f"  {lo:6.3f}–{hi:6.3f} {int(m.sum()):4d} "
                + " ".join(f"{np.median(pe[k][m]):15.2e}" for k in CASES)
                + f" {np.median(pe[SH_ONLY][m]/pe[SH_CH][m]):6.1f}×"
            )

    sp = {
        # `cov` is keyed per case AND per anomaly now; TABLE 4 quotes the target
        k: np.sqrt(np.trace(tmc["cov"][k][res["target"]][np.ix_([0, 2], [0, 2])]) / 2)
        for k in CASES
    }
    print(f"\n  TABLE 4 — single-interior detail and detection limit")
    print(
        f"  {'quantity':38s} " + " ".join(f"{k:>12}" for k in CASES) + f" {'ratio':>8}"
    )
    print(
        f"  {'analytic position 1σ (x–z) [LU]':38s} "
        + " ".join(f"{sp[k]:12.2e}" for k in CASES)
        + f" {sp[SH_ONLY]/sp[SH_CH]:7.1f}×"
    )
    print(
        f"  {'smallest detectable anomaly β_0':38s} "
        + " ".join(f"{det['thr'][k]:12.2e}" for k in CASES)
        + f" {det['thr'][SH_ONLY]/det['thr'][SH_CH]:7.1f}×"
    )
    # These two rows are per-OBSERVABLE, not per-case: they describe the SH and
    # CH coefficient blocks themselves, which the joint fit shares.  Printed
    # under the SH-only and CH-only columns, where those blocks stand alone.
    print(
        f"  {'discrepancy-to-noise, RMS ΔCS/σ':38s} {res['snr_sh']:12.1f} "
        f"{res['snr_ch']:12.1f}"
    )
    print(
        f"  {'post-fit residual [σ]':38s} {res['rms_post_sh']:12.2f} "
        f"{res['rms_post_ch']:12.2f}"
    )

    # ── TABLE 5 — the reach map, reduced to numbers ───────────────────────
    rm = res["reach"]
    print("\n  TABLE 5 — reach: fraction of the body each model can see")
    print(f"  {'threshold σ_β':22s} " + " ".join(f"{k:>12}" for k in CASES))
    for thr in REACH_LEVELS_MASS:
        cells = []
        for k in CASES:
            v = rm["sigma"][k]
            ok = np.isfinite(v)
            cells.append(f"{100.0*np.sum(v[ok] < thr)/max(1, ok.sum()):11.1f}%")
        print(f"  {'below ' + f'{thr:.0e}':22s} " + " ".join(f"{c:>12}" for c in cells))
    a_, b_ = rm["sigma"][SH_ONLY], rm["sigma"][CH_ONLY]
    ok = np.isfinite(a_) & np.isfinite(b_)
    ratio = (b_ / a_)[ok]
    print(
        f"  {'dynamic range, best/worst':22s} "
        + " ".join(
            f"{np.nanmax(rm['sigma'][k][np.isfinite(rm['sigma'][k])])/np.nanmin(rm['sigma'][k][np.isfinite(rm['sigma'][k])]):11.0f}x"
            for k in CASES
        )
    )
    print(
        f"  ⇒ CH alone beats SH alone over {100.0*np.sum(ratio < 1)/ratio.size:.0f}% "
        f"of the cross-section: up to\n    {1.0/ratio.min():.0f}× better under the "
        f"patch, down to {ratio.max():.0f}× worse on the far side.  The two do "
        f"not\n    differ in QUALITY, they differ in SHAPE — one flat field of "
        f"view against one\n    sharp peak — which is why the joint fit is worth "
        f"making."
    )
    # ── TABLE 5b — the same reach, for POSITION ───────────────────────────
    rp = rm["sigma_pos"]
    print(
        f"\n  TABLE 5b — reach, position: fraction of the body where a test "
        f"anomaly (β = {rm['beta_test']:.2f}) is placed to"
    )
    print(f"  {'threshold 1σ [LU]':22s} " + " ".join(f"{k:>12}" for k in CASES))
    for thr in REACH_LEVELS_POS:
        cells = []
        for k in CASES:
            v = rp[k]
            ok = np.isfinite(v)
            cells.append(f"{100.0*np.sum(v[ok] < thr)/max(1, ok.sum()):11.1f}%")
        print(f"  {'below ' + f'{thr:.0e}':22s} " + " ".join(f"{c:>12}" for c in cells))
    a_, b_ = rp[SH_ONLY], rp[CH_ONLY]
    ok = np.isfinite(a_) & np.isfinite(b_)
    ratio = (b_ / a_)[ok]
    print(
        f"  ⇒ CH alone places it better than SH alone over "
        f"{100.0*np.sum(ratio < 1)/ratio.size:.0f}% of the cross-section: up to "
        f"{1.0/ratio.min():.0f}× better\n    under the patch, down to "
        f"{ratio.max():.0f}× worse on the far side."
    )

    if not tex:
        return
    print(f"\n{'-'*70}\n  LaTeX tabular bodies\n{'-'*70}")
    print(r"  % Table 1 — mass-fraction uncertainty (median 1σ per case)")
    for nm, tr, qs, g in rows_m:
        print(
            rf"  {nm} & ${tr:+.4f}$ & "
            + " & ".join(_tex_num(qs[k][1]) for k in CASES)
            + rf" & ${g:.1f}$ \\"
        )
    print(r"  % Table 2 — position RMS error [LU]")
    for k in CASES:
        Q = q(pe[k])
        print(rf"  {k} & {_tex_num(Q[1])} & {_tex_num(Q[0])} & {_tex_num(Q[2])} \\")
    print(
        rf"  % gain (median of per-interior ratios): "
        rf"${np.median(pe[SH_ONLY]/pe[SH_CH]):.1f}$"
    )


def run_experiment(
    Lmax_sh=6,
    eps=0.02,
    ch_modes=(4, 4),
    n_cyl_pts=1000,  # field samples in the cylinder -- geometry, NOT an MC size
    detail=False,  # verbose narrative; the tables at the end carry the numbers
    # Every Monte-Carlo size below reads n_<role>_<experiment>:
    #   n_truth_*  how many truth interiors are drawn, one noisy fit each
    #   n_cloud_*  extra draws for the ONE interior the ellipse panel shows
    # with _m = mass fractions (experiment 1), _p = positions (experiment 2).
    # ── experiment 1 — MASS FRACTIONS  (400 linear fits) ──────────────────
    # ONE noisy fit per interior: the MC is over BODIES, not over noise.  The
    # covariance is checked separately by the n_cloud_m draws at one interior.
    n_truth_m=500,
    n_cloud_m=1000,  # the cloud behind fig 2a's covariance ellipses
    truth_mag=(0.01, 0.06),  # |beta_j| drawn uniformly in this band, sign random
    seed_mass=101,  # which set of truth interiors gets drawn
    # ── experiment 2 — POSITIONS  (300 nonlinear fits + the cloud) ────────
    # Same design, and these are TRF fits with a numerical Jacobian: they
    # dominate the runtime, so raise n_truth_p last and the mass sizes first.
    n_truth_p=500,
    n_cloud_p=1000,  # the cloud behind fig 3's covariance ellipse
    pos_spread=0.20,  # truth positions jitter within this radius of the site
    seed_pos=202,
    pos_start_offset=0.03,  # how far the nonlinear fit starts from the truth
    # ── the CH cylinder ────────────────────────────────────────────────────
    cyl_radius=0.12,
    cyl_height=0.40,
    cyl_gap=0.005,  # lift of the cylinder base above the +z pole
    target=0,  # which anomaly the cylinder is placed over / the CH target
    # ── detection sweep (figure 2b) ────────────────────────────────────────
    n_sweep=1000,  # noise draws at each grid point, per case
    det_range=(-4.5, -0.7),  # log10 span of the true-anomaly sweep
    det_n=16,
    det_acc=25.0,  # RMS error, as a % of the anomaly, that counts as "measured"
    map_n=61,  # grid resolution of the reach map (x-z slice, body-masked)
    outdir="Images",
    verbose=True,
):
    """
    `eps` is the RELATIVE measurement precision applied EQUALLY to both
    observables, PER COEFFICIENT: σ_i = eps·|coefficient_i| with a noise floor
    (see `od_sigma`), on the FULL measured coefficients (bulk included — what an
    OD solution actually delivers).  Same fractional data quality on the global
    Stokes and the local CH coefficients, so the comparison reflects geometry,
    not the two bases' different natural units.  What is FITTED is the
    discrepancy between that measurement and the known constant-density model.
    The Phi-to-field fit producing the CH coefficients from field samples is
    unweighted; the weights live here, on the coefficients.
    """
    V, F, tm, Rb = load_eros()
    Rref = Rb
    zmax = V[:, 2].max()
    names, P, beta_true = mascon_arrays()
    bulk = Bulk(V, F)
    beta_bulk = bulk_fraction(beta_true)

    if verbose:
        print(SEP)
        print(
            "  Interior-density recovery: SH  vs  CH  vs  SH+CH   "
            "(Eros, normalized units)"
        )
        print(SEP)
        print(
            f"  Brillouin R* = {Rb:.3f} LU,  z_max = {zmax:.3f} LU,  "
            f"volume = {bulk.volume:.4f} LU³"
        )
        print(
            f"  BULK: constant-density polyhedron, β̃ = 1 − Σβ = {beta_bulk:.3f} "
            f"of M*  (C̄20 = {bulk.stokes(2, Lmax_sh, Rref)[0]:+.4f})"
        )
        print("  anomalies (truth mass fractions β_j, + = excess, − = deficit):")
        for nm, p, fr in zip(names, P, beta_true):
            print(f"    {nm:22s} p={np.round(p,3)}  β={fr:+.3f}  depth={zmax-p[2]:.3f}")

    # cylinder of near-surface data over the anomaly (+z pole)
    cyl = Cylinder(
        center=np.array([0.0, 0.0, zmax + cyl_gap]),
        radius=cyl_radius,
        height=cyl_height,
    )
    obs = cylinder_points(cyl, n=n_cyl_pts)
    obs = obs[~inside_body(tm, V, F, obs)]
    r_obs = np.linalg.norm(obs, axis=1)

    # Are the truth anomalies physically realizable?  β < 0 is a deficit, not a
    # negative mass; the real constraint is on the density it implies.
    a_min, d_surf, d_obs, b_max = admissibility(
        P, beta_true, beta_bulk, bulk.volume, obs, tm
    )

    # DISCREPANCY designs: every column is (point mass at p_j) − (same mass
    # spread through the body), so β is estimated against the constant-density
    # model rather than against vacuum.
    A_sh = A_stokes_contrast(P, bulk, 2, Lmax_sh, Rref)
    pinvPhi = ch_pinv_for(cyl, obs, ch_modes)  # UNWEIGHTED Phi-to-field fit, trunc. SVD
    A_ch = A_ch_contrast(P, bulk, obs, cyl, ch_modes)

    # OD-like per-coefficient noise on the FULL measured coefficients (bulk +
    # anomalies), the same relative rule on both observables.
    y_sh_tot = sh_coefficients_total(beta_true, P, bulk, 2, Lmax_sh, Rref)
    y_ch_tot = ch_coefficients_total(beta_true, P, bulk, obs, pinvPhi)
    sig_sh = od_sigma(y_sh_tot, eps)
    sig_ch = od_sigma(y_ch_tot, eps)
    blocks = case_blocks(A_sh, sig_sh, A_ch, sig_ch)
    if verbose:
        print(
            f"  cylinder over anomaly: {len(obs)} vacuum pts, "
            f"|r|∈[{r_obs.min():.2f},{r_obs.max():.2f}] ⊂ Brillouin {Rb:.2f}"
        )
        print(
            f"  observables: SH deg 2..{Lmax_sh} ({A_sh.shape[0]} coeffs)"
            f" | CH modes {ch_modes} ({2 * ch_modes[0] * ch_modes[1]} cols)"
            f"  [no Σβ=1 row — the mass budget is structural]"
        )
        print(
            f"  weights: OD-like σ_i = {eps}·|coeff_i| (floor 10% of RMS)  →  "
            f"σ_SH ∈ [{sig_sh.min():.2e}, {sig_sh.max():.2e}], "
            f"σ_CH ∈ [{sig_ch.min():.2e}, {sig_ch.max():.2e}]"
        )

    # ── PART 1 — mass fractions ────────────────────────────────────────────
    # Covariance per case, then reduced to the number that gets quoted — the
    # same two steps PART 2 takes.  A vector parameter there, separate scalars
    # here, so the reducer differs (`posterior_rms` vs `posterior_sigma`) and
    # nothing else does.
    C_mass = {k: mass_fraction_covariance(blocks[k]) for k in CASES}
    sd_mass = {k: posterior_sigma(C_mass[k]) for k in CASES}
    one_ = np.ones(len(P))
    sd_bulk = {k: float(np.sqrt(one_ @ C_mass[k] @ one_)) for k in CASES}
    mass_gain = sd_mass[SH_ONLY] / sd_mass[SH_CH]
    if verbose and detail:
        print(f"\n{'-'*70}\n  PART 1 — MASS-FRACTION UNCERTAINTY (1σ on β_j)\n{'-'*70}")
        print(
            f"  {'anomaly':22s} {'depth':>6} "
            + " ".join(f"{k:>11s}" for k in CASES)
            + f" {'gain':>7}"
        )
        for j, (nm, p) in enumerate(zip(names, P)):
            print(
                f"  {nm:22s} {zmax-p[2]:6.3f} "
                + " ".join(f"{sd_mass[k][j]:11.2e}" for k in CASES)
                + f" {mass_gain[j]:6.1f}×"
            )

    # ── PART 2 — position of the near-surface anomaly ───────────────────────
    C_pos = position_covariance(
        target, P, obs, cyl, ch_modes, sig_sh, sig_ch, Lmax_sh, Rref, pinvPhi
    )
    rms_pos = {k: posterior_rms(C_pos[k]) for k in CASES}
    pos_gain = rms_pos[SH_ONLY] / rms_pos[SH_CH]
    if verbose and detail:
        print(
            f"\n{'-'*70}\n  PART 2 — POSITION OF NEAR-SURFACE ANOMALY "
            f"(β={beta_true[target]:+.3f})\n{'-'*70}"
        )
        print(
            "  position 1σ RMS:  "
            + "   ".join(f"{k}={rms_pos[k]:.3e} LU" for k in CASES)
            + f"   → {pos_gain:.0f}× tighter"
        )

    # ══ EXPERIMENTS 1 & 2 — MONTE-CARLO OVER THE TRUTH ═════════════════════
    # Experiment 1 resamples the truth MASSES, experiment 2 the truth POSITION
    # of the shallow anomaly.  Each draw is a different interior, refitted from
    # scratch, so the spread reported below is over INTERIORS — "would this
    # conclusion survive a different body?" — rather than over noise draws at
    # one hard-coded truth, which only answers "how precise is this one fit?".
    tmm = truth_mc_masses(
        P,
        bulk,
        obs,
        cyl,
        ch_modes,
        Lmax_sh,
        Rref,
        eps,
        n_truth=n_truth_m,
        seed=seed_mass,
        beta_ref=beta_true,
        tgt=target,
        n_cloud=n_cloud_m,
        mag=truth_mag,
    )
    tp = truth_mc_position(
        P,
        beta_true,
        bulk,
        obs,
        cyl,
        ch_modes,
        Lmax_sh,
        Rref,
        eps,
        V,
        F,
        tm,
        n_truth=n_truth_p,
        seed=seed_pos,
        start_offset=pos_start_offset,
        spread=pos_spread,
        n_cloud=n_cloud_p,
    )
    tp_err, tp_dax = tp["err"], tp["d_axis"]
    if verbose and detail:
        q = lambda v: (np.percentile(v, 10), np.median(v), np.percentile(v, 90))
        print(
            f"\n{'='*70}\n  EXPERIMENT 1 — MASS FRACTIONS, Monte-Carlo over "
            f"{n_truth_m} truth interiors\n{'='*70}"
        )
        print(f"  truth |β| ~ U[0.01,0.06] with random signs; positions fixed")
        print(
            f"  {'anomaly':22s} "
            + " ".join(f"{k + '  10/50/90%':>26}" for k in CASES)
            + f" {'gain':>7}"
        )
        for j, nm in enumerate(names):
            g = np.median(tmm["sig"][SH_ONLY][:, j] / tmm["sig"][SH_CH][:, j])
            print(
                f"  {nm:22s} "
                + "  ".join(
                    " ".join(f"{x:8.2e}" for x in q(tmm["sig"][k][:, j])) for k in CASES
                )
                + f" {g:6.1f}×"
            )
        gb = np.median(tmm["bulk_sig"][SH_ONLY] / tmm["bulk_sig"][SH_CH])
        print(
            f"  {'BODY β̃':22s} "
            + "  ".join(
                " ".join(f"{x:8.2e}" for x in q(tmm["bulk_sig"][k])) for k in CASES
            )
            + f" {gb:6.1f}×"
        )
        print(
            f"\n{'='*70}\n  EXPERIMENT 2 — ANOMALY POSITION, Monte-Carlo over "
            f"{n_truth_p} truth positions\n{'='*70}"
        )
        print(
            f"  the shallow anomaly is jittered within {pos_spread} LU of its "
            f"site (one noise draw each)"
        )
        print(f"  {'':22s} {'RMS err 10/50/90%  [LU]':>28} {'gain over SH':>13}")
        for k in CASES:
            g = (
                f" {np.median(tp_err[SH_ONLY] / tp_err[k]):12.1f}×"
                if k != SH_ONLY
                else ""
            )
            print(f"  {k:22s} " + " ".join(f"{x:9.2e}" for x in q(tp_err[k])) + g)
        edges = np.quantile(tp_dax, [0.0, 0.25, 0.5, 0.75, 1.0])
        edges[-1] += 1e-9
        print(f"\n  by horizontal distance from the cylinder axis (quartiles):")
        print(
            f"  {'range [LU]':>14} {'n':>4} "
            + " ".join(f"{'median ' + k:>15s}" for k in CASES)
            + f" {'gain':>8}"
        )
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (tp_dax >= lo) & (tp_dax < hi)
            if m.sum() == 0:
                continue
            print(
                f"  {lo:5.3f}–{hi:5.3f} {int(m.sum()):4d} "
                + " ".join(f"{np.median(tp_err[k][m]):15.2e}" for k in CASES)
                + f" {np.median(tp_err[SH_ONLY][m] / tp_err[SH_CH][m]):7.1f}×"
            )
        print(
            "  ⇒ the gain barely moves across the site, so it is a property "
            "of the patch\n    covering the anomaly — not of the anomaly "
            "landing on one lucky spot."
        )

    # ── COEFFICIENT SPECTRA: homogeneous vs heterogeneous, pre/post fit ────
    # One noisy realization, fitted jointly, so fig 3 can show the residual
    # collapsing from the pre-fit discrepancy onto the noise floor.
    rng_sp = np.random.default_rng(99)
    d_sh, d_ch = A_sh @ beta_true, A_ch @ beta_true  # = CS_hetero − CS_homog
    dat_sh = d_sh + rng_sp.normal(0.0, sig_sh)
    dat_ch = d_ch + rng_sp.normal(0.0, sig_ch)
    Aw = np.vstack([A_sh / sig_sh[:, None], A_ch / sig_ch[:, None]])
    yw = np.concatenate([dat_sh / sig_sh, dat_ch / sig_ch])
    beta_hat, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    if verbose and detail:
        snr = lambda d, sg: float(np.sqrt(np.mean((d / sg) ** 2)))
        print(
            f"\n  discrepancy-to-noise per coefficient (RMS of ΔCS/σ):"
            f"  SH {snr(d_sh, sig_sh):.1f}   CH {snr(d_ch, sig_ch):.1f}"
        )

    r_sh = (dat_sh - A_sh @ beta_hat) / sig_sh
    r_ch = (dat_ch - A_ch @ beta_hat) / sig_ch
    snr_sh = float(np.sqrt(np.mean((d_sh / sig_sh) ** 2)))
    snr_ch = float(np.sqrt(np.mean((d_ch / sig_ch) ** 2)))

    spectra = dict(
        sh=dict(
            homog=bulk.stokes(2, Lmax_sh, Rref),
            hetero=y_sh_tot,
            diff=d_sh,
            sigma=sig_sh,
            data=dat_sh,
            model=A_sh @ beta_hat,
        ),
        ch=dict(
            homog=pinvPhi @ bulk.field(obs),
            hetero=y_ch_tot,
            diff=d_ch,
            sigma=sig_ch,
            data=dat_ch,
            model=A_ch @ beta_hat,
        ),
        beta_hat=beta_hat,
        Lmin=2,
        Lmax=Lmax_sh,
        ch_modes=ch_modes,
    )
    if verbose and detail:
        snr = lambda d, sg: float(np.sqrt(np.mean((d / sg) ** 2)))
        print(
            f"\n  discrepancy-to-noise per coefficient (RMS of ΔCS/σ):"
            f"  SH {snr(d_sh, sig_sh):.1f}   CH {snr(d_ch, sig_ch):.1f}"
        )

    # smallest detectable anomaly (part of Experiment 1: positions fixed)
    f_base = beta_true.copy()
    mu_grid = np.logspace(*det_range, det_n)  # swept truth mass fraction beta_0
    det = detection_sweep(A_sh, sig_sh, A_ch, sig_ch, f_base, mu_grid, n_mc=n_sweep)
    det["acc"] = det_acc
    if verbose and detail:
        print(f"\n  smallest detectable anomaly (3σ fit scatter):")
        for k in CASES:
            g = (
                f"   → {det['thr'][SH_ONLY]/det['thr'][k]:.0f}× smaller anomaly "
                f"detectable"
                if k != SH_ONLY
                else ""
            )
            print(f"    {k:9s}: μ_min = {det['thr'][k]:.2e}{g}")

    # where each observable can see an anomaly AT ALL — the panel that says what
    # the CH-only column in the tables means geometrically
    reach = reach_map(
        bulk, [obs], [pinvPhi], [sig_ch], sig_sh, Lmax_sh, Rref, V, F, tm, n=map_n
    )

    res = dict(
        V=V,
        F=F,
        Rb=Rb,
        zmax=zmax,
        cyl=cyl,
        obs=obs,
        P=P,
        names=names,
        beta_true=beta_true,
        bulk=bulk,
        beta_bulk=beta_bulk,
        adm=dict(a_min=a_min, d_surf=d_surf, d_obs=d_obs, b_max=b_max),
        target=target,
        sd_mass=sd_mass,
        sd_bulk=sd_bulk,
        mass_gain=mass_gain,
        C_pos=C_pos,
        rms_pos=rms_pos,
        pos_gain=pos_gain,
        reach=reach,
        det=det,
        snr_sh=snr_sh,
        snr_ch=snr_ch,
        rms_post_sh=float(np.sqrt(np.mean(r_sh**2))),
        rms_post_ch=float(np.sqrt(np.mean(r_ch**2))),
        pos_spread=pos_spread,
        truth_mc=dict(**tmm, **tp),
        spectra=spectra,
        Lmax_sh=Lmax_sh,
        ch_modes=ch_modes,
        sig_sh=sig_sh,
        sig_ch=sig_ch,
        # `eps` is not a diagnostic — `sweep_lmax_sh` reads it back so the sweep
        # is run at the SAME precision as the tables it extends, rather than at
        # whatever this module's default happens to be that week.
        eps=eps,
    )
    make_plots(res, outdir=outdir)
    if verbose:
        results_report(res)
    return res


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════


def set_axes_true_shape(ax, pts, pad=0.04):
    """
    Make a 3-D axes show the body's TRUE proportions.

    `ax.set_box_aspect([1, 1, 1])` makes the drawing BOX cubic, which is not the
    same thing: with autoscaled limits the three axes then cover different data
    ranges and an elongated body (Eros spans 1.60 × 0.71 × 0.58 LU) is stretched
    into something round.  Setting the box aspect to the data extents instead
    keeps the shape honest and the framing tight.
    """
    pts = np.asarray(pts, float)
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-9)
    lo, hi = lo - pad * span, hi + pad * span
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    # Thin the ticks.  The default locator fills each axis with as many labels as
    # its DATA range warrants, but a 3-D axis is drawn foreshortened — the short
    # axis of an elongated body ends up with ~8 labels crammed into a couple of
    # projected centimetres, running into each other and into the axis label.
    # Scale the count with the axis's share of the largest extent, so the long
    # axis keeps a useful number and the short ones are not overcrowded.
    frac = (hi - lo) / (hi - lo).max()
    for axis, f in zip((ax.xaxis, ax.yaxis, ax.zaxis), frac):
        axis.set_major_locator(mpl.ticker.MaxNLocator(nbins=max(3, round(5 * f))))


def draw_cylinder(ax, cyl, color=ACCENT, alpha=0.20, n_th=48, lw=0.9, label=None):
    """
    Draw a `Cylinder` AS a cylinder — translucent lateral surface plus the two
    end rings — instead of scattering the field points that happen to sample it.
    The sample cloud shows where the data is, but it reads as noise; the solid
    body shows what the CH patch actually covers.
    """
    th = np.linspace(0.0, 2.0 * np.pi, n_th)
    zz = np.array([0.0, cyl.height])
    TH, ZZ = np.meshgrid(th, zz)
    loc = np.stack([cyl.radius * np.cos(TH), cyl.radius * np.sin(TH), ZZ], axis=-1)
    g = loc @ cyl.rot().T + cyl.center
    ax.plot_surface(
        g[..., 0],
        g[..., 1],
        g[..., 2],
        color=color,
        alpha=alpha,
        linewidth=0,
        shade=False,
        zorder=2,
    )
    for z0 in zz:  # end rings, so the footprint reads even at grazing angles
        c = np.stack(
            [cyl.radius * np.cos(th), cyl.radius * np.sin(th), np.full_like(th, z0)],
            axis=-1,
        )
        gg = c @ cyl.rot().T + cyl.center
        ax.plot(
            gg[:, 0],
            gg[:, 1],
            gg[:, 2],
            color=color,
            lw=lw,
            alpha=0.85,
            zorder=3,
            label=label if z0 == 0.0 else None,
        )


def draw_cylinder_2d(ax, cyl, i=0, j=2, color=ACCENT, lw=1.4, ls="-", label=None):
    """
    The cylinder's OUTLINE in a 2-D projection (default the x-z plane).

    The 2-D twin of `draw_cylinder`.  A map panel wants the instrument's
    footprint, not the sample points that happen to fill it: the scatter says
    where the draws landed, which is an artefact of `n_cyl_pts` and the seed,
    while the outline says what the geometry actually is and stays legible when
    six of them are drawn at once.

    The cylinder can be tilted, so the silhouette is the convex hull of its
    projected hull points rather than a rectangle.
    """
    from scipy.spatial import ConvexHull

    pts = cylinder_hull(cyl)[:, [i, j]]
    h = ConvexHull(pts)
    loop = np.append(h.vertices, h.vertices[0])
    ax.plot(
        pts[loop, 0], pts[loop, 1], color=color, lw=lw, ls=ls, label=label, zorder=6
    )


def cylinder_hull(cyl, n_th=16):
    """Points spanning a cylinder — for including it in the axis limits."""
    th = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    zz = np.array([0.0, cyl.height])
    TH, ZZ = np.meshgrid(th, zz)
    loc = np.stack([cyl.radius * np.cos(TH), cyl.radius * np.sin(TH), ZZ], axis=-1)
    return (loc.reshape(-1, 3) @ cyl.rot().T) + cyl.center


def _sigma_shade(w):
    """
    Colour for the noise gradient under the coefficient spectra, `w` running
    from 1 at the 3-sigma edge to 0 at the axis.

    A blend between two cool grey-blues rather than a matplotlib colormap: the
    ramp has to sit UNDER coloured data without competing with it, so both ends
    are desaturated and the span is narrow.  Returned as an RGB triple, which is
    what `fill_between` wants.
    """
    lo = np.array([0.760, 0.800, 0.870])  # near the axis, the darkest
    hi = np.array([0.945, 0.957, 0.976])  # at 3 sigma, almost white
    return tuple(lo + (hi - lo) * float(np.clip(w, 0.0, 1.0)))


def sigma_bands(ax, xs, y_sig, pad=0.4):
    """
    Shade the 1σ, 2σ and 3σ noise levels under a coefficient spectrum, and set
    the x-limits to match.

    A CONTINUOUS ramp, not three flat steps: shading the noise as a gradient
    that darkens toward the axis says "more likely here" without drawing edges
    the eye reads as features.  Built from many thin fills because
    `fill_between` takes one colour — 64 of them is visually continuous at print
    resolution — with a hairline at 1, 2 and 3 sigma so the levels can still be
    READ off, each named once just outside the right spine.

    Carried FLAT to the frame, `pad` beyond the first and last group: stopped at
    the groups, the fill left a blank strip at either end of the axis.
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
            facecolor=_sigma_shade(t / 3.0),  # 1 at the outer edge, ~0 at the axis
            edgecolor="none",
            zorder=1,
        )
    for k in (3, 2, 1):
        ax.plot(xe, k * ye, color="#9aa6bd", lw=0.6, zorder=1)
        # outside the frame: inside, the label would sit on its own hairline,
        # which now runs on to the right spine
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


def lognormal_overlay(ax, v, bins, color, ls="-", name="", npts=400):
    """
    Fit a LOG-normal to a positive, log-binned series and draw it in counts.

    Not a Gaussian: these errors are positive, span decades, and sit on a log
    axis.  By KS on the four series these panels show, the log-normal wins on
    three, ties on the fourth, and is never rejected (p = 0.15 to 0.90); a
    normal is rejected on the position ones (KS 0.18-0.22 vs 0.08-0.11).  On a
    log axis a log-normal is just a Gaussian in ln x, hence the clean fit.

    Returns (median, sigma_factor): the natural centre is the median exp(mu),
    the natural width the MULTIPLICATIVE exp(sigma) — [median/f, median*f] is
    the 1-sigma interval.
    """
    lv = np.log(np.asarray(v, float))
    mu, sd = float(lv.mean()), float(lv.std(ddof=1))
    dlog = math.log(bins[1] / bins[0])  # bins are uniform in ln x
    x = np.logspace(math.log10(bins[0]), math.log10(bins[-1]), npts)
    y = (
        len(lv)
        * dlog
        / (sd * math.sqrt(2.0 * math.pi))
        * np.exp(-0.5 * ((np.log(x) - mu) / sd) ** 2)
    )
    med, fac = math.exp(mu), math.exp(sd)
    e = int(math.floor(math.log10(abs(med))))
    lab = (
        rf"{name} Log-normal fit: $\mu={med / 10 ** e:.2f}"
        rf"\times 10^{{{e}}}$, $\sigma=\times{fac:.2f}$"
    ).lstrip()
    ax.plot(x, y, color=color, lw=2.0, ls=ls, zorder=7, label=lab)
    return med, fac


def cov_ellipse(ax, mean2, cov2, color, nsig=1.0, **kw):
    """1σ (or nσ) error ellipse of a 2-D covariance, for the estimate clouds."""
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.maximum(vals, 0.0)
    t = np.linspace(0.0, 2.0 * np.pi, 240)
    e = vecs @ (nsig * np.sqrt(vals)[:, None] * np.array([np.cos(t), np.sin(t)]))
    ax.plot(mean2[0] + e[0], mean2[1] + e[1], color=color, **kw)


# Level of the one predicted-covariance contour every estimate-cloud panel
# draws, and the half-width of their zoom insets in sigma of the WIDEST model
# an inset shows (the parent axes keep their own 3.4-3.9 sigma framing).
COV_NSIG = 2.0
ZOOM_HALF = 4.5


def cov_ellipse_nsig(ax, mean2, cov2, color, lw=1.8, **kw):
    """
    The COV_NSIG contour of one covariance — the one contour these panels draw.

    2σ: it encloses ~86% of a 2-D Gaussian, so it outlines the BODY of the
    cloud the reader sees.  A 1σ ellipse (39%) sits well inside the scatter and
    reads as tighter than the estimator is; a 3σ one (99%) runs round the last
    few stray draws and leaves the core unmarked.  One level only: drawing two
    doubled the curves in every panel for one extra number.
    """
    cov_ellipse(ax, mean2, cov2, color, nsig=COV_NSIG, lw=lw, **kw)


def draw_silhouette(ax, V, F, i, j, color="0.82", edge="0.6", zorder=0):
    """Filled cross-section silhouette of the shape projected onto axes (i, j)."""
    from matplotlib.collections import PolyCollection

    tris2d = V[F][:, :, [i, j]]
    ax.add_collection(
        PolyCollection(
            tris2d, facecolors=color, edgecolors="none", alpha=0.9, zorder=zorder
        )
    )
    # outer envelope (upper/lower j vs i) for a crisp outline
    xi = V[:, i]
    order = np.argsort(xi)
    ax.plot(V[order, i], V[order, j], ",", color=edge, alpha=0.0)  # keep autoscale sane


# Every panel below is written to its OWN file: the paper places the figures
# individually, so nothing is composed into a multi-panel sheet here.  The sizes
# are single-panel canvases; `bbox_inches="tight"` trims whatever the
# equal-aspect panels leave over.
FS = (7.2, 5.4)  # default standalone panel
FS_SQ = (6.6, 6.2)  # equal-aspect scatter (the position clouds)
FS_WIDE = (8.4, 5.0)  # equal-aspect panel with a wide footprint (silhouette)


# 3-D axis labels sit BEYOND their tick labels, and mplot3d's default labelpad
# of 4 pt was already marginal — at FONT_SCALE > 1 the label lands on top of the
# numbers.  Scale the pad with the text so the gap stays proportional.
LPAD3D = 10 * FONT_SCALE


def _save3d(fig, outdir, name, right=0.92, left=0.02, bottom=0.04, top=0.97):
    """
    Save a 3-D panel WITHOUT the tight crop.

    mplot3d places the axis labels outside the axes' reported bounding box, so
    `bbox_inches="tight"` computes a crop that does not contain them and slices
    the z label off the right edge.  pad_inches only buys margin until the label
    grows again — measured, it overhangs the tight bbox by ~0.42 in at
    FONT_SCALE 1.35, and the fix would have to be re-tuned on every font change.
    Reserving the margin inside the figure and saving the whole canvas is exact
    and costs no extra white border.  (A 3-D panel carrying a colour bar does not
    need this: the bar sits outboard of the z label and pulls the tight bbox out
    past it on its own.)
    """
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(os.path.join(outdir, name))


def _save(fig, outdir, name, pad=None):
    """Tight-layout and tight-crop one standalone panel to `outdir/name`.

    No dpi here: `savefig.dpi` (300) is set in the rcParams block above, and only
    rasterized content is affected by it.  `pad` sets `pad_inches` — 3-D panels
    need PAD3D, see above.
    """
    fig.tight_layout()
    kw = {"pad_inches": pad} if pad is not None else {}
    fig.savefig(os.path.join(outdir, name), bbox_inches="tight", **kw)


def hist_legend(ax):
    """
    Legend ABOVE a log-normal histogram panel, never inside it.

    These panels carry two histograms PLUS their two fitted curves, and the fit
    labels quote mu and sigma, so the box is both tall and wide — placed in any
    corner it sits on the bars, because the two histograms are separated along x
    and between them they occupy most of the width.  Two columns, filled
    column-wise, so each histogram lands directly above its own fit.
    """
    ax.legend(
        fontsize=8 * FONT_SCALE,
        ncol=2,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01, 1.0, 0.2),
        mode="expand",
        borderaxespad=0.0,
        handletextpad=0.5,
        columnspacing=1.2,
    )


def bouguer_map(
    beta,
    positions,
    bulk,
    V,
    outdir,
    fname,
    names=None,
    marks=None,
    R_map=None,
    n_lon=181,
    n_lat=91,
    at="sphere",
    F=None,
    clearance=0.02,
):
    """
    BOUGUER MAP: the heterogeneous truth MINUS the constant-density shape model,
    on a sphere, in latitude and longitude.

    What is differenced.  The truth is beta~ U_CD + sum_j beta_j U_pt and the
    reference is the SAME shape at constant density carrying the SAME total mass,
    so the difference is exactly the contrast the estimator fits,

        Delta(r) = sum_j beta_j [ pt_j(r) - U_CD(r) ]  =  A_field_contrast @ beta

    — one call, no separate forward model.  Because both terms carry unit total
    mass, the monopole cancels identically and what remains is the shape of the
    heterogeneity, not the body's mass.

    Plotted quantity is the RADIAL gravity disturbance, Delta g_r = -Delta a . rhat,
    the sign flipped so that a mass EXCESS reads positive, as a gravity anomaly
    conventionally does (this file stores accelerations as a = +grad U, so
    a . rhat is negative outside a positive mass).

    WHERE it is evaluated, `at`:

      "sphere"   (default) just outside the Brillouin sphere, the smallest
                 radius on which an exterior spherical-harmonic series is
                 guaranteed to converge — the natural surface for comparing a
                 GLOBAL field, and the one the SH experiments live on.
      "surface"  a constant `clearance` above the local terrain, found by
                 ray-casting the shape along each map direction.  The analogue
                 of a terrestrial Bouguer map, and the honest choice when the
                 question is WHICH heterogeneity sits WHERE.

    The distinction matters for an elongated body.  Measured on Eros: the
    Brillouin sphere grazes the long-axis tips (0.018 LU of clearance) but
    stands 0.73 LU off the waist — 85% of the Brillouin radius, with a median
    altitude of 0.58 LU.  A localized anomaly's signature falls off with
    distance, so the sphere systematically SUPPRESSES anomalies under the waist
    relative to those under the tips: part of "the lobes dominate" in the
    default map is that geometry, not their masses.  "surface" removes it by
    putting every point the same height above the terrain.

    Either way the map is EXACT, not truncated: it is what a perfect instrument
    would see, and the point of the experiments is how little of it survives
    degree <= L_SH.
    """
    V = np.asarray(V, float)
    lon = np.linspace(-180.0, 180.0, n_lon)
    lat = np.linspace(-90.0, 90.0, n_lat)
    LON, LAT = np.meshgrid(lon, lat)
    cl, sl = np.cos(np.radians(LAT)), np.sin(np.radians(LAT))
    ux = (cl * np.cos(np.radians(LON))).ravel()
    uy = (cl * np.sin(np.radians(LON))).ravel()
    uz = sl.ravel()
    u = np.column_stack([ux, uy, uz])
    if at == "sphere":
        r_eval = np.full(
            len(u),
            float(np.linalg.norm(V, axis=1).max()) * 1.02 if R_map is None else R_map,
        )
    elif at == "surface":
        if F is None or not _HAVE_TRIMESH:
            raise ValueError('at="surface" needs the faces F and trimesh')
        # ray-cast the shape from the origin along every map direction; the
        # clearance keeps the points strictly OUTSIDE, since a point sitting
        # exactly on a face is a boundary case for the polyhedral kernel
        tm_ = trimesh.Trimesh(V, np.asarray(F, int), process=False)
        loc, i_ray, _ = tm_.ray.intersects_location(
            ray_origins=np.zeros_like(u), ray_directions=u, multiple_hits=False
        )
        r_surf = np.full(len(u), np.nan)
        r_surf[i_ray] = np.linalg.norm(loc, axis=1)
        # a ray can miss on a numerically awkward edge; fall back to the median
        r_surf[~np.isfinite(r_surf)] = np.nanmedian(r_surf)
        r_eval = r_surf + clearance
    else:
        raise ValueError('at must be "sphere" or "surface"')
    obs = r_eval[:, None] * u

    d = A_field_contrast(np.asarray(positions, float), bulk, obs) @ np.asarray(
        beta, float
    )
    n = len(obs)
    dgr = -(d[n : 2 * n] * ux + d[2 * n : 3 * n] * uy + d[3 * n :] * uz)
    dgr = dgr.reshape(LAT.shape)

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    v = float(np.percentile(np.abs(dgr), 99)) or 1.0
    # gouraud, not flat: the disturbance is a smooth potential field and the
    # cell edges of a flat mesh read as structure that is not there
    c = ax.pcolormesh(
        lon,
        lat,
        dgr,
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
        shading="gouraud",
        rasterized=True,
    )
    # contours over the colour: a filled map alone is hard to read a VALUE off,
    # and the zero line is where the truth crosses the constant-density model
    lv = np.linspace(-v, v, 11)
    ax.contour(
        lon,
        lat,
        dgr,
        levels=lv[lv != 0],
        colors="k",
        linewidths=0.45,
        alpha=0.35,
        zorder=2,
    )
    ax.contour(
        lon, lat, dgr, levels=[0.0], colors="k", linewidths=1.3, alpha=0.8, zorder=3
    )
    # let the locator pick round ticks: the contour levels are a linspace and
    # reusing them put values like 0.2631 on the bar
    cb = fig.colorbar(c, ax=ax, pad=0.02, fraction=0.030)
    cb.set_label(r"$\Delta g_r$ [LU$^{-2}$]")

    # ONE MARKER SHAPE PER ANOMALY, named in the legend rather than written on
    # the map: six labels on a 360x180 field collide with each other and with
    # the poles, and boxing them to stay legible hides the field underneath.
    # Shape says WHICH anomaly, fill says its SIGN, so the two read independently
    # and the sign survives in greyscale through the (+)/(-) in the label.
    MK = ["o", "s", "^", "D", "P", "X", "<", ">"]
    Pa = np.asarray(positions, float)
    a_lon = np.degrees(np.arctan2(Pa[:, 1], Pa[:, 0]))
    a_lat = np.degrees(np.arcsin(Pa[:, 2] / np.linalg.norm(Pa, axis=1)))
    for j, (lo_, la_) in enumerate(zip(a_lon, a_lat)):
        lab = (
            None if names is None else (f"{names[j]}  ({'+' if beta[j] > 0 else '−'})")
        )
        ax.plot(
            lo_,
            la_,
            MK[j % len(MK)],
            ms=10,
            mec="k",
            mew=0.9,
            ls="none",
            zorder=6,
            color=COLOR[0] if beta[j] > 0 else COLOR[2],
            label=lab,
        )
    # cylinders as a wide hollow RING, so where one sits over an anomaly it
    # encircles that anomaly's marker instead of hiding it
    for m in marks or []:
        m = np.asarray(m, float)
        ax.plot(
            np.degrees(np.arctan2(m[1], m[0])),
            np.degrees(np.arcsin(m[2] / np.linalg.norm(m))),
            marker="o",
            mfc="none",
            mec=ACCENT,
            mew=2.0,
            ms=19,
            ls="none",
            zorder=5,
        )
    if marks:
        ax.plot(
            [],
            [],
            marker="o",
            mfc="none",
            mec=ACCENT,
            mew=2.0,
            ms=13,
            ls="none",
            label="CH cylinder",
        )
    # plate carree: one degree of longitude the same length as one of latitude,
    # so the anomaly footprints keep their true relative shape
    ax.set_aspect("equal")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlabel(r"Longitude  [$^\circ$]")
    ax.set_ylabel(r"Latitude  [$^\circ$]")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.set_axisbelow(False)
    n_e = len(Pa) + (1 if marks else 0)
    ax.legend(
        fontsize=8.5 * FONT_SCALE,
        ncol=min(4, n_e),
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01, 1.0, 0.2),
        mode="expand",
        borderaxespad=0.0,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    _save(fig, outdir, fname)
    return fig


def centred_rows_legend(ax, ncol, fontsize=7.5, y0=1.02, dy=0.075, **kw):
    """
    A legend above `ax` whose every ROW is centred.

    `ax.legend(ncol=n)` fills column-wise and leaves the remainder of the last
    row hanging off one side — seven entries in three columns came out 3/3/1,
    left-aligned, which reads as a mistake.  One legend PER ROW, each centred on
    the axes, makes any entry count look deliberate.  `ax.legend` replaces the
    previous legend, so earlier rows are re-attached with `add_artist`.
    """
    handles, labels = ax.get_legend_handles_labels()
    rows = [
        list(range(i, min(i + ncol, len(handles))))
        for i in range(0, len(handles), ncol)
    ]
    for r, idx in enumerate(rows[::-1]):  # bottom row first, so the top ends last
        lg = ax.legend(
            [handles[i] for i in idx],
            [labels[i] for i in idx],
            loc="lower center",
            ncol=len(idx),
            frameon=False,
            fontsize=fontsize * FONT_SCALE,
            bbox_to_anchor=(0.5, y0 + dy * r),
            **kw,
        )
        if r < len(rows) - 1:
            ax.add_artist(lg)


def case_hist(ax, series, xlabel, n_label):
    """
    The house error histogram, once, for however many cases there are.

    fig 2a(a) and fig 3(a) drew the same panel with two hard-coded series each;
    with three they would have drawn it with six, so it is written once here.
    Same bins (26, log-spaced over the pooled range), same alpha, same black
    log-normal overlays distinguished by LINE STYLE rather than colour — the
    colours are already spoken for by the histograms underneath.
    """
    allv = np.concatenate([series[k] for k in CASES])
    bins = np.logspace(np.log10(allv.min() * 0.8), np.log10(allv.max() * 1.2), 26)
    for k in CASES:
        ax.hist(
            series[k],
            bins=bins,
            color=CASE_COLOR[k],
            alpha=0.75,
            edgecolor="k",
            lw=0.5,
            label=k,
        )
    # the fitted PDF carries the centre AND the width, and its legend entry
    # reports both
    for k in CASES:
        lognormal_overlay(ax, series[k], bins, "k", ls=CASE_LS[k], name=k)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Truth Interiors  (of {n_label})  [-]")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    hist_legend(ax)


def reach_panels(
    maps,
    x,
    z,
    V,
    P,
    star,
    cyls,
    titles,
    levels,
    ceiling,
    cbar_label,
    path,
    cyl_lw=1.4,
):
    """
    One reach figure — SH only | CH only | SH + CH — for one family of maps,
    MASS or POSITION, shared by pt1 and pt2 so every reach figure reads alike.

    THREE PANELS in one file, the one place these modules depart from
    one-panel-per-file: the entire content is the COMPARISON between the
    fields, and splitting them would ask the reader to hold one in their head
    while looking at the next.  Drawn like `bouguer_map` — gouraud shading
    because sigma is a smooth function of position and flat cell edges read as
    structure that is not there, and contours over the colour because a filled
    map alone is hard to read a value off.  The contours sit at `levels`, the
    matching reach table's thresholds, so the figure and the table agree, and
    the colour stops at `ceiling`, the prior: a cell there is one the data do
    not inform at all.

    `maps` is {case: 2-D sigma}; `star` indexes the anomaly drawn as a star;
    `cyls` are outlined on the panels that use them; `titles` names each case.
    """
    fin = np.concatenate([maps[k][np.isfinite(maps[k])] for k in CASES])
    vmin, vmax = float(np.percentile(fin, 1)), float(min(fin.max(), ceiling))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.3), sharex=True, sharey=True)
    for axm, key in zip(axes, CASES):
        cmesh = axm.pcolormesh(
            x,
            z,
            maps[key],
            norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
            cmap="viridis_r",
            shading="gouraud",
            rasterized=True,
            zorder=2,
        )
        axm.contour(
            x,
            z,
            maps[key],
            levels=list(levels),
            colors="k",
            linewidths=0.45,
            alpha=0.35,
            zorder=3,
        )
        # The cylinders are drawn ONLY on the panels that use them.  Outlining
        # them over SH-only would say the global solution knows where the
        # spacecraft flew, and the whole point of that panel is that it does
        # not: its map is the same wherever the patches are put.
        if key != SH_ONLY:
            for c in cyls:
                draw_cylinder_2d(axm, c, lw=cyl_lw)
        for j, p_a in enumerate(P):
            axm.plot(
                p_a[0],
                p_a[2],
                "*" if j == star else "o",
                color="w",
                mec="k",
                mew=0.8,
                ms=14 if j == star else 9,
                zorder=5,
            )
        axm.set_title(titles[key], fontsize=10.5 * FONT_SCALE)
        axm.set_xlabel("x [LU]")
        # Limits from the BODY, not from the data: the cylinder outlines reach
        # far outside it (in pt2 they point six ways and would widen x by 40%),
        # and autoscaling to them shrinks the map — which is the content — to a
        # band across the middle.  Padded enough to show each patch's footprint
        # sitting off the surface, then clipped.
        _pad = 0.08 * max(np.ptp(V[:, 0]), np.ptp(V[:, 2]))
        axm.set_xlim(V[:, 0].min() - _pad, V[:, 0].max() + _pad)
        axm.set_ylim(V[:, 2].min() - _pad, V[:, 2].max() + _pad)
        axm.set_aspect("equal")
    axes[0].set_ylabel("z [LU]")
    # shrink: the panels are equal-aspect and do not fill the canvas height, so
    # an unshrunk bar stands taller than the maps it describes
    cbm = fig.colorbar(cmesh, ax=axes, pad=0.02, fraction=0.030, shrink=0.95)
    cbm.set_label(cbar_label, fontsize=10.5 * FONT_SCALE)
    fig.savefig(path, bbox_inches="tight")


def make_plots(res, outdir="Images"):
    os.makedirs(outdir, exist_ok=True)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    V, F, P = res["V"], res["F"], res["P"]
    cyl, names = res["cyl"], res["names"]
    tgt = res["target"]

    # ---- FIG 1: the interior model in 3-D -----------------------------------
    det = res["det"]
    ft = res["beta_true"]
    fig = plt.figure(figsize=(8.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, len(F) // 8000)
    pc = Poly3DCollection(
        V[F[::step]], alpha=0.18, facecolor="#9ecae1", edgecolor="0.55", linewidths=0.1
    )
    ax.add_collection3d(pc)
    draw_cylinder(ax, cyl, label="CH Cylinder")
    for i_a, (nm, p_a) in enumerate(zip(names, P)):
        mk = "*" if i_a == tgt else "o"
        sz = 240 if i_a == tgt else 120
        # colour by SIGN so an excess and a deficit are never confused
        ax.scatter(
            *p_a,
            c=COLOR[0] if ft[i_a] > 0 else COLOR[2],
            s=sz,
            marker=mk,
            edgecolor="k",
            depthshade=False,
        )
        # short name (first word); leading spaces nudge it right of the marker,
        # and the target sits lower so its label clears the cylinder above it
        dz = -0.05 if i_a == tgt else 0.0
        ax.text(
            p_a[0],
            p_a[1],
            p_a[2] + dz,
            f"    {nm.split()[0]}",
            fontsize=9.5 * FONT_SCALE,
        )
    ax.set_xlabel("x [LU]", labelpad=LPAD3D)
    ax.set_ylabel("y [LU]", labelpad=LPAD3D)
    ax.set_zlabel("z [LU]", labelpad=LPAD3D)
    set_axes_true_shape(ax, np.vstack([V, cylinder_hull(cyl)]))
    ax.scatter([], [], color=COLOR[0], label=r"Anomaly $\beta_j>0$")
    ax.scatter([], [], color=COLOR[2], label=r"Anomaly $\beta_j<0$")
    # the target is drawn as a star rather than a dot; nothing said so.  Same
    # colour as the marker itself, which is set by the SIGN of its own beta.
    ax.scatter(
        [],
        [],
        color=COLOR[0] if ft[tgt] > 0 else COLOR[2],
        marker="*",
        s=130,
        edgecolor="k",
        label=f"{names[tgt].split()[0]} (CH target)",
    )
    ax.legend(loc="upper left", fontsize=8 * FONT_SCALE)

    _save3d(fig, outdir, PREFIX + "fig1_geometry.pdf")

    # ---- FIG 1b: Bouguer map of the truth interior -------------------------
    # BOTH evaluation surfaces: they answer different questions and, for a body
    # this elongated, they disagree about which anomaly dominates — see the
    # `at` note in `bouguer_map`.
    for _at, _stem in (
        ("sphere", "fig1b_bouguer_sphere"),
        ("surface", "fig1c_bouguer_surface"),
    ):
        bouguer_map(
            ft,
            P,
            res["bulk"],
            V,
            outdir,
            PREFIX + _stem + ".pdf",
            names=[n.split()[0] for n in names],
            marks=[cyl.center],
            at=_at,
            F=F,
        )

    # ---- FIG 2a: EXPERIMENT 1 — mass recovery over TRUTH MASS FRACTIONS ----
    # Same three questions fig 3 asks of position, one file each: (a) how the
    # per-truth error is distributed, (b) which interiors were drawn, (c) for
    # one of them, does the analytic covariance actually describe the
    # estimator's scatter — (c) splitting again, one file per companion anomaly.
    tmc = res["truth_mc"]
    m_err = tmc["m_err"]
    bts, jt = tmc["betas"], tmc["m_tgt"]

    # (a) distribution of the per-truth mass-fraction error on the CH target
    fig, ax = plt.subplots(figsize=FS)
    case_hist(
        ax,
        m_err,
        r"Mass-fraction Error, One Fit per Truth Interior,  $\beta_0$  [-]",
        len(m_err[SH_ONLY]),
    )
    _save(fig, outdir, PREFIX + "fig2a_massfraction_hist.pdf")

    # (b) which interiors were drawn: |beta| uniform, sign random, per component
    fig, ax = plt.subplots(figsize=FS)
    rj = np.random.default_rng(5)
    for k in range(bts.shape[1]):
        xk = k + rj.uniform(-0.16, 0.16, len(bts))
        ax.scatter(
            xk,
            bts[:, k],
            s=26,
            color=COLOR[3],
            edgecolor="k",
            lw=0.35,
            alpha=0.8,
            zorder=3,
            label="Truth draws" if k == 0 else None,
        )
    ax.plot(range(len(ft)), ft, "k*", ms=17, zorder=6, label="Nominal truth")
    ax.plot(
        range(len(ft)),
        tmc["m_rep_beta"],
        "P",
        color=ACCENT,
        ms=11,
        zorder=5,
        label="Interior used for the ellipses",
    )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split()[0] for n in names], fontsize=10 * FONT_SCALE)
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_ylabel(r"Truth Mass Fraction  $\beta_j$  [-]")
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    # above the axes: the draws fill the panel top to bottom, so any in-axes
    # legend lands on data
    ax.legend(
        fontsize=9 * FONT_SCALE,
        ncol=3,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01, 1.0, 0.2),
        mode="expand",
        borderaxespad=0.0,
    )
    _save(fig, outdir, PREFIX + "fig2a_massfraction_truths.pdf")

    # (c) ONE interior: the estimate clouds against the PREDICTED covariance,
    # for both of the other anomalies in turn, ONE FIGURE EACH.  CH collapses
    # the target direction (x) only and leaves the lobes almost untouched, so
    # each ellipse is a vertical sliver — that anisotropy IS the result.
    from matplotlib.patches import Rectangle

    others = [k for k in range(len(names)) if k != jt]
    for row, ko in enumerate(others):
        fig, axc = plt.subplots(figsize=FS)
        # each panel now stands alone, so each carries the full legend
        first = True
        ix = np.ix_([jt, ko], [jt, ko])
        cA = tmc["m_cloud"][SH_ONLY][:, [jt, ko]]
        cB = tmc["m_cloud"][SH_CH][:, [jt, ko]]
        c_ch = tmc["m_cloud"][CH_ONLY][:, [jt, ko]]
        cov_sh, cov_shch = tmc["m_cov"][SH_ONLY][ix], tmc["m_cov"][SH_CH][ix]
        cov_ch = tmc["m_cov"][CH_ONLY][ix]
        mu_sh, mu_shch, mu_ch = cA.mean(axis=0), cB.mean(axis=0), c_ch.mean(axis=0)
        tru = tmc["m_rep_beta"][[jt, ko]]
        axc.scatter(
            cA[:, 0],
            cA[:, 1],
            s=36,
            color=COLOR[2],
            edgecolor="none",
            alpha=0.40,
            zorder=3,
            label="SH" if first else None,
        )
        # black outline: a blue ellipse on a blue cloud is unreadable
        cov_ellipse_nsig(
            axc,
            mu_sh,
            cov_sh,
            "k",
            lw=2.4,
            zorder=6,
            label=rf"SH Analytical {COV_NSIG:.0f}$\sigma$" if first else None,
        )
        # CH-only in the MAIN axes, not just the zoom.  Along the target (x) it
        # is as tight as SH+CH and is a sliver at this scale, but along the lobe
        # (y) it is the WIDEST of the three — the patch barely sees the lobes,
        # 1.2x SH's spread for the +x lobe and 2x for the -x one — and only the
        # parent can show that.  Under SH+CH, so the tighter cloud stays on top.
        axc.scatter(
            c_ch[:, 0],
            c_ch[:, 1],
            s=26,
            color=CASE_COLOR[CH_ONLY],
            edgecolor="none",
            alpha=0.45,
            zorder=3.5,
            label=CH_ONLY if first else None,
        )
        cov_ellipse_nsig(
            axc,
            mu_ch,
            cov_ch,
            "k",
            lw=2.2,
            ls="-.",
            zorder=6.5,
            label=(rf"{CH_ONLY} Analytical {COV_NSIG:.0f}$\sigma$" if first else None),
        )
        axc.scatter(
            cB[:, 0],
            cB[:, 1],
            s=22,
            color=COLOR[0],
            edgecolor="none",
            alpha=0.65,
            zorder=4,
            label="SH + CH" if first else None,
        )
        cov_ellipse_nsig(
            axc,
            mu_shch,
            cov_shch,
            "k",
            lw=2.4,
            ls=":",
            zorder=7,
            label=rf"SH + CH Analytical {COV_NSIG:.0f}$\sigma$" if first else None,
        )
        axc.plot(
            *tru,
            "*",
            color="w",
            ms=15,
            mec="k",
            mew=1.1,
            zorder=8,
            label="Truth" if first else None,
        )
        # per-axis limits, not equal aspect: the two axes carry different
        # components.  Each holds 3.6 sigma of the WIDEST model on that axis —
        # SH along the target, CH-only along the lobe — so no cloud runs off;
        # sized to SH alone, the CH-only cloud was cut off top and bottom.
        rx, ry = (
            3.6
            * max(
                max(np.sqrt(C[a, a]), abs(m[a] - tru[a]))
                for C, m in ((cov_sh, mu_sh), (cov_ch, mu_ch), (cov_shch, mu_shch))
            )
            for a in (0, 1)
        )
        axc.set_xlim(mu_sh[0] - rx, mu_sh[0] + rx)
        axc.set_ylim(mu_sh[1] - ry, mu_sh[1] + ry)
        axc.set_ylabel(
            rf"$\beta$  {names[ko].split()[0]}  [-]", fontsize=10 * FONT_SCALE
        )
        axc.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
        axc.yaxis.get_offset_text().set_fontsize(8)
        axc.grid(True, which="both", ls=":", alpha=0.45)
        axc.set_axisbelow(True)

        # SH+CH is ~20x tighter in the target direction, so at the SH scale its
        # predicted ellipse collapses to a line.  Inset at its OWN scale, one
        # magnification per axis: the gain is anisotropic, so a square zoom box
        # would be as wide as the panel and magnify nothing.  Sized to the
        # WIDER of the two models it shows — sized to SH+CH alone it cropped
        # the CH-only cloud — at ZOOM_HALF sigma, which leaves a margin round
        # both clouds instead of running them into the frame.  Never wider
        # than the parent, though: along the lobe axis CH-only is the widest
        # model there is, and the box would run off the panel.
        rbx, rby = (
            min(
                ZOOM_HALF
                * max(
                    np.sqrt(cov_shch[a, a]),
                    np.sqrt(cov_ch[a, a]),
                    abs(mu_shch[a] - tru[a]),
                ),
                0.92 * r_par,  # keeps the box's own edges inside the frame
            )
            for a, r_par in ((0, rx), (1, ry))
        )
        axin = axc.inset_axes([0.665, 0.07, 0.315, 0.40], facecolor="white")
        axin.set_zorder(10)
        axin.patch.set_alpha(1.0)
        axin.scatter(
            cB[:, 0],
            cB[:, 1],
            s=26,
            color=COLOR[0],
            edgecolor="none",
            alpha=0.45,
            zorder=2,
        )
        # CH-only in the zoom as well: its WIDTH along the target, which the
        # parent cannot resolve, set next to SH+CH's — how much of the joint
        # fit's tightness the patch alone already supplies.
        axin.scatter(
            c_ch[:, 0],
            c_ch[:, 1],
            s=26,
            color=CASE_COLOR[CH_ONLY],
            edgecolor="none",
            alpha=0.45,
            zorder=1,
        )
        # BLACK, like the other two predicted ellipses, with its own dash
        # pattern.  Colour on this figure means the CLOUD a model produced;
        # every predicted contour is black so the three read as one family, and
        # the STYLE tells them apart (SH solid, CH-only dash-dot, SH+CH dotted).
        cov_ellipse_nsig(
            axin,
            mu_ch,
            cov_ch,
            "k",
            lw=1.8,
            ls="-.",
            zorder=5,
        )
        cov_ellipse_nsig(axin, mu_shch, cov_shch, "k", lw=2.2, ls=":", zorder=6)
        axin.plot(*tru, "*", color="w", ms=12, mec="k", mew=0.9, zorder=8)
        axin.set_xlim(mu_shch[0] - rbx, mu_shch[0] + rbx)
        axin.set_ylim(mu_shch[1] - rby, mu_shch[1] + rby)
        # same wording as the position panel's inset.  White backing: the title
        # sits OUTSIDE the inset's own patch, over the parent SH scatter, and
        # points showing through the letters make it hard to read at print size.
        axin.set_title(
            "CH / SH + CH Zoom",
            fontsize=7 * FONT_SCALE,
            color=COLOR[0],
            pad=4,  # clears the inset's own top spine
            bbox=dict(fc="white", ec="none", pad=1.5),
        )
        axin.tick_params(labelsize=5.5 * FONT_SCALE, pad=1)
        # 3 ticks collide at this width, so 2 -- and the TOP y label is pruned:
        # it sits at the frame's top-left corner, exactly where the title is, and
        # once the component is negative the minus sign makes it wide enough to
        # run into the title text.
        axin.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
        axin.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2, prune="upper"))
        # plain, not sci: an inset's offset text is drawn OUTSIDE its frame,
        # so a "x1e-2" would float onto the parent axes
        axin.ticklabel_format(style="plain", useOffset=False)
        for sp_ in axin.spines.values():
            sp_.set_edgecolor(COLOR[0])
        # dashed grey box, not a ring: a circle in the SH+CH colour would read
        # as that case's covariance ellipse
        axc.add_patch(
            Rectangle(
                (mu_shch[0] - rbx, mu_shch[1] - rby),
                2 * rbx,
                2 * rby,
                fill=False,
                ec="0.35",
                ls="--",
                lw=1.1,
                zorder=9,
            )
        )
        # above the axes, not inside: an opaque box here would sit on the SH
        # cloud, the widest thing in the panel.  Rows CENTRED — each cloud next
        # to its own contour, 4 over 3 — and left-aligned that reads as a mistake.
        # y0 clears the 1e-2 offset text on the top spine.
        centred_rows_legend(
            axc, 4, y0=1.06, dy=0.075, handletextpad=0.4, columnspacing=1.0
        )
        axc.set_xlabel(
            rf"$\beta$  {names[jt].split()[0]}  [-]", fontsize=10 * FONT_SCALE
        )
        _save(fig, outdir, f"{PREFIX}fig2a_massfraction_cov{row + 1}.pdf")

    # ---- FIG 2b: estimator performance vs anomaly size ---------------------
    # MAIN AXES — performance: the MC RMS error as a PERCENT OF THE ANOMALY
    # being estimated.  The RMS error itself is flat at sigma (linear estimator,
    # truth-independent weights), so the relative error is sigma/beta_0, a
    # slope -1 line.  Read horizontally at any accuracy level it gives the
    # smallest anomaly each model can pin down that well, and the SH/SH+CH gap
    # is the same everywhere.
    # INSET — validation: the same MC RMS divided by the ANALYTIC sigma.  Those
    # two are computed independently (draws vs (A^T W A)^-1), so the ratio is a
    # real consistency test and not an identity; it should sit at 100%.
    # WIDER than the other panels.  The legend carries two entries per model and
    # there are three models now: at 7.6 in the six entries ran off the right
    # edge of the axes whatever the column count, and the threshold annotation
    # collided with the gain bar underneath it.
    fig, ax = plt.subplots(figsize=(10.4, 5.9))
    mug, acc = det["mu_grid"], det["acc"]
    rel = {k: 100.0 * det["rms"][k] / mug for k in CASES}

    ax.axhspan(acc, 1e6, color="0.5", alpha=0.15, lw=0, zorder=1)
    ax.axhline(acc, color="0.3", ls="-", lw=1.4, zorder=4)
    ax.text(
        mug.min() * 0.80,
        acc * 1.14,
        rf"Identifiability Threshold: {acc:.0f}" + (r"$\%$" if USE_TEX else "%"),
        fontsize=9 * FONT_SCALE,
        ha="left",
        va="bottom",
        color="0.15",
        zorder=6,
    )

    for k in CASES:
        ax.plot(
            mug,
            100.0 * det["sd"][k] / mug,
            color=CASE_COLOR[k],
            lw=1.4,
            ls="--",
            alpha=0.85,
            zorder=3,
            label=rf"{k}: $100\,\sigma/\beta_0$",
        )
        ax.plot(
            mug,
            rel[k],
            CASE_MARKER[k],
            color=CASE_COLOR[k],
            ms=6,
            mec="k",
            mew=0.5,
            ls="none",
            zorder=5,
            label=rf"{k}: MC RMS Error",
        )

    # the gain, read horizontally at the accuracy threshold itself.  The bar
    # spans SH-only to SH+CH, the two ends of the argument; CH-only gets a tick
    # of its own so its detection limit can be read off the same line.
    xs = {k: 100.0 * det["sd"][k] / acc for k in CASES}
    ax.plot([xs[SH_CH], xs[SH_ONLY]], [acc, acc], color="k", lw=1.8, zorder=6)
    for k in CASES:
        ax.plot([xs[k], xs[k]], [acc / 1.7, acc * 1.7], color="k", lw=1.8, zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(mug.min() * 0.7, mug.max() * 1.5)
    ax.set_ylim(min(rel[k].min() for k in CASES) * 0.45, 2.0e4)
    # NOT "true anomaly": that is the orbital element (the angle from
    # periapsis), and this is a mass fraction — the collision would be
    # actively misleading in an asteroid-gravity paper.
    ax.set_xlabel(r"Anomaly Mass Fraction $\beta_0$ (truth)  [-]")
    ax.set_ylabel(r"MC RMS Error / $\beta_0$  " + (r"[$\%$]" if USE_TEX else "[%]"))
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    # three models x two entries each: three columns of two, which is symmetric
    # and, on the wider canvas, fits inside the axes
    ax.legend(fontsize=8.5 * FONT_SCALE, loc="lower left", framealpha=0.93, ncol=3)
    _save(fig, outdir, PREFIX + "fig2b_detection.pdf")

    # ---- FIG 3: EXPERIMENT 2 — position recovery over TRUTH POSITIONS -------
    # One file each: (a) how the per-truth error is distributed, (b) where the
    # truths were drawn, (c) the estimator's own scatter for one of them.
    tmc = res["truth_mc"]
    pe = tmc["err"]
    pos, cylc = tmc["pos"], res["cyl"].center

    # (a) distribution of the per-truth RMS position error
    fig, ax = plt.subplots(figsize=FS)
    case_hist(
        ax,
        pe,
        "Position Error, One Fit per Truth Interior [LU]",
        len(pe[SH_ONLY]),
    )
    _save(fig, outdir, PREFIX + "fig3_position_hist.pdf")

    # (b) where the truth anomalies were drawn
    fig, ax = plt.subplots(figsize=FS_WIDE)
    draw_silhouette(ax, V, F, 0, 2)
    ax.scatter(
        pos[:, 0],
        pos[:, 2],
        s=34,
        color=COLOR[3],
        edgecolor="k",
        lw=0.4,
        alpha=0.85,
        zorder=5,
        label="Truth draws",
    )
    ax.plot(P[tgt][0], P[tgt][2], "k*", ms=17, zorder=7, label="Nominal site")
    ax.plot(
        cylc[0],
        cylc[2],
        marker="v",
        color=ACCENT,
        ms=12,
        zorder=6,
        label="CH cylinder",
    )
    ax.set_xlabel("x [LU]")
    ax.set_ylabel("z [LU]")
    ax.set_aspect("equal")
    ax.legend(fontsize=9 * FONT_SCALE, loc="lower right")
    _save(fig, outdir, PREFIX + "fig3_position_truths.pdf")

    # (c) ONE truth: every anomaly, every plane, every model.
    # A 3x3 GRID — rows are the anomalies, columns the coordinate planes.  This
    # was a single x-z panel for the shallow anomaly alone, which showed one
    # ninth of the covariance the estimator actually has: a 3-D position has
    # three planes and the experiment has three anomalies.  The grid shows the
    # ANISOTROPY (a patch localizes far better ACROSS its line of sight than
    # ALONG it, which only the x-y and y-z cells reveal) and the fall-off with
    # distance from the cylinder, in one read.
    #
    # Each cell is scaled to its WIDEST model so nothing is cut off, and carries
    # a zoom inset at the scale of its tightest one wherever those two differ by
    # more than 5x.  Which model is widest flips between rows — SH where the
    # patch overlooks the anomaly, CH-only where it does not — which is itself
    # the result, so the panel never fixes one model as the reference.
    from matplotlib.patches import Rectangle

    planes = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    n_a = len(P)
    fig, axs = plt.subplots(n_a, 3, figsize=(11.0, 3.5 * n_a))
    axs = np.atleast_2d(axs)
    for ja in range(n_a):
        tru3 = tmc["rep_truths"][ja]
        for ic, (ia, ib, la, lb) in enumerate(planes):
            axg = axs[ja, ic]
            ixp = np.ix_([ia, ib], [ia, ib])
            # SCALE: the main axes hold the LARGEST model, so nothing is cut
            # off, and an INSET re-draws the same cell at the scale of the
            # SMALLEST — the two differ by ~20x in every cell (which way round
            # depends on whether a patch overlooks that anomaly), so one scale
            # cannot show both.  This is the same main-plus-zoom construction
            # the single-panel version used, repeated per cell.
            sig_of = {
                c: max(
                    np.sqrt(tmc["cov"][c][ja][ia, ia]),
                    np.sqrt(tmc["cov"][c][ja][ib, ib]),
                )
                for c in CASES
            }
            big = max(sig_of, key=sig_of.get)
            small = min(sig_of, key=sig_of.get)
            # ALL THREE clouds, not just the widest.  Drawing one meant two of
            # the three models appeared in a cell as a bare outline with no
            # scatter behind it, and the reader could not see which estimator
            # actually produced which spread.  Widest first, so the tight ones
            # land on top instead of being buried.
            for cse in sorted(CASES, key=lambda c: -sig_of[c]):
                cl = tmc["cloud"][cse][ja][:, [ia, ib]]
                axg.scatter(
                    cl[:, 0],
                    cl[:, 1],
                    s=20,
                    color=CASE_COLOR[cse],
                    edgecolor="none",
                    alpha=0.40,
                    zorder=2 + CASES.index(cse),
                )
            cloud_sh = tmc["cloud"][big][ja][:, [ia, ib]]
            # every PREDICTED ellipse in black, the STYLE naming the model —
            # the same convention fig 2a's inset uses
            for case, ls_ in ((c, LS_OF_CASE[c]) for c in CASES):
                cov_ellipse_nsig(
                    axg,
                    tmc["cloud"][case][ja][:, [ia, ib]].mean(axis=0),
                    tmc["cov"][case][ja][ixp],
                    "k",
                    lw=1.8,
                    ls=ls_,
                    zorder=6,
                    label=(
                        f"{case} {COV_NSIG:.0f}$\\sigma$"
                        if (ja == 0 and ic == 0)
                        else None
                    ),
                )
            axg.plot(
                tru3[ia],
                tru3[ib],
                "*",
                color="w",
                ms=13,
                mec="k",
                mew=1.0,
                zorder=8,
                label="Truth" if (ja == 0 and ic == 0) else None,
            )
            c_ = tmc["cloud"][big][ja][:, [ia, ib]].mean(axis=0)
            r_ = 3.9 * sig_of[big]  # the whole widest cloud, with a margin
            axg.set_xlim(c_[0] - r_, c_[0] + r_)
            axg.set_ylim(c_[1] - r_, c_[1] + r_)
            axg.set_aspect("equal")
            axg.set_xlabel(f"{la} [LU]", fontsize=9 * FONT_SCALE)
            axg.set_ylabel(f"{lb} [LU]", fontsize=9 * FONT_SCALE)
            axg.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
            axg.tick_params(labelsize=7.5 * FONT_SCALE)
            axg.xaxis.get_offset_text().set_fontsize(7)
            axg.yaxis.get_offset_text().set_fontsize(7)
            axg.grid(True, which="both", ls=":", alpha=0.45)
            axg.set_axisbelow(True)
            if ic == 1:
                axg.set_title(names[ja], fontsize=10.5 * FONT_SCALE)

            # THE ZOOM, and only when it would actually zoom.  In cells where
            # the three models are within a few x of each other the inset
            # magnified nothing and just covered the data, so it is drawn only
            # past 5x.  It carries the models that are NEAR the smallest — the
            # big one is already legible in the main axes, and drawing it here
            # too put a huge arc across the inset that read as a glitch.
            ratio = sig_of[big] / max(sig_of[small], 1e-30)
            near = [c for c in CASES if sig_of[c] <= 3.0 * sig_of[small]]
            if ratio >= 5.0:
                azi = axg.inset_axes([0.60, 0.05, 0.36, 0.36], facecolor="white")
                azi.set_zorder(10)
                azi.patch.set_alpha(1.0)
                cs_ = tmc["cloud"][small][ja][:, [ia, ib]]
                # ZOOM_HALF sigma of the widest model the inset shows, as in
                # fig 2a — at the tightest one's scale the clouds ran into
                # the frame on every side
                rz = ZOOM_HALF * max(sig_of[c] for c in near)
                cz = cs_.mean(axis=0)
                for cse in sorted(near, key=lambda c: -sig_of[c]):
                    cl = tmc["cloud"][cse][ja][:, [ia, ib]]
                    azi.scatter(
                        cl[:, 0],
                        cl[:, 1],
                        s=14,
                        color=CASE_COLOR[cse],
                        edgecolor="none",
                        alpha=0.45,
                        zorder=2 + CASES.index(cse),
                    )
                for case in near:
                    cov_ellipse_nsig(
                        azi,
                        tmc["cloud"][case][ja][:, [ia, ib]].mean(axis=0),
                        tmc["cov"][case][ja][ixp],
                        "k",
                        lw=1.4,
                        ls=LS_OF_CASE[case],
                        zorder=6,
                    )
                azi.plot(
                    tru3[ia], tru3[ib], "*", color="w", ms=8, mec="k", mew=0.8, zorder=8
                )
                azi.set_xlim(cz[0] - rz, cz[0] + rz)
                azi.set_ylim(cz[1] - rz, cz[1] + rz)
                azi.set_aspect("equal")
                # Fig 2a's inset conventions, so the two covariance figures
                # read the same way: two pruned ticks per axis rather than
                # none, PLAIN tick format (an inset's offset text is drawn
                # outside its frame and would float onto the parent), spines in
                # the zoomed model's colour, and a white-backed title carrying
                # the model and the magnification.
                azi.tick_params(labelsize=5.5 * FONT_SCALE, pad=1)
                azi.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
                # Y ticks chosen from the MIDDLE of the range only.  Left to a
                # locator over the whole range, the topmost round value sat at
                # the very top of the frame and its label landed under the inset
                # title in the tall x-z and y-z cells, whatever the pruning.
                # Restricting the candidates to the central 60% keeps every
                # label clear of the title while still giving ROUND values —
                # placing ticks at fixed fractions did the former but printed
                # numbers like 0.00211.
                _y0, _y1 = azi.get_ylim()
                _lo, _hi = _y0 + 0.2 * (_y1 - _y0), _y1 - 0.2 * (_y1 - _y0)
                _tk = [
                    t
                    for t in mpl.ticker.MaxNLocator(nbins=2).tick_values(_lo, _hi)
                    if _lo <= t <= _hi
                ]
                azi.set_yticks(_tk)
                _yf = mpl.ticker.ScalarFormatter(useOffset=False)
                _yf.set_scientific(False)  # no offset text floating outside
                azi.yaxis.set_major_formatter(_yf)
                azi.ticklabel_format(axis="x", style="plain", useOffset=False)
                for sp_ in azi.spines.values():
                    sp_.set_edgecolor(CASE_COLOR[small])
                azi.set_title(
                    f"{small} zoom  ({r_ / rz:.0f}" + r"$\times$)",
                    fontsize=6.5 * FONT_SCALE,
                    color=CASE_COLOR[small],
                    pad=4,  # clears the inset's own top spine
                    bbox=dict(fc="white", ec="none", pad=1.5),
                )
                # the zoom REGION on the parent, as fig 2a marks it: a dashed
                # grey box, never a ring — a ring in a model's colour reads as
                # that model's covariance ellipse
                axg.add_patch(
                    Rectangle(
                        (cz[0] - rz, cz[1] - rz),
                        2 * rz,
                        2 * rz,
                        fill=False,
                        ec="0.35",
                        ls="--",
                        lw=1.0,
                        zorder=9,
                    )
                )
    # Which cloud a cell shows is a per-cell decision, so no single axes can
    # carry all three into the key; these proxies do.  The zoom inset is NOT in
    # the legend — it is labelled in its own corner, with the magnification.
    ax0 = axs[0, 0]
    for c in CASES:
        ax0.scatter(
            [],
            [],
            s=20,
            color=CASE_COLOR[c],
            edgecolor="none",
            alpha=0.6,
            label=f"{c}",
        )
    handles, labels = ax0.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=9 * FONT_SCALE,
        frameon=False,
        bbox_to_anchor=(0.5, 0.008),
    )
    fig.savefig(
        os.path.join(outdir, PREFIX + "fig3_position_cov.pdf"), bbox_inches="tight"
    )

    # ---- FIG 3b: REACH — where each model can see an anomaly at all --------
    # Two figures from one construction (`reach_panels`): the 1σ on a test
    # anomaly's MASS, and on its POSITION — can a model weigh an anomaly there,
    # and can it place one.
    rm = res["reach"]
    titles = {k: k for k in CASES}
    reach_panels(
        rm["sigma"],
        rm["x"],
        rm["z"],
        V,
        P,
        res["target"],
        [res["cyl"]],
        titles,
        REACH_LEVELS_MASS,
        PRIOR_SIGMA,
        r"Mass-fraction 1$\sigma$  [-]",
        os.path.join(outdir, PREFIX + "fig3b_reach_map.pdf"),
    )
    reach_panels(
        rm["sigma_pos"],
        rm["x"],
        rm["z"],
        V,
        P,
        res["target"],
        [res["cyl"]],
        titles,
        REACH_LEVELS_POS,
        rm["pos_prior"],
        r"Position 1$\sigma$  [LU]",
        os.path.join(outdir, PREFIX + "fig3b_reach_map_position.pdf"),
    )

    # ---- FIG 4: residual power spectrum, before and after the fit ----------
    # Per-degree (SH) / per-radial-mode (CH) RMS of the WHITENED residual,
    #     PRE-fit  = measured − homogeneous model   (= the discrepancy + noise)
    #     POST-fit = measured − (homogeneous + A β̂)
    # Whitened by σ_i, so the y-axis is dimensionless, a linear scale is
    # meaningful, and "1" is exactly the noise floor: a post-fit spectrum
    # sitting on 1 says the fit has consumed the signal and σ is the right size.
    sp = res["spectra"]
    Lmin, Lmax = sp["Lmin"], sp["Lmax"]
    n_m, n_n = sp["ch_modes"]

    def _groups(key):
        """(x values, list of index arrays) for the spectrum grouping."""
        if key == "sh":  # group by degree n
            xs, gr, acc = [], [], 0
            for n in range(Lmin, Lmax + 1):
                k = 2 * (n + 1)
                xs.append(n)
                gr.append(np.arange(acc, acc + k))
                acc += k
            return np.array(xs), gr
        pair = np.arange(2 * n_m * n_n) // 2  # group by azimuthal order m
        azi = pair // n_n
        xs = np.arange(n_m)
        return xs, [np.where(azi == m)[0] for m in xs]

    for key, xlab in [
        ("sh", "SH degree $n$  [-]"),
        ("ch", "CH azimuthal order $m$  [-]"),
    ]:
        fig, ax = plt.subplots(figsize=FS)
        d = sp[key]
        pre = d["data"]  # measured − homogeneous
        post = d["data"] - d["model"]  # measured − fitted
        xs, gr = _groups(key)
        rms = lambda v: np.array([np.sqrt(np.mean(v[g] ** 2)) for g in gr])
        # absolute residuals in the coefficients' own units, so the reference is
        # σ itself — one curve, labelled 1σ — rather than an abstract band about 1.
        # CAVEAT: reading the ratio red/dashed off the plot is only approximate.
        # σ_i varies within a group (od_sigma is per-coefficient), and
        # RMS|r| / RMS(σ)  ≠  RMS(r/σ) unless σ is constant across the group.
        # Worst case here is SH degree 3: the plot reads 0.30, the actual
        # whitened statistic is 0.67.  The EXACT consistency numbers are the
        # discrepancy-to-noise and post-fit residual rows of TABLE 4, which
        # are properly whitened.
        y_pre, y_post, y_sig = rms(np.abs(pre)), rms(np.abs(post)), rms(d["sigma"])
        # NOISE as three nested bands — 1σ, 2σ, 3σ — rather than one 1σ curve,
        # which invited only "on it or off it"; the bands say how many sigma
        # every point sits at.  `sigma_bands` also sets the x-limits.
        sigma_bands(ax, xs, y_sig)
        ax.plot(
            xs,
            y_pre,
            "-o",
            color=COLOR[2],
            lw=1.8,
            ms=9,
            mec="k",
            mew=0.7,
            zorder=5,
            label=r"PRE-fit: measured $-$ homogeneous",
        )
        ax.plot(
            xs,
            y_post,
            "-s",
            color=COLOR[0],
            lw=1.8,
            ms=8,
            mec="k",
            mew=0.7,
            zorder=6,
            label=r"POST-fit: measured $-$ (homog. $+$ A$\hat\beta$)",
        )

        ax.set_xticks(xs)
        ax.set_xlabel(xlab)
        # NOT "residual": the pre-fit curve is the discrepancy between the
        # measured field and the homogeneous model, the bands are the noise, and
        # only the post-fit curve is a residual proper.  What the three share is
        # that each is the RMS of a coefficient group — a power spectrum.
        ax.set_ylabel("RMS Coefficient Power  [-]")
        ax.set_yscale("log")
        # the bands reach down to zero, which a log axis cannot show, so the
        # floor still comes from the CURVES
        ax.set_ylim(0.5 * min(y_post.min(), y_sig.min()), 2.5 * y_pre.max())
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.45)
        ax.set_axisbelow(True)

        # legend in a reserved strip under the axes — an in-axes legend here
        # covered the degree-3 peak.  One column, not three: a single-panel
        # canvas cannot fit these labels side by side.
        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout(rect=[0, 0.13, 1, 1])
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=1,
            fontsize=9.5 * FONT_SCALE,
            frameon=False,
            bbox_to_anchor=(0.5, 0.012),
        )
        fig.savefig(
            os.path.join(outdir, f"{PREFIX}fig4_coefficients_{key}.pdf"),
            bbox_inches="tight",
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — HOW GOOD WOULD THE GLOBAL FIELD HAVE TO BE?  (L_SH SWEEP)
# ═══════════════════════════════════════════════════════════════════════════
# Everything above compares one near-surface patch against a global field
# truncated at ONE degree.  That is the comparison a mission actually faces,
# but it invites the obvious objection: the patch is winning only because the
# global solution was cut short.  The SH series is complete and the CH patch is
# finite, so more degrees MUST close the gap.  The question is whether it closes
# at a degree anyone will ever have.
#
# The sweep answers it directly: hold the cylinder, the anomalies and the
# precision fixed, and walk L_SH up.  Everything here is ANALYTIC — the Fisher
# covariances of TABLE 1 and TABLE 4, not a Monte-Carlo per degree — so the
# whole curve costs about what one extra figure costs.
#
# THE NOISE RULE.  `alphas` is a tuple, so a second rule can be added to check
# how far the answer leans on it — alpha = 0, every degree known to the same
# relative precision, is the natural stress test and the best case for SH, and
# the conclusion survives it.  The default sweeps ONE rule, the experiment's
# own, so the sweep reads as an extension of the tables rather than a separate
# study run under its own conditions.
#
#   alpha = 0.10  `od_sigma`'s default: sigma grows as exp(alpha*(n - 2)), a
#                 stand-in for that loss of short-wavelength sensitivity.  This
#                 is the rule the main experiment runs on, so the sweep passes
#                 through TABLE 4 at L = Lmax_sh.
#
# The rules are applied to BOTH observables, which is what `eps` means in
# `run_experiment` ("the same relative precision on both").  CAVEAT: `od_sigma`
# infers a degree from the SH packing, so on the CH vector it is reading a
# degree that does not exist — the CH coefficients are indexed by (m, n) radial
# modes, not by spherical-harmonic degree.  It still penalizes the CH block, but
# through an ordering that carries no physical meaning.  `ch_alpha` is exposed
# so that reading can be pinned independently; `ch_alpha=0` is the defensible
# choice, and the sweep prints which convention is in force.


def _sh_count(L, Lmin=2):
    """How many packed coefficients degrees Lmin..L occupy."""
    return sum(2 * (n + 1) for n in range(Lmin, L + 1))


def _first_reach(L_values, curve, level):
    """
    Smallest degree in `L_values` at which `curve` first drops to `level`, by
    linear interpolation in log(sigma) between the bracketing degrees; None if
    it never gets there.  This is the number the sweep exists to produce: the
    degree a GLOBAL solution needs to match what one LOCAL patch already gives.
    """
    c = np.asarray(curve, float)
    for i in range(len(c)):
        if c[i] <= level:
            if i == 0:
                return float(L_values[0])
            y0, y1 = np.log(c[i - 1]), np.log(c[i])
            t = (np.log(level) - y0) / (y1 - y0)
            return float(L_values[i - 1] + t * (L_values[i] - L_values[i - 1]))
    return None


def sweep_lmax_sh(res, L_values=None, alphas=(0.10,), ch_alpha=None, verbose=True):
    """
    Mass-fraction and position 1-sigma for every case and EVERY anomaly as L_SH
    walks up, at each noise rule in `alphas`, with their posterior/prior ratios
    R and the R >= PRIOR_RATIO_THRESHOLD flags — pt2's sweep, for this
    experiment.  Reads its geometry and precision out of `res` so the sweep and
    the tables describe one experiment.
    """
    P, beta_true, bulk = res["P"], res["beta_true"], res["bulk"]
    cyl, obs, ch_modes = res["cyl"], res["obs"], res["ch_modes"]
    Rref, target, eps = res["Rb"], res["target"], res["eps"]
    names = res["names"]
    # 2..14 by default: `Bulk.stokes` is a tetrahedral quadrature whose Gauss
    # order grows with Lmax (20 s at 14, 35 s at 16, two minutes at 22), and by
    # degree 12 the SH-only curve has been flat for three degrees — the plateau
    # is the result, and paying another minute to extend it does not sharpen it.
    L_values = list(range(2, 15)) if L_values is None else list(L_values)
    Ltop = max(L_values)

    # ONE quadrature at the top degree, then SLICED.  The packing runs
    # [C20,S20,C21,S21,C22,S22,C30,...], strictly degree by degree, so the first
    # `_sh_count(L)` rows ARE the degree-L truncation.  `Bulk.stokes` picks its
    # Gauss order from Lmax, so a sliced high-order result and a directly
    # computed low-order one differ at the 1e-11 level — eleven orders below the
    # 2% noise this whole study is about.  The alternative is re-integrating
    # 14744 faces once per degree: two minutes at L = 22 against thirty seconds
    # once, and the sweep would be dominated by an exactness nobody can measure.
    cs_top = bulk.stokes(2, Ltop, Rref)
    A_sh_top = A_stokes(P, 2, Ltop, Rref) - cs_top[:, None]

    # The CH block does not depend on L_SH at all — that is the point of the
    # comparison — so it is built once and reused at every degree.  CH-only is
    # therefore a FLAT LINE across the sweep, and drawing it is worth the ink:
    # it is the level the global field is trying to reach, and the degree at
    # which SH-only crosses it (if it ever does) is the sweep's headline.
    pinvPhi = ch_pinv_for(cyl, obs, ch_modes)
    A_ch = A_ch_contrast(P, bulk, obs, cyl, ch_modes)
    y_ch_tot = ch_coefficients_total(beta_true, P, bulk, obs, pinvPhi)

    out = {}
    for a in alphas:
        a_ch = a if ch_alpha is None else ch_alpha
        sig_ch = od_sigma(y_ch_tot, eps, alpha=a_ch)
        mass = {k: [] for k in CASES}
        mass_prior = {k: [] for k in CASES}
        pos = {k: [] for k in CASES}
        pos_prior = {k: [] for k in CASES}
        ncf = []
        for L in L_values:
            nk = _sh_count(L)
            A_sh = A_sh_top[:nk]
            # sigma is rebuilt from the TRUNCATED measured vector, not sliced
            # from a top-degree sigma: the noise floor is a fraction of the RMS
            # of what was actually measured, and a solution that stops at L
            # never saw the degrees above it.
            sig_sh = od_sigma(cs_top[:nk] + A_sh @ beta_true, eps, alpha=a)
            blk = case_blocks(A_sh, sig_sh, A_ch, sig_ch)
            # EVERY anomaly's position, each with the others held — TABLE 2's
            # linearization, applied beyond the target — seeded with
            # POS_PRIOR_SIGMA so that R is defined where the data see nothing
            C_pos = [
                position_covariance(
                    j,
                    P,
                    obs,
                    cyl,
                    ch_modes,
                    sig_sh,
                    sig_ch,
                    L,
                    Rref,
                    pinvPhi,
                    prior_sigma=POS_PRIOR_SIGMA,
                )
                for j in range(len(P))
            ]
            for k in CASES:
                m_sig = posterior_sigma(
                    mass_fraction_covariance(blk[k], prior_sigma=PRIOR_SIGMA)
                )
                mass[k].append(m_sig)
                mass_prior[k].append(m_sig / PRIOR_SIGMA >= PRIOR_RATIO_THRESHOLD)
                p_sig = np.array([posterior_rms(c[k]) for c in C_pos])
                pos[k].append(p_sig)
                pos_prior[k].append(p_sig / POS_PRIOR_SIGMA >= PRIOR_RATIO_THRESHOLD)
            ncf.append(nk)
        out[a] = dict(
            mass={k: np.array(v) for k, v in mass.items()},  # (n_L, n_anom)
            # Marginal uncertainty reduction: R near one is flagged "prior".
            # Reuse the stored sigmas; no additional covariance evaluations.
            mass_prior_ratio={k: np.array(v) / PRIOR_SIGMA for k, v in mass.items()},
            mass_prior={k: np.array(v) for k, v in mass_prior.items()},
            pos={k: np.array(v) for k, v in pos.items()},  # (n_L, n_anom)
            pos_prior_ratio={k: np.array(v) / POS_PRIOR_SIGMA for k, v in pos.items()},
            pos_prior={k: np.array(v) for k, v in pos_prior.items()},
            n_coef=np.array(ncf),
            ch_alpha=a_ch,
        )
    return dict(
        L=np.array(L_values),
        alphas=list(alphas),
        by_alpha=out,
        target=target,
        names=names,
        eps=eps,
        ch_modes=ch_modes,
        L_nominal=res["Lmax_sh"],
        prior_ratio_threshold=PRIOR_RATIO_THRESHOLD,
    )


def sweep_report(sw):
    """The sweep's numbers, in the same voice as `results_report`."""
    L, tgt, names = sw["L"], sw["target"], sw["names"]
    Ln = sw["L_nominal"]
    i_nom = int(np.argmin(np.abs(L - Ln)))
    print(f"\n{SEP}")
    print("  TABLE 6 — L_SH sweep: does the near-surface patch survive a")
    print("            better global field?")
    print(SEP)
    print(
        f"  target anomaly: {names[tgt]} | CH modes {sw['ch_modes']} | "
        f"eps = {sw['eps']}"
    )
    print(
        f"  Prior-dominated: R = posterior sigma / prior sigma >= "
        f"{sw['prior_ratio_threshold']:.2f} (position uses the RMS sigma)."
    )
    for a in sw["alphas"]:
        d = sw["by_alpha"][a]
        rule = (
            "flat relative precision, the BEST CASE FOR SH"
            if a == 0.0
            else f"sigma ~ exp({a}*(n-2)), the main experiment's rule"
        )
        print(f"\n  alpha = {a}  ({rule})")
        if d["ch_alpha"] != a:
            print(f"    [CH block held at alpha = {d['ch_alpha']}]")
        print(
            f"  {'L':>3} {'n_coef':>7} "
            + " ".join(f"{'sig_b ' + k:>13}" for k in CASES)
            + " | "
            + " ".join(f"{'pos ' + k:>13}" for k in CASES)
        )
        ms = {k: d["mass"][k][:, tgt] for k in CASES}
        ps = {k: d["pos"][k][:, tgt] for k in CASES}
        bad = d["mass_prior"][SH_ONLY][:, tgt] | d["pos_prior"][SH_ONLY][:, tgt]
        for i, Li in enumerate(L):
            mark = "  <- main run" if i == i_nom else ""
            if bad[i]:
                mark = "  (*)" + mark
            print(
                f"  {Li:3d} {d['n_coef'][i]:7d} "
                + " ".join(f"{ms[k][i]:13.2e}" for k in CASES)
                + " | "
                + " ".join(f"{ps[k][i]:13.2e}" for k in CASES)
                + mark
            )
        if bad.any():
            print(
                "  (*) The SH-only mass or position sigma remains near its prior "
                "sigma;\n      a gain against it uses a prior-dominated baseline."
            )
        # The headline, and the reason CH-only earns a column: it is FLAT in L,
        # so it is a fixed bar the global field is trying to clear.  Two
        # questions, then: what degree does SH need to match the JOINT fit, and
        # what degree does it need just to match the PATCH ALONE?
        for lbl, cs, tgts in (
            ("mass fraction", ms[SH_ONLY], (ms[SH_CH][i_nom], ms[CH_ONLY][i_nom])),
            ("position", ps[SH_ONLY], (ps[SH_CH][i_nom], ps[CH_ONLY][i_nom])),
        ):
            for what, tv in zip(("SH+CH", "CH"), tgts):
                hit = _first_reach(L, cs, tv)
                if hit is None:
                    print(
                        f"  => {lbl}: SH-only does not reach the L={Ln} {what} "
                        f"value ({tv:.2e}) at any degree up to {L[-1]}"
                    )
                else:
                    print(
                        f"  => {lbl}: SH-only would need degree {hit:.1f} to "
                        f"match the L={Ln} {what} value ({tv:.2e})"
                    )
        print(
            f"  => SH/SH+CH gain at L={L[0]}: {ms[SH_ONLY][0]/ms[SH_CH][0]:.1f}x "
            f"mass, {ps[SH_ONLY][0]/ps[SH_CH][0]:.1f}x position;  at L={L[-1]}: "
            f"{ms[SH_ONLY][-1]/ms[SH_CH][-1]:.1f}x mass, "
            f"{ps[SH_ONLY][-1]/ps[SH_CH][-1]:.1f}x position"
        )
    # per-anomaly, at the two ends: the patch is local, and the table should
    # show that it stays local however far the global expansion is pushed
    a0 = sw["alphas"][0]
    d0 = sw["by_alpha"][a0]
    print(f"\n  per-anomaly mass gain (alpha = {a0}), SH / SH+CH")
    print(f"  {'anomaly':22s} {f'L={L[0]}':>9} {f'L={L[i_nom]}':>9} {f'L={L[-1]}':>9}")
    for j, nm in enumerate(names):
        g = d0["mass"][SH_ONLY][:, j] / d0["mass"][SH_CH][:, j]
        pr = d0["mass_prior"][SH_ONLY][:, j]
        cells = "".join(
            f"{g[i]:8.1f}x" if not pr[i] else f"{'(prior)':>9s}" for i in (0, i_nom, -1)
        )
        print(f"  {nm:22s}{cells}")


def make_sweep_plots(sw, outdir="Images"):
    """
    Posterior/prior sigma ratios R of the mass fraction and the position
    against L_SH — pt2 fig 6c/6d, drawn for this experiment.  One file per
    quantity, one cell per anomaly with every case in each cell, and ONE y range
    across the pair, which is dimensionless throughout.  R = 1 is the prior
    itself; a point at or above the threshold has barely moved off it and
    carries a cross.

    NO GAIN PANELS.  They plotted ratios of the curves already drawn here, which
    is the same information a second time: with SH-only on the same axes the
    reader takes the ratio off the gap between the curves.  The crossing degrees
    the gains existed to show are quoted outright in TABLE 6.

    The cases carrying the CH block look flat in the target's cell, and that IS
    the content: those solutions are set by the patch, not by how far the
    global expansion is pushed.
    """
    os.makedirs(outdir, exist_ok=True)
    L, names = sw["L"], sw["names"]
    Ln = sw["L_nominal"]
    thr = sw["prior_ratio_threshold"]
    # ONE noise rule, as in pt2: `sweep_lmax_sh` sweeps the experiment's own
    # alpha by default, and a cell already carries one curve per case
    d_main = sw["by_alpha"][sw["alphas"][0]]

    def _centred_legend(fig, handles, labels, ncol, fontsize=9.0):
        """
        pt2's legend: rows of at most `ncol`, each its own CENTRED figure legend
        hung below the canvas (`bbox_inches="tight"` grows the crop to take it
        in).  The offsets are pt2's in INCHES, so the rows keep pt2's spacing on
        a one-row grid, where the same figure fractions would overlap them.
        """
        fig.tight_layout()
        h = fig.get_figheight()
        for r, i0 in enumerate(range(0, len(handles), ncol)):
            idx = list(range(i0, min(i0 + ncol, len(handles))))
            fig.legend(
                [handles[i] for i in idx],
                [labels[i] for i in idx],
                loc="upper center",
                ncol=len(idx),
                fontsize=fontsize * FONT_SCALE,
                frameon=False,
                bbox_to_anchor=(0.5, -(0.096 + 0.371 * r) / h),
            )

    def _grid(getter, ylab, fname, ylim, flags):
        """One cell per ANOMALY, every case drawn in each (pt2's `_grid`)."""
        n = len(names)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(11.4, 3.2 * nrows), sharex=True, sharey=True
        )
        axs = np.atleast_2d(axs)
        for j, (ax, nm) in enumerate(zip(axs.ravel(), names)):
            # CH-only and SH+CH COINCIDE wherever the patch carries the
            # solution, and one curve then hides the other outright.  CH-only
            # draws as a large HOLLOW ring and the rest as smaller filled
            # markers, so a filled marker inside a ring is visibly two series
            # at one value rather than a single series.
            for rank, k in enumerate(CASES):
                y = np.asarray(getter(d_main, k, j))
                hollow = k == CH_ONLY
                ax.plot(
                    L,
                    y,
                    "-",
                    color=CASE_COLOR[k],
                    marker=CASE_MARKER[k],
                    ms=(10 if hollow else 6),
                    mfc=("none" if hollow else CASE_COLOR[k]),
                    mec=(CASE_COLOR[k] if hollow else "k"),
                    mew=(1.5 if hollow else 0.6),
                    label=k,
                    zorder=3 + rank,
                )
                # Keep the case marker and overlay a cross for little reduction
                # from the prior.  Every degree is drawn, the flagged ones too.
                fl = np.asarray(flags(d_main, k, j), bool)
                ax.plot(
                    L[fl],
                    y[fl],
                    ls="none",
                    marker="x",
                    ms=9,
                    color="0.15",
                    mew=1.3,
                    zorder=10 + rank,
                )
            ax.axhline(1.0, color="0.35", lw=1.0, zorder=1, label=r"Prior $R = 1$")
            ax.axhline(
                thr,
                color="0.55",
                lw=1.0,
                ls="--",
                zorder=1,
                label=rf"Threshold $R = {thr:.2f}$",
            )
            ax.axvline(Ln, color="0.55", lw=1.0, ls=":", zorder=1)
            ax.set_title(nm, fontsize=10.5 * FONT_SCALE)
            ax.set_yscale("log")
            ax.grid(True, axis="y", which="both", ls=":", alpha=0.45)
            ax.set_axisbelow(True)
        for ax in axs.ravel()[n:]:
            ax.set_visible(False)
        for ax in axs[-1]:
            ax.set_xlabel(
                r"SH Maximum Degree $L_{\mathrm{SH}}$  [-]", fontsize=10 * FONT_SCALE
            )
        # ONE label for the whole figure: the quantity is the same in every cell
        fig.supylabel(ylab, fontsize=10.5 * FONT_SCALE)
        axs[0, 0].set_xlim(L[0] - 0.4, L[-1] + 0.4)
        axs[0, 0].set_xticks(L[:: max(1, len(L) // 6)])
        axs[0, 0].set_ylim(*ylim)  # axes are shared, so one call sets all
        handles, labels = axs[0, 0].get_legend_handles_labels()
        if any(np.any(flags(d_main, k, j)) for k in CASES for j in range(n)):
            handles.append(
                mpl.lines.Line2D(
                    [], [], ls="none", marker="x", ms=9, color="0.15", mew=1.3
                )
            )
            labels.append(rf"Cross: Prior-Dominated ($R \geq {thr:.2f}$)")
        _centred_legend(fig, handles, labels, 3)
        fig.savefig(os.path.join(outdir, fname), bbox_inches="tight")

    _mass_ratio = lambda d, k, j: d["mass_prior_ratio"][k][:, j]
    _mass_prior = lambda d, k, j: d["mass_prior"][k][:, j]
    _pos_ratio = lambda d, k, j: d["pos_prior_ratio"][k][:, j]
    _pos_prior = lambda d, k, j: d["pos_prior"][k][:, j]

    def _span(getter):
        v = np.concatenate(
            [
                np.asarray(getter(d_main, k, j), float)
                for k in CASES
                for j in range(len(names))
            ]
        )
        v = v[np.isfinite(v) & (v > 0)]
        return (0.6 * v.min(), 1.9 * v.max())

    # ONE y range across the pair, with R = 1 always in frame
    m_lim, p_lim = _span(_mass_ratio), _span(_pos_ratio)
    ratio_lim = (min(m_lim[0], p_lim[0]), max(1.2, m_lim[1], p_lim[1]))
    _grid(
        _mass_ratio,
        r"Mass-Fraction $R = \sigma_{\mathrm{post}} / \sigma_{\mathrm{prior}}$  [-]",
        PREFIX + "fig5a_lsh_mass_prior_ratio.pdf",
        ylim=ratio_lim,
        flags=_mass_prior,
    )
    _grid(
        _pos_ratio,
        r"Position $R = \sigma_{\mathrm{post}} / \sigma_{\mathrm{prior}}$  [-]",
        PREFIX + "fig5b_lsh_position_prior_ratio.pdf",
        ylim=ratio_lim,
        flags=_pos_prior,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    res = run_experiment(
        Lmax_sh=6,  # observable spherical-harmonic degree (tracking limit)
        eps=0.02,  # relative measurement precision (same on SH & field)
        ch_modes=(8, 8),  # (n_m, n_n) cylindrical-harmonic truncation
        n_cyl_pts=200,
        n_sweep=1000,  # detection-sweep noise draws per grid point
        outdir="Images",
        verbose=True,
    )
    # The L_SH sweep extends TABLE 1 and TABLE 4 along one axis, so it runs on
    # the experiment that produced them rather than rebuilding its own geometry.
    sw = sweep_lmax_sh(res)
    sweep_report(sw)
    make_sweep_plots(sw, outdir="Images")

    print("\nSaved to Images/ (one file per panel):")
    for _f in (
        PREFIX + "fig1_geometry.pdf",
        PREFIX + "fig1b_bouguer_sphere.pdf",
        PREFIX + "fig1c_bouguer_surface.pdf",
        PREFIX + "fig2a_massfraction_hist.pdf",
        PREFIX + "fig2a_massfraction_truths.pdf",
        PREFIX + "fig2a_massfraction_cov1.pdf",
        PREFIX + "fig2a_massfraction_cov2.pdf",
        PREFIX + "fig2b_detection.pdf",
        PREFIX + "fig3_position_hist.pdf",
        PREFIX + "fig3_position_truths.pdf",
        PREFIX + "fig3_position_cov.pdf",
        PREFIX + "fig3b_reach_map.pdf",
        PREFIX + "fig3b_reach_map_position.pdf",
        PREFIX + "fig4_coefficients_sh.pdf",
        PREFIX + "fig4_coefficients_ch.pdf",
        PREFIX + "fig5a_lsh_mass_prior_ratio.pdf",
        PREFIX + "fig5b_lsh_position_prior_ratio.pdf",
    ):
        print("  " + _f)
    print("Done.")
    # every figure at once, the sweep ones included: `make_plots` used to
    # call this itself, before `make_sweep_plots` had drawn anything
    plt.show()
