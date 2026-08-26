"""
Interior-density recovery: Spherical Harmonics vs Spherical + Cylindrical Harmonics
===================================================================================
Author: Giovanni Fereoli / experiment build

Question
--------
Can a combined SH + CH gravity model recover INTERIOR density better than SH
alone — because cylindrical harmonics sit close to the surface and therefore
carry localized, near-surface information that global spherical harmonics smear
out?

Model  —  a SCALED CONSTANT-DENSITY BULK plus localized anomalies
-----------------------------------------------------------------
The interior is NOT a cloud of mascons that sums to the total mass.  It is the
constant-density polyhedron of the shape model, scaled by β̃, plus a few
localized mascons that carry the departures from homogeneity:

    U_T(r) = β̃ · U_CD(r) + Σ_j β_j · U_pt(r; p_j),     β_j = m_j / M* ,

with M* = GM known from tracking (M* = 1 in normalized units).  The mass budget
β̃ M* + Σ_j m_j = M* fixes the bulk scale outright,

    β̃ = 1 − Σ_j β_j ,

so β̃ is NOT an independent unknown, the total mass is exact by construction, and
the old "Σ f = 1 known to σ_M" pseudo-observation disappears.  The estimated
vector is β = {β_j}: β_j > 0 is an over-dense concentration, β_j < 0 a mass
deficit.  Only the product m_j = Δρ_j v_j enters the field, so the mass fraction
— not the density contrast — is the identifiable parameter.

Substituting β̃ isolates the DISCREPANCY between the measured field and the
constant-density model, which is what the estimator actually fits:

    ΔU(r) = U_T(r) − U_CD(r) = Σ_j β_j [ U_pt(r; p_j) − U_CD(r) ] ,

so every design column is a CONTRAST against the homogeneous body.  Here Eros
carries one shallow anomaly under the +z pole (the CH target, whose position is
recovered too) and two deep ones in the lobes.

Physical mechanism
------------------
A deep anomaly is a LOW-degree (quadrupole-scale) feature — spherical harmonics
constrain it well.  The small near-surface anomaly writes its signature mostly
into HIGH-degree coefficients that are truncated/noisy from orbit, so SH is
nearly blind to it and it stays degenerate with the bulk.  Interior cylindrical
harmonics converge inside the Brillouin sphere right down to the surface (where
exterior SH diverges); low-altitude data over a cylinder above the anomaly,
represented by CH, injects exactly the local information SH is missing —
resolving the anomaly's mass fraction and position.

Both observables are LINEAR in β, so mass-fraction recovery is a linear Gaussian
inverse problem with an exact posterior covariance.  Position recovery is
nonlinear and handled by a linearized (Fisher) covariance.

Truth / units
-------------
Eros shape (`3dmeshes/eros.pk`), normalized units (LU), total mass M* = 1.  The
bulk observables are computed from the shape once and to machine precision: its
Stokes coefficients by exact tetrahedral quadrature of the solid harmonics (the
integrand is a polynomial), its near-surface field by polyhedral_gravity.

Two observables  (both fitted as DISCREPANCIES from the constant-density model)
--------------------------------------------------------------------------------
  A (SH)     : fully-normalized Stokes coefficients C̄_nm, S̄_nm, degree 2..L_SH,
               the global gravity field from tracking.  No total-mass row: the
               budget is enforced by β̃ = 1 − Σβ.
  B (SH+CH)  : the above  PLUS  the CH coefficients of the near-surface field
               (potential + acceleration) in a cylinder just above the anomaly,
               obtained by an UNWEIGHTED fit of the Bessel–Fourier basis Φ to the
               sampled field, c = Φ⁺ field.  Only the part of the near-surface
               field that Φ can represent survives that fit — the projector
               P_CH = Φ(ΦᵀΦ)⁻¹Φᵀ says how much (printed as a diagnostic).

Weights  (see `od_sigma`)
-------------------------
Estimation happens in COEFFICIENT space and is weighted there, per coefficient:
σ_i = eps·max(|coefficient_i|, floor), an OD-like "each coefficient known to a
fixed fraction of itself, above an absolute noise floor".  The inner fit that
manufactures the CH coefficients from field samples is deliberately unweighted.
The analytic covariance (Parts 1–2) and the Monte-Carlo fits (Experiments 1–2)
are fed the SAME (A, σ) blocks, so they describe one and the same estimator.

Experiments
-----------
  1. MASS FRACTION.  Posterior σ on each anomaly's β_j (and on the derived bulk
     fraction β̃), SH vs SH+CH — the near-surface anomaly gains the most.
  2. POSITION.  The near-surface anomaly's location; linearized position
     covariance (error ellipsoid), SH vs SH+CH.

Formulae
--------
SH (exterior) unit-mass Stokes basis, mass at (r,φ,λ), ref radius R*:
    {C̄,S̄}_nm  =  (r/R*)^n P̄_nm(sinφ) {cos,sin}(mλ)
CH (interior) basis in a cylinder (axis ẑ, radius R_cyl, extension α):
    U_mn = J_m(k_mn ρ) e^{-k_mn z} {cos,sin}(mφ),   k_mn = j_{m,n}/(α R_cyl)
Point mass at p, field at x:  U = Gm/|x-p|,  g = -Gm (x-p)/|x-p|³   (G=1 in LU).
Constant-density bulk, unit mass:  C̄_nm = (1/Vol) ∫_body (r/R*)^n P̄_nm cos mλ dV.
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
COLOR = ["#E6001A", "#F08C00", "#0077BB", "#1a9641", "#762a83"]
# Render figure text with a real LaTeX engine (exact document fonts) or with
# matplotlib's own mathtext (much faster).  A full run is several times slower
# with USE_TEX on, because every label is a separate LaTeX compile.
# Override from the shell without editing:  GLOBAL_NO_TEX=1 python ...
USE_TEX = False  # os.environ.get("GLOBAL_NO_TEX", "") == ""

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
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "figure.dpi": 110,
    }
)
G = 1.0  # gravitational constant in normalized units
G_SI = 6.67430e-11  # polyhedral_gravity works in SI; divided out to get G = 1
SEP = "=" * 70


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


def fully_normalized_legendre(nmax: int, x: float) -> np.ndarray:
    P = np.zeros((nmax + 1, nmax + 1))
    P[0, 0] = 1.0
    if nmax == 0:
        return P
    sx = math.sqrt(max(0.0, 1.0 - x * x))
    for m in range(1, nmax + 1):
        P[m, m] = math.sqrt((2.0 * m + 1.0) / (2.0 * m)) * sx * P[m - 1, m - 1]
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


def fully_normalized_legendre_v(nmax: int, x) -> np.ndarray:
    """Vectorized `fully_normalized_legendre`: same recursions, (nmax+1, nmax+1, N)."""
    x = np.atleast_1d(np.asarray(x, float))
    P = np.zeros((nmax + 1, nmax + 1, x.size))
    P[0, 0] = 1.0
    if nmax == 0:
        return P
    sx = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    for m in range(1, nmax + 1):
        P[m, m] = math.sqrt((2.0 * m + 1.0) / (2.0 * m)) * sx * P[m - 1, m - 1]
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


def sh_stokes_basis(pts, Lmin, Lmax, Rref):
    """
    Vectorized `sh_stokes_of_point`: row i is the unit-mass Stokes signature of a
    point mass at pts[i], in the SAME coefficient order.  (N, n_coeff)
    """
    pts = np.atleast_2d(np.asarray(pts, float))
    r = np.linalg.norm(pts, axis=1)
    lam = np.arctan2(pts[:, 1], pts[:, 0])
    Pb = fully_normalized_legendre_v(Lmax, pts[:, 2] / r)
    cols = []
    for n in range(Lmin, Lmax + 1):
        rr = (r / Rref) ** n
        for m in range(0, n + 1):
            base = rr * Pb[n, m]
            cols.append(base * np.cos(m * lam))
            cols.append(base * np.sin(m * lam))
    return np.column_stack(cols)


def sh_stokes_of_point(p, Lmin, Lmax, Rref):
    """Unit-mass fully-normalized Stokes coefficients of a point mass at p."""
    x, y, z = p
    r = math.sqrt(x * x + y * y + z * z)
    lam = math.atan2(y, x)
    Pb = fully_normalized_legendre(Lmax, z / r)
    out = []
    for n in range(Lmin, Lmax + 1):
        rr = (r / Rref) ** n
        for m in range(0, n + 1):
            out.append(rr * Pb[n, m] * math.cos(m * lam))
            out.append(rr * Pb[n, m] * math.sin(m * lam))
    return np.asarray(out)


def A_stokes(positions, Lmin, Lmax, Rref):
    """Design matrix: mascon masses -> Stokes coefficients (n_coeff, n_mascon)."""
    return np.column_stack([sh_stokes_of_point(p, Lmin, Lmax, Rref) for p in positions])


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
# looks.  Over a patch the Bessel-Fourier columns are close to linearly
# dependent (cond(Phi) ~ 1e16), and `np.linalg.pinv`'s DEFAULT cutoff is
# machine-epsilon based — max(M,N)*eps*s_max, about 9e-12 relative — so it keeps
# directions whose singular value is ~1e-13 of the largest.  Their coefficients
# come out as (projection)/sigma, i.e. enormous, and cancel again when
# multiplied back by Phi.  That, and nothing else, is why raw CH coefficients
# come out at 1e10 and refuse to decay with m.  Measured here at (8,8), 200 pts:
#
#   rcond    kept   ||c||    fit err   m-spectrum last/first
#   default    82   9.5e+09    2.1 %   3.13   (RISES)
#   1e-8       46   8.8e+05    4.7 %   0.18
#   1e-6       31   6.9e+03    6.9 %   1.2e-04
#   1e-4       16   1.6e+02   10.0 %   3.8e-07
#
# Truncating is not cosmetic: the discarded directions carry noise, not signal,
# so counting them as information inflates what the CH block appears to know.
# `cylinder_mass_estimation_BENNU_TAG.fit_coefficients` has always passed an
# explicit cond for this reason; the reference `..._MOV` script does not (it
# builds a regularization block, `order_weights`, and then drops it before the
# solve).  NOTE: with a large truncation and few sample points the system goes
# UNDERdetermined (e.g. (25,25) = 1250 columns against 4x190 = 760 rows), and
# numpy's minimum-norm solution then looks well behaved for a different reason —
# do not read that as the basis being well conditioned.
CH_RCOND = 1e-4


def ch_pinv(Phi, rcond=None):
    """Truncated-SVD pseudo-inverse of the CH basis (see CH_RCOND)."""
    return np.linalg.pinv(Phi, rcond=CH_RCOND if rcond is None else rcond)


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
# The interior is NOT a cloud of mascons that sums to the total mass.  It is the
# constant-density SHAPE MODEL scaled by β̃, plus N localized mascons carrying
# the departures from homogeneity,
#
#     U_T(r) = β̃ · U_CD(r) + Σ_j β_j · U_pt(r; p_j),      β_j = m_j / M* ,
#
# and the mass budget β̃ M* + Σ_j m_j = M* fixes the bulk scale outright,
#
#     β̃ = 1 − Σ_j β_j ,
#
# so β̃ is NOT an independent unknown and the total mass is M* by construction —
# the old "Σ f = 1 known to σ_M" pseudo-observation is gone with it.  What is
# left to estimate is β = {β_j}.  Substituting β̃ isolates the discrepancy
# between the measured field and the constant-density model,
#
#     ΔU(r) = U_T(r) − U_CD(r) = Σ_j β_j [ U_pt(r; p_j) − U_CD(r) ] ,
#
# which is what every design matrix below is: column j is a CONTRAST, the
# signature of taking a fraction β_j out of the homogeneous body and
# concentrating it at p_j.  β_j > 0 is a local excess (over-dense), β_j < 0 a
# deficit (under-dense); only the product m_j = Δρ_j v_j is identifiable, so the
# fraction, not the density contrast, is the estimated parameter.
#
# Both pieces of U_CD are computed once from the shape, to machine precision:
#   Stokes    — the integrand (r/R*)^n P̄_nm(sinφ){cos,sin}(mλ) is a solid
#               harmonic, i.e. a homogeneous POLYNOMIAL of degree n, so a
#               tetrahedral Gauss rule of degree ≥ n is exact (no model error).
#   near-field— polyhedral_gravity (Werner–Scheeres), which converges right down
#               to the surface where the SH series does not.


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
    def stokes(self, Lmin, Lmax, Rref, chunk=200_000):
        """
        Fully-normalized Stokes coefficients of the unit-mass constant-density
        polyhedron, in the ordering of `sh_stokes_of_point`:

            C̄_nm = (1/Vol) ∫_body (r/R*)^n P̄_nm(sinφ) cos mλ dV

        Each tetrahedron (origin, v0, v1, v2) is integrated with a Duffy-mapped
        tensor Gauss rule; the integrand is a polynomial of degree ≤ Lmax and the
        map contributes (1−u)²(1−v), so `ng` points per direction integrate
        degree 2·ng−1 ≥ Lmax+2 EXACTLY.  Signed volumes make the decomposition
        valid for a concave body whatever the origin.
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


def A_field_contrast(positions, obs, bulk):
    """Same contrast, in the near-surface field [U; ax; ay; az]:  Δfield = A β."""
    return A_field(positions, obs) - bulk.field(obs)[:, None]


def stokes_total(beta, positions, bulk, Lmin, Lmax, Rref):
    """Full Stokes vector of the truth model  β̃·CD + Σ β_j pt_j."""
    return (
        bulk.stokes(Lmin, Lmax, Rref)
        + A_stokes_contrast(positions, bulk, Lmin, Lmax, Rref) @ beta
    )


def field_total(beta, positions, obs, bulk):
    """Full near-surface field of the truth model  β̃·CD + Σ β_j pt_j."""
    return bulk.field(obs) + A_field_contrast(positions, obs, bulk) @ beta


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — OBSERVATION WEIGHTS, INFORMATION / COVARIANCE
# ═══════════════════════════════════════════════════════════════════════════
# TWO least-squares problems live in this code and they are weighted DIFFERENTLY
# on purpose:
#
#   1. Building the CH coefficients from sampled field values (Φ c = field).
#      UNWEIGHTED ordinary least squares — c = Φ⁺ field.  The field samples are
#      a synthetic product of one instrument over one small patch; there is no
#      per-sample error model to impose, and imposing one would just be a knob.
#
#   2. Estimating the mass fractions β from the COEFFICIENT discrepancies.
#      WEIGHTED, with a per-coefficient σ that imitates an OD solution: each
#      coefficient is delivered to a fixed FRACTION of its own magnitude, with an
#      absolute noise floor below which the solution cannot resolve anything.
#      This is the V that whitens the cost — one σ per coefficient, not one σ per
#      block, so a strong low-degree term and a weak high-degree one are not
#      given the same absolute weight.


def od_sigma(cs, eps, floor_frac=0.1):
    """
    OD-like 1σ for a measured coefficient vector `cs`:

        σ_i = eps · max( |cs_i| , floor_frac · RMS(cs) )

    The first branch is "every coefficient is known to eps of itself" — the
    relative precision an orbit-determination solution quotes.  The second is the
    absolute noise floor: an OD solution cannot resolve a coefficient far below
    the scale of the field it is fitting, and without it the entries that are
    identically zero by construction (the S̄_n0 sine terms) would carry infinite
    weight.  `cs` must be the FULL measured vector (bulk + anomalies), since that
    is what the instrument delivers before the constant-density model is removed.

    Caveat worth stating in a paper: constant relative precision across all
    degrees is optimistic at high degree, where a real OD solution degrades
    faster than the signal does.  A degree-dependent rule slots in here.
    """
    cs = np.asarray(cs, float)
    floor = floor_frac * float(np.sqrt(np.mean(cs**2)))
    return eps * np.maximum(np.abs(cs), floor)


def _col(sig):
    """σ as a column so `A / _col(σ)` whitens rows for scalar OR per-row σ."""
    s = np.asarray(sig, float)
    return s if s.ndim == 0 else s[:, None]


def fisher_masses(blocks, prior_sigma=1.0):
    """
    Fisher information for the anomaly mass-fraction vector β, built from the
    SAME (A, σ) coefficient blocks the Monte-Carlo fit uses:

        F = Σ_blocks  A_wᵀ A_w ,     A_w = A / σ   (row-wise; σ may be a vector)

    so the analytic covariance of Parts 1–2 and the actual fits of Experiments
    1–2 are guaranteed to describe one and the same estimator.  Every design
    matrix must be a contrast one (`A_stokes_contrast`, `ch_coeff_design`).
    There is no total-mass block: the budget β̃ = 1 − Σβ is enforced structurally
    by the parameterization, not by a pseudo-observation.  A weak Gaussian prior
    keeps everything finite.
    """
    n = blocks[0][0].shape[1]
    Fi = np.eye(n) / prior_sigma**2
    for A, sig in blocks:
        Aw = A / _col(sig)
        Fi = Fi + Aw.T @ Aw
    return Fi


def posterior_sigma(Fi):
    return np.sqrt(np.diag(np.linalg.inv(Fi)))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — THE ANOMALIES (contrasts on top of the constant-density bulk)
# ═══════════════════════════════════════════════════════════════════════════

# name, position [LU], truth mass fraction β_j = m_j / M*.  The bulk of the mass
# is NOT here — it is in the constant-density polyhedron, which keeps
# β̃ = 1 − Σβ = 0.96 of M*.  These are the departures from homogeneity:
# β_j > 0 an over-dense concentration, β_j < 0 a mass deficit.  Index 0 is the
# shallow anomaly (the CH target); 1 and 2 sit deep in the two lobes.
MASCONS = [
    ("near-surface anomaly", np.array([0.00, 0.00, 0.22]), 0.03),
    ("+x lobe excess", np.array([0.42, 0.00, 0.00]), 0.05),
    ("-x lobe deficit", np.array([-0.45, 0.00, 0.00]), -0.04),
]


def mascon_arrays():
    """names, positions, truth mass FRACTIONS β (not ratios summing to one)."""
    names = [m[0] for m in MASCONS]
    P = np.array([m[1] for m in MASCONS])
    f = np.array([m[2] for m in MASCONS])
    return names, P, f


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════


def stokes_power_spectrum(p, Lmax, Rref):
    """RMS Stokes magnitude per degree for a unit mass at p (shows where its
    signature lives: shallow → high degree, deep → low degree)."""
    x, y, z = p
    r = math.sqrt(x * x + y * y + z * z)
    lam = math.atan2(y, x)
    Pb = fully_normalized_legendre(Lmax, z / r)
    out = []
    for n in range(0, Lmax + 1):
        acc = 0.0
        for m in range(0, n + 1):
            c = (r / Rref) ** n * Pb[n, m]
            acc += (c * math.cos(m * lam)) ** 2 + (c * math.sin(m * lam)) ** 2
        out.append(math.sqrt(acc))
    return np.asarray(out)


def position_covariance(
    idx, P, obs, cyl, ch_modes, sig_sh, sig_ch, Lmax, Rref, pinvPhi
):
    """
    Linearized position covariance of anomaly `idx`, with its truth mass
    fraction, from SH-only and SH+CH.  Position partials by central differences,
    in the same COEFFICIENT space and with the same per-coefficient weights as
    the fits: the CH partial is Φ⁺ ∂(field)/∂p, i.e. the position sensitivity of
    the coefficients the unweighted inner fit would return.  (Other masses /
    positions held fixed — the near-surface anomaly is the target.)  The bulk
    term β̃·U_CD does not move with p, so it drops out of ∂y/∂p entirely: only
    the σ's carry its (large) presence, through the relative-precision rule.
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
    Fi_A = Jsh_w.T @ Jsh_w
    Jch = pinvPhi @ jac(lambda p: point_mass_field(p, obs), 4 * len(obs))
    Jch_w = Jch / _col(sig_ch)
    Fi_B = Fi_A + Jch_w.T @ Jch_w

    def summ(Fi):
        C = np.linalg.inv(Fi)
        return dict(cov=C, rms=float(np.sqrt(np.trace(C) / 3.0)))

    return summ(Fi_A), summ(Fi_B)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5b — ACTUAL LEAST-SQUARES FIT (Monte-Carlo, like the reference code)
# ═══════════════════════════════════════════════════════════════════════════


def ch_coeff_design(P, obs, cyl, ch_modes, bulk):
    """
    β → CYLINDRICAL-HARMONIC coefficient design matrix.
    For each anomaly we evaluate the near-surface field of the CONTRAST (unit
    mass at p_j minus the same mass spread through the body) and fit the
    Bessel–Fourier basis Φ by ORDINARY (unweighted) least squares, exactly as the
    reference script fits CH coefficients to a sampled field:
    A_ch = Φ⁺ A_field_contrast.  The weighting deliberately enters one level up,
    on the COEFFICIENTS (see `od_sigma`), not on the field samples that produce
    them.  Fitting the contrast is the near-surface form of ΔU: the
    constant-density field is known from the shape and subtracted before the
    anomalies are estimated.
    """
    Phi = cyl_basis(cyl, obs, *ch_modes)
    return ch_pinv(Phi) @ A_field_contrast(P, obs, bulk)  # (n_ch, n_anom)


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


def monte_carlo_fit(blocks, m_true, n_mc=400, seed=7):
    """Monte-Carlo over noise: returns recovered masses (n_mc, n_mascon)."""
    rng = np.random.default_rng(seed)
    return np.array([ls_fit_once(blocks, m_true, rng) for _ in range(n_mc)])


def detection_sweep(A_sh, sig_sh, A_ch, sig_ch, f_base, mu_grid, n_mc=250, seed=11):
    """
    Smallest detectable anomaly with vs without the cylinder.  Sweeps the true
    shallow-anomaly fraction β_0 over `mu_grid`, keeps the two deep anomalies
    fixed (the bulk takes up the slack, β̃ = 1 − Σβ), and for each value runs the
    LS fit (SH-only and SH+CH) over noise.  Returns, per grid value, the
    recovered anomaly mean and scatter for both cases.
    """
    out = {k: [] for k in ("muA", "sdA", "muB", "sdB")}
    for mu in mu_grid:
        m_true = f_base.copy()
        m_true[0] = mu
        A = monte_carlo_fit([(A_sh, sig_sh)], m_true, n_mc, seed)
        B = monte_carlo_fit([(A_sh, sig_sh), (A_ch, sig_ch)], m_true, n_mc, seed)
        out["muA"].append(A[:, 0].mean())
        out["sdA"].append(A[:, 0].std())
        out["muB"].append(B[:, 0].mean())
        out["sdB"].append(B[:, 0].std())
    for k in out:
        out[k] = np.asarray(out[k])
    out["mu_grid"] = np.asarray(mu_grid)

    # 3σ detection threshold = smallest true anomaly whose recovery exceeds 3×scatter
    def thr(mu, sd):
        floor = np.median(sd)  # scatter is ~flat in the fraction (linear problem)
        return 3.0 * floor

    out["thr_A"] = thr(out["mu_grid"], out["sdA"])
    out["thr_B"] = thr(out["mu_grid"], out["sdB"])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5c — EXPERIMENT 2: fit anomaly POSITION with masses FIXED (MC)
# ═══════════════════════════════════════════════════════════════════════════
# (Experiment 1 — masses free, positions fixed — is the linear fit above.)


def _pos_forward(pos0, masses, lobe_pos, Lmax, Rref, obs, pinvPhi, use_ch, bulk):
    """
    Forward observables when ONLY the shallow anomaly's position pos0=(x,y,z) is
    unknown; all three mass fractions and the two deep positions are known.  The
    model is the full field β̃·CD + Σ β_j pt_j — the bulk is an additive constant
    here (β̃ is fixed with the masses), present so the forward model is the same
    one the paper writes down.
    """
    positions = [pos0, lobe_pos[0], lobe_pos[1]]
    y_sh = bulk_fraction(masses) * bulk.stokes(2, Lmax, Rref)
    for mj, pj in zip(masses, positions):
        y_sh = y_sh + mj * sh_stokes_of_point(pj, 2, Lmax, Rref)
    blocks = [y_sh]
    if use_ch:
        field = bulk_fraction(masses) * bulk.field(obs)
        for mj, pj in zip(masses, positions):
            field = field + mj * point_mass_field(pj, obs)
        blocks.append(pinvPhi @ field)
    return blocks


def _pos_residual(
    pos0,
    data_blocks,
    sig_blocks,
    masses,
    lobe_pos,
    Lmax,
    Rref,
    obs,
    pinvPhi,
    use_ch,
    bulk,
):
    model = _pos_forward(pos0, masses, lobe_pos, Lmax, Rref, obs, pinvPhi, use_ch, bulk)
    return np.concatenate(
        [(mo - da) / s for mo, da, s in zip(model, data_blocks, sig_blocks)]
    )


def position_mc(
    P,
    f_true,
    obs,
    cyl,
    ch_modes,
    Lmax,
    Rref,
    sig_sh,
    sig_ch,
    bulk,
    use_ch=False,
    n_mc=150,
    seed=21,
    start_offset=0.03,
):
    """
    EXPERIMENT 2 — Monte-Carlo NONLINEAR least-squares recovery of the anomaly
    POSITION (x, y, z) with all mass fractions FIXED at truth.  Each draw builds
    noisy data, starts from a guess offset by `start_offset` LU, and fits (scipy
    TRF).  Returns recovered positions (n_mc, 3).
    """
    Phi = cyl_basis(cyl, obs, *ch_modes)
    pinvPhi = ch_pinv(Phi)
    masses = f_true
    lobe_pos = [P[1], P[2]]
    pos_true = P[0].copy()
    truth_blocks = _pos_forward(
        pos_true, masses, lobe_pos, Lmax, Rref, obs, pinvPhi, use_ch, bulk
    )
    sig_blocks = [sig_sh, sig_ch][: len(truth_blocks)]

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_mc):
        data = [
            tb + rng.normal(0.0, s, size=tb.shape)
            for tb, s in zip(truth_blocks, sig_blocks)
        ]
        pos0 = pos_true + start_offset
        sol = least_squares(
            _pos_residual,
            pos0,
            method="trf",
            args=(
                data,
                sig_blocks,
                masses,
                lobe_pos,
                Lmax,
                Rref,
                obs,
                pinvPhi,
                use_ch,
                bulk,
            ),
            xtol=1e-12,
            ftol=1e-12,
            max_nfev=400,
        )
        out.append(sol.x)
    return np.asarray(out)


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

    With `center` and `spread` they are drawn uniformly in a BALL of that radius
    about the nominal site — the anomaly is somewhere near where we thought, not
    anywhere in the asteroid.  That is the relevant population here, because the
    cylinder was placed FOR this site; scattering the truth through the whole
    interior instead answers "what if we pointed the patch at nothing", which is
    a different question.  Without them, uniform over the whole body.
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
    P, bulk, obs, cyl, ch_modes, Lmax, Rref, eps, n_truth=40, n_noise=60, seed=101
):
    """
    Redraw the truth MASSES; for each, rebuild σ from that truth's own field and
    refit.  Returns (sigA, sigB, bulkA, bulkB, betas) with sig* of shape
    (n_truth, n_anom): the per-parameter 1σ each truth interior yields.
    """
    rng = np.random.default_rng(seed)
    betas = draw_truth_masses(n_truth, rng)
    Phi = cyl_basis(cyl, obs, *ch_modes)
    pinvPhi = ch_pinv(Phi)
    A_sh = A_stokes_contrast(P, bulk, 2, Lmax, Rref)
    A_ch = ch_coeff_design(P, obs, cyl, ch_modes, bulk)
    sigA = np.empty((n_truth, len(P)))
    sigB = np.empty_like(sigA)
    bulkA, bulkB = np.empty(n_truth), np.empty(n_truth)
    for i, b in enumerate(betas):
        s_sh = od_sigma(stokes_total(b, P, bulk, 2, Lmax, Rref), eps)
        s_ch = od_sigma(pinvPhi @ field_total(b, P, obs, bulk), eps)
        mcA = monte_carlo_fit([(A_sh, s_sh)], b, n_mc=n_noise, seed=7 + i)
        mcB = monte_carlo_fit([(A_sh, s_sh), (A_ch, s_ch)], b, n_mc=n_noise, seed=7 + i)
        sigA[i], sigB[i] = mcA.std(0), mcB.std(0)
        bulkA[i], bulkB[i] = (1.0 - mcA.sum(1)).std(), (1.0 - mcB.sum(1)).std()
    return sigA, sigB, bulkA, bulkB, betas


def truth_mc_position(
    P,
    f_true,
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
    n_truth=25,
    n_noise=12,
    seed=202,
    start_offset=0.03,
    spread=0.12,
    n_noise_rep=250,
):
    """
    Redraw the truth POSITION of the shallow anomaly near its nominal site;
    for each, rebuild the data and refit, SH-only and SH+CH.  Returns
    a dict with per-truth RMS position error, where each truth sat relative to
    the cylinder, and — for one representative draw (`i_rep`, the truth closest
    to the nominal site) — the full cloud of recovered positions for both cases,
    so the estimator's own scatter and covariance can be shown.
    """
    rng = np.random.default_rng(seed)
    pts = draw_truth_positions(n_truth, V, F, tm, rng, center=P[0], spread=spread)
    Phi = cyl_basis(cyl, obs, *ch_modes)
    pinvPhi = ch_pinv(Phi)
    axis = cyl.rot() @ np.array([0.0, 0.0, 1.0])
    lobe_pos = [P[1], P[2]]
    errA, errB = np.empty(n_truth), np.empty(n_truth)
    d_ax, dep = np.empty(n_truth), np.empty(n_truth)
    zmax = V[:, 2].max()
    i_rep = int(np.argmin(np.linalg.norm(pts - P[0], axis=1)))
    clouds = {}
    for i, p0 in enumerate(pts):
        Pi = P.copy()
        Pi[0] = p0
        s_sh = od_sigma(stokes_total(f_true, Pi, bulk, 2, Lmax, Rref), eps)
        s_ch = od_sigma(pinvPhi @ field_total(f_true, Pi, obs, bulk), eps)
        v = p0 - cyl.center
        d_ax[i] = np.linalg.norm(v - np.dot(v, axis) * axis)
        dep[i] = zmax - p0[2]
        n_draw = n_noise_rep if i == i_rep else n_noise
        for use_ch, out in ((False, errA), (True, errB)):
            blocks = _pos_forward(
                p0, f_true, lobe_pos, Lmax, Rref, obs, pinvPhi, use_ch, bulk
            )
            sig_b = [s_sh, s_ch][: len(blocks)]
            r = np.random.default_rng(seed + 1000 * i + int(use_ch))
            acc = []
            for _ in range(n_draw):
                data = [
                    tb + r.normal(0.0, sg, size=tb.shape)
                    for tb, sg in zip(blocks, sig_b)
                ]
                sol = least_squares(
                    _pos_residual,
                    p0 + start_offset,
                    method="trf",
                    args=(
                        data,
                        sig_b,
                        f_true,
                        lobe_pos,
                        Lmax,
                        Rref,
                        obs,
                        pinvPhi,
                        use_ch,
                        bulk,
                    ),
                    xtol=1e-10,
                    ftol=1e-10,
                    max_nfev=300,
                )
                acc.append(sol.x)
            acc = np.asarray(acc)
            out[i] = np.sqrt(np.mean(np.sum((acc - p0) ** 2, axis=1)))
            if i == i_rep:
                clouds["B" if use_ch else "A"] = acc
        if i == i_rep:
            # J^T W J from the position partials — the covariance the estimator
            # HAS, rather than the one this particular set of draws happened to
            # produce.  Cheap, and it needs no Monte-Carlo at all.
            cA, cB = position_covariance(
                0, Pi, obs, cyl, ch_modes, s_sh, s_ch, Lmax, Rref, pinvPhi
            )
            clouds["covA"], clouds["covB"] = cA["cov"], cB["cov"]
    return dict(
        errA=errA,
        errB=errB,
        pos=pts,
        d_axis=d_ax,
        depth=dep,
        i_rep=i_rep,
        rep_truth=pts[i_rep],
        cloudA=clouds["A"],
        cloudB=clouds["B"],
        covA=clouds["covA"],
        covB=clouds["covB"],
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
    ft, bb = res["f_true"], res["beta_bulk"]
    q = lambda v: np.percentile(v, [10, 50, 90])
    rows_m = []
    for k, nm in enumerate(names):
        a, b = q(tmc["sigA"][:, k]), q(tmc["sigB"][:, k])
        g = np.median(tmc["sigA"][:, k] / tmc["sigB"][:, k])
        rows_m.append((nm, ft[k], a, b, g))
    a, b = q(tmc["bulkA"]), q(tmc["bulkB"])
    rows_m.append(
        (r"body $\tilde\beta$", bb, a, b, np.median(tmc["bulkA"] / tmc["bulkB"]))
    )

    print(f"\n{SEP}\n  RESULTS  (figures carry no numbers; quote from here)\n{SEP}")
    print(
        "\n  TABLE 1 — mass-fraction uncertainty, "
        f"{len(tmc['sigA'])} truth interiors, 1σ, median [10–90%]"
    )
    print(
        f"  {'component':22s} {'truth β':>9} {'σ SH':>10} {'[10–90%]':>21} "
        f"{'σ SH+CH':>10} {'[10–90%]':>21} {'gain':>7}"
    )
    for nm, tr, a, b, g in rows_m:
        nm_ = nm.replace(r"$\tilde\beta$", "β̃")
        print(
            f"  {nm_:22s} {tr:+9.4f} {a[1]:10.2e} "
            f"[{a[0]:8.2e},{a[2]:8.2e}] {b[1]:10.2e} "
            f"[{b[0]:8.2e},{b[2]:8.2e}] {g:6.1f}×"
        )

    eA, eB, dax = tmc["errA"], tmc["errB"], tmc["d_axis"]
    print(
        f"\n  TABLE 2 — anomaly position, {len(eA)} truth interiors, "
        f"RMS error [LU], median [10–90%]"
    )
    print(f"  {'case':22s} {'median':>10} {'[10–90%]':>21} {'gain':>7}")
    for lab, e in (("SH only", eA), ("SH + CH", eB)):
        Q = q(e)
        g = np.median(eA / eB) if lab == "SH + CH" else float("nan")
        gs = f"{g:6.1f}×" if np.isfinite(g) else " " * 7
        print(f"  {lab:22s} {Q[1]:10.2e} [{Q[0]:8.2e},{Q[2]:8.2e}] {gs}")
    ed = np.quantile(dax, [0, 0.25, 0.5, 0.75, 1.0])
    ed[-1] += 1e-9
    print(f"\n  TABLE 3 — position gain vs distance from the cylinder axis")
    print(
        f"  {'range [LU]':>16} {'n':>4} {'median SH':>11} {'median SH+CH':>13} "
        f"{'gain':>7}"
    )
    for lo, hi in zip(ed[:-1], ed[1:]):
        m = (dax >= lo) & (dax < hi)
        if m.sum():
            print(
                f"  {lo:6.3f}–{hi:6.3f} {int(m.sum()):4d} {np.median(eA[m]):11.2e} "
                f"{np.median(eB[m]):13.2e} {np.median(eA[m]/eB[m]):6.1f}×"
            )

    sA = np.sqrt(np.trace(tmc["covA"][np.ix_([0, 2], [0, 2])]) / 2)
    sB = np.sqrt(np.trace(tmc["covB"][np.ix_([0, 2], [0, 2])]) / 2)
    print(f"\n  TABLE 4 — single-interior detail and detection limit")
    print(f"  {'quantity':38s} {'SH':>12} {'SH + CH':>12} {'ratio':>8}")
    print(
        f"  {'analytic position 1σ (x–z) [LU]':38s} {sA:12.2e} {sB:12.2e} "
        f"{sA/sB:7.1f}×"
    )
    print(
        f"  {'smallest detectable anomaly β_0':38s} {det['thr_A']:12.2e} "
        f"{det['thr_B']:12.2e} {det['thr_A']/det['thr_B']:7.1f}×"
    )
    print(
        f"  {'discrepancy-to-noise, RMS ΔCS/σ':38s} {res['snr_sh']:12.1f} "
        f"{res['snr_ch']:12.1f}"
    )
    print(
        f"  {'post-fit residual [σ]':38s} {res['rms_post_sh']:12.2f} "
        f"{res['rms_post_ch']:12.2f}"
    )

    if not tex:
        return
    print(f"\n{'-'*70}\n  LaTeX tabular bodies\n{'-'*70}")
    print(r"  % Table 1 — mass-fraction uncertainty")
    for nm, tr, a, b, g in rows_m:
        print(
            rf"  {nm} & ${tr:+.4f}$ & {_tex_num(a[1])} & {_tex_num(b[1])} "
            rf"& ${g:.1f}$ \\"
        )
    print(r"  % Table 2 — position RMS error [LU]")
    for lab, e in (("SH only", eA), ("SH + CH", eB)):
        Q = q(e)
        print(
            rf"  {lab} & {_tex_num(Q[1])} & {_tex_num(Q[0])} & " rf"{_tex_num(Q[2])} \\"
        )
    print(rf"  % gain (median of per-interior ratios): ${np.median(eA/eB):.1f}$")


def run_experiment(
    Lmax_sh=6,
    eps=0.02,
    ch_modes=(8, 8),
    n_cyl_pts=200,
    n_mc=400,
    n_mc_nl=150,
    n_truth_m=100,  # truth-mass draws for the mass experiment
    n_truth_p=100,  # truth-position draws for the position experiment
    n_noise_p=12,  # noise draws per truth position
    detail=False,  # verbose narrative; the tables at the end carry the numbers
    pos_spread=0.20,  # truth positions jitter within this radius of the site
    outdir="Images",
    verbose=True,
):
    """
    `eps` is the RELATIVE measurement precision applied EQUALLY to both
    observables, PER COEFFICIENT: σ_i = eps·|coefficient_i| with a noise floor
    (see `od_sigma`), on the FULL measured coefficients (bulk included, since
    that is what an OD solution actually delivers).  Same fractional data quality
    on the global Stokes coefficients and on the local CH coefficients, so the
    comparison reflects geometry, not the (different) natural units of the two.
    What is FITTED is the discrepancy between that measurement and the known
    constant-density model.  The inner fit that produces the CH coefficients from
    field samples is unweighted; the weights live here, on the coefficients.
    """
    V, F, tm, Rb = load_eros()
    Rref = Rb
    zmax = V[:, 2].max()
    names, P, f_true = mascon_arrays()
    bulk = Bulk(V, F)
    beta_bulk = bulk_fraction(f_true)
    target = 0  # near-surface anomaly

    if verbose:
        print(SEP)
        print("  Interior-density recovery: SH  vs  SH+CH   (Eros, normalized units)")
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
        for nm, p, fr in zip(names, P, f_true):
            print(f"    {nm:22s} p={np.round(p,3)}  β={fr:+.3f}  depth={zmax-p[2]:.3f}")

    # cylinder of near-surface data over the anomaly (+z pole)
    cyl = Cylinder(center=np.array([0.0, 0.0, zmax + 0.005]), radius=0.12, height=0.40)
    obs = cylinder_points(cyl, n=n_cyl_pts)
    obs = obs[~inside_body(tm, V, F, obs)]
    r_obs = np.linalg.norm(obs, axis=1)

    # DISCREPANCY designs: every column is (point mass at p_j) − (same mass
    # spread through the body), so β is estimated against the constant-density
    # model rather than against vacuum.
    A_sh = A_stokes_contrast(P, bulk, 2, Lmax_sh, Rref)
    A_fd = A_field_contrast(P, obs, bulk)
    Phi = cyl_basis(cyl, obs, *ch_modes)
    pinvPhi = ch_pinv(Phi)  # UNWEIGHTED inner CH fit, truncated SVD
    A_ch = ch_coeff_design(P, obs, cyl, ch_modes, bulk)

    # OD-like per-coefficient noise on the FULL measured coefficients (bulk +
    # anomalies), the same relative rule on both observables.
    y_sh_tot = stokes_total(f_true, P, bulk, 2, Lmax_sh, Rref)
    y_fd_tot = field_total(f_true, P, obs, bulk)
    y_ch_tot = pinvPhi @ y_fd_tot
    sig_sh = od_sigma(y_sh_tot, eps)
    sig_ch = od_sigma(y_ch_tot, eps)
    blocksA = [(A_sh, sig_sh)]  # SH only
    blocksB = [(A_sh, sig_sh), (A_ch, sig_ch)]  # SH + CH
    if verbose:
        print(
            f"  cylinder over anomaly: {len(obs)} vacuum pts, "
            f"|r|∈[{r_obs.min():.2f},{r_obs.max():.2f}] ⊂ Brillouin {Rb:.2f}"
        )
        d_fd = A_fd @ f_true
        rep_ch = float(np.linalg.norm(ch_projector(Phi) @ d_fd) / np.linalg.norm(d_fd))
        print(
            f"  observables: SH deg 2..{Lmax_sh} ({A_sh.shape[0]} coeffs)"
            f" | CH modes {ch_modes} ({Phi.shape[1]} cols)"
            f"  [no Σβ=1 row — the mass budget is structural]"
        )
        print(
            f"  weights: OD-like σ_i = {eps}·|coeff_i| (floor 10% of RMS)  →  "
            f"σ_SH ∈ [{sig_sh.min():.2e}, {sig_sh.max():.2e}], "
            f"σ_CH ∈ [{sig_ch.min():.2e}, {sig_ch.max():.2e}]"
        )
        print(
            f"  inner CH fit (Φ c = field) is UNWEIGHTED; its span captures "
            f"{rep_ch:.3f} of the near-surface discrepancy"
        )
        print(
            f"  discrepancy / full field:  SH "
            f"{np.sqrt(np.mean((A_sh @ f_true)**2)) / np.sqrt(np.mean(y_sh_tot**2)):.3f}"
            f"   near-surface "
            f"{np.sqrt(np.mean(d_fd**2)) / np.sqrt(np.mean(y_fd_tot**2)):.3f}"
        )

    # ── PART 1 — mass fractions ────────────────────────────────────────────────
    Fi_A = fisher_masses(blocksA)
    Fi_B = fisher_masses(blocksB)
    sdA, sdB = posterior_sigma(Fi_A), posterior_sigma(Fi_B)
    improve = sdA / sdB
    if verbose and detail:
        print(f"\n{'-'*70}\n  PART 1 — MASS-FRACTION UNCERTAINTY (1σ on β_j)\n{'-'*70}")
        print(
            f"  {'anomaly':22s} {'depth':>6} {'σ_SH':>10} {'σ_SH+CH':>10} {'gain':>7}"
        )
        for nm, p, a, b in zip(names, P, sdA, sdB):
            print(f"  {nm:22s} {zmax-p[2]:6.3f} {a:10.2e} {b:10.2e} {a/b:6.1f}×")

    # ── PART 2 — position of the near-surface anomaly ───────────────────────
    posA, posB = position_covariance(
        target, P, obs, cyl, ch_modes, sig_sh, sig_ch, Lmax_sh, Rref, pinvPhi
    )
    if verbose and detail:
        print(
            f"\n{'-'*70}\n  PART 2 — POSITION OF NEAR-SURFACE ANOMALY "
            f"(β={f_true[target]:+.3f})\n{'-'*70}"
        )
        print(
            f"  position 1σ RMS:  SH={posA['rms']:.3e} LU   SH+CH={posB['rms']:.3e} LU"
            f"   → {posA['rms']/posB['rms']:.0f}× tighter"
        )

    # the mass-budget panel marks the RECOVERED body fraction, which needs one
    # ordinary noise-MC at the nominal truth; the truth-MC below reports spreads
    mc_nom = monte_carlo_fit(blocksB, f_true, n_mc=n_mc)
    bulk_nominal = (
        float((1.0 - mc_nom.sum(1)).mean()),
        float((1.0 - mc_nom.sum(1)).std()),
    )

    # ══ EXPERIMENTS 1 & 2 — MONTE-CARLO OVER THE TRUTH ═════════════════════
    # Experiment 1 resamples the truth MASSES, experiment 2 the truth POSITION
    # of the shallow anomaly.  Each draw is a different interior, refitted from
    # scratch, so the spread reported below is over INTERIORS — "would this
    # conclusion survive a different body?" — rather than over noise draws at
    # one hard-coded truth, which only answers "how precise is this one fit?".
    tm_sigA, tm_sigB, tm_bA, tm_bB, tm_betas = truth_mc_masses(
        P,
        bulk,
        obs,
        cyl,
        ch_modes,
        Lmax_sh,
        Rref,
        eps,
        n_truth=n_truth_m,
        n_noise=max(40, n_mc // 8),
    )
    tp = truth_mc_position(
        P,
        f_true,
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
        n_noise=n_noise_p,
        spread=pos_spread,
    )
    tp_eA, tp_eB, tp_pos = tp["errA"], tp["errB"], tp["pos"]
    tp_dax, tp_dep = tp["d_axis"], tp["depth"]
    if verbose and detail:
        q = lambda v: (np.percentile(v, 10), np.median(v), np.percentile(v, 90))
        print(
            f"\n{'='*70}\n  EXPERIMENT 1 — MASS FRACTIONS, Monte-Carlo over "
            f"{n_truth_m} truth interiors\n{'='*70}"
        )
        print(f"  truth |β| ~ U[0.01,0.06] with random signs; positions fixed")
        print(
            f"  {'anomaly':22s} {'σ_SH  10/50/90%':>26} {'σ_SH+CH  10/50/90%':>26}"
            f" {'gain':>7}"
        )
        for k, nm in enumerate(names):
            a, b = q(tm_sigA[:, k]), q(tm_sigB[:, k])
            print(
                f"  {nm:22s} "
                + " ".join(f"{x:8.2e}" for x in a)
                + "  "
                + " ".join(f"{x:8.2e}" for x in b)
                + f" {np.median(tm_sigA[:, k] / tm_sigB[:, k]):6.1f}×"
            )
        a, b = q(tm_bA), q(tm_bB)
        print(
            f"  {'BODY β̃':22s} "
            + " ".join(f"{x:8.2e}" for x in a)
            + "  "
            + " ".join(f"{x:8.2e}" for x in b)
            + f" {np.median(tm_bA / tm_bB):6.1f}×"
        )
        print(
            f"\n{'='*70}\n  EXPERIMENT 2 — ANOMALY POSITION, Monte-Carlo over "
            f"{n_truth_p} truth positions\n{'='*70}"
        )
        print(
            f"  the shallow anomaly is jittered within {pos_spread} LU of its "
            f"site ({n_noise_p} noise draws each)"
        )
        print(f"  {'':22s} {'RMS err 10/50/90%  [LU]':>28} {'median gain':>12}")
        print(f"  {'SH only':22s} " + " ".join(f"{x:9.2e}" for x in q(tp_eA)))
        print(
            f"  {'SH + CH':22s} "
            + " ".join(f"{x:9.2e}" for x in q(tp_eB))
            + f" {np.median(tp_eA / tp_eB):11.1f}×"
        )
        edges = np.quantile(tp_dax, [0.0, 0.25, 0.5, 0.75, 1.0])
        edges[-1] += 1e-9
        print(f"\n  by horizontal distance from the cylinder axis (quartiles):")
        print(
            f"  {'range [LU]':>14} {'n':>4} {'median SH':>11} {'median SH+CH':>13}"
            f" {'gain':>8}"
        )
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (tp_dax >= lo) & (tp_dax < hi)
            if m.sum() == 0:
                continue
            print(
                f"  {lo:5.3f}–{hi:5.3f} {int(m.sum()):4d} "
                f"{np.median(tp_eA[m]):11.2e} {np.median(tp_eB[m]):13.2e} "
                f"{np.median(tp_eA[m] / tp_eB[m]):7.1f}×"
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
    d_sh, d_ch = A_sh @ f_true, A_ch @ f_true  # = CS_hetero − CS_homog
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
    f_base = f_true.copy()
    mu_grid = np.logspace(-4.5, -0.7, 16)  # true anomaly mass-fraction sweep
    det = detection_sweep(
        A_sh, sig_sh, A_ch, sig_ch, f_base, mu_grid, n_mc=max(150, n_mc // 2)
    )
    if verbose and detail:
        print(f"\n  smallest detectable anomaly (3σ fit scatter):")
        print(f"    SH only : μ_min = {det['thr_A']:.2e}")
        print(
            f"    SH + CH : μ_min = {det['thr_B']:.2e}   "
            f"→ {det['thr_A']/det['thr_B']:.0f}× smaller anomaly detectable"
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
        f_true=f_true,
        bulk=bulk,
        beta_bulk=beta_bulk,
        target=target,
        sdA=sdA,
        sdB=sdB,
        improve=improve,
        posA=posA,
        posB=posB,
        det=det,
        bulk_nominal=bulk_nominal,
        snr_sh=snr_sh,
        snr_ch=snr_ch,
        rms_post_sh=float(np.sqrt(np.mean(r_sh**2))),
        rms_post_ch=float(np.sqrt(np.mean(r_ch**2))),
        pos_spread=pos_spread,
        truth_mc=dict(
            sigA=tm_sigA, sigB=tm_sigB, bulkA=tm_bA, bulkB=tm_bB, betas=tm_betas, **tp
        ),
        spectra=spectra,
        Lmax_sh=Lmax_sh,
        ch_modes=ch_modes,
        sig_sh=sig_sh,
        sig_ch=sig_ch,
    )
    make_plots(res, outdir=outdir)
    if verbose:
        results_report(res)
    return res


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════


def draw_mass_budget(
    ax, names, beta, beta_bulk=None, recovered=None, rec_label="SH + CH"
):
    """
    Waterfall of the mass budget, shared by pt1/pt2/pt3.

    The point it makes visible: the interior is NOT the mascons.  The
    constant-density polyhedron carries a mass fraction β̃ of its own, each
    anomaly then adds (β_j > 0) or removes (β_j < 0) a few per cent, and the
    running total closes at exactly M* = 1.  That closure IS the constraint
    β̃ = 1 − Σβ_j — mass conservation — rather than something imposed as an
    extra observation.

    The bars are the TRUTH budget — the simulated interior, i.e. what was put
    in.  β̃ there is not an assumption of the estimator: it is 1 − Σβ of the
    chosen truth anomalies.  Pass `recovered=(mean, std)` to also mark the
    ESTIMATED body fraction, which comes out of the fit as 1 − Σβ̂ evaluated on
    every Monte-Carlo draw and carries a real uncertainty.  Seeing both is the
    point: the estimator is never told β̃, it reconstructs it.

    The y-axis is zoomed onto the region the anomalies occupy, so the body's
    bar runs off the bottom of the axis (it is labelled with its value); the
    alternative is a 0–1 axis on which every anomaly step is invisible.
    """
    beta = np.asarray(beta, float)
    if beta_bulk is None:
        beta_bulk = bulk_fraction(beta)
    vals = np.concatenate([[beta_bulk], beta])
    ends = np.cumsum(vals)
    start = ends - vals
    x = np.arange(len(vals))
    lo = min(beta_bulk, ends.min()) - 0.035
    hi = max(1.0, ends.max()) + 0.030

    cols = ["0.55"] + [COLOR[0] if v > 0 else COLOR[2] for v in beta]
    ax.bar(x[0], beta_bulk - lo, bottom=lo, color=cols[0], edgecolor="k", width=0.62)
    ax.bar(x[1:], vals[1:], bottom=start[1:], color=cols[1:], edgecolor="k", width=0.62)
    for k in range(len(vals) - 1):  # waterfall connectors
        ax.plot(
            [x[k] + 0.31, x[k + 1] - 0.31], [ends[k]] * 2, color="0.5", lw=0.9, ls=":"
        )
    ax.axhline(1.0, color="k", ls="--", lw=1.5)
    ax.text(
        x[0] - 0.42,
        1.0,
        r"$M^*=1$",
        va="bottom",
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["BODY\n(const. density)"] + [n.split()[0] for n in names], fontsize=8
    )
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Cumulative Mass Fraction of $M^*$")
    ax.grid(True, axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    handles = [
        Patch(fc="0.55", ec="k", label=r"Body $\tilde\beta$ (truth)"),
        Patch(fc=COLOR[0], ec="k", label=r"Anomaly $\beta_j>0$"),
        Patch(fc=COLOR[2], ec="k", label=r"Anomaly $\beta_j<0$"),
    ]
    if recovered is not None:
        mu, sd = float(recovered[0]), float(recovered[1])
        # marker offset sideways so it does not sit on the bar's value label;
        # the note goes in the empty top-left corner rather than on a leader
        # line across the bars
        ax.errorbar(
            [x[0] + 0.26],
            [mu],
            yerr=[sd],
            fmt="D",
            ms=7,
            color="k",
            mfc="w",
            mew=1.6,
            capsize=4,
            zorder=6,
        )
        handles.append(
            Line2D(
                [],
                [],
                marker="D",
                ls="none",
                color="k",
                mfc="w",
                mew=1.6,
                label=r"estimated $\tilde\beta$ $\pm1\sigma$",
            )
        )
    ax.legend(handles=handles, fontsize=8, loc="lower right")


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


def draw_cylinder(ax, cyl, color="crimson", alpha=0.20, n_th=48, lw=0.9, label=None):
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


def cylinder_hull(cyl, n_th=16):
    """Points spanning a cylinder — for including it in the axis limits."""
    th = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    zz = np.array([0.0, cyl.height])
    TH, ZZ = np.meshgrid(th, zz)
    loc = np.stack([cyl.radius * np.cos(TH), cyl.radius * np.sin(TH), ZZ], axis=-1)
    return (loc.reshape(-1, 3) @ cyl.rot().T) + cyl.center


def cov_ellipse(ax, mean2, cov2, color, nsig=1.0, **kw):
    """1σ (or nσ) error ellipse of a 2-D covariance, for the estimate clouds."""
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.maximum(vals, 0.0)
    t = np.linspace(0.0, 2.0 * np.pi, 240)
    e = vecs @ (nsig * np.sqrt(vals)[:, None] * np.array([np.cos(t), np.sin(t)]))
    ax.plot(mean2[0] + e[0], mean2[1] + e[1], color=color, **kw)


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


def make_plots(res, outdir="Images"):
    os.makedirs(outdir, exist_ok=True)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    V, F, P = res["V"], res["F"], res["P"]
    cyl, obs, names = res["cyl"], res["obs"], res["names"]
    tgt = res["target"]

    # ---- FIG 1: the interior model in 3-D -----------------------------------
    det = res["det"]
    ft = res["f_true"]
    fig = plt.figure(figsize=(8.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, len(F) // 8000)
    pc = Poly3DCollection(
        V[F[::step]], alpha=0.18, facecolor="#9ecae1", edgecolor="0.55", linewidths=0.1
    )
    ax.add_collection3d(pc)
    draw_cylinder(ax, cyl, label="CH cylinder")
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
        ax.text(p_a[0], p_a[1], p_a[2], f"  {nm.split()[0]}", fontsize=8)
    ax.set_title("Interior: constant-density body plus signed anomalies")
    ax.set_xlabel("x [LU]")
    ax.set_ylabel("y [LU]")
    ax.set_zlabel("z [LU]")
    set_axes_true_shape(ax, np.vstack([V, cylinder_hull(cyl)]))
    ax.scatter([], [], color=COLOR[0], label=r"anomaly $\beta_j>0$")
    ax.scatter([], [], color=COLOR[2], label=r"anomaly $\beta_j<0$")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, "global_fig1_geometry.pdf"),
        dpi=180,
        bbox_inches="tight",
    )

    # ---- FIG 2: mass budget + EXPERIMENT 1 + detection ----------------------
    fig = plt.figure(figsize=(19.5, 5.8))
    # no wspace= here: it sets locally_modified_subplot_params(), which makes
    # tight_layout below declare every axes incompatible, warn, and do nothing.
    # Horizontal spacing comes from tight_layout's w_pad instead.
    gs1 = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.45, 1.15])
    fig.suptitle(
        "Experiment 1: mass fractions, positions fixed",
        fontweight="bold",
        y=1.02,
    )
    draw_mass_budget(
        fig.add_subplot(gs1[0, 0]),
        names,
        ft,
        res["beta_bulk"],
        recovered=res["bulk_nominal"],
    )

    # (b) EXPERIMENT 1 — spread of sigma_beta ACROSS TRUTH INTERIORS
    ax = fig.add_subplot(gs1[0, 1])
    tmc = res["truth_mc"]
    allA = np.column_stack([tmc["sigA"], tmc["bulkA"]])
    allB = np.column_stack([tmc["sigB"], tmc["bulkB"]])
    labels = [n.replace(" ", "\n", 1) for n in names] + ["BODY\n$\\tilde\\beta$"]
    xpos = np.arange(len(labels), dtype=float) * 1.35  # extra room per group
    w = 0.44
    medA, medB = np.median(allA, axis=0), np.median(allB, axis=0)
    for off, M, med, col, lab in (
        (-w / 2, allA, medA, COLOR[2], "SH only"),
        (+w / 2, allB, medB, COLOR[0], "SH + CH"),
    ):
        lo = med - np.percentile(M, 10, axis=0)
        hi = np.percentile(M, 90, axis=0) - med
        ax.bar(
            xpos + off,
            med,
            w,
            yerr=np.vstack([lo, hi]),
            capsize=5,
            color=col,
            edgecolor="k",
            label=lab,
            error_kw=dict(lw=1.3, ecolor="0.2"),
        )
    # the one number the figure keeps: how much CH improves each component
    gain = np.median(allA / allB, axis=0)
    top = np.percentile(allB, 90, axis=0)
    for k in range(len(labels)):
        ax.text(
            xpos[k] + w / 2,
            top[k] * 1.45,
            rf"${gain[k]:.0f}\times$",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=COLOR[0],
        )
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(xpos[0] - 0.9, xpos[-1] + 0.9)
    ax.set_ylim(top=np.percentile(allA, 90, axis=0).max() * 6)
    ax.set_ylabel(r"Mass-fraction 1$\sigma$   $\sigma_\beta$", fontsize=11)
    ax.set_title(
        "Mass-fraction uncertainty over truth interiors\n"
        "(bar = median, whiskers = 10-90 percentile)",
        fontsize=11,
    )
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.93)

    # (c) smallest detectable anomaly: recovered anomaly vs true anomaly
    ax = fig.add_subplot(gs1[0, 2])
    mug = det["mu_grid"]
    ax.plot(mug, mug, "k--", lw=1, label="Perfect Recovery")
    ax.errorbar(
        mug,
        np.abs(det["muA"]),
        yerr=det["sdA"],
        fmt="o",
        color=COLOR[2],
        ms=5,
        capsize=3,
        label="SH",
    )
    ax.errorbar(
        mug,
        np.abs(det["muB"]),
        yerr=det["sdB"],
        fmt="s",
        color=COLOR[0],
        ms=5,
        capsize=3,
        label="SH + CH",
    )
    ax.axhline(
        det["thr_A"],
        color=COLOR[2],
        ls=":",
        lw=1.5,
        label=r"SH $3\sigma$ floor",
    )
    ax.axhline(
        det["thr_B"],
        color=COLOR[0],
        ls=":",
        lw=1.5,
        label=r"SH + CH $3\sigma$ floor",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"True Anomaly $\beta_0$")
    ax.set_ylabel(r"Recovered Anomaly  $|\hat\mu|$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # (b) the truth mass budget: body + anomalies = M*
    fig.tight_layout(w_pad=2.2)
    fig.savefig(
        os.path.join(outdir, "global_fig2_massfraction.pdf"),
        dpi=180,
        bbox_inches="tight",
    )

    # ---- FIG 3: EXPERIMENT 2 — position recovery over TRUTH POSITIONS -------
    # (a) how the per-truth error is distributed, (b) where the truths were
    # drawn, (c) the estimator's own scatter for one of them.
    tmc = res["truth_mc"]
    eA, eB = tmc["errA"], tmc["errB"]
    pos, cylc = tmc["pos"], res["cyl"].center
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))

    # (a) distribution of the per-truth RMS position error
    ax = axes[0]
    bins = np.logspace(
        np.log10(min(eA.min(), eB.min()) * 0.8),
        np.log10(max(eA.max(), eB.max()) * 1.2),
        26,
    )
    ax.hist(
        eA,
        bins=bins,
        color=COLOR[2],
        alpha=0.75,
        edgecolor="k",
        lw=0.5,
        label="SH",
    )
    ax.hist(
        eB,
        bins=bins,
        color=COLOR[0],
        alpha=0.75,
        edgecolor="k",
        lw=0.5,
        label="SH + CH",
    )
    for v, col in ((np.median(eA), COLOR[2]), (np.median(eB), COLOR[0])):
        ax.axvline(v, color=col, ls="--", lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Position RMS Error [LU]")
    ax.set_ylabel(f"Truth Interiors  (of {len(eA)})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # (b) where the truth anomalies were drawn
    ax = axes[1]
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
        color="crimson",
        ms=12,
        zorder=6,
        label="CH cylinder",
    )
    ax.set_xlabel("x [LU]")
    ax.set_ylabel("z [LU]")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="lower right")

    # (c) ONE truth: BOTH estimates with their analytic covariances.  The main
    # axes are scaled to the SH cloud; SH+CH is ~20x tighter and collapses to a
    # dot there, so it gets an inset at its own scale.
    ax = axes[2]
    ix = np.ix_([0, 2], [0, 2])
    cA, cB, p_rep = tmc["cloudA"], tmc["cloudB"], tmc["rep_truth"]
    covA, covB = tmc["covA"][ix], tmc["covB"][ix]
    xzA, xzB = cA[:, [0, 2]], cB[:, [0, 2]]
    muA, muB = xzA.mean(axis=0), xzB.mean(axis=0)
    tru = p_rep[[0, 2]]

    ax.scatter(
        xzA[:, 0],
        xzA[:, 1],
        s=20,
        color=COLOR[2],
        edgecolor="none",
        alpha=0.45,
        zorder=3,
        label="SH",
    )
    cov_ellipse(
        ax,
        muA,
        covA,
        COLOR[2],
        nsig=1.0,
        lw=2.2,
        zorder=6,
        label=r"SH 1$\sigma$ analytic",
    )
    ax.scatter(
        xzB[:, 0],
        xzB[:, 1],
        s=16,
        color=COLOR[0],
        edgecolor="none",
        alpha=0.7,
        zorder=4,
        label="SH + CH",
    )
    cov_ellipse(
        ax,
        muB,
        covB,
        COLOR[0],
        nsig=1.0,
        lw=2.2,
        zorder=7,
        label=r"SH + CH 1$\sigma$ analytic",
    )
    ax.plot(*tru, "*", color=COLOR[1], ms=17, mec="k", mew=0.8, zorder=8, label="Truth")
    rA = 3.4 * max(np.sqrt(covA[0, 0]), np.sqrt(covA[1, 1]), np.linalg.norm(muA - tru))
    ax.set_xlim(muA[0] - rA, muA[0] + rA)
    ax.set_ylim(muA[1] - rA, muA[1] + rA)
    ax.set_aspect("equal")
    ax.set_xlabel("x [LU]")
    ax.set_ylabel("z [LU]")
    ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
    sA = np.sqrt(np.trace(covA) / 2)
    sB = np.sqrt(np.trace(covB) / 2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)

    # inset at the SH+CH scale
    # opaque, or the SH cloud shows through and the zoom is unreadable
    # [x0, y0, width, height] in axes fraction — the last two set the size
    axin = ax.inset_axes([0.62, 0.05, 0.35, 0.35], facecolor="white")
    axin.set_zorder(10)
    axin.patch.set_alpha(1.0)
    axin.scatter(
        xzB[:, 0], xzB[:, 1], s=14, color=COLOR[0], edgecolor="none", alpha=0.6
    )
    cov_ellipse(axin, muB, covB, COLOR[0], nsig=1.0, lw=1.8)
    cov_ellipse(axin, muB, np.cov(xzB.T), "0.4", nsig=1.0, lw=1.3, ls="--")
    axin.plot(*muB, "P", color="k", ms=8)
    axin.plot(*tru, "*", color=COLOR[1], ms=13, mec="k", mew=0.6)
    rB = 3.4 * max(np.sqrt(covB[0, 0]), np.sqrt(covB[1, 1]), np.linalg.norm(muB - tru))
    axin.set_xlim(muB[0] - rB, muB[0] + rB)
    axin.set_ylim(muB[1] - rB, muB[1] + rB)
    axin.set_aspect("equal")
    axin.set_title("SH + CH zoom", fontsize=8, color=COLOR[0], pad=2)
    axin.tick_params(labelsize=5.5)
    axin.locator_params(nbins=3)
    for sp_ in axin.spines.values():
        sp_.set_edgecolor(COLOR[0])
    # no indicate_inset_zoom: the zoomed region is smaller than a marker here,
    # so the connectors are long diagonals pointing at nothing visible
    ax.plot(*muB, "o", mfc="none", mec=COLOR[0], ms=15, mew=1.6, zorder=9)

    fig.suptitle(
        "Experiment 2: anomaly position, Monte-Carlo over truth " "interiors",
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout(w_pad=2.0)
    fig.savefig(
        os.path.join(outdir, "global_fig3_position.pdf"), dpi=180, bbox_inches="tight"
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

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    handles = None
    for ax, (key, name, xlab) in zip(
        axes,
        [
            ("sh", "SPHERICAL harmonics", "SH degree $n$"),
            ("ch", "CYLINDRICAL harmonics", "CH azimuthal order $m$"),
        ],
    ):
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
        # pre/post RMS in the panel titles, which are properly whitened.
        y_pre, y_post, y_sig = rms(np.abs(pre)), rms(np.abs(post)), rms(d["sigma"])
        rms_pre = np.sqrt(np.mean((pre / d["sigma"]) ** 2))
        rms_post = np.sqrt(np.mean((post / d["sigma"]) ** 2))
        ax.plot(xs, y_sig, "-o", lw=1.6, color="0.30", zorder=2, label=r"1$\sigma$")
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
        ax.set_ylabel("RMS |residual|")
        ax.set_yscale("log")
        ax.set_ylim(0.5 * min(y_post.min(), y_sig.min()), 2.5 * y_pre.max())
        ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.45)
        ax.set_axisbelow(True)
        for sd_ in ("top", "right"):
            ax.spines[sd_].set_visible(False)
        ax.set_title(
            name,
            fontweight="bold",
            fontsize=11,
            pad=9,
        )
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    # one legend for both panels, in a reserved strip under the axes — an
    # in-axes legend here covered the degree-3 peak
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=9.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.savefig(
        os.path.join(outdir, "global_fig4_coefficients.pdf"),
        dpi=180,
        bbox_inches="tight",
    )

    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    res = run_experiment(
        Lmax_sh=6,  # observable spherical-harmonic degree (tracking limit)
        eps=0.02,  # relative measurement precision (same on SH & field)
        ch_modes=(8, 8),  # (n_m, n_n) cylindrical-harmonic truncation
        n_cyl_pts=200,
        n_mc=400,  # noise draws for the linear mass fit
        n_mc_nl=150,  # noise draws for the nonlinear masses+position fit
        outdir="Images",
        verbose=True,
    )
    print(
        "\nSaved: Images/global_fig1_geometry.pdf, "
        "global_fig2_massfraction.pdf, global_fig3_position.pdf, "
        "global_fig4_coefficients.pdf"
    )
    print("Done.")
