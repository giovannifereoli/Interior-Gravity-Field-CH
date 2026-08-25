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
mpl.rcParams.update(
    {
        "axes.prop_cycle": mpl.cycler(color=COLOR),
        "font.family": "serif",
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


def run_experiment(
    Lmax_sh=6,
    eps=0.02,
    ch_modes=(8, 8),
    n_cyl_pts=200,
    n_mc=400,
    n_mc_nl=150,
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
    if verbose:
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
    if verbose:
        print(
            f"\n{'-'*70}\n  PART 2 — POSITION OF NEAR-SURFACE ANOMALY "
            f"(β={f_true[target]:+.3f})\n{'-'*70}"
        )
        print(
            f"  position 1σ RMS:  SH={posA['rms']:.3e} LU   SH+CH={posB['rms']:.3e} LU"
            f"   → {posA['rms']/posB['rms']:.0f}× tighter"
        )

    # ══ EXPERIMENT 1 — MASS FRACTIONS (all positions FIXED) ════════════════
    # Monte-Carlo linear least squares on coefficient observables: SH Stokes
    # discrepancy (Case A), plus the fitted CH coefficients of the near-surface
    # discrepancy (Case B).  The three anomaly positions are held at truth, and
    # the bulk fraction β̃ = 1 − Σβ follows from the fit rather than being fitted.
    mcA = monte_carlo_fit(blocksA, f_true, n_mc=n_mc)
    mcB = monte_carlo_fit(blocksB, f_true, n_mc=n_mc)
    fitA = dict(
        mean=mcA.mean(0),
        std=mcA.std(0),
        samples=mcA,
        bulk_mean=(1.0 - mcA.sum(1)).mean(),
        bulk_std=(1.0 - mcA.sum(1)).std(),
    )
    fitB = dict(
        mean=mcB.mean(0),
        std=mcB.std(0),
        samples=mcB,
        bulk_mean=(1.0 - mcB.sum(1)).mean(),
        bulk_std=(1.0 - mcB.sum(1)).std(),
    )
    if verbose:
        print(
            f"\n{'='*70}\n  EXPERIMENT 1 — MASS FRACTIONS, positions FIXED "
            f"({n_mc} draws)\n{'='*70}"
        )
        print(
            f"  {'quantity':22s} {'truth':>8} | {'SH: mean±std':>20} {'err%':>6}"
            f" | {'SH+CH: mean±std':>20} {'err%':>6}"
        )
        for k, nm in enumerate(names):
            ta = f"{fitA['mean'][k]:+.4f}±{fitA['std'][k]:.4f}"
            tb = f"{fitB['mean'][k]:+.4f}±{fitB['std'][k]:.4f}"
            ea = 100 * abs(fitA["mean"][k] - f_true[k]) / abs(f_true[k])
            eb = 100 * abs(fitB["mean"][k] - f_true[k]) / abs(f_true[k])
            print(
                f"  {nm:22s} {f_true[k]:+8.3f} | {ta:>20} {ea:5.1f}% | {tb:>20} {eb:5.1f}%"
            )
        ta = f"{fitA['bulk_mean']:+.4f}±{fitA['bulk_std']:.4f}"
        tb = f"{fitB['bulk_mean']:+.4f}±{fitB['bulk_std']:.4f}"
        print(
            f"  {'BULK β̃ = 1 − Σβ':21s} {beta_bulk:+8.3f} | {ta:>20} {'':5} | {tb:>20}"
        )

    # ── COEFFICIENT SPECTRA: homogeneous vs heterogeneous, pre/post fit ────
    # The whole parameterization in one picture.  The HOMOGENEOUS body gives
    # CS_CD (SH) and Φ⁺·U_CD (CH); the HETEROGENEOUS truth adds the anomalies;
    # their difference ΔCS is the only thing the estimator ever sees, and it has
    # to stand above the per-coefficient noise σ.  One noisy realization is then
    # fitted so the post-fit model and residual can be shown against the data.
    rng_sp = np.random.default_rng(99)
    d_sh, d_ch = A_sh @ f_true, A_ch @ f_true  # = CS_hetero − CS_homog
    dat_sh = d_sh + rng_sp.normal(0.0, sig_sh)
    dat_ch = d_ch + rng_sp.normal(0.0, sig_ch)
    Aw = np.vstack([A_sh / sig_sh[:, None], A_ch / sig_ch[:, None]])
    yw = np.concatenate([dat_sh / sig_sh, dat_ch / sig_ch])
    beta_hat, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
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
    if verbose:
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
    if verbose:
        print(f"\n  smallest detectable anomaly (3σ fit scatter):")
        print(f"    SH only : μ_min = {det['thr_A']:.2e}")
        print(
            f"    SH + CH : μ_min = {det['thr_B']:.2e}   "
            f"→ {det['thr_A']/det['thr_B']:.0f}× smaller anomaly detectable"
        )

    # ══ EXPERIMENT 2 — ANOMALY POSITION (all masses FIXED) ═════════════════
    # Monte-Carlo NONLINEAR least squares for the anomaly's (x,y,z); the three
    # mass fractions are held at truth.  Nothing is estimated jointly with mass.
    if verbose:
        print(
            f"\n{'='*70}\n  EXPERIMENT 2 — ANOMALY POSITION, masses FIXED "
            f"({n_mc_nl} draws)\n{'='*70}"
        )
    posA_nl = position_mc(
        P,
        f_true,
        obs,
        cyl,
        ch_modes,
        Lmax_sh,
        Rref,
        sig_sh,
        sig_ch,
        bulk,
        use_ch=False,
        n_mc=n_mc_nl,
    )
    posB_nl = position_mc(
        P,
        f_true,
        obs,
        cyl,
        ch_modes,
        Lmax_sh,
        Rref,
        sig_sh,
        sig_ch,
        bulk,
        use_ch=True,
        n_mc=n_mc_nl,
    )
    pos_rmsA = float(np.sqrt(np.mean(np.sum((posA_nl - P[target]) ** 2, axis=1))))
    pos_rmsB = float(np.sqrt(np.mean(np.sum((posB_nl - P[target]) ** 2, axis=1))))
    nl = dict(nlA=posA_nl, nlB=posB_nl, pos_rmsA=pos_rmsA, pos_rmsB=pos_rmsB)
    if verbose:
        biasA = np.linalg.norm(posA_nl.mean(0) - P[target])
        biasB = np.linalg.norm(posB_nl.mean(0) - P[target])
        print(
            f"  anomaly POSITION RMS error:  SH={pos_rmsA:.3e} LU  "
            f"SH+CH={pos_rmsB:.3e} LU  → {pos_rmsA/pos_rmsB:.0f}× tighter"
        )
        print(
            f"  recovered-mean bias:         SH={biasA:.2e}  SH+CH={biasB:.2e}  "
            f"(both ≈ unbiased)"
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
        fitA=fitA,
        fitB=fitB,
        det=det,
        nl=nl,
        spectra=spectra,
        Lmax_sh=Lmax_sh,
        ch_modes=ch_modes,
        sig_sh=sig_sh,
        sig_ch=sig_ch,
    )
    make_plots(res, outdir=outdir)
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
    for k, v in enumerate(vals):
        ax.text(
            x[k],
            ends[k],
            f"{v:.3f}" if k == 0 else f"{v:+.3f}",
            ha="center",
            va="bottom" if v > 0 else "top",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["BODY\n(const. density)"] + [n.split()[0] for n in names], fontsize=8
    )
    ax.set_ylim(lo, hi)
    ax.set_ylabel("cumulative mass fraction of $M^*$")
    ax.set_title(
        r"TRUTH mass budget:  $\tilde\beta + \sum_j \beta_j = 1$"
        "\n(body bar runs off the axis bottom)",
        fontsize=10.5,
    )
    ax.grid(True, axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    handles = [
        Patch(fc="0.55", ec="k", label=r"body $\tilde\beta$ (truth)"),
        Patch(fc=COLOR[0], ec="k", label=r"anomaly $\beta_j>0$"),
        Patch(fc=COLOR[2], ec="k", label=r"anomaly $\beta_j<0$"),
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
        ax.text(
            0.015,
            0.985,
            f"TRUTH      $\\tilde\\beta$ = {beta_bulk:.4f}   (bars)\n"
            f"ESTIMATED  $\\tilde\\beta$ = {mu:.4f} ± {sd:.4f}   (◇)\n"
            f"{rec_label} — derived as $1-\\sum_j\\hat\\beta_j$, "
            f"never assumed",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="top",
            family="monospace",
            bbox=dict(fc="white", ec="0.6", alpha=0.93),
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
                label=r"estimated $\tilde\beta$ ±1σ",
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

    # ---- FIG 1: geometry + mass-fraction recovery + detection ---------------
    fitA, fitB, det = res["fitA"], res["fitB"], res["det"]
    ft = res["f_true"]
    fig = plt.figure(figsize=(23, 5.4))
    fig.suptitle(
        "EXPERIMENT 1 — mass fractions, positions fixed "
        f"({len(fitA['samples'])} MC draws)",
        fontweight="bold",
        y=1.02,
    )

    # (a) the interior model and the near-surface data
    ax = fig.add_subplot(1, 4, 1, projection="3d")
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
        ax.text(
            p_a[0], p_a[1], p_a[2], f"  {nm.split()[0]} ({ft[i_a]:+.3f})", fontsize=8
        )
    ax.set_title(
        "Interior = constant-density BODY "
        f"($\\tilde\\beta$ = {res['beta_bulk']:.3f})\n"
        f"+ {len(P)} anomalies ($\\beta_j$, signed)"
    )
    ax.set_xlabel("x [LU]")
    ax.set_ylabel("y [LU]")
    ax.set_zlabel("z [LU]")
    set_axes_true_shape(ax, np.vstack([V, cylinder_hull(cyl)]))
    ax.scatter([], [], color=COLOR[0], label=r"anomaly $\beta_j>0$")
    ax.scatter([], [], color=COLOR[2], label=r"anomaly $\beta_j<0$")
    ax.legend(loc="upper left", fontsize=8)

    # (b) actual-fit recovery ERROR (RMS = bias ⊕ MC scatter) per quantity
    ax = fig.add_subplot(1, 4, 3)
    labels = [n.replace(" ", "\n", 1) for n in names] + ["BODY\n$\\tilde\\beta$"]
    xpos = np.arange(len(labels))
    truth = np.concatenate([ft, [res["beta_bulk"]]])
    meanA = np.concatenate([fitA["mean"], [fitA["bulk_mean"]]])
    stdA = np.concatenate([fitA["std"], [fitA["bulk_std"]]])
    meanB = np.concatenate([fitB["mean"], [fitB["bulk_mean"]]])
    stdB = np.concatenate([fitB["std"], [fitB["bulk_std"]]])
    rmsA = np.sqrt((meanA - truth) ** 2 + stdA**2)
    rmsB = np.sqrt((meanB - truth) ** 2 + stdB**2)
    w = 0.38
    ax.bar(xpos - w / 2, rmsA, w, color=COLOR[2], edgecolor="k", label="SH only")
    ax.bar(xpos + w / 2, rmsB, w, color=COLOR[0], edgecolor="k", label="SH + CH")
    ax.set_yscale("log")
    for i_b in range(len(labels)):
        ax.text(
            i_b + w / 2,
            rmsB[i_b],
            f"{rmsA[i_b]/rmsB[i_b]:.0f}×",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mass-fraction recovery RMS error")
    ax.set_title("Recovery error (bias ⊕ MC scatter)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=10)

    # (c) smallest detectable anomaly: recovered anomaly vs true anomaly
    ax = fig.add_subplot(1, 4, 4)
    mug = det["mu_grid"]
    ax.plot(mug, mug, "k--", lw=1, label="perfect recovery")
    ax.errorbar(
        mug,
        np.abs(det["muA"]),
        yerr=det["sdA"],
        fmt="o",
        color=COLOR[2],
        ms=5,
        capsize=3,
        label="SH only",
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
        label=f"SH 3σ floor = {det['thr_A']:.1e}",
    )
    ax.axhline(
        det["thr_B"],
        color=COLOR[0],
        ls=":",
        lw=1.5,
        label=f"SH+CH 3σ floor = {det['thr_B']:.1e}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"true anomaly mass fraction  $\beta_0$")
    ax.set_ylabel(r"recovered anomaly  $|\hat\mu|$")
    ax.set_title(
        f"Smallest detectable anomaly "
        f"({det['thr_A']/det['thr_B']:.0f}× smaller with CH)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # (b) the truth mass budget: body + anomalies = M*
    draw_mass_budget(
        fig.add_subplot(1, 4, 2),
        names,
        ft,
        res["beta_bulk"],
        recovered=(fitB["bulk_mean"], fitB["bulk_std"]),
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, "global_fig1_massfraction.pdf"),
        dpi=180,
        bbox_inches="tight",
    )

    # ---- FIG 2: EXPERIMENT 2 — anomaly POSITION recovery (masses fixed) ------
    # The nonlinear MC fit, not the linearized covariance: with Parts 1–2 and
    # Experiments 1–2 now sharing one observation model the two agree, so only
    # the actual fit is plotted.
    nl = res["nl"]
    nlA, nlB = nl["nlA"], nl["nlB"]  # (n_mc, 3) position samples
    p0 = P[tgt]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))

    # (a) position-error magnitude histogram
    ax = axes[0]
    errA = np.linalg.norm(nlA - p0, axis=1)
    errB = np.linalg.norm(nlB - p0, axis=1)
    bins = np.logspace(np.log10(min(errB.min(), 1e-6)), np.log10(errA.max() + 1e-9), 30)
    ax.hist(errA, bins=bins, color=COLOR[2], alpha=0.6, label="SH only")
    ax.hist(errB, bins=bins, color=COLOR[0], alpha=0.7, label="SH + CH")
    ax.set_xscale("log")
    ax.set_xlabel(r"position error $|\hat p - p_{\rm true}|$ [LU]")
    ax.set_ylabel("MC count")
    ax.set_title(
        f"Position error (RMS {nl['pos_rmsA']:.1e}→{nl['pos_rmsB']:.1e} LU, "
        f"{nl['pos_rmsA']/nl['pos_rmsB']:.0f}×)"
    )
    ax.legend(fontsize=9)

    # (b,c) recovered position clouds over Eros silhouette (x–z, y–z)
    for ax, (i_c, j_c), lbl in [
        (axes[1], (0, 2), ("x", "z")),
        (axes[2], (1, 2), ("y", "z")),
    ]:
        draw_silhouette(ax, V, F, i_c, j_c)
        ax.scatter(
            nlA[:, i_c], nlA[:, j_c], s=12, color=COLOR[2], alpha=0.5, label="SH only"
        )
        ax.scatter(
            nlB[:, i_c], nlB[:, j_c], s=12, color=COLOR[0], alpha=0.7, label="SH + CH"
        )
        ax.plot(p0[i_c], p0[j_c], "k*", ms=16, label="truth", zorder=6)
        sA = max(nlA[:, i_c].std(), nlA[:, j_c].std())
        ax.set_xlim(p0[i_c] - 5 * sA, p0[i_c] + 5 * sA)
        ax.set_ylim(p0[j_c] - 5 * sA, p0[j_c] + 5 * sA)
        ax.set_xlabel(f"{lbl[0]} [LU]")
        ax.set_ylabel(f"{lbl[1]} [LU]")
        ax.set_title(f"Recovered position ({lbl[0]}–{lbl[1]})")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

        # zoom inset: SH+CH cloud is far tighter — invisible at the SH scale
        axin = ax.inset_axes([0.04, 0.06, 0.32, 0.32])
        axin.scatter(nlB[:, i_c], nlB[:, j_c], s=8, color=COLOR[0], alpha=0.7)
        axin.plot(p0[i_c], p0[j_c], "k*", ms=8)
        sB = max(nlB[:, i_c].std(), nlB[:, j_c].std(), 1e-9)
        axin.set_xlim(p0[i_c] - 4 * sB, p0[i_c] + 4 * sB)
        axin.set_ylim(p0[j_c] - 4 * sB, p0[j_c] + 4 * sB)
        axin.set_title("SH+CH zoom (×%d)" % round(sA / sB), fontsize=8, color=COLOR[0])
        axin.tick_params(labelsize=6)
        axin.set_aspect("equal")
        for sp_ in axin.spines.values():
            sp_.set_edgecolor(COLOR[0])
    fig.suptitle(
        f"EXPERIMENT 2 — anomaly position, masses fixed ({len(nlA)} MC draws):  "
        f"RMS  SH {nl['pos_rmsA']:.2e} LU → SH+CH {nl['pos_rmsB']:.2e} LU  "
        f"({nl['pos_rmsA']/nl['pos_rmsB']:.0f}× tighter)",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        os.path.join(outdir, "global_fig2_position.pdf"), dpi=180, bbox_inches="tight"
    )

    # ---- FIG 3: residual power spectrum, before and after the fit ----------
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
            label="PRE-fit: measured − homogeneous",
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
            label=r"POST-fit: measured − (homog. + A$\hat\beta$)",
        )

        ax.set_xticks(xs)
        ax.set_xlabel(xlab)
        ax.set_ylabel("RMS |residual|  per " + ("degree" if key == "sh" else "order"))
        ax.set_yscale("log")
        ax.set_ylim(0.5 * min(y_post.min(), y_sig.min()), 2.5 * y_pre.max())
        ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.45)
        ax.set_axisbelow(True)
        for sd_ in ("top", "right"):
            ax.spines[sd_].set_visible(False)
        ax.set_title(
            f"{name}\nRMS  pre-fit {rms_pre:.1f}$\\,\\sigma$   →   "
            f"post-fit {rms_post:.2f}$\\,\\sigma$",
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
        os.path.join(outdir, "global_fig3_coefficients.pdf"),
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
        "\nSaved: Images/global_fig1_massfraction.pdf, "
        "global_fig2_position.pdf, global_fig3_coefficients.pdf"
    )
    print("Done.")
