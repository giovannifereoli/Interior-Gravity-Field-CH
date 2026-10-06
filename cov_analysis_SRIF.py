#!/usr/bin/env python3
"""
cov_analysis_SRIF.py — how many CH degrees / orders can a realistic orbit-
determination campaign actually deliver over the pt1 polar cylinder of Eros?

Multi-arc square-root information filter (SRIF) covariance analysis built on
the SAME objects as the density scripts (`cylinder_mass_estimation_GLOBAL*`):
the pt1 cylinder over the +z pole, its 200 field samples, the constant-density
bulk + anomaly-mascon truth, and the [A_mn, B_mn] packing of `cyl_basis`.

What is estimated
-----------------
The RAW CH coefficients A_mn, B_mn of the CH_MASTER basis, CH DEGREE m
(the azimuthal index, the convention of the density scripts) and ORDER n (the
radial Bessel index), B_0n ≡ 0 dropped: (12, 12), the density scripts'
"new" modes, i.e. 276 parameters, no rotation, no truncation.  The Bessel
columns of neighbouring orders n are nearly dependent over one patch (cond Φ
~ 1e18): the OD knows the field to a few 0.1% while single coefficients stay
near their prior; fewer orders per degree raise the coefficient SNR at the
price of a larger omission (BASIS_CASES prints the trade-off).  Nothing regularizes the
near-dependence except the prior, so weakly determined combinations stay at
their prior.  The one place it is handled is the DEFINITION of the truth and
of the prior: the coefficients of a field f are  c = Φ⁺ f  with the truncated
SVD of the density scripts (TRUTH_RCOND, the "new" preset), the truth for the
true interior and, for the prior, refit to a Monte Carlo of shape models and
bulk densities (what the filter's fixed constant-density bulk can get wrong;
prior = the FULL sample covariance of the refit coefficients, scaled so that
no truth coefficient exceeds PRIOR_SNR_MAX prior σ,
of the rank of Φ⁺: the filter carries it as  c = σ ∘ (Q z),  z ~ N(0, I)).  The coefficient
covariance is exported (od_ch_covariance.npz), and so is that of every
density-script preset ((8,8), (12,12)) the basis contains, to stand in for the
ad-hoc `od_sigma` rule.

OD model (km, s; body-fixed rotating frame, ω = 2π / 5.270 h about +z)
------------------------------------------------------------------------
  dynamics   r̈ = g(r) − 2ω×v − ω×(ω×r) + Rᵀ(t) a_SA
             truth g = β̃·bulk + Σ β_j·point masses (polyhedral bulk), RK4 10 s;
             the FILTER model is  bulk + CH(A, B)  inside the cylinder and
             the known field outside it (the global field of the mapping
             orbits); the non-gravitational mismodelling (SRP, thermal,
             desaturations) goes to a_SA.
  campaign   N_ARCS low-pass sorties from a circular polar home orbit
             (25 / 35 / 50 km; NEAR flew 50 and 35 km): a burn at apoapsis
             lowers the periapsis into the cylinder (1.5–5 km above its base,
             up to ±0.85 R off the axis, i.e. inclination 75–90°), the burn at
             the next apoapsis recircularizes.  One inertial plane (normal
             toward the Sun's equatorial projection, the polar plane closest
             to a terminator orbit); the timing, i.e. the body phase at
             periapsis, sets the pass azimuth over the cylinder.  Tracking
             arc = the ±2.5 h about periapsis, inside the two burns.  Screened
             for ≥ 1 km altitude and chosen greedily to cover the cylinder
             volume in (ρ, φ, z).
  data       DSN radiometric tracking along the Earth line of sight (fixed
             inertial direction, rotated into the body frame at every epoch;
             no data while Eros occults the Earth, ray-cast on the mesh):
             two-way X-band Doppler, one 60-s count per epoch (0.1 mm/s), and
             two-way ranging every 10 min (1 m);  nadir NavCam landmarks
             (IFOV 0.3 mrad, 0.25 px) every 5 min, with emission / incidence /
             FOV / occlusion visibility.  The s/c attitude knowledge (2e-5 rad
             per axis, white between epochs) is ONE error shared by every
             landmark of the same image (joint whitening); the radio data
             do not depend on it.
  locals     per arc: r, v, a_SA.
  globals    A_mn, B_mn (prior: full covariance of a shape × bulk-density Monte
             Carlo, scaled to |c_true|/σ ≤ PRIOR_SNR_MAX), landmark positions.
  consider   (i) the CH omission: the part of the contrast field the truth
             coefficients leave out, one unestimated column s (truth s = 1, filter
             s = 0) — a bias R⁻¹ R_s;  (ii) constant instrument / frame
             errors, unestimated columns c ~ N(0, σ²): range bias (ranging
             only, Doppler unbiased) and camera alignment — a covariance
             R⁻¹ R_c (R⁻¹ R_c)ᵀ added to the formal one.
  stoch.     a_SA: inertial stochastic acceleration, piecewise constant over
  accel.     1-h batches, uncorrelated between batches, 1e-12 km/s² per axis
             (batched dynamic model compensation; no SNC).  Sensitivity runs:
             σ_SA × 10 and 20-min batches at the same PSD σ²τ.
  filter     MULTI-ARC SRIF: the locals are arc-specific, A_mn, B_mn and the landmarks
             are common to every arc.  Arcs are processed in sequence; at each
             arc boundary the finished arc's locals are marginalized out and the
             next arc's start at their prior (a_SA likewise at every batch
             boundary), which leaves exactly the summed per-arc global
             information.  The formal σ of every state is kept at every epoch.

A coefficient counts as RESOLVED when σ_post/σ_prior < 1/√2, i.e. the data
carry more information on it than the prior does, and ACCURATE when the same
holds for the total error √(σ_post² + σ_consider² + bias²).  A truncation sweep re-solves the same
information array estimating only m < N, n ≤ N and gives the
acceleration-field error vs N (noise rises, truncation + aliasing fall): the
degree / order the OD can actually use.

Run time
--------
The campaign design (screening + full-truth propagation, ~3 min) and the
prior Monte Carlo are cached in CACHE_DIR (~/.cache/od_srif), keyed by the
source of the code that computes them, the settings it reads and the shape
model: they are recomputed only when one of those changes (OD_NO_CACHE=1
forces it).  The SRIF updates its triangular array in O(m n²) per epoch
(LAPACK tpqrt) and inverts it only where σ are recorded.  OD_QUICK=1 runs the
nominal filter only (no σ_SA cases, no basis sweep) while iterating on the
figures; the exported npz is the same.  OD_NO_TEX=1 draws without LaTeX.
"""

import hashlib
import inspect
import os
import sys
import time
import types

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.lapack import dtpmqrt, dtpqrt, dtrtri
from scipy.spatial import cKDTree
from polyhedral_gravity import GravityEvaluable, Polyhedron, PolyhedronIntegrity

import cylinder_mass_estimation_GLOBAL as G  # also applies the shared plot style

# NOTE: For J1, perform pure covariance propagation using Monte Carlo.
# It is going to be a paper on dynamics.  Perhaps CMDA.  Then you will
# write a paper on JGCD or something related to engineering applications,
# complete with cases.  There, you can run multiple cases while also
# implementing the actual variable density (i.e., increase priori P for sure).
# Perhaps the first is about pinpoint landing with OpNav and altimeter,
# and the second is about using a transponder and stationkeeping to better
# fit those coefficients in order to estimate a mascon beneath the surface.
# You can also collaborate and use MCMC or optical flow to complete the circle.
# I really like this!

# NOTE: Your trajectory may not break azimuthal symmetry enough — e.g., it may favor odd
# m harmonics if your path spirals or moves more in rho than phi.

# TODO: Why m=2,4 doesnt improve? Is the trajectory?
# TODO: pick realistic and good sensor suite, implement realistic models, cadence and noise, minimum height, etc.
# TODO: Covariance realism isn't the aim here (consider convariance or SNC), it's just to see how those parameters estimation evolve. Does Jay agree?
# TODO: Check A and rotations, math in general, pipeline, etc.
# TODO: optical flow exploitation?


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# ── Eros ───────────────────────────────────────────────────────────────────
GM_KM3S2 = 4.4621e-4  # Miller et al. (2002)
VOLUME_KM3 = 2503.0  # Thomas et al. (2002): fixes the length unit of the mesh
SPIN_PERIOD_H = 5.270
OMEGA = 2.0 * np.pi / (SPIN_PERIOD_H * 3600.0)
W_OMEGA = np.array([[0.0, -OMEGA, 0.0], [OMEGA, 0.0, 0.0], [0.0, 0.0, 0.0]])  # [ω×]

# ── CH patch (pt1 of the density scripts) ──────────────────────────────────
CYL_RADIUS, CYL_HEIGHT, CYL_GAP = 0.12, 0.40, 0.005  # LU, as run_experiment
N_CYL_PTS = 200  # GLOBAL main(): run_experiment(n_cyl_pts=200)
CH_MASTER = (12, 12)  # degrees m < 12, orders n ≤ 12: the density scripts' "new" modes
PRESETS = G.CH_PRESETS  # "old" (8,8)/1e-4 and "new" (12,12)/1e-6
TRUTH_RCOND = PRESETS["new"]["rcond"]  # c = Φ⁺f of the truth and of the prior draws
BASIS_CASES = [  # (modes, truth rcond): the nominal filter re-run on other bases
    ((6, 6), 1e-6),  # degrees the OD informs, fewer orders
    ((12, 12), 1e-4),  # looser truth / prior definition
    ((8, 8), 1e-6),  # the density scripts' "old" modes
    ((12, 6), 1e-6),
    ((12, 4), 1e-6),
    ((12, 3), 1e-6),
]
EPS_RULE = 1e-3  # the od_sigma eps of GLOBAL main(), for comparison
# CH prior: Monte Carlo of shape models × bulk densities, each refit as c = Φ⁺f
PRIOR_DRAWS = 300
PRIOR_SHAPE_SIGMA = 0.030  # km: radial shape error, the ±1% volume of Thomas et al. (2002) over R̄ ≈ 8.4 km
PRIOR_SHAPE_CORR = 1.0  # km: its correlation length
PRIOR_RHO_SIGMA = 0.03 / 2.67  # bulk density 2.67 ± 0.03 g/cm³ (Yeomans et al. 2000)
PRIOR_SNR_MAX = 1.0  # the MC covariance is scaled so the largest |c_true|/σ_prior is this: all below the noise

# ── campaign: low-pass sorties from a home orbit ───────────────────────────
R_HOME_KM = (25.0, 35.0, 50.0)  # circular polar home orbit = apoapsis of the sortie ellipse
HOME_PSI = 0.5 * np.pi  # inertial pass direction over the pole: the polar plane with its normal
#                         along the Sun's equatorial projection (closest to terminator: arrays
#                         on the Sun while the camera points nadir), fixed for the campaign
ARC_HALF = 9000.0  # s: tracking arc ±2.5 h about periapsis, inside the two apoapsis burns
DT_INT = 10.0  # s: RK4 step
DT_OBS = 60.0  # s: filter epoch = one Doppler count
N_ARCS = 30
N_SPARE = 10  # extra greedy picks kept in case the full-truth recheck drops some
H_P_KM = (1.5, 3.0, 5.0)  # periapsis height above the cylinder base
RHO_FRAC = (-0.85, -0.45, 0.0, 0.45, 0.85)  # periapsis offset from the axis, × R_cyl, either side
N_TH0 = 24  # body phases at periapsis (the sortie timing): 15° steps of pass azimuth
MIN_ALT_KM = 1.0
DWELL_TAU = 60.0  # s: coverage utility 1 − exp(−dwell/τ) per (ρ, φ, z) bin
COV_BINS = (3, 8, 4)  # equal-area ρ × φ × z bins of the cylinder

# ── measurements: DSN radio + landmarks ────────────────────────────────────
SIG_DOPPLER = 1e-7  # km/s: two-way X-band Doppler, 60-s count (0.1 mm/s)
SIG_RANGE = 1e-3  # km: two-way ranging point (1 m)
RANGE_EVERY = 10  # epochs (10 min)
EARTH_DIR = np.array([  # inertial, from Eros; 36° from SUN_DIR, 70° from the pole
    np.sin(np.radians(70.0)) * np.cos(np.radians(25.0)),
    np.sin(np.radians(70.0)) * np.sin(np.radians(25.0)),
    np.cos(np.radians(70.0)),
])
IMG_EVERY = 5  # epochs (5 min)
CAM_HALF_FOV = np.radians(20.0)  # NavCam-class (OSIRIS-REx TAGCAMS 44° × 32°)
CAM_IFOV = 3e-4  # rad/px
SIG_PIX = 0.25 * CAM_IFOV  # landmark centroiding
SIG_ATT = 2e-5  # rad per axis: star-tracker attitude knowledge, white between epochs and
#                 COMMON to every landmark of one image (one s/c attitude)
EMISSION_MAX = np.radians(60.0)
INCIDENCE_MAX = np.radians(70.0)
SUN_DIR = np.array([np.sin(np.radians(40.0)), 0.0, np.cos(np.radians(40.0))])
N_LMK_CAP, N_LMK_GLOBAL = 30, 50  # landmarks on the polar cap / elsewhere

# ── consider parameters: constant over the campaign, NOT estimated ─────────
# Their a-priori uncertainty is carried through the filter as unestimated
# columns of the SRIF array; the estimate error they cause adds to the formal σ.
CONSIDER = [  # (name, components, σ)
    ("Range Bias", 1, 1e-3),  # km: station / transponder delay calibration (1 m), ranging only
    ("Camera Alignment", 3, 5e-5),  # rad: NavCam vs star tracker, s/c frame
]
N_CON = sum(c for _, c, _ in CONSIDER)

# ── a-priori σ (scaled coordinates make every one of these 1) ──────────────
SIG_R0, SIG_V0 = 0.1, 1e-4  # km, km/s
SIG_LMK = 1e-2  # km
SIG_SA = 1e-12  # km/s² per inertial axis and batch: stochastic acceleration
SA_BATCH = 3600.0  # s: batch length (piecewise constant, uncorrelated)
SA_CASES = [  # (print name, σ_SA, batch); 20-min batches at the SAME PSD σ²τ
    ("nominal", SIG_SA, SA_BATCH),
    ("sigma_SA x 10", 10 * SIG_SA, SA_BATCH),
    ("20-min batches", SIG_SA * np.sqrt(SA_BATCH / 1200.0), 1200.0),
]
RESOLVED = 1.0 / np.sqrt(2.0)

OUTDIR = "Images"
PREFIX = "od_"
NPZ_PATH = "od_ch_covariance.npz"
CACHE_DIR = os.path.expanduser("~/.cache/od_srif")  # campaign + prior draws, off the synced tree
USE_CACHE = os.environ.get("OD_NO_CACHE", "") == ""
QUICK = os.environ.get("OD_QUICK", "") != ""  # nominal filter only: no σ_SA cases, no basis sweep
N_LOC = 9  # r, v, a_SA(3)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL: body, cylinder, truth, CH coefficients
# ═══════════════════════════════════════════════════════════════════════════


class Model:
    """Everything shared by every stage: mesh, units, truth, cylinder, samples."""

    def __init__(self):
        self.V, self.F, self.tm, self.Rb = G.load_eros()
        self.bulk = G.Bulk(self.V, self.F)
        self.names, self.P, self.beta = G.mascon_arrays()
        self.beta_b = G.bulk_fraction(self.beta)
        self.LU = (VOLUME_KM3 / self.bulk.volume) ** (1.0 / 3.0)  # km
        self.ACC = GM_KM3S2 / self.LU**2  # normalized accel -> km/s²
        self.GRAD = GM_KM3S2 / self.LU**3  # normalized gradient -> 1/s²
        zmax = self.V[:, 2].max()
        self.cyl = G.Cylinder(
            center=np.array([0.0, 0.0, zmax + CYL_GAP]),
            radius=CYL_RADIUS,
            height=CYL_HEIGHT,
        )
        obs = G.cylinder_points(self.cyl, n=N_CYL_PTS)
        self.obs = obs[~G.inside_body(self.tm, self.V, self.F, obs)]
        self.kd = cKDTree(self.V * self.LU)


# ═══════════════════════════════════════════════════════════════════════════
# RUN CACHE
# ═══════════════════════════════════════════════════════════════════════════

HERE = os.path.dirname(os.path.abspath(__file__))


def _ours(mod):
    f = getattr(mod, "__file__", None)
    return f is not None and os.path.dirname(os.path.abspath(f)) == HERE


def _blob(v):
    """Bytes of a setting: arrays exactly, containers item by item, no addresses."""
    if isinstance(v, np.ndarray):
        return repr((v.dtype.str, v.shape)).encode() + np.ascontiguousarray(v).tobytes()
    if isinstance(v, (list, tuple)):
        return b"(" + b",".join(_blob(u) for u in v) + b")"
    if isinstance(v, dict):
        return b"{" + b",".join(_blob(k) + b":" + _blob(u) for k, u in v.items()) + b"}"
    if callable(v):
        return getattr(v, "__qualname__", type(v).__name__).encode()
    r = repr(v)
    return (type(v).__name__ if " at 0x" in r else r).encode()


def fingerprint(roots, data=()):
    """
    16-hex SHA-1 of what the functions / classes `roots` compute from: the
    source of every function and class of this project they reach by name
    (here and in the density scripts, recursively), their default arguments,
    the module-level settings they read, and the arrays `data`.  Edits to the
    filter or the figures leave it alone; any edit on that path changes it.
    """
    h = hashlib.sha1(np.__version__.encode())
    seen, todo = set(), list(roots)
    while todo:
        obj = todo.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        h.update(inspect.getsource(obj).encode())
        fns = [v for v in vars(obj).values() if isinstance(v, types.FunctionType)] if isinstance(obj, type) else [obj]
        for fn in fns:
            h.update(_blob((fn.__defaults__, fn.__kwdefaults__)))
            names, codes = set(), [fn.__code__]
            while codes:  # global and attribute names, nested functions included
                co = codes.pop()
                names.update(co.co_names)
                codes += [c for c in co.co_consts if isinstance(c, types.CodeType)]
            mods = [v for v in fn.__globals__.values() if isinstance(v, types.ModuleType) and _ours(v)]
            for nm in sorted(names):
                for v in [fn.__globals__.get(nm)] + [getattr(m, nm, None) for m in mods]:
                    if isinstance(v, (types.FunctionType, type)):
                        if _ours(sys.modules.get(v.__module__)):
                            todo.append(v)
                    elif isinstance(v, (int, float, str, tuple, list, dict, np.ndarray)):
                        h.update(nm.encode() + _blob(v))
    for d in data:
        h.update(_blob(d))
    return h.hexdigest()[:16]


def cached(name, roots, M, compute):
    """
    compute() → dict of arrays, kept as CACHE_DIR/<name>_<fingerprint>.npz
    (fingerprint of `roots` and of the model arrays) and reloaded while
    neither changes; OD_NO_CACHE=1 recomputes and rewrites it.  Returns the
    dict and the file it was loaded from (None when computed).
    """
    data = (M.V, M.F, M.P, M.beta, M.obs, M.LU, M.cyl.center, M.cyl.radius, M.cyl.height)
    path = os.path.join(CACHE_DIR, f"{name}_{fingerprint(roots, data)}.npz")
    if USE_CACHE and os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}, path.replace(os.path.expanduser("~"), "~")
    out = compute()
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path[:-4] + ".part.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)  # never a half-written cache
    return out, None


def rotz(th):
    """Rz(θ), vectorized over any shape of θ -> (..., 3, 3)."""
    th = np.asarray(th, float)
    c, s = np.cos(th), np.sin(th)
    o, z = np.ones_like(th), np.zeros_like(th)
    return np.stack(
        [np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)],
        -2,
    )


def in_cylinder(cyl, x):
    """x in LU (..., 3) -> inside the CH cylinder."""
    tp = (x - cyl.center) @ cyl.rot()
    rho = np.hypot(tp[..., 0], tp[..., 1])
    return (rho <= cyl.radius) & (tp[..., 2] >= 0.0) & (tp[..., 2] <= cyl.height)


def cyl_local(cyl, x):
    """(ρ, φ, z) of x (LU) in the cylinder frame."""
    tp = (x - cyl.center) @ cyl.rot()
    return np.hypot(tp[..., 0], tp[..., 1]), np.arctan2(tp[..., 1], tp[..., 0]), tp[..., 2]


def ch_grad(cyl, x, ch_modes=CH_MASTER, chunk=3000):
    """∇Φ at x (LU): (n, 3, n_cols) normalized acceleration of every CH mode."""
    out = []
    for i in range(0, len(x), chunk):
        xi = x[i : i + chunk]
        Phi = G.cyl_basis(cyl, xi, *ch_modes)
        n = len(xi)
        out.append(Phi[n:].reshape(3, n, -1).transpose(1, 0, 2))
    return np.concatenate(out) if out else np.zeros((0, 3, 2 * np.prod(ch_modes)))


def ch_basis(cyl, obs, ch_modes=CH_MASTER):
    """
    The raw CH columns of `cyl_basis` that enter the filter: every A_mn and
    B_mn except B_0n ≡ 0.  Returns T (the columns at the samples, [U; a]),
    `cols` (their index in the full cyl_basis packing), the degree m, trig
    (0 = cos/A, 1 = sin/B) and order n of each, and the full Φ.  Columns keep
    the cyl_basis order (m, n, trig).
    """
    nm, nn = ch_modes
    Phi = G.cyl_basis(cyl, obs, nm, nn)
    j = np.arange(Phi.shape[1])
    m, n, trig = j // (2 * nn), (j // 2) % nn + 1, j % 2
    cols = np.flatnonzero((m > 0) | (trig == 0))
    return dict(T=Phi[:, cols], cols=cols, m=m[cols], trig=trig[cols], n=n[cols], Phi=Phi,
                modes=ch_modes)


def shape_density_fields(M):
    """
    Fields at the samples of the CH prior Monte Carlo ([U; a] per draw, cached on
    M: they do not depend on the basis).  Each draw is a shape model, the mesh
    displaced along its area-weighted vertex normals by a smooth Gaussian random
    field (σ PRIOR_SHAPE_SIGMA, correlation length PRIOR_SHAPE_CORR; random
    Fourier features), with a bulk density ρ = ρ0 (1 + PRIOR_RHO_SIGMA ε).  Its
    field is what the filter's fixed constant-density bulk misses:
        f_i = μ_i · bulk_i(obs) − bulk(obs),   μ_i = (ρ_i / ρ0)(V_i / V0).
    """
    if getattr(M, "prior_fields", None) is not None:
        return M.prior_fields
    rng, nf = np.random.default_rng(404), 300
    V, F = M.V, M.F
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    f0 = M.bulk.field(M.obs)
    Fd = np.empty((len(f0), PRIOR_DRAWS))
    for i in range(PRIOR_DRAWS):
        w = rng.normal(0.0, M.LU / PRIOR_SHAPE_CORR, (3, nf))
        h = np.sqrt(2.0 / nf) * np.cos(V @ w + rng.uniform(0.0, 2.0 * np.pi, nf)).sum(axis=1)
        Bi = G.Bulk(V + (PRIOR_SHAPE_SIGMA / M.LU) * h[:, None] * vn, F)
        mu = (1.0 + PRIOR_RHO_SIGMA * rng.standard_normal()) * Bi.volume / M.bulk.volume
        Fd[:, i] = mu * Bi.field(M.obs) - f0
    M.prior_fields = Fd
    return Fd


def ch_truth_and_prior(M, B, rcond=TRUTH_RCOND):
    """
    Raw coefficients of the truth interior, its CH omission, and the prior:
    c = Φ⁺ f with the truncated SVD at TRUTH_RCOND, for the truth and refit to
    every shape / bulk-density draw of shape_density_fields.  The prior is
    their FULL sample covariance, inflated:  P = κ · C Cᵀ / N, with κ set so
    the largest |c_true|/σ_prior is PRIOR_SNR_MAX (the prior is far less
    certain than the truth's size: the data, not the prior, must bring it out).  It
    has the rank r of Φ⁺ (the refits span what the truth definition allows),
    so it ties the nearly dependent A_mn, B_mn together.  Returns c_true,
    σ_prior = √diag P, f, Q (K × r) with Q Qᵀ = P / σσᵀ (the filter carries
    the scaled coefficients as Q z, z ~ N(0, I_r)) and κ.
    """
    Pinv = G.ch_pinv(B["Phi"], rcond)[B["cols"]]
    f = G.A_field_contrast(M.P, M.bulk, M.obs) @ M.beta
    Cd = Pinv @ shape_density_fields(M)
    c = Pinv @ f
    P = Cd @ Cd.T / Cd.shape[1]
    kap = (np.max(np.abs(c) / np.sqrt(np.diag(P))) / PRIOR_SNR_MAX) ** 2
    P *= kap
    s = np.sqrt(np.diag(P))
    w, U = np.linalg.eigh(P / np.outer(s, s))
    k = w > 1e-12 * w.max()
    return c, s, f, U[:, k] * np.sqrt(w[k]), kap


# ═══════════════════════════════════════════════════════════════════════════
# FORCE MODELS AND PROPAGATION
# ═══════════════════════════════════════════════════════════════════════════


class TruthGravity:
    """
    Full truth β̃·bulk + Σβ_j·point masses at r (km).  Returns the acceleration
    (km/s²), its gradient (1/s²) and the normalized CONTRAST acceleration
    (truth minus the full-mass bulk = what CH + omission must carry).
    """

    def __init__(self, M):
        self.M = M
        self.ev = GravityEvaluable(
            Polyhedron(
                polyhedral_source=(M.V, M.F),
                density=1.0 / M.bulk.volume,
                integrity_check=PolyhedronIntegrity.DISABLE,
            )
        )

    def __call__(self, r_km):
        M = self.M
        x = np.atleast_2d(r_km) / M.LU
        res = self.ev(computation_points=x, parallel=True)
        if len(x) == 1 and np.isscalar(res[0]):
            res = [res]
        g = np.array([q[1] for q in res]) / G.G_SI
        t6 = np.array([q[2] for q in res]) / G.G_SI  # [xx, yy, zz, xy, xz, yz]
        Tb = t6[:, [0, 3, 4, 3, 1, 5, 4, 5, 2]].reshape(-1, 3, 3)
        d = x[:, None, :] - M.P[None]
        r = np.linalg.norm(d, axis=2)
        gp = -d / r[..., None] ** 3
        Tp = (
            3.0 * d[..., :, None] * d[..., None, :]
            - (r**2)[..., None, None] * np.eye(3)
        ) / (r**5)[..., None, None]
        pm = np.einsum("j,njk->nk", M.beta, gp)
        a = M.beta_b * g + pm
        Tg = M.beta_b * Tb + np.einsum("j,njkl->nkl", M.beta, Tp)
        con = pm - M.beta.sum() * g
        return M.ACC * a, M.GRAD * Tg, con


class CloudGravity:
    """Cheap screening model: the bulk as a cloud of equal interior mascons."""

    def __init__(self, M, n=500, seed=5):
        rng = np.random.default_rng(seed)
        lo, hi = M.V.min(0), M.V.max(0)
        q = rng.uniform(lo, hi, (8 * n, 3))
        q = q[G.inside_body(M.tm, M.V, M.F, q)]
        self.Q = np.vstack([q[:n], M.P])
        self.w = np.concatenate([np.full(n, M.beta_b / n), M.beta])
        self.M = M

    def __call__(self, r_km):
        x = np.atleast_2d(r_km) / self.M.LU
        d = x[:, None, :] - self.Q[None]
        r3 = np.linalg.norm(d, axis=2) ** 3
        return self.M.ACC * -np.einsum("j,njk->nk", self.w, d / r3[..., None]), None, None


def deriv(grav, X):
    """Rotating-frame state derivative; also hands back the gradient and contrast."""
    r, v = X[:, :3], X[:, 3:]
    a, Tg, con = grav(r)
    acc = a - 2.0 * v @ W_OMEGA.T - r @ (W_OMEGA @ W_OMEGA).T
    return np.hstack([v, acc]), Tg, con


def rk4(grav, X, dt):
    k1, T1, c1 = deriv(grav, X)
    X2 = X + 0.5 * dt * k1
    k2, T2, c2 = deriv(grav, X2)
    X3 = X + 0.5 * dt * k2
    k3, T3, c3 = deriv(grav, X3)
    X4 = X + dt * k3
    k4, T4, c4 = deriv(grav, X4)
    Xn = X + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return Xn, (X, X2, X3, X4), (T1, T2, T3, T4), (c1, c2, c3, c4)


def propagate_nodes(grav, X0, half=ARC_HALF, dt=DT_INT):
    """States at −half..+half (step dt) from the t = 0 state X0, (Na, 2n+1, 6)."""
    n = int(round(half / dt))
    back, fwd = [X0], [X0]
    for _ in range(n):
        back.append(rk4(grav, back[-1], -dt)[0])
        fwd.append(rk4(grav, fwd[-1], dt)[0])
    return np.stack(back[::-1] + fwd[1:], axis=1)


def propagate_truth(grav, X0, half=ARC_HALF, dt=DT_INT):
    """
    Full-truth arcs from −half to +half, keeping the RK4 stage positions,
    gravity gradients and contrast accelerations: the variational equations
    integrated with the same tableau see A(t) at exactly these points.
    """
    n = int(round(half / dt))
    X = X0.copy()
    for _ in range(n):
        X = rk4(grav, X, -dt)[0]
    N = 2 * n
    Na = len(X0)
    nodes = np.empty((Na, N + 1, 6))
    rS = np.empty((Na, N, 4, 3))
    TS = np.empty((Na, N, 4, 3, 3))
    cS = np.empty((Na, N, 4, 3))
    nodes[:, 0] = X
    for j in range(N):
        X, st, Ts, cs = rk4(grav, X, dt)
        for s in range(4):
            rS[:, j, s] = st[s][:, :3]
            TS[:, j, s] = Ts[s]
            cS[:, j, s] = cs[s]
        nodes[:, j + 1] = X
    return nodes, rS, TS, cS


# ═══════════════════════════════════════════════════════════════════════════
# CAMPAIGN DESIGN: candidate flyovers, screening, greedy coverage
# ═══════════════════════════════════════════════════════════════════════════


def candidate_flybys(M):
    """
    Body-fixed periapsis (t = 0) states of every candidate sortie and their
    parameters (h_p, ρ_p, ψ, θ0, r_home): apoapsis on the circular home orbit
    (vis-viva periapsis speed), inertial plane through HOME_PSI, θ0 = body
    phase at periapsis.
    """
    zb = M.cyl.center[2] * M.LU
    Rc = M.cyl.radius * M.LU
    psi = HOME_PSI
    chi = psi + 0.5 * np.pi  # offset ⟂ to the pass: periapsis
    u = np.array([np.cos(psi), np.sin(psi), 0.0])
    X0, meta = [], []
    for hp in H_P_KM:
        for rf in RHO_FRAC:
            P0 = np.array([rf * Rc * np.cos(chi), rf * Rc * np.sin(chi), zb + hp])
            rp = np.linalg.norm(P0)
            for ra in R_HOME_KM:
                vI = np.sqrt(2.0 * GM_KM3S2 * ra / (rp * (ra + rp))) * u
                for it in range(N_TH0):
                    th0 = 2.0 * np.pi * it / N_TH0
                    Rz = rotz(th0)
                    rB = Rz.T @ P0
                    vB = Rz.T @ vI - W_OMEGA @ rB
                    X0.append(np.concatenate([rB, vB]))
                    meta.append((hp, rf, psi, th0, ra))
    return np.array(X0), np.array(meta)


def sortie_budget(X0, meta):
    """Half period (s) of each sortie ellipse and its Δv (km/s): deorbit + recircularize."""
    ra, rp = meta[:, 4], np.linalg.norm(X0[:, :3], axis=1)
    half = np.pi * np.sqrt((0.5 * (ra + rp)) ** 3 / GM_KM3S2)
    dv = 2.0 * (np.sqrt(GM_KM3S2 / ra) - np.sqrt(2.0 * GM_KM3S2 * rp / (ra * (ra + rp))))
    return half, dv


def coverage_dwell(M, nodes, dt):
    """Dwell (s) of each arc in each equal-area (ρ, φ, z) bin of the cylinder."""
    nr, nphi, nz = COV_BINS
    x = nodes[..., :3] / M.LU
    inside = in_cylinder(M.cyl, x)
    rho, phi, z = cyl_local(M.cyl, x)
    ir = np.minimum((rho / M.cyl.radius) ** 2 * nr, nr - 1).astype(int)
    ip = np.minimum((phi + np.pi) / (2 * np.pi) * nphi, nphi - 1).astype(int)
    iz = np.clip(z / M.cyl.height * nz, 0, nz - 1).astype(int)
    b = (ir * nphi + ip) * nz + iz
    D = np.zeros((len(nodes), nr * nphi * nz))
    for a in range(len(nodes)):
        np.add.at(D[a], b[a][inside[a]], dt)
    return D


def greedy_cover(D, n_pick, tau=DWELL_TAU):
    cov = np.zeros(D.shape[1])
    picked = []
    for _ in range(min(n_pick, len(D))):
        gain = (np.exp(-cov / tau) - np.exp(-(cov + D) / tau)).sum(1)
        gain[picked] = -np.inf
        k = int(np.argmax(gain))
        if gain[k] <= 0:
            break
        picked.append(k)
        cov += D[k]
    return picked, cov


def design_campaign(M, truth):
    t0 = time.time()
    X0, meta = candidate_flybys(M)
    half, _ = sortie_budget(X0, meta)
    assert half.min() > ARC_HALF, "the tracking arc would cross an apoapsis burn"
    nodes = propagate_nodes(CloudGravity(M), X0, dt=2 * DT_INT)
    alt = M.kd.query(nodes[..., :3].reshape(-1, 3))[0].reshape(nodes.shape[:2])
    ok = alt.min(1) >= MIN_ALT_KM
    D = coverage_dwell(M, nodes, 2 * DT_INT)
    D[~ok] = 0.0
    order, _ = greedy_cover(D, N_ARCS + N_SPARE)
    print(
        f"  candidates {len(X0)}  (h_p × ρ_p × r_home × θ0; apoapsis burns at "
        f"±{half.min() / 3600:.1f}–{half.max() / 3600:.1f} h, arc ±{ARC_HALF / 3600:.1f} h), "
        f"{ok.sum()} clear {MIN_ALT_KM:.0f} km on the screening model "
        f"[{time.time() - t0:.0f} s]"
    )

    t0 = time.time()
    nodes, rS, TS, cS = propagate_truth(truth, X0[order])
    alt = M.kd.query(nodes[..., :3].reshape(-1, 3))[0].reshape(nodes.shape[:2])
    keep = np.flatnonzero(alt.min(1) >= MIN_ALT_KM)[:N_ARCS]
    print(
        f"  full-truth recheck: {len(keep)}/{len(order)} greedy picks kept "
        f"(min altitude {alt[keep].min():.2f} km) [{time.time() - t0:.0f} s]"
    )
    sel = np.asarray(order)[keep]
    return dict(
        X0=X0[sel],
        meta=meta[sel],
        nodes=nodes[keep],
        rS=rS[keep],
        TS=TS[keep],
        cS=cS[keep],
        alt=alt[keep],
    )


# ═══════════════════════════════════════════════════════════════════════════
# VARIATIONAL EQUATIONS (per 60 s interval)
# ═══════════════════════════════════════════════════════════════════════════


def arc_transitions(M, Bs, c_true, rS, TS, cS, th0):
    """
    STM blocks of one arc over every filter interval.

    Returns Φ (n_int, 6, 6), Ψ (n_int, 6, 3+K+1) for the dynamic parameters
    [a_SA, A/B, s], the fraction of each interval spent inside the cylinder and
    the omission acceleration (km/s², per axis) at the in-cylinder stages.
    """
    N = rS.shape[0]
    K = Bs["T"].shape[1]
    nd = 3 + K + 1
    t_st = -ARC_HALF + DT_INT * (np.arange(N)[:, None] + np.array([0.0, 0.5, 0.5, 1.0]))

    A = np.zeros((N, 4, 6, 6))
    A[..., :3, 3:] = np.eye(3)
    A[..., 3:, :3] = TS + OMEGA**2 * np.diag([1.0, 1.0, 0.0])  # −[ω×]²
    A[..., 3:, 3:] = -2.0 * W_OMEGA

    Bm = np.zeros((N, 4, 6, 6 + nd))  # padded: [0_6 | B]
    Bm[..., 3:, 6:9] = rotz(th0 + OMEGA * t_st).swapaxes(-1, -2)  # Rᵀ(t)
    x = rS / M.LU
    inside = in_cylinder(M.cyl, x)
    om = np.zeros((0, 3))
    if inside.any():
        gC = ch_grad(M.cyl, x[inside], Bs["modes"])[..., Bs["cols"]]  # (n_in, 3, K)
        om = cS[inside] - gC @ c_true
        blk = Bm[inside]
        blk[:, 3:, 9 : 9 + K] = M.ACC * gC
        blk[:, 3:, -1] = M.ACC * om
        Bm[inside] = blk

    q = int(round(DT_OBS / DT_INT))
    n_int = N // q
    Y = np.zeros((n_int, 6, 6 + nd))
    Y[:, :, :6] = np.eye(6)
    h = DT_INT
    for k in range(q):
        Aj, Bj = A[k::q], Bm[k::q]
        k1 = Aj[:, 0] @ Y + Bj[:, 0]
        k2 = Aj[:, 1] @ (Y + 0.5 * h * k1) + Bj[:, 1]
        k3 = Aj[:, 2] @ (Y + 0.5 * h * k2) + Bj[:, 2]
        k4 = Aj[:, 3] @ (Y + h * k3) + Bj[:, 3]
        Y = Y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    f_in = inside.reshape(n_int, q * 4).mean(1)
    return Y[:, :, :6], Y[:, :, 6:], f_in, M.ACC * om


# ═══════════════════════════════════════════════════════════════════════════
# MEASUREMENTS: DSN RADIOMETRIC TRACKING AND OPNAV LANDMARKS
# ═══════════════════════════════════════════════════════════════════════════


def earth_los(M, camp):
    """
    Earth line of sight in the body frame at every filter epoch, and whether
    the radio link is clear: no data while Eros occults the Earth (ray-cast on
    the mesh from the s/c).  Returns (n_arc, n_epoch, 3) and (n_arc, n_epoch).
    """
    q = int(round(DT_OBS / DT_INT))
    r = camp["nodes"][:, ::q, :3]
    t = -ARC_HALF + DT_OBS * np.arange(r.shape[1])
    th = camp["meta"][:, 3, None] + OMEGA * t[None, :]
    c, s = np.cos(th), np.sin(th)  # rotz(th)ᵀ EARTH_DIR
    k = np.stack([c * EARTH_DIR[0] + s * EARTH_DIR[1],
                  -s * EARTH_DIR[0] + c * EARTH_DIR[1],
                  np.full_like(th, EARTH_DIR[2])], axis=-1)
    hit = M.tm.ray.intersects_any(ray_origins=r.reshape(-1, 3) / M.LU,
                                  ray_directions=k.reshape(-1, 3))
    return k, ~hit.reshape(th.shape)


def visible(L, Nrm, r, sun_B):
    """FOV / emission / incidence test of landmarks L (normals Nrm) from r (km)."""
    d = L - r
    u = d / np.linalg.norm(d, axis=-1)[..., None]
    b = -r / np.linalg.norm(r)
    return (
        (u @ b > np.cos(CAM_HALF_FOV))
        & (np.sum(-u * Nrm, -1) > np.cos(EMISSION_MAX))
        & (Nrm @ sun_B > np.cos(INCIDENCE_MAX))
    )


def image_epochs(camp):
    """(arc, epoch, s/c position, Sun in body frame) of every image."""
    q = int(round(DT_OBS / DT_INT))
    n_ep = camp["nodes"].shape[1] // q + 1
    out = []
    for a in range(len(camp["nodes"])):
        th0 = camp["meta"][a, 3]
        for e in range(0, n_ep, IMG_EVERY):
            t = -ARC_HALF + DT_OBS * e
            out.append((a, e, camp["nodes"][a, q * e, :3], rotz(th0 + OMEGA * t).T @ SUN_DIR))
    return out


def fps(P, n, seed_pts=None):
    """Farthest-point sample of n rows of P, away from seed_pts too."""
    if n <= 0 or len(P) == 0:
        return np.zeros(0, int)
    dmin = np.full(len(P), np.inf)
    if seed_pts is not None and len(seed_pts):
        dmin = cKDTree(seed_pts).query(P)[0]
    pick = [int(np.argmax(dmin)) if np.isfinite(dmin).all() else int(np.argmax(P[:, 2]))]
    for _ in range(min(n, len(P)) - 1):
        dmin = np.minimum(dmin, np.linalg.norm(P - P[pick[-1]], axis=1))
        pick.append(int(np.argmax(dmin)))
    return np.array(pick)


def pick_landmarks(M, imgs, n_cand=4000, seed=6):
    """Landmarks seen ≥ 3 times: N_LMK_CAP on the polar cap, N_LMK_GLOBAL elsewhere."""
    tm = M.tm
    rng = np.random.default_rng(seed)
    p = tm.area_faces / tm.area_faces.sum()
    idx = rng.choice(len(p), n_cand, replace=False, p=p)
    L = tm.triangles_center[idx] * M.LU
    Nrm = tm.face_normals[idx]
    cnt = np.zeros(n_cand, int)
    for _, _, r, sun in imgs:
        cnt += visible(L, Nrm, r, sun)
    ok = cnt >= 3
    cap = ok & (np.hypot(L[:, 0], L[:, 1]) < 1.5 * M.cyl.radius * M.LU) & (L[:, 2] > 0)
    i_cap = np.flatnonzero(cap)[fps(L[cap], N_LMK_CAP)]
    rest = ok & ~cap
    i_rest = np.flatnonzero(rest)[fps(L[rest], N_LMK_GLOBAL, L[i_cap])]
    sel = np.concatenate([i_cap, i_rest])
    return L[sel], Nrm[sel]


def image_visibility(M, imgs, L, Nrm):
    """Visible landmark indices per image, with the occlusion ray test."""
    vis = {}
    for a, e, r, sun in imgs:
        j = np.flatnonzero(visible(L, Nrm, r, sun))
        if len(j):
            o = L[j] / M.LU + 1e-4 * Nrm[j]
            dvec = r - L[j]
            dvec /= np.linalg.norm(dvec, axis=1)[:, None]
            j = j[~M.tm.ray.intersects_any(ray_origins=o, ray_directions=dvec)]
        vis[(a, e)] = j
    return vis


class Layout:
    """Column layout of one arc's state, and its scaling (prior σ)."""

    def __init__(self, sig_c, n_lmk, sig_sa=SIG_SA):
        self.K = len(sig_c)
        self.nL = n_lmk
        self.D = np.concatenate(
            [[SIG_R0] * 3, [SIG_V0] * 3, [sig_sa] * 3, sig_c, [SIG_LMK] * (3 * n_lmk)]
        )
        self.n = len(self.D)
        self.i_ch = N_LOC
        self.i_lmk = N_LOC + self.K
        self.dyn = np.concatenate(
            [np.arange(6, 9), np.arange(N_LOC, N_LOC + self.K), [self.n]]
        )
        self.Ddyn = np.concatenate([self.D[6:9], sig_c, [1.0]])  # s unscaled
        self.Dc = np.concatenate([[sg] * c for _, c, sg in CONSIDER])  # consider σ
        self.ncol = self.n + 1 + len(self.Dc)  # [state | omission s | consider c]


def meas_rows(lay, X, L, vis_j, k, radio, ranging):
    """
    Whitened, scaled rows [H | 0 | H_c] of one epoch, m × (n + 1 + N_CON).

    Radio along the Earth line of sight k (body frame), the Earth–Eros part
    known from the ephemerides:  Doppler  ρ̇ = −k·(v + ω×r),  range
    ρ = −k·r + b  (b the range bias; Doppler unbiased).  One s/c attitude
    serves every landmark of an image, so its knowledge error (SIG_ATT, white
    between epochs) is common to their rows:  R = diag(σ²) + σ_att² B Bᵀ,
    B = ∂(measurement)/∂(pointing) (zero for the radio rows), and the epoch is
    whitened as a whole.  The range bias and the constant camera alignment
    (s/c frame) are the consider columns H_c (scaled by their σ, so
    c ~ N(0, I)).
    """
    r, n = X[:3], lay.n
    b = -r / np.linalg.norm(r)  # nadir: camera axis
    e1 = np.cross(b, [0.0, 1.0, 0.0] if abs(b[0]) > 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    A_sc = np.column_stack([e1, np.cross(b, e1), b])  # s/c frame → body frame
    H, B, sd, Hc = [], [], [], []
    if radio:
        h = np.zeros(n)
        h[0:3], h[3:6] = -W_OMEGA.T @ k, -k
        H.append(h)
        B.append(np.zeros(3))
        sd.append(SIG_DOPPLER)
        Hc.append(np.zeros(N_CON))
        if ranging:
            h = np.zeros(n)
            h[0:3] = -k
            H.append(h)
            B.append(np.zeros(3))
            sd.append(SIG_RANGE)
            Hc.append(np.concatenate([[1.0], np.zeros(3)]))
    for j in [] if vis_j is None else vis_j:  # landmark line of sight, two angles
        d = L[j] - r
        dist = np.linalg.norm(d)
        w = d / dist
        p1 = e1 - (e1 @ w) * w
        p1 /= np.linalg.norm(p1)
        c = lay.i_lmk + 3 * j
        for pv in (p1, np.cross(w, p1)):
            h = np.zeros(n)
            h[0:3] = -pv / dist
            h[c : c + 3] = pv / dist
            bc = np.cross(w, pv)
            H.append(h)
            B.append(bc)
            sd.append(SIG_PIX)
            Hc.append(np.concatenate([[0.0], bc @ A_sc]))
    if not H:
        return np.zeros((0, lay.ncol))
    B = np.array(B)
    Lc = cholesky(np.diag(np.square(sd)) + SIG_ATT**2 * B @ B.T, lower=True)
    A = np.hstack([np.array(H) * lay.D, np.zeros((len(H), 1)), np.array(Hc) * lay.Dc])
    return solve_triangular(Lc, A, lower=True)


# ═══════════════════════════════════════════════════════════════════════════
# SRIF
# ═══════════════════════════════════════════════════════════════════════════


def _qr_r(A):
    return np.linalg.qr(A, mode="r")


def _qr_append(R, H):
    """
    R of the QR of [R; H]: R (n × ncol) upper triangular in its first n
    columns, H a few rows.  LAPACK's triangular-pentagonal QR (tpqrt, then
    tpmqrt on the omission / consider columns): O(m n²) for the O(n³) of a full QR.
    """
    n = R.shape[0]
    Rn, V, T, i1 = dtpqrt(0, min(n, 32), R[:, :n], H[:, :n])
    Rx, _, i2 = dtpmqrt(0, V, T, R[:, n:], H[:, n:], trans="T")
    assert i1 == i2 == 0
    return np.hstack([Rn, Rx])


def _tri_inv(R):
    """R⁻¹ of an upper-triangular R (LAPACK trtri)."""
    Ri, info = dtrtri(R)
    assert info == 0
    return Ri


def marginal_ch(RG, K):
    """
    Scaled CH covariance, omission bias and consider sensitivity from the global
    SRIF array  R x + R_s s + R_c c = z.  The filter sets s = 0 and c = 0 while
    the truth has s = 1 and c ~ N(0, I) (scaled), so  x̂ − x = R⁻¹ (R_s + R_c c):
    the bias R⁻¹ R_s and the consider covariance E Eᵀ, E = R⁻¹ R_c.
    """
    ng = RG.shape[0]
    Rinv = solve_triangular(RG[:, :ng], np.eye(ng))
    Sc = Rinv[:K] @ Rinv[:K].T
    bias = (Rinv @ RG[:, ng])[:K]
    return Sc, bias, (Rinv @ RG[:, ng + 1 :])[:K]


def subset_solution(RG, K, est, x_true, Q=None):
    """
    Estimate only the CH columns `est` (the other CH columns fixed at zero,
    their truth left in the data with the omission) from the same information
    array  R x + R_s s + R_c c = z:
        x̂_r − x_r = R_r⁺ (R_h x_h + R_s + R_c c),   cov = (R_rᵀ R_r)⁻¹.
    Returns, scaled and for the `est` columns: F with cov = F Fᵀ, the consider
    sensitivity E = R_r⁺ R_c and the deterministic error d = R_r⁺ (R_h x_h + R_s).
    With Q (the full prior, ch_truth_and_prior) RG must come from a run with
    the identity CH prior: its data information J = RᵀR − I_CH is combined with
    the marginal prior of the `est` columns, x_r = Q_r z, z ~ N(0, I) (the
    landmarks keep theirs), and the same three are mapped back through Q_r.
    """
    ng = RG.shape[0]
    if Q is not None:
        J = RG.T @ RG
        J[:K, :K] -= np.eye(K)
        r, k = Q.shape[1], int(est.sum())
        idx = np.concatenate([np.flatnonzero(est), np.arange(K, ng)])
        h = np.flatnonzero(~est)
        T = np.zeros((len(idx), r + ng - K))
        T[:k, :r], T[k:, r:] = Q[est], np.eye(ng - K)
        A = T.T @ J[np.ix_(idx, idx)] @ T
        A[:r, :r] += np.eye(r)
        Ai = np.linalg.inv(A)
        d = Q[est] @ (Ai @ T.T @ (J[np.ix_(idx, h)] @ x_true[h] + J[idx, ng]))[:r]
        E = Q[est] @ (Ai @ T.T @ J[idx, ng + 1 :])[:r]
        return Q[est] @ np.linalg.cholesky(0.5 * (Ai[:r, :r] + Ai[:r, :r].T)), E, d
    R, Rs, Rc = RG[:, :ng], RG[:, ng], RG[:, ng + 1 :]
    h = np.concatenate([~est, np.zeros(ng - K, bool)])
    Rr = R[:, ~h]
    k = int(est.sum())
    d = np.linalg.lstsq(Rr, R[:, h] @ x_true[h] + Rs, rcond=None)[0][:k]
    E = np.linalg.lstsq(Rr, Rc, rcond=None)[0][:k]
    F = solve_triangular(np.linalg.qr(Rr, mode="r"), np.eye(Rr.shape[1]))[:k]
    return F, E, d


def truncation_sweep(RG, Bs, c_true, sig_c, f, Q=None):
    """
    `subset_solution` for the square (N, N) set, degree m < N and order n ≤ N,
    for every N.  Returns rows (N, n estimated, n resolved, noise, consider,
    deterministic error) with the errors as RMS acceleration-field error at
    the samples relative to the contrast acceleration (noise and consider 1σ;
    deterministic = aliasing + truncation + omission).
    """
    ng, K = RG.shape[0], len(sig_c)
    x_true = np.concatenate([c_true / sig_c, np.zeros(ng - K)])
    na = Bs["T"].shape[0] // 4
    Ta = Bs["T"][na:] * sig_c  # scaled columns, accelerations
    om = f[na:] - Bs["T"][na:] @ c_true
    nf = np.linalg.norm(f[na:])
    out = []
    for N in range(2, max(Bs["m"].max() + 1, Bs["n"].max()) + 1):
        est = (Bs["m"] < N) & (Bs["n"] <= N)
        F, E, d = subset_solution(RG, K, est, x_true, Q)
        dc = -x_true[:K]
        dc[est] = d
        out.append(
            (N, est.sum(), np.sum(np.sqrt(np.sum(F**2, axis=1)) < RESOLVED),
             np.linalg.norm(Ta[:, est] @ F) / nf, np.linalg.norm(Ta[:, est] @ E) / nf,
             np.linalg.norm(Ta @ dc - om) / nf)
        )
    return np.array(out)


def _restart(R, cols):
    """
    Marginalize the states `cols` out of the SRIF array and restart them at
    their prior.  R is upper triangular, so only its rows up to max(cols)
    involve them: the rows below are left as they are.
    """
    t, c = cols.max() + 1, len(cols)
    rest = np.setdiff1d(np.arange(R.shape[1]), cols)  # + omission and consider columns
    Rp = _qr_r(R[:t][:, np.concatenate([cols, rest])])
    Rt = np.zeros((t, R.shape[1]))
    Rt[:c, cols] = np.eye(c)
    Rt[c:, rest] = Rp[c:, c:]
    R = R.copy()
    R[:t] = _qr_r(Rt)
    return R


def srif_campaign(camp, arcs_stm, lay, L, vis, los, batch=SA_BATCH, Q=None, record=True):
    """
    Multi-arc SRIF in scaled coordinates (every prior = I).

    State [arc locals | CH | landmarks] + the consider columns; CH and the
    landmarks are common to all arcs.  At each arc start the previous arc's
    locals are marginalized (their information on the globals is kept) and the
    new arc's locals start at their prior, so the global information is the sum
    of the per-arc contributions — the classic multi-arc solution; inside
    an arc a_SA is piecewise constant over `batch` s and restarted the same way
    at each batch boundary (uncorrelated batches).  The time update is the
    deterministic map  R_x Φ⁻¹, R_p − R_x Φ⁻¹ Ψ (top 6 rows, re-triangularized);
    each epoch ends with one triangular QR update (_qr_append), after which
    the formal σ of every state and its consider σ (the unestimated
    instrument / frame columns, √diag E Eᵀ with E = R⁻¹ R_c) are recorded —
    at every epoch with `record`, else at the end of each arc only (sig[:, -1],
    all the case tables and RG users need).  With Q (K × r, Q Qᵀ = the scaled CH prior, ch_truth_and_prior)
    the filter carries z, x_CH = Q z with z ~ N(0, I_r), in place of the K
    coefficients, and every output but RG is mapped back to them; without Q
    the CH prior is I (diagonal).  DFS = r − tr Σ_z.
    """
    K = lay.K
    lz = lay if Q is None else Layout(np.ones(Q.shape[1]), lay.nL, lay.D[6])
    n, r = lz.n, lz.K
    Qs = None if Q is None else lay.Ddyn[3 : 3 + K, None] * Q  # ∂c/∂z = σ_c Q

    def to_x(A):  # rows of a z-state array → rows of the coefficient state
        if Q is None:
            return A
        return np.vstack([A[:N_LOC], Q @ A[N_LOC : N_LOC + r], A[N_LOC + r :]])

    q = int(round(DT_OBS / DT_INT))
    nb = int(round(batch / DT_OBS))
    Dd = lz.D[:6]
    loc, sa = np.arange(N_LOC), np.arange(6, 9)
    R = np.hstack([np.eye(n), np.zeros((n, lz.ncol - n))])
    sig, sigc, Enav, hist = [], [], [], []
    for a, (Phi, Psi, _, _) in enumerate(arcs_stm):
        if a:
            R = _restart(R, loc)
        if Q is not None:
            Psi = np.concatenate([Psi[..., :3], Psi[..., 3 : 3 + K] @ Qs, Psi[..., 3 + K :]], axis=-1)
        nodes = camp["nodes"][a]
        ne = len(Phi) + 1
        for e in range(ne):
            if e:
                i = e - 1
                if i and i % nb == 0:
                    R = _restart(R, sa)
                Pt = Phi[i] / Dd[:, None] * Dd[None, :]
                St = Psi[i] / Dd[:, None] * lz.Ddyn[None, :]
                Rd = np.linalg.solve(Pt.T, R[:6, :6].T).T  # R_x Φ⁻¹ (R_x: the top 6 rows)
                R[:6, lz.dyn] -= Rd @ St
                R[:6, :6] = Rd
                R[:6] = _qr_r(R[:6])  # triangular again
            H = meas_rows(lz, nodes[q * e], L, vis.get((a, e)),
                          los[0][a, e], los[1][a, e], e % RANGE_EVERY == 0)
            if len(H):
                R = _qr_append(R, H)
            if record or e == ne - 1:
                Ri = _tri_inv(R[:, :n])
                E = Ri @ R[:, n + 1 :]
                sig.append(np.sqrt(np.sum(to_x(Ri) ** 2, axis=1)))
                sigc.append(np.sqrt(np.sum(to_x(E) ** 2, axis=1)))
        Enav.append(E[:6])
        v = sig[-1][N_LOC : N_LOC + K] ** 2
        hist.append((np.sum(np.sqrt(v) < RESOLVED), np.sum(1.0 - v)))
    RG = R[N_LOC:, N_LOC:]
    Sc, bias, Ec = marginal_ch(RG, r)
    dfs = r - np.trace(Sc)
    if Q is not None:
        Sc, bias, Ec = Q @ Sc @ Q.T, Q @ bias, Q @ Ec
    Ri = to_x(_tri_inv(R[:, :n]))
    P = Ri @ Ri.T
    d = np.sqrt(np.diag(P))
    shp = (len(arcs_stm), -1, lay.n)
    return dict(
        RG=RG, Sc=Sc, bias=bias, Ec=Ec, dfs=dfs, hist=np.array(hist),
        sig=np.array(sig).reshape(shp), sigc=np.array(sigc).reshape(shp),
        Enav=np.array(Enav), corr=P / np.outer(d, d), D=lay.D,
    )


def basis_summary(Bs, c_true, sig_c, f, r):
    """
    One BASIS SWEEP row of an SRIF result r: coefficients, truth omission (% of
    the contrast acceleration), resolved, SNR = |c|/σ_post > 3 (formal, and with
    the total σ: formal + consider + omission bias), accurate, DFS and the highest
    resolved degree m.
    """
    na = Bs["T"].shape[0] // 4
    om = np.linalg.norm(f[na:] - Bs["T"][na:] @ c_true) / np.linalg.norm(f[na:])
    v = np.sqrt(np.diag(r["Sc"]))
    a = np.sqrt(v**2 + np.sum(r["Ec"] ** 2, axis=1) + r["bias"] ** 2)
    res = v < RESOLVED
    cs = np.abs(c_true / sig_c)
    return (
        len(sig_c), 100 * om, res.sum(), np.sum(cs / v > 3), np.sum(cs / a > 3),
        np.sum(a < RESOLVED), r["dfs"], Bs["m"][res].max() if res.any() else -1,
    )


def basis_case(M, camp, L, vis, los, modes, rcond):
    """The nominal filter on another CH basis / truth definition (BASIS_CASES)."""
    Bs = ch_basis(M.cyl, M.obs, modes)
    c, s0, f, Q, _ = ch_truth_and_prior(M, Bs, rcond)
    stm = [
        arc_transitions(M, Bs, c, camp["rS"][a], camp["TS"][a], camp["cS"][a], camp["meta"][a, 3])
        for a in range(len(camp["nodes"]))
    ]
    r = srif_campaign(camp, stm, Layout(s0, len(L), SIG_SA), L, vis, los, Q=Q, record=False)
    return basis_summary(Bs, c, s0, f, r)


# ═══════════════════════════════════════════════════════════════════════════
# COEFFICIENT COVARIANCE FOR THE DENSITY SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════


def preset_covariance(RG, Bs, c_true, sig_c, ch_modes, Q=None):
    """
    OD solution of one CH preset: estimate only its (m < n_m, n ≤ n_n) columns
    (`subset_solution`) and write it in the preset's cyl_basis packing.
    Returns Σ_c (formal + consider), Σ_c formal and the noise-free estimate ĉ
    (truth + deterministic error), B_0n entries zero.
    """
    nm, nn = ch_modes
    ng, K = RG.shape[0], len(sig_c)
    est = (Bs["m"] < nm) & (Bs["n"] <= nn)
    F, E, d = subset_solution(RG, K, est, np.concatenate([c_true / sig_c, np.zeros(ng - K)]), Q)
    s = sig_c[est]
    idx = 2 * (Bs["m"][est] * nn + Bs["n"][est] - 1) + Bs["trig"][est]
    Sf, St, c0 = np.zeros((2 * nm * nn,) * 2), np.zeros((2 * nm * nn,) * 2), np.zeros(2 * nm * nn)
    Sf[np.ix_(idx, idx)] = F @ F.T * np.outer(s, s)
    St[np.ix_(idx, idx)] = Sf[np.ix_(idx, idx)] + E @ E.T * np.outer(s, s)
    c0[idx] = c_true[est] + d * s
    return St, Sf, c0


def od_whitener(Sigma_c, rtol=1e-12):
    """W with WᵀW = Σ_c⁺ (eigen pseudo-inverse): weight a CH design as W @ A."""
    lam, Q = np.linalg.eigh(0.5 * (Sigma_c + Sigma_c.T))
    k = lam > rtol * lam.max()
    return (Q[:, k] / np.sqrt(lam[k])).T


def load_od_covariance(preset="new", path=NPZ_PATH, consider=True):
    """
    The exported OD covariance of one CH preset, for the density scripts:

        d = load_od_covariance("new")
        W = od_whitener(d["Sigma_c"])          # replaces 1/od_sigma
        A_w = W @ A_ch                         # instead of A_ch / od_sigma

    `consider=True` (default) is formal + consider (instrument and frame
    errors); False the formal SRIF covariance alone.  d["bias_c"] is the
    deterministic error of the OD estimate against the preset's own
    coefficients Φ⁺f (omission, truncation, aliasing).  Both are in the
    preset's cyl_basis packing, B_0n rows and columns zero.  Valid for the
    pt1 cylinder and samples it was computed on (d["obs"]).
    """
    z = np.load(path)
    if "Sigma_c_" + preset not in z.files:
        raise KeyError(f"preset {preset!r} not exported: its modes exceed ch_master {tuple(z['ch_master'])}")
    out = {k: z[k] for k in z.files if not k.startswith(("Sigma_c_", "bias_c_"))}
    out["Sigma_c"] = z["Sigma_c_" + preset + ("" if consider else "_formal")]
    out["bias_c"] = z["bias_c_" + preset]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════


def complete(Bs, ok):
    """Largest square (N, N), m < N and n ≤ N, whose coefficients all pass `ok` (0 if none)."""
    N = 0
    while N <= Bs["m"].max() and ok[(Bs["m"] <= N) & (Bs["n"] <= N + 1)].all():
        N += 1
    return N


def by_degree(Bs, ratio, acc_ratio, snr, bias_sig, Sc_scaled):
    """Per degree m and per order n."""
    res, acc = ratio < RESOLVED, acc_ratio < RESOLVED
    dfs = 1.0 - np.diag(Sc_scaled)
    print(
        "\n  per DEGREE m  (resolved: σ_post/σ_prior < 1/√2;"
        "\n                accurate: √(σ_post² + σ_consider² + bias²)/σ_prior < 1/√2;"
        "\n                VR: Σ(1 − σ_post²/σ_prior²), the DFS only under a diagonal prior)"
    )
    print("    m   coeffs  resolved  accurate  max n res.  med σ/σ0   med SNR   max SNR  SNR>3   med|bias|/σ     VR")
    for m in np.unique(Bs["m"]):
        k = Bs["m"] == m
        print(
            f"   {m:2d}   {k.sum():5d}    {np.sum(res[k]):5d}     {np.sum(acc[k]):5d}     "
            f"{Bs['n'][k & res].max() if (k & res).any() else 0:5d}     "
            f"{np.median(ratio[k]):8.3f}  {np.median(snr[k]):9.2e}  {snr[k].max():7.2f}  {np.sum(snr[k] > 3):5d}   "
            f"{np.median(bias_sig[k]):9.2e}   {dfs[k].sum():6.2f}"
        )
    print("\n  per ORDER n")
    print("    n   coeffs  resolved  accurate  max m res.  med σ/σ0   med SNR   max SNR  SNR>3     VR")
    for n in np.unique(Bs["n"]):
        k = Bs["n"] == n
        print(
            f"   {n:2d}   {k.sum():5d}    {np.sum(res[k]):5d}     {np.sum(acc[k]):5d}     "
            f"{Bs['m'][k & res].max() if (k & res).any() else -1:5d}     "
            f"{np.median(ratio[k]):8.3f}  {np.median(snr[k]):9.2e}  {snr[k].max():7.2f}  {np.sum(snr[k] > 3):5d}   {dfs[k].sum():6.2f}"
        )


def rms_by_m(x, n_n):
    m = np.arange(len(x)) // (2 * n_n)
    return np.array([np.sqrt(np.mean(x[m == i] ** 2)) for i in np.unique(m)])


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════
# Five files, the standard OD set: (1) geometry, (2) 1σ of the arc-local
# states and landmarks at every filter epoch (scatter), (3) SNR of the CH
# coefficients A_mn, B_mn per degree m after each flyover, (4) their
# correlations, (5) degree spectrum.  Every plotted σ (and correlation) is
# formal + consider.  GLOBAL house style throughout (Okabe–Ito colours, Title
# Case labels with units, frameless legends above the axes).

# Real LaTeX (the paper's typeface); OD_NO_TEX=1 falls back to mathtext.  Every
# label below is valid in both modes: maths in $...$, no bare unicode.
USE_TEX = os.environ.get("OD_NO_TEX", "") == ""
mpl.rcParams.update(
    {"text.usetex": USE_TEX, "font.family": "serif" if USE_TEX else "STIXGeneral"}
)
FSC = G.FONT_SCALE
LEG = dict(fontsize=8.5 * FSC, frameon=False)
PCT = r"[$\%$]" if USE_TEX else "[%]"
R_SHOW = 30.0  # km: tracks are drawn inside this radius
EPOCH = dict(ls="none", marker="o", ms=2.4, mew=0, alpha=0.45, rasterized=True)  # every epoch
MK = dict(marker="o", ms=6.5, mec="k", mew=0.4, lw=1.2, zorder=3)  # per flyover / degree
END = dict(ls="none", marker="o", ms=6.5, mec="k", mew=0.4, zorder=3)  # end of arc
XI = r"CH Coefficients $A_{mn}, B_{mn}$"


def _save(fig, outdir, name):
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    G._savefig(fig, os.path.join(outdir, PREFIX + name), bbox_inches="tight")
    plt.close(fig)


def _grid(ax):
    ax.grid(True, which="major", ls=":", alpha=0.45)
    ax.set_axisbelow(True)


def _km_cyl(M):
    c = M.cyl
    return G.Cylinder(
        center=c.center * M.LU, radius=c.radius * M.LU, height=c.height * M.LU,
        alpha=c.alpha, R=c.R,
    )


def _tracks(M, camp):
    """Per arc (km, body frame): the track inside R_SHOW and its in-cylinder part."""
    out = []
    for p in camp["nodes"][..., :3]:
        p = p.copy()
        p[np.linalg.norm(p, axis=1) > R_SHOW] = np.nan
        q = p.copy()
        q[~in_cylinder(M.cyl, np.nan_to_num(p) / M.LU) | np.isnan(p[:, 0])] = np.nan
        out.append((p, q))
    return out


def _degree_map(mmax):
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, mmax + 1.5), 256)
    return mpl.cm.ScalarMappable(norm=norm, cmap="viridis")


def _ch_cov(res):
    """Scaled end-of-campaign CH covariance, formal + consider: every plotted σ includes consider."""
    return res["Sc"] + res["Ec"] @ res["Ec"].T


def _rms_m(v, Bs, ms):
    """RMS of v over the CH coefficients of each degree m (last axis)."""
    k = [Bs["m"] == m for m in ms]
    return np.array([np.sqrt(np.mean(v[..., kk] ** 2, axis=-1)) for kk in k]).T


def fig_geometry(M, camp, L, n_seen, occ, outdir):
    """fig1: the flyovers, the cylinder, the landmarks and the Earth-occulted epochs."""
    V = M.V * M.LU
    cyl = _km_cyl(M)
    trk = _tracks(M, camp)
    pts = np.vstack([V, G.cylinder_hull(cyl)] + [p[~np.isnan(p[:, 0])] for p, _ in trk])
    fig = plt.figure(figsize=(8.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    G.shaded_body(ax, V, M.F[:: max(1, len(M.F) // 8000)])
    G.draw_cylinder(ax, cyl, label="CH Cylinder")
    for k, (p, q) in enumerate(trk):
        ax.plot(*p.T, color=G.COLOR[2], lw=0.7, alpha=0.7,
                label="Flyover Arcs" if k == 0 else None)
        ax.plot(*q.T, color=G.COLOR[0], lw=2.2,
                label="Inside the Cylinder" if k == 0 else None)
    if len(occ):
        ax.scatter(*occ.T, color=G.COLOR[1], s=6, depthshade=False, linewidths=0, zorder=5,
                   label="Earth Occulted")
    sc = ax.scatter(*L.T, c=n_seen, cmap="viridis", s=28, depthshade=False,
                    edgecolors="k", linewidths=0.4, zorder=6, label="Landmarks")
    for set_lab, a in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel), "xyz"):
        set_lab(rf"${a}$ [km]", labelpad=G.LPAD3D)
    G.set_axes_true_shape(ax, pts)
    ax.legend(loc="upper left", markerscale=1.3, **LEG)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.12)
    cb.set_label("Images per Landmark [-]")
    _save(fig, outdir, "fig1_geometry.pdf")


def fig_nav(res, Bs, outdir):
    """
    fig2: 1σ (formal + consider) of every estimated state except the CH
    coefficients at every filter epoch (scatter).
    """
    D = res["D"]
    Na, ne, _ = res["sig"].shape
    K = len(Bs["m"])
    S = np.hypot(res["sig"], res["sigc"]).reshape(Na * ne, -1) * D  # formal + consider; prior 1
    lmk = lambda X: np.median(
        1e3 * np.linalg.norm(X[:, N_LOC + K :].reshape(len(X), -1, 3), axis=-1), axis=1)
    rss = lambda X, i, f: f * np.linalg.norm(X[:, i : i + 3], axis=1)
    rows = [
        (r"Position $1\sigma$ [m]", [
            (rss(S, 0, 1e3), np.sqrt(3) * SIG_R0 * 1e3, "Spacecraft (RSS)", G.COLOR[0]),
            (lmk(S), np.sqrt(3) * SIG_LMK * 1e3, "Landmarks (Median)", G.COLOR[2]),
        ]),
        (r"Velocity $1\sigma$ [mm/s]", [
            (rss(S, 3, 1e6), np.sqrt(3) * SIG_V0 * 1e6, "Spacecraft (RSS)", G.COLOR[0]),
        ]),
        (r"Stochastic Acceleration $1\sigma$ [nm/s$^2$]", [
            (1e12 * S[:, 6 + i], 1e12 * D[6], rf"$a_{{\mathrm{{SA}},{c}}}$ (Inertial)", col)
            for i, (c, col) in enumerate(zip("xyz", (G.COLOR[3], G.COLOR[1], G.COLOR[4])))
        ]),
    ]
    x = (np.arange(1, Na + 1)[:, None] + 0.8 * (np.arange(ne) / (ne - 1) - 0.5)).ravel()
    xe = np.arange(1, Na + 1) + 0.4
    end = lambda v: v.reshape(Na, ne)[:, -1]
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7.6, 10.6))
    for k, (ax, (ylab, series)) in enumerate(zip(axs, rows)):
        for y, prior, lab, col in series:
            ax.plot(x, y, color=col, **EPOCH)
            ax.plot(xe, end(y), color=col, label=lab, **END)
            ax.axhline(prior, color=col, ls="--", lw=1.1, zorder=1)
        ax.plot([], [], color="0.35", ls="--", lw=1.1, label="Prior")
        if k == 0:  # the marker key, once
            ax.plot([], [], color="0.35", label="Every Filter Epoch",
                    **{**EPOCH, "ms": 4, "alpha": 0.7})
            ax.plot([], [], color="0.6", label="End of Arc", **END)
        ax.set_yscale("log" if k < 2 else "linear")  # a_SA spans < 10%: linear
        ax.set_ylabel(ylab)
        _grid(ax)
        G.centred_rows_legend(ax, 3 if k == 0 else 4, fontsize=8.5, y0=1.01, dy=0.12,
                              handletextpad=0.3, columnspacing=1.2)
    axs[-1].set_xlim(0.4, Na + 0.8)
    axs[-1].set_xticks([1] + list(range(5, Na + 1, 5)))
    axs[-1].set_xlabel(rf"Flyover (Each Arc $\pm {ARC_HALF / 3600:g}$ h) [-]")
    _save(fig, outdir, "fig2_nav_sigma.pdf")


def fig_ch(res, Bs, c_true, sig_c, outdir):
    """
    fig3: SNR of the A_mn, B_mn of each degree m after each flyover (σ formal +
    consider).
    """
    K = len(sig_c)
    ms = np.unique(Bs["m"])
    ch = slice(N_LOC, N_LOC + K)
    r = np.vstack([np.ones(K), np.hypot(res["sig"][:, -1, ch], res["sigc"][:, -1, ch])])  # 0 = prior
    k = np.arange(len(r))
    rms_c = _rms_m(c_true, Bs, ms)
    snr = rms_c[None, :] / _rms_m(r * sig_c, Bs, ms)
    sm = _degree_map(ms[-1])
    fig, ax = plt.subplots(figsize=(7.8, 5.2), layout="constrained")
    for i, m in enumerate(ms):
        c = sm.to_rgba(m)
        ax.plot(k, snr[:, i], color=c, **MK)
    ax.axhline(3.0, color="k", ls="--", lw=1.0, label=r"$\mathrm{SNR} = 3$")
    ax.set_ylabel(r"$\mathrm{SNR} = \mathrm{RMS}_m(A, B)\,/\,\mathrm{RMS}_m(\sigma)$ [-]")
    ax.set_yscale("log")
    _grid(ax)
    G.centred_rows_legend(ax, 1, fontsize=8.5, y0=1.0, handletextpad=0.4)
    ax.set_xlim(-0.6, k[-1] + 0.6)
    ax.set_xticks(range(0, k[-1] + 1, 5))
    ax.set_xlabel("Flyovers Processed [-]")
    cb = fig.colorbar(sm, ax=ax, ticks=ms, fraction=0.04, pad=0.02, aspect=40)
    cb.set_label(r"CH Degree $m$ [-]")
    cb.ax.minorticks_off()
    _save(fig, outdir, "fig3_ch_degree.pdf")


def fig_corr(res, Bs, outdir):
    """
    fig4: posterior correlations (formal + consider) of A_mn, B_mn, by degree m and, inside
    each degree, A_mn then B_mn (dotted divider) with n ascending.  Degrees the
    data leave at the prior (σ/σ0 > 0.99 for every coefficient: an identity
    block, no correlation to show) are left out.
    """
    P = _ch_cov(res)
    r = np.sqrt(np.diag(P))
    keep = [m for m in np.unique(Bs["m"]) if r[Bs["m"] == m].min() < 0.99]
    o = np.lexsort((Bs["n"], Bs["trig"], Bs["m"]))
    o = o[np.isin(Bs["m"][o], keep)]
    ms, tr = Bs["m"][o], Bs["trig"][o]
    C = (P / np.outer(r, r))[np.ix_(o, o)]
    fig, ax = plt.subplots(figsize=G.FS_SQ)
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    for e in np.flatnonzero(np.diff(ms)) + 0.5:
        ax.axhline(e, color="k", lw=0.6)
        ax.axvline(e, color="k", lw=0.6)
    for e in np.flatnonzero(np.diff(tr) & (np.diff(ms) == 0)) + 0.5:  # A | B
        ax.axhline(e, color="k", lw=0.5, ls=":")
        ax.axvline(e, color="k", lw=0.5, ls=":")
    u, cnt = np.unique(ms, return_counts=True)
    cen = [np.flatnonzero(ms == m).mean() for m in u]
    ax.minorticks_off()
    ax.set_xticks(cen, [str(m) for m in u])
    ax.set_yticks(cen, [str(m) for m in u])
    for t in ax.get_xticklabels() + ax.get_yticklabels():  # thin blocks: small labels
        if cnt[u == int(t.get_text())][0] < 3:
            t.set_fontsize(0.55 * t.get_fontsize())
    ax.set_xlabel(XI + r", by Degree $m$ [-]")
    ax.set_ylabel(XI + r", by Degree $m$ [-]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Posterior Correlation [-]")
    _save(fig, outdir, "fig4_correlation.pdf")


def fig_spectrum(res, Bs, c_true, sig_c, outdir):
    """fig5: per-degree-m RMS of the truth, the prior and the nominal posterior (formal + consider)."""
    ms = np.unique(Bs["m"])
    ln = dict(lw=1.2)
    fig, ax = plt.subplots(figsize=G.FS)
    ax.plot(ms, _rms_m(c_true, Bs, ms), "-o", color="k", ms=7.5, label="Truth", **ln)
    ax.plot(ms, _rms_m(sig_c, Bs, ms), "-s", color="0.4", ms=8, mfc="none", mew=1.4,
            label=r"Prior $\sigma_{\mathrm{prior}}$", **ln)
    ax.plot(ms, _rms_m(np.sqrt(np.diag(_ch_cov(res))) * sig_c, Bs, ms), "-^", color=G.COLOR[2],
            ms=8, mec="k", mew=0.5, label=r"Posterior $\sigma_{\mathrm{post}}$", **ln)
    ax.set_yscale("log")
    ax.set_xticks(ms)
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim(ms[0] - 0.6, ms[-1] + 0.6)
    ax.set_xlabel(r"CH Degree $m$ [-]")
    ax.set_ylabel(r"RMS of $A_{mn}, B_{mn}$ over Degree $m$ [-]")
    _grid(ax)
    G.centred_rows_legend(ax, 3, fontsize=8.5, handletextpad=0.3, columnspacing=1.6)
    _save(fig, outdir, "fig5_spectrum.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    T0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    print(G.SEP)
    print("  SRIF covariance analysis: resolvable CH degree/order over the pt1 cylinder")
    print(G.SEP)
    M = Model()
    print(
        f"  LU = {M.LU:.3f} km, GM/LU² = {M.ACC:.3e} km/s², cylinder base "
        f"{M.cyl.center[2] * M.LU:.2f} km, R = {M.cyl.radius * M.LU:.2f} km, "
        f"H = {M.cyl.height * M.LU:.2f} km, {len(M.obs)} samples"
    )

    # ── raw CH coefficients, truth, prior ────────────────────────────────
    Bs = ch_basis(M.cyl, M.obs)
    K = Bs["T"].shape[1]
    t0 = time.time()
    Fd, hit = cached("prior", (shape_density_fields, Model), M, lambda: dict(Fd=shape_density_fields(M)))
    M.prior_fields = Fd["Fd"]
    c_true, sig_c, f_true, Qp, kap = ch_truth_and_prior(M, Bs)
    na = len(M.obs)
    rq = np.abs(c_true) / sig_c
    print(
        f"  CH prior: {PRIOR_DRAWS} shape models (radial σ {1e3 * PRIOR_SHAPE_SIGMA:.0f} m, correlation "
        f"{PRIOR_SHAPE_CORR:.1f} km) × bulk density (σ {100 * PRIOR_RHO_SIGMA:.1f}%), each refit as "
        f"c = Φ⁺f [{time.time() - t0:.0f} s{', draws cached' if hit else ''}]\n  prior = their full covariance × {kap:.3g} "
        f"(σ × {np.sqrt(kap):.3g}), rank {Qp.shape[1]}/{K}; their acceleration RMS is "
        f"{100 * np.sqrt(np.mean(M.prior_fields[na:] ** 2) / np.mean(f_true[na:] ** 2)):.1f}% of the "
        f"truth contrast; |c_true|/σ_prior median {np.median(rq):.1f}, max {rq.max():.0f}"
    )
    om = f_true - Bs["T"] @ c_true
    print(
        f"  CH coefficients: {K} raw A_mn, B_mn of {CH_MASTER} (B_0n dropped; degree m ≤ "
        f"{Bs['m'].max()}, order n ≤ {Bs['n'].max()}); truth / prior c = Φ⁺f at rcond "
        f"{TRUTH_RCOND:.0e}\n  truth omission {100 * np.linalg.norm(om) / np.linalg.norm(f_true):.1f}% "
        f"of the contrast field ({100 * np.linalg.norm(om[na:]) / np.linalg.norm(f_true[na:]):.1f}% "
        f"of its acceleration)"
    )

    # ── campaign ─────────────────────────────────────────────────────────
    camp, hit = cached("campaign", (design_campaign, TruthGravity, Model), M,
                       lambda: design_campaign(M, TruthGravity(M)))
    if hit:
        print(f"  campaign: cached design ({hit}; OD_NO_CACHE=1 redesigns)")
    _, dv = sortie_budget(camp["X0"], camp["meta"])
    ra, nra = np.unique(camp["meta"][:, 4], return_counts=True)
    print(
        f"  sorties per home orbit: "
        + ", ".join(f"{r:.0f} km × {k}" for r, k in zip(ra, nra))
        + f";  Δv deorbit + recircularize {1e3 * dv.min():.2f}–{1e3 * dv.max():.2f} m/s "
        f"per sortie, {1e3 * dv.sum():.0f} m/s in all"
    )
    Na = len(camp["nodes"])
    t0 = time.time()
    stm = [
        arc_transitions(
            M, Bs, c_true, camp["rS"][a], camp["TS"][a], camp["cS"][a], camp["meta"][a, 3]
        )
        for a in range(Na)
    ]
    dwell = np.array([s[2].sum() * DT_OBS for s in stm])
    om_orb = np.linalg.norm(np.concatenate([s[3] for s in stm]), axis=1)
    print(
        f"  {Na} arcs: in-cylinder dwell {dwell.min():.0f}–{dwell.max():.0f} s "
        f"(total {dwell.sum() / 60:.0f} min); omission there RMS "
        f"{np.sqrt(np.mean(om_orb**2)):.1e} km/s²; STMs [{time.time() - t0:.0f} s]"
    )

    imgs = image_epochs(camp)
    L, Nrm = pick_landmarks(M, imgs)
    vis = image_visibility(M, imgs, L, Nrm)
    nvis = np.array([len(v) for v in vis.values()])
    print(
        f"  {len(L)} landmarks; {len(imgs)} images, landmarks per image "
        f"median {np.median(nvis):.0f} (max {nvis.max()}, "
        f"{np.mean(nvis == 0) * 100:.0f}% empty)"
    )
    n_seen = np.bincount(np.concatenate(list(vis.values())), minlength=len(L))
    los = earth_los(M, camp)
    occ = ~los[1]
    print(
        f"  radio: Doppler {1e6 * SIG_DOPPLER:.2f} mm/s every {DT_OBS:.0f} s, range "
        f"{1e3 * SIG_RANGE:.0f} m every {RANGE_EVERY * DT_OBS / 60:.0f} min; Earth occulted at "
        f"{100 * occ.mean():.0f}% of the epochs (per arc {occ.sum(1).min()}–{occ.sum(1).max()})"
    )
    print(
        f"  filter state: {N_LOC} local + {K} CH + {3 * len(L)} landmark; consider: "
        f"omission + {N_CON} instrument / frame ("
        + ", ".join(f"{nm} {sg:.0e}" for nm, _, sg in CONSIDER) + ")"
    )

    # ── SRIF ─────────────────────────────────────────────────────────────
    cases = {}
    for name, ss, bt in SA_CASES[:1] if QUICK else SA_CASES:  # nominal first: per-epoch σ for fig2
        t0 = time.time()
        cases[name] = srif_campaign(camp, stm, Layout(sig_c, len(L), ss), L, vis, los, batch=bt, Q=Qp,
                                    record=name == "nominal")
        print(
            f"  SRIF [{name}: σ_SA {ss:.1e} km/s², {bt / 60:.0f}-min batches] "
            f"done [{time.time() - t0:.0f} s]"
        )
    nom = cases["nominal"]
    t0 = time.time()
    info = srif_campaign(camp, stm, Layout(sig_c, len(L)), L, vis, los, record=False)  # data information in c
    print(f"  SRIF [nominal, identity CH prior: data information for the subset solutions] "
          f"done [{time.time() - t0:.0f} s]")

    S0 = np.outer(sig_c, sig_c)
    Sc_f = nom["Sc"] * S0  # formal
    Sc = Sc_f + nom["Ec"] @ nom["Ec"].T * S0  # formal + consider
    sig_post = np.sqrt(np.diag(Sc_f))
    sig_con = np.sqrt(np.sum(nom["Ec"] ** 2, axis=1)) * sig_c
    ratio = sig_post / sig_c
    snr = np.abs(c_true) / sig_post
    bias = nom["bias"] * sig_c
    bias_sig = np.abs(bias) / sig_post
    acc_ratio = np.sqrt(sig_post**2 + sig_con**2 + bias**2) / sig_c

    by_degree(Bs, ratio, acc_ratio, snr, bias_sig, nom["Sc"])

    res = ratio < RESOLVED
    acc = acc_ratio < RESOLVED
    print(
        f"\n  RESOLVED {res.sum()}/{K} coefficients, DFS "
        f"{nom['dfs']:.1f} of {Qp.shape[1]}; complete square (N, N) = "
        f"{complete(Bs, res)}; highest resolved degree m = "
        f"{Bs['m'][res].max() if res.any() else '-'}, order n = {Bs['n'][res].max() if res.any() else '-'}"
        f"\n  ACCURATE (formal + consider + omission) {acc.sum()}/{K}; "
        f"complete square (N, N) = {complete(Bs, acc)}"
    )
    print("    case              resolved (N)   accurate (N)    DFS    end-of-arc pos. / vel. (formal)")
    for name, r in cases.items():
        v = np.sqrt(np.diag(r["Sc"]))
        a = np.sqrt(v**2 + np.sum(r["Ec"] ** 2, axis=1) + r["bias"] ** 2)
        Se = r["sig"][:, -1] * r["D"]
        print(
            f"    {name:16s}  {np.sum(v < RESOLVED):4d} ({complete(Bs, v < RESOLVED):2d})"
            f"     {np.sum(a < RESOLVED):4d} ({complete(Bs, a < RESOLVED):2d})"
            f"    {r['dfs']:5.1f}    "
            f"{1e3 * np.median(np.linalg.norm(Se[:, :3], axis=1)):5.2f} m / "
            f"{1e6 * np.median(np.linalg.norm(Se[:, 3:6], axis=1)):5.3f} mm/s"
        )
    Sn = nom["sig"] * nom["D"]
    Cn = nom["sigc"] * nom["D"]
    pos = np.linalg.norm(Sn[..., :3], axis=-1)  # (arc, epoch)
    pos_t = np.sqrt(pos**2 + np.sum(Cn[..., :3] ** 2, axis=-1))
    s3 = np.linalg.norm(Sn[:, -1, N_LOC + K :].reshape(Na, -1, 3), axis=-1)
    s3t = np.sqrt(s3**2 + np.sum(Cn[:, -1, N_LOC + K :].reshape(Na, -1, 3) ** 2, axis=-1))
    print(
        f"  nominal, end of each arc (median over arcs), formal → formal + consider:"
        f"\n    position {1e3 * np.median(pos[:, -1]):.2f} → {1e3 * np.median(pos_t[:, -1]):.2f} m, "
        f"velocity {1e6 * np.median(np.linalg.norm(Sn[:, -1, 3:6], axis=1)):.3f} → "
        f"{1e6 * np.median(np.sqrt(np.sum(Sn[:, -1, 3:6] ** 2 + Cn[:, -1, 3:6] ** 2, axis=1))):.3f} mm/s, "
        f"a_SA {1e12 * np.median(np.linalg.norm(Sn[:, -1, 6:9], axis=1)):.1f} nm/s² (prior "
        f"{1e12 * np.sqrt(3) * SIG_SA:.1f}); landmarks (final) median {1e3 * np.median(s3[-1]):.2f} → "
        f"{1e3 * np.median(s3t[-1]):.2f} m"
    )
    dp = np.diff(pos, axis=1)
    print(
        f"  within an arc the position σ GROWS at {100 * np.mean(dp > 0):.0f}% of the epochs "
        f"(occultations, image gaps, a_SA batch restarts): median step +{1e3 * np.median(dp[dp > 0]):.2f} m, "
        f"largest +{1e3 * dp.max():.2f} m"
    )
    print("  consider breakdown  (median over the CH coefficients of σ_consider/σ_formal; "
          "end-of-arc position, median)")
    i0 = 0
    for nm, c, sg in CONSIDER:
        g = slice(i0, i0 + c)
        rx = np.sqrt(np.sum(nom["Ec"][:, g] ** 2, axis=1)) / np.sqrt(np.diag(nom["Sc"]))
        rp = np.sqrt(np.sum((nom["Enav"][:, :3, g] * nom["D"][:3, None]) ** 2, axis=(1, 2)))
        print(f"    {nm:20s} σ {sg:.0e}:  CH {np.median(rx):6.3f} (max {rx.max():5.2f}),"
              f"  position {1e3 * np.median(rp):6.3f} m")
        i0 += c
    rx = sig_con / sig_post
    print(f"    {'all':20s}          :  CH {np.median(rx):6.3f} (max {rx.max():5.2f})")
    if res.any():
        print(
            f"  consider (omission) bias: median |bias|/σ over resolved coefficients "
            f"{np.median(bias_sig[res]):.2f}, max {bias_sig[res].max():.1f}"
        )

    sweep = truncation_sweep(info["RG"], Bs, c_true, sig_c, f_true, Qp)
    print(
        "\n  TRUNCATION SWEEP  (estimate only m < N, n ≤ N; RMS acceleration"
        "\n                     error at the samples / contrast acceleration, omission included)"
    )
    print("    N  estimated  resolved   noise 1σ   consider   determ.    total")
    for Lc, ne, nr, sn, sc, sd in sweep:
        print(
            f"   {Lc:2.0f}   {ne:6.0f}    {nr:6.0f}    {100 * sn:7.2f}%  {100 * sc:7.2f}%  "
            f"{100 * sd:7.2f}%  {100 * np.sqrt(sn**2 + sc**2 + sd**2):7.2f}%"
        )
    tot = np.linalg.norm(sweep[:, 3:6], axis=1)
    print(f"  best truncation: (N, N) = ({sweep[np.argmin(tot), 0]:.0f}, {sweep[np.argmin(tot), 0]:.0f}) ({100 * tot.min():.2f}%)")

    # ── basis sweep: fewer orders per degree vs a looser truth definition ──
    t0 = time.time()
    if QUICK:
        print("\n  BASIS SWEEP skipped (OD_QUICK)")
    else:
        print(
            "\n  BASIS SWEEP  (nominal filter re-run on each CH basis and truth / prior rcond;"
            "\n               omission = acceleration the truth coefficients leave out)"
        )
        print("    modes      rcond   coeffs  omission  resolved  SNR>3 (total)  accurate    DFS  max m res.")
        rows = [("*", CH_MASTER, TRUTH_RCOND, basis_summary(Bs, c_true, sig_c, f_true, nom))]
        rows += [(" ", md, rc, basis_case(M, camp, L, vis, los, md, rc)) for md, rc in BASIS_CASES]
        for tag, md, rc, (k, o, nr, ns, nt, na_, dfs, mr) in rows:
            print(f"  {tag} {str(md):9s}  {rc:.0e}   {k:4d}    {o:5.1f}%    {nr:4d}    {ns:4d}  ({nt:3d})     {na_:4d}"
                  f"    {dfs:5.1f}     {mr:3d}")
        print(f"  (* nominal)  [{time.time() - t0:.0f} s]")

    # ── Σ_c for the density-script presets ───────────────────────────────
    print("\n  CH coefficients (density-script presets) — OD (formal + consider) vs od_sigma(eps=1e-3)")
    export = {}
    for name, pr in PRESETS.items():
        modes, rc = pr["ch_modes"], pr["rcond"]
        if modes[0] > CH_MASTER[0] or modes[1] > CH_MASTER[1]:
            print(f"   [{name}] modes {modes}: outside the {CH_MASTER} basis, not exported")
            continue
        Sp, Sp_f, c_od = preset_covariance(info["RG"], Bs, c_true, sig_c, modes, Qp)
        Pinv = G.ch_pinv_for(M.cyl, M.obs, modes, rc)
        c_tot = G.ch_coefficients_total(M.beta, M.P, M.bulk, M.obs, Pinv)
        s_rule = G.od_sigma(c_tot, EPS_RULE, kind="ch", n_max=modes[1])
        s_od = np.sqrt(np.clip(np.diag(Sp), 0, None))
        b_c = np.where(np.diag(Sp) > 0, c_od - c_tot, 0.0)  # OD estimate vs the preset's Φ⁺f
        d = dict(
            c=rms_by_m(c_tot, modes[1]),
            od=rms_by_m(s_od, modes[1]),
            rule=rms_by_m(s_rule, modes[1]),
            modes=modes,
            rcond=rc,
        )
        export["Sigma_c_" + name] = Sp
        export["Sigma_c_" + name + "_formal"] = Sp_f
        export["bias_c_" + name] = b_c
        eps_eff = EPS_RULE * d["od"] / d["rule"]
        print(f"   [{name}] modes {modes}, rcond {rc:.0e}: effective eps per m")
        print("     " + "  ".join(f"m{m}:{e:.1e}" for m, e in enumerate(eps_eff)))

        A_ch = Pinv @ G.A_field_contrast(M.P, M.bulk, M.obs)
        W = od_whitener(Sp)
        Aw = W @ A_ch
        Ci = np.linalg.inv(np.eye(len(M.beta)) + Aw.T @ Aw)
        s_beta_od = np.sqrt(np.diag(Ci))
        b_beta = Ci @ Aw.T @ (W @ b_c)
        s_beta_rule = np.sqrt(
            np.diag(np.linalg.inv(G.fisher_mass_fractions([(A_ch, s_rule)], 1.0)))
        )
        print("     CH-only σ_β     " + "  ".join(f"{n[:12]:>12s}" for n in M.names))
        print("       SRIF f + c    " + "  ".join(f"{s:12.2e}" for s in s_beta_od))
        print("       od_sigma      " + "  ".join(f"{s:12.2e}" for s in s_beta_rule))
        print("       OD bias       " + "  ".join(f"{b:+12.2e}" for b in b_beta))
        print("       truth β       " + "  ".join(f"{b:+12.3f}" for b in M.beta))

    # ── figures, export ──────────────────────────────────────────────────
    q = int(round(DT_OBS / DT_INT))
    fig_geometry(M, camp, L, n_seen, camp["nodes"][:, ::q, :3][occ], OUTDIR)
    fig_nav(nom, Bs, OUTDIR)
    fig_ch(nom, Bs, c_true, sig_c, OUTDIR)
    fig_corr(nom, Bs, OUTDIR)
    fig_spectrum(nom, Bs, c_true, sig_c, OUTDIR)
    np.savez(
        NPZ_PATH,
        obs=M.obs,
        cyl_center=M.cyl.center,
        cyl_radius=M.cyl.radius,
        cyl_height=M.cyl.height,
        alpha=M.cyl.alpha,
        ch_master=np.array(CH_MASTER),
        truth_rcond=TRUTH_RCOND,
        m=Bs["m"], trig=Bs["trig"], n=Bs["n"], cols=Bs["cols"], T=Bs["T"],
        Sigma_ch=Sc, Sigma_ch_formal=Sc_f, c_true=c_true, sigma_prior=sig_c, bias_c=bias,
        Sigma_prior_ch=Qp @ Qp.T * np.outer(sig_c, sig_c), prior_inflate=kap,
        resolved=res, accurate=acc, sweep=sweep,
        **export,
    )
    print(f"\n  saved {NPZ_PATH} and {OUTDIR}/{PREFIX}fig*.pdf   [{time.time() - T0:.0f} s]")
    return dict(M=M, Bs=Bs, camp=camp, cases=cases, info=info, Q=Qp, sweep=sweep)


if __name__ == "__main__":
    main()
