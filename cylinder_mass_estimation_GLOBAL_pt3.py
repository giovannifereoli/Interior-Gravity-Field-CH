"""
Part 3 — WHERE should the cylinders go?  Adaptive vs geometric placement
========================================================================
Author: Giovanni Fereoli / experiment build

Question
--------
pt2 placed the CH network by farthest-point sampling: pure geometry — it knows
nothing about the interior, the measurements, or the SH solution.  It is
deterministic, reproducible and independent of everything, which is exactly why
it is attractive.  But is it any good?  Two smarter, SH-aware criteria are
implemented here and raced against it:

  (1) SH MODEL ERROR / UNCERTAINTY  — put cylinders where SH is worst.
        eps(r)      = |g_truth(r) - g_SH,L(r)|          truncation / model error
        sigma_g(r)  = sqrt( tr[ H_g(r) P_SH H_g(r)^T ] ),  H_g = d g / d c
      The first needs a truth field (simulation, crossovers, tracking residuals);
      the second needs a posterior covariance P_SH on the Stokes coefficients.

  (2) INCREMENTAL FISHER INFORMATION (D-optimal sensor placement) — put
      cylinders where they measure something SH CANNOT.  For a candidate
      cylinder at surface site s, with y = h(c, theta_s) and H = [H_SH  H_CH(s)],

        P_perp   = I - H_SH (H_SH^T W H_SH)^-1 H_SH^T W
        DeltaF(s)= H_CH(s)^T P_perp^T W P_perp H_CH(s)

      is the information about the cylinder's own coefficients that survives
      after everything SH can explain has been projected away.  Sites are picked
      greedily by max det(DeltaF) (or max trace).

Four networks are then built on the SAME candidate pool and compared on the SAME
interior, using pt2's estimation machinery:

    FPS       farthest-point (pt2 baseline)   uses: geometry only
    SH-ERR    max eps(r)                      uses: geometry + truth field
    SH-SIG    max sigma_g(r)                  uses: geometry + P_SH
    FISHER    max det(DeltaF(s))              uses: geometry + both basis sets

INTERIOR MODEL (as in pt1/pt2)
------------------------------
The mass is in the CONSTANT-DENSITY POLYHEDRON scaled by beta_tilde = 1 - sum
beta; the mascons are the localized departures from homogeneity, beta_j = m_j/M*,
positive for an excess and negative for a deficit.  Everything estimated is a
contrast against that constant-density model, and there is no sum-beta = 1
pseudo-observation.  This matters most for criterion (1): with the bulk in the
truth field, eps(r) = |g_truth - g_SH,L| is the truncation error of the SHAPE
with the anomalies as a small perturbation on top — so it is even less of an
interior map than it was before.

FAIRNESS (this is the whole experiment — read this)
---------------------------------------------------
pt2 put an anomaly under every farthest-point cylinder, which would rig this
race.  Here the truth anomalies are drawn at RANDOM inside the body (rejection
sampling, fixed seed), independent of every strategy, deliberately including deep
ones and shallow ones under the long-axis tips where no cylinder is allowed to
go.  All four strategies choose from one common candidate pool and are subject to
the same minimum-separation constraint, so only the SCORE differs.

Metrics: per-anomaly mass-fraction sigma_beta, its geometric mean, and the
WORST-CASE sigma_beta (the coverage question), plus nonlinear position recovery.
The whole race is then repeated over several independent random interiors,
because a single draw cannot rank placement rules.

WHAT COMES OUT (Eros, L=6, 6 cylinders, 5 mascons)
--------------------------------------------------
1. Taken literally, all three criteria are near-perfect proxies for RADIUS
   (Spearman |rho| > 0.93 with log r): inside the Brillouin sphere the factor
   (R*/r)^n swamps everything, so each one just says "go to the lowest points of
   the shape", and raw eps and raw sigma_g select IDENTICAL sites.
2. Worse, with a degree-only (Kaula) covariance the addition theorem makes
   sigma_g EXACTLY a function of |r| — idea 1b then carries literally zero
   directional information.  It only becomes a map once P_SH has real structure
   (here: a Fisher inverse with a tracking coverage gap), and even then the
   radial trend has to be removed before the gap is what gets selected.
3. The raw SH-error criterion DOES beat farthest-point, and the win survives
   averaging: 7/8 independent interiors, median 1.56x on worst-case sigma_beta
   (FISHER close behind, 1.50x, 7/8).  The detrended criteria, which throw the
   radial signal away, do not.
4. But that win is not adaptivity.  With the constant-density bulk in the truth
   field, eps(r) is dominated by the SHAPE's degree>L truncation error — the
   Spearman correlation between the bulk-only and full eps maps is 0.98 — so the
   selected network barely moves between interiors (the script prints the
   fraction of shared sites).  What the criterion really says is "go to the
   lowest points, where the degree-L model is worst": a better GEOMETRIC rule,
   precomputable from the shape and the truncation degree with no interior
   knowledge at all.
=> Report it as an SH-truncation-error placement map, with farthest-point as the
   safe interior-agnostic default.  The script prints whichever verdict the
   numbers support, and the sentence to go with it.
   (Historical note: with the older mascons-in-vacuum truth — no bulk — eps(r)
   was an interior artefact and NOTHING beat farthest-point.  The parameterization
   is what changed the answer, which is itself worth knowing.)

Everything heavy (Legendre, Stokes design, Bessel basis, the constant-density
bulk, fits) is reused from `cylinder_mass_estimation_GLOBAL` (as G) and
`..._GLOBAL_pt2` (as P2); this file adds only the SH field synthesis (needed for
the error/uncertainty maps) and the placement criteria.

Units: Eros normalized (LU), total mass M* = 1, G = 1.
"""

from __future__ import annotations
import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import cylinder_mass_estimation_GLOBAL as G          # pt1 machinery
import cylinder_mass_estimation_GLOBAL_pt2 as P2     # pt2 network machinery

COLOR = G.COLOR
mpl.rcParams.update({"axes.prop_cycle": mpl.cycler(color=COLOR),
                     "figure.dpi": 110})
SEP = "=" * 78

# cylinder geometry — identical to pt2 so the networks are comparable
CYL_RADIUS, CYL_HEIGHT, CYL_ALPHA, CYL_LIFT = 0.12, 0.32, 100.0, 0.03

# Strategies whose score never touches the interior: geometry (FPS), the SH
# formal-uncertainty map (built from the tracking geometry alone) and the
# incremental Fisher scan (built from the two basis sets).  Their networks are
# IDENTICAL for every truth interior — which is what the robustness sweep
# exploits, and what the verdict must say out loud if one of them wins.
INTERIOR_INDEPENDENT = ("FPS (geometry)", "SH-SIG detrend", "FISHER logdet")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — SPHERICAL-HARMONIC FIELD SYNTHESIS
# (pt1 only builds Stokes coefficients FROM masses; to map the SH model error we
#  need the other direction: the field the truncated SH model predicts.)
# ═══════════════════════════════════════════════════════════════════════════


def legendre_bar(nmax: int, x):
    """
    Fully-normalized associated Legendre P̄_nm(x), vectorized over x.
    Same recursions as G.fully_normalized_legendre (checked in _selftest).
    Returns (nmax+1, nmax+1, N).
    """
    x = np.atleast_1d(np.asarray(x, float))
    P = np.zeros((nmax + 1, nmax + 1, x.size))
    P[0, 0] = 1.0
    if nmax == 0:
        return P
    sx = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    for m in range(1, nmax + 1):
        P[m, m] = np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * sx * P[m - 1, m - 1]
    for m in range(0, nmax):
        P[m + 1, m] = np.sqrt(2.0 * m + 3.0) * x * P[m, m]
    for m in range(0, nmax + 1):
        for n in range(m + 2, nmax + 1):
            a = np.sqrt(((2 * n + 1) * (2 * n - 1)) / ((n - m) * (n + m)))
            b = np.sqrt(((2 * n + 1) * (n + m - 1) * (n - m - 1))
                        / ((2 * n - 3) * (n - m) * (n + m)))
            P[n, m] = a * x * P[n - 1, m] - b * P[n - 2, m]
    return P


def sh_pot_basis(pts, Lmin, Lmax, Rref):
    """
    Exterior SH POTENTIAL basis at `pts`, in the SAME coefficient order as
    G.sh_stokes_of_point(p, Lmin, Lmax, Rref), so that

        U_L(r) = sh_pot_basis(r) @ ( Σ_j f_j · G.sh_stokes_of_point(p_j) ).

    The (2-delta_m0)/(2n+1) factor is the Legendre addition theorem written for
    pt1's normalization: G.fully_normalized_legendre seeds P̄_11 with
    sqrt(3/2) rather than the geodesy sqrt(3), i.e. its P̄_nm are the 4π-
    normalized ones divided by sqrt(2) for m>=1.  With this factor the sum is
    the exact multipole expansion of Σ_j f_j/|r-p_j| (verified in _selftest).
    Returns (N, n_coeff).
    """
    pts = np.atleast_2d(np.asarray(pts, float))
    r = np.linalg.norm(pts, axis=1)
    lam = np.arctan2(pts[:, 1], pts[:, 0])
    Pb = legendre_bar(Lmax, pts[:, 2] / r)
    cols = []
    for n in range(Lmin, Lmax + 1):
        rad = Rref ** n / r ** (n + 1) / (2.0 * n + 1.0)
        for m in range(0, n + 1):
            base = (1.0 if m == 0 else 2.0) * rad * Pb[n, m]
            cols.append(base * np.cos(m * lam))
            cols.append(base * np.sin(m * lam))
    return np.column_stack(cols)


def sh_grad_basis(pts, Lmin, Lmax, Rref, h=1e-4):
    """
    Gradient of each SH potential basis function, dU/dx: (N, 3, n_coeff).
    Central differences — the same +grad(U) sign convention used by
    G.point_mass_field and G.cyl_basis, so all three live in one measurement
    space [U; ax; ay; az].
    """
    pts = np.atleast_2d(np.asarray(pts, float))
    out = np.empty((len(pts), 3, sh_pot_basis(pts[:1], Lmin, Lmax, Rref).shape[1]))
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        out[:, k, :] = (sh_pot_basis(pts + d, Lmin, Lmax, Rref)
                        - sh_pot_basis(pts - d, Lmin, Lmax, Rref)) / (2 * h)
    return out


def sh_meas_basis(pts, Lmin, Lmax, Rref):
    """SH basis stacked as [U; ax; ay; az] → (4N, n_coeff): H_SH for a patch."""
    B = sh_pot_basis(pts, Lmin, Lmax, Rref)
    dB = sh_grad_basis(pts, Lmin, Lmax, Rref)
    return np.vstack([B, dB[:, 0, :], dB[:, 1, :], dB[:, 2, :]])


def stokes_vector(P, f, Lmin, Lmax, Rref, bulk=None):
    """
    Stokes coefficients of the interior model (the 'estimated' SH field).
    With `bulk` this is the full β̃·CD + Σ β_j pt_j; bulk=None degenerates to
    mascons in vacuum, which is only used by the self-test.
    """
    if bulk is None:
        return G.A_stokes(P, Lmin, Lmax, Rref) @ f
    return G.stokes_total(f, P, bulk, Lmin, Lmax, Rref)


def truth_field(pts, P, f, bulk=None):
    """
    Exact potential and acceleration of the interior model at pts → (U, g).
    With `bulk`, the constant-density polyhedron scaled by β̃ = 1 − Σβ is
    included — and it DOMINATES: the anomalies perturb it by a few per cent.
    That matters here, because eps = |g_truth − g_SH,L| then maps the SH
    truncation error of the SHAPE, not of a mascon cloud in vacuum.
    """
    pts = np.atleast_2d(np.asarray(pts, float))
    n = len(pts)
    U = np.zeros(n)
    g = np.zeros((n, 3))
    for m_j, p_j in zip(f, P):
        pm = m_j * G.point_mass_field(p_j, pts)
        U += pm[:n]
        g += pm[n:].reshape(3, n).T
    if bulk is not None:
        fb = G.bulk_fraction(f) * bulk.field(pts)
        U += fb[:n]
        g += fb[n:].reshape(3, n).T
    return U, g


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — PLACEMENT CRITERION MAPS
# ═══════════════════════════════════════════════════════════════════════════


def map_sh_error(pts, P, f, Lmax, Rref, bulk=None):
    """
    IDEA 1a — SH MODEL ERROR.  eps(r) = |g_truth(r) - g_SH,L(r)| : where the
    degree-L expansion fails to represent the real field.  In a mission this is
    fed by crossover / tracking residuals instead of a simulated truth.

    With the bulk included, both sides carry it — the truth exactly, the model
    through the bulk's own Stokes coefficients — so what is left is the genuine
    degree > L truncation error, which for a shape like Eros is mostly the
    SHAPE's, with the anomalies as a small perturbation on top.
    """
    c = stokes_vector(P, f, 0, Lmax, Rref, bulk=bulk)
    g_sh = sh_grad_basis(pts, 0, Lmax, Rref) @ c
    _, g_true = truth_field(pts, P, f, bulk=bulk)
    return np.linalg.norm(g_true - g_sh, axis=1)


def kaula_variances(Lmax, K=1e-2, n_known=1, known_sig=1e-6):
    """
    Per-coefficient variances for a Kaula rule sigma_n = K/n^2 applied to the
    standard 4π-normalized Stokes coefficients, EXPRESSED IN PT1'S CONVENTION
    (degrees <= n_known — GM and the centre of mass — taken as known).

    pt1's basis absorbs a (2-delta_m0)/(2n+1) factor (see sh_pot_basis), so an
    isotropic Kaula prior is NOT equal variance per column: it is
    sigma_n^2 (2n+1)^2/(2-delta_m0).  Only with this weighting do the degree sums
    close under the addition theorem — which is what makes the resulting
    sigma_g map exactly radial (asserted in _selftest).  Column order matches
    sh_pot_basis: for each n, for each m, the cos then the sin column.
    """
    var = []
    for n in range(0, Lmax + 1):
        s = known_sig if n <= n_known else K / n ** 2
        for m in range(0, n + 1):
            v = s ** 2 * (2 * n + 1) ** 2 / (1.0 if m == 0 else 2.0)
            var += [v, v]
    return np.asarray(var)


def kaula_covariance(Lmax, K=1e-2, n_known=1):
    """
    DIAGONAL, degree-only stand-in for P_SH.

    Careful — this one is a trap, and the script demonstrates it: if the
    covariance depends only on degree n, the Legendre addition theorem makes
    Σ_m |∇b_nm(r)|^2 rotationally invariant, so sigma_g(r) is EXACTLY a function
    of |r| alone.  The resulting "uncertainty map" carries no directional
    information whatsoever — it can only say "go to the low points of the shape".
    Verified in _selftest; use tracking_covariance() for a map that means
    something.
    """
    return np.diag(kaula_variances(Lmax, K=K, n_known=n_known))


def tracking_covariance(Lmax, Rref, R_obs=1.25, n_obs=800, gap_lon=(40.0, 150.0),
                        gap_keep=0.06, sig=3e-4, K=1e-2, seed=3):
    """
    A posterior covariance P_SH that actually has structure: the Fisher inverse
    of a synthetic tracking geometry with a COVERAGE GAP.  Potential+acceleration
    are sampled on a sphere of radius R_obs·R*, but only a fraction `gap_keep` of
    the samples survive inside the longitude sector `gap_lon` (an unobserved
    swath, as any real mission has).  A Kaula prior regularizes the inversion.

        P_SH = ( H^T H / sig^2  +  Prior^-1 )^-1 ,   H = [b; grad b]

    sigma_g built from this P_SH is anisotropic and peaks over the gap, which is
    what idea 1b is supposed to exploit.
    """
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(4 * n_obs, 3))
    p = R_obs * Rref * p / np.linalg.norm(p, axis=1)[:, None]
    lon = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360.0
    in_gap = (lon >= gap_lon[0]) & (lon <= gap_lon[1])
    keep = (~in_gap) | (rng.random(len(p)) < gap_keep)
    p = p[keep][:n_obs]
    H = sh_meas_basis(p, 0, Lmax, Rref)
    Fi = H.T @ H / sig ** 2 + np.diag(1.0 / kaula_variances(Lmax, K=K))
    return np.linalg.inv(Fi), p


def map_sh_sigma(pts, Lmax, Rref, P_sh):
    """
    IDEA 1b — SH FORMAL UNCERTAINTY.  sigma_g(r)^2 = tr[ H_g P_SH H_g^T ] with
    H_g = dg/dc: where the estimated SH field is least trustworthy.  The map is
    interior-agnostic by construction (it knows the data geometry, not the
    masses); with a degree-only P_SH it degenerates to a function of |r| alone,
    so pass a covariance with real structure (see tracking_covariance).
    """
    H = sh_grad_basis(pts, 0, Lmax, Rref)          # (N,3,n_c)
    return np.sqrt(np.einsum("nic,cd,nid->n", H, P_sh, H))


def detrend_radial(score, pts):
    """
    Remove the trivial radial trend from a criterion map.

    Both idea-1 maps carry the factor (R*/r)^n, so raw eps and sigma_g are
    dominated by upward continuation: they simply rank the sites by how deep
    inside the Brillouin sphere they sit, and end up selecting the SAME points
    (the low waist) whatever the interior is.  Fitting log(score) = a + b·log(r)
    and returning the residual leaves "where is SH bad FOR THIS RADIUS", which is
    the interior-driven part the criterion was supposed to find.
    """
    r = np.log(np.linalg.norm(pts, axis=1))
    y = np.log(np.maximum(score, 1e-300))
    b, a = np.polyfit(r, y, 1)
    return y - (a + b * r)


def delta_fisher(site, tm, V, F, ch_modes, Lmax, Rref, n_pts, seed=1,
                 normalize=True, delta=1e-6):
    """
    IDEA 2 — INCREMENTAL FISHER INFORMATION of a candidate cylinder at `site`.

        M          = P_perp H_CH  =  H_CH - H_SH (H_SH^+ H_CH)      (stable form)
        DeltaF     = M^T W M                       (W = I/sigma^2, dropped: the
                                                    ranking is scale-free)

    With `normalize` the CH columns are scaled to unit norm first, so
    diag(DeltaF)_k = ||P_perp phi_k||^2 / ||phi_k||^2 is the FRACTION of CH mode
    k that spherical harmonics cannot reproduce here (0..1) — comparable across
    sites regardless of how strong the basis happens to be.  Scores returned:
        logdet = Σ log(lambda_i + delta)   D-optimality (delta = weak prior)
        trace  = mean orthogonal fraction  A-optimality, interpretable in [0,1]
    """
    cyl, obs = cylinder_at(site, tm, V, F, n_pts=n_pts, seed=seed)
    if len(obs) < 20:
        return dict(logdet=-np.inf, trace=0.0, n_obs=len(obs))
    H_ch = G.cyl_basis(cyl, obs, *ch_modes)                 # (4N, n_ch)
    H_sh = sh_meas_basis(obs, 0, Lmax, Rref)                # (4N, n_c)
    # SAME trap as the CH pseudo-inverse (see G.CH_RCOND): H_SH over a single
    # patch has cond ~ 1e23, and the default machine-epsilon cutoff keeps 49/56
    # directions down to sigma ~ 1e-21 of the largest.  Projecting through those
    # is projecting through noise, so P_perp — and every DeltaF score built on
    # it — inherits it.  Truncate at the same relative level (33/56 survive).
    coef, *_ = np.linalg.lstsq(H_sh, H_ch, rcond=G.CH_RCOND)
    M = H_ch - H_sh @ coef                                  # P_perp H_CH
    if normalize:
        nrm = np.linalg.norm(H_ch, axis=0)
        nrm[nrm == 0] = 1.0
        M = M / nrm
    dF = M.T @ M
    lam = np.linalg.eigvalsh(0.5 * (dF + dF.T))
    lam = np.clip(lam, 0.0, None)
    return dict(logdet=float(np.sum(np.log(lam + delta))),
                trace=float(np.trace(dF) / dF.shape[0]), n_obs=len(obs))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — CANDIDATE POOL, SELECTION RULES, NETWORK BUILDER
# ═══════════════════════════════════════════════════════════════════════════


def cylinder_at(site, tm, V, F, n_pts=180, seed=2):
    """pt2's cylinder geometry at one surface site: axis = outward radial."""
    d = site / np.linalg.norm(site)
    cyl = G.Cylinder(center=site + CYL_LIFT * d, radius=CYL_RADIUS,
                     height=CYL_HEIGHT, alpha=CYL_ALPHA, R=P2.rot_z_to(d))
    obs = G.cylinder_points(cyl, n=n_pts, seed=seed + 1)
    return cyl, obs[~G.inside_body(tm, V, F, obs)]          # vacuum only


def candidate_pool(V, F, tm, Rb, n_cand=140, brillouin_frac=0.80,
                   min_vacuum=0.40, n_pts=180, seed=1):
    """
    Common candidate sites for ALL strategies: the inside-Brillouin surface
    (sides / waist — the tips sit ON R* where exterior SH does not diverge),
    thinned by farthest-point sampling for even coverage, then filtered to sites
    whose cylinder is mostly outside the body (a cylinder buried in a concavity
    is a bad cylinder for every strategy, so it is removed from everyone's pool).
    """
    surf = V[np.linalg.norm(V, axis=1) < brillouin_frac * Rb]
    idx = P2.farthest_point_sample(surf, n_cand,
                                   start_idx=int(np.argmin(surf[:, 2])), seed=seed)
    cand = surf[idx]
    frac = np.array([len(cylinder_at(s, tm, V, F, n_pts=n_pts)[1]) / n_pts
                     for s in cand])
    return cand[frac >= min_vacuum], surf


def greedy_by_score(cand, score, k, min_sep):
    """
    Greedy max-score selection with a minimum-separation constraint (so a smart
    criterion cannot cheat by stacking every cylinder on one hot spot).  If the
    constraint starves the selection it is relaxed by 20% and retried.
    """
    sep = float(min_sep)
    while True:
        chosen, s = [], np.array(score, float).copy()
        while len(chosen) < k and np.isfinite(s).any():
            j = int(np.nanargmax(s))
            chosen.append(j)
            s[np.linalg.norm(cand - cand[j], axis=1) < sep] = -np.inf
        if len(chosen) == k or sep < 1e-3:
            return np.asarray(chosen)
        sep *= 0.8


def network_at(sites, tm, V, F, n_pts=180, seed=2):
    """pt2-compatible network dicts {cyl, obs, dir, surf} at given sites."""
    net = []
    for s in sites:
        cyl, obs = cylinder_at(s, tm, V, F, n_pts=n_pts, seed=seed)
        net.append(dict(cyl=cyl, obs=obs, dir=s / np.linalg.norm(s), surf=s))
    return net


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRUTH INTERIOR, DRAWN INDEPENDENTLY OF EVERY STRATEGY
# ═══════════════════════════════════════════════════════════════════════════


def place_mascons_random(V, tm, n_shallow=4, n_deep=1, seed=5, min_sep=0.20,
                         shallow_max=0.09, deep_min=0.17, n_try=6000):
    """
    Rejection-sample ANOMALIES uniformly inside the body: `n_shallow` shallow
    ones (depth < shallow_max, anywhere on the body — including under the
    long-axis tips, where the candidate pool has no site) and `n_deep` deep ones
    (the case no near-surface patch can see).  Truth mass fractions
    β_j = m_j/M* are a few per cent with mixed signs (over- and under-dense);
    they do NOT sum to one — the remaining β̃ = 1 − Σβ stays in the
    constant-density polyhedron.  Depth = distance to the nearest surface vertex.
    """
    rng = np.random.default_rng(seed)
    lo, hi = V.min(0), V.max(0)
    pts = rng.uniform(lo, hi, (n_try, 3))
    pts = pts[G.inside_body(tm, V, None, pts)]
    depth = cKDTree(V).query(pts)[0]

    def pick(pool, n, taken):
        out = []
        for p in pool[rng.permutation(len(pool))]:
            if all(np.linalg.norm(p - q) > min_sep for q in taken + out):
                out.append(p)
            if len(out) == n:
                break
        return out

    taken = pick(pts[depth < shallow_max], n_shallow, [])
    taken += pick(pts[depth > deep_min], n_deep, taken)
    P = np.array(taken)
    dep = cKDTree(V).query(P)[0]
    order = np.argsort(dep)                       # shallowest first
    P, dep = P[order], dep[order]
    f = rng.uniform(0.015, 0.05, len(P)) * rng.choice([-1.0, 1.0], len(P))
    names = [f"m{i} (d={d:.2f})" for i, d in enumerate(dep)]
    return names, P, f, dep


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════


def run(Lmax_sh=6, eps=0.02, ch_modes=(6, 6), n_cyl=6, n_cand=140,
        n_pts=180, n_mc=300, n_mc_pos=30, kaula_K=1e-2, gap_lon=(40.0, 150.0),
        do_position=True, n_sweep=8, outdir="Images", verbose=True):
    t0 = time.time()
    V, F, tm, Rb = G.load_eros()
    Rref = Rb

    # ── truth interior — independent of every placement strategy ────────────
    #    (anomalies on top of the constant-density bulk, which keeps β̃ = 1 − Σβ)
    names, P, f_true, depth = place_mascons_random(V, tm)
    bulk = G.Bulk(V, F)
    beta_bulk = G.bulk_fraction(f_true)
    n_m = len(P)

    # ── common candidate pool ───────────────────────────────────────────────
    cand, surf = candidate_pool(V, F, tm, Rb, n_cand=n_cand, n_pts=n_pts)
    lift = cand + CYL_LIFT * cand / np.linalg.norm(cand, axis=1)[:, None]

    if verbose:
        print(SEP)
        print("  PART 3 — ADAPTIVE vs GEOMETRIC placement of the CH network (Eros)")
        print(SEP)
        print(f"  Brillouin R* = {Rb:.3f} LU | candidate sites {len(cand)}/{n_cand} "
              f"({n_cand-len(cand)} dropped as too buried) | {n_cyl} cylinders "
              f"| {n_m} anomalies")
        print(f"  BULK: constant-density polyhedron, β̃ = 1 − Σβ = {beta_bulk:.3f} of M*")
        for nm, p, fr in zip(names, P, f_true):
            print(f"    {nm:14s} p={np.round(p,3)}  β={fr:+.3f}  "
                  f"|r|={np.linalg.norm(p):.2f}")

    # ── the criterion maps, on the candidate pool ───────────────────────────
    P_kaula = kaula_covariance(Lmax_sh, K=kaula_K)
    P_track, obs_track = tracking_covariance(Lmax_sh, Rref, gap_lon=gap_lon,
                                             K=kaula_K)
    err_raw = map_sh_error(lift, P, f_true, Lmax_sh, Rref, bulk=bulk)
    err_det = detrend_radial(err_raw, lift)
    sig_kau = map_sh_sigma(lift, Lmax_sh, Rref, P_kaula)   # diagnostic (radial)
    sig_trk = map_sh_sigma(lift, Lmax_sh, Rref, P_track)
    sig_det = detrend_radial(sig_trk, lift)                # the gap-seeking part
    fish = [delta_fisher(s, tm, V, F, ch_modes, Lmax_sh, Rref, n_pts) for s in cand]
    sc_fis = np.array([d["logdet"] for d in fish])
    sc_tr = np.array([d["trace"] for d in fish])

    # ── the five networks (same pool, same separation rule) ─────────────────
    fps_idx = P2.farthest_point_sample(cand, n_cyl,
                                       start_idx=int(np.argmin(cand[:, 2])))
    d_fps = min(np.linalg.norm(cand[i] - cand[j])
                for i in fps_idx for j in fps_idx if i != j)
    min_sep = 0.5 * d_fps
    #  raw sigma_g is NOT raced: being ~purely radial it selects exactly the same
    #  sites as raw eps (asserted below), so it would be a duplicate column.
    picks = {
        "FPS (geometry)": fps_idx,
        "SH-ERR raw": greedy_by_score(cand, err_raw, n_cyl, min_sep),
        "SH-ERR detrend": greedy_by_score(cand, err_det, n_cyl, min_sep),
        "SH-SIG detrend": greedy_by_score(cand, sig_det, n_cyl, min_sep),
        "FISHER logdet": greedy_by_score(cand, sc_fis, n_cyl, min_sep),
    }
    picks_sig_raw = greedy_by_score(cand, sig_trk, n_cyl, min_sep)
    nets = {k: network_at(cand[i], tm, V, F, n_pts=n_pts) for k, i in picks.items()}

    if verbose:
        rlog = np.log(np.linalg.norm(lift, axis=1))
        print(f"\n{'-'*78}\n  PLACEMENT CRITERIA (min separation {min_sep:.3f} LU "
              f"= 0.5 × farthest-point spacing)\n{'-'*78}")
        print(f"  eps |g-g_SH|            : {err_raw.min():.2e} … {err_raw.max():.2e} "
              f"({err_raw.max()/err_raw.min():.0f}× spread)")
        print(f"  sigma_g (tracking P_SH) : {sig_trk.min():.2e} … {sig_trk.max():.2e} "
              f"({sig_trk.max()/sig_trk.min():.0f}× spread, "
              f"coverage gap at lon {gap_lon[0]:.0f}–{gap_lon[1]:.0f}°)")
        print(f"  DeltaF orthogonal frac. : {sc_tr.min():.3f} … {sc_tr.max():.3f}  "
              f"(1 = CH mode invisible to SH)")
        print("\n  How much is each criterion just a proxy for radius?  "
              "(Spearman rank corr. with log r)")
        for nm, s in (("eps raw", err_raw), ("eps detrended", err_det),
                      ("sigma_g Kaula", sig_kau), ("sigma_g tracking", sig_trk),
                      ("sigma_g tr. detr.", sig_det), ("logdet DeltaF", sc_fis)):
            print(f"    {nm:20s} rho = {spearmanr(s, rlog).statistic:+.3f}")
        print("    ⇒ EVERY criterion taken literally is a near-perfect proxy for "
              "radius: they all\n      reduce to 'go to the lowest points of the "
              "shape'.  With a degree-only (Kaula)\n      covariance sigma_g is "
              "EXACTLY radial (rho = -1, proved in _selftest) — it has no\n      "
              "directional information at all.  Detrending is what makes idea 1 "
              "informative.")
        same_raw = set(picks_sig_raw.tolist()) == set(picks["SH-ERR raw"].tolist())
        print(f"    raw sigma_g picks the SAME {n_cyl} sites as raw eps: {same_raw} "
              f"(rho = {spearmanr(err_raw, sig_trk).statistic:+.3f}) → not raced "
              f"separately")
        in_gap = ((np.degrees(np.arctan2(cand[:, 1], cand[:, 0])) % 360)[
            picks["SH-SIG detrend"]] >= gap_lon[0]) & (
            (np.degrees(np.arctan2(cand[:, 1], cand[:, 0])) % 360)[
                picks["SH-SIG detrend"]] <= gap_lon[1])
        print(f"    detrended sigma_g puts {in_gap.sum()}/{n_cyl} cylinders inside "
              f"the tracking coverage gap")
        for k, i in picks.items():
            same = len(set(i.tolist()) & set(fps_idx.tolist()))
            d = [min(np.linalg.norm(cand[a] - cand[b]) for b in i if b != a) for a in i]
            print(f"    {k:16s} sites {np.sort(i)}  ({same}/{n_cyl} shared with FPS, "
                  f"mean NN spacing {np.mean(d):.3f} LU)")

    # ── observables (pt1/pt2 rule: OD-like per-coefficient weights, σ_i =
    #    eps·|coeff_i| above a noise floor, on the FULL measured coefficients;
    #    every design matrix is a contrast against the bulk, and there is no
    #    Σβ = 1 row — the mass budget is structural) ───────────────────────────
    A_sh = G.A_stokes_contrast(P, bulk, 2, Lmax_sh, Rref)
    sig_sh = G.od_sigma(G.stokes_total(f_true, P, bulk, 2, Lmax_sh, Rref), eps)
    base = [(A_sh, sig_sh)]

    cases = {"SH only": base}
    sig_ch = {}
    for k, net in nets.items():
        blocks = P2.ch_blocks_for(P, net, ch_modes, eps, f_true, bulk)
        sig_ch[k] = [s for _, s in blocks]
        cases[k] = base + blocks

    # ── A) MASS-RATIO recovery ──────────────────────────────────────────────
    # β̃ = 1 − Σβ is estimated too, as a derived quantity on every MC draw
    sig_by_case, bulk_by_case = {}, {}
    for k, b in cases.items():
        mc = G.monte_carlo_fit(b, f_true, n_mc=n_mc)
        sig_by_case[k] = mc.std(0)
        bt = 1.0 - mc.sum(1)
        bulk_by_case[k] = (float(bt.mean()), float(bt.std()))
    if verbose:
        print(f"\n{'-'*78}\n  A) MASS-FRACTION sigma_beta  ({n_mc} MC least-squares "
              f"fits)\n{'-'*78}")
        print(f"  {'anomaly':14s} " + " ".join(f"{k:>16s}" for k in cases))
        for i, nm in enumerate(names):
            print(f"  {nm:14s} " + " ".join(f"{sig_by_case[k][i]:16.2e}" for k in cases))
        print(f"  {'-'*76}")
        print(f"  {'BODY β̃':14s} " +
              " ".join(f"{bulk_by_case[k][1]:16.2e}" for k in cases))
        print(f"    (truth β̃ = {beta_bulk:.4f};  recovered "
              f"{bulk_by_case['FPS (geometry)'][0]:.4f} ± "
              f"{bulk_by_case['FPS (geometry)'][1]:.4f} with the FPS network)")
        print(f"  {'geom. mean':14s} " +
              " ".join(f"{np.exp(np.mean(np.log(sig_by_case[k]))):16.2e}" for k in cases))
        print(f"  {'WORST case':14s} " +
              " ".join(f"{sig_by_case[k].max():16.2e}" for k in cases))

    # ── B) POSITION recovery ────────────────────────────────────────────────
    pos_rms = {}
    if do_position:
        if verbose:
            print(f"\n{'-'*78}\n  B) POSITION RMS error [LU]  ({n_mc_pos} nonlinear MC "
                  f"fits per mascon)\n{'-'*78}")
        pos_bounds = (V.min(0) - 0.05, V.max(0) + 0.05)
        any_net = nets["FPS (geometry)"]
        for k in cases:
            use = k != "SH only"
            net = nets[k] if use else any_net
            sig_list = sig_ch[k] if use else sig_ch["FPS (geometry)"]
            rms = np.zeros(n_m)
            for j in range(n_m):
                cloud = P2.position_mc_net(j, P, f_true, net, ch_modes, Lmax_sh,
                                           Rref, sig_sh, sig_list, use, bulk,
                                           bounds=pos_bounds, n_mc=n_mc_pos)
                rms[j] = np.sqrt(np.mean(np.sum((cloud - P[j]) ** 2, axis=1)))
            pos_rms[k] = rms
        if verbose:
            print(f"  {'anomaly':14s} " + " ".join(f"{k:>16s}" for k in cases))
            for i, nm in enumerate(names):
                print(f"  {nm:14s} " + " ".join(f"{pos_rms[k][i]:16.2e}" for k in cases))
            print(f"  {'-'*76}")
            print(f"  {'WORST case':14s} " +
                  " ".join(f"{pos_rms[k].max():16.2e}" for k in cases))

    res = dict(V=V, F=F, tm=tm, Rb=Rb, surf=surf, cand=cand, lift=lift, picks=picks,
               nets=nets, names=names, P=P, f_true=f_true, depth=depth,
               bulk=bulk, beta_bulk=beta_bulk, bulk_by_case=bulk_by_case,
               err_raw=err_raw, err_det=err_det, sig_kau=sig_kau,
               sig_trk=sig_trk, sig_det=sig_det, obs_track=obs_track,
               gap_lon=gap_lon, sc_fis=sc_fis, sc_tr=sc_tr,
               cases=list(cases.keys()), sig_by_case=sig_by_case,
               pos_rms=pos_rms, n_cyl=n_cyl, min_sep=min_sep, Lmax_sh=Lmax_sh)
    sweep, stab = None, None
    if n_sweep > 1:
        sweep, stab = robustness_sweep(res, eps=eps, ch_modes=ch_modes,
                                       Lmax_sh=Lmax_sh, n_pts=n_pts, n_mc=n_mc,
                                       seeds=tuple(21 + i for i in range(n_sweep)),
                                       verbose=verbose)
        res["sweep"], res["stability"] = sweep, stab
    make_plots(res, outdir=outdir)
    if verbose:
        verdict(res, sweep=sweep, stability=stab)
        print(f"\n  [{time.time()-t0:.1f} s]")
    return res


def robustness_sweep(res, eps=0.02, ch_modes=(6, 6), Lmax_sh=6,
                     n_pts=180, n_mc=300, seeds=(21, 22, 23, 24, 25, 26, 27, 28),
                     verbose=True):
    """
    Repeat the mass-fraction race over several INDEPENDENT random interiors.

    A single draw cannot settle "is adaptive placement worth it?" — the winner
    could be an accident of where the anomalies happened to land.  Only the two
    eps-based criteria depend on the interior, so the candidate pool, the Fisher
    scan, the sigma_g map and their three networks are computed once and reused;
    per seed only the eps maps, their networks and the fits are redone.  (With
    the constant-density bulk in the truth field, even those two barely move: the
    SH truncation error is now dominated by the SHAPE, and the anomalies only
    perturb it — one more reason not to expect an interior-adaptive win.)
    Returns {strategy: array of worst-case sigma_beta ratios vs FPS}.
    """
    V, F, tm, Rb = res["V"], res["F"], res["tm"], res["Rb"]
    bulk = res["bulk"]
    cand, lift, n_cyl, min_sep = res["cand"], res["lift"], res["n_cyl"], res["min_sep"]
    fixed = {k: res["picks"][k] for k in INTERIOR_INDEPENDENT}
    fixed_nets = {k: res["nets"][k] for k in fixed}
    ratios = {k: [] for k in res["cases"]}
    sites = {"SH-ERR raw": [], "SH-ERR detrend": []}   # do the picks even move?
    if verbose:
        print(f"\n{SEP}\n  ROBUSTNESS — {len(seeds)} independent random interiors\n{SEP}")
    for sd in seeds:
        _, P, f_true, _ = place_mascons_random(V, tm, seed=sd)
        err_raw = map_sh_error(lift, P, f_true, Lmax_sh, Rb, bulk=bulk)
        nets = dict(fixed_nets)
        for key, score in (("SH-ERR raw", err_raw),
                           ("SH-ERR detrend", detrend_radial(err_raw, lift))):
            idx = greedy_by_score(cand, score, n_cyl, min_sep)
            sites[key].append(set(idx.tolist()))
            nets[key] = network_at(cand[idx], tm, V, F, n_pts=n_pts)
        A_sh = G.A_stokes_contrast(P, bulk, 2, Lmax_sh, Rb)
        sig_sh = G.od_sigma(G.stokes_total(f_true, P, bulk, 2, Lmax_sh, Rb), eps)
        base = [(A_sh, sig_sh)]
        wc = {}
        for k in ratios:
            blocks = base if k == "SH only" else \
                base + P2.ch_blocks_for(P, nets[k], ch_modes, eps, f_true, bulk)
            wc[k] = G.monte_carlo_fit(blocks, f_true, n_mc=n_mc).std(0).max()
        for k in ratios:
            ratios[k].append(wc["FPS (geometry)"] / wc[k])
    ratios = {k: np.asarray(v) for k, v in ratios.items()}
    # site stability: fraction of each draw's cylinders shared with the first
    # draw's.  Near 1 means the "adaptive" criterion is not actually adapting.
    stab = {k: 1.0 for k in fixed}          # reused unchanged for every interior
    stab.update({k: float(np.mean([len(a & v[0]) / n_cyl for a in v]))
                 for k, v in sites.items() if v})
    if verbose:
        print("  worst-case sigma_beta improvement over FPS, across interiors")
        print(f"  {'strategy':18s} {'median':>9s} {'min':>9s} {'max':>9s} "
              f"{'# draws better':>16s}")
        for k, v in ratios.items():
            print(f"  {k:18s} {np.median(v):8.2f}× {v.min():8.2f}× {v.max():8.2f}× "
                  f"{int((v > 1).sum()):11d}/{len(v)}")
        print("\n  site stability (fraction of cylinders shared with the first "
              "interior's pick):")
        for k in res["cases"]:
            if k in stab:
                tag = ("   (fixed by construction: the score never sees the interior)"
                       if k in INTERIOR_INDEPENDENT else "")
                print(f"    {k:18s} {stab[k]:.2f}{tag}")
        print("    ⇒ three of the four criteria never look at the interior at all, and "
              "the eps-based\n      ones barely move either: with the constant-density "
              "bulk in the truth field eps(r)\n      is dominated by the SHAPE's "
              "degree>L truncation error.  Any win here is a\n      better GEOMETRIC "
              "rule, not adaptivity.")
    return ratios, stab


def verdict(res, sweep=None, stability=None):
    """Turn the numbers into the sentence the paper actually needs."""
    sig, cases = res["sig_by_case"], res["cases"]
    ref = "FPS (geometry)"
    print(f"\n{SEP}\n  VERDICT — is adaptive placement worth a second optimizer?\n{SEP}")
    print(f"  {'strategy':18s} {'geo-mean sig_b':>15s} {'worst sig_b':>13s} "
          f"{'vs FPS (worst)':>16s}")
    for k in cases:
        gm = np.exp(np.mean(np.log(sig[k])))
        wc = sig[k].max()
        rel = sig[ref].max() / wc
        print(f"  {k:18s} {gm:15.2e} {wc:13.2e} {rel:15.2f}×")
    single = min((k for k in cases if k != "SH only"), key=lambda k: sig[k].max())
    print(f"\n  On THIS interior the best network is {single} "
          f"({sig[ref].max()/sig[single].max():.2f}× vs farthest-point).")

    if sweep is not None:
        cand_k = [k for k in cases if k not in ("SH only", ref)]
        best = max(cand_k, key=lambda k: np.median(sweep[k]))
        gain = float(np.median(sweep[best]))
        n_dr = len(sweep[best])
        nb = int((sweep[best] > 1).sum())
        survives = gain >= 1.2 and nb >= 0.75 * n_dr
        moves = None if stability is None else stability.get(best)
        print(f"  Over {n_dr} independent interiors the best criterion is {best} at a"
              f"\n  MEDIAN {gain:.2f}× (winning {nb}/{n_dr} draws)" +
              ("." if survives else
               " — and the ratios straddle 1, so the\n  ranking flips from one interior "
               "to the next."))
        fixed_by_constr = best in INTERIOR_INDEPENDENT
        if not survives:
            print("\n  → CONCLUSION.  None of the SH-aware criteria beats farthest-point\n"
                  "    sampling once you average over interiors: single-draw wins are\n"
                  "    noise, not signal.  Two reasons, both visible above:\n"
                  "      • taken literally, every criterion is a near-perfect proxy for\n"
                  "        radius (|rho| > 0.93), so it stacks cylinders on the low waist\n"
                  "        and loses the coverage that makes a network work;\n"
                  "      • what a network needs is to be NEAR EVERY anomaly, and with the\n"
                  "        anomalies unknown, even spacing is the optimal hedge.\n"
                  "    Farthest-point sampling is therefore the right choice for the\n"
                  "    paper — geometric, deterministic, reproducible — and adaptive\n"
                  "    placement belongs in one forward-looking sentence.")
        else:
            print(f"\n  → CONCLUSION.  The win SURVIVES averaging: {best} beats\n"
                  f"    farthest-point on {nb}/{n_dr} independent interiors.  But read what\n"
                  f"    it actually is before calling it adaptive placement:")
            if fixed_by_constr:
                print("      • its score never touches the interior — it is built from "
                      "the\n        tracking geometry and the two basis sets alone, so it "
                      "selects the\n        SAME six sites for every truth interior;")
            elif moves is not None:
                print(f"      • its network barely moves between interiors "
                      f"({moves:.0%} of the\n        cylinders are shared with the first "
                      f"draw's pick), because with the\n        constant-density bulk in "
                      f"the truth field eps(r) is dominated by the\n        SHAPE's "
                      f"degree>L truncation error, not by the anomalies;")
            print("      • and every criterion here correlates with radius at "
                  "|rho| > 0.93, i.e.\n        it says 'go to the lowest points, where the "
                  "degree-L model is worst'.\n"
                  "    So this is a better GEOMETRIC rule — precomputable from the shape,\n"
                  "    the tracking geometry and the SH degree, with no interior knowledge\n"
                  "    — rather than a criterion that adapts to the interior.  Report it as\n"
                  "    such, with farthest-point as the safe default.")
    else:
        print("  (run with n_sweep > 1: a single interior cannot rank placement rules)")
    if sweep is not None and gain >= 1.2 and nb >= 0.75 * n_dr:
        print("\n  Suggested sentence:\n"
              '    "Cylinder sites are chosen by farthest-point sampling of the\n'
              '     inside-Brillouin surface.  Selecting instead the sites at which the\n'
              '     degree-L spherical-harmonic model departs most from the\n'
              '     constant-density field tightens the worst-case mass-fraction\n'
              '     uncertainty by a further factor of roughly %.1f; since that map is\n'
              '     fixed by the shape, the tracking geometry and the truncation degree,\n'
              '     knowledge of the interior."' % gain)
    else:
        print("\n  Suggested sentence:\n"
              '    "More sophisticated placement strategies could exploit the SH solution,\n'
              '     for example by selecting cylinders in regions of large SH uncertainty or\n'
              '     by maximizing the incremental Fisher information provided by candidate\n'
              '     cylinders. Such adaptive placement is beyond the scope of this work."')


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════


_SHORT = {"SH only": "SH", "FPS (geometry)": "FPS", "SH-ERR raw": "ERR\nraw",
          "SH-ERR detrend": "ERR\ndetr", "SH-SIG detrend": "SIG\ndetr",
          "FISHER logdet": "FISHER"}


def _lonlat(p):
    r = np.linalg.norm(p, axis=1)
    return np.degrees(np.arctan2(p[:, 1], p[:, 0])), np.degrees(np.arcsin(p[:, 2] / r))


def make_plots(res, outdir="Images"):
    os.makedirs(outdir, exist_ok=True)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    V, F, P, cand, picks = res["V"], res["F"], res["P"], res["cand"], res["picks"]
    names = res["names"]
    fps = picks["FPS (geometry)"]

    # ---- FIG 1: criterion maps + what each strategy picks ------------------
    fig = plt.figure(figsize=(18.5, 9.5))
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    step = max(1, len(F) // 9000)
    ax.add_collection3d(Poly3DCollection(V[F[::step]], alpha=0.10,
                        facecolor="#9ecae1", edgecolor="0.6", linewidths=0.1))
    for c in res["nets"]["FPS (geometry)"]:
        G.draw_cylinder(ax, c["cyl"])
    ft = res["f_true"]
    for nm, p, b in zip(names, P, ft):   # colour by SIGN, label with β_j
        ax.scatter(p[0], p[1], p[2], s=80, depthshade=False, edgecolor="k",
                   color=G.COLOR[0] if b > 0 else G.COLOR[2])
        # name only: this panel is one of six, and the β values are in the
        # printed table — full labels collide at this size
        ax.text(p[0], p[1], p[2], f"  {nm.split()[0]}", fontsize=7.5)
    ax.plot([], [], color="crimson", lw=2, label="FPS network")
    ax.scatter([], [], color=G.COLOR[0], label=r"anomaly $\beta_j>0$")
    ax.scatter([], [], color=G.COLOR[2], label=r"anomaly $\beta_j<0$")
    ax.set_xlabel("x [LU]"); ax.set_ylabel("y [LU]"); ax.set_zlabel("z [LU]")
    G.set_axes_true_shape(ax, np.vstack(
        [V] + [G.cylinder_hull(c["cyl"]) for c in res["nets"]["FPS (geometry)"]]))
    ax.set_title("Constant-density BODY "
                 f"($\\tilde\\beta$ = {res['beta_bulk']:.3f})\n"
                 f"+ {len(P)} random anomalies ($\\tilde\\beta+\\sum\\beta_j=1$)")
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    ax.legend(fontsize=8, loc="upper left")

    lon, lat = _lonlat(cand)
    panels = [
        (2, np.log10(res["err_raw"]), "SH-ERR raw",
         r"$\log_{10}\epsilon=|g-g_{SH}|$  (idea 1a)"
         "\ndominated by upward continuation $(R^*/r)^n$"),
        (3, res["err_det"], "SH-ERR detrend",
         r"$\epsilon$ with the radial trend removed"
         "\nwhere SH is bad FOR ITS RADIUS → the interior"),
        (4, np.log10(res["sig_kau"]), None,
         r"$\log_{10}\sigma_g$, degree-only (Kaula) $P_{SH}$"
         "\nEXACTLY radial: no directional information"),
        (5, res["sig_det"], "SH-SIG detrend",
         r"detrended $\sigma_g$, tracking $P_{SH}$  (idea 1b)"
         "\nanisotropic: peaks over the coverage gap"),
        (6, res["sc_fis"], "FISHER logdet",
         r"$\log\det\Delta F$  (idea 2)"
         "\nwhat a cylinder sees and SH cannot"),
    ]
    mlon, mlat = _lonlat(P)
    for pos, score, key, title in panels:
        a = fig.add_subplot(2, 3, pos)
        sc = a.scatter(lon, lat, c=score, s=42, cmap="viridis")
        a.scatter(mlon, mlat, marker="x", s=70, color="crimson", lw=2,
                  label="mascon (sub-point)")
        a.scatter(lon[fps], lat[fps], s=230, marker="s", facecolor="none",
                  edgecolor="k", lw=1.1, label="FPS picks (baseline)")
        if key is not None:
            a.scatter(lon[picks[key]], lat[picks[key]], s=150, facecolor="none",
                      edgecolor="k", lw=2.4, label="this criterion's picks")
        if pos == 5:  # show where the tracking data are missing
            for g in res["gap_lon"]:
                a.axvline(((g + 180) % 360) - 180, color="crimson", ls="--", lw=1.2)
        plt.colorbar(sc, ax=a, label="score")
        a.set_xlabel("longitude [deg]"); a.set_ylabel("latitude [deg]")
        a.set_title(title, fontsize=10)
        a.legend(fontsize=7, loc="lower left", framealpha=0.9)
    fig.suptitle("Where should the cylinders go?  SH-aware criterion maps on the "
                 "common candidate pool", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt3_fig1_criteria.pdf"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 2: mass-fraction recovery, all strategies ---------------------
    cases, sig = res["cases"], res["sig_by_case"]
    has_sweep = "sweep" in res
    ncol = 3 if has_sweep else 2
    fig, axes = plt.subplots(1, ncol, figsize=(19 if has_sweep else 15.5, 5.8),
                             gridspec_kw={"width_ratios": [2.4, 1, 1.3][:ncol]})
    a1, a2 = axes[0], axes[1]
    nb = res["bulk_by_case"]
    vals = {k: np.append(sig[k], nb[k][1]) for k in cases}
    x = np.arange(len(P) + 1)
    w = 0.8 / len(cases)
    cols = ["0.55"] + COLOR[:len(cases) - 1]
    for i, (k, col) in enumerate(zip(cases, cols)):
        a1.bar(x + (i - (len(cases) - 1) / 2) * w, vals[k], w, color=col,
               edgecolor="k", lw=0.5, label=k)
    a1.axvline(len(P) - 0.5, color="0.6", ls="--", lw=1.2)
    a1.set_yscale("log"); a1.set_xticks(x)
    a1.set_xticklabels([n.replace(" ", "\n") for n in names]
                       + ["BODY\n$\\tilde\\beta$"], fontsize=8)
    a1.set_ylabel(r"mass-fraction 1$\sigma$ uncertainty $\sigma_\beta$")
    a1.set_title("Mass-fraction recovery per anomaly (sorted shallow to deep)\n"
                 r"plus the derived body fraction $\tilde\beta=1-\sum\beta_j$")
    a1.grid(True, axis="y", which="both", alpha=0.3); a1.legend(fontsize=8)

    gm = [np.exp(np.mean(np.log(sig[k]))) for k in cases]
    wc = [sig[k].max() for k in cases]
    xx = np.arange(len(cases))
    a2.bar(xx - 0.2, gm, 0.4, color=COLOR[2], edgecolor="k", label="geometric mean")
    a2.bar(xx + 0.2, wc, 0.4, color=COLOR[0], edgecolor="k", label="worst case")
    a2.set_yscale("log"); a2.set_xticks(xx)
    a2.set_xticklabels([_SHORT.get(k, k) for k in cases], fontsize=8)
    a2.set_ylabel(r"$\sigma_\beta$")
    a2.set_title("Aggregate on THIS interior:\ncoverage is the worst-case bar")
    a2.grid(True, axis="y", which="both", alpha=0.3); a2.legend(fontsize=8)

    if has_sweep:   # the verdict panel: does any criterion survive averaging?
        a3 = axes[2]
        sw = res["sweep"]
        keys = [k for k in cases if k != "SH only"]
        for i, k in enumerate(keys):
            v = sw[k]
            a3.scatter(np.full_like(v, i) + np.random.default_rng(i).uniform(
                -0.13, 0.13, len(v)), v, s=26, color=cols[cases.index(k)],
                edgecolor="k", lw=0.4, zorder=3)
            a3.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="k", lw=2.2,
                    zorder=4)
        a3.axhline(1.0, color="crimson", ls="--", lw=1.5,
                   label="parity with farthest-point")
        a3.set_yscale("log")
        a3.set_xticks(range(len(keys)))
        a3.set_xticklabels([_SHORT.get(k, k) for k in keys], fontsize=8)
        a3.set_ylabel(r"worst-case $\sigma_\beta$ improvement over FPS")
        a3.set_title(f"Across {len(sw[keys[0]])} independent interiors\n"
                     "(bar = median; below the line = worse than geometry)")
        a3.grid(True, axis="y", which="both", alpha=0.3); a3.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt3_fig2_massratio.pdf"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 3: position recovery ------------------------------------------
    if res["pos_rms"]:
        fig, ax = plt.subplots(figsize=(13.5, 5.6))
        pr = res["pos_rms"]
        # NOT fig 2's `x`: that one carries an extra slot for the derived body
        # fraction, which has no position — this axis is one bar per anomaly
        xp = np.arange(len(P))
        for i, (k, col) in enumerate(zip(cases, cols)):
            ax.bar(xp + (i - (len(cases) - 1) / 2) * w, pr[k], w, color=col,
                   edgecolor="k", lw=0.5, label=k)
        ax.set_yscale("log"); ax.set_xticks(xp)
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
        ax.set_ylabel("position RMS error [LU]")
        ax.set_title("Position recovery of randomly placed mascons: "
                     "geometric vs SH-aware networks")
        ax.grid(True, axis="y", which="both", alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "global_pt3_fig3_position.pdf"),
                    dpi=180, bbox_inches="tight")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST — the SH synthesis added here must reproduce the point-mass field
# ═══════════════════════════════════════════════════════════════════════════


def _selftest(verbose=True):
    rng = np.random.default_rng(0)
    # 1) vectorized Legendre == pt1's scalar version
    for x in rng.uniform(-0.99, 0.99, 5):
        a = legendre_bar(8, x)[:, :, 0]
        b = G.fully_normalized_legendre(8, float(x))
        assert np.allclose(a, b, atol=1e-12), "Legendre mismatch"
    # 2) synthesis converges to the exact mascon potential / acceleration
    P = np.array([[0.30, 0.05, -0.02], [-0.35, -0.10, 0.06], [0.0, 0.0, 0.20]])
    f = np.array([0.5, 0.3, 0.2])
    Rref, L = 0.86, 30
    pts = rng.normal(size=(20, 3))
    pts = 1.6 * Rref * pts / np.linalg.norm(pts, axis=1)[:, None]   # well outside
    c = stokes_vector(P, f, 0, L, Rref)
    U_sh = sh_pot_basis(pts, 0, L, Rref) @ c
    g_sh = sh_grad_basis(pts, 0, L, Rref) @ c
    U_t, g_t = truth_field(pts, P, f)
    eU = np.max(np.abs(U_sh - U_t) / np.abs(U_t))
    eg = np.max(np.linalg.norm(g_sh - g_t, axis=1) / np.linalg.norm(g_t, axis=1))
    assert eU < 1e-8 and eg < 1e-6, f"synthesis error U={eU:.1e} g={eg:.1e}"
    # 3) the claim the experiment rests on: a degree-only covariance gives a
    #    sigma_g that depends on |r| ONLY (addition theorem) — same radius,
    #    scattered directions, identical value.
    sph = rng.normal(size=(40, 3))
    sph = 0.5 * sph / np.linalg.norm(sph, axis=1)[:, None]
    Pk = kaula_covariance(6)
    B = sh_pot_basis(sph, 0, 6, Rref)                     # exact, no differencing
    s_pot = np.sqrt(np.einsum("nc,cd,nd->n", B, Pk, B))
    sp_pot = float(np.ptp(s_pot) / np.mean(s_pot))
    s_kau = map_sh_sigma(sph, 6, Rref, Pk)                # gradient: FD-limited
    sp_grad = float(np.ptp(s_kau) / np.mean(s_kau))
    assert sp_pot < 1e-12, f"Kaula sigma_U is not radial ({sp_pot:.1e})"
    assert sp_grad < 1e-5, f"Kaula sigma_g is not radial ({sp_grad:.1e})"
    # 4) the constant-density bulk: unit mass and centre of mass at the origin
    #    are exact properties of the quadrature, and the two INDEPENDENT paths to
    #    its field — tetrahedral-quadrature Stokes + SH synthesis, and
    #    polyhedral_gravity — must agree outside the Brillouin sphere to the
    #    degree-L truncation.  This pins both the unit-mass normalization of
    #    `Bulk.stokes` and the SI-G division inside `Bulk.field`.
    Vb, Fb, tmb, Rb = G.load_eros()
    bulk = G.Bulk(Vb, Fb)
    c01 = bulk.stokes(0, 1, Rb)
    assert abs(c01[0] - 1.0) < 1e-12, f"bulk mass != 1 ({c01[0]})"
    assert np.max(np.abs(c01[2:])) < 1e-9, "bulk centre of mass is not the origin"
    q = rng.normal(size=(12, 3))
    q = 3.0 * Rb * q / np.linalg.norm(q, axis=1)[:, None]
    U_q = sh_pot_basis(q, 0, 6, Rb) @ bulk.stokes(0, 6, Rb)
    U_p = bulk.field(q)[: len(q)]
    eb = float(np.max(np.abs(U_q - U_p) / np.abs(U_p)))
    assert eb < 5e-3, f"bulk Stokes vs polyhedral_gravity disagree ({eb:.1e})"
    if verbose:
        print(f"  selftest OK — SH synthesis vs exact mascon field: "
              f"|dU|/U = {eU:.1e}, |dg|/g = {eg:.1e}")
        print(f"  selftest OK — degree-only covariance ⇒ isotropic map: "
              f"sigma_U radial to {sp_pot:.1e} (exact), "
              f"sigma_g to {sp_grad:.1e} (central-difference limited)")
        print(f"  selftest OK — constant-density bulk: mass = {c01[0]:.12f}, "
              f"|COM| < 1e-9, quadrature Stokes vs polyhedral_gravity at 3R* "
              f"agree to {eb:.1e} (degree-6 truncation)")


if __name__ == "__main__":
    _selftest()
    res = run(
        Lmax_sh=6,
        eps=0.02,
        ch_modes=(6, 6),
        n_cyl=6,
        n_cand=140,
        n_mc=300,
        n_mc_pos=30,
        do_position=True,
        n_sweep=8,
        outdir="Images",
        verbose=True,
    )
    print("\nSaved: Images/global_pt3_fig1_criteria.pdf, "
          "global_pt3_fig2_massratio.pdf, global_pt3_fig3_position.pdf")
    print("Done.")
