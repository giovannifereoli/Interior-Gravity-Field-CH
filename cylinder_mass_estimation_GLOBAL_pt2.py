"""
Part 2 — A NETWORK of cylindrical harmonics for ARBITRARY interior mascons
==========================================================================
Author: Giovanni Fereoli / experiment build

Question (different from pt1)
-----------------------------
pt1 showed that ONE near-surface CH cylinder resolves ONE anomaly sitting under
it.  A real body has many mass concentrations at unknown, scattered locations,
and a single cylinder only helps whatever is beneath it.

    Does SH + a NETWORK of CH cylinders (near-surface data all around the body)
    let us estimate the mass fraction and position of MANY, arbitrarily placed
    anomalies — where SH alone, and SH + a single cylinder, cannot?

Interior model (same as pt1)
----------------------------
Mass lives in the CONSTANT-DENSITY POLYHEDRON scaled by β̃ = 1 − Σβ; the mascons
are the localized DEPARTURES from homogeneity, β_j = m_j/M*, positive for an
excess and negative for a deficit.  Every design matrix is therefore a contrast
against the constant-density model — what is fitted is ΔU = U_measured − U_CD —
and there is no Σβ = 1 pseudo-observation, the budget being structural.  Two
consequences for the experiment: pt1's "core" mascon stood in for the bulk, so
here the deep mascon is a genuine deep ANOMALY; and the noise is referred to the
full measured field (bulk included), which the anomalies perturb by a few %.

Idea
----
Place ~6 anomalies at scattered interior locations of Eros and build a NETWORK
of CH cylinders on the surface, each a patch of near-surface / low-altitude data
with its own Bessel–Fourier expansion.  Cylinders go on the surface INSIDE the
Brillouin sphere (the sides / waist), where exterior SH is weakest and CH
converges; the long-axis tips, which sit on the Brillouin sphere, are skipped.

Three observation models are compared:
    A  : SH only               (global Stokes, degree 2..L, + total mass)
    A1 : SH + ONE CH cylinder  (pt1-style, single near-surface patch)
    AN : SH + the CH NETWORK   (all cylinders' coefficients)

Since an anomaly's localized signature is captured by whatever cylinder is near
it, the NETWORK constrains every anomaly, the single cylinder only its
neighbour, and SH alone leaves them degenerate.

Both observables are linear in β (mass-fraction experiment = linear LS) and
nonlinear in position (position experiment = TRF, every anomaly's
position free in one joint fit).  All heavy machinery
(Legendre, Stokes design, Bessel basis, constant-density bulk, fits) is reused
from `cylinder_mass_estimation_GLOBAL`, imported as G.

Units: Eros normalized (LU), total mass M* = 1, G = 1.
"""

from __future__ import annotations
import os
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy import stats

import cylinder_mass_estimation_GLOBAL as G  # reuse pt1 machinery

COLOR = G.COLOR
ACCENT = G.ACCENT
# Every panel is its own file here too (see pt1): same canvases, same saver.
FS, _save = G.FS, G._save
FONT_SCALE = G.FONT_SCALE  # one knob for all text sizes; see GLOBAL
FS_BAR = (9.0, 5.4)  # bars: 7 groups x 4 models plus wrapped tick labels
FS_COR = (6.8, 5.8)  # correlation matrix + its own colour bar
mpl.rcParams.update({"axes.prop_cycle": mpl.cycler(color=COLOR), "figure.dpi": 110})
# ── the five models' colours, named once ────────────────────────────────────
# Constrained from several directions: no orange and no pink (both read badly
# against the vermillion), and no two blues — the sky blue that sat here was too
# close to the SH blue to tell apart in a six-cell small multiple.  What is left
# that stays separable under colour-vision deficiency is blue, violet, teal,
# olive, vermillion.  The teal and olive are Paul Tol's; the rest are Okabe-Ito,
# and the three shared with pt1 keep pt1's assignment.
CASE_COLORS = (COLOR[2], G.CH_VIOLET, "#44AA99", "#999933", COLOR[0])

SEP = "=" * 72

# ═══════════════════════════════════════════════════════════════════════════
# NETWORK OF CH CYLINDERS
# ═══════════════════════════════════════════════════════════════════════════


def rot_z_to(d):
    """Rotation matrix R with R @ [0,0,1] = d  (aligns a cylinder axis to d)."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(d, z):
        return np.eye(3)
    if np.allclose(d, -z):
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, d)
    s = np.linalg.norm(v)
    c = float(np.dot(z, d))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def farthest_point_sample(pts, k, start_idx=None, seed=0):
    """
    Pick k well-spread points (greedy farthest-point sampling).  If `start_idx`
    is given the first point is fixed there (used to anchor the network at the
    body's underside); otherwise the first point is random.
    """
    rng = np.random.default_rng(seed)
    first = int(start_idx) if start_idx is not None else int(rng.integers(len(pts)))
    idx = [first]
    d = np.linalg.norm(pts - pts[first], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    return np.asarray(idx)


def build_network(
    V,
    F,
    tm,
    n_cyl=6,
    radius=0.12,  # cylinder radius [LU]
    height=0.32,  # cylinder height [LU]
    gap=0.03,  # lift of the base above its surface vertex [LU]
    alpha=100.0,  # Bessel extension; the Cylinder default, named here so a
    #                 caller can change it without editing the constructor
    n_pts=180,  # field samples drawn per cylinder
    brillouin_frac=0.80,
    seed=1,
):
    """
    Network of CH cylinders on the INSIDE-Brillouin surface of the body.

    The `brillouin_frac` parameter restricts candidate mesh vertices to those
    whose radial distance from the origin is less than `brillouin_frac * Rb`,
    where `Rb` is the maximum vertex radius. This excludes the elongated tips
    and concentrates the cylinders around the body's sides and waist.

    Farthest-point sampling is anchored at the −z (underside) extreme so the
    network is guaranteed a cylinder below the body—otherwise, on Eros's
    flat-in-z shape, the greedy sampler tends to double up on the top face.

    `gap` lifts each cylinder off the vertex it is anchored to, along the
    outward normal, so its base clears the local terrain — pt1 calls the same
    quantity `cyl_gap`.  It is 0.03 LU here against pt1's 0.005 because these
    cylinders sit on the curved sides rather than flat on the +z pole, where a
    thinner gap lets the rim cut back into the body.

    Returns a list of dicts: {cyl, obs, dir, surf}.
    """
    Rb = float(np.linalg.norm(V, axis=1).max())
    surf = V[np.linalg.norm(V, axis=1) < brillouin_frac * Rb]  # sides / waist
    start_idx = int(np.argmin(surf[:, 2]))  # anchor the first cylinder below
    idx = farthest_point_sample(surf, n_cyl, start_idx=start_idx, seed=seed)
    net = []
    for s in surf[idx]:
        d = s / np.linalg.norm(s)  # outward radial normal
        cyl = G.Cylinder(
            center=s + gap * d,
            radius=radius,
            height=height,
            alpha=alpha,
            R=rot_z_to(d),
        )
        obs = G.cylinder_points(cyl, n=n_pts, seed=seed + 1)
        obs = obs[~G.inside_body(tm, V, F, obs)]  # vacuum only
        net.append(dict(cyl=cyl, obs=obs, dir=d, surf=s))
    return net


# I dont like this, also what does the 4% comment mean
def place_mascons(net, seed=2):
    """
    Scatter ANOMALIES for the experiment: one near-surface anomaly under each of
    the first (n_cyl-1) cylinder SITES, plus one DEEP anomaly near the centre of
    the shape (the hard case, far from every near-surface patch).

    Cylinders are not assigned to anomalies: they are near-surface data patches,
    all entering the same joint least squares, and every anomaly is estimated
    from all of them at once.  Shallow anomalies are merely PLACED under
    cylinder sites so that "covered" and "uncovered" cases both exist; the names
    say where things sit, not who owns what.

    Returns names, positions and truth mass fractions β_j = m_j/M* — a few per
    cent each, mixed signs (over- and under-dense), NOT ratios summing to one:
    the remaining β̃ = 1 − Σβ stays in the constant-density polyhedron.
    """
    names, pos = [], []
    for i, c in enumerate(net[:-1]):
        names.append(_site_name(c["dir"], names))
        pos.append(0.72 * c["surf"])  # just inside the surface, under cylinder i
    # NOT the bulk — that is the polyhedron carrying beta~ = 1 - sum(beta).
    # This is a genuine deep ANOMALY, ~4% of the body mass, and it is the hard
    # case: the one anomaly with no near-surface patch above it.  Avoid "core"
    # in the name; in the old parameterization a "core" mascon stood in for the
    # body, and the word still reads that way.
    names.append("Deep Interior")
    pos.append(np.array([0.08, 0.0, 0.0]))
    pos = np.array(pos)
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.015, 0.05, len(pos)) * rng.choice([-1.0, 1.0], len(pos))
    f[-1] = abs(f[-1])  # keep the deep one an EXCESS (a dense concentration)
    return names, pos, f


# A readable name per outward direction, in place of m0/m1/...  Farthest-point
# sampling can land two sites on the same face, so repeats get a numeral.
_DIR_NAME = {
    "+x": "East Lobe",
    "-x": "West Lobe",
    "+y": "North Flank",
    "-y": "South Flank",
    "+z": "Upper Face",
    "-z": "Underside",
}


def _site_name(d, taken):
    base = _DIR_NAME[_axis_label(d)]
    if base not in taken:
        return base
    n = sum(1 for t in taken if t.startswith(base)) + 1
    return f"{base} {'II III IV V VI'.split()[n - 2]}"


def _axis_label(d):
    ax = "xyz"[int(np.argmax(np.abs(d)))]
    return ("+" if d[np.argmax(np.abs(d))] > 0 else "-") + ax


# ═══════════════════════════════════════════════════════════════════════════
# POSITION FIT (ALL anomaly positions free at once, masses fixed) — network
# ═══════════════════════════════════════════════════════════════════════════
# Every anomaly moves in the same fit: 3N unknowns (18 for six anomalies).  The
# alternative — fit one anomaly while the other five sit at their TRUE
# positions — returns the CONDITIONAL error, the one you would get if the rest
# of the interior were already known exactly.  It never is, and positions trade
# off against each other (two anomalies can share a low-degree Stokes signature
# between them), so the conditional error is optimistic: at the nominal L_SH = 6
# the joint (marginal) error is ~3-4× larger for SH alone, 1-4× with the
# network, and it is the one an observer actually faces.
#
# Data vs unknowns.  Counting is not the problem at the nominal setup: SH gives
# 45 informative Stokes coefficients for L_SH = 6 (the 50 packed entries include
# the five S̄_n0 ≡ 0), and each cylinder another 72, against 18 unknowns.  The
# joint Jacobian is full rank in every case, condition ~10^2.5-10^3.  Counting
# DOES bite in the L_SH sweep: SH alone has 5, 12 informative coefficients at
# L_SH = 2, 3, fewer than 18, so some position combinations need the prior.
# `position_sigma_net` reports how much each anomaly's marginal sigma shrinks
# relative to its prior; rank deficiency alone does not set this ratio flag.


def _pos_forward_net(positions, masses, bulk, Lmax, Rref, ch_data, use_sh=True):
    """
    Full forward model β̃·CD + Σ β_k pt_k at the anomaly positions `positions`
    (N, 3) — all of them free.  The bulk term is an additive constant here (β̃
    is fixed with the masses), but it is written out so the forward model is the
    one the parameterization defines rather than a mascons-in-vacuum stand-in.

    `use_sh=False` drops the global block entirely — the CH-only model, which
    has no SH observation at all.
    """
    positions = np.asarray(positions, float)
    blocks = []
    if use_sh:
        # one batched Stokes evaluation for all the anomalies, not one call each
        # — see the same note in `G._pos_forward`; with 6 anomalies it matters more
        S = G.sh_stokes_basis(positions, 2, Lmax, Rref)
        y_sh = G.bulk_fraction(masses) * bulk.stokes(2, Lmax, Rref)
        for mk, Sk in zip(masses, S):
            y_sh = y_sh + mk * Sk
        blocks.append(y_sh)
    for pinvPhi, obs in ch_data:  # each cylinder's CH coefficients
        field = G.bulk_fraction(masses) * bulk.field(obs)
        for mk, pk in zip(masses, positions):
            field = field + mk * G.point_mass_field(pk, obs)
        blocks.append(pinvPhi @ field)
    return blocks


def _pos_jacobian_net(positions, masses, Lmax, Rref, ch_data, use_sh=True, h=1e-5):
    """
    ∂(forward blocks)/∂(all positions): one (n_b, 3N) matrix per block, in the
    order `_pos_forward_net` returns them, columns ordered (anomaly, x/y/z) —
    i.e. `positions.ravel()`.

    Block-sparse by construction: the forward model is a SUM over anomalies, so
    column (j, a) sees anomaly j's own term only, and the bulk not at all (it
    does not move with p).  Central differences on that one term cost 1/N of a
    full forward evaluation each, which is what keeps the 3N-parameter fit about
    as cheap as fitting one anomaly at a time was.
    """
    positions = np.asarray(positions, float)
    step = h * np.eye(3)
    plus = (positions[:, None, :] + step).reshape(-1, 3)  # row 3j + a
    minus = (positions[:, None, :] - step).reshape(-1, 3)
    w = np.repeat(np.asarray(masses, float), 3) / (2 * h)
    blocks = []
    if use_sh:
        dS = G.sh_stokes_basis(plus, 2, Lmax, Rref) - G.sh_stokes_basis(
            minus, 2, Lmax, Rref
        )
        blocks.append(dS.T * w)
    for pinvPhi, obs in ch_data:
        dF = np.column_stack(
            [
                G.point_mass_field(a, obs) - G.point_mass_field(b, obs)
                for a, b in zip(plus, minus)
            ]
        )
        blocks.append((pinvPhi @ dF) * w)
    return blocks


def _ch_data(net, use_net, pinv_list, ch_modes):
    """[(Φ⁺, obs)] for the cylinders `use_net` selects, and their indices."""
    if use_net is True:
        idx = list(range(len(net)))
    elif use_net is False or use_net is None:
        idx = []
    else:
        idx = list(use_net)
    if not idx:
        return [], idx
    # Phi and its pseudo-inverse do not depend on the truth, so a caller looping
    # over truths can build them once and pass them in; without that this
    # rebuilds a (n_pts x n_modes) SVD on every single call.
    pv = (
        pinv_list
        if pinv_list is not None
        else [G.ch_pinv_for(c["cyl"], c["obs"], ch_modes) for c in net]
    )
    return [(pv[k], net[k]["obs"]) for k in idx], idx


def position_mc_net(
    P,
    beta_true,
    net,
    ch_modes,
    Lmax,
    Rref,
    sig_sh,
    sig_ch_list,
    use_net,
    bulk,
    bounds=None,
    n_mc=240,
    seed=13,
    start_jitter=0.02,
    pinv_list=None,
    use_sh=True,
):
    """
    Monte-Carlo NONLINEAR least-squares (scipy TRF) recovery of ALL anomaly
    positions at once — 3N unknowns, mass fractions fixed at the truth.
    `use_net` selects the CH patches: False → no CH at all, True → the whole
    network, or a SEQUENCE OF INDICES → just those cylinders, so the mass
    experiment's reduced configurations can be run here too.  `use_sh=False`
    drops the global block, which with the full network is the CH-ONLY model.

    What moves where.  The TRUTH is `P`, and it is the caller that moves it:
    `truth_mc_position_net` redraws every anomaly inside a ball about its
    nominal site before calling this.  In here the truth is fixed; each draw
    adds noise to its data and fits.  `start_jitter` is only the solver's
    INITIAL GUESS, truth + U(±start_jitter) per coordinate: the position
    problem is nonlinear, so an iterative solver needs somewhere to start, and
    starting exactly on the truth would flatter it.  The mass problem has no
    such knob because it is LINEAR — its least squares is solved in closed
    form, with no iteration and so no starting point.

    `bounds` (a per-coordinate box, tiled over the anomalies) keep the solver on
    the body, so a weakly-constrained near-central anomaly degrades to a
    large-but-finite error instead of diverging.  Returns positions
    (n_mc, N, 3): still an actual fit, just robustified.
    """
    ch_data, idx = _ch_data(net, use_net, pinv_list, ch_modes)
    sig_blocks = ([sig_sh] if use_sh else []) + [sig_ch_list[k] for k in idx]
    sig_all = np.concatenate(sig_blocks)
    n = len(P)
    p_true = np.asarray(P, float).copy()
    truth = _pos_forward_net(p_true, beta_true, bulk, Lmax, Rref, ch_data, use_sh)
    if bounds is None:
        bounds = (-np.inf, np.inf)
    else:
        bounds = tuple(np.tile(np.broadcast_to(b, 3), n) for b in bounds)

    def resid(x, data):
        model = _pos_forward_net(
            x.reshape(n, 3), beta_true, bulk, Lmax, Rref, ch_data, use_sh
        )
        return np.concatenate([(m - d) / s for m, d, s in zip(model, data, sig_blocks)])

    def jac(x, data):
        J = _pos_jacobian_net(x.reshape(n, 3), beta_true, Lmax, Rref, ch_data, use_sh)
        return np.vstack(J) / sig_all[:, None]

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_mc):
        data = [t + rng.normal(0, s, size=t.shape) for t, s in zip(truth, sig_blocks)]
        p0 = (p_true + rng.uniform(-start_jitter, start_jitter, p_true.shape)).ravel()
        if np.isfinite(bounds[0]).all():
            # TRF needs a strictly feasible start
            p0 = np.clip(p0, bounds[0] + 1e-9, bounds[1] - 1e-9)
        sol = least_squares(
            resid,
            p0,
            jac=jac,  # block-sparse central differences, see `_pos_jacobian_net`
            args=(data,),
            bounds=bounds,
            method="trf",
            # Automatically account for differently sensitive coordinates
            x_scale="jac",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            # Allow difficult cases to converge
            max_nfev=2000,
        )
        out.append(sol.x.reshape(n, 3))
    return np.asarray(out)


# ═══════════════════════════════════════════════════════════════════════════
# MONTE-CARLO OVER THE TRUTH  (same design as pt1's `truth_mc_masses`)
# ═══════════════════════════════════════════════════════════════════════════


def precompute_ch(P, bulk, net, ch_modes):
    """
    Per cylinder: the truncated pseudo-inverse of its Bessel-Fourier basis and
    the CH design it induces.  Neither depends on the truth, so both are built
    ONCE and reused for every drawn interior — otherwise each truth would pay
    for `n_cyl` SVDs of a (4*n_pts x n_modes) matrix.
    """
    out = []
    for c in net:
        pinv = G.ch_pinv_for(c["cyl"], c["obs"], ch_modes)
        out.append(
            dict(
                pinv=pinv, obs=c["obs"], A=pinv @ G.A_field_contrast(P, bulk, c["obs"])
            )
        )
    return out


def case_blocks(pre, A_sh, sig_sh, sig_ch, n_cyl, c0, c1):
    """
    The five observation models, as lists of (design, sigma) blocks.

    CH-only sits second, right beside SH-only: the two single-observable models
    are what the three combined ones have to be read against.  Without it the
    table says the network helps and cannot say what the network IS — see the
    reach map, where SH is a broad flat field and the network a set of sharp
    peaks.
    """
    base = [(A_sh, sig_sh)]
    ch = [(pre[k]["A"], sig_ch[k]) for k in range(n_cyl)]
    return {
        "SH": base,
        f"{n_cyl}-CH": ch,
        f"SH + 1 CH ({c0})": base + ch[:1],
        f"SH + 2 CH ({c0},{c1})": base + ch[:2],
        f"SH + {n_cyl}-CH": base + ch,
    }


# Which cylinders each case uses, and whether it carries the SH block at all.
# The mass fit reads its cases from `case_blocks`; the POSITION fit is nonlinear
# and has to rebuild them, so both read this instead of writing the tuple twice.
def case_config(keys, n_cyl):
    """{case: (cylinder indices, uses the SH block)} in `keys` order."""
    return dict(
        zip(
            keys,
            (
                ([], True),  # SH only
                (list(range(n_cyl)), False),  # n-CH only
                ([0], True),
                ([0, 1], True),
                (list(range(n_cyl)), True),
            ),
        )
    )


def truth_mc_masses_net(
    P,
    bulk,
    net,
    ch_modes,
    Lmax,
    Rref,
    eps,
    n_cyl,
    c0,
    c1,
    n_truth=400,
    seed=101,
    mag=(0.015, 0.05),
):
    """
    Redraw the truth MASS FRACTIONS; for each interior rebuild every sigma from
    that truth's own field and refit, in all four observation models.

    Identical in construction to `G.truth_mc_masses`, so pt1 and pt2 report the
    same kind of number:
      PREDICTED  sig[case]  — analytic (A^T W A)^-1, no sampling;
      REALIZED   dev[case]  — the error made fitting ONE noisy realization of
                              that interior.
    Computed independently, so comparing them is a real consistency test.  All
    four cases get fresh generators on the same seed, seeing the same SH noise.

    ONE draw per interior, as in pt1: the Monte-Carlo is over BODIES, and an
    observer gets one realization per body.  The covariance check here is
    TABLE 1b, realized RMS against predicted 1-sigma over the interiors; the
    dense single-interior cloud that isolates it lives in pt1.

    beta_tilde = 1 - sum(beta) is DERIVED: its variance is 1^T C 1 and its error
    is minus the sum of the anomaly errors — never a free parameter.
    """
    rng = np.random.default_rng(seed)
    betas = rng.uniform(mag[0], mag[1], size=(n_truth, len(P))) * rng.choice(
        [-1.0, 1.0], size=(n_truth, len(P))
    )
    A_sh = G.A_stokes_contrast(P, bulk, 2, Lmax, Rref)
    pre = precompute_ch(P, bulk, net, ch_modes)
    keys = list(case_blocks(pre, A_sh, None, [None] * n_cyl, n_cyl, c0, c1))
    one = np.ones(len(P))

    sig = {k: np.empty((n_truth, len(P))) for k in keys}
    cor = {k: np.empty((n_truth, len(P), len(P))) for k in keys}
    bulk_sig = {k: np.empty(n_truth) for k in keys}
    dev = {k: np.empty((n_truth, len(P))) for k in keys}
    dev_bulk = {k: np.empty(n_truth) for k in keys}
    for i, b in enumerate(betas):
        sig_sh = G.od_sigma(G.sh_coefficients_total(b, P, bulk, 2, Lmax, Rref), eps)
        sig_ch = [
            G.od_sigma(G.ch_coefficients_total(b, P, bulk, q["obs"], q["pinv"]), eps)
            for q in pre
        ]
        cases = case_blocks(pre, A_sh, sig_sh, sig_ch, n_cyl, c0, c1)
        for k, blocks in cases.items():
            C = G.mass_fraction_covariance(blocks)
            sig[k][i] = np.sqrt(np.diag(C))
            # SEPARABILITY, not precision: sigma says how well each anomaly is
            # known, this says whether it can be told apart from the others.
            d = np.sqrt(np.diag(C))
            cor[k][i] = C / np.outer(d, d)
            bulk_sig[k][i] = np.sqrt(one @ C @ one)
            r = np.random.default_rng(7 + i)
            e = G.ls_fit_once(blocks, b, r) - b
            dev[k][i] = e
            dev_bulk[k][i] = -e.sum()
    return dict(
        betas=betas,
        sig=sig,
        bulk_sig=bulk_sig,
        dev=dev,
        dev_bulk=dev_bulk,
        cases=keys,
        corr={k: np.median(cor[k], axis=0) for k in keys},
    )


def _jitter_inside(p, spread, V, F, tm, rng, n_try=200):
    """A truth position drawn uniformly in a ball about `p`, kept inside the body."""
    for _ in range(n_try):
        u = rng.normal(size=3)
        q = p + spread * rng.uniform() ** (1 / 3) * u / np.linalg.norm(u)
        if G.inside_body(tm, V, F, q[None, :])[0]:
            return q
    return p.copy()


def truth_mc_position_net(
    P,
    beta_true,
    bulk,
    net,
    ch_modes,
    Lmax,
    Rref,
    eps,
    V,
    F,
    tm,
    n_cyl,
    keys,
    n_truth=400,
    seed=202,
    spread=0.06,
    pos_bounds=None,
):
    """
    Redraw the truth POSITIONS of every anomaly and refit ALL of them jointly
    in EVERY observation model — SH only, +1 CH, +2 CH, full network — so this
    mirrors the mass experiment, which also fits every beta at once, and the two
    read side by side.  One noisy fit per interior; the loop over interiors
    supplies the sample, as in pt1.

    Returns {case: (n_truth, n_anom)} of the realized position error [LU].
    """
    rng = np.random.default_rng(seed)
    pre = precompute_ch(P, bulk, net, ch_modes)
    pinv_list = [q["pinv"] for q in pre]
    # the same configurations the mass experiment uses
    cfg = case_config(keys, n_cyl)
    err = {k: np.empty((n_truth, len(P))) for k in keys}
    for i in range(n_truth):
        Pi = np.array([_jitter_inside(p, spread, V, F, tm, rng) for p in P])
        sig_sh = G.od_sigma(
            G.sh_coefficients_total(beta_true, Pi, bulk, 2, Lmax, Rref), eps
        )
        sig_ch = [
            G.od_sigma(
                G.ch_coefficients_total(beta_true, Pi, bulk, q["obs"], q["pinv"]), eps
            )
            for q in pre
        ]
        for k in keys:
            idx_k, use_sh_k = cfg[k]
            c = position_mc_net(
                Pi,
                beta_true,
                net,
                ch_modes,
                Lmax,
                Rref,
                sig_sh,
                sig_ch,
                idx_k,
                bulk,
                bounds=pos_bounds,
                n_mc=1,
                seed=seed + 1000 * i,
                pinv_list=pinv_list,
                use_sh=use_sh_k,
            )
            err[k][i] = np.linalg.norm(c[0] - Pi, axis=1)
    return err


def reach_position_joint(reach, P, beta, pre, sig_sh, sig_ch, Lmax, Rref):
    """
    The reach map's POSITION panel, redone so the test anomaly is located
    TOGETHER with the six — 3 + 18 unknowns, every mass fixed — as every other
    position result in pt2 is.  `G.reach_map` places a lone test anomaly in an
    otherwise homogeneous body, the six not in the model at all; that is pt1's
    question and stays pt1's.

    Same grid, same frozen nominal weights, same test mass and the same seed
    (1/R² on every coordinate, R the circumscribing radius) as `G.reach_map`,
    so only the joint part is new.  The test anomaly's marginal information is
    the Schur complement

        S(p) = F_tt(p) − F_t6(p) F_66⁻¹ F_6t(p) ,

    F_66 the six anomalies' 18×18 information (the same at every map point),
    F_6t the cross term.  Information adds over data sets, so each case is a sum
    of the SH and per-cylinder pieces, as in the fits.  Near one of the six the
    two signatures coincide and S collapses toward the seed: two point masses
    at one place cannot be told apart, and the map shows it.

    Returns {case: (n_z, n_x) map}, NaN outside the body, keyed like
    `reach["sigma_pos"]` so it drops in for it.
    """
    XX, ZZ = np.meshgrid(reach["x"], reach["z"], indexing="xy")
    pts = np.column_stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()])
    ins = np.flatnonzero(np.isfinite(reach["sigma"][G.SH_ONLY].ravel()))
    n_in, n6 = ins.size, 3 * len(P)
    ch_data = [(q["pinv"], q["obs"]) for q in pre]
    sig = [sig_sh] + list(sig_ch)
    J6 = _pos_jacobian_net(P, beta, Lmax, Rref, ch_data)
    # every grid point as its own "anomaly": column block i is point i's
    # partials alone, which is exactly the block-sparsity that function exploits
    Jt = _pos_jacobian_net(
        pts[ins], np.full(n_in, reach["beta_test"]), Lmax, Rref, ch_data
    )
    # per data set (SH, then each cylinder): F_66, F_6t, F_tt
    parts = []
    for a, b, s in zip(J6, Jt, sig):
        a, b = a / s[:, None], (b / s[:, None]).reshape(len(s), n_in, 3)
        parts.append(
            (a.T @ a, np.einsum("dj,dia->jia", a, b), np.einsum("dia,dib->iab", b, b))
        )
    seed = 1.0 / reach["pos_prior"] ** 2

    def marginal(use):
        F66 = seed * np.eye(n6) + sum(parts[u][0] for u in use)
        M = sum(parts[u][1] for u in use)
        Ftt = seed * np.eye(3) + sum(parts[u][2] for u in use)
        X = np.linalg.solve(F66, M.reshape(n6, -1)).reshape(M.shape)
        C = np.linalg.inv(Ftt - np.einsum("jia,jib->iab", M, X))
        out = np.full(XX.size, np.nan)
        out[ins] = np.sqrt(np.einsum("kii->k", C) / 3.0)  # posterior_rms, stacked
        return out.reshape(XX.shape)

    ch = list(range(1, len(sig)))
    return {
        G.SH_ONLY: marginal([0]),
        G.CH_ONLY: marginal(ch),
        G.SH_CH: marginal([0] + ch),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════


def run(
    Lmax_sh=6,
    eps=0.02,
    ch_modes=(6, 6),
    n_cyl=6,
    # ── truth-mass draws (experiment A) ────────────────────────────────────
    n_truth_m=400,
    truth_mag=(0.015, 0.05),
    seed_mass=101,
    # ── truth-position draws (experiment B) ────────────────────────────────
    n_truth_p=400,
    pos_spread=0.06,
    seed_pos=202,
    map_n=61,  # grid resolution of the reach map (x-z slice, body-masked)
    outdir="Images",
    verbose=True,
):
    V, F, tm, Rb = G.load_eros()
    Rref = Rb
    net = build_network(V, F, tm, n_cyl=n_cyl)
    names, P, beta_true = place_mascons(net)
    bulk = G.Bulk(V, F)
    beta_bulk = G.bulk_fraction(beta_true)
    c0, c1 = _axis_label(net[0]["dir"]), _axis_label(net[1]["dir"])

    if verbose:
        print(SEP)
        print("  PART 2 — SH + CH NETWORK for arbitrary interior anomalies (Eros)")
        print(SEP)
        print(
            f"  Brillouin R* = {Rb:.3f} LU | {n_cyl} network cylinders | "
            f"{len(P)} anomalies"
        )
        print(
            f"  BULK: constant-density polyhedron, β̃ = 1 − Σβ = {beta_bulk:.3f}"
            " of M*"
        )
        print(
            f"  weights: OD-like σ_i = {eps}·|coeff_i| (floor 10% of RMS); the "
            "Φ-to-field fit is unweighted"
        )
        print("  CH cylinder sites (farthest-point order; all enter the joint fit):")
        for i_c, c in enumerate(net):
            tag = (
                "   <- the 1-CH case"
                if i_c == 0
                else "   <- added by the 2-CH case" if i_c == 1 else ""
            )
            print(
                f"    C{i_c} {_axis_label(c['dir']):>3s}  surface="
                f"{np.round(c['surf'], 3)}  |r|={np.linalg.norm(c['surf']):.2f}" + tag
            )

    # ── EXPERIMENT A — MASS FRACTIONS over TRUTH INTERIORS ──────────────────
    tmm = truth_mc_masses_net(
        P,
        bulk,
        net,
        ch_modes,
        Lmax_sh,
        Rref,
        eps,
        n_cyl,
        c0,
        c1,
        n_truth=n_truth_m,
        seed=seed_mass,
        mag=truth_mag,
    )

    # ── EXPERIMENT B — POSITIONS over TRUTH INTERIORS ───────────────────────
    pos_bounds = (V.min(0) - 0.05, V.max(0) + 0.05)  # keep the solver on the body
    pos_err = truth_mc_position_net(
        P,
        beta_true,
        bulk,
        net,
        ch_modes,
        Lmax_sh,
        Rref,
        eps,
        V,
        F,
        tm,
        n_cyl,
        tmm["cases"],
        n_truth=n_truth_p,
        seed=seed_pos,
        spread=pos_spread,
        pos_bounds=pos_bounds,
    )

    # ── COEFFICIENT SPECTRA at the NOMINAL truth, pre/post fit ─────────────
    # One noisy realization, fitted jointly on SH + the whole network, so the
    # residual can be watched collapsing from the pre-fit discrepancy onto the
    # noise floor.  The network's CH coefficients are POOLED: every cylinder
    # carries the same (n_m, n_n) mode layout, so they group by azimuthal order
    # exactly as a single patch does in pt1.
    pre_ch = precompute_ch(P, bulk, net, ch_modes)
    A_sh_n = G.A_stokes_contrast(P, bulk, 2, Lmax_sh, Rref)
    sig_sh_n = G.od_sigma(
        G.sh_coefficients_total(beta_true, P, bulk, 2, Lmax_sh, Rref), eps
    )
    A_ch_n = np.vstack([q["A"] for q in pre_ch])
    sig_ch_n = np.concatenate(
        [
            G.od_sigma(
                G.ch_coefficients_total(beta_true, P, bulk, q["obs"], q["pinv"]), eps
            )
            for q in pre_ch
        ]
    )
    rng_sp = np.random.default_rng(99)
    d_sh, d_ch = A_sh_n @ beta_true, A_ch_n @ beta_true  # = CS_hetero - CS_homog
    dat_sh = d_sh + rng_sp.normal(0.0, sig_sh_n)
    dat_ch = d_ch + rng_sp.normal(0.0, sig_ch_n)
    Aw = np.vstack([A_sh_n / sig_sh_n[:, None], A_ch_n / sig_ch_n[:, None]])
    yw = np.concatenate([dat_sh / sig_sh_n, dat_ch / sig_ch_n])
    beta_hat, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    spectra = dict(
        Lmin=2,
        Lmax=Lmax_sh,
        ch_modes=ch_modes,
        n_cyl=n_cyl,
        sh=dict(sigma=sig_sh_n, data=dat_sh, model=A_sh_n @ beta_hat),
        ch=dict(sigma=sig_ch_n, data=dat_ch, model=A_ch_n @ beta_hat),
    )

    # ── REACH: where each model can see an anomaly at all ─────────────────
    # `G.reach_map` takes LISTS, so the whole network goes in as-is and the
    # CH information sums over the patches exactly as the joint fit does.
    sig_ch_nom = [
        G.od_sigma(
            G.ch_coefficients_total(beta_true, P, bulk, q["obs"], q["pinv"]), eps
        )
        for q in pre_ch
    ]
    reach = G.reach_map(
        bulk,
        [q["obs"] for q in pre_ch],
        [q["pinv"] for q in pre_ch],
        sig_ch_nom,
        sig_sh_n,
        Lmax_sh,
        Rref,
        V,
        F,
        tm,
        n=map_n,
    )
    # POSITION reach, joint with the six like every other position result here;
    # the mass panel keeps pt1's lone test anomaly
    reach["sigma_pos"] = reach_position_joint(
        reach, P, beta_true, pre_ch, sig_sh_n, sig_ch_nom, Lmax_sh, Rref
    )

    # A single nominal interior, for the admissibility table
    a_min, d_surf, d_obs, b_max = G.admissibility(
        P, beta_true, beta_bulk, bulk.volume, np.vstack([c["obs"] for c in net]), tm
    )

    res = dict(
        V=V,
        F=F,
        Rb=Rb,
        net=net,
        names=names,
        P=P,
        beta_true=beta_true,
        bulk=bulk,
        beta_bulk=beta_bulk,
        n_cyl=n_cyl,
        cases=tmm["cases"],
        truth_mc=tmm,
        pos_err=pos_err,
        pos_spread=pos_spread,
        spectra=spectra,
        reach=reach,
        adm=dict(a_min=a_min, d_surf=d_surf, d_obs=d_obs, b_max=b_max),
        # not diagnostics: `sweep_lmax_sh` reads these back so the sweep runs on
        # the same configuration as the tables it extends
        Lmax_sh=Lmax_sh,
        ch_modes=ch_modes,
        eps=eps,
    )
    if verbose:
        results_report(res)
    make_plots(res, outdir=outdir)
    return res


# ═══════════════════════════════════════════════════════════════════════════
# REPORT  (figures carry no numbers; every quotable value is printed here)
# ═══════════════════════════════════════════════════════════════════════════


def _tex_num(v, nd=2):
    if v == 0 or not np.isfinite(v):
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    return rf"{v / 10 ** e:.{nd}f}\times10^{{{e}}}"


def _lognorm_fit(v):
    """
    (median, multiplicative sigma, KS p-value) of a log-normal fitted to `v` > 0.

    The SAME estimator `G.lognormal_overlay` draws, so the table and the curve on
    the figure describe one fit: mu = mean of log v, sigma = sd of log v, reported
    as the median exp(mu) and the MULTIPLICATIVE exp(sigma) — the 1-sigma band is
    [median/sigma, median*sigma], not median +/- sigma.

    The p-value is a Kolmogorov-Smirnov test of log v against the fitted normal,
    and it is worth reading before quoting a row.  At n = 600 per panel the
    log-normal is REJECTED in 11 of the 12 mass panels and 6 of the 12 position
    ones, so these fits are a summary of location and width — NOT a claim that
    the errors are log-normal.  Quote median and sigma; do not quote the shape.

    Two things drive it, both measured rather than assumed:

    SHAPE.  Both quantities are magnitudes and so are left-skewed in log, but by
    different amounts — skew(log|e|) = -1.53 against skew(log|Delta p|) = -0.92,
    where a log-normal needs 0.  The mass quantity is a SINGLE scalar |e| (one
    draw per interior), hence half-normal, with a shoulder at zero no log-normal
    follows; the position quantity is |Delta p|, a norm over THREE components,
    which self-averages and lands much closer.  With sigma held fixed, synthetic
    draws are rejected 95% of the time for |e| at n = 240 against 30% for
    |Delta p| — the shape difference is real, and it is why position still fares
    better here (6 of 12 rejected against 11 of 12).

    SAMPLE SIZE.  KS power grows with n, so this verdict is not portable.  The
    same synthetic |e| is rejected 26% of the time at n = 80 and 95% at n = 240.
    An earlier run at n = 240 (mass) against n = 80 (position) made position look
    uniformly log-normal; equalising both at 600 showed that was the smaller
    sample, not a better model.  That is why `run` now draws the same n for the
    two experiments.

    A third effect rescues individual panels: the observed value is sigma_i |z|,
    so a sigma that VARIES across interiors adds a normal term in log and drags
    the sum back toward normal.  Synthetically, |e| at n = 240 goes from 95%
    rejected at a sigma spread of x1.0 to 26% at x2.0 and 4% at x3.0.
    """
    v = np.asarray(v, float).ravel()
    lv = np.log(v[v > 0])
    ks = stats.kstest(lv, "norm", args=(lv.mean(), lv.std(ddof=1)))
    return float(np.exp(lv.mean())), float(np.exp(lv.std(ddof=1))), float(ks.pvalue)


def results_report(res):
    names, tmm, cases = res["names"], res["truth_mc"], res["cases"]
    ft, sig, bsig = res["beta_true"], tmm["sig"], tmm["bulk_sig"]
    net_k = cases[-1]
    rms = lambda M: np.sqrt(np.mean(np.asarray(M) ** 2, axis=0))
    print(f"\n{SEP}\n  RESULTS  (figures carry no numbers; quote from here)\n{SEP}")

    # ── TABLE 0 — are the truth anomalies physically realizable? ────────────
    adm = res["adm"]
    print(
        f"\n  TABLE 0 — physical admissibility of the truth anomalies "
        f"(excess ceiling Δρ/ρ = {G.EXCESS_CONTRAST:.2f})"
    )
    print(
        f"  {'anomaly':16s} {'β':>8} {'a_min':>8} {'to surf':>8} {'to obs':>8}"
        f" {'β_max':>9}  verdict"
    )
    for k, nm in enumerate(names):
        v = (
            "breaches the surface"
            if adm["a_min"][k] > adm["d_surf"][k]
            else (
                "field points inside it"
                if adm["a_min"][k] > adm["d_obs"][k]
                else "buried, clear of the data — exact"
            )
        )
        print(
            f"  {nm:16s} {ft[k]:+8.3f} {adm['a_min'][k]:8.3f} "
            f"{adm['d_surf'][k]:8.3f} {adm['d_obs'][k]:8.3f} "
            f"{adm['b_max'][k]:+9.4f}  {v}"
        )

    # ── TABLE 1 — mass-fraction uncertainty per case ────────────────────────
    print(
        f"\n  TABLE 1 — mass-fraction 1σ, {len(tmm['betas'])} truth interiors,"
        " median over interiors"
    )
    print(
        f"  {'anomaly':16s} "
        + " ".join(f"{k:>18s}" for k in cases)
        + f" {'net gain':>9}"
    )
    for i, nm in enumerate(names):
        row = [np.median(sig[k][:, i]) for k in cases]
        print(
            f"  {nm:16s} "
            + " ".join(f"{v:18.2e}" for v in row)
            + f" {row[0] / row[-1]:8.1f}×"
        )
    print(f"  {'-' * 74}")
    row = [np.median(bsig[k]) for k in cases]
    print(
        f"  {'BODY β̃ = 1−Σβ':16s} "
        + " ".join(f"{v:18.2e}" for v in row)
        + f" {row[0] / row[-1]:8.1f}×"
    )
    g = np.median(sig[cases[0]] / sig[net_k], axis=0)
    print(
        f"  network gain vs SH:  min {g.min():.0f}×   median "
        f"{np.median(g):.0f}×   max {g.max():.0f}×"
    )

    # ── TABLE 1b — does the analytic covariance predict the error made? ─────
    print(
        f"\n  TABLE 1b — covariance consistency, "
        f"{len(tmm['betas'])} noisy fits per case (one per interior): "
        "realized RMS(estimate − truth) vs predicted 1σ"
    )
    print(
        f"  {'anomaly':16s} "
        + " ".join(
            f"{'realized/pred ' + k.split('(')[0]:>22s}" for k in (cases[0], net_k)
        )
    )
    for i, nm in enumerate(names):
        cells = []
        for k in (cases[0], net_k):
            r = rms(tmm["dev"][k][:, i])
            pr = rms(sig[k][:, i])
            cells.append(f"{r:9.2e} /{pr:9.2e} {r / pr:5.2f}")
        print(f"  {nm:16s} " + " ".join(f"{c:>22s}" for c in cells))
    cells = []
    for k in (cases[0], net_k):
        r, pr = rms(tmm["dev_bulk"][k].ravel()), rms(bsig[k])
        cells.append(f"{r:9.2e} /{pr:9.2e} {r / pr:5.2f}")
    print(f"  {'BODY β̃':16s} " + " ".join(f"{c:>22s}" for c in cells))

    # ── TABLE 2 — position ─────────────────────────────────────────────────
    pe = res["pos_err"]
    print(
        f"\n  TABLE 2 — anomaly position, {len(pe[cases[0]])} truth interiors,"
        " RMS error [LU]; all positions fitted jointly"
    )
    print(
        f"  {'anomaly':16s} "
        + " ".join(f"{k:>18s}" for k in cases)
        + f" {'net gain':>9}"
    )
    for i, nm in enumerate(names):
        row = [rms(pe[k][:, i]) for k in cases]
        print(
            f"  {nm:16s} "
            + " ".join(f"{v:18.2e}" for v in row)
            + f" {row[0] / row[-1]:8.1f}×"
        )

    # ── TABLE 2b — the small-multiple panels, as numbers ───────────────────
    # Each cell of the two histogram figures gets a log-normal; these are those
    # fits.  `median ratio` is SH median / network median -- a THIRD gain
    # definition beside TABLE 1's ratio of analytic sigmas and the RMS ratio the
    # figures annotate.  They differ because |e| is skewed: expect the median
    # ratio to sit a little above the RMS one.
    print(
        "\n  TABLE 2b — log-normal fit to each panel of the histogram figures"
        f" ({len(tmm['betas'])} / {len(pe[cases[0]])} interiors)"
    )
    print(
        f"  {'quantity':14s} {'anomaly':16s} {'SH median':>11} {'σx':>6} {'p':>6}"
        f" {'net median':>12} {'σx':>6} {'p':>6} {'med ratio':>10}"
    )
    for title, data in (
        ("mass fraction", {k: np.abs(tmm["dev"][k]) for k in cases}),
        ("position", pe),
    ):
        for i, nm in enumerate(names):
            ma, fa, pa = _lognorm_fit(data[cases[0]][:, i])
            mb, fb, pb = _lognorm_fit(data[net_k][:, i])
            print(
                f"  {title:14s} {nm:16s} {ma:11.2e} {fa:6.2f} {pa:6.3f}"
                f" {mb:12.2e} {fb:6.2f} {pb:6.3f} {ma / mb:9.1f}×"
            )
    print(
        "  p is a KS test of the fit; p < 0.05 rejects the log-normal.  At this n"
        " most rows\n  are rejected — |e| from one draw is half-normal and"
        " |Δp| is a 3-D norm, neither\n  log-normal.  Quote the median and σ as"
        " location and width, not the shape (see\n  `_lognorm_fit` for the"
        " measured skew, and why n must match between the two)."
    )

    # ── TABLE 3 — separability ─────────────────────────────────────────────
    # A component can be precisely determined and still be inseparable from its
    # neighbour: sigma and correlation answer different questions.
    cor = tmm["corr"]
    off = ~np.eye(len(names), dtype=bool)
    print(
        f"\n  TABLE 3 — separability: posterior |correlation| between anomalies"
        " (median over interiors)"
    )
    print(f"  {'observation model':22s} {'max |ρ|':>9} {'mean |ρ|':>10}   worst pair")
    for k in cases:
        M = cor[k]
        a, b = np.unravel_index(np.argmax(np.abs(M) * off), M.shape)
        print(
            f"  {k:22s} {np.abs(M[off]).max():9.3f} {np.abs(M[off]).mean():10.3f}"
            f"   {names[a]} <-> {names[b]}  ({M[a, b]:+.3f})"
        )
    MA, MB = cor[cases[0]], cor[net_k]
    a, b = np.unravel_index(np.argmax(np.abs(MA) * off), MA.shape)
    print(
        f"  worst SH pair {names[a]} <-> {names[b]}:"
        f" {MA[a, b]:+.3f} → {MB[a, b]:+.3f} with the network"
    )
    a, b = np.unravel_index(np.argmax(np.abs(MB) * off), MB.shape)
    print(
        f"  worst NETWORK pair {names[a]} <-> {names[b]}:"
        f" {MA[a, b]:+.3f} → {MB[a, b]:+.3f}"
        "   (near-surface data localizes across the line of sight, not along it)"
    )

    # ── TABLE 4 — how much signal the joint fit consumes ───────────────────
    sp = res["spectra"]
    print("\n  TABLE 4 — coefficient residuals at the nominal truth, whitened by σ")
    print(f"  {'observable':22s} {'PRE-fit RMS':>12} {'POST-fit RMS':>13}")
    for key, nm in (("sh", "SH (degree 2..L)"), ("ch", f"CH ({res['n_cyl']} patches)")):
        d = sp[key]
        pre = np.sqrt(np.mean((d["data"] / d["sigma"]) ** 2))
        post = np.sqrt(np.mean(((d["data"] - d["model"]) / d["sigma"]) ** 2))
        print(f"  {nm:22s} {pre:12.2f} {post:13.2f}")
    print(
        "  (PRE = discrepancy-to-noise; POST ≈ 1 means the fit consumed the "
        "signal and σ is the right size)"
    )

    # ── LaTeX bodies ───────────────────────────────────────────────────────
    # ── TABLE 5 — reach: what each model can see, and where ───────────────
    rm = res["reach"]
    mk_cases = [G.SH_ONLY, G.CH_ONLY, G.SH_CH]
    lbl = {
        G.SH_ONLY: "SH",
        G.CH_ONLY: f"{res['n_cyl']}-CH",
        G.SH_CH: f"SH + {res['n_cyl']}-CH",
    }
    print("\n  TABLE 5 — reach: fraction of the body each model can see")
    print(f"  {'threshold σ_β':24s} " + " ".join(f"{lbl[k]:>14}" for k in mk_cases))
    for thr in G.REACH_LEVELS_MASS:
        cells = []
        for k in mk_cases:
            v = rm["sigma"][k]
            ok = np.isfinite(v)
            cells.append(f"{100.0 * np.sum(v[ok] < thr) / max(1, ok.sum()):13.1f}%")
        print(f"  {'below ' + f'{thr:.0e}':24s} " + " ".join(f"{c:>14}" for c in cells))
    a_, b_ = rm["sigma"][G.SH_ONLY], rm["sigma"][G.CH_ONLY]
    ok = np.isfinite(a_) & np.isfinite(b_)
    ratio = (b_ / a_)[ok]
    print(
        f"  {'dynamic range, best/worst':24s} "
        + " ".join(
            f"{np.nanmax(rm['sigma'][k][np.isfinite(rm['sigma'][k])]) / np.nanmin(rm['sigma'][k][np.isfinite(rm['sigma'][k])]):13.0f}x"
            for k in mk_cases
        )
    )
    print(
        f"  ⇒ the network alone beats SH alone over "
        f"{100.0 * np.sum(ratio < 1) / ratio.size:.0f}% of the cross-section: up "
        f"to\n    {1.0 / ratio.min():.0f}× better under a patch, down to "
        f"{ratio.max():.0f}× worse in the gaps BETWEEN patches.\n    Those gaps "
        f"are where the deep anomaly sits, which is why adding patches never\n"
        f"    turns it into an easy case."
    )
    # ── TABLE 5b — the same reach, for POSITION ───────────────────────────
    rp = rm["sigma_pos"]
    print(
        f"\n  TABLE 5b — reach, position: fraction of the body where a test "
        f"anomaly (β = {rm['beta_test']:.2f}), located jointly with the six, "
        "is placed to"
    )
    print(f"  {'threshold 1σ [LU]':24s} " + " ".join(f"{lbl[k]:>14}" for k in mk_cases))
    for thr in G.REACH_LEVELS_POS:
        cells = []
        for k in mk_cases:
            v = rp[k]
            ok = np.isfinite(v)
            cells.append(f"{100.0 * np.sum(v[ok] < thr) / max(1, ok.sum()):13.1f}%")
        print(f"  {'below ' + f'{thr:.0e}':24s} " + " ".join(f"{c:>14}" for c in cells))
    a_, b_ = rp[G.SH_ONLY], rp[G.CH_ONLY]
    ok = np.isfinite(a_) & np.isfinite(b_)
    ratio = (b_ / a_)[ok]
    print(
        f"  ⇒ the network alone places it better than SH alone over "
        f"{100.0 * np.sum(ratio < 1) / ratio.size:.0f}% of the cross-section: "
        f"up to\n    {1.0 / ratio.min():.0f}× better under a patch, down to "
        f"{ratio.max():.0f}× worse in the gaps."
    )

    print(f"\n{'-' * 74}\n  LaTeX tabular bodies\n{'-' * 74}")
    print("  % Table 1 — mass-fraction 1 sigma per observation model")
    for i, nm in enumerate(names):
        row = [np.median(sig[k][:, i]) for k in cases]
        print(
            f"  {nm} & "
            + " & ".join(f"${_tex_num(v)}$" for v in row)
            + rf" & ${row[0] / row[-1]:.1f}$ \\"
        )
    row = [np.median(bsig[k]) for k in cases]
    print(
        r"  body $\tilde\beta$ & "
        + " & ".join(f"${_tex_num(v)}$" for v in row)
        + rf" & ${row[0] / row[-1]:.1f}$ \\"
    )
    print("  % Table 2 — position RMS error [LU] per observation model")
    for i, nm in enumerate(names):
        row = [rms(pe[k][:, i]) for k in cases]
        print(
            f"  {nm} & "
            + " & ".join(f"${_tex_num(v)}$" for v in row)
            + rf" & ${row[0] / row[-1]:.1f}$ \\"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════


def make_plots(res, outdir="Images"):
    """
    Three figures, in pt1's conventions: PDF, no numbers in titles (the tables
    carry them), bracketed units on every axis, "(MC)" wherever a quantity comes
    from sampling, and the analytic 1-sigma drawn on top of the realized error
    so bar-vs-tick is a visible consistency check.
    """
    os.makedirs(outdir, exist_ok=True)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    V, F, P, net, names = res["V"], res["F"], res["P"], res["net"], res["names"]
    n_cyl, cases, tmm = res["n_cyl"], res["cases"], res["truth_mc"]
    ft, bulk = res["beta_true"], res["bulk"]
    net_k = cases[-1]
    rms = lambda M: np.sqrt(np.mean(np.asarray(M) ** 2, axis=0))
    lab = [n.replace(" ", "\n", 1) for n in names]  # full name, wrapped once

    # ---- FIG 1: geometry ---------------------------------------------------
    # Geometry only, at pt1's fig-1 size.  The mass budget lived here as a
    # second panel; the truth beta_j and beta~ are in TABLE 0/1 instead.
    fig = plt.figure(figsize=(8.6, 7.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    step = max(1, len(F) // 9000)
    ax.add_collection3d(
        Poly3DCollection(
            V[F[::step]],
            alpha=0.10,
            facecolor="#9ecae1",
            edgecolor="0.6",
            linewidths=0.1,
        )
    )
    for i_c, c in enumerate(net):
        G.draw_cylinder(ax, c["cyl"])
        t = c["surf"] + 0.42 * c["dir"]  # past the far end, never on the tube
        ax.text(
            t[0],
            t[1],
            t[2],
            f"C{i_c} ({_axis_label(c['dir'])})",
            fontsize=8 * FONT_SCALE,
            color=ACCENT,
            ha="center",
            bbox=dict(fc="white", ec="0.7", alpha=0.8, pad=1.2, lw=0.4),
        )
    for nm, q, b in zip(names, P, ft):
        ax.scatter(
            q[0],
            q[1],
            q[2],
            s=90,
            depthshade=False,
            edgecolor="k",
            color=COLOR[0] if b > 0 else COLOR[2],
        )
        ax.text(q[0], q[1], q[2], f"  {nm}", fontsize=8 * FONT_SCALE)
    ax.plot([], [], color=ACCENT, lw=2, label="CH cylinders")
    ax.scatter([], [], color=COLOR[0], label=r"Anomaly $\beta_j>0$")
    ax.scatter([], [], color=COLOR[2], label=r"Anomaly $\beta_j<0$")
    ax.set_xlabel("x [LU]", labelpad=G.LPAD3D)
    ax.set_ylabel("y [LU]", labelpad=G.LPAD3D)
    ax.set_zlabel("z [LU]", labelpad=G.LPAD3D)
    G.set_axes_true_shape(ax, np.vstack([V] + [G.cylinder_hull(c["cyl"]) for c in net]))
    ax.legend(fontsize=9 * FONT_SCALE, loc="upper left")

    G._save3d(fig, outdir, "global_pt2_fig1_geometry.pdf")

    # ---- FIG 1b: Bouguer map of the truth interior -------------------------
    # Same construction as pt1's, six anomalies instead of three and every
    # cylinder of the network marked: it shows at a glance which parts of the
    # surface the network covers and which heterogeneity sits under them.
    # both evaluation surfaces, as in pt1
    for _at, _stem in (
        ("sphere", "fig1b_bouguer_sphere"),
        ("surface", "fig1c_bouguer_surface"),
    ):
        G.bouguer_map(
            ft,
            P,
            bulk,
            V,
            outdir,
            "global_pt2_" + _stem + ".pdf",
            names=names,
            marks=[c["cyl"].center for c in net],
            at=_at,
            F=F,
        )

    # ---- FIG 2 / FIG 3: the two experiments, drawn identically -------------
    # Specular by construction: same four observation models, same bar chart,
    # same pooled histogram with a log-normal fit.  Read them side by side and
    # the only difference is what is being recovered — mass or position.  Bars
    # and histogram go to separate files, so each experiment is two figures.
    # One colour per model, shared by the bar panels and the histograms so a
    # reader carries the key from one figure to the next.  CH-only takes the
    # spare purple; every previously existing case keeps the colour it had.
    colmap = dict(zip(cases, CASE_COLORS))

    def _cases_panel(ax, vals, pred, ylabel, title):
        """Bars per anomaly across the four models; ticks = analytic 1σ if given."""
        n = len(vals[cases[0]])
        x = np.arange(n, dtype=float) * 1.25
        # five models per group now, so the bars narrow and the offsets are
        # generated rather than written out
        w = 1.05 / len(cases)
        offs = (np.arange(len(cases)) - (len(cases) - 1) / 2) * w
        for k, off in zip(cases, offs):
            ax.bar(x + off, vals[k], w, color=colmap[k], edgecolor="k", label=k)
            if pred is not None:
                ax.plot(
                    x + off, pred[k], "_", ms=9, mew=2.0, color="k", ls="none", zorder=6
                )
        if pred is not None:
            ax.plot(
                [],
                [],
                "_",
                ms=9,
                mew=2.0,
                color="k",
                ls="none",
                label=r"Analytic 1$\sigma$",
            )
        g = vals[cases[0]] / vals[cases[-1]]
        for q in range(n):
            ax.text(
                x[q] + offs[-1],
                vals[cases[-1]][q] * 1.15,
                rf"${g[q]:.0f}\times$",
                ha="center",
                va="bottom",
                fontsize=8 * FONT_SCALE,
                color=COLOR[0],
                zorder=7,
                # opaque backing: the label is centred over the SHORTEST bar of
                # its group, so at paper font sizes it grows wide enough to run
                # over the taller neighbour beside it ("10x" read as "0x")
                bbox=dict(fc="white", ec="none", pad=1.0),
            )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(
            lab + ([r"BODY $\tilde\beta$"] if n > len(lab) else []),
            fontsize=9 * FONT_SCALE,
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.45)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8 * FONT_SCALE, ncol=2)

    def _hist_panel(series, show, xlabel, fname):
        """
        Error distribution per anomaly, one curve per model in `show`.

        SMALL MULTIPLES, not one pooled histogram.  Pooling the six anomalies
        into a single set of curves is correct but unreadable: their sigmas span
        ~15x under SH against ~3.6x under the network, so the pooled SH curve is a
        six-component mixture ~x3.0 wide from the mixing alone.  With the
        single-draw term (x3.04) on top, each pooled curve spans ~x4.8 while the
        curves are only ~7x apart, and the separation the bar panel reports reads
        as overlap.
        One cell per anomaly removes the mixing term: inside a cell every case
        refers to the SAME sigma, so the shift between them IS that anomaly's
        gain, printed in the corner.  The shared x-axis keeps the anomaly-to-
        anomaly offsets visible as the groups sliding left and right.

        `show` is a SUBSET of the models, not all of them: the bar panel carries
        all five, but five overlapping histograms per cell is mud.  The three
        that earn a curve are the two single-observable models and the full
        network — the intermediate 1-CH and 2-CH configurations differ from the
        network by less than the single-draw scatter, so they would add ink
        without adding a distinguishable curve.
        """
        # LEFT LIMIT from a percentile, not from the minimum.  With one draw per
        # interior the plotted quantity is |e|, and |e| reaches arbitrarily close
        # to zero on a lucky draw; one such draw would stretch the log axis over
        # a decade of empty space.
        arrs = {k: np.asarray(series[k]) for k in show}
        lo = min(np.percentile(arrs[k], 2) for k in show)
        hi = max(arrs[k].max() for k in show) * 1.2
        bins = np.logspace(np.log10(lo), np.log10(hi), 26)
        # line style per case, because the log-normal overlays are all black:
        # the histograms carry the colour, the fits carry the dash pattern
        ls_of = dict(zip(show, ("-", ":", "--", "-.", (0, (3, 1, 1, 1)))))
        # Compact stand-ins for the corner box, where three full model names
        # ("SH + 6-CH") will not fit in a small multiple — compact, but the
        # cylinder count stays: a bare "CH" reads as ONE cylinder.  The joint
        # case is "SH+6-CH", NOT "net": that was its name back when only two models were
        # drawn and only one of them carried the network, and it stopped being
        # unambiguous the moment CH-only — which is equally the network — joined
        # the panel.  Named per case rather than by position, and anything not
        # in the map keeps its full name, so adding a model cannot silently
        # relabel another one.
        n_c = res["n_cyl"]
        _short = {cases[0]: "SH", cases[1]: f"{n_c}-CH", cases[-1]: f"SH+{n_c}-CH"}
        short = {k: _short.get(k, k) for k in show}
        fig, axs = plt.subplots(2, 3, figsize=(11.4, 6.2), sharex=True, sharey=True)
        cmax = 0.0
        for i, (ax, nm) in enumerate(zip(axs.ravel(), names)):
            for k in show:
                c, _, _ = ax.hist(
                    np.clip(arrs[k][:, i], lo, None),
                    bins=bins,
                    color=colmap[k],
                    alpha=0.75,
                    edgecolor="k",
                    lw=0.4,
                    label=k,
                )
                cmax = max(cmax, c.max())
            # log-normal fitted to each cell separately, drawn AND quoted here in
            # the same style pt1's histograms use.  `lognormal_overlay` writes a
            # label carrying mu and sigma, which would put a different legend on
            # every cell, so only the first cell contributes generic entries and
            # the per-cell values go in the corner box instead.
            txt = []
            for k in show:
                med, fac = G.lognormal_overlay(
                    ax, arrs[k][:, i], bins, "k", ls=ls_of[k]
                )
                ax.get_lines()[-1].set_label(
                    f"Log-normal fit, {short[k]}" if i == 0 else "_nolegend_"
                )
                ex = int(np.floor(np.log10(abs(med))))
                txt.append(
                    rf"{short[k]}: $\mu={med / 10 ** ex:.2f}\times10^{{{ex}}}$,"
                    rf" $\sigma=\times{fac:.2f}$"
                )
            # The NAME is a title, as in every other small multiple in this
            # project; only the NUMBERS live in the corner box, which a title
            # cannot carry.  The two used to be one in-axes block here and a
            # title in the sweep grids, so the same anomaly was labelled two
            # different ways in two figures.
            ax.set_title(nm, fontsize=10.5 * FONT_SCALE)
            ax.text(
                0.03,
                0.96,
                "\n".join(txt),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.0 * FONT_SCALE,
                linespacing=1.32,
                bbox=dict(fc="white", ec="0.8", lw=0.5, alpha=0.92, pad=2.5),
            )
            ax.set_xscale("log")
            ax.grid(True, which="both", ls=":", alpha=0.45)
            ax.set_axisbelow(True)
        # headroom for the corner box: the tallest bars sit mid-panel and would
        # otherwise run into it (the axes share y, so one call sets all six).
        # Taller than it was, because the box now carries three fit lines.
        axs[0, 0].set_ylim(0, cmax * 1.95)
        for ax in axs[-1]:
            ax.set_xlabel(xlabel, fontsize=10 * FONT_SCALE)
        for ax in axs[:, 0]:
            ax.set_ylabel(
                f"Truth Interiors  (of {len(arrs[show[0]])})  [-]",
                fontsize=10 * FONT_SCALE,
            )
        h, lab_ = axs[0, 0].get_legend_handles_labels()
        fig.legend(
            h,
            lab_,
            loc="lower center",
            # TWO columns, not one per model: the entries are three histograms
            # followed by their three fits, and matplotlib fills column-wise, so
            # ncol=3 interleaves them.  With ncol=2 each model sits on a row
            # beside its own fitted curve.
            ncol=2,
            frameon=False,
            fontsize=9 * FONT_SCALE,
            bbox_to_anchor=(0.5, 0.005),
        )
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.savefig(os.path.join(outdir, fname), bbox_inches="tight")

    # FIG 2 — MASS FRACTIONS
    real, pred = {}, {}
    for k in cases:
        real[k] = np.append(rms(tmm["dev"][k]), rms(tmm["dev_bulk"][k]))
        pred[k] = np.append(rms(tmm["sig"][k]), rms(tmm["bulk_sig"][k]))
    fig, ax = plt.subplots(figsize=FS_BAR)
    _cases_panel(
        ax,
        real,
        pred,
        r"Mass-fraction Error  [-]  (MC)",
        "Each added patch constrains the anomalies it covers",
    )
    _save(fig, outdir, "global_pt2_fig2_massratio_bars.pdf")

    # one value per (interior, anomaly): with a single draw per interior that is
    # just |e|, and `dev[k]` is already (n_truth, n_anom).  NOT a mean over
    # axis 1 -- that axis is the ANOMALIES now, not the noise draws.
    per_truth = {k: np.abs(tmm["dev"][k]) for k in cases}
    # SH alone, the network alone, and the two together — see `_hist_panel`
    hist_show = [cases[0], cases[1], cases[-1]]
    _hist_panel(
        per_truth,
        hist_show,
        r"Mass-fraction Error  [-]",
        "global_pt2_fig2_massratio_hist.pdf",
    )

    # FIG 3 — POSITIONS  (same two panels, same order, same two files)
    pe = res["pos_err"]
    fig, ax = plt.subplots(figsize=FS_BAR)
    _cases_panel(
        ax,
        {k: rms(pe[k]) for k in cases},
        None,
        "Position RMS Error  [LU]  (MC)",
        "The same patches, recovering position instead of mass",
    )
    _save(fig, outdir, "global_pt2_fig3_position_bars.pdf")

    _hist_panel(
        pe,
        hist_show,
        "Position Error  [LU]",
        "global_pt2_fig3_position_hist.pdf",
    )

    # ---- FIG 3b: REACH — where each model can see an anomaly at all --------
    # The same two figures as pt1 — a test anomaly's MASS 1σ and its POSITION
    # 1σ — drawn by the same `G.reach_panels`.  The position one is JOINT: the
    # test anomaly is located together with the six (`reach_position_joint`).  With a NETWORK the CH panel is
    # no longer one hot spot but several, and the GAPS between them are the
    # part worth looking at — they are where the deep anomaly lives, and why
    # it stays the hard case however many patches are added.
    rm = res["reach"]
    n_c = res["n_cyl"]
    # the network's own labels, not pt1's: "CH only" here means all of them
    titles = {
        G.SH_ONLY: G.SH_ONLY,
        G.CH_ONLY: f"{n_c}-CH",
        G.SH_CH: f"SH + {n_c}-CH",
    }
    cyls = [c["cyl"] for c in res["net"]]
    for maps, levels, ceiling, cbar_label, fname in (
        (
            rm["sigma"],
            G.REACH_LEVELS_MASS,
            G.PRIOR_SIGMA,
            r"Mass-fraction 1$\sigma$  [-]",
            "global_pt2_fig3b_reach_map.pdf",
        ),
        (
            rm["sigma_pos"],
            G.REACH_LEVELS_POS,
            rm["pos_prior"],
            r"Position 1$\sigma$  [LU]",
            "global_pt2_fig3b_reach_map_position.pdf",
        ),
    ):
        # the DEEP anomaly (last) is the starred one: it is the case no patch
        # reaches, where pt1 stars the near-surface target
        G.reach_panels(
            maps,
            rm["x"],
            rm["z"],
            V,
            P,
            len(P) - 1,
            cyls,
            titles,
            levels,
            ceiling,
            cbar_label,
            os.path.join(outdir, fname),
            cyl_lw=1.1,
        )

    # ---- FIG 4: separability -----------------------------------------------
    # sigma says how WELL each anomaly is known; this says whether it can be
    # told APART from the others.  A component can be precise and still be
    # inseparable from its neighbour, which is the degeneracy real interior
    # models live with: every mass element contributes to every coefficient.
    # Median over interiors, elementwise: a summary for display, not a matrix
    # to invert.  It matters that this is a median — the correlation depends on
    # the truth through the relative-noise weights, and a single interior can be
    # far from typical (the nominal one puts Upper Face <-> Deep Interior at
    # -0.96, against a median of -0.28).
    cor = tmm["corr"]
    short = [n.replace(" ", "\n", 1) for n in names]
    # one file per observation model; with no shared layout to place them, the
    # x-label names the model that a left/right position used to imply.
    # CH-only earns a panel here for the same reason it earns a column in the
    # tables: separability is where the patches differ most from the global
    # field, because a patch breaks a degeneracy by SEEING one anomaly and not
    # its neighbour, which is a thing only local data can do.  The "sh"/"net"
    # file tags are the ones this figure has always written, so they are left
    # alone and CH-only takes "ch".
    for tag, k in zip(("sh", "ch", "net"), (cases[0], cases[1], net_k)):
        fig, ax = plt.subplots(figsize=FS_COR)
        M = cor[k]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        for a in range(len(names)):
            for b in range(len(names)):
                if a == b:
                    continue
                ax.text(
                    b,
                    a,
                    f"{M[a, b]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5 * FONT_SCALE,
                    color="w" if abs(M[a, b]) > 0.55 else "0.15",
                )
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(short, fontsize=7 * FONT_SCALE, rotation=45, ha="right")
        ax.set_yticklabels(short, fontsize=7 * FONT_SCALE)
        ax.set_xticks(np.arange(len(names) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", lw=1.2)
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel(k, fontsize=10 * FONT_SCALE)
        fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.04, label="posterior correlation  [-]"
        )
        fig.savefig(
            os.path.join(outdir, f"global_pt2_fig4_separability_{tag}.pdf"),
            bbox_inches="tight",
        )

    # ---- FIG 5: coefficient residuals, before and after the fit ------------
    # Same construction as pt1's coefficient figure.  Per-degree (SH) and
    # per-azimuthal-order (CH) RMS of the residual in the coefficients' own
    # units, with 1-sigma drawn as its own curve:
    #     PRE-fit  = measured - homogeneous          (the discrepancy + noise)
    #     POST-fit = measured - (homogeneous + A beta_hat)
    # A post-fit curve sitting on the sigma curve says the fit has consumed the
    # signal and sigma is the right size.
    # The network's CH coefficients are POOLED into one curve set: every
    # cylinder carries the same (n_m, n_n) mode layout, so they group by
    # azimuthal order exactly as a single patch does in pt1.  Drawing the six
    # patches separately puts 18 curves on one axes and is unreadable; the
    # per-patch numbers, if wanted, are in TABLE 4.
    # CAVEAT: reading the ratio off the plot is approximate — sigma varies
    # within a group, and RMS|r| / RMS(sigma) != RMS(r/sigma).  The exact
    # whitened numbers are in TABLE 4.
    sp = res["spectra"]
    Lmin, Lmax = sp["Lmin"], sp["Lmax"]
    n_m, n_n = sp["ch_modes"]

    def _groups(key):
        if key == "sh":  # group by degree n
            xs, gr, acc = [], [], 0
            for n in range(Lmin, Lmax + 1):
                k = 2 * (n + 1)
                xs.append(n)
                gr.append(np.arange(acc, acc + k))
                acc += k
            return np.array(xs), gr
        per = 2 * n_m * n_n  # coefficients per cylinder
        idx = np.arange(sp["n_cyl"] * per)  # pooled over the whole network
        azi = ((idx % per) // 2) // n_n
        xs = np.arange(n_m)
        return xs, [np.where(azi == m)[0] for m in xs]

    for key, xlab in [
        ("sh", r"SH degree $n$  [-]"),
        ("ch", r"CH azimuthal order $m$  [-]"),
    ]:
        fig, ax = plt.subplots(figsize=FS)
        d = sp[key]
        pre, post = d["data"], d["data"] - d["model"]
        xs, gr = _groups(key)
        r = lambda v: np.array([np.sqrt(np.mean(v[g] ** 2)) for g in gr])
        y_pre, y_post, y_sig = r(np.abs(pre)), r(np.abs(post)), r(d["sigma"])
        # NOISE as three nested 1σ/2σ/3σ bands, drawn exactly as pt1 draws
        # them (`G.sigma_bands`, which also sets the x-limits)
        G.sigma_bands(ax, xs, y_sig)
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
        # see the note in `G.make_plots`: this is a power spectrum, and only one
        # of the three curves on it is a residual
        ax.set_ylabel("RMS Coefficient Power  [-]")
        ax.set_yscale("log")
        # the bands reach down to zero, which a log axis cannot show, so the
        # floor still comes from the CURVES
        ax.set_ylim(0.5 * min(y_post.min(), y_sig.min()), 2.5 * y_pre.max())
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.45)
        ax.set_axisbelow(True)

        # legend in a reserved strip below the axes, one column: a single-panel
        # canvas cannot fit these three labels side by side (as in pt1)
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
            os.path.join(outdir, f"global_pt2_fig5_coefficients_{key}.pdf"),
            bbox_inches="tight",
        )


# ═══════════════════════════════════════════════════════════════════════════
# L_SH SWEEP — DOES THE NETWORK SURVIVE A BETTER GLOBAL FIELD?
# ═══════════════════════════════════════════════════════════════════════════
# pt1 asks this of ONE patch over ONE anomaly.  The network changes the
# question in a way worth separating out: a single patch can only ever help the
# anomaly beneath it, so as L_SH rises it is racing the global field on one
# target.  The network covers most of the body, and the anomaly it CANNOT cover
# — the deep one, with no patch above it — is precisely the one the global field
# is best at.  So the two curves should behave differently, and the interesting
# number is not whether the network keeps its edge on the covered anomalies
# (pt1 already says it does) but whether it keeps any edge on the deep one.
#
# Analytic throughout — the same Fisher covariances TABLE 1 quotes — so the
# whole sweep costs one `Bulk.stokes` call at the top degree.  See the long note
# in `cylinder_mass_estimation_GLOBAL.sweep_lmax_sh` for what `alpha` means and
# why both noise rules have to be run.


# Weak Gaussian seed on every position coordinate, the position twin of
# `G.PRIOR_SIGMA`: about the body's half-length, so it only matters where the
# data leave a combination of positions unconstrained (SH alone at L_SH = 2, 3).
POS_PRIOR_SIGMA = 1.0  # LU
PRIOR_RATIO_THRESHOLD = 0.95  # flag less than or equal to 5% sigma reduction


def position_covariance_net(
    J_sh, J_ch, subset, sig_sh, sig_ch, use_sh=True, prior_sigma=POS_PRIOR_SIGMA
):
    """
    Linearized JOINT covariance (3N x 3N) of every anomaly position, from SH plus
    the cylinders in `subset` — the covariance of the fit `position_mc_net`
    actually runs, all positions free together, so the sweep and TABLE 2
    describe one estimator.  Anomaly j's 3x3 MARGINAL is the diagonal block
    [3j:3j+3]; inverting that block alone instead would be the conditional
    error, with the other anomalies pinned at the truth.

    J_sh (n_sh, 3N) and J_ch (a list, one (n_coef, 3N) per cylinder) are the
    unweighted partials from `_pos_jacobian_net`; the weights are applied here,
    per coefficient, exactly as in the fits.  The bulk term does not move with p
    and is absent from them.  `prior_sigma` seeds the information so an
    under-determined case still inverts — see `position_sigma_net`.
    """
    Fi = np.eye(J_ch[0].shape[1] if len(J_ch) else J_sh.shape[1]) / prior_sigma**2
    if use_sh:
        Jw = J_sh / sig_sh[:, None]
        Fi = Fi + Jw.T @ Jw
    for k in subset:
        Jw = J_ch[k] / sig_ch[k][:, None]
        Fi = Fi + Jw.T @ Jw
    return np.linalg.inv(Fi)


def position_sigma_net(
    J_sh,
    J_ch,
    subset,
    sig_sh,
    sig_ch,
    use_sh=True,
    ratio_threshold=PRIOR_RATIO_THRESHOLD,
):
    """
    Per-anomaly position 1σ — `G.posterior_rms` of its MARGINAL 3x3 block of
    the joint covariance — and a prior-dominated flag.  Compute the covariance
    once and compare R = sigma / POS_PRIOR_SIGMA with `ratio_threshold`.
    The isotropic coordinate prior has the same RMS sigma, POS_PRIOR_SIGMA.
    R near one means little uncertainty reduction; R much smaller than one
    means substantial reduction.  This summarizes each anomaly's position,
    rather than flagging its three coordinates separately.
    Returns (sigma (N,), prior_bound (N,) bool).
    """
    C = position_covariance_net(
        J_sh,
        J_ch,
        subset,
        sig_sh,
        sig_ch,
        use_sh,
        prior_sigma=POS_PRIOR_SIGMA,
    )
    sigma = np.array(
        [
            G.posterior_rms(C[3 * j : 3 * j + 3, 3 * j : 3 * j + 3])
            for j in range(len(C) // 3)
        ]
    )
    return sigma, sigma / POS_PRIOR_SIGMA >= ratio_threshold


def sweep_lmax_sh(res, L_values=None, alphas=(0.10,), ch_alpha=None):
    """
    Mass-fraction and position 1-sigma for all four observation models as L_SH
    walks up, at each noise rule in `alphas`.  The position sigma is each
    anomaly's marginal from the JOINT covariance, all positions free.  Geometry, truth and precision all
    come out of `res`, so the sweep and TABLE 1 describe one experiment.
    """
    net, P, beta_true, bulk = res["net"], res["P"], res["beta_true"], res["bulk"]
    n_cyl, ch_modes = res["n_cyl"], res["ch_modes"]
    Rref, eps, keys = res["Rb"], res["eps"], list(res["cases"])
    # 2..14: `Bulk.stokes` is a quadrature whose Gauss order grows with Lmax
    # (20 s at 14, two minutes at 22), and the plateau is established well
    # before the top of this range.
    L_values = list(range(2, 15)) if L_values is None else list(L_values)
    Ltop = max(L_values)

    # ONE quadrature at the top degree, then sliced — the SH packing runs
    # strictly degree by degree, so the first G._sh_count(L) rows ARE the
    # degree-L truncation.  See the same note in pt1.
    cs_top = bulk.stokes(2, Ltop, Rref)
    A_sh_top = G.A_stokes(P, 2, Ltop, Rref) - cs_top[:, None]

    # nothing about the network depends on L_SH — that is the comparison
    pre = precompute_ch(P, bulk, net, ch_modes)
    # the models as (cylinders, uses SH), so the mass covariance and the position
    # covariance below cannot drift apart on what a case means
    cfg = case_config(keys, n_cyl)
    # Position partials, likewise: independent of L_SH except through the SH
    # truncation, which slices them exactly as it slices A_sh_top.
    J_sh_top = _pos_jacobian_net(P, beta_true, Ltop, Rref, [], use_sh=True)[0]
    J_ch = _pos_jacobian_net(
        P, beta_true, Ltop, Rref, [(q["pinv"], q["obs"]) for q in pre], use_sh=False
    )
    # the two anomalies the sweep is really about: one under a patch, and the
    # deep one no patch reaches
    i_cov, i_deep = 0, len(P) - 1

    out = {}
    for a in alphas:
        a_ch = a if ch_alpha is None else ch_alpha
        sig_ch = [
            G.od_sigma(
                G.ch_coefficients_total(beta_true, P, bulk, q["obs"], q["pinv"]),
                eps,
                alpha=a_ch,
            )
            for q in pre
        ]
        mass = {k: [] for k in keys}
        mass_prior = {k: [] for k in keys}
        # EVERY anomaly, not just a covered one and the deep one.  Two named
        # examples asked the reader to take on trust that they stood for the
        # rest; with all of them the figure shows the spread directly.
        pos = {k: [] for k in keys}
        pos_prior = {k: [] for k in keys}
        ncf = []
        for L in L_values:
            k_n = G._sh_count(L)
            A_sh = A_sh_top[:k_n]
            # sigma rebuilt from the TRUNCATED measured vector: a solution that
            # stops at L never saw the degrees above it, and the noise floor is
            # a fraction of the RMS of what was actually measured
            sig_sh = G.od_sigma(cs_top[:k_n] + A_sh @ beta_true, eps, alpha=a)
            for k in keys:
                idx_k, use_sh_k = cfg[k]
                blocks = ([(A_sh, sig_sh)] if use_sh_k else []) + [
                    (pre[c]["A"], sig_ch[c]) for c in idx_k
                ]
                m_sig = G.posterior_sigma(
                    G.mass_fraction_covariance(blocks, prior_sigma=G.PRIOR_SIGMA)
                )
                mass[k].append(m_sig)
                mass_prior[k].append(m_sig / G.PRIOR_SIGMA >= PRIOR_RATIO_THRESHOLD)
                # the JOINT position covariance, all anomalies free — the
                # linearized twin of TABLE 2's fit
                p_sig, p_pr = position_sigma_net(
                    J_sh_top[:k_n],
                    J_ch,
                    idx_k,
                    sig_sh,
                    sig_ch,
                    use_sh_k,
                    ratio_threshold=PRIOR_RATIO_THRESHOLD,
                )
                pos[k].append(p_sig)
                pos_prior[k].append(p_pr)
            ncf.append(k_n)
        out[a] = dict(
            mass={k: np.array(v) for k, v in mass.items()},  # (n_L, n_anom)
            # Marginal uncertainty reduction: R near one is flagged "prior".
            # Reuse the stored sigmas; no additional covariance evaluations.
            mass_prior_ratio={k: np.array(v) / G.PRIOR_SIGMA for k, v in mass.items()},
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
        keys=keys,
        names=res["names"],
        i_cov=i_cov,
        i_deep=i_deep,
        eps=eps,
        ch_modes=ch_modes,
        n_cyl=n_cyl,
        L_nominal=res["Lmax_sh"],
        prior_ratio_threshold=PRIOR_RATIO_THRESHOLD,
    )


def sweep_report(sw):
    """The sweep's numbers, in the same voice as `results_report`."""
    L, keys, names = sw["L"], sw["keys"], sw["names"]
    Ln = sw["L_nominal"]
    i_nom = int(np.argmin(np.abs(L - Ln)))
    sh_k, net_k = keys[0], keys[-1]
    print(f"\n{SEP}")
    print("  TABLE 6 — L_SH sweep: does the network survive a better global field?")
    print(SEP)
    print(
        f"  covered anomaly: {names[sw['i_cov']]} | deep anomaly: "
        f"{names[sw['i_deep']]}"
    )
    print(f"  {sw['n_cyl']} cylinders, CH modes {sw['ch_modes']}, eps = {sw['eps']}")
    print(
        f"  Prior-dominated: R = posterior sigma / prior sigma >= "
        f"{sw['prior_ratio_threshold']:.2f} (position uses per-anomaly RMS sigma)."
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
            print(f"    [CH blocks held at alpha = {d['ch_alpha']}]")
        print(
            f"  {'L':>3} {'n_coef':>7} | {'sig_b covered':>14} {'gain':>7} | "
            f"{'sig_b deep':>12} {'gain':>7} | {'pos deep [LU]':>14} {'gain':>7}"
        )
        mc_s, mc_n = d["mass"][sh_k][:, sw["i_cov"]], d["mass"][net_k][:, sw["i_cov"]]
        md_s, md_n = d["mass"][sh_k][:, sw["i_deep"]], d["mass"][net_k][:, sw["i_deep"]]
        pd_s = d["pos"][sh_k][:, sw["i_deep"]]
        pd_n = d["pos"][net_k][:, sw["i_deep"]]
        bad = (
            d["mass_prior"][sh_k][:, sw["i_cov"]]
            | d["mass_prior"][sh_k][:, sw["i_deep"]]
            | d["pos_prior"][sh_k][:, sw["i_deep"]]
        )
        for i, Li in enumerate(L):
            mark = "  <- main run" if i == i_nom else ""
            if bad[i]:
                mark = "  (*)" + mark
            print(
                f"  {Li:3d} {d['n_coef'][i]:7d} | {mc_n[i]:14.2e} "
                f"{mc_s[i]/mc_n[i]:6.1f}x | {md_n[i]:12.2e} "
                f"{md_s[i]/md_n[i]:6.1f}x | {pd_n[i]:14.2e} "
                f"{pd_s[i]/pd_n[i]:6.1f}x{mark}"
            )
        if bad.any():
            print(
                "  (*) At least one displayed SH-only sigma remains near its "
                "prior sigma;\n      the corresponding gain uses a "
                "prior-dominated baseline."
            )
        # CH-only is FLAT in L — it has no SH block to extend — so it is a fixed
        # bar the global field is trying to clear.  Quoted rather than given a
        # column: the table is already eight wide.
        ch_k = keys[1]
        print(
            f"  [{ch_k}, flat in L: covered {d['mass'][ch_k][0, sw['i_cov']]:.2e}, "
            f"deep {d['mass'][ch_k][0, sw['i_deep']]:.2e}, "
            f"deep position {d['pos'][ch_k][0, sw['i_deep']]:.2e} LU]"
        )
        for lbl, cs, tv in (
            ("covered mass", mc_s, mc_n[i_nom]),
            ("deep mass", md_s, md_n[i_nom]),
            ("deep position", pd_s, pd_n[i_nom]),
        ):
            hit = G._first_reach(L, cs, tv)
            print(
                f"  => {lbl}: SH-only would need degree {hit:.1f} to match the "
                f"L={Ln} network ({tv:.2e})"
                if hit is not None
                else f"  => {lbl}: SH-only does not reach the L={Ln} network "
                f"({tv:.2e}) at any degree up to {L[-1]}"
            )
    # the whole point of a NETWORK: how the benefit is distributed over the body
    a0 = sw["alphas"][0]
    d0 = sw["by_alpha"][a0]
    print(f"\n  per-anomaly mass gain (alpha = {a0}), SH-only / {net_k}")
    print(f"  {'anomaly':22s} {f'L={L[0]}':>9} {f'L={L[i_nom]}':>9} {f'L={L[-1]}':>9}")
    for j, nm in enumerate(names):
        g = d0["mass"][sh_k][:, j] / d0["mass"][net_k][:, j]
        pr = d0["mass_prior"][sh_k][:, j]
        tag = "  <- no patch above it" if j == sw["i_deep"] else ""
        cells = "".join(
            f"{g[i]:8.1f}x" if not pr[i] else f"{'(prior)':>9s}" for i in (0, i_nom, -1)
        )
        print(f"  {nm:22s}{cells}{tag}")


def make_sweep_plots(sw, outdir="Images"):
    """
    Posterior/prior sigma ratios R of the mass fraction and the position
    against L_SH.  One file per quantity, one cell per anomaly with every model
    in each cell, and ONE y range across the pair.

    NO GAIN PANELS.  They plotted ratios of curves already on these axes, which
    is the same information twice over — with SH-only drawn in every cell the
    reader takes the ratio off the gap.  TABLE 6 quotes the crossings.
    """
    os.makedirs(outdir, exist_ok=True)
    L, keys, names = sw["L"], sw["keys"], sw["names"]
    Ln = sw["L_nominal"]
    # ONE noise rule: `sweep_lmax_sh` sweeps the experiment's own alpha by
    # default, and a cell already carries one curve per model
    d_main = sw["by_alpha"][sw["alphas"][0]]
    # the case colours pt2 uses everywhere else (see fig 2's colmap)
    # CH-only takes the spare purple; every previously existing case keeps
    # the colour a reader already knows it by.
    colmap = dict(zip(keys, CASE_COLORS))
    mk = dict(zip(keys, ("o", "D", "^", "v", "s")))

    def _mark_nominal(ax, Ln):
        ax.axvline(Ln, color="0.55", lw=1.0, ls=":", zorder=1)
        ax.annotate(
            "Main Run",
            xy=(Ln, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8.5 * FONT_SCALE,
            color="0.35",
            ha="left",
            va="top",
        )

    def _centred_legend(fig, handles, labels, ncol, fontsize=9.0):
        """
        A legend whose every row is CENTRED, in rows of at most `ncol`.

        One `fig.legend(ncol=n)` fills column-wise and left-aligns whatever is
        left over, so five entries in two columns come out as a ragged 3-then-2
        block hanging off the left.  Drawing one legend PER ROW instead — each
        its own centred figure legend — makes 3-over-2 sit symmetrically under
        the axes, and a row that happens to be full looks no different.
        `Figure.legends` is a list, so the calls stack rather than replace.
        """
        rows = [
            list(range(i, min(i + ncol, len(handles))))
            for i in range(0, len(handles), ncol)
        ]
        # Placed BELOW the canvas (negative y, anchored by its top) rather than
        # inside a strip reserved with tight_layout's `rect`.  Reserving a strip
        # means guessing its height, and guessing high leaves a band of dead
        # space between the x label and the legend; `bbox_inches="tight"` at
        # save time grows the crop to include whatever hangs below, so the gap
        # is exactly the offset asked for.
        fig.tight_layout()
        for r, idx in enumerate(rows):
            fig.legend(
                [handles[i] for i in idx],
                [labels[i] for i in idx],
                loc="upper center",
                ncol=len(idx),
                fontsize=fontsize * FONT_SCALE,
                frameon=False,
                bbox_to_anchor=(0.5, -0.015 - 0.058 * r),
            )

    def _grid(
        getter, ylab, fname, gain=False, ylim=None, flags=None, prior_ratio=False
    ):
        """
        One cell per ANOMALY, every model drawn in each.

        This was two named panels, "covered" and "deep", which asked the reader
        to accept that those two anomalies stood in for the other four.  A small
        multiple per anomaly drops the question: the spread BETWEEN cells is the
        result — patches help enormously where they look and barely at all in
        the gaps — and no cell has to be given a role for the point to land.
        """
        n = len(names)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(11.4, 3.2 * nrows), sharex=True, sharey=True
        )
        axs = np.atleast_2d(axs)
        for j, (ax, nm) in enumerate(zip(axs.ravel(), names)):
            # The CH-only network and the full joint fit COINCIDE on most of
            # these cells — the patches carry the solution, so adding the global
            # block back changes almost nothing — and one curve then hides the
            # other outright.  Same fix as pt1's position panel: CH-only draws
            # as a large HOLLOW ring and the joint fit as a smaller filled
            # marker, so a filled marker inside a ring is visibly two series at
            # one value rather than a single series.
            for rank, k in enumerate(keys):
                y = getter(d_main, k, j)
                if gain:
                    y = getter(d_main, keys[0], j) / y
                hollow = k == keys[1]
                ax.plot(
                    L,
                    y,
                    "-",
                    color=colmap[k],
                    marker=mk[k],
                    ms=(10 if hollow else 6),
                    mfc=("none" if hollow else colmap[k]),
                    mec=(colmap[k] if hollow else "k"),
                    mew=(1.5 if hollow else 0.6),
                    label=k,
                    zorder=3 + rank,
                )
                if flags is not None:
                    # Keep the model marker and overlay a cross for little
                    # reduction from the prior.  Hollow markers identify CH-only.
                    fl = np.asarray(flags(d_main, k, j), bool)
                    ax.plot(
                        L[fl],
                        np.asarray(y)[fl],
                        ls="none",
                        marker="x",
                        ms=9,
                        color="0.15",
                        mew=1.3,
                        zorder=10 + rank,
                    )
            if gain:
                # break-even: below it the extra data has stopped paying
                ax.axhline(1.0, color="0.35", lw=1.0, zorder=1)
            if prior_ratio:
                ax.axhline(1.0, color="0.35", lw=1.0, zorder=1, label=r"Prior $R = 1$")
                ax.axhline(
                    sw["prior_ratio_threshold"],
                    color="0.55",
                    lw=1.0,
                    ls="--",
                    zorder=1,
                    label=rf"Threshold $R = {sw['prior_ratio_threshold']:.2f}$",
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
        # ONE label for the whole figure, not one per row.  Per-row labels of
        # this length ran into each other and into the offset text between the
        # rows; the quantity is the same in every cell, so it only needs saying
        # once.
        fig.supylabel(ylab, fontsize=10.5 * FONT_SCALE)
        axs[0, 0].set_xlim(L[0] - 0.4, L[-1] + 0.4)
        axs[0, 0].set_xticks(L[:: max(1, len(L) // 6)])
        if ylim is not None:
            axs[0, 0].set_ylim(*ylim)  # axes are shared, so one call sets all
        handles, labels = axs[0, 0].get_legend_handles_labels()
        if flags is not None and any(
            np.any(flags(d_main, k, j)) for k in keys for j in range(n)
        ):
            handles.append(
                mpl.lines.Line2D(
                    [],
                    [],
                    ls="none",
                    marker="x",
                    ms=9,
                    color="0.15",
                    mew=1.3,
                )
            )
            labels.append(
                rf"Cross: Prior-Dominated ($R \geq {sw['prior_ratio_threshold']:.2f}$)"
            )
        _centred_legend(fig, handles, labels, 3)
        fig.savefig(os.path.join(outdir, fname), bbox_inches="tight")

    # EVERY degree is drawn, the prior-bound ones included, and tagged
    _mass_prior_g = lambda d, k, j: d["mass_prior"][k][:, j]
    _pos_prior_g = lambda d, k, j: d["pos_prior"][k][:, j]
    _mass_ratio_g = lambda d, k, j: d["mass_prior_ratio"][k][:, j]
    _pos_ratio_g = lambda d, k, j: d["pos_prior_ratio"][k][:, j]

    # ONE y range across the pair so mass and position can be read side by
    # side; R is dimensionless throughout.
    def _span(getter, gain):
        v = []
        for k in keys:
            for j in range(len(names)):
                y = np.asarray(getter(d_main, k, j), float)
                if gain:
                    y = np.asarray(getter(d_main, keys[0], j), float) / y
                v.append(y)
        v = np.concatenate(v)
        v = v[np.isfinite(v) & (v > 0)]
        return (0.6 * v.min(), 1.9 * v.max())

    mass_ratio_lim = _span(_mass_ratio_g, False)
    pos_ratio_lim = _span(_pos_ratio_g, False)
    ratio_lim = (
        min(mass_ratio_lim[0], pos_ratio_lim[0]),
        max(1.2, mass_ratio_lim[1], pos_ratio_lim[1]),
    )
    _grid(
        _mass_ratio_g,
        r"Mass-Fraction $R = \sigma_{\mathrm{post}} / \sigma_{\mathrm{prior}}$  [-]",
        "global_pt2_fig6c_lsh_mass_prior_ratio.pdf",
        ylim=ratio_lim,
        flags=_mass_prior_g,
        prior_ratio=True,
    )
    _grid(
        _pos_ratio_g,
        r"Position $R = \sigma_{\mathrm{post}} / \sigma_{\mathrm{prior}}$  [-]",
        "global_pt2_fig6d_lsh_position_prior_ratio.pdf",
        ylim=ratio_lim,
        flags=_pos_prior_g,
        prior_ratio=True,
    )


if __name__ == "__main__":
    res = run(
        Lmax_sh=6,
        eps=0.02,
        ch_modes=(6, 6),
        n_cyl=6,
        # equal counts on purpose: the log-normal KS test in TABLE 2b gains
        # power with n, so mass and position must be judged on the same n
        n_truth_m=400,  # truth interiors for the mass experiment
        n_truth_p=400,  # truth interiors for the position experiment
        pos_spread=0.06,
        outdir="Images",
        verbose=True,
    )
    # The L_SH sweep extends TABLE 1 along one axis, so it runs on the
    # experiment that produced it rather than rebuilding its own network.
    sw = sweep_lmax_sh(res)
    sweep_report(sw)
    make_sweep_plots(sw, outdir="Images")

    print("\nSaved to Images/ (one file per panel):")
    for _f in (
        "global_pt2_fig1_geometry.pdf",
        "global_pt2_fig1b_bouguer_sphere.pdf",
        "global_pt2_fig1c_bouguer_surface.pdf",
        "global_pt2_fig2_massratio_bars.pdf",
        "global_pt2_fig2_massratio_hist.pdf",
        "global_pt2_fig3_position_bars.pdf",
        "global_pt2_fig3_position_hist.pdf",
        "global_pt2_fig3b_reach_map.pdf",
        "global_pt2_fig3b_reach_map_position.pdf",
        "global_pt2_fig4_separability_sh.pdf",
        "global_pt2_fig4_separability_ch.pdf",
        "global_pt2_fig4_separability_net.pdf",
        "global_pt2_fig5_coefficients_sh.pdf",
        "global_pt2_fig5_coefficients_ch.pdf",
        "global_pt2_fig6c_lsh_mass_prior_ratio.pdf",
        "global_pt2_fig6d_lsh_position_prior_ratio.pdf",
    ):
        print("  " + _f)
    print("Done.")
    # every figure at once, the sweep ones included: `make_plots` used to
    # call this itself, before `make_sweep_plots` had drawn anything
    plt.show()
