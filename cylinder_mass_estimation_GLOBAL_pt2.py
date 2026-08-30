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
nonlinear in position (position experiment = TRF).  All heavy machinery
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

import cylinder_mass_estimation_GLOBAL as G  # reuse pt1 machinery

COLOR = G.COLOR
ACCENT = G.ACCENT
# Every panel is its own file here too (see pt1): same canvases, same saver.
FS, _save = G.FS, G._save
FS_BAR = (9.0, 5.4)  # bars: 7 groups x 4 models plus wrapped tick labels
FS_COR = (6.8, 5.8)  # correlation matrix + its own colour bar
mpl.rcParams.update({"axes.prop_cycle": mpl.cycler(color=COLOR), "figure.dpi": 110})
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
    V, F, tm, n_cyl=6, radius=0.12, height=0.32, n_pts=180, brillouin_frac=0.80, seed=1
):
    """
    Network of CH cylinders on the INSIDE-Brillouin surface of the body.
    Farthest-point sampling is anchored at the −z (underside) extreme so the
    network is guaranteed a cylinder below the body — otherwise, on Eros's
    flat-in-z shape, the greedy sampler tends to double up on the top face.
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
            center=s + 0.03 * d,
            radius=radius,
            height=height,
            alpha=100.0,
            R=rot_z_to(d),
        )
        obs = G.cylinder_points(cyl, n=n_pts, seed=seed + 1)
        obs = obs[~G.inside_body(tm, V, F, obs)]  # vacuum only
        net.append(dict(cyl=cyl, obs=obs, dir=d, surf=s))
    return net


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
# OBSERVABLE BLOCKS
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# POSITION FIT (one mascon free, masses + other positions fixed) — network
# ═══════════════════════════════════════════════════════════════════════════


def _pos_forward_net(posj, j, P, masses, Lmax, Rref, ch_data, bulk):
    """
    Full forward model β̃·CD + Σ β_k pt_k with only anomaly j's position free.
    The bulk term is an additive constant here (β̃ is fixed with the masses), but
    it is written out so the forward model is the one the parameterization
    defines rather than a mascons-in-vacuum stand-in.
    """
    positions = P.copy()
    positions[j] = posj
    y_sh = G.bulk_fraction(masses) * bulk.stokes(2, Lmax, Rref)
    for mk, pk in zip(masses, positions):
        y_sh = y_sh + mk * G.sh_stokes_of_point(pk, 2, Lmax, Rref)
    blocks = [y_sh]
    for pinvPhi, obs in ch_data:  # each cylinder's CH coefficients
        field = G.bulk_fraction(masses) * bulk.field(obs)
        for mk, pk in zip(masses, positions):
            field = field + mk * G.point_mass_field(pk, obs)
        blocks.append(pinvPhi @ field)
    return blocks


def position_mc_net(
    j,
    P,
    f_true,
    net,
    ch_modes,
    Lmax,
    Rref,
    sig_sh,
    sig_ch_list,
    use_net,
    bulk,
    bounds=None,
    n_mc=60,
    seed=13,
    start_jitter=0.02,
    pinv_list=None,
):
    """
    Monte-Carlo NONLINEAR least-squares (scipy TRF) recovery of anomaly j's
    position; mass fractions & other positions fixed.  `use_net` selects the CH
    patches: False → SH only, True → the whole network, or a SEQUENCE OF INDICES
    → just those cylinders, so the mass experiment's reduced configurations can
    be run here too.  Each draw fits noisy data from a small random offset
    (`start_jitter`) about the truth, within `bounds` — which keep the solver on
    the body, so a weakly-constrained near-central mascon degrades to a
    large-but-finite error instead of diverging.  Returns positions (n_mc, 3):
    still an actual fit, just robustified.
    """
    ch_data = []
    sig_blocks = [sig_sh]
    if use_net is True:
        idx = list(range(len(net)))
    elif use_net is False or use_net is None:
        idx = []
    else:
        idx = list(use_net)
    if idx:
        # Phi and its pseudo-inverse do not depend on the truth, so a caller
        # looping over truths can build them once and pass them in; without
        # that this rebuilds a (n_pts x n_modes) SVD on every single call.
        pv = (
            pinv_list
            if pinv_list is not None
            else [G.ch_pinv(G.cyl_basis(c["cyl"], c["obs"], *ch_modes)) for c in net]
        )
        for k in idx:
            ch_data.append((pv[k], net[k]["obs"]))
            sig_blocks.append(sig_ch_list[k])
    p_true = P[j].copy()
    truth = _pos_forward_net(p_true, j, P, f_true, Lmax, Rref, ch_data, bulk)
    if bounds is None:
        bounds = (-np.inf, np.inf)

    def resid(posj, data):
        model = _pos_forward_net(posj, j, P, f_true, Lmax, Rref, ch_data, bulk)
        return np.concatenate([(m - d) / s for m, d, s in zip(model, data, sig_blocks)])

    rng = np.random.default_rng(seed + j)
    out = []
    for _ in range(n_mc):
        data = [t + rng.normal(0, s, size=t.shape) for t, s in zip(truth, sig_blocks)]
        p0 = p_true + rng.uniform(-start_jitter, start_jitter, 3)
        sol = least_squares(
            resid,
            p0,
            args=(data,),
            method="trf",
            bounds=bounds,
            xtol=1e-12,
            ftol=1e-12,
            max_nfev=300,
        )
        out.append(sol.x)
    return np.asarray(out)


# ═══════════════════════════════════════════════════════════════════════════
# MONTE-CARLO OVER THE TRUTH  (same design as pt1's `truth_mc_masses`)
# ═══════════════════════════════════════════════════════════════════════════


def precompute_ch(P, net, ch_modes, bulk):
    """
    Per cylinder: the truncated pseudo-inverse of its Bessel-Fourier basis and
    the CH design it induces.  Neither depends on the truth, so both are built
    ONCE and reused for every drawn interior — otherwise each truth would pay
    for `n_cyl` SVDs of a (4*n_pts x n_modes) matrix.
    """
    out = []
    for c in net:
        pinv = G.ch_pinv(G.cyl_basis(c["cyl"], c["obs"], *ch_modes))
        out.append(
            dict(
                pinv=pinv, obs=c["obs"], A=pinv @ G.A_field_contrast(P, c["obs"], bulk)
            )
        )
    return out


def case_blocks(pre, A_sh, sig_sh, sig_ch, n_cyl, c0, c1):
    """The four observation models, as lists of (design, sigma) blocks."""
    base = [(A_sh, sig_sh)]
    ch = [(pre[k]["A"], sig_ch[k]) for k in range(n_cyl)]
    return {
        "SH only": base,
        f"SH + 1 CH ({c0})": base + ch[:1],
        f"SH + 2 CH ({c0},{c1})": base + ch[:2],
        f"SH + {n_cyl}-CH": base + ch,
    }


def truth_mc_masses_net(
    P,
    net,
    bulk,
    ch_modes,
    Lmax,
    Rref,
    eps,
    n_cyl,
    c0,
    c1,
    n_truth=60,
    seed=101,
    mag=(0.015, 0.05),
    n_rea=24,
):
    """
    Redraw the truth MASS FRACTIONS; for each interior rebuild every sigma from
    that truth's own field and refit, in all four observation models.

    Identical in construction to `G.truth_mc_masses`, so pt1 and pt2 report the
    same kind of number:
      PREDICTED  sig[case]  — analytic (A^T W A)^-1, no sampling;
      REALIZED   dev[case]  — errors from fitting noisy data, `n_rea` draws per
                              interior (a single draw is half-normal about sigma
                              with 76% scatter, swamping the interior-to-interior
                              spread the figure shows).
    Computed independently, so comparing them is a real consistency test.  All
    four cases get fresh generators on the same seed, seeing the same SH noise.

    beta_tilde = 1 - sum(beta) is DERIVED: its variance is 1^T C 1 and its error
    is minus the sum of the anomaly errors — never a free parameter.
    """
    rng = np.random.default_rng(seed)
    betas = rng.uniform(mag[0], mag[1], size=(n_truth, len(P))) * rng.choice(
        [-1.0, 1.0], size=(n_truth, len(P))
    )
    A_sh = G.A_stokes_contrast(P, bulk, 2, Lmax, Rref)
    pre = precompute_ch(P, net, ch_modes, bulk)
    keys = list(case_blocks(pre, A_sh, None, [None] * n_cyl, n_cyl, c0, c1))
    one = np.ones(len(P))

    sig = {k: np.empty((n_truth, len(P))) for k in keys}
    cor = {k: np.empty((n_truth, len(P), len(P))) for k in keys}
    bulk_sig = {k: np.empty(n_truth) for k in keys}
    dev = {k: np.empty((n_truth, n_rea, len(P))) for k in keys}
    dev_bulk = {k: np.empty((n_truth, n_rea)) for k in keys}

    for i, b in enumerate(betas):
        sig_sh = G.od_sigma(G.stokes_total(b, P, bulk, 2, Lmax, Rref), eps)
        sig_ch = [
            G.od_sigma(q["pinv"] @ G.field_total(b, P, q["obs"], bulk), eps)
            for q in pre
        ]
        cases = case_blocks(pre, A_sh, sig_sh, sig_ch, n_cyl, c0, c1)
        for k, blocks in cases.items():
            C = np.linalg.inv(G.fisher_masses(blocks))
            sig[k][i] = np.sqrt(np.diag(C))
            # SEPARABILITY, not precision: sigma says how well each anomaly is
            # known, this says whether it can be told apart from the others.
            d = np.sqrt(np.diag(C))
            cor[k][i] = C / np.outer(d, d)
            bulk_sig[k][i] = np.sqrt(one @ C @ one)
            r = np.random.default_rng(7 + i)
            e = np.array([G.ls_fit_once(blocks, b, r) - b for _ in range(n_rea)])
            dev[k][i] = e
            dev_bulk[k][i] = -e.sum(axis=1)
    return dict(
        betas=betas,
        sig=sig,
        bulk_sig=bulk_sig,
        dev=dev,
        dev_bulk=dev_bulk,
        cases=keys,
        n_rea=n_rea,
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
    f_true,
    net,
    bulk,
    ch_modes,
    Lmax,
    Rref,
    eps,
    V,
    F,
    tm,
    n_cyl,
    keys,
    n_truth=25,
    seed=202,
    spread=0.06,
    pos_bounds=None,
):
    """
    Redraw the truth POSITIONS of every anomaly and refit each in EVERY
    observation model — SH only, +1 CH, +2 CH, full network — so this mirrors
    the mass experiment and the two read side by side.  One noisy fit per
    interior; the loop over interiors supplies the sample, as in pt1.

    Returns {case: (n_truth, n_anom)} of the realized position error [LU].
    """
    rng = np.random.default_rng(seed)
    pre = precompute_ch(P, net, ch_modes, bulk)
    pinv_list = [q["pinv"] for q in pre]
    # the same four configurations the mass experiment uses, as cylinder subsets
    subsets = dict(zip(keys, ([], [0], [0, 1], list(range(n_cyl)))))
    err = {k: np.empty((n_truth, len(P))) for k in keys}
    for i in range(n_truth):
        Pi = np.array([_jitter_inside(p, spread, V, F, tm, rng) for p in P])
        sig_sh = G.od_sigma(G.stokes_total(f_true, Pi, bulk, 2, Lmax, Rref), eps)
        sig_ch = [
            G.od_sigma(q["pinv"] @ G.field_total(f_true, Pi, q["obs"], bulk), eps)
            for q in pre
        ]
        for j in range(len(P)):
            for k in keys:
                c = position_mc_net(
                    j,
                    Pi,
                    f_true,
                    net,
                    ch_modes,
                    Lmax,
                    Rref,
                    sig_sh,
                    sig_ch,
                    subsets[k],
                    bulk,
                    bounds=pos_bounds,
                    n_mc=1,
                    seed=seed + 1000 * i,
                    pinv_list=pinv_list,
                )
                err[k][i, j] = float(np.linalg.norm(c[0] - Pi[j]))
    return err


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════


def run(
    Lmax_sh=6,
    eps=0.02,
    ch_modes=(6, 6),
    n_cyl=6,
    # ── truth-mass draws (experiment A) ────────────────────────────────────
    n_truth_m=60,
    truth_mag=(0.015, 0.05),
    seed_mass=101,
    n_rea=24,
    # ── truth-position draws (experiment B) ────────────────────────────────
    n_truth_p=20,
    pos_spread=0.06,
    seed_pos=202,
    outdir="Images",
    verbose=True,
):
    V, F, tm, Rb = G.load_eros()
    Rref = Rb
    net = build_network(V, F, tm, n_cyl=n_cyl)
    names, P, f_true = place_mascons(net)
    bulk = G.Bulk(V, F)
    beta_bulk = G.bulk_fraction(f_true)
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
            "inner Φ-to-field fit is unweighted"
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
        net,
        bulk,
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
        n_rea=n_rea,
    )

    # ── EXPERIMENT B — POSITIONS over TRUTH INTERIORS ───────────────────────
    pos_bounds = (V.min(0) - 0.05, V.max(0) + 0.05)  # keep the solver on the body
    pos_err = truth_mc_position_net(
        P,
        f_true,
        net,
        bulk,
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
    pre_ch = precompute_ch(P, net, ch_modes, bulk)
    A_sh_n = G.A_stokes_contrast(P, bulk, 2, Lmax_sh, Rref)
    sig_sh_n = G.od_sigma(G.stokes_total(f_true, P, bulk, 2, Lmax_sh, Rref), eps)
    A_ch_n = np.vstack([q["A"] for q in pre_ch])
    sig_ch_n = np.concatenate(
        [
            G.od_sigma(q["pinv"] @ G.field_total(f_true, P, q["obs"], bulk), eps)
            for q in pre_ch
        ]
    )
    rng_sp = np.random.default_rng(99)
    d_sh, d_ch = A_sh_n @ f_true, A_ch_n @ f_true  # = CS_hetero - CS_homog
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

    # a single nominal interior, for the admissibility table
    a_min, d_surf, d_obs, b_max = G.admissibility(
        P, f_true, beta_bulk, bulk.volume, np.vstack([c["obs"] for c in net]), tm
    )

    res = dict(
        V=V,
        F=F,
        Rb=Rb,
        net=net,
        names=names,
        P=P,
        f_true=f_true,
        bulk=bulk,
        beta_bulk=beta_bulk,
        n_cyl=n_cyl,
        cases=tmm["cases"],
        truth_mc=tmm,
        pos_err=pos_err,
        pos_spread=pos_spread,
        spectra=spectra,
        adm=dict(a_min=a_min, d_surf=d_surf, d_obs=d_obs, b_max=b_max),
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


def results_report(res):
    names, tmm, cases = res["names"], res["truth_mc"], res["cases"]
    ft, sig, bsig = res["f_true"], tmm["sig"], tmm["bulk_sig"]
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
        f"{len(tmm['betas']) * tmm['n_rea']} noisy fits per case: "
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
            r = rms(tmm["dev"][k].reshape(-1, len(names))[:, i])
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
        " RMS error [LU]"
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
    ft = res["f_true"]
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
            fontsize=8,
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
        ax.text(q[0], q[1], q[2], f"  {nm}", fontsize=8)
    ax.plot([], [], color=ACCENT, lw=2, label="CH cylinders")
    ax.scatter([], [], color=COLOR[0], label=r"Anomaly $\beta_j>0$")
    ax.scatter([], [], color=COLOR[2], label=r"Anomaly $\beta_j<0$")
    ax.set_xlabel("x [LU]")
    ax.set_ylabel("y [LU]")
    ax.set_zlabel("z [LU]")
    G.set_axes_true_shape(ax, np.vstack([V] + [G.cylinder_hull(c["cyl"]) for c in net]))
    ax.legend(fontsize=9, loc="upper left")

    _save(fig, outdir, "global_pt2_fig1_geometry.pdf")

    # ---- FIG 2 / FIG 3: the two experiments, drawn identically -------------
    # Specular by construction: same four observation models, same bar chart,
    # same pooled histogram with a log-normal fit.  Read them side by side and
    # the only difference is what is being recovered — mass or position.  Bars
    # and histogram go to separate files, so each experiment is two figures.
    def _cases_panel(ax, vals, pred, ylabel, title):
        """Bars per anomaly across the four models; ticks = analytic 1σ if given."""
        n = len(vals[cases[0]])
        x = np.arange(n, dtype=float) * 1.25
        w = 0.22
        colmap = dict(zip(cases, (COLOR[2], COLOR[1], COLOR[3], COLOR[0])))
        for k, off in zip(cases, (-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w)):
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
                x[q] + 1.5 * w,
                vals[cases[-1]][q] * 1.15,
                rf"${g[q]:.0f}\times$",
                ha="center",
                va="bottom",
                fontsize=8,
                                color=COLOR[0],
            )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(
            lab + ([r"BODY $\tilde\beta$"] if n > len(lab) else []), fontsize=9
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, ncol=2)

    def _hist_panel(ax, a, b, xlabel, title):
        """Pooled error distribution, SH only vs the full network."""
        a, b = np.ravel(a), np.ravel(b)
        bins = np.logspace(
            np.log10(min(a.min(), b.min()) * 0.8),
            np.log10(max(a.max(), b.max()) * 1.2),
            26,
        )
        ax.hist(
            a,
            bins=bins,
            color=COLOR[2],
            alpha=0.75,
            edgecolor="k",
            lw=0.5,
            label="SH only",
        )
        ax.hist(
            b,
            bins=bins,
            color=COLOR[0],
            alpha=0.75,
            edgecolor="k",
            lw=0.5,
            label=f"SH + {n_cyl}-CH network",
        )
        G.lognormal_overlay(ax, a, bins, "k", ls="-", name="SH")
        G.lognormal_overlay(ax, b, bins, "k", ls="--", name="SH + CH")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Fits  (of {a.size})  [-]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    # FIG 2 — MASS FRACTIONS
    real, pred = {}, {}
    for k in cases:
        real[k] = np.append(
            rms(tmm["dev"][k].reshape(-1, len(P))), rms(tmm["dev_bulk"][k].ravel())
        )
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

    # one value per (interior, anomaly) — the RMS over that interior's noise
    # draws — so this pools the same kind of quantity the position panel does.
    # Pooling the raw signed errors instead would be a half-normal: |e| reaches
    # arbitrarily close to zero, and the histogram grows a tail down to 1e-10
    # that is an artefact of the absolute value, not of the estimator.
    per_truth = {k: np.sqrt((tmm["dev"][k] ** 2).mean(axis=1)) for k in cases}
    fig, ax = plt.subplots(figsize=FS)
    _hist_panel(
        ax,
        per_truth[cases[0]],
        per_truth[cases[-1]],
        r"Mass-fraction RMS Error over MC noise draws  [-]",
        "Distribution over truth interiors and anomalies",
    )
    _save(fig, outdir, "global_pt2_fig2_massratio_hist.pdf")

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

    fig, ax = plt.subplots(figsize=FS)
    _hist_panel(
        ax,
        pe[cases[0]],
        pe[cases[-1]],
        "Position Error, One Fit per Truth Interior  [LU]",
        "Distribution over truth interiors and anomalies",
    )
    _save(fig, outdir, "global_pt2_fig3_position_hist.pdf")

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
    # x-label names the model that a left/right position used to imply
    for tag, k in zip(("sh", "net"), (cases[0], net_k)):
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
                    fontsize=7.5,
                    color="w" if abs(M[a, b]) > 0.55 else "0.15",
                )
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(short, fontsize=7)
        ax.set_xticks(np.arange(len(names) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", lw=1.2)
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel(k, fontsize=10)
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
        ax.set_ylabel("RMS |residual|  [-]")
        ax.set_yscale("log")
        ax.set_ylim(0.5 * min(y_post.min(), y_sig.min()), 2.5 * y_pre.max())
        ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.45)
        ax.set_axisbelow(True)
        for sd_ in ("top", "right"):
            ax.spines[sd_].set_visible(False)

        # legend in a reserved strip below the axes, one column: a single-panel
        # canvas cannot fit these three labels side by side (as in pt1)
        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout(rect=[0, 0.13, 1, 1])
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=1,
            fontsize=9.5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.012),
        )
        fig.savefig(
            os.path.join(outdir, f"global_pt2_fig5_coefficients_{key}.pdf"),
            bbox_inches="tight",
        )
    plt.show()


if __name__ == "__main__":
    res = run(
        Lmax_sh=6,
        eps=0.02,
        ch_modes=(6, 6),
        n_cyl=6,
        n_truth_m=60,  # truth interiors for the mass experiment
        n_rea=24,  # noise draws per interior
        n_truth_p=20,  # truth interiors for the position experiment
        pos_spread=0.06,
        outdir="Images",
        verbose=True,
    )
    print("\nSaved to Images/ (one file per panel):")
    for _f in (
        "global_pt2_fig1_geometry.pdf",
        "global_pt2_fig2_massratio_bars.pdf",
        "global_pt2_fig2_massratio_hist.pdf",
        "global_pt2_fig3_position_bars.pdf",
        "global_pt2_fig3_position_hist.pdf",
        "global_pt2_fig4_separability_sh.pdf",
        "global_pt2_fig4_separability_net.pdf",
        "global_pt2_fig5_coefficients_sh.pdf",
        "global_pt2_fig5_coefficients_ch.pdf",
    ):
        print("  " + _f)
    print("Done.")
