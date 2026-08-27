"""
Part 2 — A NETWORK of cylindrical harmonics for ARBITRARY interior mascons
==========================================================================
Author: Giovanni Fereoli / experiment build

Question (different from pt1)
-----------------------------
pt1 showed that ONE near-surface CH cylinder resolves ONE anomaly sitting under
it.  But a real body has many mass concentrations at unknown, scattered
locations.  A single cylinder only helps whatever is beneath it.

    Does SH + a NETWORK of CH cylinders (near-surface data all around the body)
    let us estimate the mass fraction and position of MANY, arbitrarily placed
    anomalies — where SH alone, and SH + a single cylinder, cannot?

Interior model (same as pt1)
----------------------------
The mass lives in the CONSTANT-DENSITY POLYHEDRON scaled by β̃ = 1 − Σβ; the
mascons are the localized DEPARTURES from homogeneity, β_j = m_j/M*, positive
for an excess and negative for a deficit.  Every design matrix below is
therefore a contrast against the constant-density model — what is fitted is the
discrepancy ΔU = U_measured − U_CD — and there is no Σβ = 1 pseudo-observation,
because the mass budget is structural.  Note what this changes about the
experiment: pt1's "core" mascon was standing in for the bulk, so here the deep
mascon is a genuine deep ANOMALY, and the noise is referred to the full measured
field (bulk included), which the anomalies perturb by only a few per cent.

Idea
----
Place ~6 anomalies at scattered interior locations of Eros.  Build a NETWORK of
CH cylinders on the surface (each a patch of near-surface / low-altitude data,
represented by its own Bessel–Fourier expansion).  The cylinders are put on the
parts of the surface that lie INSIDE the Brillouin sphere (the sides / waist),
where exterior SH is weakest and CH converges — the long-axis tips sit on the
Brillouin sphere and are skipped.

Three observation models are compared:
    A  : SH only                    (global Stokes, degree 2..L, + total mass)
    A1 : SH + ONE CH cylinder       (pt1-style, single near-surface patch)
    AN : SH + the CH NETWORK        (all cylinders' coefficients)

Because an anomaly's localized signature is captured by whatever cylinder is
near it, the NETWORK constrains every anomaly; the single cylinder constrains
only its neighbour; SH alone leaves them degenerate.

Both observables are linear in β (mass-fraction experiment = linear LS), and
nonlinear in an anomaly's position (position experiment = TRF).  All heavy
machinery (Legendre, Stokes design, Bessel basis, the constant-density bulk,
fits) is reused from `cylinder_mass_estimation_GLOBAL` (imported as G).

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
mpl.rcParams.update({"axes.prop_cycle": mpl.cycler(color=COLOR),
                     "figure.dpi": 110})
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


def build_network(V, F, tm, n_cyl=6, radius=0.12, height=0.32, n_pts=180,
                  brillouin_frac=0.80, seed=1):
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
        cyl = G.Cylinder(center=s + 0.03 * d, radius=radius, height=height,
                         alpha=100.0, R=rot_z_to(d))
        obs = G.cylinder_points(cyl, n=n_pts, seed=seed + 1)
        obs = obs[~G.inside_body(tm, V, F, obs)]  # vacuum only
        net.append(dict(cyl=cyl, obs=obs, dir=d, surf=s))
    return net


def place_mascons(net, seed=2):
    """
    Scatter ANOMALIES for the experiment: one near-surface anomaly under each of
    the first (n_cyl-1) cylinder SITES, plus one DEEP anomaly near the centre of
    the shape (the hard case, far from every near-surface patch).

    The cylinders are not assigned to anomalies: they are near-surface data
    patches, every one of them enters the same joint least squares, and every
    anomaly is estimated from all of them at once.  The shallow anomalies are
    merely PLACED under cylinder sites so that "covered" and "uncovered" cases
    both exist; the naming reflects where things sit, not who owns what.
    Returns names, positions and
    truth mass fractions β_j = m_j/M* — a few per cent each, mixed signs
    (over- and under-dense), NOT ratios summing to one: the remaining
    β̃ = 1 − Σβ stays in the constant-density polyhedron.
    """
    names, pos = [], []
    for i, c in enumerate(net[:-1]):
        names.append(f"m{i} near {_axis_label(c['dir'])}")
        pos.append(0.72 * c["surf"])  # just inside the surface, under cylinder i
    # NOT the bulk — that is the polyhedron carrying beta~ = 1 - sum(beta).
    # This is a genuine deep ANOMALY, ~4% of the body mass, and it is the hard
    # case: the one anomaly with no near-surface patch above it.  Avoid "core"
    # in the name; in the old parameterization a "core" mascon stood in for the
    # body, and the word still reads that way.
    names.append("deep (central)")
    pos.append(np.array([0.08, 0.0, 0.0]))
    pos = np.array(pos)
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.015, 0.05, len(pos)) * rng.choice([-1.0, 1.0], len(pos))
    f[-1] = abs(f[-1])  # keep the deep one an EXCESS (a dense concentration)
    return names, pos, f


def _axis_label(d):
    ax = "xyz"[int(np.argmax(np.abs(d)))]
    return ("+" if d[np.argmax(np.abs(d))] > 0 else "-") + ax


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVABLE BLOCKS
# ═══════════════════════════════════════════════════════════════════════════


def ch_blocks_for(P, net, ch_modes, eps, f_true, bulk):
    """
    (A_ch, sigma) for every cylinder in the network — β → CH coefficients.

    The inner fit of the Bessel–Fourier basis to the sampled field is UNWEIGHTED
    (c = Φ⁺ field); A_ch is that fit applied to the CONTRAST field (mass at p_j
    minus the same mass spread through the body), i.e. to ΔU.  The weights enter
    on the COEFFICIENTS: sigma is `G.od_sigma` of the CH coefficients of the FULL
    measured field (bulk + anomalies) — one sigma per coefficient, since that is
    what the instrument delivers before the known constant-density part is
    removed.
    """
    blocks = []
    for c in net:
        Phi = G.cyl_basis(c["cyl"], c["obs"], *ch_modes)
        pinv = G.ch_pinv(Phi)                # unweighted inner fit, trunc. SVD
        A_ch = pinv @ G.A_field_contrast(P, c["obs"], bulk)
        y_ch = pinv @ G.field_total(f_true, P, c["obs"], bulk)
        blocks.append((A_ch, G.od_sigma(y_ch, eps)))
    return blocks


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
    j, P, f_true, net, ch_modes, Lmax, Rref, sig_sh, sig_ch_list, use_net, bulk,
    bounds=None, n_mc=60, seed=13, start_jitter=0.02, pinv_list=None,
):
    """
    Monte-Carlo NONLINEAR least-squares (scipy TRF) recovery of anomaly j's
    position; mass fractions & other positions fixed.  `use_net`=False → SH only;
    True → SH + whole network.  Each draw fits noisy data starting from a small
    random offset (`start_jitter`) about the truth, within `bounds` (which keep
    the solver on the body so a weakly-constrained near-central mascon degrades
    to a large-but-finite error instead of diverging).  Returns positions
    (n_mc, 3) — still an actual fit, just robustified.
    """
    ch_data = []
    sig_blocks = [sig_sh]
    if use_net:
        # Phi and its pseudo-inverse do not depend on the truth, so a caller
        # looping over truths can build them once and pass them in; without
        # that this rebuilds a (n_pts x n_modes) SVD on every single call.
        pv = pinv_list if pinv_list is not None else [
            G.ch_pinv(G.cyl_basis(c["cyl"], c["obs"], *ch_modes)) for c in net
        ]
        for c, s, pinv in zip(net, sig_ch_list, pv):
            ch_data.append((pinv, c["obs"]))
            sig_blocks.append(s)
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
        sol = least_squares(resid, p0, args=(data,), method="trf", bounds=bounds,
                            xtol=1e-12, ftol=1e-12, max_nfev=300)
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
            dict(pinv=pinv, obs=c["obs"],
                 A=pinv @ G.A_field_contrast(P, c["obs"], bulk))
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
        f"SH + {n_cyl}-CH network": base + ch,
    }


def truth_mc_masses_net(
    P, net, bulk, ch_modes, Lmax, Rref, eps, n_cyl, c0, c1,
    n_truth=60, seed=101, mag=(0.015, 0.05), n_rea=24,
):
    """
    Redraw the truth MASS FRACTIONS; for each interior rebuild every sigma from
    that truth's own field and refit, in all four observation models.

    Identical in construction to `G.truth_mc_masses`, so pt1 and pt2 report the
    same kind of number:
      PREDICTED  sig[case]  — analytic (A^T W A)^-1, no sampling at all;
      REALIZED   dev[case]  — errors from actually fitting noisy data, `n_rea`
                              draws per interior (a single draw is half-normal
                              about sigma with 76% scatter, which would swamp
                              the interior-to-interior spread the figure shows).
    The two are computed independently, so comparing them is a real consistency
    test.  All four cases get fresh generators on the same seed, so they see the
    same realization of the SH noise.

    beta_tilde = 1 - sum(beta) is DERIVED, so its variance is 1^T C 1 and its
    error is minus the sum of the anomaly errors — never a free parameter.
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
            bulk_sig[k][i] = np.sqrt(one @ C @ one)
            r = np.random.default_rng(7 + i)
            e = np.array([G.ls_fit_once(blocks, b, r) - b for _ in range(n_rea)])
            dev[k][i] = e
            dev_bulk[k][i] = -e.sum(axis=1)
    return dict(betas=betas, sig=sig, bulk_sig=bulk_sig, dev=dev,
                dev_bulk=dev_bulk, cases=keys, n_rea=n_rea)


def _jitter_inside(p, spread, V, F, tm, rng, n_try=200):
    """A truth position drawn uniformly in a ball about `p`, kept inside the body."""
    for _ in range(n_try):
        u = rng.normal(size=3)
        q = p + spread * rng.uniform() ** (1 / 3) * u / np.linalg.norm(u)
        if G.inside_body(tm, V, F, q[None, :])[0]:
            return q
    return p.copy()


def truth_mc_position_net(
    P, f_true, net, bulk, ch_modes, Lmax, Rref, eps, V, F, tm, n_cyl,
    n_truth=25, seed=202, spread=0.06, pos_bounds=None,
):
    """
    Redraw the truth POSITIONS of every anomaly and refit each one, SH-only and
    with the full network.  One noisy fit per interior — the loop over interiors
    already supplies the sample, exactly as in pt1's position experiment.
    Returns per-anomaly arrays of the realized error, (n_truth, n_anom).
    """
    rng = np.random.default_rng(seed)
    pre = precompute_ch(P, net, ch_modes, bulk)
    pinv_list = [q["pinv"] for q in pre]
    errA = np.empty((n_truth, len(P)))
    errN = np.empty_like(errA)
    for i in range(n_truth):
        Pi = np.array([_jitter_inside(p, spread, V, F, tm, rng) for p in P])
        sig_sh = G.od_sigma(G.stokes_total(f_true, Pi, bulk, 2, Lmax, Rref), eps)
        sig_ch = [
            G.od_sigma(q["pinv"] @ G.field_total(f_true, Pi, q["obs"], bulk), eps)
            for q in pre
        ]
        for j in range(len(P)):
            for use_net, out in ((False, errA), (True, errN)):
                c = position_mc_net(
                    j, Pi, f_true, net, ch_modes, Lmax, Rref, sig_sh, sig_ch,
                    use_net, bulk, bounds=pos_bounds, n_mc=1,
                    seed=seed + 1000 * i, pinv_list=pinv_list,
                )
                out[i, j] = float(np.linalg.norm(c[0] - Pi[j]))
    return errA, errN


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
        print(f"  Brillouin R* = {Rb:.3f} LU | {n_cyl} network cylinders | "
              f"{len(P)} anomalies")
        print(f"  BULK: constant-density polyhedron, β̃ = 1 − Σβ = {beta_bulk:.3f}"
              " of M*")
        print(f"  weights: OD-like σ_i = {eps}·|coeff_i| (floor 10% of RMS); the "
              "inner Φ-to-field fit is unweighted")
        print("  CH cylinder sites (farthest-point order; all enter the joint fit):")
        for i_c, c in enumerate(net):
            tag = ("   <- the 1-CH case" if i_c == 0 else
                   "   <- added by the 2-CH case" if i_c == 1 else "")
            print(f"    C{i_c} {_axis_label(c['dir']):>3s}  surface="
                  f"{np.round(c['surf'], 3)}  |r|={np.linalg.norm(c['surf']):.2f}"
                  + tag)

    # ── EXPERIMENT A — MASS FRACTIONS over TRUTH INTERIORS ──────────────────
    tmm = truth_mc_masses_net(
        P, net, bulk, ch_modes, Lmax_sh, Rref, eps, n_cyl, c0, c1,
        n_truth=n_truth_m, seed=seed_mass, mag=truth_mag, n_rea=n_rea,
    )

    # ── EXPERIMENT B — POSITIONS over TRUTH INTERIORS ───────────────────────
    pos_bounds = (V.min(0) - 0.05, V.max(0) + 0.05)  # keep the solver on the body
    errA, errN = truth_mc_position_net(
        P, f_true, net, bulk, ch_modes, Lmax_sh, Rref, eps, V, F, tm, n_cyl,
        n_truth=n_truth_p, seed=seed_pos, spread=pos_spread,
        pos_bounds=pos_bounds,
    )

    # a single nominal interior, for the geometry figure's budget panel
    a_min, d_surf, d_obs, b_max = G.admissibility(
        P, f_true, beta_bulk, bulk.volume, np.vstack([c["obs"] for c in net]), tm
    )

    res = dict(
        V=V, F=F, Rb=Rb, net=net, names=names, P=P, f_true=f_true, bulk=bulk,
        beta_bulk=beta_bulk, n_cyl=n_cyl, cases=tmm["cases"], truth_mc=tmm,
        pos_errA=errA, pos_errN=errN, pos_spread=pos_spread,
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
    print(f"\n  TABLE 0 — physical admissibility of the truth anomalies "
          f"(excess ceiling Δρ/ρ = {G.EXCESS_CONTRAST:.2f})")
    print(f"  {'anomaly':16s} {'β':>8} {'a_min':>8} {'to surf':>8} {'to obs':>8}"
          f" {'β_max':>9}  verdict")
    for k, nm in enumerate(names):
        v = ("breaches the surface" if adm["a_min"][k] > adm["d_surf"][k]
             else "field points inside it" if adm["a_min"][k] > adm["d_obs"][k]
             else "buried, clear of the data — exact")
        print(f"  {nm:16s} {ft[k]:+8.3f} {adm['a_min'][k]:8.3f} "
              f"{adm['d_surf'][k]:8.3f} {adm['d_obs'][k]:8.3f} "
              f"{adm['b_max'][k]:+9.4f}  {v}")

    # ── TABLE 1 — mass-fraction uncertainty per case ────────────────────────
    print(f"\n  TABLE 1 — mass-fraction 1σ, {len(tmm['betas'])} truth interiors,"
          " median over interiors")
    print(f"  {'anomaly':16s} " + " ".join(f"{k:>18s}" for k in cases)
          + f" {'net gain':>9}")
    for i, nm in enumerate(names):
        row = [np.median(sig[k][:, i]) for k in cases]
        print(f"  {nm:16s} " + " ".join(f"{v:18.2e}" for v in row)
              + f" {row[0] / row[-1]:8.1f}×")
    print(f"  {'-' * 74}")
    row = [np.median(bsig[k]) for k in cases]
    print(f"  {'BODY β̃ = 1−Σβ':16s} " + " ".join(f"{v:18.2e}" for v in row)
          + f" {row[0] / row[-1]:8.1f}×")
    g = np.median(sig[cases[0]] / sig[net_k], axis=0)
    print(f"  network gain vs SH:  min {g.min():.0f}×   median "
          f"{np.median(g):.0f}×   max {g.max():.0f}×")

    # ── TABLE 1b — does the analytic covariance predict the error made? ─────
    print(f"\n  TABLE 1b — covariance consistency, "
          f"{len(tmm['betas']) * tmm['n_rea']} noisy fits per case: "
          "realized RMS(estimate − truth) vs predicted 1σ")
    print(f"  {'anomaly':16s} " + " ".join(f"{'realized/pred ' + k.split('(')[0]:>22s}"
                                           for k in (cases[0], net_k)))
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
    eA, eN = res["pos_errA"], res["pos_errN"]
    print(f"\n  TABLE 2 — anomaly position, {len(eA)} truth interiors, RMS error"
          " [LU]")
    print(f"  {'anomaly':16s} {'SH only':>12} {'network':>12} {'gain':>8}")
    for i, nm in enumerate(names):
        a, b = rms(eA[:, i]), rms(eN[:, i])
        print(f"  {nm:16s} {a:12.2e} {b:12.2e} {a / b:7.1f}×")

    # ── LaTeX bodies ───────────────────────────────────────────────────────
    print(f"\n{'-' * 74}\n  LaTeX tabular bodies\n{'-' * 74}")
    print("  % Table 1 — mass-fraction 1 sigma per observation model")
    for i, nm in enumerate(names):
        row = [np.median(sig[k][:, i]) for k in cases]
        print(f"  {nm} & " + " & ".join(f"${_tex_num(v)}$" for v in row)
              + rf" & ${row[0] / row[-1]:.1f}$ \\")
    row = [np.median(bsig[k]) for k in cases]
    print(r"  body $\tilde\beta$ & " + " & ".join(f"${_tex_num(v)}$" for v in row)
          + rf" & ${row[0] / row[-1]:.1f}$ \\")
    print("  % Table 2 — position RMS error [LU]")
    for i, nm in enumerate(names):
        a, b = rms(eA[:, i]), rms(eN[:, i])
        print(f"  {nm} & ${_tex_num(a)}$ & ${_tex_num(b)}$ & ${a / b:.1f}$ \\\\")


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
    ft, beta_bulk = res["f_true"], res["beta_bulk"]
    net_k = cases[-1]
    rms = lambda M: np.sqrt(np.mean(np.asarray(M) ** 2, axis=0))
    lab = [n.split()[0] for n in names]          # same split pt1 uses

    # ---- FIG 1: geometry ---------------------------------------------------
    # Geometry only, at pt1's fig-1 size.  The mass budget lived here as a
    # second panel; the truth beta_j and beta~ are in TABLE 0/1 instead.
    fig = plt.figure(figsize=(8.6, 7.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    step = max(1, len(F) // 9000)
    ax.add_collection3d(Poly3DCollection(V[F[::step]], alpha=0.10,
                        facecolor="#9ecae1", edgecolor="0.6", linewidths=0.1))
    for i_c, c in enumerate(net):
        G.draw_cylinder(ax, c["cyl"])
        t = c["surf"] + 0.42 * c["dir"]   # past the far end, never on the tube
        ax.text(t[0], t[1], t[2], f"C{i_c} ({_axis_label(c['dir'])})",
                fontsize=8, color="#8b0000", fontweight="bold", ha="center",
                bbox=dict(fc="white", ec="0.7", alpha=0.8, pad=1.2, lw=0.4))
    for nm, q, b in zip(names, P, ft):
        ax.scatter(q[0], q[1], q[2], s=90, depthshade=False, edgecolor="k",
                   color=COLOR[0] if b > 0 else COLOR[2])
        ax.text(q[0], q[1], q[2], f"  {nm.split()[0]}", fontsize=8)
    ax.plot([], [], color="crimson", lw=2, label="CH cylinders")
    ax.scatter([], [], color=COLOR[0], label=r"anomaly $\beta_j>0$")
    ax.scatter([], [], color=COLOR[2], label=r"anomaly $\beta_j<0$")
    ax.set_title("Interior: constant-density body, anomalies, and the CH network")
    ax.set_xlabel("x [LU]"); ax.set_ylabel("y [LU]"); ax.set_zlabel("z [LU]")
    G.set_axes_true_shape(ax, np.vstack([V] + [G.cylinder_hull(c["cyl"])
                                               for c in net]))
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt2_fig1_geometry.pdf"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 2: mass-fraction recovery, four observation models ------------
    # Bars are the error actually made (RMS over truth interiors x noise draws);
    # the black ticks are the analytic 1-sigma those same interiors predict.
    # They are computed independently, so bar == tick is a real check.
    fig, ax = plt.subplots(figsize=(14, 5.8))
    x = np.arange(len(P) + 1, dtype=float) * 1.25
    w = 0.22
    colmap = dict(zip(cases, (COLOR[2], COLOR[1], COLOR[3], COLOR[0])))
    offs = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    real, pred = {}, {}
    for k in cases:
        real[k] = np.append(rms(tmm["dev"][k].reshape(-1, len(P))),
                            rms(tmm["dev_bulk"][k].ravel()))
        pred[k] = np.append(rms(tmm["sig"][k]), rms(tmm["bulk_sig"][k]))
    for k, off in zip(cases, offs):
        ax.bar(x + off, real[k], w, color=colmap[k], edgecolor="k", label=k)
        ax.plot(x + off, pred[k], "_", ms=9, mew=2.0, color="k", ls="none",
                zorder=6)
    ax.plot([], [], "_", ms=9, mew=2.0, color="k", ls="none",
            label=r"analytic 1$\sigma$ (predicted)")
    ax.axvline(x[-1] - 0.62, color="0.6", ls="--", lw=1.2)
    gains = real[cases[0]] / real[net_k]
    for i in range(len(x)):
        ax.text(x[i] + 1.5 * w, real[net_k][i] * 1.15, rf"${gains[i]:.0f}\times$",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=COLOR[0])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(lab + [r"BODY $\tilde\beta$"], fontsize=9)
    ax.set_ylabel(r"Mass-fraction Error  $\beta_j$  [-]  (MC)")
    ax.set_title("Each added near-surface patch constrains the anomalies it "
                 "covers;\nthe full network constrains all of them, and the body "
                 "fraction with them")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt2_fig2_massratio.pdf"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 3: position recovery, per anomaly + the distribution ----------
    eA, eN = res["pos_errA"], res["pos_errN"]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    xp = np.arange(len(P), dtype=float)
    ax.bar(xp - 0.2, rms(eA), 0.4, color=COLOR[2], edgecolor="k", label="SH only")
    ax.bar(xp + 0.2, rms(eN), 0.4, color=COLOR[0], edgecolor="k",
           label="SH + CH network")
    g = rms(eA) / rms(eN)
    for i in range(len(P)):
        ax.text(xp[i] + 0.2, rms(eN)[i] * 1.15, rf"${g[i]:.0f}\times$",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=COLOR[0])
    ax.set_yscale("log")
    ax.set_xticks(xp)
    ax.set_xticklabels(lab, fontsize=9)
    ax.set_ylabel("Position RMS Error over Truth Interiors  [LU]  (MC)")
    ax.set_title("Position recovery of arbitrary anomalies")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    # (b) the distribution over interiors, pooled — same treatment as pt1 fig 3
    ax = axes[1]
    fa, fn = eA.ravel(), eN.ravel()
    bins = np.logspace(np.log10(min(fa.min(), fn.min()) * 0.8),
                       np.log10(max(fa.max(), fn.max()) * 1.2), 26)
    ax.hist(fa, bins=bins, color=COLOR[2], alpha=0.75, edgecolor="k", lw=0.5,
            label="SH only")
    ax.hist(fn, bins=bins, color=COLOR[0], alpha=0.75, edgecolor="k", lw=0.5,
            label="SH + CH network")
    G.lognormal_overlay(ax, fa, bins, "k", ls="-", name="SH")
    G.lognormal_overlay(ax, fn, bins, "k", ls="--", name="SH + CH")
    ax.set_xscale("log")
    ax.set_xlabel("Position Error, One Fit per Truth Interior  [LU]")
    ax.set_ylabel(f"Fits  (of {fa.size})  [-]")
    ax.set_title("Distribution over truth interiors and anomalies")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(outdir, "global_pt2_fig3_position.pdf"),
                dpi=180, bbox_inches="tight")
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
    print("\nSaved: Images/global_pt2_fig1_geometry.pdf, "
          "global_pt2_fig2_massratio.pdf, global_pt2_fig3_position.pdf")
    print("Done.")
