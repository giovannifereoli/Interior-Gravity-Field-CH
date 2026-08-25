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
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import cylinder_mass_estimation_GLOBAL as G  # reuse pt1 machinery

COLOR = G.COLOR
mpl.rcParams.update({"axes.prop_cycle": mpl.cycler(color=COLOR),
                     "font.family": "serif", "figure.dpi": 110})
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
    names.append("deep (core)")
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
    bounds=None, n_mc=60, seed=13, start_jitter=0.02,
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
        for c, s in zip(net, sig_ch_list):
            Phi = G.cyl_basis(c["cyl"], c["obs"], *ch_modes)
            ch_data.append((G.ch_pinv(Phi), c["obs"]))
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
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════


def run(Lmax_sh=6, eps=0.02, ch_modes=(6, 6), n_cyl=6,
        n_mc=300, n_mc_pos=60, outdir="Images", verbose=True):
    V, F, tm, Rb = G.load_eros()
    Rref = Rb
    net = build_network(V, F, tm, n_cyl=n_cyl)
    names, P, f_true = place_mascons(net)
    bulk = G.Bulk(V, F)
    beta_bulk = G.bulk_fraction(f_true)
    n_m = len(P)

    if verbose:
        print(SEP)
        print("  PART 2 — SH + CH NETWORK for arbitrary interior anomalies (Eros)")
        print(SEP)
        print(f"  Brillouin R* = {Rb:.3f} LU | {n_cyl} network cylinders | "
              f"{n_m} anomalies")
        print(f"  BULK: constant-density polyhedron, β̃ = 1 − Σβ = {beta_bulk:.3f} of M*")
        print(f"  weights: OD-like σ_i = {eps}·|coeff_i| (floor 10% of RMS); the inner "
              f"Φ-to-field fit is unweighted")
        for nm, p, fr in zip(names, P, f_true):
            print(f"    {nm:16s} p={np.round(p,3)}  β={fr:+.3f}  |r|={np.linalg.norm(p):.2f}")
        print(f"  CH cylinder sites (farthest-point order; every one of them "
              f"enters the joint fit):")
        for i_c, c in enumerate(net):
            print(f"    C{i_c} {_axis_label(c['dir']):>3s}  surface="
                  f"{np.round(c['surf'], 3)}  |r|={np.linalg.norm(c['surf']):.2f}"
                  + ("   <- used by the 1-CH case" if i_c == 0 else
                     "   <- added by the 2-CH case" if i_c == 1 else ""))

    # ── observable design blocks (all of them CONTRASTS vs the bulk) ────────
    A_sh = G.A_stokes_contrast(P, bulk, 2, Lmax_sh, Rref)
    sig_sh = G.od_sigma(G.stokes_total(f_true, P, bulk, 2, Lmax_sh, Rref), eps)
    ch_blocks = ch_blocks_for(P, net, ch_modes, eps, f_true, bulk)  # per-cylinder
    sig_ch_list = [s for _, s in ch_blocks]

    base = [(A_sh, sig_sh)]
    # Farthest-point order anchors net[0] at the −z extreme and puts net[1]
    # farthest from it, so these two are the most widely separated pair.  Name
    # the reduced configurations by WHERE their cylinders actually sit, since
    # which patch is included is the whole point of the comparison.
    c0, c1 = _axis_label(net[0]["dir"]), _axis_label(net[1]["dir"])
    cases = {
        "SH only": base,
        f"SH + 1 CH ({c0})": base + [ch_blocks[0]],
        f"SH + 2 CH ({c0},{c1})": base + [ch_blocks[0], ch_blocks[1]],
        f"SH + {n_cyl}-CH network": base + ch_blocks,
    }

    # ── EXPERIMENT A — MASS FRACTIONS (positions fixed) ─────────────────────
    if verbose:
        print(f"\n{'-'*72}\n  A) MASS-FRACTION recovery (1σ on β_j over {n_mc} MC "
              f"draws)\n{'-'*72}")
        print(f"  {'anomaly':16s} " + " ".join(f"{k:>14s}" for k in cases))
    # The body fraction is ESTIMATED too — not as a free parameter, but as the
    # derived quantity β̃ = 1 − Σβ evaluated on every Monte-Carlo draw.  Mass
    # conservation makes it a function of the anomalies, so it inherits their
    # covariance: σ_β̃ = sqrt(1ᵀ Cov(β) 1), which is what the draws measure.
    sig_by_case, bulk_by_case = {}, {}
    for k, blocks in cases.items():
        mc = G.monte_carlo_fit(blocks, f_true, n_mc=n_mc)
        sig_by_case[k] = mc.std(0)
        bt = 1.0 - mc.sum(1)
        bulk_by_case[k] = (float(bt.mean()), float(bt.std()))
    if verbose:
        for i, nm in enumerate(names):
            print(f"  {nm:16s} " + " ".join(f"{sig_by_case[k][i]:14.2e}" for k in cases))
        print(f"  {'-'*72}")
        print(f"  {'BODY β̃ =1−Σβ':16s} "
              + " ".join(f"{bulk_by_case[k][1]:14.2e}" for k in cases))
        net_k = f"SH + {n_cyl}-CH network"
        print(f"    (truth β̃ = {beta_bulk:.4f};  recovered {bulk_by_case[net_k][0]:.4f}"
              f" ± {bulk_by_case[net_k][1]:.4f} with the full network)")
        gains = sig_by_case["SH only"] / sig_by_case[f"SH + {n_cyl}-CH network"]
        print(f"  network gain vs SH: min={gains.min():.0f}× median={np.median(gains):.0f}× "
              f"max={gains.max():.0f}×")

    # ── EXPERIMENT B — POSITION of arbitrary anomalies (masses fixed) ───────
    if verbose:
        print(f"\n{'-'*72}\n  B) POSITION recovery ({n_mc_pos} MC draws): SH vs {n_cyl}-CH network\n{'-'*72}")
    pos_rms = {"SH only": np.zeros(n_m), "network": np.zeros(n_m)}
    pos_clouds = {}
    pos_bounds = (V.min(0) - 0.05, V.max(0) + 0.05)  # keep the solver on the body
    for j in range(n_m):
        cloudA = position_mc_net(j, P, f_true, net, ch_modes, Lmax_sh, Rref,
                                 sig_sh, sig_ch_list, False, bulk,
                                 bounds=pos_bounds, n_mc=n_mc_pos)
        cloudN = position_mc_net(j, P, f_true, net, ch_modes, Lmax_sh, Rref,
                                 sig_sh, sig_ch_list, True, bulk,
                                 bounds=pos_bounds, n_mc=n_mc_pos)
        pos_rms["SH only"][j] = np.sqrt(np.mean(np.sum((cloudA - P[j]) ** 2, axis=1)))
        pos_rms["network"][j] = np.sqrt(np.mean(np.sum((cloudN - P[j]) ** 2, axis=1)))
        pos_clouds[j] = (cloudA, cloudN)
        if verbose:
            print(f"    {names[j]:16s} SH={pos_rms['SH only'][j]:.2e} LU  "
                  f"network={pos_rms['network'][j]:.2e} LU  "
                  f"→ {pos_rms['SH only'][j]/pos_rms['network'][j]:.0f}× tighter")

    res = dict(V=V, F=F, Rb=Rb, net=net, names=names, P=P, f_true=f_true,
               bulk=bulk, beta_bulk=beta_bulk, bulk_by_case=bulk_by_case,
               n_cyl=n_cyl, cases=list(cases.keys()), sig_by_case=sig_by_case,
               pos_rms=pos_rms, pos_clouds=pos_clouds)
    make_plots(res, outdir=outdir)
    return res


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════


def make_plots(res, outdir="Images"):
    os.makedirs(outdir, exist_ok=True)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    V, F, P, net, names = res["V"], res["F"], res["P"], res["net"], res["names"]
    n_cyl = res["n_cyl"]

    # ---- FIG 1: geometry + the truth mass budget ---------------------------
    ft, beta_bulk = res["f_true"], res["beta_bulk"]
    fig = plt.figure(figsize=(15.5, 6.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    step = max(1, len(F) // 9000)
    ax.add_collection3d(Poly3DCollection(V[F[::step]], alpha=0.10,
                        facecolor="#9ecae1", edgecolor="0.6", linewidths=0.1))
    for i_c, c in enumerate(net):
        G.draw_cylinder(ax, c["cyl"])
        # past the far end of the cylinder, so the label never sits on it
        t = c["surf"] + 0.42 * c["dir"]
        ax.text(t[0], t[1], t[2], f"C{i_c} ({_axis_label(c['dir'])})",
                fontsize=8, color="#8b0000", fontweight="bold", ha="center",
                bbox=dict(fc="white", ec="0.7", alpha=0.8, pad=1.2, lw=0.4))
    for nm, p, b in zip(names, P, ft):   # colour by SIGN, label with β_j
        ax.scatter(p[0], p[1], p[2], s=90, depthshade=False, edgecolor="k",
                   color=G.COLOR[0] if b > 0 else G.COLOR[2])
        ax.text(p[0], p[1], p[2], f"  {nm.split()[0]} ({b:+.3f})", fontsize=8)
    ax.plot([], [], color="crimson", lw=2, label=f"{n_cyl} CH cylinders")
    ax.scatter([], [], color=G.COLOR[0], label=r"anomaly $\beta_j>0$")
    ax.scatter([], [], color=G.COLOR[2], label=r"anomaly $\beta_j<0$")
    ax.set_title("Interior = constant-density BODY "
                 f"($\\tilde\\beta$ = {beta_bulk:.3f})\n"
                 f"+ {len(P)} anomalies + CH network ({n_cyl} cylinders)")
    ax.set_xlabel("x [LU]"); ax.set_ylabel("y [LU]"); ax.set_zlabel("z [LU]")
    G.set_axes_true_shape(ax, np.vstack([V] + [G.cylinder_hull(c["cyl"])
                                              for c in net]))
    ax.legend(fontsize=9, loc="upper left")

    net_key = f"SH + {n_cyl}-CH network"
    G.draw_mass_budget(fig.add_subplot(1, 2, 2), names, ft, beta_bulk,
                       recovered=res["bulk_by_case"][net_key],
                       rec_label=net_key)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt2_fig1_geometry.pdf"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 2: mass-fraction uncertainty per anomaly, 4 cases -------------
    fig, ax = plt.subplots(figsize=(14, 5.8))
    cases = res["cases"]  # SH / 1CH / 2CH / network
    sig = res["sig_by_case"]
    nb = res["bulk_by_case"]
    # the derived body fraction is shown alongside the anomalies it comes from
    vals = {c: np.append(sig[c], nb[c][1]) for c in cases}
    x = np.arange(len(P) + 1)
    w = 0.2
    colmap = {cases[0]: COLOR[2], cases[1]: COLOR[1], cases[2]: COLOR[3],
              cases[3]: COLOR[0]}
    offs = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    for c, off in zip(cases, offs):
        ax.bar(x + off, vals[c], w, color=colmap[c], edgecolor="k", label=c)
    ax.axvline(len(P) - 0.5, color="0.6", ls="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names]
                       + ["BODY\n$\\tilde\\beta$ (derived)"], fontsize=8)
    ax.set_ylabel(r"mass-fraction 1$\sigma$ uncertainty $\sigma_\beta$")
    ax.set_title("Mass-fraction recovery: SH → +1 CH → +2 CH → full CH network\n"
                 "(each added patch constrains the anomalies it covers; the full "
                 "network constrains ALL — and the body fraction with them)")
    gains = vals[cases[0]] / vals[cases[3]]
    for i in range(len(x)):
        ax.text(x[i] + 1.5 * w, vals[cases[3]][i], f"{gains[i]:.0f}×", ha="center",
                va="bottom", fontsize=8, fontweight="bold")
    ax.grid(True, axis="y", which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt2_fig2_massratio.pdf"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 3: position recovery per anomaly, SH vs network ---------------
    fig, ax = plt.subplots(figsize=(13, 5.6))
    pr = res["pos_rms"]
    ax.bar(x - 0.2, pr["SH only"], 0.4, color=COLOR[2], edgecolor="k", label="SH only")
    ax.bar(x + 0.2, pr["network"], 0.4, color=COLOR[0], edgecolor="k",
           label=f"SH + {n_cyl}-CH network")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax.set_ylabel("position RMS error [LU]")
    ax.set_title("Position recovery of arbitrary anomalies: SH vs CH network")
    g = pr["SH only"] / pr["network"]
    for i in range(len(P)):
        ax.text(x[i] + 0.2, pr["network"][i], f"{g[i]:.0f}×", ha="center",
                va="bottom", fontsize=8, fontweight="bold")
    ax.grid(True, axis="y", which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt2_fig3_position.pdf"),
                dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    res = run(
        Lmax_sh=6,
        eps=0.02,
        ch_modes=(6, 6),
        n_cyl=6,
        n_mc=300,
        n_mc_pos=60,
        outdir="Images",
        verbose=True,
    )
    print("\nSaved: Images/global_pt2_fig1_geometry.pdf, "
          "global_pt2_fig2_massratio.pdf, global_pt2_fig3_position.pdf")
    print("Done.")
