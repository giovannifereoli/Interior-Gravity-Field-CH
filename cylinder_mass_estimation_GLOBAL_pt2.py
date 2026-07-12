"""
Part 2 — A NETWORK of cylindrical harmonics for ARBITRARY interior mascons
==========================================================================
Author: Giovanni Fereoli / experiment build

Question (different from pt1)
-----------------------------
pt1 showed that ONE near-surface CH cylinder resolves ONE mascon sitting under
it.  But a real body has many mass concentrations at unknown, scattered
locations.  A single cylinder only helps whatever is beneath it.

    Does SH + a NETWORK of CH cylinders (near-surface data all around the body)
    let us estimate the mass ratio and position of MANY, arbitrarily placed
    mascons — where SH alone, and SH + a single cylinder, cannot?

Idea
----
Place ~6 mascons at scattered interior locations of Eros.  Build a NETWORK of
CH cylinders on the surface (each a patch of near-surface / low-altitude data,
represented by its own Bessel–Fourier expansion).  The cylinders are put on the
parts of the surface that lie INSIDE the Brillouin sphere (the sides / waist),
where exterior SH is weakest and CH converges — the long-axis tips sit on the
Brillouin sphere and are skipped.

Three observation models are compared:
    A  : SH only                    (global Stokes, degree 2..L, + total mass)
    A1 : SH + ONE CH cylinder       (pt1-style, single near-surface patch)
    AN : SH + the CH NETWORK        (all cylinders' coefficients)

Because a mascon's localized signature is captured by whatever cylinder is near
it, the NETWORK constrains every mascon; the single cylinder constrains only its
neighbour; SH alone leaves them degenerate.

Both observables are linear in the mascon masses (mass-ratio experiment =
linear LS), and nonlinear in a mascon's position (position experiment = TRF).
All heavy machinery (Legendre, Stokes design, Bessel basis, fits) is reused
from `cylinder_mass_estimation_GLOBAL` (imported as G).

Units: Eros normalized (LU), total mass M = 1, G = 1.
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
    Scatter mascons for the experiment: one near-surface mascon under each of
    the first (n_cyl-1) cylinders, plus one DEEP core mascon (the hard case that
    no near-surface patch sits close to).  Returns names, positions, mass ratios.
    """
    names, pos = [], []
    for i, c in enumerate(net[:-1]):
        names.append(f"m{i} near {_axis_label(c['dir'])}")
        pos.append(0.72 * c["surf"])  # just inside the surface, under cylinder i
    names.append("core (deep)")
    pos.append(np.array([0.08, 0.0, 0.0]))
    pos = np.array(pos)
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.6, 1.4, len(pos))
    f = f / f.sum()  # mass ratios sum to 1
    return names, pos, f


def _axis_label(d):
    ax = "xyz"[int(np.argmax(np.abs(d)))]
    return ("+" if d[np.argmax(np.abs(d))] > 0 else "-") + ax


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVABLE BLOCKS
# ═══════════════════════════════════════════════════════════════════════════


def ch_blocks_for(P, net, ch_modes, eps, f_true):
    """(A_ch, sigma) for every cylinder in the network — mascon→CH coefficients."""
    blocks = []
    for c in net:
        A_ch = G.ch_coeff_design(P, c["obs"], c["cyl"], ch_modes)
        sig = eps * float(np.sqrt(np.mean((A_ch @ f_true) ** 2)))
        blocks.append((A_ch, sig))
    return blocks


# ═══════════════════════════════════════════════════════════════════════════
# POSITION FIT (one mascon free, masses + other positions fixed) — network
# ═══════════════════════════════════════════════════════════════════════════


def _pos_forward_net(posj, j, P, masses, Lmax, Rref, ch_data):
    positions = P.copy()
    positions[j] = posj
    y_sh = np.zeros(len(G.sh_stokes_of_point(positions[0], 2, Lmax, Rref)))
    for mk, pk in zip(masses, positions):
        y_sh = y_sh + mk * G.sh_stokes_of_point(pk, 2, Lmax, Rref)
    blocks = [y_sh]
    for pinvPhi, obs in ch_data:  # each cylinder's CH coefficients
        field = np.zeros(4 * len(obs))
        for mk, pk in zip(masses, positions):
            field = field + mk * G.point_mass_field(pk, obs)
        blocks.append(pinvPhi @ field)
    return blocks


def position_mc_net(
    j, P, f_true, net, ch_modes, Lmax, Rref, sig_sh, sig_ch_list, use_net,
    bounds=None, n_mc=60, seed=13, start_jitter=0.02,
):
    """
    Monte-Carlo NONLINEAR least-squares (scipy TRF) recovery of mascon j's
    position; masses & other positions fixed.  `use_net`=False → SH only;
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
            ch_data.append((np.linalg.pinv(Phi), c["obs"]))
            sig_blocks.append(s)
    p_true = P[j].copy()
    truth = _pos_forward_net(p_true, j, P, f_true, Lmax, Rref, ch_data)
    if bounds is None:
        bounds = (-np.inf, np.inf)

    def resid(posj, data):
        model = _pos_forward_net(posj, j, P, f_true, Lmax, Rref, ch_data)
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


def run(Lmax_sh=6, eps=0.02, sig_M=1e-3, ch_modes=(6, 6), n_cyl=6,
        n_mc=300, n_mc_pos=60, outdir="Images", verbose=True):
    V, F, tm, Rb = G.load_eros()
    Rref = Rb
    net = build_network(V, F, tm, n_cyl=n_cyl)
    names, P, f_true = place_mascons(net)
    n_m = len(P)

    if verbose:
        print(SEP)
        print("  PART 2 — SH + CH NETWORK for arbitrary interior mascons (Eros)")
        print(SEP)
        print(f"  Brillouin R* = {Rb:.3f} LU | {n_cyl} network cylinders | "
              f"{n_m} mascons")
        for nm, p, fr in zip(names, P, f_true):
            print(f"    {nm:16s} p={np.round(p,3)}  f={fr:.3f}  |r|={np.linalg.norm(p):.2f}")

    # ── observable design blocks ────────────────────────────────────────────
    A_sh = G.A_stokes(P, 2, Lmax_sh, Rref)
    A_M = np.ones((1, n_m))
    sig_sh = eps * float(np.sqrt(np.mean((A_sh @ f_true) ** 2)))
    ch_blocks = ch_blocks_for(P, net, ch_modes, eps, f_true)   # per-cylinder
    sig_ch_list = [s for _, s in ch_blocks]

    base = [(A_sh, sig_sh), (A_M, sig_M)]
    # net[0] and net[1] are the two most-opposite cylinders (farthest-point order)
    cases = {
        "SH only": base,
        "SH + 1 CH": base + [ch_blocks[0]],
        "SH + 2 CH (opposite)": base + [ch_blocks[0], ch_blocks[1]],
        f"SH + {n_cyl}-CH network": base + ch_blocks,
    }

    # ── EXPERIMENT A — MASS RATIOS (positions fixed) ────────────────────────
    if verbose:
        print(f"\n{'-'*72}\n  A) MASS-RATIO recovery (1σ over {n_mc} MC draws)\n{'-'*72}")
        print(f"  {'mascon':16s} " + " ".join(f"{k:>14s}" for k in cases))
    sig_by_case = {}
    for k, blocks in cases.items():
        mc = G.monte_carlo_fit(blocks, f_true, n_mc=n_mc)
        sig_by_case[k] = mc.std(0)
    if verbose:
        for i, nm in enumerate(names):
            print(f"  {nm:16s} " + " ".join(f"{sig_by_case[k][i]:14.2e}" for k in cases))
        gains = sig_by_case["SH only"] / sig_by_case[f"SH + {n_cyl}-CH network"]
        print(f"  network gain vs SH: min={gains.min():.0f}× median={np.median(gains):.0f}× "
              f"max={gains.max():.0f}×")

    # ── EXPERIMENT B — POSITION of arbitrary mascons (masses fixed) ─────────
    if verbose:
        print(f"\n{'-'*72}\n  B) POSITION recovery ({n_mc_pos} MC draws): SH vs {n_cyl}-CH network\n{'-'*72}")
    pos_rms = {"SH only": np.zeros(n_m), "network": np.zeros(n_m)}
    pos_clouds = {}
    pos_bounds = (V.min(0) - 0.05, V.max(0) + 0.05)  # keep the solver on the body
    for j in range(n_m):
        cloudA = position_mc_net(j, P, f_true, net, ch_modes, Lmax_sh, Rref,
                                 sig_sh, sig_ch_list, use_net=False,
                                 bounds=pos_bounds, n_mc=n_mc_pos)
        cloudN = position_mc_net(j, P, f_true, net, ch_modes, Lmax_sh, Rref,
                                 sig_sh, sig_ch_list, use_net=True,
                                 bounds=pos_bounds, n_mc=n_mc_pos)
        pos_rms["SH only"][j] = np.sqrt(np.mean(np.sum((cloudA - P[j]) ** 2, axis=1)))
        pos_rms["network"][j] = np.sqrt(np.mean(np.sum((cloudN - P[j]) ** 2, axis=1)))
        pos_clouds[j] = (cloudA, cloudN)
        if verbose:
            print(f"    {names[j]:16s} SH={pos_rms['SH only'][j]:.2e} LU  "
                  f"network={pos_rms['network'][j]:.2e} LU  "
                  f"→ {pos_rms['SH only'][j]/pos_rms['network'][j]:.0f}× tighter")

    res = dict(V=V, F=F, Rb=Rb, net=net, names=names, P=P, f_true=f_true,
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

    # ---- FIG 1: geometry — Eros + mascons + CH network ---------------------
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, len(F) // 9000)
    ax.add_collection3d(Poly3DCollection(V[F[::step]], alpha=0.10,
                        facecolor="#9ecae1", edgecolor="0.6", linewidths=0.1))
    for c in net:
        o = c["obs"]
        ax.scatter(o[:, 0], o[:, 1], o[:, 2], s=3, color="crimson", alpha=0.5)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=90, color="k", depthshade=False)
    for nm, p in zip(names, P):
        ax.text(p[0], p[1], p[2], "  " + nm.split()[0], fontsize=8)
    ax.scatter([], [], color="crimson", label=f"{n_cyl} CH cylinders")
    ax.scatter([], [], color="k", label="mascons")
    ax.set_title(f"Eros + {len(P)} scattered mascons\n+ CH network ({n_cyl} cylinders)")
    ax.set_xlabel("x [LU]"); ax.set_ylabel("y [LU]"); ax.set_zlabel("z [LU]")
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt2_fig1_geometry.png"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 2: mass-ratio uncertainty per mascon, 4 cases -----------------
    fig, ax = plt.subplots(figsize=(14, 5.8))
    cases = res["cases"]  # SH / 1CH / 2CH / network
    sig = res["sig_by_case"]
    x = np.arange(len(P))
    w = 0.2
    colmap = {cases[0]: COLOR[2], cases[1]: COLOR[1], cases[2]: COLOR[3],
              cases[3]: COLOR[0]}
    offs = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    for c, off in zip(cases, offs):
        ax.bar(x + off, sig[c], w, color=colmap[c], edgecolor="k", label=c)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax.set_ylabel(r"mass-ratio 1$\sigma$ uncertainty $\sigma_f$")
    ax.set_title("Mass-ratio recovery: SH → +1 CH → +2 CH (opposite) → CH network\n"
                 "(each added cylinder constrains the mascons it covers; "
                 "the full network constrains ALL)")
    gains = sig[cases[0]] / sig[cases[3]]
    for i in range(len(P)):
        ax.text(x[i] + 1.5 * w, sig[cases[3]][i], f"{gains[i]:.0f}×", ha="center",
                va="bottom", fontsize=8, fontweight="bold")
    ax.grid(True, axis="y", which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt2_fig2_massratio.png"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 3: position recovery per mascon, SH vs network ----------------
    fig, ax = plt.subplots(figsize=(13, 5.6))
    pr = res["pos_rms"]
    ax.bar(x - 0.2, pr["SH only"], 0.4, color=COLOR[2], edgecolor="k", label="SH only")
    ax.bar(x + 0.2, pr["network"], 0.4, color=COLOR[0], edgecolor="k",
           label=f"SH + {n_cyl}-CH network")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax.set_ylabel("position RMS error [LU]")
    ax.set_title("Position recovery of arbitrary mascons: SH vs CH network")
    g = pr["SH only"] / pr["network"]
    for i in range(len(P)):
        ax.text(x[i] + 0.2, pr["network"][i], f"{g[i]:.0f}×", ha="center",
                va="bottom", fontsize=8, fontweight="bold")
    ax.grid(True, axis="y", which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt2_fig3_position.png"),
                dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    res = run(
        Lmax_sh=6,
        eps=0.02,
        sig_M=1e-3,
        ch_modes=(6, 6),
        n_cyl=6,
        n_mc=300,
        n_mc_pos=60,
        outdir="Images",
        verbose=True,
    )
    print("\nSaved: Images/global_pt2_fig1_geometry.png, "
          "global_pt2_fig2_massratio.png, global_pt2_fig3_position.png")
    print("Done.")
