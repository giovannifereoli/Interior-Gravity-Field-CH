"""
Part 2b — COMPONENTS WITH A PURPOSE: placing mass elements to target specific
          gravity coefficients, and asking whether CH can separate them
==============================================================================
Author: Giovanni Fereoli / experiment build

Question (different from pt2)
-----------------------------
pt2 scattered its anomalies at geometrically convenient spots — under cylinder
sites, plus one deep one.  That answers "does the network cover the body?", but
it says nothing about the way interior models are actually built.  Following the
Bennu interior model of Cavale & Scheeres (Icarus 460, 2026, 117260), a real
parameterization places each component WHERE IT IS NEEDED, so that it carries
targeted sensitivity to a particular block of Stokes coefficients:

    B22   equatorial plane, prescribed longitude      -> C22, S22
    B21   prescribed latitude and longitude           -> C21, S21
    BN    on the +z axis, near the surface            -> J3, J4  (C30, C40)
    BS    on the −z axis, near the surface            -> J3, J4
    torus ring in the equatorial plane, at the origin -> J2  (C20)
    core  near the centre                             -> the weakly observable one

Components sit near the EXTERIOR (a fraction of the local surface radius),
because a mass element's contribution to degree n scales as (r/R*)^n: the same
reason the paper puts its boulders at 170 m of Bennu's ~250 m.

    But — and this is the point the paper makes and this script measures —
    "every mass component contributes to all gravity coefficients", so the
    densities "cannot be adjusted independently for individual terms but must be
    solved simultaneously".  Targeting buys sensitivity, not separability.

    So: does adding near-surface CH data BREAK that degeneracy?

Interior model (as in pt1/pt2)
------------------------------
Mass lives in the constant-density polyhedron, scaled by β̃ = 1 − Σβ; the
components carry only the departures from homogeneity, β_j = m_j/M*, positive
for compaction and negative for porosity — the natural reading for a rubble
pile, where these are void/packing variations rather than distinct lithologies.
β̃ is derived, never assigned, and is estimated along with everything else.

Components may be EXTENDED, not just points: each one is a set of sub-points
with weights summing to one, so the torus is a genuine ring rather than a mascon
at the origin.  A point component is the one-sub-point case.

What this script produces
-------------------------
  1. TARGETING MATRIX.  Per unit mass fraction, the share of each Stokes
     coefficient driven by each component.  This is where "B22 targets C22/S22"
     is either verified or falsified — for Eros it is only partly true, and the
     script says which parts.
  2. RECOVERY.  σ_β per component, SH alone vs SH + the CH network, plus the
     derived body fraction β̃.
  3. CORRELATION.  The posterior correlation matrix of the component densities,
     SH alone vs SH + CH.  This is the quantitative form of "must be solved
     simultaneously", and the measure of what the near-surface data buys.

Placement of the CH cylinders is pt2's farthest-point rule, which knows nothing
about where the components are — unlike pt2, where anomalies were deliberately
put under cylinders, nothing here is rigged in the network's favour.

Units: Eros normalized (LU), total mass M* = 1, G = 1.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import cylinder_mass_estimation_GLOBAL as G          # pt1 machinery
import cylinder_mass_estimation_GLOBAL_pt2 as P2     # pt2 network machinery

COLOR = G.COLOR
mpl.rcParams.update({"axes.prop_cycle": mpl.cycler(color=COLOR),
                     "figure.dpi": 110})
SEP = "=" * 78


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — COMPONENTS PLACED TO TARGET SPECIFIC COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════


def surface_radius(tm, V, F, d, r_hi, n_iter=40):
    """
    Distance from the origin to the surface along direction `d`, by bisection on
    "is this point still inside?".  Assumes the body is star-shaped about the
    origin along this ray — true enough for Eros about its centre of mass, and
    only ever used to place a component at a FRACTION of the result.
    """
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    lo, hi = 0.0, float(r_hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if G.inside_body(tm, V, F, (mid * d)[None, :])[0]:
            lo = mid
        else:
            hi = mid
    return lo


def _sph(lat_deg, lon_deg):
    """Unit vector from latitude / longitude in degrees."""
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def place_components(V, F, tm, Rb, lon22=30.0, lat21=45.0, lon21=120.0,
                     frac=0.80, frac_z=0.85, n_ring=24,
                     torus_mode="conforming", verbose=True):
    """
    Build the targeted component set.  Returns

        names   : component labels
        comps   : list of (points (k,3), weights (k,)) — weights sum to 1
        beta    : truth mass fractions β_j (compaction > 0, porosity < 0)
        target  : the Stokes block each component is MEANT to drive
        centres : one representative position per component, for plotting

    Radial placement is a fraction of the LOCAL surface radius, so every
    component stays inside this particular shape while sitting as far out as the
    (r/R*)^n leverage argument wants it.
    """
    r_hi = 1.05 * Rb
    spec = [
        # name,   direction,            radial fraction, target block
        ("core",  _sph(0.0, 0.0),       0.16,            "low degree / β̃"),
        ("torus", None,                 frac,            "J2  (C20)"),
        ("B22",   _sph(0.0, lon22),     frac,            "C22, S22"),
        ("B21",   _sph(lat21, lon21),   frac,            "C21, S21"),
        ("BN",    _sph(+90.0, 0.0),     frac_z,          "J3, J4"),
        ("BS",    _sph(-90.0, 0.0),     frac_z,          "J3, J4"),
    ]
    names, comps, target, centres = [], [], [], []
    for nm, d, fr, tg in spec:
        if nm == "torus":                       # extended: an equatorial ring
            lons = np.linspace(0.0, 360.0, n_ring, endpoint=False)
            dirs = np.array([_sph(0.0, lo) for lo in lons])
            r_s = np.array([surface_radius(tm, V, F, d_, r_hi) for d_ in dirs])
            if torus_mode == "circle":
                # a geometric circle: ONE radius, so it is inscribed by the
                # NARROWEST equatorial direction.  Faithful to a torus, but on
                # an elongated body it throws away most of the (r/R*)^n leverage.
                pts = (fr * r_s.min()) * dirs
            else:
                # "conforming": the ring follows the equatorial outline at a
                # fixed fraction of the LOCAL surface radius.  This is the better
                # analogue of a torus sitting at the equatorial bulge when the
                # body is not round, and it is what keeps the component useful.
                pts = (fr * r_s)[:, None] * dirs
            w = np.full(n_ring, 1.0 / n_ring)
        else:                                   # point component
            r_s = surface_radius(tm, V, F, d, r_hi)
            pts = (fr * r_s * d)[None, :]
            w = np.array([1.0])
        names.append(nm)
        comps.append((pts, w))
        target.append(tg)
        # a full ring has its centroid at the origin, which is not where it IS:
        # label and report it by a representative sub-point instead
        centres.append(pts[0] if len(pts) > 1 else pts[0])

    # truth: a few per cent each, mixed signs — compaction and porosity
    beta = np.array([+0.035, -0.025, +0.030, -0.020, +0.028, -0.018])

    if verbose:
        print(f"  components (positions FIXED, densities ESTIMATED):")
        for nm, c, b, tg, (pts, _) in zip(names, centres, beta, target, comps):
            kind = f"ring({len(pts)})" if len(pts) > 1 else "point"
            r_rep = float(np.mean(np.linalg.norm(pts, axis=1)))
            print(f"    {nm:9s} {kind:8s} r={r_rep:.3f} LU  "
                  + (f"(equatorial ring, {torus_mode})           "
                     if len(pts) > 1 else
                     f"lat={np.degrees(np.arcsin(np.clip(c[2]/max(np.linalg.norm(c),1e-9),-1,1))):+6.1f}° "
                     f"lon={np.degrees(np.arctan2(c[1], c[0])):+7.1f}°  ")
                  + f"β={b:+.3f}  lever (r/R*)²={(r_rep/Rb)**2:.3f} "
                    f"⁴={(r_rep/Rb)**4:.4f}  → targets {tg}")
    return names, comps, beta, target, np.asarray(centres)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DESIGN MATRICES FOR EXTENDED COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════
# Identical in meaning to G.A_stokes_contrast / G.A_field_contrast, but a column
# is now the weighted sum over a component's sub-points minus the bulk (the
# weights sum to 1, so the bulk is subtracted exactly once).


def A_stokes_raw(comps, Lmin, Lmax, Rref):
    """
    Each component's OWN unit-mass Stokes signature, with NO bulk subtraction.

    This is the right object for the targeting question — "which coefficients
    does putting mass here influence?" — and it is NOT what the estimator sees.
    The estimator works on contrasts (component minus the bulk mass it displaces),
    and every contrast column inherits the SAME −CS_CD term; on a body with a
    large C20/C22 like Eros that common term swamps the columns and makes any
    column-normalized comparison meaningless.  Hence: targeting on the raw
    signature, estimation on the contrast.
    """
    return np.column_stack([
        sum(wi * G.sh_stokes_of_point(p, Lmin, Lmax, Rref) for p, wi in zip(pts, w))
        for pts, w in comps])


def A_stokes_comp(comps, bulk, Lmin, Lmax, Rref):
    cs_cd = bulk.stokes(Lmin, Lmax, Rref)
    cols = []
    for pts, w in comps:
        c = sum(wi * G.sh_stokes_of_point(p, Lmin, Lmax, Rref)
                for p, wi in zip(pts, w))
        cols.append(c - cs_cd)
    return np.column_stack(cols)


def A_field_comp(comps, obs, bulk):
    f_cd = bulk.field(obs)
    cols = []
    for pts, w in comps:
        f = sum(wi * G.point_mass_field(p, obs) for p, wi in zip(pts, w))
        cols.append(f - f_cd)
    return np.column_stack(cols)


def stokes_total_comp(beta, comps, bulk, Lmin, Lmax, Rref):
    return (bulk.stokes(Lmin, Lmax, Rref)
            + A_stokes_comp(comps, bulk, Lmin, Lmax, Rref) @ beta)


def field_total_comp(beta, comps, obs, bulk):
    return bulk.field(obs) + A_field_comp(comps, obs, bulk) @ beta


def ch_blocks_comp(comps, net, ch_modes, eps, beta, bulk):
    """pt2's per-cylinder CH blocks, for extended components."""
    blocks = []
    for c in net:
        Phi = G.cyl_basis(c["cyl"], c["obs"], *ch_modes)
        pinv = G.ch_pinv(Phi)                # unweighted inner fit, trunc. SVD
        A_ch = pinv @ A_field_comp(comps, c["obs"], bulk)
        y_ch = pinv @ field_total_comp(beta, comps, c["obs"], bulk)
        blocks.append((A_ch, G.od_sigma(y_ch, eps)))
    return blocks


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — TARGETING DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════


def coeff_labels(Lmin, Lmax):
    """Labels in the column order of G.sh_stokes_of_point, and a mask dropping
    the S̄_n0 entries, which are identically zero by construction."""
    lab, keep = [], []
    for n in range(Lmin, Lmax + 1):
        for m in range(n + 1):
            lab += [f"C{n}{m}", f"S{n}{m}"]
            keep += [True, m > 0]
    return np.array(lab), np.array(keep)


def targeting_matrix(A_sh):
    """
    Share of each Stokes coefficient driven by each component, per UNIT mass
    fraction: column-normalized |A|.  Answers "if this coefficient moves, who
    moved it?", which is the claim targeted placement makes.  It deliberately
    does NOT use the truth β — targeting is a property of the design — and it
    must be fed the RAW signature (`A_stokes_raw`), not the contrast.
    """
    M = np.abs(A_sh)
    return M / (M.sum(axis=1, keepdims=True) + 1e-300)


def corr_from_fisher(blocks):
    """Posterior correlation matrix of the component mass fractions."""
    C = np.linalg.inv(G.fisher_masses(blocks))
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════


def run(Lmax_sh=6, eps=0.02, ch_modes=(6, 6), n_cyl=6, n_mc=300,
        torus_mode="conforming", outdir="Images", verbose=True):
    V, F, tm, Rb = G.load_eros()
    Rref = Rb
    bulk = G.Bulk(V, F)

    if verbose:
        print(SEP)
        print("  PART 2b — TARGETED COMPONENTS: sensitivity vs separability (Eros)")
        print(SEP)
    names, comps, beta, target, centres = place_components(
        V, F, tm, Rb, torus_mode=torus_mode, verbose=verbose)
    beta_bulk = G.bulk_fraction(beta)
    n_c = len(names)

    # the CH network is pt2's, and knows nothing about where the components are
    net = P2.build_network(V, F, tm, n_cyl=n_cyl)

    A_sh = A_stokes_comp(comps, bulk, 2, Lmax_sh, Rref)
    sig_sh = G.od_sigma(stokes_total_comp(beta, comps, bulk, 2, Lmax_sh, Rref), eps)
    ch_blocks = ch_blocks_comp(comps, net, ch_modes, eps, beta, bulk)
    blocksA = [(A_sh, sig_sh)]                    # SH only
    blocksB = blocksA + ch_blocks                 # SH + CH network

    if verbose:
        print(f"\n  BODY: constant-density polyhedron, β̃ = 1 − Σβ = {beta_bulk:.4f} of M*")
        print(f"  observables: SH deg 2..{Lmax_sh} ({A_sh.shape[0]} coeffs)"
              f" | {n_cyl} CH cylinders (farthest-point, blind to the components)")

    # ── 1. TARGETING ────────────────────────────────────────────────────────
    lab, keep = coeff_labels(2, Lmax_sh)
    T = targeting_matrix(A_stokes_raw(comps, 2, Lmax_sh, Rref))
    if verbose:
        print(f"\n{'-'*78}\n  1) TARGETING — share of each coefficient driven by each "
              f"component\n{'-'*78}")
        print("  (computed on each component's OWN signature; the estimator itself "
              "works on\n   contrasts against the bulk, which share a common term "
              "and cannot be compared this way)")
        print(f"  {'component':8s} {'intended target':18s} {'actually dominates':28s} "
              f"{'share':>6}")
        for j, nm in enumerate(names):
            k = np.where(keep)[0]
            best = k[np.argsort(-T[k, j])][:3]
            print(f"  {nm:8s} {target[j]:18s} "
                  f"{', '.join(f'{lab[b]}' for b in best):28s} "
                  f"{T[best[0], j]:5.0%}")
        # did each targeted coefficient actually end up dominated by its component?
        print(f"\n  {'coefficient':12s} {'dominated by':12s} {'share':>6}   "
              f"{'intended':12s}")
        want = {"C22": "B22", "S22": "B22", "C21": "B21", "S21": "B21",
                "C30": "BN/BS", "C40": "BN/BS", "C20": "torus"}
        n_ok = 0
        for cf, who in want.items():
            k = int(np.where(lab == cf)[0][0])
            j = int(np.argmax(T[k]))
            ok = names[j] in who.split("/")
            n_ok += ok
            print(f"  {cf:12s} {names[j]:12s} {T[k, j]:5.0%}   {who:12s} "
                  f"{'OK' if ok else '<-- NOT the intended one'}")
        print(f"  ⇒ {n_ok}/{len(want)} targeted coefficients are actually dominated "
              f"by their component.")
        r_eq = max(np.linalg.norm(centres[names.index(nm)])
                   for nm in ("B22",))
        r_pol = np.linalg.norm(centres[names.index("BN")])
        print(f"\n  Why the misses: a component's pull on degree n scales as "
              f"(r/R*)^n, and Eros is\n    FLAT IN Z — the +z surface sits at "
              f"{r_pol/0.85:.2f} LU against {r_eq/0.80:.2f} LU on the long axis, so a "
              f"polar\n    component is stuck at ({r_pol/r_eq:.2f})^n of an "
              f"equatorial one's leverage: {(r_pol/r_eq)**4:.3f} at degree 4.\n"
              f"    The zonal targets J3/J4 are therefore weak HERE in a way they "
              f"are not on a\n    round body like Bennu; the same is true of a "
              f"circular torus, whose radius is\n    capped by the narrowest "
              f"equatorial direction.  This is a property of the SHAPE,\n"
              f"    not of the targeting idea.")

    # ── 2. RECOVERY ─────────────────────────────────────────────────────────
    mcA = G.monte_carlo_fit(blocksA, beta, n_mc=n_mc)
    mcB = G.monte_carlo_fit(blocksB, beta, n_mc=n_mc)
    sigA, sigB = mcA.std(0), mcB.std(0)
    btA, btB = 1.0 - mcA.sum(1), 1.0 - mcB.sum(1)
    if verbose:
        print(f"\n{'-'*78}\n  2) DENSITY RECOVERY — 1σ on β_j ({n_mc} MC fits)\n{'-'*78}")
        print(f"  {'component':10s} {'truth β':>9} {'σ SH only':>12} "
              f"{'σ SH+CH':>12} {'gain':>7}")
        for j, nm in enumerate(names):
            print(f"  {nm:10s} {beta[j]:+9.3f} {sigA[j]:12.2e} {sigB[j]:12.2e} "
                  f"{sigA[j]/sigB[j]:6.1f}×")
        print(f"  {'BODY β̃':10s} {beta_bulk:+9.4f} {btA.std():12.2e} "
              f"{btB.std():12.2e} {btA.std()/btB.std():6.1f}×")

    # ── 3. SEPARABILITY ─────────────────────────────────────────────────────
    corrA, corrB = corr_from_fisher(blocksA), corr_from_fisher(blocksB)
    off = ~np.eye(n_c, dtype=bool)
    if verbose:
        print(f"\n{'-'*78}\n  3) SEPARABILITY — posterior correlation between "
              f"component densities\n{'-'*78}")
        print(f"  max |correlation|:  SH only {np.abs(corrA[off]).max():.3f}   "
              f"SH + CH {np.abs(corrB[off]).max():.3f}")
        print(f"  mean |correlation|: SH only {np.abs(corrA[off]).mean():.3f}   "
              f"SH + CH {np.abs(corrB[off]).mean():.3f}")
        a = np.unravel_index(np.argmax(np.abs(corrA) * off), corrA.shape)
        print(f"  worst SH pair: {names[a[0]]}–{names[a[1]]} "
              f"({corrA[a]:+.3f} → {corrB[a]:+.3f} with CH)")
        print("  ⇒ targeting buys SENSITIVITY, not separability: the components "
              "still have to be\n    solved simultaneously.  The near-surface data "
              "is what reduces the coupling.")

    res = dict(V=V, F=F, Rb=Rb, tm=tm, bulk=bulk, net=net, names=names,
               comps=comps, centres=centres, beta=beta, beta_bulk=beta_bulk,
               target=target, A_sh=A_sh, T=T, lab=lab, keep=keep,
               sigA=sigA, sigB=sigB, btA=btA, btB=btB,
               corrA=corrA, corrB=corrB, Lmax_sh=Lmax_sh, n_cyl=n_cyl)
    make_plots(res, outdir=outdir)
    return res


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════


def make_plots(res, outdir="Images"):
    os.makedirs(outdir, exist_ok=True)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    V, F, names, beta = res["V"], res["F"], res["names"], res["beta"]
    centres, comps = res["centres"], res["comps"]

    # ---- FIG 1: where the components are + the mass budget ------------------
    fig = plt.figure(figsize=(18, 6.8))
    # no wspace=: see the note in _GLOBAL.py — it silently disables the
    # tight_layout call below
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    step = max(1, len(F) // 9000)
    ax.add_collection3d(Poly3DCollection(V[F[::step]], alpha=0.10,
                        facecolor="#9ecae1", edgecolor="0.6", linewidths=0.1))
    for c in res["net"]:
        G.draw_cylinder(ax, c["cyl"], alpha=0.12, lw=0.7)
    for nm, (pts, _), b in zip(names, comps, beta):
        col = COLOR[0] if b > 0 else COLOR[2]
        if len(pts) > 1:                     # extended: draw the ring as a RING
            loop = np.vstack([pts, pts[:1]])
            ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=col, lw=2.6,
                    zorder=5)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=95, color=col,
                       edgecolor="k", depthshade=False, lw=0.5, zorder=5)
    # one short label per component, nudged radially outward so they do not pile
    # up in the middle; the targets live in fig 2 and in the printed table
    for nm, c, b in zip(names, centres, beta):
        d = c / max(np.linalg.norm(c), 1e-9)
        t = c + 0.22 * d
        ax.text(t[0], t[1], t[2], f"{nm} {b:+.3f}", fontsize=8, ha="center",
                fontweight="bold", zorder=6,
                color=COLOR[0] if b > 0 else COLOR[2],
                bbox=dict(fc="white", ec="0.75", alpha=0.85, pad=1.2, lw=0.4))
    ax.scatter([], [], color=COLOR[0], label=r"compaction $\beta_j>0$")
    ax.scatter([], [], color=COLOR[2], label=r"porosity $\beta_j<0$")
    ax.plot([], [], color="crimson", lw=2, label=f"{res['n_cyl']} CH cylinders")
    ax.set_xlabel("x [LU]"); ax.set_ylabel("y [LU]"); ax.set_zlabel("z [LU]")
    ax.set_title("Components placed to TARGET specific coefficients\n"
                 f"on a constant-density body ($\\tilde\\beta$ = {res['beta_bulk']:.3f})"
                 "\n(targets: see fig. 2)", fontsize=10.5)
    G.set_axes_true_shape(ax, np.vstack([V] + [G.cylinder_hull(c["cyl"])
                                              for c in res["net"]]))
    ax.view_init(elev=24, azim=-58)
    ax.legend(fontsize=8, loc="upper left")
    G.draw_mass_budget(fig.add_subplot(gs[0, 1]), names, beta, res["beta_bulk"],
                       recovered=(res["btB"].mean(), res["btB"].std()),
                       rec_label=f"SH + {res['n_cyl']}-CH network")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt2b_fig1_components.pdf"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 2: the targeting matrix ---------------------------------------
    T, lab, keep = res["T"], res["lab"], res["keep"]
    fig, ax = plt.subplots(figsize=(17, 4.6))
    im = ax.imshow(T[keep].T, aspect="auto", cmap="magma_r", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="share of that coefficient (per unit $\\beta$)")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    kl = lab[keep]
    ax.set_xticks(range(len(kl))); ax.set_xticklabels(kl, rotation=90, fontsize=6)
    ax.set_title("TARGETING: which component drives which Stokes coefficient\n"
                 "(column-normalized; every component still touches every "
                 "coefficient — that is the degeneracy)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt2b_fig2_targeting.pdf"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 3: recovery + the correlation matrices -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4),
                             gridspec_kw={"width_ratios": [1.35, 1, 1]})
    ax = axes[0]
    x = np.arange(len(names) + 1)
    vA = np.append(res["sigA"], res["btA"].std())
    vB = np.append(res["sigB"], res["btB"].std())
    ax.bar(x - 0.2, vA, 0.4, color=COLOR[2], edgecolor="k", label="SH only")
    ax.bar(x + 0.2, vB, 0.4, color=COLOR[0], edgecolor="k",
           label=f"SH + {res['n_cyl']}-CH network")
    ax.axvline(len(names) - 0.5, color="0.6", ls="--", lw=1.2)
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels(names + [r"BODY $\tilde\beta$"], fontsize=8)
    ax.set_ylabel(r"1$\sigma$ uncertainty $\sigma_\beta$  (MC)")
    ax.set_title("Density recovery per component")
    for i in range(len(x)):
        ax.text(x[i] + 0.2, vB[i], rf"${vA[i]/vB[i]:.0f}\times$", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.grid(True, axis="y", which="both", alpha=0.3); ax.legend(fontsize=9)

    for ax, M, ttl in [(axes[1], res["corrA"], "SH only"),
                       (axes[2], res["corrB"], f"SH + {res['n_cyl']}-CH network")]:
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45,
                                                             fontsize=8)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        off = ~np.eye(len(names), dtype=bool)
        ax.set_title(f"Correlation: {ttl}\n"
                     rf"max $|\rho|$ = {np.abs(M[off]).max():.2f}, "
                     f"mean = {np.abs(M[off]).mean():.2f}", fontsize=10)
        for a in range(len(names)):
            for b in range(len(names)):
                if a != b:
                    ax.text(b, a, f"{M[a, b]:+.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="w" if abs(M[a, b]) > 0.55 else "k")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Targeting buys sensitivity; the near-surface data buys "
                 "separability", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_pt2b_fig3_recovery_corr.pdf"),
                dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    res = run(Lmax_sh=6, eps=0.02, ch_modes=(6, 6), n_cyl=6, n_mc=300,
              outdir="Images", verbose=True)
    print("\nSaved: Images/global_pt2b_fig1_components.pdf, "
          "global_pt2b_fig2_targeting.pdf, global_pt2b_fig3_recovery_corr.pdf")
    print("Done.")
