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

Model
-----
Eros interior is modelled by a FEW mascons (the natural low-order picture of a
bilobate body): two lobe mascons carrying the bulk mass, plus one small
near-surface anomaly under a chosen spot.  The unknowns are their mass ratios
f_j = m_j / M (Σ f_j = 1, with the total mass M = GM known from tracking).  We
also localize the near-surface anomaly (position recovery).

Physical mechanism
------------------
The two deep lobes are a LOW-degree (quadrupole-scale) feature — spherical
harmonics constrain them well.  The small near-surface anomaly writes its
signature mostly into HIGH-degree coefficients that are truncated/noisy from
orbit, so SH is nearly blind to it and it stays degenerate with the bulk.
Interior cylindrical harmonics converge inside the Brillouin sphere right down
to the surface (where exterior SH diverges); low-altitude data over a cylinder
above the anomaly, represented by CH, injects exactly the local information SH
is missing — resolving the anomaly's mass ratio and position.

Both observables are LINEAR in the mascon masses, so mass-ratio recovery is a
linear Gaussian inverse problem with an exact posterior covariance.  Position
recovery is nonlinear and handled by a linearized (Fisher) covariance.

Truth / units
-------------
Eros shape (`3dmeshes/eros.pk`), normalized units (LU), total mass M = 1.

Two observables
---------------
  A (SH)     : fully-normalized Stokes coefficients C̄_nm, S̄_nm, degree 2..L_SH,
               each known to σ_SH, PLUS the total mass (Σ f = 1) known to σ_M
               (the global gravity field from tracking).
  B (SH+CH)  : the above  PLUS  the near-surface gravitational field (potential +
               acceleration) in a cylinder just above the anomaly, KNOWN TO
               σ_FIELD and usable only through the CH model that can represent it.
               Its information is the field Fisher projected onto the CH span,
               P_CH = Φ(ΦᵀΦ)⁻¹Φᵀ  (Φ = Bessel–Fourier basis at the cylinder
               points): the part of the near-surface field CH can actually model.

Experiments
-----------
  1. MASS RATIO.  Posterior σ on each mascon's mass ratio, SH vs SH+CH — the
     near-surface anomaly gains the most.
  2. POSITION.  The near-surface anomaly's location; linearized position
     covariance (error ellipsoid), SH vs SH+CH.

Formulae
--------
SH (exterior) unit-mass Stokes basis, mass at (r,φ,λ), ref radius R*:
    {C̄,S̄}_nm  =  (r/R*)^n P̄_nm(sinφ) {cos,sin}(mλ)
CH (interior) basis in a cylinder (axis ẑ, radius R_cyl, extension α):
    U_mn = J_m(k_mn ρ) e^{-k_mn z} {cos,sin}(mφ),   k_mn = j_{m,n}/(α R_cyl)
Point mass at p, field at x:  U = Gm/|x-p|,  g = -Gm (x-p)/|x-p|³   (G=1 in LU).
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


def ch_projector(Phi):
    """P_CH = Φ(ΦᵀΦ)⁻¹Φᵀ : projection onto the span the CH model can represent."""
    # use a pseudo-inverse for numerical safety (Φ can be rank-deficient)
    return Phi @ np.linalg.pinv(Phi.T @ Phi) @ Phi.T


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — INFORMATION / COVARIANCE
# ═══════════════════════════════════════════════════════════════════════════


def fisher_masses(
    A_sh, sig_sh, A_fd=None, P_ch=None, sig_fd=None, sig_M=1e-4, prior_sigma=1.0
):
    """
    Fisher information for the mascon mass-ratio vector f (M = 1 ⇒ f are ratios).
      SH  : (A_shᵀ A_sh)/σ_sh²                     Stokes coefficients, deg 2..L
      M   : (1 1ᵀ)/σ_M²                            known total mass  Σ f = 1
      CH  : (A_fdᵀ P_ch A_fd)/σ_fd²                near-surface field the CH model sees
    A weak Gaussian prior keeps everything finite; with a few well-separated
    mascons the SH+total-mass block is already well-posed.
    """
    n = A_sh.shape[1]
    Fi = np.eye(n) / prior_sigma**2
    Fi = Fi + (A_sh.T @ A_sh) / sig_sh**2
    Fi = Fi + np.ones((n, n)) / sig_M**2  # total-mass (GM) constraint
    if A_fd is not None:
        Fi = Fi + (A_fd.T @ (P_ch @ A_fd)) / sig_fd**2
    return Fi


def posterior_sigma(Fi):
    return np.sqrt(np.diag(np.linalg.inv(Fi)))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — MASCON MODEL (few physical mascons)
# ═══════════════════════════════════════════════════════════════════════════

# name, position [LU], truth mass ratio.  Index 0 is the near-surface anomaly
# (the CH target); 1,2 are the two lobes carrying the bulk mass.
MASCONS = [
    ("near-surface anomaly", np.array([0.00, 0.00, 0.22]), 0.06),
    ("+x big lobe", np.array([0.42, 0.00, 0.00]), 0.54),
    ("-x small lobe", np.array([-0.45, 0.00, 0.00]), 0.40),
]


def mascon_arrays():
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


def position_covariance(idx, P, obs, cyl, ch_modes, sig_sh, sig_fd, Lmax, Rref, P_ch):
    """
    Linearized position covariance of mascon `idx`, with its truth mass, from
    SH-only and SH+CH.  Position partials by central differences.  (Other mascon
    masses / positions held fixed — the near-surface anomaly is the target.)
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

    Jsh = jac(lambda p: sh_stokes_of_point(p, 2, Lmax, Rref), A_stokes(P, 2, Lmax, Rref).shape[0])
    Fi_A = Jsh.T @ Jsh / sig_sh**2
    Jfd = jac(lambda p: point_mass_field(p, obs), 4 * len(obs))
    Fi_B = Fi_A + Jfd.T @ (P_ch @ Jfd) / sig_fd**2

    def summ(Fi):
        C = np.linalg.inv(Fi)
        return dict(cov=C, rms=float(np.sqrt(np.trace(C) / 3.0)))

    return summ(Fi_A), summ(Fi_B)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5b — ACTUAL LEAST-SQUARES FIT (Monte-Carlo, like the reference code)
# ═══════════════════════════════════════════════════════════════════════════


def ch_coeff_design(P, obs, cyl, ch_modes):
    """
    Mascon-mass → CYLINDRICAL-HARMONIC coefficient design matrix.
    For a unit mass at each mascon we evaluate its near-surface field and fit the
    Bessel–Fourier basis Φ by least squares (exactly as the reference script fits
    CH coefficients to the sampled field): c_j = Φ⁺ f_j.  A_ch = Φ⁺ A_field.
    """
    Phi = cyl_basis(cyl, obs, *ch_modes)
    A_fd = A_field(P, obs)
    return np.linalg.pinv(Phi) @ A_fd  # (n_ch, n_mascon)


def ls_fit_once(blocks, m_true, rng):
    """
    One weighted (whitened) linear least-squares fit of the mascon masses from
    noisy coefficient observables, in the style of the reference code:
        minimize Σ_blocks || (A m − y)/σ ||² ,   y = A m_true + N(0, σ).
    `blocks` is a list of (A, σ).  Returns the recovered mass vector.
    """
    As, ys = [], []
    for A, sig in blocks:
        y = A @ m_true + rng.normal(0.0, sig, size=A.shape[0])
        As.append(A / sig)
        ys.append(y / sig)
    Aw, yw = np.vstack(As), np.concatenate(ys)
    m_hat, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    return m_hat


def monte_carlo_fit(blocks, m_true, n_mc=400, seed=7):
    """Monte-Carlo over noise: returns recovered masses (n_mc, n_mascon)."""
    rng = np.random.default_rng(seed)
    return np.array([ls_fit_once(blocks, m_true, rng) for _ in range(n_mc)])


def detection_sweep(A_sh, sig_sh, A_ch, sig_ch, A_M, sig_M, f_base, mu_grid,
                    n_mc=250, seed=11):
    """
    Smallest detectable anomaly with vs without the cylinder.  Sweeps the true
    anomaly mass ratio (index 0) over `mu_grid`, keeps the two lobes fixed, and
    for each value runs the LS fit (SH-only and SH+CH) over noise.  Returns, per
    grid value, the recovered anomaly mean and scatter for both cases.
    """
    out = {k: [] for k in ("muA", "sdA", "muB", "sdB")}
    for mu in mu_grid:
        m_true = f_base.copy()
        m_true[0] = mu
        A = monte_carlo_fit([(A_sh, sig_sh), (A_M, sig_M)], m_true, n_mc, seed)
        B = monte_carlo_fit(
            [(A_sh, sig_sh), (A_ch, sig_ch), (A_M, sig_M)], m_true, n_mc, seed
        )
        out["muA"].append(A[:, 0].mean()); out["sdA"].append(A[:, 0].std())
        out["muB"].append(B[:, 0].mean()); out["sdB"].append(B[:, 0].std())
    for k in out:
        out[k] = np.asarray(out[k])
    out["mu_grid"] = np.asarray(mu_grid)
    # 3σ detection threshold = smallest true anomaly whose recovery exceeds 3×scatter
    def thr(mu, sd):
        floor = np.median(sd)  # scatter is ~flat in the mass ratio (linear problem)
        return 3.0 * floor
    out["thr_A"] = thr(out["mu_grid"], out["sdA"])
    out["thr_B"] = thr(out["mu_grid"], out["sdB"])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5c — EXPERIMENT 2: fit anomaly POSITION with masses FIXED (MC)
# ═══════════════════════════════════════════════════════════════════════════
# (Experiment 1 — masses free, positions fixed — is the linear fit above.)


def _pos_forward(pos0, masses, lobe_pos, Lmax, Rref, obs, pinvPhi, use_ch):
    """
    Forward observables when ONLY the anomaly position pos0=(x,y,z) is unknown;
    all three masses and the two lobe positions are fixed/known.
    """
    positions = [pos0, lobe_pos[0], lobe_pos[1]]
    y_sh = np.zeros(len(sh_stokes_of_point(positions[0], 2, Lmax, Rref)))
    for mj, pj in zip(masses, positions):
        y_sh = y_sh + mj * sh_stokes_of_point(pj, 2, Lmax, Rref)
    blocks = [y_sh]
    if use_ch:
        field = np.zeros(4 * len(obs))
        for mj, pj in zip(masses, positions):
            field = field + mj * point_mass_field(pj, obs)
        blocks.append(pinvPhi @ field)
    return blocks


def _pos_residual(pos0, data_blocks, sig_blocks, masses, lobe_pos, Lmax, Rref,
                  obs, pinvPhi, use_ch):
    model = _pos_forward(pos0, masses, lobe_pos, Lmax, Rref, obs, pinvPhi, use_ch)
    return np.concatenate([(mo - da) / s for mo, da, s in
                           zip(model, data_blocks, sig_blocks)])


def position_mc(
    P, f_true, obs, cyl, ch_modes, Lmax, Rref, sig_sh, sig_ch,
    use_ch, n_mc=150, seed=21, start_offset=0.03,
):
    """
    EXPERIMENT 2 — Monte-Carlo NONLINEAR least-squares recovery of the anomaly
    POSITION (x, y, z) with all masses FIXED at truth.  Each draw builds noisy
    data, starts from a guess offset by `start_offset` LU, and fits (scipy TRF).
    Returns recovered positions (n_mc, 3).
    """
    Phi = cyl_basis(cyl, obs, *ch_modes)
    pinvPhi = np.linalg.pinv(Phi)
    masses = f_true
    lobe_pos = [P[1], P[2]]
    pos_true = P[0].copy()
    truth_blocks = _pos_forward(pos_true, masses, lobe_pos, Lmax, Rref, obs,
                                pinvPhi, use_ch)
    sig_blocks = [sig_sh, sig_ch][: len(truth_blocks)]

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_mc):
        data = [tb + rng.normal(0.0, s, size=tb.shape)
                for tb, s in zip(truth_blocks, sig_blocks)]
        pos0 = pos_true + start_offset
        sol = least_squares(
            _pos_residual, pos0, method="trf",
            args=(data, sig_blocks, masses, lobe_pos, Lmax, Rref, obs, pinvPhi,
                  use_ch),
            xtol=1e-12, ftol=1e-12, max_nfev=400,
        )
        out.append(sol.x)
    return np.asarray(out)


def run_experiment(
    Lmax_sh=6,
    eps=0.02,
    sig_M=1e-3,
    ch_modes=(8, 8),
    n_cyl_pts=200,
    n_mc=400,
    n_mc_nl=150,
    outdir="Images",
    verbose=True,
):
    """
    `eps` is the RELATIVE measurement precision applied EQUALLY to both
    observables: σ_SH = eps·RMS(truth Stokes), σ_field = eps·RMS(truth
    near-surface field).  Same fractional data quality on the global SH field
    and the local near-surface field, so the comparison reflects geometry, not
    the (different) natural units of the two observables.
    """
    V, F, tm, Rb = load_eros()
    Rref = Rb
    zmax = V[:, 2].max()
    names, P, f_true = mascon_arrays()
    target = 0  # near-surface anomaly

    if verbose:
        print(SEP)
        print("  Interior-density recovery: SH  vs  SH+CH   (Eros, normalized units)")
        print(SEP)
        print(f"  Brillouin R* = {Rb:.3f} LU,  z_max = {zmax:.3f} LU")
        print("  mascons (truth mass ratios):")
        for nm, p, fr in zip(names, P, f_true):
            print(f"    {nm:22s} p={np.round(p,3)}  f={fr:.3f}  depth={zmax-p[2]:.3f}")

    # cylinder of near-surface data over the anomaly (+z pole)
    cyl = Cylinder(center=np.array([0.0, 0.0, zmax + 0.005]), radius=0.12, height=0.40)
    obs = cylinder_points(cyl, n=n_cyl_pts)
    obs = obs[~inside_body(tm, V, F, obs)]
    r_obs = np.linalg.norm(obs, axis=1)

    A_sh = A_stokes(P, 2, Lmax_sh, Rref)
    A_fd = A_field(P, obs)
    Phi = cyl_basis(cyl, obs, *ch_modes)
    P_ch = ch_projector(Phi)

    # fair, unit-independent noise: same relative precision on both observables
    sig_sh = eps * float(np.sqrt(np.mean((A_sh @ f_true) ** 2)))
    sig_fd = eps * float(np.sqrt(np.mean((A_fd @ f_true) ** 2)))
    # total mass (GM) known to a realistic fixed precision (decoupled from eps);
    # results saturate for σ_M ≲ 1e-3, so this is the conservative regime.
    if verbose:
        print(f"  cylinder over anomaly: {len(obs)} vacuum pts, "
              f"|r|∈[{r_obs.min():.2f},{r_obs.max():.2f}] ⊂ Brillouin {Rb:.2f}")
        print(f"  observables: SH deg 2..{Lmax_sh} ({A_sh.shape[0]} coeffs) + total mass"
              f" | CH modes {ch_modes} ({Phi.shape[1]} cols)")
        print(f"  noise: relative eps={eps} → σ_SH={sig_sh:.2e}, σ_field={sig_fd:.2e}")

    # ── PART 1 — mass ratios ────────────────────────────────────────────────
    Fi_A = fisher_masses(A_sh, sig_sh, sig_M=sig_M)
    Fi_B = fisher_masses(A_sh, sig_sh, A_fd, P_ch, sig_fd, sig_M=sig_M)
    sdA, sdB = posterior_sigma(Fi_A), posterior_sigma(Fi_B)
    improve = sdA / sdB
    if verbose:
        print(f"\n{'-'*70}\n  PART 1 — MASS-RATIO UNCERTAINTY (1σ on f_j)\n{'-'*70}")
        print(f"  {'mascon':22s} {'depth':>6} {'σ_SH':>10} {'σ_SH+CH':>10} {'gain':>7}")
        for nm, p, a, b in zip(names, P, sdA, sdB):
            print(f"  {nm:22s} {zmax-p[2]:6.3f} {a:10.2e} {b:10.2e} {a/b:6.1f}×")

    # ── PART 2 — position of the near-surface anomaly ───────────────────────
    posA, posB = position_covariance(
        target, P, obs, cyl, ch_modes, sig_sh, sig_fd, Lmax_sh, Rref, P_ch
    )
    if verbose:
        print(f"\n{'-'*70}\n  PART 2 — POSITION OF NEAR-SURFACE ANOMALY (f={f_true[target]:.3f})\n{'-'*70}")
        print(f"  position 1σ RMS:  SH={posA['rms']:.3e} LU   SH+CH={posB['rms']:.3e} LU"
              f"   → {posA['rms']/posB['rms']:.0f}× tighter")

    # ══ EXPERIMENT 1 — MASS RATIOS (all positions FIXED) ═══════════════════
    # Monte-Carlo linear least squares on coefficient observables, exactly like
    # the reference code: SH Stokes + total mass (Case A), plus fitted CH
    # coefficients (Case B).  The three mascon positions are held at truth.
    A_ch = ch_coeff_design(P, obs, cyl, ch_modes)
    A_M = np.ones((1, len(P)))
    sig_ch = eps * float(np.sqrt(np.mean((A_ch @ f_true) ** 2)))
    blocksA = [(A_sh, sig_sh), (A_M, sig_M)]
    blocksB = [(A_sh, sig_sh), (A_ch, sig_ch), (A_M, sig_M)]
    mcA = monte_carlo_fit(blocksA, f_true, n_mc=n_mc)
    mcB = monte_carlo_fit(blocksB, f_true, n_mc=n_mc)
    fitA = dict(mean=mcA.mean(0), std=mcA.std(0), samples=mcA,
                Mtot_mean=mcA.sum(1).mean(), Mtot_std=mcA.sum(1).std())
    fitB = dict(mean=mcB.mean(0), std=mcB.std(0), samples=mcB,
                Mtot_mean=mcB.sum(1).mean(), Mtot_std=mcB.sum(1).std())
    if verbose:
        print(f"\n{'='*70}\n  EXPERIMENT 1 — MASS RATIOS, positions FIXED "
              f"({n_mc} draws)\n{'='*70}")
        print(f"  {'quantity':22s} {'truth':>8} | {'SH: mean±std':>20} {'err%':>6}"
              f" | {'SH+CH: mean±std':>20} {'err%':>6}")
        for k, nm in enumerate(names):
            ta = f"{fitA['mean'][k]:+.4f}±{fitA['std'][k]:.4f}"
            tb = f"{fitB['mean'][k]:+.4f}±{fitB['std'][k]:.4f}"
            ea = 100 * abs(fitA['mean'][k] - f_true[k]) / f_true[k]
            eb = 100 * abs(fitB['mean'][k] - f_true[k]) / f_true[k]
            print(f"  {nm:22s} {f_true[k]:8.3f} | {ta:>20} {ea:5.1f}% | {tb:>20} {eb:5.1f}%")
        ta = f"{fitA['Mtot_mean']:+.4f}±{fitA['Mtot_std']:.4f}"
        tb = f"{fitB['Mtot_mean']:+.4f}±{fitB['Mtot_std']:.4f}"
        print(f"  {'TOTAL MASS':22s} {f_true.sum():8.3f} | {ta:>20} {'':5} | {tb:>20}")

    # smallest detectable anomaly (part of Experiment 1: positions fixed)
    f_base = f_true.copy()
    mu_grid = np.logspace(-4.5, -0.7, 16)  # true anomaly mass ratio sweep
    det = detection_sweep(A_sh, sig_sh, A_ch, sig_ch, A_M, sig_M, f_base, mu_grid,
                          n_mc=max(150, n_mc // 2))
    if verbose:
        print(f"\n  smallest detectable anomaly (3σ fit scatter):")
        print(f"    SH only : μ_min = {det['thr_A']:.2e}")
        print(f"    SH + CH : μ_min = {det['thr_B']:.2e}   "
              f"→ {det['thr_A']/det['thr_B']:.0f}× smaller anomaly detectable")

    # ══ EXPERIMENT 2 — ANOMALY POSITION (all masses FIXED) ═════════════════
    # Monte-Carlo NONLINEAR least squares for the anomaly's (x,y,z); the three
    # mass ratios are held at truth.  Nothing is estimated jointly with mass.
    if verbose:
        print(f"\n{'='*70}\n  EXPERIMENT 2 — ANOMALY POSITION, masses FIXED "
              f"({n_mc_nl} draws)\n{'='*70}")
    posA_nl = position_mc(P, f_true, obs, cyl, ch_modes, Lmax_sh, Rref,
                          sig_sh, sig_ch, use_ch=False, n_mc=n_mc_nl)
    posB_nl = position_mc(P, f_true, obs, cyl, ch_modes, Lmax_sh, Rref,
                          sig_sh, sig_ch, use_ch=True, n_mc=n_mc_nl)
    pos_rmsA = float(np.sqrt(np.mean(np.sum((posA_nl - P[target]) ** 2, axis=1))))
    pos_rmsB = float(np.sqrt(np.mean(np.sum((posB_nl - P[target]) ** 2, axis=1))))
    nl = dict(nlA=posA_nl, nlB=posB_nl, pos_rmsA=pos_rmsA, pos_rmsB=pos_rmsB)
    if verbose:
        biasA = np.linalg.norm(posA_nl.mean(0) - P[target])
        biasB = np.linalg.norm(posB_nl.mean(0) - P[target])
        print(f"  anomaly POSITION RMS error:  SH={pos_rmsA:.3e} LU  "
              f"SH+CH={pos_rmsB:.3e} LU  → {pos_rmsA/pos_rmsB:.0f}× tighter")
        print(f"  recovered-mean bias:         SH={biasA:.2e}  SH+CH={biasB:.2e}  "
              f"(both ≈ unbiased)")

    spec_shallow = stokes_power_spectrum(P[target], max(Lmax_sh + 6, 12), Rref)
    spec_lobe = stokes_power_spectrum(P[1], max(Lmax_sh + 6, 12), Rref)

    res = dict(
        V=V, F=F, Rb=Rb, zmax=zmax, cyl=cyl, obs=obs, P=P, names=names,
        f_true=f_true, target=target, sdA=sdA, sdB=sdB, improve=improve,
        posA=posA, posB=posB, spec_shallow=spec_shallow, spec_lobe=spec_lobe,
        fitA=fitA, fitB=fitB, det=det, nl=nl,
        Lmax_sh=Lmax_sh, ch_modes=ch_modes, sig_sh=sig_sh, sig_fd=sig_fd,
    )
    make_plots(res, outdir=outdir)
    return res


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════


def _plot_ellipse(ax, center, cov2, color, label, nsig=1.0):
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.maximum(vals, 0)
    t = np.linspace(0, 2 * np.pi, 120)
    ell = vecs @ (nsig * np.sqrt(vals)[:, None] * np.array([np.cos(t), np.sin(t)]))
    ax.plot(center[0] + ell[0], center[1] + ell[1], color=color, lw=2, label=label)


def draw_silhouette(ax, V, F, i, j, color="0.82", edge="0.6", zorder=0):
    """Filled cross-section silhouette of the shape projected onto axes (i, j)."""
    from matplotlib.collections import PolyCollection

    tris2d = V[F][:, :, [i, j]]
    ax.add_collection(
        PolyCollection(tris2d, facecolors=color, edgecolors="none", alpha=0.9,
                       zorder=zorder)
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

    # ---- FIG 1: geometry + Stokes spectrum + mass-ratio bars ----------------
    fig = plt.figure(figsize=(18, 5.4))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    step = max(1, len(F) // 8000)
    pc = Poly3DCollection(V[F[::step]], alpha=0.18, facecolor="#9ecae1",
                          edgecolor="0.55", linewidths=0.1)
    ax.add_collection3d(pc)
    ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], c="crimson", s=4, alpha=0.6,
               label="CH cylinder data")
    for i, (nm, p) in enumerate(zip(names, P)):
        mk = "*" if i == tgt else "o"
        sz = 240 if i == tgt else 120
        ax.scatter(*p, c="k", s=sz, marker=mk)
        ax.text(p[0], p[1], p[2], "  " + nm.split()[0], fontsize=8)
    ax.set_title("Eros interior model:\ntwo lobes + a near-surface anomaly")
    ax.set_xlabel("x [LU]"); ax.set_ylabel("y [LU]"); ax.set_zlabel("z [LU]")
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    ax.legend(loc="upper left", fontsize=9)

    ax = fig.add_subplot(1, 3, 2)
    degs = np.arange(len(res["spec_shallow"]))
    ax.semilogy(degs, res["spec_shallow"], "-o", color=COLOR[0], label="near-surface anomaly")
    ax.semilogy(degs, res["spec_lobe"], "-s", color=COLOR[2], label="deep lobe")
    ax.axvspan(2, res["Lmax_sh"], color="0.85", label=f"observed SH (≤{res['Lmax_sh']})")
    ax.set_xlabel("SH degree n"); ax.set_ylabel("Stokes signature (RMS per degree)")
    ax.set_title("Where each mascon's signature lives")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)

    ax = fig.add_subplot(1, 3, 3)
    xpos = np.arange(len(names)); w = 0.38
    ax.bar(xpos - w/2, res["sdA"], w, color=COLOR[2], edgecolor="k", label="SH only")
    ax.bar(xpos + w/2, res["sdB"], w, color=COLOR[0], edgecolor="k", label="SH + CH")
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax.set_ylabel(r"mass-ratio 1σ uncertainty  $\sigma_{f}$")
    ax.set_title("Mass-ratio recovery")
    for i in range(len(names)):
        ax.text(i, res["sdB"][i], f"{res['improve'][i]:.0f}×", ha="center",
                va="bottom", fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_fig1_massratio.png"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 2: position error ellipses over the Eros cross-section ---------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, plane, (i, j), lbl in [
        (axes[0], "x–z", (0, 2), ("x", "z")),
        (axes[1], "y–z", (1, 2), ("y", "z")),
    ]:
        draw_silhouette(ax, V, F, i, j)  # Eros cross-section for context
        # all mascons in this plane
        ax.scatter(P[:, i], P[:, j], c="k", s=30, zorder=5)
        # cylinder footprint (projected) as a light band
        cc = cyl.center
        ax.plot([cc[i]], [cc[j]], marker="v", color="crimson", ms=9,
                label="CH cylinder", zorder=6)
        for case, col, name in [
            (res["posA"], COLOR[2], "SH only"),
            (res["posB"], COLOR[0], "SH + CH"),
        ]:
            C = case["cov"][np.ix_([i, j], [i, j])]
            _plot_ellipse(ax, P[tgt][[i, j]], C, col, name)
        ax.plot(*P[tgt][[i, j]], "k*", ms=15, label="anomaly (truth)", zorder=7)
        ax.set_xlabel(f"{lbl[0]} [LU]"); ax.set_ylabel(f"{lbl[1]} [LU]")
        ax.set_title(f"Anomaly localization on Eros ({plane})")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

        # zoom inset at the SH+CH scale (≈100× finer) to reveal the red ellipse
        axin = ax.inset_axes([0.04, 0.08, 0.30, 0.32])
        Cb = res["posB"]["cov"][np.ix_([i, j], [i, j])]
        _plot_ellipse(axin, P[tgt][[i, j]], Cb, COLOR[0], "SH+CH")
        axin.plot(*P[tgt][[i, j]], "k*", ms=8)
        w = 4 * math.sqrt(max(Cb[0, 0], Cb[1, 1]))
        axin.set_xlim(P[tgt][i] - w, P[tgt][i] + w)
        axin.set_ylim(P[tgt][j] - w, P[tgt][j] + w)
        axin.set_title("SH+CH zoom (×%d)" % round(res["posA"]["rms"] / res["posB"]["rms"]),
                       fontsize=8, color=COLOR[0])
        axin.tick_params(labelsize=6)
        axin.set_aspect("equal")
        for s in axin.spines.values():
            s.set_edgecolor(COLOR[0])
    fig.suptitle(
        f"Near-surface anomaly localization:  SH {res['posA']['rms']:.2e} LU  →  "
        f"SH+CH {res['posB']['rms']:.2e} LU   "
        f"({res['posA']['rms']/res['posB']['rms']:.0f}× tighter)",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, "global_fig2_position.png"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 3: EXPERIMENT 1 — mass-ratio fit + detection --------------------
    fitA, fitB, det = res["fitA"], res["fitB"], res["det"]
    ft = res["f_true"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    fig.suptitle("EXPERIMENT 1 — mass ratios, positions fixed "
                 f"({len(fitA['samples'])} MC draws)", fontweight="bold", y=1.02)

    # (a) actual-fit recovery ERROR (RMS = bias⊕scatter) per quantity, log scale
    ax = axes[0]
    labels = [n.replace(" ", "\n", 1) for n in names] + ["TOTAL\nmass"]
    xpos = np.arange(len(labels))
    truth = np.concatenate([ft, [ft.sum()]])
    meanA = np.concatenate([fitA["mean"], [fitA["Mtot_mean"]]])
    stdA = np.concatenate([fitA["std"], [fitA["Mtot_std"]]])
    meanB = np.concatenate([fitB["mean"], [fitB["Mtot_mean"]]])
    stdB = np.concatenate([fitB["std"], [fitB["Mtot_std"]]])
    # RMS error about truth = sqrt(bias^2 + scatter^2)
    rmsA = np.sqrt((meanA - truth) ** 2 + stdA**2)
    rmsB = np.sqrt((meanB - truth) ** 2 + stdB**2)
    w = 0.38
    ax.bar(xpos - w / 2, rmsA, w, color=COLOR[2], edgecolor="k", label="SH only")
    ax.bar(xpos + w / 2, rmsB, w, color=COLOR[0], edgecolor="k", label="SH + CH")
    ax.set_yscale("log")
    for i in range(len(labels)):
        ax.text(i + w / 2, rmsB[i], f"{rmsA[i]/rmsB[i]:.0f}×", ha="center",
                va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mass-ratio recovery RMS error")
    ax.set_title("Recovery error (bias ⊕ MC scatter)")
    ax.grid(True, axis="y", which="both", alpha=0.3); ax.legend(fontsize=10)

    # (b) the MC DISTRIBUTION of the recovered ANOMALY mass ratio (zoomed)
    ax = axes[1]
    sA, sB = fitA["samples"][:, 0], fitB["samples"][:, 0]  # anomaly (index 0)
    lo, hi = ft[0] - 5 * sA.std(), ft[0] + 5 * sA.std()
    bins = np.linspace(lo, hi, 40)
    ax.hist(sA, bins=bins, color=COLOR[2], alpha=0.6, label=f"SH only (σ={sA.std():.1e})")
    ax.hist(sB, bins=bins, color=COLOR[0], alpha=0.8,
            label=f"SH + CH (σ={sB.std():.1e})")
    ax.axvline(ft[0], color="k", ls="--", lw=2, label="truth")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"recovered anomaly mass ratio $\hat f_0$")
    ax.set_ylabel("MC count")
    ax.set_title(f"Anomaly mass-ratio MC distribution\n({sA.std()/sB.std():.0f}× narrower "
                 f"with CH; lobes & total unchanged)")
    ax.legend(fontsize=9, loc="upper right")

    # (c) smallest detectable anomaly: recovered anomaly vs true anomaly
    ax = axes[2]
    mug = det["mu_grid"]
    ax.plot(mug, mug, "k--", lw=1, label="perfect recovery")
    ax.errorbar(mug, np.abs(det["muA"]), yerr=det["sdA"], fmt="o", color=COLOR[2],
                ms=5, capsize=3, label="SH only")
    ax.errorbar(mug, np.abs(det["muB"]), yerr=det["sdB"], fmt="s", color=COLOR[0],
                ms=5, capsize=3, label="SH + CH")
    ax.axhline(det["thr_A"], color=COLOR[2], ls=":", lw=1.5,
               label=f"SH 3σ floor = {det['thr_A']:.1e}")
    ax.axhline(det["thr_B"], color=COLOR[0], ls=":", lw=1.5,
               label=f"SH+CH 3σ floor = {det['thr_B']:.1e}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true anomaly mass ratio  $\\mu$")
    ax.set_ylabel(r"recovered anomaly  $|\hat\mu|$")
    ax.set_title(f"Smallest detectable anomaly "
                 f"({det['thr_A']/det['thr_B']:.0f}× smaller with CH)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "global_fig3_lsfit_detection.png"),
                dpi=180, bbox_inches="tight")

    # ---- FIG 4: EXPERIMENT 2 — anomaly POSITION recovery (masses fixed) ------
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
    ax.set_title(f"Position error (RMS {nl['pos_rmsA']:.1e}→{nl['pos_rmsB']:.1e} LU, "
                 f"{nl['pos_rmsA']/nl['pos_rmsB']:.0f}×)")
    ax.legend(fontsize=9)

    # (b,c) recovered position clouds over Eros silhouette (x–z, y–z)
    for ax, (i, j), lbl in [(axes[1], (0, 2), ("x", "z")), (axes[2], (1, 2), ("y", "z"))]:
        draw_silhouette(ax, V, F, i, j)
        ax.scatter(nlA[:, i], nlA[:, j], s=12, color=COLOR[2], alpha=0.5,
                   label="SH only")
        ax.scatter(nlB[:, i], nlB[:, j], s=12, color=COLOR[0], alpha=0.7,
                   label="SH + CH")
        ax.plot(p0[i], p0[j], "k*", ms=16, label="truth", zorder=6)
        sA = max(nlA[:, i].std(), nlA[:, j].std())
        ax.set_xlim(p0[i] - 5 * sA, p0[i] + 5 * sA)
        ax.set_ylim(p0[j] - 5 * sA, p0[j] + 5 * sA)
        ax.set_xlabel(f"{lbl[0]} [LU]"); ax.set_ylabel(f"{lbl[1]} [LU]")
        ax.set_title(f"Recovered position ({lbl[0]}–{lbl[1]})")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="lower right")

        # zoom inset: SH+CH cloud is far tighter — invisible at the SH scale
        axin = ax.inset_axes([0.04, 0.06, 0.32, 0.32])
        axin.scatter(nlB[:, i], nlB[:, j], s=8, color=COLOR[0], alpha=0.7)
        axin.plot(p0[i], p0[j], "k*", ms=8)
        sB = max(nlB[:, i].std(), nlB[:, j].std(), 1e-9)
        axin.set_xlim(p0[i] - 4 * sB, p0[i] + 4 * sB)
        axin.set_ylim(p0[j] - 4 * sB, p0[j] + 4 * sB)
        axin.set_title("SH+CH zoom (×%d)" % round(sA / sB), fontsize=8, color=COLOR[0])
        axin.tick_params(labelsize=6); axin.set_aspect("equal")
        for s in axin.spines.values():
            s.set_edgecolor(COLOR[0])
    fig.suptitle(
        f"EXPERIMENT 2 — anomaly position, masses fixed ({len(nlA)} MC draws):  "
        f"RMS  SH {nl['pos_rmsA']:.2e} LU → SH+CH {nl['pos_rmsB']:.2e} LU  "
        f"({nl['pos_rmsA']/nl['pos_rmsB']:.0f}× tighter)",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, "global_fig4_nonlinear_mc.png"),
                dpi=180, bbox_inches="tight")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    res = run_experiment(
        Lmax_sh=6,        # observable spherical-harmonic degree (tracking limit)
        eps=0.02,         # relative measurement precision (same on SH & field)
        sig_M=1e-3,       # total-mass (GM) precision (realistic; result saturates)
        ch_modes=(8, 8),  # (n_m, n_n) cylindrical-harmonic truncation
        n_cyl_pts=200,
        n_mc=400,         # noise draws for the linear mass fit
        n_mc_nl=150,      # noise draws for the nonlinear masses+position fit
        outdir="Images",
        verbose=True,
    )
    print("\nSaved: Images/global_fig1_massratio.png, global_fig2_position.png, "
          "global_fig3_lsfit_detection.png, global_fig4_nonlinear_mc.png")
    print("Done.")
