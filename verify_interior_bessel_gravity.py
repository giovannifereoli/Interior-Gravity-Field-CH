#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
=====================================================================
 Numerical verification AND figures for

   "Closed-form resummation and exactness of the interior spherical
    Bessel gravity field"  (G. Fereoli, August 2026)
=====================================================================

Everything the paper asserts is checked numerically, from the two
lemmas up to the three parts of Theorem 1 and its corollaries -- and
every figure is then drawn with those same verified routines, so a
curve can never drift away from the test that certifies it.

---------------------------------------------------------------------
A NOTE ON UNITS
---------------------------------------------------------------------
Everything radial is done in the dimensionless variable of Def. (xdef),
x = r/Rs, x' = r'/Rs, on the interval [0,1] -- exactly as the paper
states the boxed identity Eq. (gen):

    S_n(x,x') := sum_{l>=0} [2/E(a_ln)] j_n(a_ln x) j_n(a_ln x')
               = (1/(2n+1)) * x_<^n / x_>^(n+1),        x,x' in [0,1]

with x_< = min(x,x'), x_> = max(x,x').  Both sides are dimensionless.
Rewritten in physical radii the right-hand side picks up one factor of
Rs, since x_<^n / x_>^(n+1) = Rs * r_<^n / r_>^(n+1); that factor is
what cancels the 1/Rs prefactor of Eq. (Vi) in Part II.  This script
verifies the dimensionless form directly, and the potential tests then
carry the Rs factors explicitly, so the end-to-end result
V_i = G int dm'/|r-r'| is checked in physical units as well.

---------------------------------------------------------------------
WHAT IS CHECKED
---------------------------------------------------------------------
 1  Lemma "bc"     : Robin condition  <=>  j_{n-1}(alpha) = 0
 2  Lemma "norm"   : Dini integral, ||f_l||^2 = E/(2 alpha^2), orthogonality
 3  Appendix 5     : local Green's function -- ODE, jump, BC, Wronskian C=2n+1
 4  Theorem (I)    : the generating identity S_n(x,x') (many n, x, x')
 5  Appendix 6     : n=0 shell theorem and the zeta(2)-type series
 6  Corollary rigid: Dirichlet eigenvalues give G - image term (NOT Newton)
 7  Part II Step 2 : addition theorem + Legendre expansion of 1/|r-r'|
 8  Theorem (II)   : exactness for a *continuous* density (uniform sphere),
                     evaluated INSIDE the mass, at the surface, and outside
 9  Theorem (II)   : exactness for a general non-symmetric body via the
                     literal triple sum of Eqs. (Vi)+(coeffs), with the
                     field point both outside and INSIDE the mass shell
10  Remark quantif.: different Rs give different coefficients, same potential
11  Theorem (III)  : for r > Rs the series still converges but NOT to V

---------------------------------------------------------------------
WHAT IS DRAWN  (into ./Images, each a vector PDF + a 400 dpi PNG)
---------------------------------------------------------------------
 fig1_generating_identity   Theorem (I): series vs closed form
 fig5_uniform_sphere        Theorem (II): exact INSIDE a real density
 fig6_region_of_validity    Theorem (III): where exactness stops
 interior_bessel_gravity_figures.pdf   all three in one document

Run:   python3 verify_interior_bessel_gravity.py            # checks + figures
       python3 verify_interior_bessel_gravity.py --tests    # checks only
       python3 verify_interior_bessel_gravity.py --figures  # figures only
Deps:  numpy, scipy, matplotlib, and a LaTeX installation with mathrsfs
       (set USETEX = False below to fall back on matplotlib mathtext).
"""

from __future__ import annotations

import argparse
import os
import time
import numpy as np
from scipy.special import spherical_jn, lpmv, gammaln
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------
# Global truncation parameters (raise for more accuracy, lower for speed)
# ---------------------------------------------------------------------
L_MAX = 3000  # number of radial modes l kept in every l-sum (checks)
N_MAX = 36  # highest spherical-harmonic degree n in the triple sum
G = 1.0  # gravitational constant (units are irrelevant here)

L_PLOT = 2000  # radial modes for the kernel figures
N_POT = 20  # degree truncation for the potential figures
L_POT = 1200  # radial modes for the potential figures

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images")

USETEX = True  # LaTeX typesetting (matches the rest of the repo)
BM = r"\bm" if USETEX else r"\mathbf"  # \bm exists only under real LaTeX

np.seterr(all="ignore")


# #####################################################################
# PART I -- shared routines and the numerical verification
# #####################################################################
# =====================================================================
# 0.  Bookkeeping helpers
# =====================================================================
RESULTS = []


def check(name, err, tol, note=""):
    ok = bool(np.isfinite(err) and err <= tol)
    RESULTS.append((name, ok))
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name:<50s} err={err:9.2e}  tol={tol:7.1e}  {note}"
    )
    return ok


def check_min(name, val, thresh, note=""):
    """Used where the paper predicts a *failure* (deviation must be large)."""
    ok = bool(np.isfinite(val) and val >= thresh)
    RESULTS.append((name, ok))
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name:<50s} val={val:9.2e}  min={thresh:7.1e}  {note}"
    )
    return ok


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =====================================================================
# 1.  Special functions and eigenvalues
# =====================================================================
def jn(n, z):
    """Spherical Bessel j_n, extended to n = -1 via j_{-1}(z) = cos z / z."""
    z = np.asarray(z, dtype=float)
    if n == -1:
        return np.cos(z) / z
    return spherical_jn(n, z)


def djn(n, z):
    """d/dz j_n(z), extended to n = -1."""
    z = np.asarray(z, dtype=float)
    if n == -1:
        return -np.sin(z) / z - np.cos(z) / z**2
    return spherical_jn(n, z, derivative=True)


_ZTAB_CACHE = {}


def zeros_table(m_max, L):
    """
    tab[m] = first L positive zeros of the spherical Bessel function j_m,
    for m = -1, 0, 1, ..., m_max.

    Built by the interlacing property: between two consecutive zeros of
    j_{m-1} there is exactly one zero of j_m.  This is bullet-proof
    (no reliance on asymptotic guesses) and needs no external tables.
    """
    key = (m_max, L)
    if key in _ZTAB_CACHE:
        return _ZTAB_CACHE[key]

    m_max = max(m_max, 0)
    tab = {-1: (np.arange(L) + 0.5) * np.pi}  # zeros of cos z / z
    cur = np.pi * np.arange(1, L + m_max + 2, dtype=float)  # zeros of sin z / z
    tab[0] = cur[:L].copy()

    for m in range(1, m_max + 1):
        f = lambda t, mm=m: spherical_jn(mm, t)
        nxt = np.empty(cur.size - 1)
        for k in range(nxt.size):
            nxt[k] = brentq(f, cur[k], cur[k + 1], xtol=1e-13, maxiter=200)
        cur = nxt
        tab[m] = cur[:L].copy()

    _ZTAB_CACHE[key] = tab
    return tab


def alphas(n, L, tab):
    """Eigenvalues alpha^i_{ln}: the first L positive roots of j_{n-1}."""
    return tab[n - 1][:L]


def E_norm(n, a):
    """Normalisation constant  E = alpha^2 j_n(alpha)^2   (Def. E)."""
    return a**2 * jn(n, a) ** 2


# =====================================================================
# 2.  Series acceleration
# =====================================================================
def accelerate(terms):
    """
    The l-terms decay like 1/l^2 and oscillate, so the partial sums S_L
    behave like  S + a/L + b/L^2 + (oscillation).  Averaging the partial
    sums over a dyadic window kills the oscillation; two stages of
    Richardson extrapolation on three nested windows kill the a/L and
    b/L^2 bias.  Returns (accelerated, raw_partial_sum).

    This is a summation device only -- it never assumes the value of the
    limit, so it cannot manufacture agreement with the closed form.
    """
    S = np.cumsum(np.asarray(terms, dtype=float))
    L = S.size
    if L < 32:
        return S[-1], S[-1]
    c1 = S[L // 8 : L // 4].mean()  # window centred on ~L/4
    c2 = S[L // 4 : L // 2].mean()  # window centred on ~L/2
    c3 = S[L // 2 :].mean()  # window centred on ~L
    return (8.0 * c3 - 6.0 * c2 + c1) / 3.0, S[-1]


def S_spectral(n, x, xp, L, tab, dirichlet=False):
    """
    Left-hand side of the generating identity (dimensionless form):

        sum_l  f_l(x) f_l(x') / (lambda_l ||f_l||_w^2)

    dirichlet=False -> Robin/Neumann-type eigenvalues j_{n-1}(a)=0 (the paper)
    dirichlet=True  -> Dirichlet eigenvalues   j_n(a)=0  (Corollary "rigidity")
    """
    if dirichlet:
        a = tab[n][:L]  # zeros of j_n
        w = 2.0 / (a**2 * jn(n - 1, a) ** 2)  # 1/(lam ||f||^2)
    else:
        a = alphas(n, L, tab)  # zeros of j_{n-1}
        w = 2.0 / E_norm(n, a)  # = 1/(lam ||f||^2)
    terms = w * jn(n, a * x) * jn(n, a * xp)
    return accelerate(terms)


def G_closed(n, x, xp):
    """Closed-form radial Green's function  x_<^n x_>^-(n+1) / (2n+1)."""
    lo, hi = min(x, xp), max(x, xp)
    return lo**n * hi ** (-(n + 1)) / (2 * n + 1)


def G_closed_dirichlet(n, x, xp):
    """Green's function under the Dirichlet condition: Newton MINUS image."""
    lo, hi = min(x, xp), max(x, xp)
    return (lo**n * hi ** (-(n + 1)) - (x * xp) ** n) / (2 * n + 1)


# =====================================================================
# TEST 1 -- Lemma "bc": eigenvalue condition <=> Robin boundary condition
# =====================================================================
def test_boundary_condition(tab):
    banner(
        "TEST 1 -- Lemma (boundary condition):  (n+1) j_n(a) + a j_n'(a) = 0"
        "  <=>  j_{n-1}(a) = 0"
    )
    worst_res, worst_rob = 0.0, 0.0
    for n in range(0, 9):
        a = alphas(n, 12, tab)
        worst_res = max(worst_res, np.max(np.abs(jn(n - 1, a))))
        rob = (n + 1) * jn(n, a) + a * djn(n, a)
        # normalise by the size of the individual terms
        scale = np.abs((n + 1) * jn(n, a)) + np.abs(a * djn(n, a))
        worst_rob = max(worst_rob, np.max(np.abs(rob) / np.maximum(scale, 1e-300)))
    check("eigenvalues are genuine roots of j_{n-1}", worst_res, 1e-11)
    check("Robin condition holds at every eigenvalue", worst_rob, 1e-10)

    # the recurrence (eq. recurrence) that the proof of Lemma bc uses
    z = np.linspace(0.3, 40.0, 4000)
    err = 0.0
    for n in range(0, 8):
        lhs = z * djn(n, z)
        rhs = z * jn(n - 1, z) - (n + 1) * jn(n, z)
        err = max(err, np.max(np.abs(lhs - rhs)) / np.max(np.abs(rhs)))
    check("recurrence z j_n' = z j_{n-1} - (n+1) j_n", err, 1e-11)

    # closed form for n = 0
    a0 = alphas(0, 20, tab)
    err0 = np.max(np.abs(a0 - (np.arange(20) + 0.5) * np.pi)) / a0[-1]
    check("n=0 closed form  alpha_{l0} = (l+1/2)pi", err0, 1e-14)


# =====================================================================
# TEST 2 -- Lemma "norm": Dini integral, ||f||^2 = E/(2 alpha^2), orthogonality
# =====================================================================
def test_norm_lemma(tab):
    banner(
        "TEST 2 -- Lemma (meaning of E): ||f_l||_w^2 = E/(2 alpha^2), and"
        " orthogonality"
    )

    # (a) the Dini-type identity eq.(dini), for arbitrary alpha (not just roots)
    err = 0.0
    for n in range(0, 7):
        for al in [0.7, 2.3, 5.9, 11.4, 23.1]:
            num = quad(
                lambda r, n=n, al=al: jn(n, al * r) ** 2 * r**2, 0.0, 1.0, limit=400
            )[0]
            ana = 0.5 * (jn(n, al) ** 2 - jn(n - 1, al) * jn(n + 1, al))
            err = max(err, abs(num - ana) / max(abs(ana), 1e-16))
    check("Dini identity  int_0^1 j_n(a r)^2 r^2 dr", err, 1e-8)

    # (b) at the eigenvalues the cross term dies and ||f||^2 = E/(2 a^2)
    err = 0.0
    for n in range(0, 7):
        for a in alphas(n, 6, tab):
            num = quad(
                lambda r, n=n, a=a: jn(n, a * r) ** 2 * r**2, 0.0, 1.0, limit=400
            )[0]
            ana = E_norm(n, a) / (2.0 * a**2)
            err = max(err, abs(num - ana) / abs(ana))
    check("||f_l||_w^2 = E/(2 alpha^2) at eigenvalues", err, 1e-8)

    # (c) orthogonality of distinct eigenfunctions in the weighted product
    err = 0.0
    for n in range(0, 5):
        a = alphas(n, 6, tab)
        for i in range(6):
            for k in range(i + 1, 6):
                v = quad(
                    lambda r, n=n, ai=a[i], ak=a[k]: jn(n, ai * r)
                    * jn(n, ak * r)
                    * r**2,
                    0.0,
                    1.0,
                    limit=400,
                )[0]
                nrm = np.sqrt(E_norm(n, a[i]) * E_norm(n, a[k])) / (2 * a[i] * a[k])
                err = max(err, abs(v) / nrm)
    check("orthogonality <f_l,f_k>_w = 0 for l != k", err, 1e-8)


# =====================================================================
# TEST 3 -- Appendix 5: the local Green's function
# =====================================================================
def test_local_green():
    banner("TEST 3 -- Appendix A.5: local Green's function  (ODE, BC, jump, C=2n+1)")

    def Lop(n, f, r, h=1e-5):
        """L_n f = -(r^2 f')' + n(n+1) f  by central differences."""
        d = lambda t: (f(t + h) - f(t - h)) / (2 * h)
        p = lambda t: t**2 * d(t)
        return -(p(r + h) - p(r - h)) / (2 * h) + n * (n + 1) * f(r)

    # (a) homogeneous solutions r^n and r^-(n+1) both annihilate L_n
    err = 0.0
    for n in range(0, 7):
        for r in [0.23, 0.51, 0.87]:
            for s in (n, -(n + 1)):
                val = Lop(n, lambda t, s=s: t ** float(s), r)
                err = max(err, abs(val) / max(abs(r ** float(s)) * (2 * n + 2), 1e-16))
    check("Cauchy-Euler roots s = n, -(n+1) solve L_n f = 0", err, 1e-5)

    # (b) weighted Wronskian  -r^2 (u v' - u' v) = 2n+1  (constant)
    err = 0.0
    for n in range(0, 9):
        for r in [0.13, 0.4, 0.77, 1.0, 1.9]:
            u, du = r**n, n * r ** (n - 1)
            v, dv = r ** (-(n + 1)), -(n + 1) * r ** (-(n + 2))
            W = -(r**2) * (u * dv - du * v)
            err = max(err, abs(W - (2 * n + 1)) / (2 * n + 1))
    check("weighted Wronskian -r^2(uv'-u'v) = 2n+1", err, 1e-12)

    # (c) jump condition  -r'^2 [dG(r'+) - dG(r'-)] = 1  fixes C = 2n+1
    err = 0.0
    h = 1e-6
    for n in range(0, 9):
        for rp in [0.2, 0.5, 0.9]:
            dplus = (G_closed(n, rp + 2 * h, rp) - G_closed(n, rp + h, rp)) / h
            dminus = (G_closed(n, rp - h, rp) - G_closed(n, rp - 2 * h, rp)) / h
            err = max(err, abs(-(rp**2) * (dplus - dminus) - 1.0))
    check("jump condition -r'^2 [G'(+)-G'(-)] = 1", err, 1e-4, "(finite differences)")

    # (d) continuity of G itself at r = r'
    err = 0.0
    for n in range(0, 9):
        for rp in [0.2, 0.5, 0.9]:
            err = max(
                err,
                abs(G_closed(n, rp - 1e-9, rp) - G_closed(n, rp + 1e-9, rp))
                / abs(G_closed(n, rp, rp)),
            )
    check("G_n continuous across the source", err, 1e-7)

    # (e) Robin BC at the outer boundary x = 1, and regularity at 0
    err = 0.0
    for n in range(0, 9):
        for rp in [0.2, 0.5, 0.9]:
            dG = (G_closed(n, 1.0, rp) - G_closed(n, 1.0 - h, rp)) / h
            val = (n + 1) * G_closed(n, 1.0, rp) + dG
            err = max(err, abs(val) / abs((n + 1) * G_closed(n, 1.0, rp)))
    check("G_n obeys (n+1)G + dG/dr = 0 at x = 1", err, 1e-5)


# =====================================================================
# TEST 4 -- Theorem, Part (I): the generating identity
# =====================================================================
def test_generating_identity(tab, L=L_MAX):
    banner(
        "TEST 4 -- Theorem (I): generating identity  sum_l 2/E j_n j_n"
        " = x_<^n / ((2n+1) x_>^(n+1))"
    )

    pairs = [
        (0.10, 0.70),
        (0.70, 0.10),
        (0.30, 0.85),
        (0.85, 0.30),
        (0.45, 0.45),
        (0.62, 0.62),
        (0.05, 0.99),
        (0.50, 1.00),
        (1.00, 0.40),
        (0.90, 0.95),
    ]

    # Error metric: |series - closed form| / (|closed form| + 1e-3).  This is
    # relative where the kernel is O(1) and absolute where the kernel
    # underflows (large n with small x_<), which is where double-precision
    # cancellation, not the identity, sets the accuracy floor.
    worst, worst_rel, worst_abs, worst_raw = 0.0, 0.0, 0.0, 0.0
    print("     n     x       x'        series          closed form      rel.err")
    for n in [0, 1, 2, 3, 5, 8]:
        for x, xp in pairs:
            acc, raw = S_spectral(n, x, xp, L, tab)
            ref = G_closed(n, x, xp)
            worst = max(worst, abs(acc - ref) / (abs(ref) + 1e-3))
            worst_abs = max(worst_abs, abs(acc - ref))
            if abs(ref) > 1e-2:
                worst_rel = max(worst_rel, abs(acc - ref) / abs(ref))
                worst_raw = max(worst_raw, abs(raw - ref) / abs(ref))
            if n in (0, 3) and (x, xp) in [(0.10, 0.70), (0.45, 0.45), (0.85, 0.30)]:
                e = abs(acc - ref) / abs(ref)
                print(
                    f"    {n:2d}  {x:5.2f}  {xp:5.2f}   {acc: .12e}  {ref: .12e}  {e:8.1e}"
                )
    check(
        "generating identity, 6 degrees x 10 point pairs",
        worst,
        1e-5,
        f"(max abs {worst_abs:.1e}, max rel {worst_rel:.1e})",
    )
    print(
        f"      raw (unaccelerated) partial sums would give rel.err"
        f" {worst_raw:.1e} at L = {L}"
    )

    # symmetry in (x,x') -- this is what makes Corollary "validity inside
    # the mass" work: no assumption about which radius is larger
    err = 0.0
    for n in [0, 2, 4]:
        for x, xp in [(0.2, 0.8), (0.35, 0.9)]:
            s1 = S_spectral(n, x, xp, L, tab)[0]
            s2 = S_spectral(n, xp, x, L, tab)[0]
            err = max(err, abs(s1 - s2) / abs(s1))
    check("kernel symmetric under x <-> x' (Cor. inside mass)", err, 1e-12)

    # convergence rate of the RAW partial sums: O(1/L), worst on the diagonal
    print("\n  Raw (unaccelerated) convergence, n=2, x=x'=0.55  [Remark (b)]:")
    n, x = 2, 0.55
    a = alphas(n, L, tab)
    t = (2.0 / E_norm(n, a)) * jn(n, a * x) ** 2
    S = np.cumsum(t)
    ref = G_closed(n, x, x)
    prev = None
    for Lc in [125, 250, 500, 1000, 2000]:
        if Lc <= L:
            e = abs(S[Lc - 1] - ref) / ref
            ratio = "" if prev is None else f"  (ratio {prev / e:5.2f})"
            print(f"      L = {Lc:5d}   rel.err = {e:.3e}{ratio}")
            prev = e
    print("      -> error halves when L doubles: conditional/O(1/L) behaviour,")
    print("         exactly the marginality flagged in Remark (b).")


# =====================================================================
# TEST 5 -- Appendix 6: shell theorem and the zeta-type series
# =====================================================================
def test_shell_theorem(tab, L=L_MAX):
    banner("TEST 5 -- Appendix A.6: n=0 shell theorem, and the zeta(2)-type series")

    # G_0 = 1/r_>  : constant inside the shell, Newtonian outside
    err = 0.0
    for x, xp in [(0.2, 0.6), (0.4, 0.6), (0.59, 0.6), (0.61, 0.6), (0.9, 0.6)]:
        s = S_spectral(0, x, xp, L, tab)[0]
        err = max(err, abs(s - 1.0 / max(x, xp)) * max(x, xp))
    check("G_0(x,x') = 1/x_>  (Newton's shell theorem)", err, 5e-6)

    # sum_l 1/((l+1/2)pi)^2 = 1/2
    a = alphas(0, L, tab)
    s = np.sum(1.0 / a**2)
    tail = 1.0 / (np.pi**2 * (L + 0.5))  # analytic tail estimate
    check("sum_l ((l+1/2)pi)^-2 = 1/2", abs(s + tail - 0.5), 1e-6)

    # E = 1 for every n = 0 eigenvalue
    check("E(alpha_{l0}) = 1", np.max(np.abs(E_norm(0, a) - 1.0)), 1e-12)

    # the reduced n=0 identity written out in the Remark
    err = 0.0
    for x, xp in [(0.3, 0.8), (0.8, 0.3), (0.55, 0.55)]:
        t = np.sin(a * x) * np.sin(a * xp) / a**2
        val = accelerate(t)[0]
        err = max(err, abs(val - min(x, xp) / 2.0) / (min(x, xp) / 2.0))
    check("sum sin(ax)sin(ax')/a^2 = x_</2", err, 5e-6)


# =====================================================================
# TEST 6 -- Corollary "eigenvalue rigidity"
# =====================================================================
def test_rigidity(tab, L=L_MAX):
    banner(
        "TEST 6 -- Corollary (eigenvalue rigidity): Dirichlet eigenvalues"
        " produce an image term"
    )

    worst, biggest_gap = 0.0, 0.0
    print("       n     x      x'    Dirichlet sum     Newton kernel    image/Newton")
    for n in [0, 1, 2, 4]:
        for x, xp in [(0.25, 0.7), (0.7, 0.25), (0.5, 0.5), (0.4, 0.95)]:
            acc = S_spectral(n, x, xp, L, tab, dirichlet=True)[0]
            ref_D = G_closed_dirichlet(n, x, xp)  # Newton MINUS image
            ref_N = G_closed(n, x, xp)  # pure Newton
            worst = max(worst, abs(acc - ref_D) / abs(ref_D))
            gap = abs(acc - ref_N) / abs(ref_N)
            biggest_gap = max(biggest_gap, gap)
            if (x, xp) in [(0.25, 0.7), (0.4, 0.95)]:
                print(
                    f"      {n:2d}  {x:5.2f}  {xp:5.2f}  {acc: .8e}  {ref_N: .8e}"
                    f"   {gap:8.1e}"
                )
    check("Dirichlet spectral sum = [x_<^n x_>^-(n+1) - (xx')^n]/(2n+1)", worst, 5e-5)
    check_min(
        "...and it differs from the pure Newton kernel",
        biggest_gap,
        1e-1,
        "(the spurious harmonic image)",
    )
    print("      -> only the Robin condition j_{n-1}(alpha)=0 kills the image;")
    print("         the eigenvalue condition is forced, not chosen.")


# =====================================================================
# TEST 7 -- Part II, Step 2: addition theorem + Legendre expansion of 1/|r-r'|
# =====================================================================
def cart(r, phi, lam):
    return np.array(
        [r * np.cos(phi) * np.cos(lam), r * np.cos(phi) * np.sin(lam), r * np.sin(phi)]
    )


def c_nm(n, m):
    """(2 - delta_0m) (n-m)! / (n+m)!  -- note lpmv's Condon-Shortley phase
    cancels because P_nm appears squared."""
    return (2.0 - (m == 0)) * np.exp(gammaln(n - m + 1) - gammaln(n + m + 1))


def test_addition_theorem():
    banner("TEST 7 -- Part II Step 2: addition theorem and 1/|r-r'| expansion")

    rng = np.random.default_rng(7)
    err = 0.0
    for _ in range(40):
        phi, phip = rng.uniform(-1.5, 1.5, 2)
        lam, lamp = rng.uniform(-np.pi, np.pi, 2)
        cg = np.sin(phi) * np.sin(phip) + np.cos(phi) * np.cos(phip) * np.cos(
            lam - lamp
        )
        for n in [0, 1, 2, 3, 6, 10]:
            s = sum(
                c_nm(n, m)
                * lpmv(m, n, np.sin(phi))
                * lpmv(m, n, np.sin(phip))
                * np.cos(m * (lam - lamp))
                for m in range(n + 1)
            )
            err = max(err, abs(s - lpmv(0, n, cg)))
    check("addition theorem for associated Legendre functions", err, 1e-9)

    # 1/|r-r'| = sum_n r_<^n / r_>^(n+1) P_n(cos gamma)
    err = 0.0
    for _ in range(20):
        r, rp = rng.uniform(0.10, 0.45), rng.uniform(0.60, 0.95)
        if rng.random() < 0.5:
            r, rp = rp, r
        phi, phip = rng.uniform(-1.5, 1.5, 2)
        lam, lamp = rng.uniform(-np.pi, np.pi, 2)
        cg = np.sin(phi) * np.sin(phip) + np.cos(phi) * np.cos(phip) * np.cos(
            lam - lamp
        )
        lo, hi = min(r, rp), max(r, rp)
        s = sum(lo**n / hi ** (n + 1) * lpmv(0, n, cg) for n in range(200))
        ref = 1.0 / np.linalg.norm(cart(r, phi, lam) - cart(rp, phip, lamp))
        err = max(err, abs(s - ref) / ref)
    check("Legendre expansion of the Newtonian kernel", err, 1e-8)


# =====================================================================
# TEST 8 -- Theorem (II) for a continuous density: the uniform sphere
# =====================================================================
def uniform_sphere_series(r, a_body, Rs, M, L, tab):
    """
    Interior spherical Bessel potential of a homogeneous ball of radius
    a_body and mass M, built literally from Eq. (coeffs) with n = m = 0.

    Radial mass integral (closed form, verified against quadrature below):
        int_M j_0(alpha r'/Rs) dm' = 3 M j_1(alpha Rt) / (alpha Rt),
        Rt = a_body / Rs.   (dm' has mass units and j_0 is dimensionless,
        so no factor of Rs can appear here.)
    """
    a = alphas(0, L, tab)
    Rt = a_body / Rs
    E = E_norm(0, a)  # == 1, kept explicit on purpose
    I = 3.0 * M * jn(1, a * Rt) / (a * Rt)
    A = 2.0 * (2 - 1) * (2 * 0 + 1) * 1.0 / (M * E) * I  # A^i_{l00}
    terms = jn(0, a * r / Rs) * A * lpmv(0, 0, 0.0)
    acc, _ = accelerate(terms)
    return G * M / Rs * acc


def uniform_sphere_exact(r, a_body, M):
    if r <= a_body:
        return G * M * (3 * a_body**2 - r**2) / (2 * a_body**3)
    return G * M / r


def test_uniform_sphere(tab, L=L_MAX):
    banner(
        "TEST 8 -- Theorem (II), continuous density: homogeneous ball,"
        " field points INSIDE the mass"
    )

    M, a_body = 1.0, 0.8
    # first: the closed-form radial integral used above really is the
    # mass integral of Eq. (coeffs)
    rho = 3 * M / (4 * np.pi * a_body**3)
    err = 0.0
    for Rs in [0.8, 1.0]:
        for al in alphas(0, 5, tab):
            num = quad(
                lambda rp, al=al, Rs=Rs: jn(0, al * rp / Rs) * rho * 4 * np.pi * rp**2,
                0.0,
                a_body,
                limit=400,
            )[0]
            Rt = a_body / Rs
            ana = 3.0 * M * jn(1, al * Rt) / (al * Rt)
            err = max(err, abs(num - ana) / abs(ana))
    check("mass integral in Eq.(coeffs) evaluated correctly", err, 1e-9)

    print("\n      Rs      r      r/a     V_series          V_exact          rel.err")
    worst = 0.0
    for Rs in [0.80, 1.00, 1.60]:
        for r in [0.0, 0.15, 0.4, 0.6, 0.79, 0.8, 0.85, 0.95 * Rs]:
            if r >= Rs:
                continue
            vs = uniform_sphere_series(r, a_body, Rs, M, L, tab)
            ve = uniform_sphere_exact(r, a_body, M)
            e = abs(vs - ve) / abs(ve)
            worst = max(worst, e)
            tag = "in " if r < a_body else "out"
            print(f"     {Rs:4.2f}  {r:5.3f}  {tag}   {vs: .10e}  {ve: .10e}  {e:8.1e}")
    check("V_i = true potential, inside and outside the mass", worst, 1e-6)
    print("      -> exactness holds for r < a (inside the body) as well as")
    print("         a < r < Rs, for every admissible choice of Rs.")


# =====================================================================
# TEST 9 -- Theorem (II) for a general body: the literal triple sum
# =====================================================================
def V_bessel_triple_sum(src, masses, field, Rs, Nmax, L, tab):
    """
    Literal evaluation of Eqs. (Vi) + (coeffs):

        V_i = (G M*/Rs) sum_l sum_n sum_m j_n(a_ln r/Rs) P_nm(sin phi)
                          [ A_lnm cos(m lam) + B_lnm sin(m lam) ]

    with A, B the mass integrals of Eq. (coeffs) (here discrete sums over
    point masses).  No use is made of Theorem (I) anywhere in this routine.
    Vectorised over l; the l-sum is accelerated at the very end.
    """
    rj, phij, lamj = src[:, 0], src[:, 1], src[:, 2]
    r, phi, lam = field
    Mstar = masses.sum()
    xj = rj / Rs
    x = r / Rs

    T = np.zeros(L)  # T[l] = contribution of radial mode l
    for n in range(Nmax + 1):
        a = alphas(n, L, tab)
        E = E_norm(n, a)
        jx = jn(n, a * x)  # (L,)
        jxj = jn(n, np.outer(a, xj))  # (L,J)
        pre = 2.0 * (2 * n + 1) / (Mstar * E)  # (L,)
        for m in range(n + 1):
            fac = (2.0 - (m == 0)) * np.exp(gammaln(n - m + 1) - gammaln(n + m + 1))
            Pfield = lpmv(m, n, np.sin(phi))
            if Pfield == 0.0:
                continue
            Psrc = lpmv(m, n, np.sin(phij))  # (J,)
            wc = masses * Psrc * np.cos(m * lamj)
            ws = masses * Psrc * np.sin(m * lamj)
            A = pre * fac * (jxj @ wc)
            B = pre * fac * (jxj @ ws)
            T += jx * Pfield * (A * np.cos(m * lam) + B * np.sin(m * lam))

    acc, raw = accelerate(T)
    return G * Mstar / Rs * acc, G * Mstar / Rs * raw


def V_newton(src, masses, field):
    p = cart(*field)
    tot = 0.0
    for (rj, phij, lamj), mj in zip(src, masses):
        tot += G * mj / np.linalg.norm(p - cart(rj, phij, lamj))
    return tot


def test_general_body(tab, L=L_MAX, Nmax=N_MAX):
    banner(
        "TEST 9 -- Theorem (II), general non-symmetric body:"
        " literal triple sum vs Newton"
    )

    # a lumpy, non-symmetric body: three point masses, Brillouin radius 0.75
    src = np.array(
        [
            [0.15, 0.30, 0.50],
            [0.30, -0.80, 2.10],
            [0.75, 1.10, -1.30],
        ]
    )
    masses = np.array([1.0, 2.0, 0.5])
    Rp = src[:, 0].max()
    Rs = 1.0

    fields = [
        ("outside all mass, inside Rs", (0.95, 0.20, 0.40)),
        ("outside all mass, inside Rs", (0.90, -1.00, 2.50)),
        ("INSIDE the mass shell", (0.50, 0.60, -0.90)),
        ("INSIDE the mass shell", (0.22, -0.40, 1.70)),
        ("near the centre", (0.05, 0.10, 0.10)),
    ]

    print(f"      Brillouin radius R* = {Rp:.3f},  reference radius Rs = {Rs:.3f}")
    print(f"      truncations: N_max = {Nmax}, L_max = {L}\n")
    print(
        "      field point                   V_series          V_Newton         rel.err"
    )
    worst = 0.0
    for label, f in fields:
        vs, _ = V_bessel_triple_sum(src, masses, f, Rs, Nmax, L, tab)
        vn = V_newton(src, masses, f)
        e = abs(vs - vn) / abs(vn)
        worst = max(worst, e)
        print(f"      r={f[0]:4.2f} {label:<26s} {vs: .8e}  {vn: .8e}  {e:8.1e}")
    check(
        "triple sum reproduces Newton's integral",
        worst,
        5e-4,
        "(limited by N_max truncation)",
    )
    print("      -> including field points INSIDE the mass distribution,")
    print("         where the exterior spherical-harmonic series is invalid.")
    return src, masses, Rp


# =====================================================================
# TEST 10 -- Remark on the quantifiers: different Rs, same potential
# =====================================================================
def test_Rs_independence(tab, src, masses, L=1200, Nmax=24):
    banner(
        "TEST 10 -- Remark (quantifiers): different Rs give different"
        " coefficients but the same V"
    )

    field = (0.55, 0.35, -0.70)
    vn = V_newton(src, masses, field)
    print("        Rs      V_series          rel.err vs Newton")
    worst = 0.0
    for Rs in [0.80, 1.00, 1.50, 2.20]:
        vs, _ = V_bessel_triple_sum(src, masses, field, Rs, Nmax, L, tab)
        e = abs(vs - vn) / abs(vn)
        worst = max(worst, e)
        print(f"      {Rs:5.2f}   {vs: .10e}   {e:8.1e}")
    check("V independent of the free parameter Rs (for r < Rs)", worst, 5e-4)

    # the coefficient sets really are different -- no scaling identity
    n = 2
    a = alphas(n, 8, tab)
    c1 = jn(n, a * (0.3 / 1.0))
    c2 = jn(n, a * (0.3 / 2.2))
    rel = np.max(np.abs(c1 - c2)) / np.max(np.abs(c1))
    check_min("coefficient sets for different Rs genuinely differ", rel, 1e-1)


# =====================================================================
# TEST 11 -- Theorem (III): convergence everywhere, exactness only r < Rs
# =====================================================================
def test_region_of_validity(tab, src, masses, L=1500, Nmax=30):
    banner(
        "TEST 11 -- Theorem (III): the series converges for r > Rs but"
        " NOT to the true potential"
    )

    src2 = src.copy()
    src2[:, 0] = np.array([0.25, 0.45, 0.70])
    Rs = 1.0

    print(
        "        r/Rs    V_series          V_Newton          rel.diff   |S(L)-S(L/2)|"
    )
    worst_inside = 0.0
    max_outside = 0.0
    for x in [0.80, 0.95, 0.99, 1.05, 1.20, 1.50, 1.90]:
        f = (x * Rs, 0.25, 0.65)
        vs, _ = V_bessel_triple_sum(src2, masses, f, Rs, Nmax, L, tab)
        vs_half, _ = V_bessel_triple_sum(src2, masses, f, Rs, Nmax, L // 2, tab)
        vn = V_newton(src2, masses, f)
        e = abs(vs - vn) / abs(vn)
        stab = abs(vs - vs_half) / abs(vs)
        flag = "  <- inside" if x < 1 else "  <- OUTSIDE"
        print(f"      {x:5.2f}   {vs: .8e}   {vn: .8e}   {e:8.1e}   {stab:8.1e}{flag}")
        if x < 1.0:
            worst_inside = max(worst_inside, e)
        else:
            max_outside = max(max_outside, e)

    check("exact for r < Rs", worst_inside, 1e-3)
    check_min(
        "NOT the true potential for r > Rs",
        max_outside,
        1e-1,
        "(unphysical continuation)",
    )
    print("      (the deviation grows continuously from 0 as r crosses Rs, so")
    print("       just outside the boundary it is necessarily small)")
    print("      -> the series remains finite and stable under refinement of L")
    print("         outside Rs (it converges); it simply converges to the")
    print("         potential of a different, unphysical density.")

    # a clean analytic illustration at n = 0: the eigenbasis
    # sin((l+1/2)pi x) is odd about x=0 and even about x=1, so beyond x=1
    # the resummation returns the MIRRORED kernel min(2-x, x')/(x x')
    print("\n      Analytic identification of the continuation (n = 0):")
    a = alphas(0, L_MAX, tab)
    err = 0.0
    for x, xp in [(1.30, 0.90), (1.55, 0.60), (1.80, 0.95)]:
        s = accelerate((2.0 / E_norm(0, a)) * jn(0, a * x) * jn(0, a * xp))[0]
        mirrored = min(2.0 - x, xp) / (x * xp)
        newton = 1.0 / max(x, xp)
        print(
            f"        x={x:4.2f}, x'={xp:4.2f}:  series={s: .6e}"
            f"   mirror={mirrored: .6e}   Newton={newton: .6e}"
        )
        err = max(err, abs(s - mirrored) / abs(mirrored))
    check("continuation for x>1 is the mirrored kernel", err, 1e-4)


# #####################################################################
# PART II -- figures, drawn with the routines verified in Part I
# #####################################################################
# ---------------------------------------------------------------------
# House style: serif/LaTeX text, colour-blind-safe palette, thin frames,
# inward ticks, and figures sized for a 7.2 in journal text block so the
# lettering stays legible at print size.
# ---------------------------------------------------------------------
mpl.rcParams.update(
    {
        "text.usetex": USETEX,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}"
        r"\usepackage{amssymb}"
        r"\usepackage{mathrsfs}"
        r"\usepackage{bm}",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 130,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": False,
        "lines.linewidth": 1.3,
        "lines.markeredgewidth": 0.9,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.75",
        "legend.fancybox": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 2.4,  # long enough to show dash patterns
        "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.35,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.compression": 6,
    }
)

C_SERIES = "#0077BB"  # blue   -- the Bessel series
C_CLOSED = "#E6001A"  # red    -- the closed form / ground truth
C_ALT = "#009988"     # teal   -- third case
C_ORANGE = "#F08C00"  # orange -- Dirichlet / second case
C_PURPLE = "#AA3377"  # purple -- fourth case
C_GREY = "#7f7f7f"
C_MARK = "#882255"    # dark red -- annotations on the continuation

FIGW = 7.2  # full text-block width, inches


def save(fig, stem, pdf=None):
    """Write one figure as vector PDF + PNG, and add it to the book."""
    fig.savefig(os.path.join(OUT, stem + ".pdf"))
    fig.savefig(os.path.join(OUT, stem + ".png"))
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


def band(ax, x0, x1, color="#cfe6f7", alpha=0.5):
    """Shade a region of validity behind everything else."""
    return ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0, zorder=-5)


def tag(ax, x, y, text, **kw):
    """Annotate at data-x, axes-fraction-y (never clipped by autoscaling)."""
    kw.setdefault("fontsize", 8)
    kw.setdefault("color", C_GREY)
    return ax.text(x, y, text, transform=ax.get_xaxis_transform(), **kw)

# ---------------------------------------------------------------------
# The accelerated sum as an explicit linear functional (needed to
# evaluate the kernel on a whole 2-D grid at once).
# ---------------------------------------------------------------------
def accel_weights(L):
    """
    Weight vector u with  sum_l u_l t_l  ==  accelerate(t)[0].
    (windowed Cesaro means + two Richardson stages, written out)
    """
    l = np.arange(L)

    def window(a, b):
        w = np.where(l < b, b - np.maximum(a, l), 0.0)
        return w / (b - a)

    u1 = window(L // 8, L // 4)
    u2 = window(L // 4, L // 2)
    u3 = window(L // 2, L)
    return (8.0 * u3 - 6.0 * u2 + u1) / 3.0


def kernel_grid(n, xs, xps, L, tab, dirichlet=False):
    """S_n(x,x') on the outer grid xs x xps, vectorised."""
    if dirichlet:
        a = tab[n][:L]
        w = 2.0 / (a**2 * jn(n - 1, a) ** 2)
    else:
        a = alphas(n, L, tab)
        w = 2.0 / E_norm(n, a)
    u = accel_weights(L) * w
    Jx = jn(n, np.outer(xs, a))  # (nx, L)
    Jp = jn(n, np.outer(xps, a))  # (np, L)
    return (Jx * u) @ Jp.T




# =====================================================================
# FIGURE 1 -- Theorem (I)
# =====================================================================
def fig1(tab):
    xp = 0.60
    xs = np.linspace(0.02, 1.0, 400)
    degrees = [0, 1, 2, 4]

    fig, axes = plt.subplots(2, 2, figsize=(FIGW, 5.3), layout="constrained")
    for ax, n in zip(axes.ravel(), degrees):
        ser = kernel_grid(n, xs, np.array([xp]), L_PLOT, tab)[:, 0]
        lo, hi = np.minimum(xs, xp), np.maximum(xs, xp)
        ref = lo**n * hi ** (-(n + 1)) / (2 * n + 1)

        ax.axvline(xp, color=C_GREY, ls=":", lw=0.9, zorder=1)
        ax.plot(
            xs,
            ref,
            "-",
            color=C_CLOSED,
            lw=2.8,
            alpha=0.5,
            solid_capstyle="round",
            zorder=2,
            label=r"closed form  $x_<^{n}\,/\,[(2n{+}1)\,x_>^{\,n+1}]$",
        )
        ax.plot(
            xs[::7],
            ser[::7],
            "o",
            color=C_SERIES,
            ms=3.2,
            mfc="none",
            mew=0.85,
            zorder=3,
            label=r"$\sum_{\ell} 2\,j_n j_n / \mathscr{E}$  ($L=%d$)" % L_PLOT,
        )
        tag(ax, xp, 0.04, r"$\;x=x'$", va="bottom", ha="left")
        ax.set_title(r"$n=%d\quad$(source at $x'=%.2f$)" % (n, xp))
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(r"$x=r/\check{R}^{*}_{e}$")
        ax.set_ylabel(r"$\mathcal{G}_{%d}(x,x')$" % n)

    # one shared legend under the grid, so no panel has to give up space
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        r"Theorem, Part (I): the $\ell$-sum of spherical Bessel products "
        r"collapses to a power law",
        fontsize=11,
    )
    return fig


# =====================================================================
# FIGURE 5 -- Theorem (II) for a real continuous density
# =====================================================================
def fig5(tab):
    M, a_body = 1.0, 0.8
    cols = [C_SERIES, C_ALT, C_PURPLE]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGW, 3.2),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    ax = axes[0]
    band(ax, 0, a_body, color="0.85", alpha=0.5)
    rr = np.linspace(0, 1.6, 400)
    ax.plot(
        rr,
        [uniform_sphere_exact(r, a_body, M) for r in rr],
        color=C_CLOSED,
        lw=2.8,
        alpha=0.5,
        zorder=1,
        label=r"exact potential",
    )
    for k, (Rs, col) in enumerate(zip([0.80, 1.00, 1.60], cols)):
        rr = np.linspace(0.0, 0.995 * Rs, 90)
        vs = np.array(
            [uniform_sphere_series(r, a_body, Rs, M, L_PLOT, tab) for r in rr]
        )
        ax.plot(
            rr[k::3],  # stagger the three sets so none is hidden by another
            vs[k::3],
            ["o", "s", "^"][k],
            ms=2.7,
            color=col,
            mfc="none",
            mew=0.85,
            zorder=2,
            label=r"series, $\check{R}^{*}_{e}=%.2f$" % Rs,
        )
    tag(ax, a_body / 2, 0.06, r"inside the mass", ha="center", color="0.35")
    ax.set_xlim(0, 1.62)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$V$")
    ax.set_title(r"homogeneous ball of radius $a=0.8$")
    ax.legend(loc="upper right", fontsize=7.6)

    ax = axes[1]
    for Rs, col in zip([0.80, 1.00, 1.60], cols):
        rr = np.linspace(0.005, 0.99 * Rs, 70)
        vs = np.array(
            [uniform_sphere_series(r, a_body, Rs, M, L_PLOT, tab) for r in rr]
        )
        ve = np.array([uniform_sphere_exact(r, a_body, M) for r in rr])
        ax.semilogy(
            rr,
            np.abs(vs - ve) / ve + 1e-17,
            lw=1.1,
            color=col,
            label=r"$\check{R}^{*}_{e}=%.2f$" % Rs,
        )
    ax.axvline(a_body, color=C_GREY, ls=":", lw=0.9)
    tag(ax, a_body, 0.05, r"$\;$body surface", ha="left", color="0.35", fontsize=7.5)
    ax.set_ylim(1e-17, 1e-5)
    ax.set_xlim(0, 1.62)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"relative error")
    ax.set_title(r"exactness holds through the surface")
    ax.legend(loc="upper left", fontsize=7.6)

    fig.suptitle(
        r"Theorem, Part (II) + Corollary (validity inside the mass)", fontsize=11
    )
    return fig


# =====================================================================
# FIGURE 6 -- Theorem (III): region of validity
# =====================================================================
def fig6(tab):
    """
    Two bodies, one message.  Left: a homogeneous ball almost filling the
    reference sphere -- exact to machine precision inside, departing right
    at the boundary.  Right: a lumpy three-lump body, scanned along a ray
    that stays clear of every source radius (so the Legendre truncation
    stays converged) -- same boundary, same behaviour.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIGW, 5.4),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.5, 1]},
    )

    # ---------------- left: homogeneous ball, a = 0.95, Rs = 1 -----------
    M, a_body, Rs = 1.0, 0.95, 1.0
    xs = np.linspace(0.02, 2.0, 220)
    vs = np.array(
        [uniform_sphere_series(x * Rs, a_body, Rs, M, L_PLOT, tab) for x in xs]
    )
    vn = np.array([uniform_sphere_exact(x * Rs, a_body, M) for x in xs])

    ax = axes[0, 0]
    band(ax, 0, 1)
    ax.plot(xs, vn, color=C_CLOSED, lw=2.8, alpha=0.5, zorder=1, label=r"true potential")
    ax.plot(
        xs[::4],
        vs[::4],
        "o",
        color=C_SERIES,
        ms=2.8,
        mfc="none",
        mew=0.85,
        zorder=2,
        label=r"interior Bessel series",
    )
    ax.axvline(1.0, color="k", lw=1.0)
    ax.axvline(a_body, color=C_GREY, ls=":", lw=0.9)
    ax.set_ylabel(r"$V$")
    ax.set_title(r"homogeneous ball, $R^{*}_{e}=0.95$, $\check{R}^{*}_{e}=1$")
    ax.legend(loc="upper right", fontsize=7.6)

    ax = axes[1, 0]
    band(ax, 0, 1)
    ax.semilogy(xs, np.abs(vs - vn) / np.abs(vn) + 1e-17, color=C_SERIES, lw=1.2)
    ax.axvline(1.0, color="k", lw=1.0)
    ax.set_ylim(1e-16, 5)
    ax.set_ylabel(r"relative deviation")
    ax.set_xlabel(r"$r/\check{R}^{*}_{e}$")
    tag(ax, 0.5, 0.09, r"exact to machine precision", ha="center", color="#11555f")
    ax.axvline(2 - a_body, color=C_MARK, ls="-.", lw=1.0)
    tag(ax, 2 - a_body, 0.22, r"$\;2-\tilde{R}_{e}$", color=C_MARK, ha="left")

    # ---------------- right: lumpy body, ray clear of all sources --------
    src = np.array([[0.20, 0.30, 0.50], [0.35, -0.80, 2.10], [0.50, 1.10, -1.30]])
    masses = np.array([1.0, 2.0, 0.5])
    Rs = 1.0
    xs2 = np.linspace(0.60, 2.0, 56)  # never crosses a source radius
    vs2, vn2 = [], []
    for x in xs2:
        f = (x * Rs, 0.25, 0.65)
        vs2.append(V_bessel_triple_sum(src, masses, f, Rs, N_POT, L_POT, tab)[0])
        vn2.append(V_newton(src, masses, f))
    vs2, vn2 = np.array(vs2), np.array(vn2)

    ax = axes[0, 1]
    band(ax, 0, 1)
    ax.plot(
        xs2,
        vn2,
        color=C_CLOSED,
        lw=2.8,
        alpha=0.5,
        zorder=1,
        label=r"Newton $G\!\int\!\mathrm{d}m'/\|%s{r}-%s{r}'\|$" % (BM, BM),
    )
    ax.plot(
        xs2,
        vs2,
        "o",
        color=C_SERIES,
        ms=2.8,
        mfc="none",
        mew=0.85,
        zorder=2,
        label=r"interior Bessel series",
    )
    ax.axvline(1.0, color="k", lw=1.0)
    ax.axvline(0.50, color=C_GREY, ls=":", lw=0.9)
    ax.set_ylabel(r"$V$")
    ax.set_title(r"three-lump body, $R^{*}_{e}=0.5$, $\check{R}^{*}_{e}=1$")
    ax.legend(loc="upper right", fontsize=7.6)
    tag(
        ax,
        1.52,
        0.70,
        "converges, but to an\nunphysical continuation",
        fontsize=8,
        color=C_MARK,
        ha="center",
        va="top",
        linespacing=1.4,
    )

    ax = axes[1, 1]
    band(ax, 0, 1)
    ax.semilogy(xs2, np.abs(vs2 - vn2) / np.abs(vn2) + 1e-16, color=C_SERIES, lw=1.2)
    ax.axvline(1.0, color="k", lw=1.0)
    ax.axhline(1e-6, color=C_GREY, ls="--", lw=0.9)
    ax.text(
        0.03,
        1.4e-6,
        r"$N_{\max}$ truncation floor",
        transform=ax.get_yaxis_transform(),
        fontsize=7.4,
        color="0.35",
        va="bottom",
    )
    ax.axvline(1.5, color=C_MARK, ls="-.", lw=1.0)
    tag(ax, 1.5, 0.22, r"$\;2-\tilde{R}_{e}$", color=C_MARK, ha="left")
    ax.set_ylim(1e-11, 5)
    ax.set_ylabel(r"relative deviation")
    ax.set_xlabel(r"$r/\check{R}^{*}_{e}$")

    for ax in axes[0]:
        ax.set_xlim(0, 2.0)
        tag(ax, 0.5, 0.08, r"exact", fontsize=9.5, color="#11555f", ha="center")
    for ax in axes[1]:
        tag(ax, 1.0, 0.90, r"$\;\check{R}^{*}_{e}$", va="top", ha="left", color="0.2")

    fig.suptitle(
        r"Theorem, Part (III): the series converges on all of "
        r"$\mathbb{R}^{3}$, but equals $V$ only for $r<\check{R}^{*}_{e}$"
        "\n"
        r"(dash-dotted: where the mirrored continuation first parts "
        r"company with Newton)",
        fontsize=10.5,
    )
    return fig


# =====================================================================
# main
# =====================================================================
def run_tests(tab):
    """Run every check in Part I; returns 0 iff they all pass."""
    t0 = time.time()
    test_boundary_condition(tab)
    test_norm_lemma(tab)
    test_local_green()
    test_generating_identity(tab)
    test_shell_theorem(tab)
    test_rigidity(tab)
    test_addition_theorem()
    test_uniform_sphere(tab)
    src, masses, Rp = test_general_body(tab)
    test_Rs_independence(tab, src, masses)
    test_region_of_validity(tab, src, masses)

    banner("SUMMARY")
    npass = sum(1 for _, ok in RESULTS if ok)
    for name, ok in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    print(f"  {npass}/{len(RESULTS)} checks passed ({time.time() - t0:.1f} s)")
    return 0 if npass == len(RESULTS) else 1


def run_figures(tab):
    """Draw each figure: one PDF + one PNG each, plus a combined document."""
    os.makedirs(OUT, exist_ok=True)

    # sanity: the linear-functional form of the accelerator must agree
    rng = np.random.default_rng(0)
    t = rng.normal(size=L_PLOT) / (np.arange(1, L_PLOT + 1) ** 2)
    assert abs(accel_weights(L_PLOT) @ t - accelerate(t)[0]) < 1e-14

    figures = [
        ("fig1_generating_identity", fig1),
        ("fig5_uniform_sphere", fig5),
        ("fig6_region_of_validity", fig6),
    ]
    book = os.path.join(OUT, "interior_bessel_gravity_figures.pdf")
    with PdfPages(book) as pdf:
        for stem, fn in figures:
            t1 = time.time()
            save(fn(tab), stem, pdf)
            print(f"  {stem}  ({time.time() - t1:.1f} s)")
        pdf.infodict()["Title"] = "Interior spherical Bessel gravity field -- figures"
    print(f"      -> {OUT}")
    print(f"      -> combined document: {os.path.basename(book)}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Verify the interior spherical Bessel gravity field and "
        "draw the paper figures (default: do both)."
    )
    ap.add_argument("--tests", action="store_true", help="run the checks only")
    ap.add_argument("--figures", action="store_true", help="draw the figures only")
    args = ap.parse_args(argv)
    do_tests = args.tests or not args.figures
    do_figs = args.figures or not args.tests

    t0 = time.time()
    print(__doc__.split("WHAT IS CHECKED")[0])

    # One eigenvalue table serves both halves: build it for whichever of
    # the two truncations is the more demanding.
    m_max = max(N_MAX - 1 if do_tests else 0, max(N_POT - 1, 8) if do_figs else 0)
    L = max(L_MAX if do_tests else 0, L_PLOT if do_figs else 0)
    print(
        f"Building eigenvalue table (roots of j_m, m <= {m_max}, "
        f"{L} roots each) ...",
        flush=True,
    )
    tab = zeros_table(m_max, L)
    print(f"  done in {time.time() - t0:.1f} s")

    status = 0
    if do_tests:
        status = run_tests(tab)
    if do_figs:
        banner("FIGURES")
        run_figures(tab)

    print(f"\n  total wall time {time.time() - t0:.1f} s")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
