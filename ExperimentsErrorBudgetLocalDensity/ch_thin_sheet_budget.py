"""TAG dM error budget and mass kernel in notes/ch_thin_sheet_error.pdf
(run from anywhere, ~2 min).

Part 1 applies the pipeline's own fit + Wahr inversion (same field points, same
SVD cutoff) to point masses rho*dh*dA from the DTMs, one assumption at a time.
Part 2 computes the same numbers from the mass kernel w(x) = h.g(x),
h = A+^T f_M, gives the upper bounds, and writes notes/ch_thin_sheet_kernel.pdf.
Part 3 checks the closed forms: rim formula and Cauchy-Schwarz bound for the
band limit, and the second-order thickness and cell-size terms.
"""
import os, sys, time, ast, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.chdir(ROOT)
import cylinder_mass_estimation_BENNU_TAG as T
t0 = time.time()
src = open(T.__file__).read(); i = src.index("    setup = dict(", src.index('if __name__ == "__main__"'))
node = ast.parse(src[i:].lstrip().split(chr(10)*2)[0]).body[0].value
setup = eval(compile(ast.Expression(node), "setup", "eval"), vars(T))
ch = T.calibrated_basis(setup, verbose=False); setup.update(alpha=ch["alpha"], m_max=ch["m_max"], n_max=ch["n_max"])
res = T.run_bennu_tag(**setup, cond=ch["cond"], do_covariance=True, verbose=False)
print(f"sigma_dM {res['sigma_dM']:.1f} kg = {abs(res['sigma_dM']/res['dM_true'])*100:.2f}%")
print(f"pipeline run {time.time()-t0:.0f}s  basis ({res['alpha']},{res['m_max']},{res['n_max']})")
A, cond, zd = res["design_matrix"], res["cond"], res["zeros_dict"]
Rs, al, M, N = res["R_star"], res["alpha"], res["m_max"], res["n_max"]
P, pp, z0, rho = res["pts_cart"], res["pp"], res["z_sheet"], res["density"]
GX, GY = np.meshgrid(res["gx"], res["gy"], indexing="ij")
if GX.shape != res["dh"].shape: GX, GY = np.meshgrid(res["gx"], res["gy"])
hp, hq, dh = res["h_pre"], res["h_post"], res["dh"]
dA = (res["gx"][1]-res["gx"][0])*(res["gy"][1]-res["gy"][0])
r = np.hypot(GX-res["cx"], GY-res["cy"])
ok = np.isfinite(dh) & (np.abs(dh) > 0)
Mt = res["dM_true"]
def field(x, y, z, dm, chunk=48):
    src = np.column_stack([x, y, z]); out = np.zeros(4*len(P))
    for i in range(0, len(P), chunk):
        sl = slice(i, i+chunk); d = P[sl, None, :]-src[None]
        rr = np.sqrt(np.einsum("ijk,ijk->ij", d, d)); ir3 = T.G_W/rr**3
        gx = -(ir3*d[..., 0])@dm; gy = -(ir3*d[..., 1])@dm; gz = -(ir3*d[..., 2])@dm
        c, s = np.cos(pp[sl]), np.sin(pp[sl]); n = len(rr)
        out[4*i:4*(i+n):4] = (T.G_W/rr)@dm
        out[4*i+1:4*(i+n):4] = gx*c+gy*s; out[4*i+2:4*(i+n):4] = -gx*s+gy*c; out[4*i+3:4*(i+n):4] = gz
    return out
def dM_of(b):
    c, *_ = T.fit_coefficients(A, b[0::4], b[1::4], b[2::4], b[3::4], cond=cond)
    return T.wahr_invert(c, Rs, al, M, N, zd)[0]
pct = lambda v: (v/Mt-1)*100
print(f"truth {Mt:+.4e}  pipeline dM_est {res['dM_est']:+.4e} ({pct(res['dM_est']):+.2f}%)  refined {res['dM_refined']:+.4e} ({pct(res['dM_refined']):+.2f}%)")
print(f"check wahr(d_coeffs) {pct(T.wahr_invert(res['d_coeffs'],Rs,al,M,N,zd)[0]):+.2f}%")
hm = 0.5*(hp+hq)
out = {}
for lab, msk in [("in R*", ok & (r < Rs)), ("patch", ok)]:
    x, y, d = GX[msk], GY[msk], dh[msk]; dm = rho*d*dA
    a = dM_of(field(x, y, np.full_like(x, z0), dm))
    b = dM_of(field(x, y, hm[msk], dm))
    o = d/(2*np.sqrt(3))
    c = dM_of(field(x, y, hm[msk]+o, dm/2)+field(x, y, hm[msk]-o, dm/2))
    out[lab] = (a, b, c)
    print(f"[{lab}] flat@z0 {pct(a):+.2f}%  mid-height pts {pct(b):+.2f}%  +thickness {pct(c):+.2f}%   ({time.time()-t0:.0f}s)")
a, b, c = out["patch"]; e = res["dM_est"]
print(f"budget (patch, % of truth): band+cutoff+leak {pct(a):+.2f} | wander {(b-a)/Mt*100:+.2f} | thickness {(c-b)/Mt*100:+.2f} | mesh/discret {(e-c)/Mt*100:+.2f} | total {pct(e):+.2f}")
ai, bi, ci = out["in R*"]; print(f"leakage from outside R* (flat) {(a-ai)/Mt*100:+.2f}%  band+cutoff only {pct(ai):+.2f}%")

# ── Part 2: mass kernel w(x) = h.g(x), the estimator's response to 1 kg at x ──
from scipy.special import jv, jn_zeros
fM = np.zeros(A.shape[1])          # Wahr: dM = sum_n (R*/G) J1(j0n/alpha) dA_0n
for n in range(1, N+1): fM[2*(n-1)] = (Rs/T.G_W)*jv(1, zd[0][n-1]/al)
Uu, sv, Vt = np.linalg.svd(A, full_matrices=False)
conds = [cond, 1e-4, 1e-6, 1e-8]
def hvec(cc):
    k = sv > cc*sv[0]; return Uu[:, k] @ ((Vt[k] @ fM)/sv[k])
H = np.column_stack([hvec(cc) for cc in conds])
print("kept singular values:", [int((sv > cc*sv[0]).sum()) for cc in conds], "of", len(sv))
def kern(x, y, z, chunk=48):
    """w at sources (x,y,z) for each column of H: transpose of field()."""
    src = np.column_stack([x, y, z]); w = np.zeros((len(x), H.shape[1]))
    for i in range(0, len(P), chunk):
        sl = slice(i, i+chunk); d = P[sl, None, :]-src[None]
        rr = np.sqrt(np.einsum("ijk,ijk->ij", d, d)); ir3 = T.G_W/rr**3
        c, s = np.cos(pp[sl]), np.sin(pp[sl]); h = H[4*i:4*(i+len(rr))].reshape(len(rr), 4, -1)
        hx = h[:, 1]*c[:, None]-h[:, 2]*s[:, None]; hy = h[:, 1]*s[:, None]+h[:, 2]*c[:, None]
        w += (T.G_W/rr).T @ h[:, 0]-(ir3*d[..., 0]).T @ hx-(ir3*d[..., 1]).T @ hy-(ir3*d[..., 2]).T @ h[:, 3]
    return w
x, y, d = GX[ok], GY[ok], dh[ok]; dm = rho*d*dA; rr = r[ok]; inR = rr < Rs; one = inR.astype(float)
hm_ = hm[ok]; zeta = hm_-z0; o = d/(2*np.sqrt(3))
w0 = kern(x, y, np.full_like(x, z0)); wm = kern(x, y, hm_)
wt = 0.5*(kern(x, y, hm_+o)+kern(x, y, hm_-o))
# ideal kernel (exact Fourier-Bessel projection, no fit): sum_n c_n J0(k rho) e^{k zeta} S(k dh)
Ra = al*Rs; j0 = jn_zeros(0, N); kn = j0/Ra; kN = kn[-1]
cn = 2*Rs*jv(1, kn*Rs)/(kn*Ra**2*jv(1, j0)**2)
S = lambda u: np.where(np.abs(u) < 1e-9, 1.0, np.sinh(u/2)/(u/2+1e-300))
wN = lambda rh, ze, dz: (cn*jv(0, np.outer(rh, kn))*np.exp(np.outer(ze, kn))*S(np.outer(dz, kn))).sum(1)
q = lambda v: v/Mt*100; adm = np.abs(dm)
def report(lab, a0, am, at):
    e0, et = a0-one, at-one
    print(f"[{lab}] signed b_i: basis {q(np.sum((e0*dm)[inR])):+.2f} outside {q(np.sum((e0*dm)[~inR])):+.2f} "
          f"wander {q(np.sum((am-a0)*dm)):+.2f} thickness {q(np.sum((at-am)*dm)):+.2f} total {q(np.sum(et*dm)):+.2f} | "
          f"any-density bound: basis {np.sum((np.abs(e0)*adm)[inR])/abs(Mt)*100:.1f} outside {np.sum((np.abs(e0)*adm)[~inR])/abs(Mt)*100:.1f} "
          f"wander {np.sum(np.abs(am-a0)*adm)/abs(Mt)*100:.1f} thickness {np.sum(np.abs(at-am)*adm)/abs(Mt)*100:.2f} "
          f"kappa {np.sum(np.abs(et)*adm)/abs(Mt):.2f}")
for j, cc in enumerate(conds): report(f"pipeline cond {cc:.1e}", w0[:, j], wm[:, j], wt[:, j])
report("ideal, all 22 zonal modes", wN(rr, 0*rr, 0*rr), wN(rr, zeta, 0*rr), wN(rr, zeta, d))
print(f"moved mass / net: {adm.sum()/abs(Mt):.2f} (inside R* {adm[inR].sum()/abs(Mt):.2f});  "
      f"mean |w-1| {np.average(np.abs(wt[:, 0]-one), weights=adm):.3f}, max {np.abs(wt[:, 0]-one).max():.2f}")
rg = np.linspace(1e-3, Ra-0.1, 20000); eg = np.abs(wN(rg, 0*rg, 0*rg)-(rg < Rs)); dg = np.abs(rg-Rs)
wg = wN(rg, 0*rg, 0*rg)
print(f"Gibbs: w_N(R*) = {wN(np.array([Rs]), np.zeros(1), np.zeros(1))[0]:.3f}, overshoot {max(wg[rg < Rs].max()-1, -wg[rg > Rs].min()):.3f}, "
      f"max |e| / min(1/2, 1/(pi kN |rho-R*|)) = {(eg/np.minimum(0.5, 1/(np.pi*kN*dg+1e-30))).max():.2f}")
# ── Part 3: closed forms for the band limit (rim), thickness and cell size ──
from scipy.special import sici
g1 = lambda u: 0.5-sici(kN*u)[0]/np.pi             # 1-D Gibbs ramp, int_0^inf g1 = 1/(pi kN)
print(f"rim ramp (Si) model: in {q(-np.sum((g1(Rs-rr)*dm)[inR])):+.2f} out {q(np.sum((g1(rr-Rs)*dm)[~inR])):+.2f}")
dl = 2/(np.pi*kN); ss = np.sum(dm[inR])/(np.pi*Rs**2)
si = np.sum(dm[inR & (rr > Rs-dl)])/(np.pi*(Rs**2-(Rs-dl)**2)); so = np.sum(dm[~inR & (rr < Rs+dl)])/(np.pi*((Rs+dl)**2-Rs**2))
print(f"rim formula (rings {dl:.2f} m): sig_in/sig* {si/ss:.3f} sig_out/sig* {so/ss:.3f}  b_in {q(-2*Rs*si/kN):+.2f} "
      f"b_out {q(2*Rs*so/kN):+.2f} total {q(-2*Rs*(si-so)/kN):+.2f}  (2/(pi kN R*) = {2/(np.pi*kN*Rs):.3f})")
Wn = 2*np.pi*Rs*jv(1, kn*Rs)/kn; Nn = np.pi*Ra**2*jv(1, j0)**2; TN = np.sqrt(np.pi*Rs**2-np.sum(Wn**2/Nn))
eb = np.arange(0, rr.max()+0.3, 0.3); ar = np.pi*(eb[1:]**2-eb[:-1]**2); sb = np.bincount(np.digitize(rr, eb)-1, dm, len(ar))/ar
nb = np.sqrt(np.sum(sb**2*ar)); tail = np.sqrt(nb**2-np.sum(Nn*(np.array([np.sum(dm*jv(0, k*rr)) for k in kn])/Nn)**2))
print(f"Cauchy-Schwarz: T_N {TN:.3f} m (sqrt(2R*/kN) {np.sqrt(2*Rs/kN):.3f}), ring-mean tail {tail/nb*100:.0f}% -> |b_band| <= {abs(TN*tail/Mt)*100:.1f}%")
dz, a = 0.1, np.sqrt(dA)   # w harmonic in x: cell a*a*dh average = w + (dh^2-a^2)/24 d2w/dz2
d2 = (kern(x, y, hm_+dz)[:, 0]+kern(x, y, hm_-dz)[:, 0]-2*wm[:, 0])/dz**2
print(f"second order: thickness {q(np.sum(d**2/24*d2*dm)):+.2f}%  cell {q(-a**2/24*np.sum(d2*dm)):+.3f}% "
      f"(a {a:.2f} m, (kmax a)^2/24 {(zd[M-1][-1]/Ra*a)**2/24*100:.1f}%)")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.style.use("default"); plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "axes.spines.top": False})
B, O = "#1f5fa8", "#e07b00"
fig, ax = plt.subplots(figsize=(6.6, 2.5)); bx = ax.twinx()
eb = np.arange(0, 24.5, 0.5); cb = 0.5*(eb[1:]+eb[:-1]); ib = np.digitize(rr, eb)-1
mu = np.array([w0[ib == i, 0].mean() for i in range(len(cb))]); sd = np.array([w0[ib == i, 0].std() for i in range(len(cb))])
share = np.array([adm[ib == i].sum() for i in range(len(cb))])/adm.sum()/0.5
bx.bar(cb, share, width=0.45, color="0.88", zorder=0); bx.set_ylabel("moved mass per m (share)", color="0.5")
bx.tick_params(colors="0.5"); bx.set_ylim(0, 3*share.max()); bx.spines["top"].set_visible(False)
ax.set_zorder(1); ax.patch.set_visible(False)
rp = np.linspace(0.01, 24, 2000); env = np.minimum(0.5, 1/(np.pi*kN*np.abs(rp-Rs)))
ax.fill_between(rp, (rp < Rs)-env, (rp < Rs)+env, color=B, alpha=0.12, lw=0, label="Gibbs envelope")
ax.plot(rp, rp < Rs, "k", lw=0.8, label="ideal $1_{R^*}$")
ax.plot(rp, wN(rp, 0*rp, 0*rp), color=B, lw=1.2, label="$w_N$ (exact projection)")
ax.fill_between(cb, mu-sd, mu+sd, color=O, alpha=0.25, lw=0); ax.plot(cb, mu, "o", ms=2.5, color=O, label="$w$ (pipeline), mean $\\pm$ sd")
ax.axvline(Rs, color="0.6", lw=0.6, ls=":"); ax.set_xlim(0, 24); ax.set_ylim(-0.4, 1.6)
ax.set_xlabel("distance from TAG centre $\\rho$ [m]"); ax.set_ylabel("mass kernel $w$ on $z_0$")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "notes", "ch_thin_sheet_kernel.pdf")); print(f"done {time.time()-t0:.0f}s")
