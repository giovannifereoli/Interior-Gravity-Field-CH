"""TAG thin-sheet error check: numbers in notes/ch_thin_sheet_error.pdf (run from anywhere)."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.chdir(ROOT)
import cylinder_mass_estimation_BENNU_TAG as T
from scipy.special import jn_zeros, jv
Rs, al, M, N = 8.0, 6.0, 8, 22; Ra = al*Rs
Pp = T.load_terrain_points("3dmeshes/Bennu_preTag.obj"); Pq = T.load_terrain_points("3dmeshes/Bennu_afterTag.obj")
gx, gy, GX, GY = T.common_grid(Pp, Pq, grid_res=0.30)
hp = T.height_map(Pp, GX, GY); hq = T.height_map(Pq, GX, GY); dh = hq-hp
dA = (gx[1]-gx[0])*(gy[1]-gy[0])
cx, cy = T.locate_tag_site(dh, GX, GY); r = np.hypot(GX-cx, GY-cy); foot = r < Rs
z0 = T.differential_sheet_plane(hp, hq, foot, fallback=float(hp[foot].mean()))
d = dh[foot]; zeta = 0.5*(hp+hq)[foot]-z0; rho = r[foot]; phi = np.arctan2((GY-cy)[foot], (GX-cx)[foot])
w = np.abs(d); mv = w > 1e-3
print(f"z0={z0:.3f}  pre-mean={hp[foot].mean():.3f}  cells={foot.sum()} moved(|dh|>1mm)={mv.sum()}")
print(f"dh: min {d.min():.3f} max {d.max():.3f}  sum dh*dA={d.sum()*dA:.2f} m3  exc {d[d<0].sum()*dA:.2f} dep {d[d>0].sum()*dA:.2f}")
q = lambda x, p: np.percentile(x, p)
# |dh|-weighted stats (mass-weighted)
def wq(x, p):
    i = np.argsort(x); c = np.cumsum(w[i]); return x[i][np.searchsorted(c, p/100*c[-1])]
print(f"|dh| mass-weighted: median {wq(w,50):.3f} p90 {wq(w,90):.3f} p99 {wq(w,99):.3f} max {w.max():.3f}  rms {np.sqrt(np.average(d**2,weights=w)):.3f}")
print(f"zeta mass-weighted: mean(signed dh) {np.sum(d*zeta)/d.sum():.2e}  mean|dh|-w {np.average(zeta,weights=w):.3f} std {np.sqrt(np.average(zeta**2,weights=w)):.3f} min {zeta[mv].min():.3f} max {zeta[mv].max():.3f}")
s2 = np.sum(d*(zeta**2+d**2/12))/d.sum(); print(f"s (signed) = {np.sqrt(s2) if s2>0 else s2:.3f}  thickness part sqrt(sum d*d^2/12/sum d)={np.sqrt(np.sum(d**3/12)/d.sum()) if np.sum(d**3)/d.sum()>0 else 'neg'}")
kmax = jn_zeros(M-1, N)[-1]/Ra
S = lambda u: np.where(np.abs(u) < 1e-8, 1.0, np.sinh(u/2)/(u/2+1e-300))
for lab, k in [("kmax", kmax), ("1/R*", 1/Rs)]:
    e = S(k*w)-1
    print(f"thickness err at {lab} (k={k:.3f}): mass-w mean {np.average(e,weights=w)*100:.2f}%  max {e.max()*100:.1f}%  mass frac >5%: {w[e>0.05].sum()/w.sum()*100:.1f}%")
print(f"mass frac with |dh|>0.66: {w[w>0.66].sum()/w.sum()*100:.1f}%;  lam where max|dh| gives 5%: {2*np.pi*w.max()/1.0874:.2f} m; p90: {2*np.pi*wq(w,90)/1.0874:.2f} m")
# exact in-model projection: thin-sheet return = proj(drho * G_k) per mode
modes = [(m, n, jn_zeros(m, N)[n-1]) for m in range(M) for n in range(1, N+1)]
def recon(Gfun):
    out = np.zeros_like(d); sig = {}
    for m, n, j in modes:
        k = j/Ra; norm = (2*np.pi if m == 0 else np.pi)*Ra**2/2*jv(m+1, j)**2
        B = jv(m, k*rho); f = d*Gfun(k)
        for tr, c in ((np.cos, 1), (np.sin, 0)):
            if m == 0 and c == 0: continue
            bas = B*tr(m*phi); a = np.sum(f*bas)*dA/norm; out += a*bas
    return out
G1 = lambda k: 1.0
Gt = lambda k: S(k*d)
Gw = lambda k: np.exp(k*zeta)
Gb = lambda k: np.exp(k*zeta)*S(k*d)
ref = recon(G1); pk = np.abs(ref).max(); M0 = ref.sum()*dA
print(f"band-limited truth: peak {pk:.3f} m(dh)  dM(bl) {M0:.2f} m3 vs true {d.sum()*dA:.2f}")
for lab, G in [("thickness only", Gt), ("wander only", Gw), ("both", Gb)]:
    rr = recon(G); e = rr-ref
    print(f"{lab:15s}: dM err {(rr.sum()*dA/M0-1)*100:+.2f}%   map err RMS {np.sqrt(np.mean(e**2))/pk*100:.2f}%  median|e| {np.median(np.abs(e))/pk*100:.2f}%  max {np.abs(e).max()/pk*100:.1f}% of peak")
print(f"zeta unweighted over foot: rms {np.sqrt(np.mean(zeta**2)):.3f} std {zeta.std():.3f};  h_mid std {(0.5*(hp+hq))[foot].std():.3f}; hpre std {hp[foot].std():.3f}")
ks = np.array([j/Ra for _,_,j in modes]); print(f"modes with lam<5.97m: {(ks>2*np.pi/5.97).sum()}/{len(ks)};  lam<13.3m: {(ks>2*np.pi/13.3).sum()}")
# mass error by m=0 radial index n (wander)
cum = 0; tot = []
for m, n, j in modes:
    if m: continue
    k = j/Ra; B = jv(0, k*rho); norm = 2*np.pi*Ra**2/2*jv(1, j)**2
    da = np.sum(d*(np.exp(k*zeta)-1)*B)*dA/norm; W = 2*np.pi*Rs*Ra*jv(1, j*Rs/Ra)/j
    tot.append(da*W)
tot = np.array(tot); print("dM wander err (% of bl) cumulative by n:", np.round(np.cumsum(tot)/M0*100, 1))
print("signal share |a_n^true| m=0 by n:", np.round([abs(np.sum(d*jv(0, j/Ra*rho))*dA) for _,_,j in [x for x in modes if x[0]==0]], 1))
for f in [1.5, 2, 2.5, 3, 4]:
    rr = recon(lambda k: S(k*f*d)); e = rr-ref
    print(f"thickness x{f}: max|dh| {f*w.max():.2f} m  dM {(rr.sum()*dA/M0-1)*100:+.2f}%  map RMS {np.sqrt(np.mean(e**2))/pk*100:.2f}% median {np.median(np.abs(e))/pk*100:.2f}%")
# per-column wander / combined errors (volume-weighted)
az = np.abs(zeta)
print(f"|zeta| vol-w: median {wq(az,50):.3f} p90 {wq(az,90):.3f} max {az.max():.3f}; mean zeta {np.average(zeta,weights=w):.3f} rms {np.sqrt(np.average(zeta**2,weights=w)):.3f}")
for eps in (0.05, 0.01):
    print(f"vol frac |zeta|<=eps/kmax ({eps}): {w[az<=eps/kmax].sum()/w.sum()*100:.1f}%;  lam for rms/p90/max zeta: "
          f"{2*np.pi*np.sqrt(np.average(zeta**2,weights=w))/np.log(1+eps):.0f} {2*np.pi*wq(az,90)/np.log(1+eps):.0f} {2*np.pi*az.max()/np.log(1+eps):.0f} m")
k01 = jn_zeros(0, 1)[0]/Ra
for lab, k in [("kmax", kmax), ("1/R*", 1/Rs), ("k01", k01)]:
    ew = np.exp(k*zeta)-1; eb = np.exp(k*zeta)*S(k*w)-1
    print(f"{lab} k={k:.3f}: wander |e| vol-w mean {np.average(np.abs(ew),weights=w)*100:.1f}% median {wq(np.abs(ew),50)*100:.1f}% max {np.abs(ew).max()*100:.0f}%  | both mean {np.average(np.abs(eb),weights=w)*100:.1f}% median {wq(np.abs(eb),50)*100:.1f}%")
