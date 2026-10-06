# Cylindrical Harmonics for Small-Body Gravity and Interior Density Estimation

**Giovanni Fereoli<sup>1,2</sup> · Jay W. McMahon<sup>1,2</sup>**  
<sup>1</sup>Ann and H.J. Smead Department of Aerospace Engineering Sciences, University of Colorado Boulder  
<sup>2</sup>Colorado Center for Astrodynamics Research (CCAR), University of Colorado Boulder

---

## Overview

This repository develops a **cylindrical harmonic (Fourier–Bessel) gravity expansion** for small
irregular bodies: an **accurate, locally convergent, and computationally efficient** representation
of the gravitational potential that remains valid *below* the Brillouin sphere, where the classical
exterior spherical harmonic (SH) series diverges. The formulation carries the full set of analytical
partials required for estimation, so it drops directly into orbit determination (OD), proximity
operations, and landing dynamics.

The work has since grown past forward modeling into **inference**. The question the current
experiments ask is:

> Near-surface data expanded in cylindrical harmonics (CH) carries localized information that global
> spherical harmonics smear out. Does adding CH to SH let us recover **where the mass is inside the
> body** — the mass fraction *and* the position of interior anomalies — when SH alone cannot?

The answer is pursued at two scales, which is what the upcoming conference paper is about:

* **Global scale** — a homogeneous polyhedron scaled by β̃ = 1 − Σβ, plus a handful of interior
  mascons carrying the departures from homogeneity. SH alone, CH alone, and SH + a network of CH
  cylinders are compared on identical data, with exact posterior covariances and Monte Carlo
  validation.
* **Local scale** — the OSIRIS-REx TAG event at Bennu, where the mass moved by the sampling
  contact is estimated from the *difference* of CH coefficients fitted before and after, and
  checked against the geometric ground truth from the DTM itself.

---

## Papers

| Status | Work |
|---|---|
| **Published** | *Interior Gravity Characterization of Small Celestial Bodies Using Cylindrical Harmonics* — **Celestial Mechanics and Dynamical Astronomy** 138, 12 (2026). [doi:10.1007/s10569-026-10281-7](https://doi.org/10.1007/s10569-026-10281-7) |
| **Published** | *On Cylindrical Harmonics for Local Gravity Field Modeling* — 2025 AAS/AIAA Astrodynamics Specialist Conference, Boston, MA. The conference precursor to the CMDA article. |
| **In preparation** | *Small-Body Density Estimation with Cylindrical Harmonics at Global and Local Scales* — 37th AAS/AIAA Space Flight Mechanics Meeting, New Orleans, LA, 17–21 January 2027. **This is the paper the `cylinder_mass_estimation_*` scripts produce.** |
| **Planned (2027)** | A journal extension of the SFM paper. **Undecided**: the title, whether it is one paper or two (the global-scale density experiments and the local/TAG-scale results may not belong in the same venue), and the venue itself — *Journal of Guidance, Control, and Dynamics*, *Journal of Geodesy*, and *CMDA* are all under consideration. |
| **Theory note (draft)** | *Closed-form resummation and exactness of the interior spherical Bessel gravity field*. Establishes that the interior spherical-Bessel expansion is **exact**, not merely convergent, inside the source region — and where that exactness stops. Verified end to end by [verify_interior_bessel_gravity.py](verify_interior_bessel_gravity.py). |

The interior-expansion machinery builds on **Y. Takahashi and D. J. Scheeres, "Small body surface
gravity fields via spherical harmonic expansions," *Celest. Mech. Dyn. Astron.* 119, 169–206
(2014)**, [doi:10.1007/s10569-014-9552-9](https://doi.org/10.1007/s10569-014-9552-9).

---

## What is in here

### Density estimation — the SFM 2027 paper

| File | Experiment |
|---|---|
| [cylinder_mass_estimation_GLOBAL.py](cylinder_mass_estimation_GLOBAL.py) | **Part 1.** One near-surface CH cylinder over one shallow anomaly, plus two deep lobe anomalies, on Eros. Compares `SH`, `CH`, `SH + CH` for mass fraction and anomaly position. Owns all the shared machinery (observables, noise model, covariances, figures) that the other parts import. |
| [cylinder_mass_estimation_GLOBAL_pt2.py](cylinder_mass_estimation_GLOBAL_pt2.py) | **Part 2.** A *network* of CH cylinders around the body against ~6 arbitrarily placed anomalies: SH, 6-CH, SH + 1 CH, SH + 2 CH, SH + 6-CH. Imports Part 1 as `G`. |
| [ExtraGoodCode/cylinder_mass_estimation_GLOBAL_pt3.py](ExtraGoodCode/cylinder_mass_estimation_GLOBAL_pt3.py) | **Part 3.** *Where* should the cylinders go? SH-error and Fisher-information placement criteria against plain farthest-point geometric sampling. |
| [cylinder_mass_estimation_BENNU_TAG.py](cylinder_mass_estimation_BENNU_TAG.py) | **Local scale.** Mass moved by the OSIRIS-REx TAG event, from ΔCH coefficients fitted pre/post, validated against the DTM height change, with full covariance propagation Σ<sub>Δc</sub> → Σ<sub>ΔM</sub>. |

Each script prints a numbered set of tables and writes one vector PDF per figure panel into
`Images/`, under its own filename prefix (`global_`, `global_pt2_`, `bennu_tag_`). Tables carry the
numbers; figures carry none, so a figure can be moved around a manuscript freely.

### Orbit determination and uncertainty quantification

| File | Purpose |
|---|---|
| [cov_analysis_SRIF.py](cov_analysis_SRIF.py) | Multi-arc SRIF covariance analysis over the pt1 polar cylinder: which CH degree m and order n a realistic campaign resolves. 30 low-pass sorties in the rotating frame (polyhedral + mascon truth): each drops from a circular polar home orbit (`R_HOME_KM` = 25 / 35 / 50 km, the sortie apoapsis; vis-viva periapsis speed) to a periapsis 1.5–5 km above the cylinder base and up to ±0.85 R off its axis, then recircularizes (Δv printed); one inertial plane (`HOME_PSI`, normal along the Sun's equatorial projection), the sortie timing (body phase at periapsis, 24 steps) sets the pass azimuth; tracking arc ±2.5 h about periapsis, between the two apoapsis burns; screened for ≥ 1 km altitude and picked greedily for (ρ, φ, z) coverage; DSN radiometric tracking along the Earth line of sight (two-way X-band Doppler, 60 s counts, 0.1 mm/s; two-way ranging every 10 min, 1 m; no data while Eros occults the Earth, ray-cast on the shape model) and NavCam landmarks with visibility/occlusion; per-epoch spacecraft attitude knowledge error shared by every landmark of an image (correlated measurement noise; radio unaffected); per-arc locals (r, v, batched stochastic accelerations — piecewise constant, 1 h batches, restarted at every batch); global RAW CH coefficients A_mn, B_mn of the `CH_MASTER` = (12, 12) basis (276, B_0n dropped; degree m ≤ 11, order n ≤ 12: the density scripts' "new" modes; neighbouring orders are nearly dependent, so the OD knows the field while single coefficients stay near their prior; no rotation; truth defined as c = Φ⁺f at `TRUTH_RCOND` = 1e-6; prior = the FULL sample covariance of the same refit over a Monte Carlo of 300 shape models (30 m radial error, 1 km correlation) × bulk densities (±1.1%), scaled so the largest truth coefficient sits at `PRIOR_SNR_MAX` = 1 prior σ (×9.3e4, σ × 305: every prior SNR ≤ 1), rank 35 of 276 (the rank of Φ⁺): the filter carries c = σ ∘ (Q z), z ~ N(0, I); subset / truncation solutions use the data information of an identity-prior run; `PRIOR_*` constants) and landmarks. Consider parameters: range bias (ranging only — Doppler unbiased) and camera alignment (4 columns); the CH omission enters separately as a deterministic bias. Reports resolved (formal) and accurate (formal + consider + omission) coefficients per degree and order, DFS (r − tr Σ_z under the correlated prior), the largest complete square (N, N), a truncation sweep (m < N, n ≤ N), a consider error budget and a BASIS SWEEP (`BASIS_CASES`: the nominal filter re-run on (6, 6), (8, 8), (12, 6), (12, 4), (12, 3) and on (12, 12) at a looser truth rcond 1e-4 — omission, resolved, SNR > 3 formal and total, accurate, DFS). Exports `od_ch_covariance.npz` (nominal Σ_c, formal and with consider, truth, OD bias, the prior covariance `Sigma_prior_ch`; plus Σ_c of every density-script preset the basis contains — old (8, 8) and new (12, 12); `load_od_covariance(consider=True)`, `od_whitener`) to replace `od_sigma`. Figures (`Images/od_fig1–5`): geometry (flyovers, cylinder, landmarks, Earth-occulted epochs), navigation 1σ per filter epoch (scatter, end-of-arc formal and formal + consider), A_mn, B_mn SNR per degree vs flyovers, A_mn, B_mn correlations of the informed degrees (ordered m, then A | B, then n), degree spectrum (truth / prior / posterior). Real LaTeX by default; `OD_NO_TEX=1` falls back to mathtext. |
| [cylindrical_acc_pot_UQ.py](cylindrical_acc_pot_UQ.py) | Uncertainty quantification on the fitted coefficients. |
| [cylindrical_acc_pot_OPT.py](cylindrical_acc_pot_OPT.py) | Least-squares fit of CH coefficients to polyhedral gravity (potential and acceleration jointly). |
| `cylindrical_acc_pot_SHORT_fitting_both_*.py` | Fitting variants: normalized units (`NORM`), moving cylinder (`MOV`), polar cap (`TOP`), TAG geometry (`TEST_TAG`), and two-mascon inversion with mass conservation (`INVERSION_2m_*`). |
| [cylindrical_pot_short_integration_MCI.py](cylindrical_pot_short_integration_MCI.py) | Monte Carlo integration of the potential, used as an independent check on the analytic coefficients. |

### Theory verification

[verify_interior_bessel_gravity.py](verify_interior_bessel_gravity.py) checks every claim of the
interior spherical-Bessel result numerically — the two lemmas, the Green's-function construction,
all three parts of the main theorem, the shell theorem, and the region of validity — and then draws
the paper's figures with the *same* verified routines, so a curve can never drift away from the test
that certifies it.

```bash
python3 verify_interior_bessel_gravity.py            # checks + figures
python3 verify_interior_bessel_gravity.py --tests    # checks only
python3 verify_interior_bessel_gravity.py --figures  # figures only
```

### Data and utilities

* `3dmeshes/` — shape models: Eros (12k/50k), Itokawa, Bennu, 67P/Churyumov–Gerasimenko, plus
  hollow and non-uniform density variants, and the Bennu `preTag`/`afterTag` DTM patches of the
  Nightingale site. **The TAG OBJs are local DTM patches, not closed polyhedra** — they have holes
  and a detached bottom plate, and are rebuilt into watertight slabs by the pipeline.
* [mesh_utility.py](mesh_utility.py), [mesh_plotting.py](mesh_plotting.py),
  [plot_cylinders_mascons.py](plot_cylinders_mascons.py) — mesh loading and the 3-D geometry figures.
* `VariousExperiments/`, `ExtraGoodCode/` — earlier and side experiments (MCMC, Slepian functions,
  Poisson/polyhedron tests, landing measurement models) kept for provenance.
* `Images/` — generated output; not tracked.

---

## Running the experiments

Requires Python 3.11+ with `numpy`, `scipy`, `matplotlib`, `trimesh`, `polyhedral-gravity`, and
`tqdm`. A full run of a density-estimation script takes minutes, not seconds: the Monte Carlo
validation draws hundreds of random interiors.

```bash
python3 cylinder_mass_estimation_GLOBAL.py       # Part 1  -> Images/global_*.pdf
python3 cylinder_mass_estimation_GLOBAL_pt2.py   # Part 2  -> Images/global_pt2_*.pdf
python3 cylinder_mass_estimation_BENNU_TAG.py    # TAG     -> Images/bennu_tag_*.pdf
```

Figure text is rendered with matplotlib's mathtext by default; set `USE_TEX = True` at the top of a
script for real LaTeX fonts, at a large cost in runtime. `FONT_SCALE` is the single knob for every
text size, sized so a figure dropped into a two-column manuscript at `\linewidth` stays legible.

**A note on what the numbers mean.** Every position uncertainty these scripts report is a 3-D norm
on |Δp| from the **joint** fit, with all anomalies free simultaneously — not a per-coordinate sigma,
and not the more optimistic one-at-a-time experiment that hands the estimator its neighbours as
known. Mass-fraction and position tables quote the analytic 1σ beside the Monte Carlo RMS so the
two can be read against each other, and gains are reported as ratios against the SH-only baseline.

---

## Citation

If you use this code or reproduce results from the papers, please cite:

```bibtex
@article{Fereoli2026_CylindricalHarmonics,
  author    = {Fereoli, Giovanni and McMahon, Jay W.},
  title     = {Interior Gravity Characterization of Small Celestial Bodies Using Cylindrical Harmonics},
  journal   = {Celestial Mechanics and Dynamical Astronomy},
  year      = {2026},
  volume    = {138},
  pages     = {12},
  doi       = {10.1007/s10569-026-10281-7},
  url       = {https://doi.org/10.1007/s10569-026-10281-7}
}

@inproceedings{Fereoli2025_LocalGravity,
  author    = {Fereoli, Giovanni and McMahon, Jay W.},
  title     = {On Cylindrical Harmonics for Local Gravity Field Modeling},
  booktitle = {AAS/AIAA Astrodynamics Specialist Conference},
  address   = {Boston, Massachusetts},
  year      = {2025}
}
```

The density-estimation paper is not published yet; its entry will be added once the SFM 2027
preprint number is assigned.

---

## License

Released under the **MIT License**. See the [LICENSE](LICENSE) file for details.
