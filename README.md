```
# Interior Gravity Characterization of Small Celestial Bodies Using Cylindrical Harmonics

**Giovanni Fereoli<sup>1,2</sup> · Jay W. McMahon<sup>1,2</sup>**  
<sup>1</sup>Ann and H.J. Smead Department of Aerospace Engineering Sciences, University of Colorado Boulder  
<sup>2</sup>Colorado Center for Astrodynamics Research (CCAR), University of Colorado Boulder  

---

## 🪐 Overview

This repository accompanies the paper:

> **Fereoli, G. & McMahon, J.W. (2025)**  
> *Interior Gravity Characterization of Small Celestial Bodies Using Cylindrical Harmonics*  
> *Celestial Mechanics and Dynamical Astronomy* (under review)

The project introduces a **cylindrical harmonic (Fourier–Bessel) gravity expansion** that provides an **accurate, locally convergent, and computationally efficient** representation of the gravitational potential of small irregular bodies.  
The formulation is particularly suited for **spacecraft proximity operations**, **orbit determination (OD)**, and **landing dynamics** on asteroids and comets.

---

## ⚙️ Key Features

- 🌀 **Interior Cylindrical Harmonic Expansion**  
  Derived analytically from Laplace’s equation in cylindrical coordinates.  
  Employs Fourier–Bessel functions guaranteeing convergence inside a cylindrical domain.

- 🧮 **Efficient Gravity Reconstruction**  
  Coefficients estimated via least-squares fitting to polyhedral reference fields.  
  Achieves sub-percent RMS errors in potential and acceleration.

- 🛰️ **Mission-Relevant Applications**  
  Enables precise local gravity modeling for trajectory design, OD, and landing-site analysis.

- 📈 **Validated Results**  
  Benchmarked on the Gaskell (2008) Eros shape model and Dawn FC2 datasets.  
  Demonstrates faster convergence and improved accuracy over spherical harmonics.

---

## 🧩 Repository Structure

```

📦 CylindricalHarmonics_GravityModel
├── README.md                     # Project overview (this file)
├── paper/                        # LaTeX source for the submitted manuscript
│   ├── main.tex
│   ├── figures/
│   ├── references.bib
│   └── sn-jnl.cls
├── src/                          # Core Python code
│   ├── bessel_basis.py           # Fourier–Bessel basis & normalization
│   ├── gravity_fit.py            # Coefficient estimation (least squares)
│   ├── gravity_eval.py           # Potential and acceleration computation
│   └── visualization.py          # Plotting and convergence diagnostics
├── data/
│   ├── shape_model/              # Gaskell (2008) Eros triangular mesh
│   ├── polyhedral_reference/     # Reference polyhedral gravity data
│   └── fitted_coefficients/      # Stored cylindrical harmonic coefficients
└── examples/
├── compare_polyhedral_vs_cyl.py  # Reproduce paper validation figures
└── propagate_trajectory.py       # Example trajectory propagation test

````

---

## 🧠 Theoretical Background

The gravitational potential \( U(\rho, \phi, z) \) is represented as

\[
U(\rho, \phi, z) =
\sum_{n=-N}^{N} \sum_{m=1}^{M}
C_{mn}\, J_n(k_{mn}\rho)\, \cos(n\phi)\, e^{k_{mn}z}
\]

where:

- \( J_n \): Bessel function of the first kind  
- \( k_{mn} \): radial eigenvalue (root of \( J_n' \))  
- \( C_{mn} \): fitted coefficient from least-squares optimization  

This representation satisfies **Laplace’s equation exactly** and converges **within a finite cylindrical region**, enabling high-accuracy modeling near irregular surfaces where spherical harmonics fail.

---

## 🚀 Getting Started

### Requirements

- Python ≥ 3.9  
- NumPy, SciPy, Matplotlib, SymPy  
- (optional) spiceypy for SPICE trajectory validation

### Installation

```bash
git clone https://github.com/<your-username>/CylindricalHarmonics_GravityModel.git
cd CylindricalHarmonics_GravityModel
pip install -r requirements.txt
````

### Example Usage

```python
import numpy as np
from src.bessel_basis import CylindricalBasis
from src.gravity_eval import compute_gravity

# Initialize cylindrical basis
basis = CylindricalBasis(N=5, M=5, radius=1.0)

# Load fitted coefficients
coeffs = np.load('data/fitted_coefficients/eros_cylindrical.npy', allow_pickle=True)

# Evaluate potential and acceleration
U, a = compute_gravity(rho=0.5, phi=1.2, z=0.1, coeffs=coeffs, basis=basis)

print("Potential [m^2/s^2]:", U)
print("Acceleration [m/s^2]:", a)
```

---

## 📊 Validation Example

To reproduce validation figures comparing the cylindrical and polyhedral gravity fields:

```bash
python examples/compare_polyhedral_vs_cyl.py
```

This generates RMS error and convergence plots as in the manuscript.

---

## 📚 Citation

If you use this code or reproduce results from the paper, please cite:

```bibtex
@article{Fereoli2025_CylindricalHarmonics,
  title   = {Interior Gravity Characterization of Small Celestial Bodies Using Cylindrical Harmonics},
  author  = {Fereoli, Giovanni and McMahon, Jay W.},
  journal = {Celestial Mechanics and Dynamical Astronomy},
  year    = {2025},
  note    = {under review}
}
```

---

## 📄 License

Released under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

## 👨‍🚀 Acknowledgments

Developed within the **Orbit and Small Body Research (ORCCA) Lab**
at the University of Colorado Boulder, in collaboration with the
**Colorado Center for Astrodynamics Research (CCAR)**.

Special thanks to the **Dawn** and **NEAR Shoemaker** mission teams for publicly available data.

```
