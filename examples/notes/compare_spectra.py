"""
Overlay the TPSCI and TDDFT absorption spectra for visual comparison.

Reads:
  tddft_roots.txt   — from tddft.py
  tpsci_spectrum.npy — save ω and I from Julia first:
      using NPZ
      npzwrite("tpsci_spectrum.npz", Dict("omega"=>ω, "intensity"=>I,
                                          "energies"=>e0a, "fosc"=>f))
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HA2EV = 27.2114

# ── TPSCI data ────────────────────────────────────────────────────────────────
data    = np.load("tpsci_spectrum.npz")
omega   = data["omega"] * HA2EV
intens  = data["intensity"]
e_tpsci = (data["energies"] - data["energies"][0]) * HA2EV
f_tpsci = data["fosc"]

# ── TDDFT data ────────────────────────────────────────────────────────────────
tddft = np.loadtxt("tddft_roots.txt", skiprows=2)
e_td  = tddft[:, 1]
f_td  = tddft[:, 2]

sigma_ev = 0.005 * HA2EV
e_grid   = np.linspace(0.5, 8.0, 2000)

def lorentzian(x, x0, s):
    return (s / np.pi) / ((x - x0)**2 + s**2)

spec_td = sum(f * lorentzian(e_grid, e0, sigma_ev)
              for e0, f in zip(e_td, f_td) if f > 1e-5)

# Normalise both to the same peak height for shape comparison
norm_tpsci = intens / (intens.max() + 1e-30)
norm_td    = spec_td / (spec_td.max() + 1e-30)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(omega,  norm_tpsci, lw=2.5, color="steelblue",   label="TPSCI")
ax.plot(e_grid, norm_td,    lw=2.5, color="darkorange",   label="TD-CAM-B3LYP", ls="--")

# TPSCI sticks
for e, f in zip(e_tpsci[1:], f_tpsci[1:]):
    if f > 1e-4:
        ax.plot([e, e], [0, f / (f_tpsci[1:].max() + 1e-30)],
                lw=1.2, color="steelblue", alpha=0.5)

# TDDFT sticks
for e, f in zip(e_td, f_td):
    if f > 1e-4:
        ax.plot([e, e], [0, -f / (f_td.max() + 1e-30)],
                lw=1.2, color="darkorange", alpha=0.5)

ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Excitation energy (eV)", fontsize=13)
ax.set_ylabel("Normalised absorption (arb. u.)", fontsize=13)
ax.set_title("TPSCI vs TD-CAM-B3LYP  —  Cr₂O(NH₃)₈⁴⁺", fontsize=13)
ax.legend(fontsize=12)
ax.set_xlim(0.5, 7.0)
fig.tight_layout()
fig.savefig("comparison_spectra.png", dpi=150)
print("Saved comparison_spectra.png")
