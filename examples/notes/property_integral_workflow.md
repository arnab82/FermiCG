# Property Integrals: From PySCF Through CMF to TPSCI

## The Orbital Transformation Chain

The full pipeline has two successive changes of orbital basis:

```
AO basis  --[Cact]-->  active MO basis  --[U]-->  CMF-optimized basis
```

- **`Cact`** (`mo_coeffs.npy`): the `(n_AO × n_act)` active-space MO coefficient
  matrix built by SPADE partitioning in `scf_spade.ipynb`.
- **`U`**: the `(n_act × n_act)` unitary orbital rotation found by `cmf_oo_newton`
  in `cmf.jl`.

Hamiltonian integrals go through both steps:

```
h1_AO  --[Cact.T @ h1_AO @ Cact]-->  h1_MO  --[U.T @ h1_MO @ U]-->  h1_CMF
```

`InCoreIntegrals.orbital_rotation(ints, U)` in Julia handles the second step.
**Property integrals must go through exactly the same two steps.**

---

## Step 1 — PySCF: extract and rotate to active MO basis

Add the following block to `scf_spade.ipynb` directly after the cell that saves
`ints_h1.npy` (i.e., after `Cact` is finalized):

```python
# -----------------------------------------------------------------------
# Property integrals: electric dipole (length gauge)
# -----------------------------------------------------------------------
# int1e_r gives <μ|r_x|ν>, <μ|r_y|ν>, <μ|r_z|ν> in AO basis
# shape: (3, n_AO, n_AO)
dip_ao = pymol.intor('int1e_r', comp=3)

# Transform to active MO basis using Cact  (n_AO × n_act)
# dip_mo[x, p, q] = Σ_{μν} Cact[μ,p] * dip_ao[x,μ,ν] * Cact[ν,q]
dip_mo = np.einsum('mp,xmn,nq->xpq', Cact, dip_ao, Cact)  # (3, n_act, n_act)

np.save("dipole_ints", dip_mo)
```

This produces `dipole_ints.npy` of shape `(3, n_act, n_act)` alongside the
existing `ints_h1.npy`.

If you also need the **velocity (momentum) gauge** for cross-checking:

```python
# <μ|∇|ν> in AO basis — sign convention: nabla, not -i*nabla
nabla_ao = pymol.intor('int1e_ipovlp', comp=3)  # (3, n_AO, n_AO), antisymmetric
nabla_mo = np.einsum('mp,xmn,nq->xpq', Cact, nabla_ao, Cact)
np.save("nabla_ints", nabla_mo)   # velocity-gauge dipole (imaginary, antisymmetric)
```

---

## Step 2 — CMF (Julia): apply the orbital rotation U

In `cmf.jl`, after the lines

```julia
d1   = orbital_rotation(d1, U)
ints = orbital_rotation(ints, U)
```

add:

```julia
# -----------------------------------------------------------------------
# Rotate property integrals with the same CMF unitary U
# -----------------------------------------------------------------------
# For a one-electron operator O, h'_pq = Σ_{mn} U[m,p] h_mn U[n,q]
# In Julia matrix notation (U is real): h' = U' * h * U
dip_mo  = npzread("dipole_ints.npy")     # (3, n_act, n_act) in Julia: [x, p, q]
dip_cmf = similar(dip_mo)
for x in 1:3
    dip_cmf[x,:,:] .= U' * dip_mo[x,:,:] * U
end
```

Then save `dip_cmf` together with the rest of the CMF data:

```julia
@save "data_cmf_13_cr2_morokuma.jld2" clusters init_fspace ints d1 e_cmf U dip_cmf
```

### Why this rotation is correct

`orbital_rotation(ints, U)` transforms `h1[p,q]` as

$$h'_{pq} = \sum_{mn} U_{mp}\, h_{mn}\, U_{nq} = (U^\top h\, U)_{pq}$$

(real orbitals, so $U^\dagger = U^\top$).  Any other one-electron integral
matrix transforms identically, so `U' * dip_mo[x,:,:] * U` is exact.

---

## Step 3 — TPSCI (Julia): load integrals and compute properties

In the TPSCI script (modelled on `tpsci_01.jl`), load `dip_cmf` from the
saved JLD2 file and use it after the TPSCI solve:

```julia
using QCBase
using FermiCG
using NPZ
using InCoreIntegrals
using RDM
using JLD2
using LinearAlgebra

@load "data_cmf_13_cr2_morokuma.jld2" clusters init_fspace ints d1 e_cmf U dip_cmf

# ---- build cluster bases and operators (unchanged) ----
init_fspace = FockConfig(init_fspace)
cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(
    ints, clusters, d1, [3,3,3,3,3], init_fspace, max_roots=100, verbose=1)

clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
cluster_ops   = FermiCG.compute_cluster_ops(cluster_bases, ints)
FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

# ---- TPSCI solve for nroots ----
nroots   = 4
ci_vector = FermiCG.TPSCIstate(clusters, init_fspace, R=nroots)
ci_vector = FermiCG.add_spin_focksectors(ci_vector)
eci, v0   = FermiCG.tps_ci_direct(ci_vector, cluster_ops, clustered_ham)

e0a, v0a = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham,
                              thresh_cipsi = 1e-4,
                              thresh_spin  = 1e-4,
                              thresh_foi   = 1e-6,
                              ci_max_iter  = 150,
                              nbody        = 4)

# ---- One-electron properties ----
# Transition 1-RDM for all root pairs: shape (norb, norb, nroots, nroots)
γ_aa, γ_bb = FermiCG.compute_1rdm(v0a, cluster_ops)

# Extract dipole components (each is n_act × n_act after both rotations)
μ_x = dip_cmf[1,:,:]
μ_y = dip_cmf[2,:,:]
μ_z = dip_cmf[3,:,:]

# Oscillator strengths from ground state (root 1)
f = FermiCG.compute_oscillator_strengths(e0a, γ_aa, γ_bb, μ_x, μ_y, μ_z; ref_root=1)

FermiCG.print_stick_spectrum(e0a, f; units=:ev)

ω, I = FermiCG.absorption_spectrum(e0a, f; σ=0.005, lineshape=:lorentzian)
```

---

## What happens to property integrals after HOSVD rotation?

The TPSCI script applies an HOSVD compression after each threshold:

```julia
rotations = FermiCG.hosvd(v0a, cluster_ops)
for ci in clusters
    FermiCG.rotate!(cluster_ops[ci.idx], rotations[ci.idx])
    FermiCG.rotate!(cluster_bases[ci.idx], rotations[ci.idx])
end
```

`FermiCG.rotate!` rotates the **local cluster-state indices** (the $s, t$
indices of TDMs like `A[p,s,t]`), **not** the molecular orbital indices $p$.
The `dip_cmf` matrix lives in the MO basis and is unaffected.
No update to `dip_cmf` is needed after HOSVD.

---

## Checklist

| Step | File changed | What to add |
|---|---|---|
| PySCF | `scf_spade.ipynb` | `dip_ao = mol.intor('int1e_r', comp=3)` → transform with `Cact` → save `dipole_ints.npy` |
| CMF | `cmf.jl` | Load `dipole_ints.npy`, apply `U' * dip * U` per component, save `dip_cmf` to JLD2 |
| TPSCI | `tpsci_XX.jl` | Load `dip_cmf`, call `compute_1rdm` → `compute_oscillator_strengths` → `absorption_spectrum` |

---

## Notes on Sign Conventions

- PySCF `int1e_r` returns $\langle \mu | \mathbf{r} | \nu \rangle$ (position, positive definite diagonal).
  Dipole moments computed with these are $\langle \Psi | \mathbf{r} | \Psi \rangle$, i.e.,
  the *electronic* contribution (positive). The full dipole including nuclear charges
  needs separate treatment.

- Velocity gauge: `int1e_ipovlp` returns $\langle \mu | \nabla | \nu \rangle$ (purely imaginary,
  antisymmetric). The transition dipole in velocity gauge is
  $\langle 0 | \nabla | n \rangle / \Delta E_{0n}$; it should agree with
  the length gauge for exact wavefunctions (gauge invariance test).

- The oscillator strengths in `compute_oscillator_strengths` use the
  length-gauge formula $f = \frac{2}{3} \Delta E |\langle 0 | \mathbf{r} | n \rangle|^2$,
  which requires `dip_mo` from `int1e_r`. Do not mix length and velocity
  gauge integrals in the same formula.
