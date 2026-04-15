"""
dipole_comparison.jl

Compare two ways to compute one-electron properties from a TPSCI wavefunction:

  Method A (1-RDM path):
      1. compute_1rdm  → γ_aa, γ_bb  (norb × norb × R × R)
      2. contract_1rdm_property(γ_aa, γ_bb, h_prop) → P[r1,r2]

  Method B (direct path):
      compute_1e_property_direct(v, cluster_ops, h_prop) → P[r1,r2]
      (contracts with h_prop inside the cluster loop — no full RDM stored)

Both should give identical P matrices.  This script:
  • Loads the TPSCI solution produced by tpsci.jl
  • Prints the max element-wise difference |P_A - P_B| for μ_x, μ_y, μ_z
  • Prints a table of diagonal (expectation value) and transition dipole moments
    from both methods

Prerequisites:
  Run tpsci.jl first so that data_cmf.jld2 exists and v0a / cluster_ops are
  available.  Import them as shown below, or simply include this file at the
  bottom of tpsci.jl after v0a and cluster_ops are defined.
"""

using QCBase
using FermiCG
using InCoreIntegrals
using RDM
using JLD2
using Printf
using LinearAlgebra

# --------------------------------------------------------------------------
# Load data (skip if already in scope from tpsci.jl)
# --------------------------------------------------------------------------
# @load "data_cmf.jld2"
# Assume v0a, cluster_ops, γ_aa, γ_bb, dip_cmf, e0a are already defined.
# (Either run after tpsci.jl or uncomment the load above and rebuild.)

# --------------------------------------------------------------------------
# Dipole integral matrices (CMF MO basis, shape 3 × norb × norb)
# --------------------------------------------------------------------------
μ_x = dip_cmf[1, :, :]   # (norb, norb)
μ_y = dip_cmf[2, :, :]
μ_z = dip_cmf[3, :, :]

nroots = size(γ_aa, 3)

# ==========================================================================
# Method A: 1-RDM contraction
# ==========================================================================
println("\n" * "="^70)
println(" Method A: 1-RDM contraction")
println("="^70)

P_x_A = FermiCG.contract_1rdm_property(γ_aa, γ_bb, μ_x)
P_y_A = FermiCG.contract_1rdm_property(γ_aa, γ_bb, μ_y)
P_z_A = FermiCG.contract_1rdm_property(γ_aa, γ_bb, μ_z)

# ==========================================================================
# Method B: direct (cluster-loop contraction, no full RDM)
# ==========================================================================
println("\n" * "="^70)
println(" Method B: direct cluster-operator contraction")
println("="^70)

P_x_B = FermiCG.compute_1e_property_direct(v0a, cluster_ops, μ_x)
P_y_B = FermiCG.compute_1e_property_direct(v0a, cluster_ops, μ_y)
P_z_B = FermiCG.compute_1e_property_direct(v0a, cluster_ops, μ_z)

# ==========================================================================
# Comparison: max |A - B|
# ==========================================================================
println("\n" * "="^70)
println(" Comparison: max |P_A[r1,r2] - P_B[r1,r2]|")
println("="^70)
err_x = maximum(abs.(P_x_A .- P_x_B))
err_y = maximum(abs.(P_y_A .- P_y_B))
err_z = maximum(abs.(P_z_A .- P_z_B))
@printf("   μ_x: %.3e   μ_y: %.3e   μ_z: %.3e\n", err_x, err_y, err_z)
tol = 1e-8
if max(err_x, err_y, err_z) < tol
    println("   OK — methods agree to within $tol a.u.")
else
    println("   WARNING — discrepancy > $tol: check orbital ordering or sign conventions")
end

# ==========================================================================
# Diagonal (expectation) values per root
# ==========================================================================
println("\n" * "="^70)
println(" Dipole expectation values <r|μ|r>  (a.u.)")
println("="^70)
@printf("   %-6s  %-14s  %-14s  %-14s  %-14s  %-14s  %-14s\n",
        "Root",
        "μ_x (1-RDM)", "μ_x (direct)",
        "μ_y (1-RDM)", "μ_y (direct)",
        "μ_z (1-RDM)", "μ_z (direct)")
for r in 1:nroots
    @printf("   %-6i  %-14.8f  %-14.8f  %-14.8f  %-14.8f  %-14.8f  %-14.8f\n",
            r,
            P_x_A[r,r], P_x_B[r,r],
            P_y_A[r,r], P_y_B[r,r],
            P_z_A[r,r], P_z_B[r,r])
end

# ==========================================================================
# Transition dipole moments from root 1 (off-diagonal |<1|μ|n>|)
# ==========================================================================
println("\n" * "="^70)
println(" Transition dipoles |<1|μ_α|n>| from root 1  (a.u.)")
println("="^70)
@printf("   %-6s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s\n",
        "n",
        "|μ_x| A", "|μ_x| B",
        "|μ_y| A", "|μ_y| B",
        "|μ_z| A", "|μ_z| B")
for n in 2:nroots
    @printf("   %-6i  %-12.6f  %-12.6f  %-12.6f  %-12.6f  %-12.6f  %-12.6f\n",
            n,
            abs(P_x_A[1,n]), abs(P_x_B[1,n]),
            abs(P_y_A[1,n]), abs(P_y_B[1,n]),
            abs(P_z_A[1,n]), abs(P_z_B[1,n]))
end

# ==========================================================================
# Oscillator strengths from both methods (should be identical)
# ==========================================================================
println("\n" * "="^70)
println(" Oscillator strengths f_0n from both methods")
println("="^70)
ha2ev = 27.2114
@printf("   %-6s  %-12s  %-12s  %-12s  %-14s\n",
        "n", "ΔE (eV)", "f (1-RDM)", "f (direct)", "|Δf|")
for n in 2:nroots
    ΔE = e0a[n] - e0a[1]
    ΔE > 0 || continue
    # f = (2/3) ΔE (|<0|μ_x|n>|² + |<0|μ_y|n>|² + |<0|μ_z|n>|²)
    f_A = (2/3) * ΔE * (abs2(P_x_A[1,n]) + abs2(P_y_A[1,n]) + abs2(P_z_A[1,n]))
    f_B = (2/3) * ΔE * (abs2(P_x_B[1,n]) + abs2(P_y_B[1,n]) + abs2(P_z_B[1,n]))
    @printf("   %-6i  %-12.4f  %-12.6f  %-12.6f  %-14.2e\n",
            n, ΔE * ha2ev, f_A, f_B, abs(f_A - f_B))
end
