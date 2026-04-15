using QCBase
using FermiCG
using NPZ
using InCoreIntegrals
using RDM
using JLD2
using Printf
using LinearAlgebra

@load "data_cmf.jld2"

M = 100

init_fspace = FockConfig([(3,0), (3, 3), (0, 3)])
init_fspace1= FockConfig([(2,0), (3, 3), (0, 4)])
let
    cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [10,10,10], init_fspace, max_roots=M, verbose=1);

    # CT Fock sectors
    ct_fspaces = [
        FockConfig([(2,0), (3,3), (0,4)]),
        FockConfig([(2,0), (3,3), (1,3)]),
        FockConfig([(3,0), (2,3), (1,3)]),
        FockConfig([(3,1), (3,3), (0,2)]),
        FockConfig([(3,1), (3,2), (0,3)]),
        FockConfig([(2,1), (3,3), (1,2)]),
    ]

    for fs in ct_fspaces
        cb = FermiCG.compute_cluster_eigenbasis_spin(
            ints, clusters, d1, [10,10,10], fs, max_roots=5, verbose=0)
        cluster_bases = FermiCG.merge_cluster_bases(cluster_bases, cb)
    end

clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters);
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

nroots=10


ci_vector = FermiCG.TPSCIstate(clusters, init_fspace, R=nroots)
ci_vector = FermiCG.add_spin_focksectors(ci_vector)

# form single excitonic states by adding CT sectors to the reference root
FermiCG.add_fockconfig!(ci_vector, FockConfig([(2,0), (3, 3), (0, 4)]))
FermiCG.add_fockconfig!(ci_vector, FockConfig([(2,0), (3, 3), (1, 3)]))
FermiCG.add_fockconfig!(ci_vector, FockConfig([(3,0), (2, 3), (1, 3)]))
FermiCG.add_fockconfig!(ci_vector, FockConfig([(3,1), (3, 3), (0, 2)]))
FermiCG.add_fockconfig!(ci_vector, FockConfig([(3,1), (3, 2), (0, 3)]))
FermiCG.add_fockconfig!(ci_vector, FockConfig([(2,1), (3, 3), (1, 2)]))
# ------------------------------------------------------------------
# Charge-transfer Fock sectors for excited electronic states.

# Single-electron CT (Na=6, Nb=6 conserved throughout):
#   α: C1(3,0)→C3(0,3)   new sectors: C1=(2,0), C3=(1,3)
#   α: C2(3,3)→C3(0,3)   new sectors: C2=(2,3), C3=(1,3)
#   β: C3(0,3)→C1(3,0)   new sectors: C3=(0,2), C1=(3,1)
#   β: C2(3,3)→C1(3,0)   new sectors: C2=(3,2), C1=(3,1)
#
# Double CT / spin-exchange C1↔C3 (−1α+1β on C1, +1α−1β on C3):
#   new sectors: C1=(2,1), C3=(1,2)  
# ------------------------------------------------------------------

fspace_0 = init_fspace

# α: C1 → C3
tmp_fspace = FermiCG.replace(fspace_0, (1,3), ([2,0],[1,3]))
FermiCG.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][FermiCG.ClusterConfig([1,1,1])] = zeros(Float64,nroots)

# α: C2 → C3
tmp_fspace = FermiCG.replace(fspace_0, (2,3), ([2,3],[1,3]))
FermiCG.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][FermiCG.ClusterConfig([1,1,1])] = zeros(Float64,nroots)

# β: C3 → C1
tmp_fspace = FermiCG.replace(fspace_0, (1,3), ([3,1],[0,2]))
FermiCG.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][FermiCG.ClusterConfig([1,1,1])] = zeros(Float64,nroots)

# β: C2 → C1
tmp_fspace = FermiCG.replace(fspace_0, (1,2), ([3,1],[3,2]))
FermiCG.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][FermiCG.ClusterConfig([1,1,1])] = zeros(Float64,nroots)

# Spin-exchange C1↔C3: -1α+1β on C1, +1α-1β on C3
tmp_fspace = FermiCG.replace(fspace_0, (1,3), ([2,1],[1,2]))
FermiCG.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][FermiCG.ClusterConfig([1,1,1])] = zeros(Float64,nroots)

# Add spin sectors for all CT FockConfigs
ci_vector = FermiCG.add_spin_focksectors(ci_vector)

FermiCG.eye!(ci_vector)

eci, v = FermiCG.tps_ci_direct(ci_vector, cluster_ops, clustered_ham);

e0a, v0a = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true,
                            thresh_cipsi = 1e-3,
                            thresh_foi   = 1e-6,
                            thresh_asci  = -1);
γ_aa, γ_bb = FermiCG.compute_1rdm(v0a, cluster_ops)

# ------------------------------------------------------------------
# Diagnostic 1: trace of the 1-RDM per root
# Tr(γ_aa[:,:,r,r] + γ_bb[:,:,r,r]) must equal total number of electrons.
# If these are correct the code is working; if zero there is a bug.
n_elec = sum(init_fspace[k][1] + init_fspace[k][2] for k in 1:length(clusters))
γ_total = γ_aa .+ γ_bb
@printf("\n 1-RDM trace check (should be %.1f electrons each root):\n", float(n_elec))
for r in 1:nroots
    @printf("   root %i: Tr(γ) = %.6f\n", r, tr(γ_total[:,:,r,r]))
end

# Diagnostic 2: raw transition dipole values between ref root and all others.
# If these are all zero, the transitions are symmetry/spin-forbidden (not a bug).
μ_x, μ_y, μ_z = dip_cmf[1,:,:], dip_cmf[2,:,:], dip_cmf[3,:,:]
tdm_x, tdm_y, tdm_z = FermiCG.compute_transition_dipoles(γ_aa, γ_bb, μ_x, μ_y, μ_z; ref_root=1)
@printf("\n Transition dipole moments from root 1 (a.u.):\n")
@printf("   %-6s  %-12s  %-12s  %-12s\n", "Root", "<0|μ_x|n>", "<0|μ_y|n>", "<0|μ_z|n>")
for n in 2:nroots
    @printf("   %-6i  %-12.6f  %-12.6f  %-12.6f\n", n, tdm_x[n], tdm_y[n], tdm_z[n])
end

f = FermiCG.compute_oscillator_strengths(
        e0a, γ_aa, γ_bb, μ_x, μ_y, μ_z; ref_root=1)

FermiCG.print_stick_spectrum(e0a, f; units=:ev)

ω, I = FermiCG.absorption_spectrum(e0a, f; σ=0.005, lineshape=:lorentzian)
using NPZ
npzwrite("tpsci_spectrum.npz", Dict(
    "omega"     => ω,
    "intensity" => I,
    "energies"  => e0a,
    "fosc"      => f))
    using Plots

    ha2ev = 27.2114
    
    ω_ev  = ω .* ha2ev
    ΔE_ev = (e0a[2:end] .- e0a[1]) .* ha2ev
    f_sticks = f[2:end]
    
    p = plot(ω_ev, I,
        lw=2, color=:steelblue,
        xlabel="Energy (eV)", ylabel="Absorption (arb. u.)",
        title="TPSCI absorption — Cr₂O(NH₃)₈⁴⁺",
        label="TPSCI (Lorentzian)", legend=:topright)
    
    scale = maximum(I) / (maximum(f_sticks) + 1e-10)
    for (e, fi) in zip(ΔE_ev, f_sticks)
        fi > 1e-4 || continue
        plot!(p, [e, e], [0.0, fi * scale], lw=1.5, color=:red, alpha=0.6, label=false)
    end
    
    savefig(p, "tpsci_spectrum.pdf")
    display(p)
    
# ==============================================================================
# Spin-flip 1-RDM and Spin-Orbit Coupling
# ==============================================================================
γ_ab, γ_ba = FermiCG.compute_1rdm_sf(v0a, cluster_ops)

# Load SOC integrals (MO basis, saved by property_integrals_pyscf.ipynb)
# Expected shape: (norb, norb) each
soc_Lx = npzread("soc_Lx.npy")
soc_Ly = npzread("soc_Ly.npy")
soc_Lz = npzread("soc_Lz.npy")

# SOC matrix elements H_SOC[r1,r2] broken into z, +, - components:
#   H_SOC = h_z*(γ_aa - γ_bb) + h_+*γ_ab + h_-*γ_ba
# where h_z = L_z (imaginary, stored real: multiply by im),
#       h_+ = (L_x + i*L_y)/sqrt(2),  h_- = (L_x - i*L_y)/sqrt(2)
SOC_z = zeros(ComplexF64, nroots, nroots)
SOC_p = zeros(ComplexF64, nroots, nroots)
SOC_m = zeros(ComplexF64, nroots, nroots)

h_z  = im .* soc_Lz                     # <p|L_z|q> is purely imaginary
h_pl = (soc_Lx .+ im .* soc_Ly) ./ sqrt(2)
h_mn = (soc_Lx .- im .* soc_Ly) ./ sqrt(2)

for r2 in 1:nroots, r1 in 1:nroots
    Δγ = γ_aa[:, :, r1, r2] .- γ_bb[:, :, r1, r2]
    SOC_z[r1, r2] = dot(vec(h_z),  vec(Δγ))
    SOC_p[r1, r2] = dot(vec(h_pl), vec(γ_ab[:, :, r1, r2]))
    SOC_m[r1, r2] = dot(vec(h_mn), vec(γ_ba[:, :, r1, r2]))
end

SOC_total = SOC_z .+ SOC_p .+ SOC_m

@printf("\n SOC matrix |H_SOC[r1,r2]| (cm⁻¹), threshold 1 cm⁻¹:\n")
@printf("   %-6s  %-6s  %-14s  %-14s  %-14s  %-14s\n",
        "r1", "r2", "|H_SOC|", "|SOC_z|", "|SOC_+|", "|SOC_-|")
ha2cm = 219474.63
for r2 in 1:nroots, r1 in 1:r2
    val = abs(SOC_total[r1, r2]) * ha2cm
    val < 1.0 && continue
    @printf("   %-6i  %-6i  %-14.4f  %-14.4f  %-14.4f  %-14.4f\n",
            r1, r2,
            abs(SOC_total[r1, r2]) * ha2cm,
            abs(SOC_z[r1, r2]) * ha2cm,
            abs(SOC_p[r1, r2]) * ha2cm,
            abs(SOC_m[r1, r2]) * ha2cm)
end

# ==============================================================================
# 2-RDM and Exchange Coupling
# ==============================================================================
Γ = FermiCG.compute_2rdm(v0a, cluster_ops)

# Load cluster-pair exchange integrals K_{IJ}[p,q,r,s]
# Shape: (norb_I, norb_J, norb_I, norb_J) = (pI, qJ, rI, sJ)
# Saved by property_integrals_pyscf.ipynb as K_12_exch.npy, K_13_exch.npy, K_23_exch.npy
K_12 = npzread("K_12_exch.npy")   # (orb_C1, orb_C2, orb_C1, orb_C2)
K_13 = npzread("K_13_exch.npy")   # (orb_C1, orb_C3, orb_C1, orb_C3)
K_23 = npzread("K_23_exch.npy")   # (orb_C2, orb_C3, orb_C2, orb_C3)

# Cluster orbital offsets
off = [0; cumsum([length(c.orb_list) for c in clusters])]

# J_IJ[r] = (1/2) Σ_{p,r∈I, q,s∈J} (pr|qs) Γ[p,q,r,s,r,r]
# diagonal roots only (ground-state-like expectation values per root)
nroots_diag = nroots
J12 = zeros(nroots_diag)
J13 = zeros(nroots_diag)
J23 = zeros(nroots_diag)

let
    o1 = off[1]+1:off[2]   # cluster 1 orbital indices (global)
    o2 = off[2]+1:off[3]   # cluster 2
    o3 = off[3]+1:off[4]   # cluster 3
    norb1 = length(o1); norb2 = length(o2); norb3 = length(o3)

    for rr in 1:nroots_diag
        acc12 = 0.0; acc13 = 0.0; acc23 = 0.0
        for p in 1:norb1, r in 1:norb1, q in 1:norb2, s in 1:norb2
            acc12 += K_12[p, q, r, s] * Γ[o1[p], o2[q], o1[r], o2[s], rr, rr]
        end
        for p in 1:norb1, r in 1:norb1, q in 1:norb3, s in 1:norb3
            acc13 += K_13[p, q, r, s] * Γ[o1[p], o3[q], o1[r], o3[s], rr, rr]
        end
        for p in 1:norb2, r in 1:norb2, q in 1:norb3, s in 1:norb3
            acc23 += K_23[p, q, r, s] * Γ[o2[p], o3[q], o2[r], o3[s], rr, rr]
        end
        J12[rr] = 0.5 * acc12
        J13[rr] = 0.5 * acc13
        J23[rr] = 0.5 * acc23
    end
end

@printf("\n Exchange coupling constants J_IJ (cm⁻¹) per root:\n")
@printf("   %-6s  %-14s  %-14s  %-14s\n", "Root", "J_12", "J_13", "J_23")
for rr in 1:nroots_diag
    @printf("   %-6i  %-14.4f  %-14.4f  %-14.4f\n",
            rr, J12[rr]*ha2cm, J13[rr]*ha2cm, J23[rr]*ha2cm)
end

e2a = FermiCG.compute_pt2_energy(v0a, cluster_ops, clustered_ham)
end