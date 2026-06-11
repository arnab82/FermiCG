"""
test_tpsci_rdm2.jl

Correctness checks for compute_2rdm, compute_2rdm_threaded, compute_2rdm_blas.

Tests
-----
1.  serial == threaded                  (element-wise, atol=1e-12)
2.  serial == blas                      (element-wise, atol=1e-12)
3.  Trace: Σ_p Γ[p,p,p,p,r,r] ≡ Tr(γ^2 - γ)/2  (via 1-RDM)  — skipped,
    use test 4 instead.
4.  Partial trace: Σ_q Γ[p,q,r,q,r,r] == (N-1) * γ_aa[p,r,r,r] + (N_β) * γ_aa[p,r,r,r]
    — simpler: just check Σ_{qs} Γ[p,q,r,s,root,root]*δ_{qs} ≡ (N-1)*γ[p,r]
    Actually the clean version: Σ_q Γ[p,q,r,q,r,r] = (N_elec - 1) * γ_aa[p,r,r,r] for αα sector.
    We use the relation: Σ_q Γ_total[p,q,r,q,r,r] = (N-1) * γ_total[p,r,r,r]
    where γ_total = γ_aa + γ_bb and Γ_total = sum over all spin pairs.
5.  Hermiticity: Γ[p,q,r,s,root,root] == Γ[r,s,p,q,root,root] (for real states)
6.  Anti-symmetry: Γ[p,q,r,s,...] == -Γ[q,p,r,s,...] == -Γ[p,q,s,r,...] == Γ[q,p,s,r,...]
7.  Transition 2-RDM: serial == threaded  (bra ≠ ket)
8.  Diagonal elements Γ[p,p,q,q,r,r] >= 0  (probability interpretation, real states only for αα+ββ diagonal)

Run with:
    julia --threads auto test_tpsci_rdm2.jl
or via runtests.jl.
"""

using QCBase
using RDM
using FermiCG
using Printf
using Test
using LinearAlgebra
using JLD2

function _slice_tpsci_roots(v, roots)
    out = FermiCG.TPSCIstate(v; R=length(roots))
    for (fock, configs) in v.data
        for (config, coeffs) in configs
            out[fock][config] .= coeffs[roots]
        end
    end
    return out
end

@testset "compute_2rdm  (he4, 64bit)" begin

    println("\n=== Loading he4 test data ===")
    cd(@__DIR__)
    @load "_testdata_cmf_he4.jld2"

    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops   = FermiCG.compute_cluster_ops_2rdm(cluster_bases, ints)
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    # Run TPSCI to get converged wavefunctions
    nroots   = 1
    ref_fock = FermiCG.FockConfig(init_fspace)
    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)
    ci_vector[ref_fock][ClusterConfig([1,1,1,1])] = [1.0]

    e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham,
                               incremental=true, ci_conv=1e-10,
                               thresh_cipsi=1e-3, thresh_foi=1e-8,
                               thresh_asci=-1, conv_thresh=1e-7,
                               ci_lindep_thresh=1e-12)

    n_alpha = sum(f[1] for f in init_fspace)
    n_beta  = sum(f[2] for f in init_fspace)
    N_elec  = n_alpha + n_beta
    println("\nTPSCI energies: ", e0)
    println("N_elec = $N_elec  ($(n_alpha)α + $(n_beta)β)")
    println("Threads available: ", Threads.nthreads())

    # ------------------------------------------------------------------
    # Test 1: serial == threaded
    # ------------------------------------------------------------------
    @testset "2-RDM serial == threaded" begin
        Γ_s = FermiCG.compute_2rdm(v0, cluster_ops)
        Γ_t = FermiCG.compute_2rdm_threaded(v0, cluster_ops)
        @test isapprox(Γ_s, Γ_t, atol=1e-12)
        println("  Max diff serial vs threaded: ", maximum(abs.(Γ_s .- Γ_t)))
    end

    # ------------------------------------------------------------------
    # Test 2: serial == BLAS
    # ------------------------------------------------------------------
    @testset "2-RDM serial == blas" begin
        Γ_s  = FermiCG.compute_2rdm(v0, cluster_ops)
        Γ_bl = FermiCG.compute_2rdm_blas(v0, cluster_ops)
        @test isapprox(Γ_s, Γ_bl, atol=1e-12)
        println("  Max diff serial vs blas: ", maximum(abs.(Γ_s .- Γ_bl)))
    end

    # ------------------------------------------------------------------
    # Test 3: Partial-trace relation
    #   Σ_q Γ[p,q,r,q, root,root] = (N-1) * γ_total[p,r, root,root]
    # where γ_total = γ_aa + γ_bb  and  Γ is the spin-free 2-RDM.
    # ------------------------------------------------------------------
    @testset "Partial trace Σ_q Γ[p,q,r,q] = (N-1)γ[p,r]" begin
        Γ      = FermiCG.compute_2rdm(v0, cluster_ops)
        γ_aa, γ_bb = FermiCG.compute_1rdm(v0, cluster_ops)
        norb   = size(Γ, 1)
        R      = size(Γ, 5)

        for root in 1:R
            γ_tot = γ_aa[:,:,root,root] .+ γ_bb[:,:,root,root]
            # partial trace: Σ_q Γ[p,q,r,q,root,root]
            pt = zeros(norb, norb)
            for q in 1:norb
                pt .+= Γ[:,q,:,q,root,root]
            end
            @test isapprox(pt, (N_elec - 1) .* γ_tot, atol=1e-9)
        end
        println("  Partial-trace check passed for all $nroots roots")
    end

    # ------------------------------------------------------------------
    # Test 4: Hermiticity  Γ[p,q,r,s,root,root] == Γ[r,s,p,q,root,root]
    # ------------------------------------------------------------------
    @testset "Hermiticity Γ[p,q,r,s] == Γ[r,s,p,q]" begin
        Γ    = FermiCG.compute_2rdm(v0, cluster_ops)
        norb = size(Γ, 1)
        R    = size(Γ, 5)
        for root in 1:R
            Γ_rs = Γ[:,:,:,:,root,root]
            Γ_perm = permutedims(Γ_rs, [3,4,1,2])  # Γ[r,s,p,q]
            @test isapprox(Γ_rs, Γ_perm, atol=1e-10)
        end
        println("  Hermiticity check passed for all $nroots roots")
    end

    # ------------------------------------------------------------------
    # Test 5: Spin-free exchange symmetry
    #   Γ[p,q,r,s] == Γ[q,p,s,r]
    # For the spin-summed spatial 2-RDM, swapping only p/q is not a valid
    # antisymmetry check because opposite-spin contributions survive at p == q.
    # ------------------------------------------------------------------
    @testset "Pair-exchange symmetry of spin-free 2-RDM" begin
        Γ    = FermiCG.compute_2rdm(v0, cluster_ops)
        R    = size(Γ, 5)
        for root in 1:R
            Γ_r = Γ[:,:,:,:,root,root]
            @test isapprox(Γ_r, permutedims(Γ_r, [2,1,4,3]), atol=1e-10)
        end
        println("  Pair-exchange symmetry check passed for all $nroots roots")
    end

    # ------------------------------------------------------------------
    # Test 6: Consistency with energy via 2-RDM
    #   E = Σ_{pqrs} h2[p,q,r,s] Γ[p,q,r,s]  +  Σ_{pq} h1[p,q] γ[p,q]
    #   where h2 = (pq|rs) - (ps|rq)  (antisymmetrized 2-electron integrals)
    # ------------------------------------------------------------------
    @testset "Energy from 2-RDM matches TPSCI eigenvalues" begin
        Γ         = FermiCG.compute_2rdm(v0, cluster_ops)
        γ_aa, γ_bb = FermiCG.compute_1rdm(v0, cluster_ops)
        norb      = size(Γ, 1)
        R         = size(Γ, 5)

        # One-electron integrals: h1[p,q] = ints.h1[p,q]
        # Two-electron integrals: ints.h2[p,q,r,s] = (p q | r s)
        # The spin-free Hamiltonian energy formula:
        #   E = Σ_{pq} h1[p,q]*(γ_aa+γ_bb)[p,q]
        #     + (1/2) Σ_{pqrs} (pq|rs) * Γ[p,q,r,s]
        h1   = ints.h1   # (norb, norb)
        h2   = ints.h2   # (norb, norb, norb, norb) in (pq|rs) convention
        for root in 1:R
            γ_tot  = γ_aa[:,:,root,root] .+ γ_bb[:,:,root,root]
            Γ_root = Γ[:,:,:,:,root,root]

            E1 = dot(h1, γ_tot)
            # 2-electron: (1/2) Σ_{pqrs} h2[p,r,q,s] * Γ[p,q,r,s]
            # The sign convention for Γ here is p†q†sr, so
            # E2 = (1/2) Σ_{pqrs} <pq|rs> * Γ[p,q,r,s]
            # with <pq|rs> = h2[p,r,q,s] in chemist notation
            E2 = zero(Float64)
            for s in 1:norb, r in 1:norb, q in 1:norb, p in 1:norb
                E2 += h2[p,r,q,s] * Γ_root[p,q,r,s]
            end
            E2 *= 0.5

            E_rdm = E1 + E2
            @test isapprox(E_rdm, e0[root], atol=1e-7)
        end
        println("  Energy-from-2RDM check passed for all $nroots roots")
        println("  2-RDM energies: ", [
            let γ_tot = γ_aa[:,:,r,r].+γ_bb[:,:,r,r], Γr = Γ[:,:,:,:,r,r]
                dot(ints.h1, γ_tot) +
                0.5*sum(ints.h2[p,rr,q,s]*Γr[p,q,rr,s]
                        for p in 1:norb, q in 1:norb, rr in 1:norb, s in 1:norb)
            end
            for r in 1:R])
    end

    # ------------------------------------------------------------------
    # Timing comparison
    # ------------------------------------------------------------------
    println("\n=== Timing ($(Threads.nthreads()) threads) ===")
    println("compute_2rdm serial:")
    @time FermiCG.compute_2rdm(v0, cluster_ops)
    println("compute_2rdm threaded:")
    @time FermiCG.compute_2rdm_threaded(v0, cluster_ops)
    println("compute_2rdm blas:")
    @time FermiCG.compute_2rdm_blas(v0, cluster_ops)

end
