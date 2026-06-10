"""
test_tpsci_rdm_threaded.jl

Correctness checks for the threaded 1-RDM functions:
  compute_1rdm_threaded, compute_1rdm_sf_threaded.

Tests
-----
1. Serial == threaded               (element-wise, tol=1e-12)
2. Trace of diagonal γ_aa + γ_bb   == N_alpha + N_beta  per root
3. Hermiticity of diagonal RDM      γ[p,q,r,r] == γ[q,p,r,r]
4. Transition 1-RDM: serial == threaded  (bra ≠ ket)
5. Spin-flip 1-RDM: serial == threaded
6. γ_ab transpose == γ_ba per root

Run with:
    julia --threads auto test_tpsci_rdm_threaded.jl
or via the test suite through runtests.jl.
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

@testset "compute_1rdm_threaded  (he4, 64bit)" begin

    println("\n=== Loading he4 test data ===")
    cd(@__DIR__)
    @load "_testdata_cmf_he4.jld2"

    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops   = FermiCG.compute_cluster_ops(cluster_bases, ints)
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    # Run TPSCI to get converged wavefunctions
    nroots    = 5
    ref_fock  = FermiCG.FockConfig(init_fspace)
    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)
    ci_vector[ref_fock][ClusterConfig([2,1,1,1])] = [0,1,0,0,0]
    ci_vector[ref_fock][ClusterConfig([1,2,1,1])] = [0,0,1,0,0]
    ci_vector[ref_fock][ClusterConfig([1,1,2,1])] = [0,0,0,1,0]
    ci_vector[ref_fock][ClusterConfig([1,1,1,2])] = [0,0,0,0,1]

    e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham,
                               incremental=true, ci_conv=1e-10,
                               thresh_cipsi=1e-3, thresh_foi=1e-8,
                               thresh_asci=-1, conv_thresh=1e-7,
                               ci_lindep_thresh=1e-12)

    N_elec = sum(f[1] + f[2] for f in init_fspace)
    println("\nTPSCI energies: ", e0)
    println("Threads available: ", Threads.nthreads())

    # ------------------------------------------------------------------
    # Test 1: serial == threaded for compute_1rdm (same-state)
    # ------------------------------------------------------------------
    @testset "1-RDM serial == threaded" begin
        γ_aa_s, γ_bb_s = FermiCG.compute_1rdm(v0, cluster_ops)
        γ_aa_t, γ_bb_t = FermiCG.compute_1rdm_threaded(v0, cluster_ops)

        @test isapprox(γ_aa_s, γ_aa_t, atol=1e-12)
        @test isapprox(γ_bb_s, γ_bb_t, atol=1e-12)
        println("  Max diff γ_aa: ", maximum(abs.(γ_aa_s .- γ_aa_t)))
        println("  Max diff γ_bb: ", maximum(abs.(γ_bb_s .- γ_bb_t)))
    end

    # ------------------------------------------------------------------
    # Test 2: trace = particle number per root
    # ------------------------------------------------------------------
    @testset "Trace = N_elec per root" begin
        γ_aa_t, γ_bb_t = FermiCG.compute_1rdm_threaded(v0, cluster_ops)
        norb, _, R, _ = size(γ_aa_t)

        # Total electrons from init_fspace
        n_alpha = sum(f[1] for f in init_fspace)
        n_beta  = sum(f[2] for f in init_fspace)

        for r in 1:R
            tr_aa = sum(γ_aa_t[p, p, r, r] for p in 1:norb)
            tr_bb = sum(γ_bb_t[p, p, r, r] for p in 1:norb)
            @test isapprox(tr_aa, n_alpha, atol=1e-10)
            @test isapprox(tr_bb, n_beta,  atol=1e-10)
        end
        println("  Trace α per root: ", [sum(γ_aa_t[p,p,r,r] for p in 1:norb) for r in 1:R])
        println("  Trace β per root: ", [sum(γ_bb_t[p,p,r,r] for p in 1:norb) for r in 1:R])
    end

    # ------------------------------------------------------------------
    # Test 3: Hermiticity of diagonal (same-root) RDM
    # ------------------------------------------------------------------
    @testset "Hermiticity γ[p,q,r,r] == γ[q,p,r,r]" begin
        γ_aa_t, γ_bb_t = FermiCG.compute_1rdm_threaded(v0, cluster_ops)
        norb, _, R, _  = size(γ_aa_t)
        for r in 1:R
            @test isapprox(γ_aa_t[:,:,r,r], γ_aa_t[:,:,r,r]', atol=1e-10)
            @test isapprox(γ_bb_t[:,:,r,r], γ_bb_t[:,:,r,r]', atol=1e-10)
        end
    end

    # ------------------------------------------------------------------
    # Test 4: serial == threaded for transition RDM (bra ≠ ket)
    # ------------------------------------------------------------------
    @testset "Transition 1-RDM serial == threaded" begin
        # Use first root as bra, rest as ket by slicing the state
        # Build single-root states for a clean bra/ket pair
        bra = _slice_tpsci_roots(v0, [1])
        ket = _slice_tpsci_roots(v0, [2,3,4,5])

        γ_aa_s, γ_bb_s = FermiCG.compute_1rdm(bra, ket, cluster_ops)
        γ_aa_t, γ_bb_t = FermiCG.compute_1rdm_threaded(bra, ket, cluster_ops)

        @test isapprox(γ_aa_s, γ_aa_t, atol=1e-12)
        @test isapprox(γ_bb_s, γ_bb_t, atol=1e-12)
        println("  Transition max diff γ_aa: ", maximum(abs.(γ_aa_s .- γ_aa_t)))
    end

    # ------------------------------------------------------------------
    # Test 5: serial == threaded for compute_1rdm_sf
    # ------------------------------------------------------------------
    @testset "Spin-flip 1-RDM serial == threaded" begin
        γ_ab_s, γ_ba_s = FermiCG.compute_1rdm_sf(v0, cluster_ops)
        γ_ab_t, γ_ba_t = FermiCG.compute_1rdm_sf_threaded(v0, cluster_ops)

        @test isapprox(γ_ab_s, γ_ab_t, atol=1e-12)
        @test isapprox(γ_ba_s, γ_ba_t, atol=1e-12)
        println("  Max diff γ_ab: ", maximum(abs.(γ_ab_s .- γ_ab_t)))
        println("  Max diff γ_ba: ", maximum(abs.(γ_ba_s .- γ_ba_t)))
    end

    # ------------------------------------------------------------------
    # Test 6: γ_ab[p,q,r,r] == γ_ba[q,p,r,r]  (transpose relation)
    # Holds for real eigenstates: <r|p†_α q_β|r>* = <r|q†_β p_α|r>
    # ------------------------------------------------------------------
    @testset "γ_ab transpose == γ_ba per root" begin
        γ_ab_t, γ_ba_t = FermiCG.compute_1rdm_sf_threaded(v0, cluster_ops)
        norb, _, R, _  = size(γ_ab_t)
        for r in 1:R
            @test isapprox(γ_ab_t[:,:,r,r], γ_ba_t[:,:,r,r]', atol=1e-10)
        end
    end

    # ------------------------------------------------------------------
    # Timing comparison
    # ------------------------------------------------------------------
    println("\n=== Timing: serial vs threaded ($(Threads.nthreads()) threads) ===")
    println("compute_1rdm serial:")
    @time FermiCG.compute_1rdm(v0, cluster_ops)
    println("compute_1rdm threaded:")
    @time FermiCG.compute_1rdm_threaded(v0, cluster_ops)
    println("compute_1rdm_sf serial:")
    @time FermiCG.compute_1rdm_sf(v0, cluster_ops)
    println("compute_1rdm_sf threaded:")
    @time FermiCG.compute_1rdm_sf_threaded(v0, cluster_ops)

end
