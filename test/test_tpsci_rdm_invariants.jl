"""
test_tpsci_rdm_invariants.jl

Broader invariant and API checks for TPSCI 1-RDM, spin-flip 1-RDM, and
spin-free 2-RDM code.

This is intentionally standalone rather than included in runtests.jl because
building exact local 2-RDM operators is noticeably more expensive than a normal
TPSCI test.

Run with:
    julia --threads auto --project=.. test_tpsci_rdm_invariants.jl
"""

using FermiCG
using JLD2
using LinearAlgebra
using Printf
using QCBase
using RDM
using Test

function _slice_tpsci_roots(v, roots)
    out = FermiCG.TPSCIstate(v; R=length(roots))
    for (fock, configs) in v.data
        for (config, coeffs) in configs
            out[fock][config] .= coeffs[roots]
        end
    end
    return out
end

function _manual_1rdm_property(gamma_aa, gamma_bb, prop)
    norb, _, R1, R2 = size(gamma_aa)
    out = zeros(eltype(prop), R1, R2)
    for r2 in 1:R2, r1 in 1:R1
        acc = zero(eltype(prop))
        for q in 1:norb, p in 1:norb
            acc += prop[p, q] * (gamma_aa[p, q, r1, r2] + gamma_bb[p, q, r1, r2])
        end
        out[r1, r2] = acc
    end
    return out
end

function _energy_from_rdms(ints, gamma_aa, gamma_bb, Gamma, root)
    norb = size(Gamma, 1)
    gamma_tot = gamma_aa[:, :, root, root] .+ gamma_bb[:, :, root, root]
    Gamma_root = Gamma[:, :, :, :, root, root]

    E1 = dot(ints.h1, gamma_tot)
    E2 = zero(eltype(Gamma))
    for s in 1:norb, r in 1:norb, q in 1:norb, p in 1:norb
        E2 += ints.h2[p, r, q, s] * Gamma_root[p, q, r, s]
    end
    return E1 + 0.5 * E2
end

@testset "TPSCI RDM invariants and API guards" begin
    cd(@__DIR__)
    @load "_testdata_cmf_he4.jld2"

    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)

    # Default TPSCI cluster_ops should stay lightweight and not include Ppqsr.
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints)
    @test all(!haskey(ops, "Ppqsr") for ops in cluster_ops)

    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    nroots = 2
    ref_fock = FermiCG.FockConfig(init_fspace)
    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)
    ci_vector[ref_fock][ClusterConfig([1, 1, 1, 1])] = [1.0, 0.0]
    ci_vector[ref_fock][ClusterConfig([2, 1, 1, 1])] = [0.0, 1.0]

    e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham,
                              incremental=true, ci_conv=1e-10,
                              thresh_cipsi=1e-3, thresh_foi=1e-8,
                              thresh_asci=-1, conv_thresh=1e-7,
                              ci_lindep_thresh=1e-12)

    n_alpha = sum(f[1] for f in init_fspace)
    n_beta = sum(f[2] for f in init_fspace)
    n_elec = n_alpha + n_beta

    # The full 2-RDM must refuse ordinary cluster_ops instead of silently using
    # an incomplete on-cluster approximation.
    @test_throws ErrorException FermiCG.compute_2rdm(v0, cluster_ops)

    FermiCG.add_spinfree_2rdm_ops!(cluster_ops, cluster_bases)
    @test all(haskey(ops, "Ppqsr") for ops in cluster_ops)
    for ops in cluster_ops
        norb_i = length(ops.cluster)
        for (_, Ppqsr) in ops["Ppqsr"]
            @test ndims(Ppqsr) == 3
            @test size(Ppqsr, 1) == norb_i^4
        end
    end

    gamma_aa, gamma_bb = FermiCG.compute_1rdm(v0, cluster_ops)
    gamma_aa_t, gamma_bb_t = FermiCG.compute_1rdm_threaded(v0, cluster_ops)
    gamma_ab, gamma_ba = FermiCG.compute_1rdm_sf(v0, cluster_ops)
    gamma_ab_t, gamma_ba_t = FermiCG.compute_1rdm_sf_threaded(v0, cluster_ops)

    Gamma = FermiCG.compute_2rdm(v0, cluster_ops)
    Gamma_t = FermiCG.compute_2rdm_threaded(v0, cluster_ops)
    Gamma_b = FermiCG.compute_2rdm_blas(v0, cluster_ops)

    norb = size(gamma_aa, 1)
    identity_prop = Matrix{Float64}(I, norb, norb)

    @testset "1-RDM implementation agreement" begin
        @test isapprox(gamma_aa, gamma_aa_t, atol=1e-12)
        @test isapprox(gamma_bb, gamma_bb_t, atol=1e-12)
        @test isapprox(gamma_ab, gamma_ab_t, atol=1e-12)
        @test isapprox(gamma_ba, gamma_ba_t, atol=1e-12)
    end

    @testset "1-RDM traces and transition Hermiticity" begin
        for r2 in 1:nroots, r1 in 1:nroots
            delta = r1 == r2 ? 1.0 : 0.0
            @test isapprox(tr(gamma_aa[:, :, r1, r2]), n_alpha * delta, atol=1e-10)
            @test isapprox(tr(gamma_bb[:, :, r1, r2]), n_beta * delta, atol=1e-10)

            @test isapprox(gamma_aa[:, :, r1, r2],
                           transpose(gamma_aa[:, :, r2, r1]), atol=1e-10)
            @test isapprox(gamma_bb[:, :, r1, r2],
                           transpose(gamma_bb[:, :, r2, r1]), atol=1e-10)
            @test isapprox(gamma_ab[:, :, r1, r2],
                           transpose(gamma_ba[:, :, r2, r1]), atol=1e-10)
        end

        N_from_identity = FermiCG.contract_1rdm_property(gamma_aa, gamma_bb, identity_prop)
        @test isapprox(N_from_identity, n_elec .* Matrix{Float64}(I, nroots, nroots), atol=1e-10)
    end

    @testset "1-RDM property contractions" begin
        P_from_rdm = FermiCG.contract_1rdm_property(gamma_aa, gamma_bb, ints.h1)
        P_direct = FermiCG.compute_1e_property_direct(v0, cluster_ops, ints.h1)
        @test isapprox(P_from_rdm, P_direct, atol=1e-10)

        P_list = FermiCG.contract_1rdm_property(gamma_aa, gamma_bb,
                                                [ints.h1, identity_prop])
        @test isapprox(P_list[1], P_from_rdm, atol=1e-12)
        @test isapprox(P_list[2], n_elec .* Matrix{Float64}(I, nroots, nroots), atol=1e-10)

        complex_prop = ComplexF64.(ints.h1)
        for q in 1:norb, p in 1:norb
            complex_prop[p, q] += im * (p + 2q) / (10norb)
        end
        P_complex = FermiCG.contract_1rdm_property(ComplexF64.(gamma_aa),
                                                   ComplexF64.(gamma_bb),
                                                   complex_prop)
        @test isapprox(P_complex,
                       _manual_1rdm_property(ComplexF64.(gamma_aa),
                                             ComplexF64.(gamma_bb),
                                             complex_prop),
                       atol=1e-12)
    end

    @testset "2-RDM implementation agreement" begin
        @test size(Gamma) == (norb, norb, norb, norb, nroots, nroots)
        @test isapprox(Gamma, Gamma_t, atol=1e-12)
        @test isapprox(Gamma, Gamma_b, atol=1e-12)
    end

    @testset "2-RDM transition identities" begin
        gamma_tot = gamma_aa .+ gamma_bb
        for r2 in 1:nroots, r1 in 1:nroots
            partial_trace = zeros(Float64, norb, norb)
            for q in 1:norb
                partial_trace .+= Gamma[:, q, :, q, r1, r2]
            end
            @test isapprox(partial_trace,
                           (n_elec - 1) .* gamma_tot[:, :, r1, r2],
                           atol=1e-9)

            @test isapprox(sum(Gamma[p, q, p, q, r1, r2]
                               for p in 1:norb, q in 1:norb),
                           n_elec * (n_elec - 1) * (r1 == r2 ? 1.0 : 0.0),
                           atol=1e-8)

            @test isapprox(Gamma[:, :, :, :, r1, r2],
                           permutedims(Gamma[:, :, :, :, r1, r2], [2, 1, 4, 3]),
                           atol=1e-10)
            @test isapprox(Gamma[:, :, :, :, r1, r2],
                           permutedims(Gamma[:, :, :, :, r2, r1], [3, 4, 1, 2]),
                           atol=1e-10)
        end
    end

    @testset "2-RDM energy reconstruction" begin
        for root in 1:nroots
            @test isapprox(_energy_from_rdms(ints, gamma_aa, gamma_bb, Gamma, root),
                           e0[root], atol=1e-7)
        end
    end

    @testset "Single-root bra/ket APIs match multi-root blocks" begin
        bra = _slice_tpsci_roots(v0, [1])
        ket = _slice_tpsci_roots(v0, [2])

        gamma_aa_12, gamma_bb_12 = FermiCG.compute_1rdm(bra, ket, cluster_ops)
        gamma_ab_12, gamma_ba_12 = FermiCG.compute_1rdm_sf(bra, ket, cluster_ops)
        Gamma_12 = FermiCG.compute_2rdm(bra, ket, cluster_ops)

        @test isapprox(gamma_aa_12[:, :, 1, 1], gamma_aa[:, :, 1, 2], atol=1e-12)
        @test isapprox(gamma_bb_12[:, :, 1, 1], gamma_bb[:, :, 1, 2], atol=1e-12)
        @test isapprox(gamma_ab_12[:, :, 1, 1], gamma_ab[:, :, 1, 2], atol=1e-12)
        @test isapprox(gamma_ba_12[:, :, 1, 1], gamma_ba[:, :, 1, 2], atol=1e-12)
        @test isapprox(Gamma_12[:, :, :, :, 1, 1], Gamma[:, :, :, :, 1, 2], atol=1e-12)
    end

    @printf("RDM invariant tests passed for %d roots, %d orbitals\n", nroots, norb)
end
