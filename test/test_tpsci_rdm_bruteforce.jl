"""
test_tpsci_rdm_bruteforce.jl

Independent determinant-space reference test for TPSCI RDMs.

This test builds a tiny 4-orbital, 2-cluster problem with identity cluster
bases and random orthonormal TPSCI coefficients. It then compares production
1-RDM, spin-flip 1-RDM, and spin-free 2-RDM code against a brute-force
determinant bitstring implementation of the same second-quantized operators.

Run with:
    julia --threads auto --project=.. test_tpsci_rdm_bruteforce.jl
"""

using ActiveSpaceSolvers
using FermiCG
using InCoreIntegrals
using LinearAlgebra
using Random
using StaticArrays
using Test

function _combinations_lex(n::Integer, k::Integer)
    if k == 0
        return [Int[]]
    end

    comb = collect(1:k)
    out = Vector{Vector{Int}}()
    while true
        push!(out, copy(comb))
        advanced = false
        for i in k:-1:1
            if comb[i] < n - k + i
                comb[i] += 1
                for j in i+1:k
                    comb[j] = comb[j-1] + 1
                end
                advanced = true
                break
            end
        end
        advanced || break
    end
    return out
end

function _identity_cluster_bases(clusters)
    cluster_bases = FermiCG.ClusterBasis[]
    for ci in clusters
        cb = FermiCG.ClusterBasis(ci)
        norb_i = length(ci)
        for na in 0:norb_i, nb in 0:norb_i
            ansatz = FCIAnsatz(norb_i, na, nb)
            cb[(Int16(na), Int16(nb))] =
                ActiveSpaceSolvers.Solution(ansatz, zeros(ansatz.dim),
                                            Matrix{Float64}(I, ansatz.dim, ansatz.dim))
        end
        push!(cluster_bases, cb)
    end
    return cluster_bases
end

function _all_fixed_particle_product_entries(cluster_bases, total_na, total_nb)
    nclusters = length(cluster_bases)
    sectors = [collect(keys(cb.basis)) for cb in cluster_bases]
    entries = Tuple{FockConfig{2},ClusterConfig{2}}[]

    for focks in Iterators.product(sectors...)
        sum(f[1] for f in focks) == total_na || continue
        sum(f[2] for f in focks) == total_nb || continue

        dims = [size(cluster_bases[i][focks[i]].vectors, 2) for i in 1:nclusters]
        for config in Iterators.product((1:d for d in dims)...)
            push!(entries, (FermiCG.FockConfig(collect(focks)),
                            FermiCG.ClusterConfig(collect(config))))
        end
    end
    return entries
end

function _random_orthonormal_tpsci_state(clusters, cluster_bases; total_na=2, total_nb=2, nroots=2)
    entries = _all_fixed_particle_product_entries(cluster_bases, total_na, total_nb)
    rng = MersenneTwister(20240610)
    raw = randn(rng, length(entries), nroots)
    Q = Matrix(qr(raw).Q)[:, 1:nroots]

    state = FermiCG.TPSCIstate(clusters; T=Float64, R=nroots)
    for (row, (fock, config)) in enumerate(entries)
        if !haskey(state, fock)
            FermiCG.add_fockconfig!(state, fock)
        end
        state[fock][config] = MVector{nroots,Float64}(Q[row, :])
    end
    return state
end

function _spinorb_lookup(clusters)
    lookup = Dict{Tuple{Int,Symbol},Int}()
    base = 0
    for ci in clusters
        norb_i = length(ci)
        for (local_idx, orb) in enumerate(ci.orb_list)
            lookup[(Int(orb), :alpha)] = base + local_idx
            lookup[(Int(orb), :beta)] = base + norb_i + local_idx
        end
        base += 2norb_i
    end
    return lookup
end

function _apply_annihilation(det::UInt64, spinorb::Integer)
    bit = UInt64(1) << (spinorb - 1)
    (det & bit) != 0 || return 0.0, det
    before = det & (bit - UInt64(1))
    sign = isodd(count_ones(before)) ? -1.0 : 1.0
    return sign, det ⊻ bit
end

function _apply_creation(det::UInt64, spinorb::Integer)
    bit = UInt64(1) << (spinorb - 1)
    (det & bit) == 0 || return 0.0, det
    before = det & (bit - UInt64(1))
    sign = isodd(count_ones(before)) ? -1.0 : 1.0
    return sign, det | bit
end

function _local_occ_from_state_index(norb_i, na, nb, idx)
    alpha_combs = _combinations_lex(norb_i, na)
    beta_combs = _combinations_lex(norb_i, nb)
    dima = length(alpha_combs)
    alpha_idx = mod1(idx, dima)
    beta_idx = div(idx - 1, dima) + 1
    return alpha_combs[alpha_idx], beta_combs[beta_idx]
end

function _full_coefficients_from_identity_cluster_state(state, clusters, cluster_bases)
    spinorb = _spinorb_lookup(clusters)
    coeffs = Dict{UInt64,Vector{Float64}}()
    R = size(state)[2]

    for (fock, configs) in state.data
        for (config, coeff) in configs
            det = UInt64(0)
            for ci in clusters
                ci_idx = ci.idx
                norb_i = length(ci)
                na, nb = fock[ci_idx]
                local_idx = Int(config[ci_idx])
                occ_a, occ_b = _local_occ_from_state_index(norb_i, na, nb, local_idx)

                for local_idx in occ_a
                    det |= UInt64(1) << (spinorb[(Int(ci.orb_list[local_idx]), :alpha)] - 1)
                end
                for local_idx in occ_b
                    det |= UInt64(1) << (spinorb[(Int(ci.orb_list[local_idx]), :beta)] - 1)
                end
            end
            coeffs[det] = get(coeffs, det, zeros(Float64, R)) .+ Vector(coeff)
        end
    end
    return coeffs
end

function _bruteforce_1rdm(coeffs, clusters, create_spin::Symbol, annihilate_spin::Symbol, nroots)
    norb = sum(length, clusters)
    spinorb = _spinorb_lookup(clusters)
    gamma = zeros(Float64, norb, norb, nroots, nroots)

    for (ket_det, cket) in coeffs
        for q in 1:norb, p in 1:norb
            s1, det1 = _apply_annihilation(ket_det, spinorb[(q, annihilate_spin)])
            iszero(s1) && continue
            s2, bra_det = _apply_creation(det1, spinorb[(p, create_spin)])
            iszero(s2) && continue
            cbra = get(coeffs, bra_det, nothing)
            cbra === nothing && continue

            sign = s1 * s2
            for r2 in 1:nroots, r1 in 1:nroots
                gamma[p, q, r1, r2] += sign * cbra[r1] * cket[r2]
            end
        end
    end
    return gamma
end

function _bruteforce_spinfree_2rdm(coeffs, clusters, nroots)
    norb = sum(length, clusters)
    spinorb = _spinorb_lookup(clusters)
    Gamma = zeros(Float64, norb, norb, norb, norb, nroots, nroots)
    spin_cases = ((:alpha, :alpha), (:beta, :beta), (:alpha, :beta), (:beta, :alpha))

    for (ket_det, cket) in coeffs
        for (sigma, tau) in spin_cases
            for s in 1:norb, r in 1:norb, q in 1:norb, p in 1:norb
                # Gamma[p,q,r,s] = <p'_sigma q'_tau s_tau r_sigma>
                s1, det1 = _apply_annihilation(ket_det, spinorb[(r, sigma)])
                iszero(s1) && continue
                s2, det2 = _apply_annihilation(det1, spinorb[(s, tau)])
                iszero(s2) && continue
                s3, det3 = _apply_creation(det2, spinorb[(q, tau)])
                iszero(s3) && continue
                s4, bra_det = _apply_creation(det3, spinorb[(p, sigma)])
                iszero(s4) && continue
                cbra = get(coeffs, bra_det, nothing)
                cbra === nothing && continue

                sign = s1 * s2 * s3 * s4
                for r2 in 1:nroots, r1 in 1:nroots
                    Gamma[p, q, r, s, r1, r2] += sign * cbra[r1] * cket[r2]
                end
            end
        end
    end
    return Gamma
end

@testset "TPSCI RDM brute-force determinant reference" begin
    clusters = [MOCluster(1, 1:2), MOCluster(2, 3:4)]
    cluster_bases = _identity_cluster_bases(clusters)
    ints = InCoreInts(0.0, zeros(4, 4), zeros(4, 4, 4, 4))
    cluster_ops = FermiCG.compute_cluster_ops_2rdm(cluster_bases, ints)

    psi = _random_orthonormal_tpsci_state(clusters, cluster_bases;
                                          total_na=2, total_nb=2, nroots=2)
    coeffs = _full_coefficients_from_identity_cluster_state(psi, clusters, cluster_bases)

    gamma_aa, gamma_bb = FermiCG.compute_1rdm(psi, cluster_ops)
    gamma_ab, gamma_ba = FermiCG.compute_1rdm_sf(psi, cluster_ops)
    Gamma = FermiCG.compute_2rdm(psi, cluster_ops)

    @test isapprox(gamma_aa, _bruteforce_1rdm(coeffs, clusters, :alpha, :alpha, 2), atol=1e-12)
    @test isapprox(gamma_bb, _bruteforce_1rdm(coeffs, clusters, :beta, :beta, 2), atol=1e-12)
    @test isapprox(gamma_ab, _bruteforce_1rdm(coeffs, clusters, :alpha, :beta, 2), atol=1e-12)
    @test isapprox(gamma_ba, _bruteforce_1rdm(coeffs, clusters, :beta, :alpha, 2), atol=1e-12)
    @test isapprox(Gamma, _bruteforce_spinfree_2rdm(coeffs, clusters, 2), atol=1e-12)
end
