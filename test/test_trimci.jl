using QCBase
using RDM
using FermiCG
using Printf
using Test
using LinearAlgebra
using Random
using JLD2

@testset "tpsci_trimci he4" begin
    @load "_testdata_cmf_he4.jld2"

    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops   = FermiCG.compute_cluster_ops(cluster_bases, ints)
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    nroots  = 5
    ref_fock = FermiCG.FockConfig(init_fspace)

    # ------------------------------------------------------------------
    # Build the same initial CI vector used in test_tpsci.jl
    # ------------------------------------------------------------------
    function make_ci_vector()
        cv = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)
        cv[ref_fock][ClusterConfig([2,1,1,1])] = [0,1,0,0,0]
        cv[ref_fock][ClusterConfig([1,2,1,1])] = [0,0,1,0,0]
        cv[ref_fock][ClusterConfig([1,1,2,1])] = [0,0,0,1,0]
        cv[ref_fock][ClusterConfig([1,1,1,2])] = [0,0,0,0,1]
        return cv
    end

    # ------------------------------------------------------------------
    # Reference energies from converged TPSCI (from test_tpsci.jl)
    # ------------------------------------------------------------------
    ref_e0 = [
        -16.886058279836007
        -15.435798355654288
        -15.4228104425694
        -15.422680127510118
        -15.409357434493316
    ]

    # ------------------------------------------------------------------
    # Test trim_tpsci!: should remove configs below thresh, keep the rest
    # ------------------------------------------------------------------
    @testset "trim_tpsci!" begin
        cv = make_ci_vector()
        # Set one config's coefficients to near-zero to verify it gets removed
        near_zero_coeff = 1e-10
        cv[ref_fock][ClusterConfig([2,1,1,1])] = fill(near_zero_coeff, nroots)
        len_before = length(cv)
        FermiCG.trim_tpsci!(cv, thresh=1e-8)
        # The near-zero config should be removed; the remaining 3 should stay
        @test length(cv) < len_before
        @test length(cv) == 3
    end

    # ------------------------------------------------------------------
    # Test tpsci_trimci: energies should agree with TPSCI to ~1e-3 Eh
    # (TRIMCI is an approximate method; large block_size disables block trim)
    # ------------------------------------------------------------------
    @testset "tpsci_trimci energies" begin
        ci_vector = make_ci_vector()
        Random.seed!(42)

        # Use large block_size (> expected dim) to disable Stage-1 block trim
        # and validate purely the expansion + global trim path against TPSCI.
        e0_trimci, v0_trimci = FermiCG.tpsci_trimci(
            ci_vector, cluster_ops, clustered_ham,
            thresh_cipsi     = 1e-3,
            thresh_foi       = 1e-8,
            thresh_trim      = 1e-10,
            block_size       = 10000,    # larger than any expected core dim
            thresh_asci      = -1,
            conv_thresh      = 1e-5,
            ci_conv          = 1e-10,
            ci_lindep_thresh = 1e-12,
            max_iter         = 20,
        )

        display(e0_trimci)
        # TRIMCI energies should agree with TPSCI within reasonable tolerance
        @test isapprox(abs.(e0_trimci), abs.(ref_e0), atol=1e-3)
    end

    # ------------------------------------------------------------------
    # Test tpsci_trimci with block trim enabled (small block_size)
    # ------------------------------------------------------------------
    @testset "tpsci_trimci with block trim" begin
        ci_vector = make_ci_vector()
        Random.seed!(42)

        e0_trimci, v0_trimci = FermiCG.tpsci_trimci(
            ci_vector, cluster_ops, clustered_ham,
            thresh_cipsi     = 1e-2,
            thresh_foi       = 1e-6,
            thresh_trim      = 1e-8,
            block_size       = 5,
            n_keep_per_block = 3,
            conv_thresh      = 1e-3,
            ci_conv          = 1e-8,
            max_iter         = 10,
        )

        display(e0_trimci)
        # With a larger thresh and small block, energies may differ from TPSCI;
        # just verify the method runs without error and returns the right types.
        @test length(e0_trimci) == nroots
        @test length(v0_trimci) > 0
    end
end
