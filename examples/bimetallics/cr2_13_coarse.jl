using QCBase
using RDM
using FermiCG
using Printf
using JLD2

@load "../../test/data_cmf_13_cr2_morokuma.jld2"

ref_fock = FockConfig([(3,0),(3,3),(0,3)])
nroots   = 4

clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
clustered_S2  = FermiCG.extract_S2(clusters)

local_basis_sizes(cbs) = [sum(size(sol.vectors,2) for (_,sol) in cb.basis) for cb in cbs]

function run_tpsci(cluster_bases, label; nroots=4)
    println()
    println("═══════════════════════════════════════════════════════")
    println(" Basis: $label")
    println("═══════════════════════════════════════════════════════")

    lbs = local_basis_sizes(cluster_bases)
    @printf(" Local basis sizes per cluster: %s\n", join(lbs, ", "))

    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints)
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots)
    ci_vector = FermiCG.add_spin_focksectors(ci_vector)

    function print_s2(v, e, tag)
        s2 = FermiCG.compute_expectation_value_parallel(v, cluster_ops, clustered_S2)
        @printf(" %-35s  %5s  %10s  %6s\n", tag, "Root", "Energy", "<S²>")
        for r in 1:length(e)
            @printf(" %-35s  %5i  %10.6f  %6.3f\n", "", r, e[r], s2[r])
        end
    end

    # coarse
    e1, v1 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham,
                               thresh_cipsi=1e-3, max_iter=30, thresh_foi=1e-5)
    ept1 = FermiCG.compute_pt2_energy(v1, cluster_ops, clustered_ham, thresh_foi=1e-5)
    @printf("\n cipsi=1e-3  TPS=%6i\n", length(v1))
    print_s2(v1, e1, "cipsi=1e-3")

    # fine 8e-4
    e2, v2 = FermiCG.tpsci_ci(v1, cluster_ops, clustered_ham,
                               thresh_cipsi=8e-4, max_iter=30, thresh_foi=1e-5)
    ept2 = FermiCG.compute_pt2_energy(v2, cluster_ops, clustered_ham, thresh_foi=1e-5)
    @printf("\n cipsi=8e-4  TPS=%6i\n", length(v2))
    print_s2(v2, e2, "cipsi=8e-4")

    # finest 6e-4
    e3, v3 = FermiCG.tpsci_ci(v2, cluster_ops, clustered_ham,
                               thresh_cipsi=6e-4, max_iter=30, thresh_foi=1e-5)
    ept3 = FermiCG.compute_pt2_energy(v3, cluster_ops, clustered_ham, thresh_foi=1e-5)
    @printf("\n cipsi=6e-4  TPS=%6i\n", length(v3))
    print_s2(v3, e3, "cipsi=6e-4")

    println()
    @printf(" %-10s  %5s  %14s  %14s\n", "thresh", "Root", "E(var)", "E+PT2")
    for (tag, ev, ep) in [("1e-3", e1, ept1), ("8e-4", e2, ept2), ("6e-4", e3, ept3)]
        for r in 1:length(ev)
            @printf(" %-10s  %5i  %14.8f  %14.8f\n", tag, r, ev[r], ev[r]+ep[r])
        end
    end
    return e3, e3 .+ ept3, lbs, length(v3)
end

# ── EST spinbasis (per_sector, no S±) ─────────────────────────────────────────
println("\n Building EST spinbasis per_sector thresh=1e-4 ...")
@time cb_est = FermiCG.compute_cluster_est_spinbasis(ints, clusters, d1.a, d1.b,
                verbose=0, thresh_schmidt=1e-4, init_fspace=ref_fock, delta_elec=[3,3,3],
                target_norb=10,
                est_nr=1, est_max_cycles=200, per_sector=true, apply_spin_ladder=false)

# ── Spin eigenbasis M=30 ───────────────────────────────────────────────────────
println("\n Building Spin eigenbasis M=30 ...")
@time cb_spin = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1,
                [3,3,3], ref_fock, max_roots=30, verbose=0)

# ── Run TPSCI ─────────────────────────────────────────────────────────────────
r_est  = run_tpsci(cb_est,  "EST spinbasis per_sector thresh=1e-4")
r_spin = run_tpsci(cb_spin, "Spin eigenbasis M=30")

println()
W = 90
println("═"^W)
println(" Summary — Root 1 (cipsi=6e-4)")
println("═"^W)
@printf(" %-44s  loc=%s  TPS=%5i  E= %14.8f  E+PT2= %14.8f\n",
        "EST spinbasis per_sector thresh=1e-4", join(r_est[3],  ","), r_est[4],  r_est[1][1],  r_est[2][1])
@printf(" %-44s  loc=%s  TPS=%5i  E= %14.8f  E+PT2= %14.8f\n",
        "Spin eigenbasis M=30",                join(r_spin[3], ","), r_spin[4], r_spin[1][1], r_spin[2][1])
