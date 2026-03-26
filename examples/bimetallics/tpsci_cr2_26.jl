using QCBase
using FermiCG
using InCoreIntegrals
using RDM
using JLD2
using Printf

@load "/Users/arnab/arnab/workspace/cMF_data/bimetallics/cr2_morokuma/26__3d4d_2p3p_3d4d/data_cmf_26_cr2.jld2"

# 26-orbital Cr2: clusters [(1-10),(11-16),(17-26)], fspace [(5,2),(3,3),(5,2)]
ref_fock = FockConfig(init_fspace)
M        = 40
nroots   = 4

clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)

# ── Helper: run TPSCI + PT2 ───────────────────────────────────────────────────
function run_tpsci(cluster_bases, label; nroots=4)
    println()
    println("═══════════════════════════════════════════════════════")
    println(" Basis: $label")
    println("═══════════════════════════════════════════════════════")

    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints)
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots)
    ci_vector = FermiCG.add_spin_focksectors(ci_vector)

    e_tpsci, v_tpsci = FermiCG.tps_ci_direct(ci_vector, cluster_ops, clustered_ham)
    @printf(" TPSCI (direct):  %14.8f Ha\n", e_tpsci[1])

    e_tpsci2, v_tpsci2 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham,
                                            thresh_cipsi=1e-3, max_iter=30, thresh_foi=1e-5)
    e_tpsci2, v_tpsci2 = FermiCG.tpsci_ci(v_tpsci2, cluster_ops, clustered_ham,
                                            thresh_cipsi=8e-4, max_iter=30, thresh_foi=1e-5)
    @printf(" TPSCI (cipsi):   %14.8f Ha\n", e_tpsci2[1])

    ept2 = FermiCG.compute_pt2_energy(v_tpsci2, cluster_ops, clustered_ham, thresh_foi=1e-5)
    e_total_pt2 = e_tpsci2 .+ ept2
    @printf(" TPSCI+PT2:       %14.8f Ha  (correction: %+.6e)\n", e_total_pt2[1], ept2[1])

    return e_tpsci, e_tpsci2, e_total_pt2, v_tpsci2
end

# ── EST spinbasis: per_sector, no S±, thresh_schmidt=1e-3, max_bath=4 ─────────
results_est = []
for ts in [1e-3, 5e-4, 1e-4]
    label = @sprintf("per_sector, thresh=%.0e, target_norb=16", ts)
    println("\n Building EST spinbasis ($label)...")
    @time cb = FermiCG.compute_cluster_est_spinbasis(ints, clusters, d1.a, d1.b,
                    verbose=1, thresh_schmidt=ts, init_fspace=ref_fock, delta_elec=[3,3,3],
                    est_nr=1, est_max_cycles=500, max_fci_dim=50_000_000,
                    per_sector=true, apply_spin_ladder=false, target_norb=16)
    e_dir, e_ci, e_pt2, _ = run_tpsci(cb, "EST spinbasis ($label)")
    push!(results_est, (label, e_dir, e_ci, e_pt2))
end

# ── Spin eigenbasis ───────────────────────────────────────────────────────────
println("\n Building spin eigenbasis (max_roots=$M)...")
@time cb_spin = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3], ref_fock,
                    max_roots=M, verbose=1)
e_sp_dir, e_sp_ci, e_sp_pt2, _ = run_tpsci(cb_spin, "Spin eigenbasis (max_roots=$M)")

# ── Summary ───────────────────────────────────────────────────────────────────
println()
println("═══════════════════════════════════════════════════════════════════════════════")
println(" Comparison Summary")
println("═══════════════════════════════════════════════════════════════════════════════")
@printf(" %-50s  %14s  %14s  %14s\n", "Basis", "TPSCI(direct)", "TPSCI(cipsi)", "TPSCI+PT2")
println("───────────────────────────────────────────────────────────────────────────────")
for i in 1:nroots
    println(" Root $i:")
    for (label, e_dir, e_ci, e_pt2) in results_est
        @printf("   %-50s  %14.8f  %14.8f  %14.8f\n", label, e_dir[i], e_ci[i], e_pt2[i])
    end
    @printf("   %-50s  %14.8f  %14.8f  %14.8f\n", "Spin eigenbasis (M=$M)", e_sp_dir[i], e_sp_ci[i], e_sp_pt2[i])
    println()
end
