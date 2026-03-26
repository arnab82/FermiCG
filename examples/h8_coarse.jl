using QCBase
using RDM
using FermiCG
using Printf
using JLD2

@load "../test/_testdata_cmf_h8.jld2"

ref_fock = FockConfig([(2,2),(2,2)])

clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)

local_basis_sizes(cbs) = [sum(size(sol.vectors,2) for (_,sol) in cb.basis) for cb in cbs]

function run_tpsci_coarse(cluster_bases, label; nroots=1)
    println()
    println("═══════════════════════════════════════════════")
    println(" Basis: $label")
    println("═══════════════════════════════════════════════")

    lbs = local_basis_sizes(cluster_bases)
    @printf(" Local basis sizes per cluster: %s\n", join(lbs, ", "))

    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints)
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots)
    ci_vector = FermiCG.add_spin_focksectors(ci_vector)

    e, v = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham,
                              thresh_cipsi=1e-3, max_iter=20, thresh_foi=1e-5)
    ept2 = FermiCG.compute_pt2_energy(v, cluster_ops, clustered_ham, thresh_foi=1e-5)
    @printf(" cipsi=1e-3 foi=1e-5  TPS=%6i  E= %14.8f  PT2= %14.8f  (corr: %+.2e)\n",
            length(v), e[1], e[1]+ept2[1], ept2[1])
    return e, e .+ ept2, lbs, length(v)
end

# ── EST spinbasis (per_sector, no S±) ─────────────────────────────────────────
println("\n Building EST spinbasis per_sector thresh=1e-3 ...")
@time cb_est = FermiCG.compute_cluster_est_spinbasis(ints, clusters, d1.a, d1.b,
                verbose=0, thresh_schmidt=1e-3, init_fspace=ref_fock, delta_elec=nothing,
                est_nr=1, est_max_cycles=200, per_sector=true, apply_spin_ladder=false)

# ── Spin eigenbasis M=10 ───────────────────────────────────────────────────────
println("\n Building Spin eigenbasis M=10 ...")
@time cb_spin = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [2,2], ref_fock,
                max_roots=10, verbose=0)

# ── Plain eigenbasis (no spin-preserving) M=10 ────────────────────────────────
println("\n Building Plain eigenbasis (no spin) M=10 ...")
@time cb_plain = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=10,
                rdm1a=d1.a, rdm1b=d1.b, init_fspace=ref_fock)

# ── Legacy EST basis (no S2 projection, per-sector FCI) ──────────────────────
println("\n Building Legacy EST basis (no S2 proj, per-sector) thresh=1e-3 ...")
@time cb_est_legacy = FermiCG.compute_cluster_est_basis(ints, clusters, d1.a, d1.b,
                verbose=0, thresh_schmidt=1e-3, init_fspace=ref_fock,
                delta_elec=[2,2], est_nr=1, est_max_cycles=200)

# ── Run TPSCI ─────────────────────────────────────────────────────────────────
r_est        = run_tpsci_coarse(cb_est,        "EST spinbasis per_sector thresh=1e-3")
r_spin       = run_tpsci_coarse(cb_spin,       "Spin eigenbasis M=10")
r_plain      = run_tpsci_coarse(cb_plain,      "Plain eigenbasis M=10")
r_est_legacy = run_tpsci_coarse(cb_est_legacy, "Legacy EST (no S2, per-sector) thresh=1e-3")

println()
println("═══════════════════════════════════════════════════════════════════════════════")
println(" Summary — Root 1")
println("═══════════════════════════════════════════════════════════════════════════════")
@printf(" %-40s  loc=%s  TPS=%5i  E= %14.8f  E+PT2= %14.8f\n",
        "EST spinbasis per_sector thresh=1e-3", join(r_est[3],        ","), r_est[4],        r_est[1][1],        r_est[2][1])
@printf(" %-40s  loc=%s  TPS=%5i  E= %14.8f  E+PT2= %14.8f\n",
        "Spin eigenbasis M=10",                join(r_spin[3],       ","), r_spin[4],       r_spin[1][1],       r_spin[2][1])
@printf(" %-40s  loc=%s  TPS=%5i  E= %14.8f  E+PT2= %14.8f\n",
        "Plain eigenbasis M=10 (no spin)",     join(r_plain[3],      ","), r_plain[4],      r_plain[1][1],      r_plain[2][1])
@printf(" %-40s  loc=%s  TPS=%5i  E= %14.8f  E+PT2= %14.8f\n",
        "Legacy EST (no S2, per-sector) thresh=1e-3", join(r_est_legacy[3], ","), r_est_legacy[4], r_est_legacy[1][1], r_est_legacy[2][1])
