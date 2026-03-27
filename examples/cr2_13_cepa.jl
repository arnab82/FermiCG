using QCBase
using RDM
using FermiCG
using Printf
using JLD2

@load "../test/data_cmf_13_cr2_morokuma.jld2"

ref_fock = FockConfig([(3,0),(3,3),(0,3)])
nroots   = 4

clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
clustered_S2  = FermiCG.extract_S2(clusters)

# ── Spin eigenbasis M=30 ───────────────────────────────────────────────────────
println("\n Building Spin eigenbasis ...")
@time cb_spin = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1,
                [3,3,3], ref_fock, max_roots=50, verbose=0)

cluster_ops = FermiCG.compute_cluster_ops(cb_spin, ints)
FermiCG.add_cmf_operators!(cluster_ops, cb_spin, ints, d1.a, d1.b)

lbs = [sum(size(sol.vectors,2) for (_,sol) in cb.basis) for cb in cb_spin]
@printf(" Local basis sizes: %s\n", join(lbs, ", "))

# ── TPSCI reference ────────────────────────────────────────────────────────────
ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots)
ci_vector = FermiCG.add_spin_focksectors(ci_vector)

println("\n Running TPSCI (cipsi=6e-4) ...")
e_tpsci, v_tpsci = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham,
                                      thresh_cipsi=6e-4, max_iter=30, thresh_foi=1e-5)
e_tpsci, v_tpsci = FermiCG.tpsci_ci(v_tpsci, cluster_ops, clustered_ham,
                                      thresh_cipsi=4e-4, max_iter=30, thresh_foi=1e-6)
e_tpsci, v_tpsci = FermiCG.tpsci_ci(v_tpsci, cluster_ops, clustered_ham,
                                      thresh_cipsi=2e-4, max_iter=30, thresh_foi=1e-6)
e_tpsci, v_tpsci = FermiCG.tpsci_ci(v_tpsci, cluster_ops, clustered_ham,
                                      thresh_cipsi=1e-4, max_iter=30, thresh_foi=1e-6)
s2_tpsci = FermiCG.compute_expectation_value_parallel(v_tpsci, cluster_ops, clustered_S2)
@printf("\n TPSCI  TPS=%6i\n", length(v_tpsci))
@printf(" %-5s  %14s  %6s\n", "Root", "E(var)", "<S²>")
for r in 1:nroots
    @printf(" %5i  %14.8f  %6.3f\n", r, e_tpsci[r], s2_tpsci[r])
end

ept2 = FermiCG.compute_pt2_energy(v_tpsci, cluster_ops, clustered_ham, thresh_foi=1e-5)
@printf("\n TPSCI+PT2:\n")
for r in 1:nroots
    @printf(" %5i  %14.8f\n", r, e_tpsci[r]+ept2[r])
end

# # ── CEPA ───────────────────────────────────────────────────────────────────────
# # thresh_foi=1e-3 keeps FOIS small enough for direct solve; use 1e-5 for PT2
# println("\n Running CEPA (cepa-0) ...")
# @time e_cepa = FermiCG.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
#                                       cepa_shift="cepa", thresh_foi=1e-4,
#                                       nbody=4, tol=1e-6,solver=:minres, verbose=1)

# println("\n Running ACPF ...")
# @time e_acpf = FermiCG.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
#                                       cepa_shift="acpf", thresh_foi=1e-4,
#                                       nbody=4, tol=1e-6, solver=:minres, verbose=1)

# println("\n Running AQCC ...")
# @time e_aqcc = FermiCG.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
#                                       cepa_shift="aqcc", thresh_foi=1e-4,
#                                       nbody=4, tol=1e-6, solver=:krylovkit, verbose=1)
#. SPT CEPA#
# start by defining P/Q spaces
p_spaces = Vector{ClusterSubspace}()

ssi = ClusterSubspace(clusters[1])
add_subspace!(ssi, (3,0), 1:1)
add_subspace!(ssi, (2,1), 1:1)
add_subspace!(ssi, (1,2), 1:1)
add_subspace!(ssi, (0,3), 1:1)
push!(p_spaces, ssi)

ssi = ClusterSubspace(clusters[2])
add_subspace!(ssi, (3,3), 1:1)
push!(p_spaces, ssi)

ssi = ClusterSubspace(clusters[3])
add_subspace!(ssi, (3,0), 1:1)
add_subspace!(ssi, (2,1), 1:1)
add_subspace!(ssi, (1,2), 1:1)
add_subspace!(ssi, (0,3), 1:1)
push!(p_spaces, ssi)

ci_vector = BSTstate(clusters, p_spaces, cb_spin, R=nroots) 

na = 6
nb = 6
FermiCG.fill_p_space!(ci_vector, na, nb)
FermiCG.eye!(ci_vector)
ebst, vbst = FermiCG.ci_solve(ci_vector, cluster_ops, clustered_ham)

println("\n Running CEPA (cepa-0) ...")
@time e_cepa_spt = FermiCG.do_fois_cepa(vbst, cluster_ops, clustered_ham,
                                      cepa_shift="cepa", thresh_foi=1e-4,
                                      nbody=4, tol=1e-8, verbose=1)

println("\n Running ACPF ...")
@time e_acpf_spt = FermiCG.do_fois_cepa(vbst, cluster_ops, clustered_ham,
                                      cepa_shift="acpf", thresh_foi=1e-4,
                                      nbody=4, tol=1e-8, verbose=1)

println("\n Running AQCC ...")
@time e_aqcc_spt = FermiCG.do_fois_cepa(vbst, cluster_ops, clustered_ham,
                                      cepa_shift="aqcc", thresh_foi=1e-7,
                                      nbody=4, tol=1e-8, verbose=1)



# ── Summary ────────────────────────────────────────────────────────────────────
println()
W = 85
println("═"^W)
println(" Summary — Cr2 13-orbital, spin eigenbasis, cipsi=6e-4")
println("═"^W)
@printf(" %-12s  %5s  %14s\n", "Method", "Root", "Energy")
for r in 1:nroots
    @printf(" %-12s  %5i  %14.8f\n", "TPSCI",      r, e_tpsci[r])
end
for r in 1:nroots
    @printf(" %-12s  %5i  %14.8f\n", "TPSCI+PT2",  r, e_tpsci[r]+ept2[r])
end
for (lab, ev) in [("CEPA-0", e_cepa), ("ACPF", e_acpf), ("AQCC", e_aqcc)]
    for r in 1:nroots
        @printf(" %-12s  %5i  %14.8f\n", lab, r, ev[r])
    end
end
for (lab, ev) in [("CEPA-0", e_cepa_spt), ("ACPF", e_acpf_spt), ("AQCC", e_aqcc_spt)]
    for r in 1:nroots
        @printf(" %-12s  %5i  %14.8f\n", lab, r, ev[r])
    end
end
