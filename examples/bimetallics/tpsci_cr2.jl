using QCBase
using FermiCG
using NPZ
using InCoreIntegrals
using RDM
using JLD2
using Printf

@load "./../../test/data_cmf_13_cr2_morokuma.jld2"

M = 100

init_fspace = FockConfig([(3, 0), (3, 3), (0, 3)])

# cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3], init_fspace, max_roots=M, verbose=1);
# # cluster_bases = FermiCG.compute_cluster_est_basis(ints, clusters, d1.a, d1.b,
#                     # verbose=1, thresh_schmidt=5e-5, init_fspace=init_fspace, delta_elec=[3,3,3])
# clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
# cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

# FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

# nroots=4
# ci_vector = FermiCG.TPSCIstate(clusters, init_fspace, R=nroots)
# ci_vector = FermiCG.add_spin_focksectors(ci_vector)

# e_tpsci, v_tpsci = FermiCG.tps_ci_direct(ci_vector, cluster_ops, clustered_ham)
# @printf(" TPSCI (direct):  %14.8f Ha\n", e_tpsci[1])

# e_tpsci2, v_tpsci2 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, thresh_cipsi=1e-4)
# cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3], init_fspace, max_roots=M, verbose=1);
cluster_bases = FermiCG.compute_cluster_est_spinbasis(ints, clusters, d1.a, d1.b,
                    verbose=1, thresh_schmidt=1e-5, init_fspace=init_fspace, delta_elec=[3,3,3],
                    est_max_cycles=500, max_fci_dim=5_000_000)

# @load "data_spt.jld2" e_var1 v_var cluster_bases cluster_ops
clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

nroots=4
ci_vector = FermiCG.TPSCIstate(clusters, init_fspace, R=nroots)
ci_vector = FermiCG.add_spin_focksectors(ci_vector)

e_tpsci, v_tpsci = FermiCG.tps_ci_direct(ci_vector, cluster_ops, clustered_ham)
@printf(" TPSCI (direct):  %14.8f Ha\n", e_tpsci[1])

e_tpsci2, v_tpsci2 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, thresh_cipsi=1e-3,thresh_spin=1e-3,max_iter = 40)
e_tpsci2, v_tpsci2 = FermiCG.tpsci_ci(v_tpsci2, cluster_ops, clustered_ham, thresh_cipsi=8e-4,thresh_spin=8e-4,max_iter = 40)
e_tpsci2, v_tpsci2 = FermiCG.tpsci_ci(v_tpsci2, cluster_ops, clustered_ham, thresh_cipsi=6e-4,thresh_spin=6e-4,max_iter = 40)