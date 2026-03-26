using QCBase
using FermiCG
using NPZ
using InCoreIntegrals
using RDM
using JLD2
using Printf

@load "./../../test/data_cmf_13_cr2_morokuma.jld2"

M = 20

init_fspace = FockConfig([(3, 0), (3, 3), (0, 3)])

cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3], init_fspace, max_roots=M, verbose=0);
# cluster_bases = FermiCG.compute_cluster_est_basis(ints, clusters, d1.a, d1.b,
                    # verbose=1, thresh_schmidt=5e-5, init_fspace=init_fspace, delta_elec=[3,3,3])
clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

nroots=4

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


ci_vector = BSTstate(clusters, p_spaces, cluster_bases, R=4)

na = 6
nb = 6
FermiCG.fill_p_space!(ci_vector, na, nb)
FermiCG.eye!(ci_vector)
ebst, vbst = FermiCG.ci_solve(ci_vector, cluster_ops, clustered_ham)


thresh_spf=0.01
e_var1, v_var = FermiCG.block_sparse_tucker(vbst, cluster_ops, clustered_ham,
                                               max_iter    = 20,
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = thresh_spf,
                                               thresh_spin = thresh_spf,
                                               thresh_foi  = thresh_spf/50,
                                               thresh_pt   = thresh_spf/2,
                                               ci_conv     = 5e-5,
                                               do_pt       = false,
                                               tol_tucker  = 1e-5,
                                               resolve_ss  = true,
                                               verbose     = 1)
@save "data_spt.jld2" e_var1 v_var cluster_bases


# cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3], init_fspace, max_roots=M, verbose=1);
# --- original EST (may have spin contamination) ---
# cluster_bases = FermiCG.compute_cluster_est_basis(ints, clusters, d1.a, d1.b,
#                     verbose=0, thresh_schmidt=1e-5, init_fspace=init_fspace, delta_elec=[3,3,3],
#                     est_max_cycles=500)

# --- new EST spin basis: S+/S- before SVD → spin-pure, entangled states ---
# main path used when fragment+bath FCI dim ≤ max_fci_dim (default 5M);
# falls back to fragment-only FCI + S+/S- for large clusters
cluster_bases = FermiCG.compute_cluster_est_spinbasis(ints, clusters, d1.a, d1.b,
                    verbose=1, thresh_schmidt=1e-5, init_fspace=init_fspace, delta_elec=[3,3,3],
                    est_max_cycles=500, max_fci_dim=5_000_000)

# @load "data_spt.jld2" e_var1 v_var cluster_bases cluster_ops
clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

nroots=4

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


ci_vector = BSTstate(clusters, p_spaces, cluster_bases, R=4)

na = 6
nb = 6
FermiCG.fill_p_space!(ci_vector, na, nb)
FermiCG.eye!(ci_vector)
ebst, vbst = FermiCG.ci_solve(ci_vector, cluster_ops, clustered_ham)


thresh_spf=0.008
e_var2, v_var2 = FermiCG.block_sparse_tucker(vbst, cluster_ops, clustered_ham,
                                               max_iter    = 20,
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = thresh_spf,
                                               thresh_spin = thresh_spf,
                                               thresh_foi  = thresh_spf/100,
                                               thresh_pt   = thresh_spf/2,
                                               ci_conv     = 5e-5,
                                               do_pt       = false,
                                               tol_tucker  = 1e-5,
                                               resolve_ss  = true,
                                               verbose     = 1)
@save "data_spt.jld2" e_var2 v_var2 cluster_bases
thresh_spf=0.002
e_var2, v_var2 = FermiCG.block_sparse_tucker(v_var2, cluster_ops, clustered_ham,
                                               max_iter    = 20,
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = thresh_spf,
                                               thresh_spin = thresh_spf/1.5,
                                               thresh_foi  = thresh_spf/100,
                                               thresh_pt   = thresh_spf/2,
                                               ci_conv     = 5e-5,
                                               do_pt       = false,
                                               tol_tucker  = 1e-5,
                                               resolve_ss  = true,
                                               verbose     = 1)