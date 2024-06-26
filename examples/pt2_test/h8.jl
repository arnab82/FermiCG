using QCBase
using RDM
using FermiCG
using Printf
using Test
using JLD2


@load "_testdata_cmf_h8.jld2"
ref_fock = FockConfig(init_fspace)
# Do TPS
M=20
cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3], ref_fock, max_roots=M, verbose=1);
#cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=M, init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)

clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

nroots=1

# TPSCI
#
ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots)

ci_vector = FermiCG.add_spin_focksectors(ci_vector)

display(ci_vector)
etpsci, vtpsci = FermiCG.tps_ci_direct(ci_vector, cluster_ops, clustered_ham);

ept1 = FermiCG.compute_pt2_energy(vtpsci, cluster_ops, clustered_ham, thresh_foi=1e-12)



# BST
#

# start by defining P/Q spaces
p_spaces = Vector{ClusterSubspace}()

for ci in clusters
    ssi = ClusterSubspace(clusters[ci.idx])

    num_states_in_p_space = 1
    # our clusters are near triangles, with degenerate gs, so keep two states
    add_subspace!(ssi, ref_fock[ci.idx], 1:num_states_in_p_space)
    add_subspace!(ssi, (ref_fock[ci.idx][2], ref_fock[ci.idx][1]), 1:num_states_in_p_space) # add flipped spin
    push!(p_spaces, ssi)
end

ci_vector = BSTstate(clusters, p_spaces, cluster_bases, R=1) 

na = 4
nb = 4
FermiCG.fill_p_space!(ci_vector, na, nb)
FermiCG.eye!(ci_vector)
ebst, vbst = FermiCG.ci_solve(ci_vector, cluster_ops, clustered_ham)

ept2 = FermiCG.compute_pt2_energy(vbst, cluster_ops, clustered_ham, thresh_foi=1e-64)

println(" PT2 - tpsci")
display(ept1)
println(" PT2 - bst")
display(ept2)
