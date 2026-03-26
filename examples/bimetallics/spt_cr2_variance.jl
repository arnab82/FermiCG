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

# cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3], init_fspace, max_roots=M, verbose=1);
cluster_bases = FermiCG.compute_cluster_est_spinbasis(ints, clusters, d1.a, d1.b,
                    verbose=1, thresh_schmidt=5e-5, init_fspace=init_fspace, delta_elec=[3,3,3])

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
e2_bst = FermiCG.compute_pt2_energy(v_var, cluster_ops, clustered_ham, thresh_foi=1e-6);

t1=time()
alloc= @allocated σ2 = FermiCG.compute_spt_sigma_norm_blockwise(v_var, cluster_ops, clustered_ham;
                                       H0="Hcmf", nbody=4, thresh_foi=1e-8,
                                       max_number=nothing, opt_ref=true,
                                       ci_tol=1e-6, verbose=3)

var_like = σ2 .- e_var1.^2
for i in 1:length(e_var1)
    println("Variance-like, Energy: $(var_like[i]), $(e_var1[i])")
end
@printf("Allocated memory: %.3f GB\n", alloc/1e9)
t2 = time()
println("Time taken: $(t2-t1) seconds")
t1=time()
alloc= @allocated σ2 = FermiCG.compute_spt_sigma_norm_blockwise_fast(v_var, cluster_ops, clustered_ham;
                                       H0="Hcmf", nbody=4, thresh_foi=1e-8,
                                       max_number=nothing, opt_ref=true,
                                       ci_tol=1e-6, verbose=3)

var_like_ = σ2 .- e_var1.^2
for i in 1:length(e_var1)
    println("Variance-like, Energy: $(var_like_[i]), $(e_var1[i])")
end
@printf("Allocated memory: %.3f GB\n", alloc/1e9)
t2 = time()
println("Time taken: $(t2-t1) seconds")



thresh_spf=0.008
e_var2, v_var = FermiCG.block_sparse_tucker(v_var, cluster_ops, clustered_ham,
                                               max_iter    = 30,
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
@save "data_spt.jld2" e_var2 v_var cluster_bases
t1=time()
alloc= @allocated σ2 = FermiCG.compute_spt_sigma_norm_blockwise(v_var, cluster_ops, clustered_ham;
                                       H0="Hcmf", nbody=4, thresh_foi=1e-8,
                                       max_number=nothing, opt_ref=true,
                                       ci_tol=1e-6, verbose=3   )

var_like2 = σ2 .- e_var2.^2
for i in 1:length(e_var2)
    println("Variance-like, Energy: $(var_like2[i]), $(e_var2[i])")
end

@printf("Allocated memory: %.3f GB\n", alloc/1e9)
t2 = time()
println("Time taken: $(t2-t1) seconds")

t1=time()
alloc= @allocated σ2 = FermiCG.compute_spt_sigma_norm_blockwise_fast(v_var, cluster_ops, clustered_ham;
                                       H0="Hcmf", nbody=4, thresh_foi=1e-8,
                                       max_number=nothing, opt_ref=true,
                                       ci_tol=1e-6, verbose=3   )

var_like2_ = σ2 .- e_var2.^2
for i in 1:length(e_var2)
    println("Variance-like, Energy: $(var_like2_[i]), $(e_var2[i])")
end

@printf("Allocated memory: %.3f GB\n", alloc/1e9)
t2 = time()
println("Time taken: $(t2-t1) seconds")


thresh_spf=0.006
e_var3, v_var = FermiCG.block_sparse_tucker(v_var, cluster_ops, clustered_ham,
                                               max_iter    = 20 ,
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
@save "data_spt.jld2" e_var3 v_var cluster_bases
t1=time()
alloc= @allocated σ2 = FermiCG.compute_spt_sigma_norm_blockwise(v_var, cluster_ops, clustered_ham;
                                       H0="Hcmf", nbody=4, thresh_foi=1e-8,
                                       max_number=nothing, opt_ref=true,
                                       ci_tol=1e-6, verbose=3   )

var_like3 = σ2 .- e_var3.^2
for i in 1:length(e_var3)
    println("Variance-like, Energy: $(var_like3[i]), $(e_var3[i])")
end

@printf("Allocated memory: %.3f GB\n", alloc/1e9)
t2 = time()
println("Time taken: $(t2-t1) seconds")
t1=time()
alloc= @allocated σ2 = FermiCG.compute_spt_sigma_norm_blockwise_fast(v_var, cluster_ops, clustered_ham;
                                       H0="Hcmf", nbody=4, thresh_foi=1e-8,
                                       max_number=nothing, opt_ref=true,
                                       ci_tol=1e-6, verbose=3   )

var_like3_ = σ2 .- e_var3.^2
for i in 1:length(e_var3)
    println("Variance-like, Energy: $(var_like3_[i]), $(e_var3[i])")
end

@printf("Allocated memory: %.3f GB\n", alloc/1e9)
t2 = time()
println("Time taken: $(t2-t1) seconds")




println("Variance, Energy")
for i in 1:length(e_var1)
    println("Root,variance, Energy:  $(i),$(var_like[i]), $(e_var1[i])")
    println("Root,variance, Energy:  $(i),$(var_like2[i]), $(e_var2[i])")
    println("Root,variance, Energy:  $(i),$(var_like3[i]), $(e_var3[i])")
end
for i in 1:length(e_var1)
    println("Root,variance, Energy:  $(i),$(var_like_[i]), $(e_var1[i])")
    println("Root,variance, Energy:  $(i),$(var_like2_[i]), $(e_var2[i])")
    println("Root,variance, Energy:  $(i),$(var_like3_[i]), $(e_var3[i])")
end