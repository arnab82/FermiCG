using FermiCG
using LinearAlgebra
using Printf
using NPZ
using SparseArrays
"""
    compare_hqq_builders(ref, cluster_ops, clustered_ham; thresh_foi, nbody, verbose)

Build the CEPA Q-space from `ref` at `thresh_foi`, then construct H_qq with both
`build_H_qq` (dense) and `build_H_qq_sparse` (sparse) and compare them.
Prints a report and returns `(H_dense, H_sparse)`.
Used to verify the sparse builder before using it in production.
"""
function compare_hqq_builders(ref::TPSCIstate{T,N,R}, cluster_ops, clustered_ham;
                                thresh_foi=1e-3, nbody=4, verbose=1) where {T,N,R}
#={{{=#
    ref_vec = deepcopy(ref)
    e0, ref_vec = FermiCG.tps_ci_direct(ref_vec, cluster_ops, clustered_ham, conv_thresh=1e-8)

    pt1_vec = deepcopy(ref_vec)
    pt1_vec = FermiCG.open_matvec_thread(pt1_vec, cluster_ops, clustered_ham,
                                 nbody=nbody, thresh=thresh_foi)
    FermiCG.project_out!(pt1_vec, ref)

    dim_q = length(pt1_vec)
    @printf(" FOIS dim_q = %i\n", dim_q)

    q1 = TPSCIstate(pt1_vec, R=1)

    verbose > 0 && println(" Building H_qq dense  ...")
    GC.gc(); td = @timed FermiCG.build_H_qq(q1, cluster_ops, clustered_ham)
    H_dense = td.value

    verbose > 0 && println(" Building H_qq sparse ...")
    GC.gc(); ts = @timed FermiCG.build_H_qq_sparse(q1, cluster_ops, clustered_ham)
    H_sparse = ts.value

    diff_norm = LinearAlgebra.norm(H_dense - Matrix(H_sparse))
    sym_err   = LinearAlgebra.norm(H_sparse - H_sparse')
    fill_pct  = 100.0 * nnz(H_sparse) / dim_q^2

    println()
    println("─"^70)
    println(" H_qq builder comparison")
    println("─"^70)
    @printf(" dim_q                   = %i\n",    dim_q)
    @printf(" Dense  alloc / time     = %.2f MiB / %.2f s\n", td.bytes/2^20, td.time)
    @printf(" Sparse alloc / time     = %.2f MiB / %.2f s\n", ts.bytes/2^20, ts.time)
    @printf(" nnz(sparse)             = %i  (%.3f%% fill)\n", nnz(H_sparse), fill_pct)
    @printf(" Dense  stored mem       = %.2f MiB\n", sizeof(H_dense)/2^20)
    @printf(" Sparse stored mem (CSC) = %.2f MiB\n",
            (nnz(H_sparse)*8 + (dim_q+1)*8)/2^20)
    @printf(" norm(dense - sparse)    = %.2e   ← correctness\n", diff_norm)
    @printf(" norm(sparse - sparse')  = %.2e   ← symmetry\n",   sym_err)
    println("─"^70)

    return H_dense, H_sparse
end
#=}}}=#