# BST
The Block-Sparse Tucker (BST) method approximates FCI as a linear combination of individually compressed (via HOSVD)
blocks of the Hilbert space.

## Background
Similar to TPSCI, the approach here starts with a CMF wavefunction, and systematically reintroduces the discarded tensor product
states to variationally approach FCI. Unlike with TPSCI, however, we don't assume that the final wavefunction is written as 
a purely sparse form (where only a few TPS's are needed), but rather we assume that collections of TPS's where certain numbers of clusters are "excited" can be efficiently compressed via HOSVD (although the basic idea would extend to other tensor decompositions, like CP or MPS). 

The key insight is that while the full tensor product state (TPS) basis may be prohibitively large, blocks of this space 
(corresponding to specific Fock configurations and cluster subspaces) often exhibit low-rank structure. BST exploits this 
structure by representing each block as a compressed Tucker tensor, reducing storage and computational costs while 
maintaining accuracy.

## Implementation Details

Our current implementation is written in Julia, taking advantage of its high-performance numerical computing capabilities while maintaining code readability. The implementation uses several Julia packages including `LinearAlgebra` for basic matrix operations, `TensorOperations.jl` for efficient tensor contractions, and `BlockDavidson.jl` for iterative eigensolvers. Integration with PySCF is handled through the `pyscf` Python module for computing one- and two-electron integrals from RHF calculations. While our Julia implementation already demonstrates good performance, there remain several opportunities for optimization, particularly in the tensor contraction routines and memory management strategies.

One of the most critical aspects of this implementation is the data structure used to store the compressed wavefunction. Unlike TPSCI, which uses a sparse list of individual tensor product states, BST requires storing multiple compressed tensor blocks, each potentially containing thousands of variational parameters. We address this through a nested hash table structure implemented using Julia's `OrderedDict`. The use of `OrderedDict` rather than standard `Dict` ensures deterministic iteration order, which is crucial for reproducibility and debugging, though it comes with a modest performance penalty.

The wavefunction storage scheme uses a doubly-nested hash table structure. By specifying a Fock space configuration over $N$ clusters as an immutable tuple-of-tuples $((N_1^\uparrow, N_1^\downarrow), (N_2^\uparrow, N_2^\downarrow), \ldots, (N_N^\uparrow, N_N^\downarrow))$, all tensor blocks with the same distribution of particle numbers can be organized together. Within each Fock sector, we further organize the data by Tucker configuration, which specifies the subspace ranges for each cluster as a tuple of `UnitRange` objects: $(r_1, r_2, \ldots, r_N)$ where each $r_i$ is a range like `1:5`. Consequently, an arbitrary Tucker-compressed tensor block can be accessed by two sequential hash table lookups:

```julia
TBL2 = TBL1[((N₁↑, N₁↓), (N₂↑, N₂↓), ..., (Nₙ↑, Nₙ↓))]               # (1)
Tucker_block = TBL2[(range₁, range₂, ..., rangeₙ)]                   # (2)
```

Each `Tucker_block` stores the compressed tensor representation using the Tucker decomposition format. This consists of a small core tensor $\mathcal{C}$ and orthogonal factor matrices $U^{(i)}$ for each mode, stored as:

```julia
struct Tucker{T,N,R}
    core::NTuple{R, Array{T,N}}      # R core tensors (one per root)
    factors::NTuple{N, Matrix{T}}    # N factor matrices (shared across roots)
end
```

The full tensor for root $r$ can be reconstructed (though this is rarely necessary in practice) as $\mathcal{T}^{(r)} = \mathcal{C}^{(r)} \times_1 U^{(1)} \times_2 U^{(2)} \cdots \times_N U^{(N)}$. The use of immutable `NTuple` types rather than mutable `Vector` types provides several advantages: it enables compile-time optimizations by the Julia JIT compiler, ensures type stability for better performance, and allows thread-safe read access without locks. For multi-root calculations, all roots share the same Tucker factor matrices but have separate core tensors, which significantly reduces memory usage and enables efficient state-averaged optimizations.

The compression operation is central to BST's efficiency. Each Tucker block is compressed using the Higher-Order Singular Value Decomposition (HOSVD), which we implement by performing sequential SVDs along each mode of the tensor. For each mode $i$, we reshape the $N$-dimensional tensor into a matrix via mode-$i$ matricization, compute the SVD, and retain only singular vectors with singular values $\sigma > \epsilon_{\text{thresh}}$. For multi-root calculations, we compute a joint compression basis by forming the Gram matrix $\mathbf{G} = \sum_{r=1}^R \mathbf{T}_{(i)}^{(r)} {\mathbf{T}_{(i)}^{(r)}}^T$ and diagonalizing it to find eigenvectors common to all roots. This shared basis is essential for state-averaged calculations and significantly reduces memory requirements compared to storing separate Tucker factors for each root.

The core tensor projection, transforming $\mathcal{C}_{\text{old}}$ into the compressed basis to yield $\mathcal{C}_{\text{new}}$, is performed using tensor contractions from `TensorOperations.jl`, which leverages BLAS routines for optimal performance. A critical implementation detail is that we apply compression iteratively via the `compress_iteratively` function, which repeatedly applies HOSVD until the dimension stops decreasing. This is necessary because Tucker compression is not nested—compressing an already-compressed tensor can further reduce its dimension as cross-mode correlations become more apparent. Typically 3-5 compression iterations are sufficient to reach convergence.

Memory management presents significant challenges in BST calculations. The memory bottleneck comes from several sources: the Tucker core tensors, which scale as $O(R \prod_{i=1}^N r_i)$ where $r_i$ are the compressed dimensions; the Tucker factor matrices, which scale as $O(\sum_{i=1}^N d_i r_i)$ where $d_i$ are the full dimensions; and most critically, the operator tensors stored for each cluster. Each cluster must store transition density tensors $\Gamma^{IJ}_{pqr}$ for all operator strings needed by the Hamiltonian, which scale as $O(N_{\text{orb}}^3 M M')$ where $M$ and $M'$ are the number of states in different Fock sectors. For clusters with 6 or more orbitals and large basis sets, these operator tensors can consume tens of gigabytes of memory per cluster.

To mitigate memory demands, we employ several strategies. First, we exploit local symmetries (particle number and spin projection) to store only symmetry-allowed transitions. The operator tensors are organized in hash tables indexed by Fock space transitions, avoiding storage of hard zeros. Second, for two-electron Hamiltonian terms, which would nominally require storing six-index tensors $\Gamma^{IJ}_{pqrs}$, we precontract these with the two-electron integrals to form effective local Hamiltonian matrices. This reduces the memory bottleneck to five-index tensors $\Gamma^{IJ}_{pqr}$, but still limits us to clusters of at most 6-7 orbitals in practice. Third, we provide operator caching functionality that pre-computes dense Hamiltonian blocks for frequently accessed operator combinations:

```julia
function cache_hamiltonian(bra::BSTstate, ket::BSTstate, cluster_ops, H)
    for each Fock sector transition:
        for each Tucker config pair:
            if symmetry allowed and not cached:
                H_dense = build_dense_H_term(...)
                cache[operator_key] = H_dense
end
```

This trades memory for computational time by storing pre-computed operator blocks, which can dramatically accelerate the repeated sigma builds required during CI diagonalization.

The sigma build operation, computing $|\sigma\rangle = \hat{H}|v\rangle$, is the most computationally intensive step in BST. Our implementation avoids explicit reconstruction of full tensors whenever possible. The algorithm iterates over all Fock sector pairs, checks if the transition is present in the Hamiltonian, and then for each Tucker configuration pair that is connected, computes the contribution via tensor contractions. The key optimization is that most tensor contractions occur directly in the compressed Tucker space. For example, a term like $\langle \text{bra}| \hat{p}^\dagger \hat{q} \hat{r} |\text{ket}\rangle$ is computed as:

```julia
# Contract operator tensor with ket Tucker factors and core
result_core = contract_with_operators(operator_tensor, ket.core, ket.factors)
# Then with bra Tucker factors  
sigma.core += contract_with_bra_factors(result_core, bra.factors)
```

This approach avoids the factorial scaling that would come from reconstructing full tensors. However, for some operator terms, particularly those involving many-body operators acting on multiple clusters simultaneously, we do need to work with partially reconstructed tensors, which becomes a performance bottleneck.

One aspect that affects both memory and computational performance is the handling of non-orthogonal state additions. When expanding the variational space by adding the PT1 wavefunction to the reference, the two states generally have different Tucker factor matrices. Our `nonorth_add` function handles this by constructing a union basis for each mode—essentially performing a QR decomposition on the concatenation of the two factor matrices—and then transforming both core tensors into this common basis. This operation can be expensive for large Tucker blocks, but it is essential for maintaining the Tucker format throughout the calculation. We have found that following each non-orthogonal addition with compression helps control the growth of the Tucker dimensions.

The orthonormalization of multi-root states also deserves mention. After various operations like state additions and compressions, the different roots can lose orthogonality. We restore orthonormality by extracting all core tensors into a coefficient matrix, performing Gram-Schmidt orthogonalization directly on this matrix, and then writing the orthonormalized coefficients back into the Tucker cores. This extraction and insertion is facilitated by `get_vector` and `set_vector!` functions, which flatten the nested Tucker structure into a simple matrix representation suitable for linear algebra operations.

Parallelization is naturally accommodated at several levels in our implementation. The most expensive operations—HOSVD compression, sigma builds, and CI diagonalization—are parallelized using Julia's built-in threading capabilities. The sigma build can be parallelized over Fock sector transitions, since these are independent. The HOSVD compression for different Tucker blocks can also proceed in parallel. Currently, we use shared-memory parallelism via `Threads.@threads`, and we have observed good scaling on systems with 16-32 cores. The tensor contractions themselves can also benefit from threaded BLAS operations. We have not yet implemented distributed-memory parallelism, but the algorithm structure would readily accommodate it, particularly for very large calculations where different Fock sectors or Tucker blocks could be distributed across compute nodes.

Looking forward, there are several areas where the implementation could be improved. The current handling of the first-order wavefunction can create memory pressure when the FOIS becomes very large. While we compress immediately after building the FOIS, for some systems this is insufficient, and we must prune small-amplitude configurations before storing them in memory. A more sophisticated approach, analogous to the deterministic or semistochastic PT2 corrections used in determinant-based methods, would avoid storing the full PT1 wavefunction altogether. Additionally, the operator tensor storage could be optimized by exploiting sparsity more aggressively or by using compressed storage formats. Finally, while Julia provides excellent performance for numerical computing, certain hot-path tensor contractions might benefit from hand-tuned kernels or GPU acceleration, particularly for the largest Tucker blocks.

## Performance considerations
- **`thresh_var`**: Controls compression of converged variational state. Smaller values preserve more accuracy but increase the dimension and computational cost. Typical values range from $10^{-3}$ to $10^{-5}$.
- **`thresh_foi`**: Threshold for compressed first-order interacting space. This parameter critically balances capturing important interactions while controlling the growth of the variational space. Too loose leads to missing important configurations; too tight causes memory issues. Typical values: $10^{-4}$ to $10^{-6}$.
- **`thresh_pt`**: Compression of PT1 wavefunction and expanded variational space. This is perhaps the most critical parameter for controlling memory usage, as the expanded space after adding PT1 can be very large. Values of $10^{-4}$ to $10^{-5}$ are typical.
- **`nbody`**: Include up to `nbody`-body terms in Hamiltonian. Higher values increase accuracy but significantly increase computational cost. Most calculations use `nbody=4`.
- **`max_iter`**: Maximum Tucker optimization cycles. BST typically converges in 5-20 iterations depending on system complexity and threshold choices.
- **`do_pt`**: Computing the PT1 wavefunction generally improves convergence and final accuracy but adds considerable computational cost. For very large calculations, this can be disabled to reduce memory and time requirements.
- **`resolve_ss`**: Re-diagonalizing in the compressed P-space at each iteration improves accuracy but adds cost. Generally recommended for high-accuracy calculations. 

## Index
```@index
Pages   = ["BST.md"]
```

## Documentation 
```@autodocs
Modules = [FermiCG]
Pages   = ["tucker_inner.jl","tucker_outer.jl","bst.jl"]
Order   = [:type, :function]
Depth	= 2
```

## HOSVD
```@autodocs
Modules = [FermiCG]
Pages   = ["hosvd.jl"]
Order   = [:type, :function]
Depth	= 2
```
