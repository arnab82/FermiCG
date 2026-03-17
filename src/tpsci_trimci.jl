using Printf
using Random

"""
    trim_tpsci!(s::TPSCIstate{T,N,R}; thresh=1e-8) where {T,N,R}

Remove configurations from the variational space whose total squared coefficient
across all roots is below `thresh`:

    ∑_{r=1}^{R} |c_i^{(r)}|² < thresh

This is the **global trimming** step of TRIMCI: it keeps the variational space
compact by discarding configurations that contribute negligibly to all roots.
"""
function trim_tpsci!(s::TPSCIstate{T,N,R}; thresh=1e-8) where {T,N,R}
    for (fock, configs) in s.data
        to_delete = ClusterConfig{N}[]
        for (config, coeff) in configs
            weight = sum(c * c for c in coeff)
            if weight < thresh
                push!(to_delete, config)
            end
        end
        for config in to_delete
            delete!(configs, config)
        end
    end
    prune_empty_fock_spaces!(s)
end


"""
    _trimci_block_diag(vec, cluster_ops, clustered_ham, block_size, n_keep_per_block; rng)

**Stage-1 trimming** for TRIMCI: randomly partition `vec` into blocks of size
`block_size`, diagonalize each block's Hamiltonian, and keep the top
`n_keep_per_block` configurations ranked by their weight in the lowest
`min(R, block_size)` local eigenvectors.

Returns a new `TPSCIstate` containing only the surviving configurations
(with CI coefficients from `vec`).

# Algorithm
1. Collect all (fock, config) pairs and assign to random blocks.
2. For each block: build `H_block` via `build_full_H` and diagonalize.
3. For each config i in the block compute
       w_i = ∑_{r=1}^{min(R,B)} (v_i^{(r)})²
   where v^{(r)} is the r-th lowest-energy local eigenvector.
4. Retain the `n_keep_per_block` configs with highest w_i.
"""
function _trimci_block_diag(vec::TPSCIstate{T,N,R},
                            cluster_ops, clustered_ham,
                            block_size::Int, n_keep_per_block::Int;
                            rng=Random.default_rng()) where {T,N,R}

    # ---- Collect all (fock, config) pairs ----------------------------------
    all_focks   = FockConfig{N}[]
    all_configs = ClusterConfig{N}[]

    for (fock, configs) in vec.data
        for (config, _) in configs
            push!(all_focks,   fock)
            push!(all_configs, config)
        end
    end

    n_total = length(all_focks)
    n_total > 0 || return deepcopy(vec)

    # ---- Random permutation -------------------------------------------------
    perm  = randperm(rng, n_total)
    keep  = falses(n_total)

    n_blocks = cld(n_total, block_size)   # ceil(n_total / block_size)

    for b in 1:n_blocks
        i_start   = (b - 1) * block_size + 1
        i_end     = min(b * block_size, n_total)
        block_idx = perm[i_start:i_end]   # indices into all_focks / all_configs
        B         = length(block_idx)

        # ---- Build temporary TPSCIstate for this block ----------------------
        # Coefficients are only needed to define the basis for build_full_H
        block_state = TPSCIstate(vec.clusters, T=T, R=R)
        for i in block_idx
            fock_i   = all_focks[i]
            config_i = all_configs[i]
            if !haskey(block_state.data, fock_i)
                add_fockconfig!(block_state, fock_i)
            end
            block_state[fock_i][config_i] = zeros(T, R)
        end

        # ---- Diagonalize block Hamiltonian ----------------------------------
        H_block = build_full_H(block_state, cluster_ops, clustered_ham)
        n_eig   = min(R, B)
        F       = eigen(Symmetric(H_block))

        # Weight of config i = sum of squares in the lowest n_eig eigenvectors
        local_weight = zeros(T, B)
        for r in 1:n_eig
            local_weight .+= F.vectors[:, r] .^ 2
        end

        # Keep top n_keep_per_block configs by local weight
        n_keep = min(n_keep_per_block, B)
        sorted = sortperm(local_weight, rev=true)
        for k in 1:n_keep
            keep[block_idx[sorted[k]]] = true
        end
    end

    # ---- Build output: deepcopy of vec minus the discarded configs ----------
    out = deepcopy(vec)
    for i in 1:n_total
        keep[i] && continue
        fock_i   = all_focks[i]
        config_i = all_configs[i]
        if haskey(out.data, fock_i) && haskey(out[fock_i], config_i)
            delete!(out[fock_i], config_i)
        end
    end
    prune_empty_fock_spaces!(out)
    return out
end


"""
    tpsci_trimci(ci_vector, cluster_ops, clustered_ham; kwargs...)

Run **TRIMCI** (TRIMmed Configuration Interaction) in the Tensor Product State
(TPS) basis.

TRIMCI performs graph-based expansion from the current variational core,
followed by random block diagonalization trimming (Stage 1) and a global CI
diagonalization (Stage 2). An optional global trim then removes
low-weight configurations before the next iteration.

See: https://arxiv.org/abs/2511.14734 and https://arxiv.org/pdf/2303.02232

## Algorithm per iteration

1. **Expansion**: compute the first-order interaction space FOIS = H|Ψ⟩;
   add configurations x where the accumulated coupling

       score_x = ∑_r |⟨x|H|Ψ^{(r)}⟩| > thresh_cipsi

   This is the TPS-basis analogue of the TRIMCI graph-connectivity expansion.
   Each external config x is a neighbor of the current core in the Hamiltonian
   graph, and score_x ≈ ∑_i |H_{xi} c_i| accumulated over the core.

2. **Block trim (Stage 1)**: randomly partition the expanded core into blocks
   of `block_size`; diagonalize each block; keep the top `n_keep_per_block`
   configs by local eigenvector weight. Skipped in the first iteration
   (no CI coefficients available) and when `block_size >= length(vec_var)`.

3. **Global CI (Stage 2)**: diagonalize the full Hamiltonian in the surviving
   core to get the updated variational wavefunction Ψ and energy E.

4. **Global trim**: remove configurations with ∑_r |c_i^{(r)}|² < `thresh_trim`.

5. **Convergence**: stop when max_r |E^{(t)}_r - E^{(t-1)}_r| < `conv_thresh`.

## Arguments
- `thresh_cipsi`     : expansion threshold on ∑_r|⟨x|H|Ψ^{(r)}⟩| (default 1e-4)
- `thresh_foi`       : threshold for H|Ψ⟩ vector entries (default 1e-6)
- `thresh_trim`      : global trim: remove i if ∑_r|c_i^{(r)}|² < thresh
                       (default 1e-8; set to `nothing` or 0 to disable)
- `block_size`       : configs per random block in Stage-1 trim
                       (default 200; set > dim to skip block trim)
- `n_keep_per_block` : configs retained per block (default `block_size ÷ 2`)
- `thresh_asci`      : clip the reference vector for sigma build (default nothing)
- `thresh_var`       : additional clip of variational vector (default nothing)
- `thresh_spin`      : threshold for S² residual extension (default nothing)
- `max_iter`         : maximum TRIMCI iterations (default 10)
- `conv_thresh`      : energy convergence threshold (default 1e-4)
- `nbody`            : maximum n-body terms (default 4)
- `ci_conv`          : inner CI convergence (default 1e-5)
- `ci_max_iter`      : max iterations for inner CI (default 50)
- `ci_max_ss_vecs`   : max subspace size for inner CI (default 12)
- `ci_lindep_thresh` : linear-dependence threshold for inner CI (default 1e-12)
- `davidson`         : force Davidson solver (default false; auto if mem > max_mem_ci)
- `max_mem_ci`       : memory limit (Gb) for dense H matrix (default 20.0)
- `threaded`         : use multithreading for H|Ψ⟩ (default true)
- `rng`              : RNG for block partitioning (default `Random.default_rng()`)

## Returns
- `e0::Vector{T}`              : variational energies for each root
- `vec_var::TPSCIstate{T,N,R}` : converged TRIMCI wavefunction
"""
function tpsci_trimci(ci_vector::TPSCIstate{T,N,R}, cluster_ops,
                      clustered_ham::ClusteredOperator;
    thresh_cipsi     = 1e-4,
    thresh_foi       = 1e-6,
    thresh_trim      = 1e-8,
    block_size       = 200,
    n_keep_per_block = nothing,
    thresh_asci      = nothing,
    thresh_var       = nothing,
    thresh_spin      = nothing,
    max_iter         = 10,
    conv_thresh      = 1e-4,
    nbody            = 4,
    ci_conv          = 1e-5,
    ci_max_iter      = 50,
    ci_max_ss_vecs   = 12,
    ci_lindep_thresh = 1e-12,
    davidson         = false,
    max_mem_ci       = 20.0,
    threaded         = true,
    rng              = Random.default_rng()) where {T,N,R}

    # Default: keep half the block on each block-trim pass
    n_keep_block = (n_keep_per_block !== nothing) ? n_keep_per_block : max(1, block_size ÷ 2)

    vec_var = deepcopy(ci_vector)
    vec_pt  = deepcopy(ci_vector)
    length(ci_vector) > 0 || error(" input vector has zero length")
    zero!(vec_pt)
    e0      = zeros(T, R)
    e0_last = zeros(T, R)

    clustered_S2    = extract_S2(ci_vector.clusters)
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = "Hcmf")

    println(" ci_vector        : ", size(ci_vector))
    println(" thresh_cipsi     : ", thresh_cipsi)
    println(" thresh_foi       : ", thresh_foi)
    println(" thresh_trim      : ", thresh_trim)
    println(" block_size       : ", block_size)
    println(" n_keep_per_block : ", n_keep_block)
    println(" thresh_asci      : ", thresh_asci)
    println(" thresh_var       : ", thresh_var)
    println(" thresh_spin      : ", thresh_spin)
    println(" max_iter         : ", max_iter)
    println(" conv_thresh      : ", conv_thresh)
    println(" nbody            : ", nbody)
    println(" ci_conv          : ", ci_conv)
    println(" ci_max_iter      : ", ci_max_iter)
    println(" ci_max_ss_vecs   : ", ci_max_ss_vecs)
    println(" ci_lindep_thresh : ", ci_lindep_thresh)
    println(" davidson         : ", davidson)
    println(" max_mem_ci       : ", max_mem_ci)
    println(" threaded         : ", threaded)

    # H and vec_var_old track the stored dense Hamiltonian for incremental updates.
    # need_full_H is set true whenever the space changes (trim or block-trim).
    H           = zeros(T, size(ci_vector))
    vec_var_old = deepcopy(ci_vector)
    need_full_H = true

    to = TimerOutput()

    for it in 1:max_iter

        println()
        println()
        println(" ===================================================================")
        @printf("     TRIMCI Iteration: %4i  thresh_cipsi: %12.8f\n", it, thresh_cipsi)
        println(" ===================================================================")

        # ----------------------------------------------------------------
        # [Expand] Add configurations selected in the previous iteration
        # ----------------------------------------------------------------
        if it > 1
            if thresh_var !== nothing
                l1 = length(vec_var)
                clip!(vec_var, thresh=thresh_var)
                l2 = length(vec_var)
                @printf(" Clip values < %8.1e         %6i → %6i\n", thresh_var, l1, l2)
            end

            vec_var_old = deepcopy(vec_var)

            l1 = length(vec_var)
            zero!(vec_pt)           # zero coefficients so add! only adds new config keys
            add!(vec_var, vec_pt)   # merge new config keys into variational space
            l2 = length(vec_var)
            @printf("%-50s%6i → %6i\n", " Add selected configs to current space", l1, l2)
        end

        # ----------------------------------------------------------------
        # Optional S² extension
        # ----------------------------------------------------------------
        @timeit to "s2 extension" if thresh_spin !== nothing
            spin_residual = if threaded
                open_matvec_thread(vec_var, cluster_ops, clustered_S2,
                                   nbody=nbody, thresh=thresh_spin)
            else
                open_matvec_serial(vec_var, cluster_ops, clustered_S2,
                                   nbody=nbody, thresh=thresh_spin)
            end
            spin_expval   = overlap(vec_var, spin_residual)
            spin_residual = spin_residual - (vec_var * spin_expval)
            for r in 1:R
                @printf(" S^2 Residual %12.8f\n", dot(spin_residual, spin_residual, r, r))
            end
            zero!(spin_residual)
            l1 = length(vec_var)
            add!(vec_var, spin_residual)
            l2 = length(vec_var)
            @printf("%-50s%6i → %6i\n", " Add spin completing states", l1, l2)
            flush(stdout)
        end

        # ----------------------------------------------------------------
        # [Stage 1: Block trim]
        # Randomly partition the expanded core into blocks; diagonalize each
        # block and keep the top n_keep_per_block configs by local weight.
        # Skipped on the first iteration (no meaningful CI coefficients yet)
        # and when block_size >= current space dimension (nothing to trim).
        # ----------------------------------------------------------------
        if it > 1 && block_size < length(vec_var)
            l1 = length(vec_var)
            @timeit to "block trim" begin
                vec_var = _trimci_block_diag(vec_var, cluster_ops, clustered_ham,
                                             block_size, n_keep_block; rng=rng)
            end
            l2 = length(vec_var)
            @printf("%-50s%6i → %6i\n", " TRIMCI block trim (stage 1)", l1, l2)
            if l2 < l1
                need_full_H = true
                vec_var_old = deepcopy(vec_var)
            end
            flush(stdout)
        end

        # ----------------------------------------------------------------
        # [Stage 2: Global CI] diagonalize on surviving configs
        # ----------------------------------------------------------------
        e0 = nothing
        mem_needed = sizeof(T) * length(vec_var) * length(vec_var) * 1e-9
        @printf(" Memory needed to hold full CI matrix: %12.8f (Gb) Max allowed: %12.8f (Gb)\n",
                mem_needed, max_mem_ci)
        flush(stdout)

        @timeit to "ci" begin
            if (mem_needed > max_mem_ci) || davidson == true
                orthonormalize!(vec_var)
                e0, vec_var = tps_ci_davidson(vec_var, cluster_ops, clustered_ham,
                                              conv_thresh   = ci_conv,
                                              max_iter      = ci_max_iter,
                                              max_ss_vecs   = ci_max_ss_vecs,
                                              lindep_thresh = ci_lindep_thresh)
                need_full_H = true   # Davidson does not return H; force rebuild next iteration
            else
                if it > 1 && !need_full_H
                    # Reuse stored H and update incrementally
                    e0, vec_var, H = tps_ci_direct(vec_var, cluster_ops, clustered_ham,
                                                   H_old         = H,
                                                   v_old         = vec_var_old,
                                                   conv_thresh   = ci_conv,
                                                   max_ss_vecs   = ci_max_ss_vecs,
                                                   max_iter      = ci_max_iter,
                                                   lindep_thresh = ci_lindep_thresh)
                else
                    e0, vec_var, H = tps_ci_direct(vec_var, cluster_ops, clustered_ham,
                                                   conv_thresh   = ci_conv,
                                                   max_ss_vecs   = ci_max_ss_vecs,
                                                   max_iter      = ci_max_iter,
                                                   lindep_thresh = ci_lindep_thresh)
                    need_full_H = false
                end
            end
        end
        flush(stdout)

        # ----------------------------------------------------------------
        # [Global trim] Remove configs with ∑_r |c_i^{(r)}|² < thresh_trim
        # ----------------------------------------------------------------
        if thresh_trim !== nothing && thresh_trim > 0
            l1 = length(vec_var)
            trim_tpsci!(vec_var, thresh=thresh_trim)
            l2 = length(vec_var)
            @printf("%-50s%6i → %6i\n", " TRIMCI global trim", l1, l2)
            if l2 < l1
                need_full_H = true
                vec_var_old = deepcopy(vec_var)
            end
            flush(stdout)
        end

        # ----------------------------------------------------------------
        # Barycentric (zeroth-order) energy <Ψ|H₀|Ψ>
        # ----------------------------------------------------------------
        Efock = compute_expectation_value_parallel(vec_var, cluster_ops, clustered_ham_0)
        flush(stdout)

        # ----------------------------------------------------------------
        # Optional ASCI clipping of reference vector for sigma build
        # ----------------------------------------------------------------
        vec_asci = deepcopy(vec_var)
        if thresh_asci !== nothing
            l1 = length(vec_asci)
            clip!(vec_asci, thresh=thresh_asci)
            l2 = length(vec_asci)
            @printf("%-50s%6i → %6i\n", " Length of ASCI vector", l1, l2)
        end

        # ----------------------------------------------------------------
        # Compute FOIS: sig = H|Ψ> projected outside the current core
        # ----------------------------------------------------------------
        @timeit to "matvec" begin
            sig = if threaded
                open_matvec_thread(vec_asci, cluster_ops, clustered_ham,
                                   nbody=nbody, thresh=thresh_foi)
            else
                open_matvec_serial(vec_asci, cluster_ops, clustered_ham,
                                   nbody=nbody, thresh=thresh_foi)
            end
        end
        project_out!(sig, vec_asci)
        println(" Length of FOIS vector: ", length(sig))

        # Compute H0 diagonal for PT2 correction (informational output)
        @printf(" %-50s", "Compute diagonal: ")
        flush(stdout)
        @timeit to "diagonal" @time Hd = compute_diagonal(sig, cluster_ops, "Hcmf")
        println()
        flush(stdout)

        sig_v = get_vector(sig)   # n_fois x R  matrix of <x|H|Ψ^(r)>
        norms = norm(vec_asci)

        # Print PT2 energy correction (informational only; not used for selection)
        println()
        @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)")
        for r in 1:R
            denom = Efock[r] / (norms[r] * norms[r]) .- Hd
            e2_r  = sum(sig_v[:, r] .^ 2 ./ denom)
            @printf(" %5i %12.8f %12.8f\n", r,
                    e0[r] / (norms[r] * norms[r]),
                    e0[r] / (norms[r] * norms[r]) + e2_r)
        end

        # ----------------------------------------------------------------
        # TRIMCI expansion criterion:
        #   score_x = sum_r |<x|H|Ψ^(r)>|  (accumulated graph coupling)
        #
        # Retain configuration x if score_x > thresh_cipsi.
        # This is the TPS-basis analogue of scanning H rows for neighbors
        # with large |H_ij c_i|: here we use the FOIS amplitude which
        # aggregates all such couplings into a single score per external config.
        # ----------------------------------------------------------------
        expansion_score = dropdims(sum(abs.(sig_v), dims=2), dims=2)   # length n_fois

        # Store scores in vec_pt. All roots receive the same value so that
        # clip! (which checks any-root) correctly selects by the total score.
        vec_pt = deepcopy(sig)
        v_select = repeat(expansion_score, 1, R)
        set_vector!(vec_pt, Matrix{T}(v_select))

        l1 = length(vec_pt)
        clip!(vec_pt, thresh=thresh_cipsi)
        l2 = length(vec_pt)
        @printf("%-50s%6i → %6i\n", " Length of TRIMCI selected vector", l1, l2)

        # ----------------------------------------------------------------
        # Convergence check
        # ----------------------------------------------------------------
        converged = maximum(abs.(e0_last .- e0)) < conv_thresh
        print_tpsci_iter(vec_var, it, e0, converged)
        converged && break
        e0_last .= e0
        flush(stdout)
    end

    println("")
    show(to)
    println("")
    flush(stdout)
    return e0, vec_var
end
