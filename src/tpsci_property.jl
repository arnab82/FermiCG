"""
tpsci_property.jl

One-electron property computation for TPSCI wavefunctions.

The central quantity is the transition 1-RDM:

    γ_σσ'[p,q,r1,r2] = <Ψ_{r1}|p'_{p,σ} q_{q,σ'}|Ψ_{r2}>

Two contributions are assembled:

  1. On-cluster  (p,q on the same cluster i):
        uses "Aa" / "Bb" transition density matrices already in cluster_ops,
        contracted with the cluster reduced density matrix
        ρ_i[s,t,r1,r2] = Σ_{matching configs} C_bra[r1] * C_ket[r2]

  2. Inter-cluster (p on cluster i, q on cluster j, i≠j):
        uses "A"[p,s,t] × "a"[q,s,t] (alpha-alpha) and
            "B"[p,s,t] × "b"[q,s,t] (beta-beta)
        with the appropriate fermionic sign.

Once γ is in hand, any one-electron property is just Tr(prop_ints * γ).
"""


# ---------------------------------------------------------------------------
# Internal helper: fermionic sign for p'_i q_j acting on a ket state
# ---------------------------------------------------------------------------
"""
    _rdm_sign(fock_ket::FockConfig{N}, ci_idx::Integer, cj_idx::Integer) where N

Return the fermionic sign arising from moving p'_{ci} and q_{cj} through the
string of cluster occupation operators.  Follows the same convention as
`compute_terms_state_sign`: each odd-parity operator contributes a sign
(-1)^(total electrons in ket before that cluster's position).
"""
function _rdm_sign(fock_ket::FockConfig{N}, ci_idx::Integer, cj_idx::Integer) where N
    state_sign = 1
    n_before_ci = 0
    for k in 1:ci_idx-1
        n_before_ci += fock_ket[k][1] + fock_ket[k][2]
    end
    n_before_ci % 2 != 0 && (state_sign = -state_sign)
    n_before_cj = 0
    for k in 1:cj_idx-1
        n_before_cj += fock_ket[k][1] + fock_ket[k][2]
    end
    n_before_cj % 2 != 0 && (state_sign = -state_sign)
    return state_sign
end


# ---------------------------------------------------------------------------
# Main 1-RDM function
# ---------------------------------------------------------------------------
"""
    compute_1rdm(bra::TPSCIstate{T,N,R1}, ket::TPSCIstate{T,N,R2}, cluster_ops)

Compute transition 1-RDM

    γ_aa[p,q,r1,r2] = <bra_{r1}|p'_{p,α} q_{q,α}|ket_{r2}>
    γ_bb[p,q,r1,r2] = <bra_{r1}|p'_{p,β} q_{q,β}|ket_{r2}>

Both tensors have shape (norb, norb, R1, R2).

For spin-free (singlet) one-electron properties use γ_total = γ_aa + γ_bb.
"""
function compute_1rdm(bra::TPSCIstate{T,N,R1},
                      ket::TPSCIstate{T,N,R2},
                      cluster_ops::Vector{ClusterOps{T}}) where {T,N,R1,R2}

    clusters  = bra.clusters
    norb      = sum(length(c) for c in clusters)

    # Global orbital offset for each cluster (0-based start)
    orb_offsets = zeros(Int, N)
    for i in 2:N
        orb_offsets[i] = orb_offsets[i-1] + length(clusters[i-1])
    end

    γ_aa = zeros(T, norb, norb, R1, R2)
    γ_bb = zeros(T, norb, norb, R1, R2)

    # ===========================================================
    # 1. ON-CLUSTER CONTRIBUTION
    # ===========================================================
    # For each fock sector f shared by bra and ket, and each cluster i:
    #
    #   γ_aa[p,q,r1,r2] += Σ_{s,t} ρ_i[(f,f)][s,t,r1,r2] * Aa_i[(f,f)][p,q,s,t]
    #
    # where ρ_i[s,t,r1,r2] = Σ_{configs matching at k≠i} C_bra[...s...][r1]*C_ket[...t...][r2]

    for (fock, configs_bra) in bra.data
        haskey(ket.data, fock) || continue
        configs_ket = ket.data[fock]

        for ci in clusters
            ci_idx = ci.idx
            norb_i = length(ci)
            off_i  = orb_offsets[ci_idx]
            ftrans = (fock[ci_idx], fock[ci_idx])

            haskey(cluster_ops[ci_idx], "Aa") || continue
            haskey(cluster_ops[ci_idx]["Aa"], ftrans) || continue
            haskey(cluster_ops[ci_idx], "Bb") || continue
            haskey(cluster_ops[ci_idx]["Bb"], ftrans) || continue

            Aa_i = cluster_ops[ci_idx]["Aa"][ftrans]  # [norb_i^2, n_s, n_t]  (after reshape)
            Bb_i = cluster_ops[ci_idx]["Bb"][ftrans]
            n_s  = size(Aa_i, 2)
            n_t  = size(Aa_i, 3)

            # ---- Build cluster RDM ρ_i ----
            ρ = zeros(T, n_s, n_t, R1, R2)

            # Group configs by their values at all clusters k ≠ ci_idx
            bra_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R1,T}}}}()
            for (config, coeff) in configs_bra
                key = [config[k] for k in 1:N if k != ci_idx]
                push!(get!(bra_groups, key, Pair{Int16,MVector{R1,T}}[]),
                      config[ci_idx] => coeff)
            end

            ket_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R2,T}}}}()
            for (config, coeff) in configs_ket
                key = [config[k] for k in 1:N if k != ci_idx]
                push!(get!(ket_groups, key, Pair{Int16,MVector{R2,T}}[]),
                      config[ci_idx] => coeff)
            end

            for (key, bra_list) in bra_groups
                haskey(ket_groups, key) || continue
                for (s_i, c_bra) in bra_list
                    for (t_i, c_ket) in ket_groups[key]
                        for r2 in 1:R2, r1 in 1:R1
                            ρ[s_i, t_i, r1, r2] += c_bra[r1] * c_ket[r2]
                        end
                    end
                end
            end

            # ---- Contract ρ with Aa/Bb to accumulate into γ ----
            # Aa_i stored as [norb_i^2, n_s, n_t]; reshape to [norb_i, norb_i, n_s, n_t]
            # Column-major: pq_flat = p + (q-1)*norb_i
            Aa_r = reshape(Aa_i, norb_i, norb_i, n_s, n_t)
            Bb_r = reshape(Bb_i, norb_i, norb_i, n_s, n_t)

            for r2 in 1:R2, r1 in 1:R1
                for t_i in 1:n_t, s_i in 1:n_s
                    ρval = ρ[s_i, t_i, r1, r2]
                    iszero(ρval) && continue
                    for q in 1:norb_i, p in 1:norb_i
                        γ_aa[off_i+p, off_i+q, r1, r2] += Aa_r[p, q, s_i, t_i] * ρval
                        γ_bb[off_i+p, off_i+q, r1, r2] += Bb_r[p, q, s_i, t_i] * ρval
                    end
                end
            end

        end  # ci loop
    end  # fock loop (on-cluster)


    # ===========================================================
    # 2. INTER-CLUSTER CONTRIBUTION
    # ===========================================================
    # For each ordered pair (i,j) with i≠j and each spin case:
    #
    #   γ_aa[off_i+p, off_j+q, r1, r2] +=
    #       sign * A_i[p,s_i,t_i] * a_j[q,s_j,t_j] * C_bra[r1] * C_ket[r2]
    #
    # where sign = fermionic anticommutation sign,
    #       fock_bra[i] = fock_ket[i] + (1,0),  fock_bra[j] = fock_ket[j] - (1,0)
    # (and analogously for beta-beta)

    for ci in clusters
        ci_idx = ci.idx
        norb_i = length(ci)
        off_i  = orb_offsets[ci_idx]

        for cj in clusters
            ci_idx == cj.idx && continue
            cj_idx = cj.idx
            norb_j = length(cj)
            off_j  = orb_offsets[cj_idx]

            # Precompute which cluster indices are "other" (not i or j)
            other_k = [k for k in 1:N if k != ci_idx && k != cj_idx]

            for (fock_ket, configs_ket) in ket.data

                na_i, nb_i = fock_ket[ci_idx]
                na_j, nb_j = fock_ket[cj_idx]

                # ---- Alpha-alpha: +1α at i, -1α at j ----
                if na_j >= 1
                    na_i_max = length(clusters[ci_idx]) # can't exceed norb_i
                    if na_i < na_i_max
                        fock_bra = replace(fock_ket,
                                           [ci_idx, cj_idx],
                                           [(na_i+1, nb_i), (na_j-1, nb_j)])

                        if haskey(bra.data, fock_bra)
                            configs_bra = bra.data[fock_bra]
                            ftrans_i = (fock_bra[ci_idx], fock_ket[ci_idx])
                            ftrans_j = (fock_bra[cj_idx], fock_ket[cj_idx])

                            if haskey(cluster_ops[ci_idx], "A") &&
                               haskey(cluster_ops[ci_idx]["A"], ftrans_i) &&
                               haskey(cluster_ops[cj_idx], "a") &&
                               haskey(cluster_ops[cj_idx]["a"], ftrans_j)

                                A_i = cluster_ops[ci_idx]["A"][ftrans_i]  # [norb_i, s_i, t_i]
                                a_j = cluster_ops[cj_idx]["a"][ftrans_j]  # [norb_j, s_j, t_j]
                                state_sign = _rdm_sign(fock_ket, ci_idx, cj_idx)

                                # Group by "other" cluster configs
                                bra_by_other = Dict{Vector{Int16},
                                    Vector{Tuple{Int16,Int16,MVector{R1,T}}}}()
                                for (config, coeff) in configs_bra
                                    key = [config[k] for k in other_k]
                                    push!(get!(bra_by_other, key,
                                               Tuple{Int16,Int16,MVector{R1,T}}[]),
                                          (config[ci_idx], config[cj_idx], coeff))
                                end

                                ket_by_other = Dict{Vector{Int16},
                                    Vector{Tuple{Int16,Int16,MVector{R2,T}}}}()
                                for (config, coeff) in configs_ket
                                    key = [config[k] for k in other_k]
                                    push!(get!(ket_by_other, key,
                                               Tuple{Int16,Int16,MVector{R2,T}}[]),
                                          (config[ci_idx], config[cj_idx], coeff))
                                end

                                for (key, bra_list) in bra_by_other
                                    haskey(ket_by_other, key) || continue
                                    for (s_i, s_j, c_bra) in bra_list
                                        for (t_i, t_j, c_ket) in ket_by_other[key]
                                            for r2 in 1:R2, r1 in 1:R1
                                                coeff_prod = state_sign * c_bra[r1] * c_ket[r2]
                                                iszero(coeff_prod) && continue
                                                for q in 1:norb_j
                                                    aj_val = a_j[q, s_j, t_j]
                                                    iszero(aj_val) && continue
                                                    for p in 1:norb_i
                                                        γ_aa[off_i+p, off_j+q, r1, r2] +=
                                                            A_i[p, s_i, t_i] * aj_val * coeff_prod
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end

                            end  # haskey checks
                        end  # haskey bra fock
                    end  # na_i < max
                end  # na_j >= 1  (alpha-alpha)

                # ---- Beta-beta: +1β at i, -1β at j ----
                if nb_j >= 1
                    nb_i_max = length(clusters[ci_idx])
                    if nb_i < nb_i_max
                        fock_bra = replace(fock_ket,
                                           [ci_idx, cj_idx],
                                           [(na_i, nb_i+1), (na_j, nb_j-1)])

                        if haskey(bra.data, fock_bra)
                            configs_bra = bra.data[fock_bra]
                            ftrans_i = (fock_bra[ci_idx], fock_ket[ci_idx])
                            ftrans_j = (fock_bra[cj_idx], fock_ket[cj_idx])

                            if haskey(cluster_ops[ci_idx], "B") &&
                               haskey(cluster_ops[ci_idx]["B"], ftrans_i) &&
                               haskey(cluster_ops[cj_idx], "b") &&
                               haskey(cluster_ops[cj_idx]["b"], ftrans_j)

                                B_i = cluster_ops[ci_idx]["B"][ftrans_i]
                                b_j = cluster_ops[cj_idx]["b"][ftrans_j]
                                state_sign = _rdm_sign(fock_ket, ci_idx, cj_idx)

                                bra_by_other = Dict{Vector{Int16},
                                    Vector{Tuple{Int16,Int16,MVector{R1,T}}}}()
                                for (config, coeff) in configs_bra
                                    key = [config[k] for k in other_k]
                                    push!(get!(bra_by_other, key,
                                               Tuple{Int16,Int16,MVector{R1,T}}[]),
                                          (config[ci_idx], config[cj_idx], coeff))
                                end

                                ket_by_other = Dict{Vector{Int16},
                                    Vector{Tuple{Int16,Int16,MVector{R2,T}}}}()
                                for (config, coeff) in configs_ket
                                    key = [config[k] for k in other_k]
                                    push!(get!(ket_by_other, key,
                                               Tuple{Int16,Int16,MVector{R2,T}}[]),
                                          (config[ci_idx], config[cj_idx], coeff))
                                end

                                for (key, bra_list) in bra_by_other
                                    haskey(ket_by_other, key) || continue
                                    for (s_i, s_j, c_bra) in bra_list
                                        for (t_i, t_j, c_ket) in ket_by_other[key]
                                            for r2 in 1:R2, r1 in 1:R1
                                                coeff_prod = state_sign * c_bra[r1] * c_ket[r2]
                                                iszero(coeff_prod) && continue
                                                for q in 1:norb_j
                                                    bj_val = b_j[q, s_j, t_j]
                                                    iszero(bj_val) && continue
                                                    for p in 1:norb_i
                                                        γ_bb[off_i+p, off_j+q, r1, r2] +=
                                                            B_i[p, s_i, t_i] * bj_val * coeff_prod
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end

                            end  # haskey checks
                        end  # haskey bra fock
                    end  # nb_i < max
                end  # nb_j >= 1  (beta-beta)

            end  # fock_ket loop
        end  # cj loop
    end  # ci loop (inter-cluster)

    return γ_aa, γ_bb
end


# ---------------------------------------------------------------------------
# Spin-flip 1-RDM
# ---------------------------------------------------------------------------
"""
    compute_1rdm_sf(bra::TPSCIstate{T,N,R1}, ket::TPSCIstate{T,N,R2}, cluster_ops)

Compute the spin-flip transition 1-RDM

    γ_ab[p,q,r1,r2] = <bra_{r1}|p'_{p,α} q_{q,β}|ket_{r2}>   (create α, annihilate β)
    γ_ba[p,q,r1,r2] = <bra_{r1}|p'_{p,β} q_{q,α}|ket_{r2}>   (create β, annihilate α)

Both tensors have shape (norb, norb, R1, R2).

These are needed for spin-orbit coupling matrix elements and any response
involving ΔM_S = ±1 (e.g. spin-flip transitions between states of different
multiplicity).

Two contributions per spin case:

  1. On-cluster  (p, q on the same cluster i):
        uses "Ab" = <s|p'_α q_β|t>  stored as (norb_i^2, n_s, n_t), reshaped to (norb_i,norb_i,n_s,n_t)
        and  "Ba" = <s|p'_β q_α|t>  (adjoint of Ab), same shape convention.
        bra and ket Fock sectors at cluster i differ by (±1,∓1).

  2. Inter-cluster (p on cluster i, q on cluster j, i≠j):
        α→β: uses "A"[p,s,t] (create α at i) × "b"[q,s,t] (annihilate β at j)
        β→α: uses "B"[p,s,t] (create β at i) × "a"[q,s,t] (annihilate α at j)
        with the standard fermionic anticommutation sign.
"""
function compute_1rdm_sf(bra::TPSCIstate{T,N,R1},
                         ket::TPSCIstate{T,N,R2},
                         cluster_ops::Vector{ClusterOps{T}}) where {T,N,R1,R2}

    clusters  = bra.clusters
    norb      = sum(length(c) for c in clusters)

    orb_offsets = zeros(Int, N)
    for i in 2:N
        orb_offsets[i] = orb_offsets[i-1] + length(clusters[i-1])
    end

    γ_ab = zeros(T, norb, norb, R1, R2)   # create α, annihilate β
    γ_ba = zeros(T, norb, norb, R1, R2)   # create β, annihilate α

    # ===========================================================
    # 1. ON-CLUSTER SPIN-FLIP CONTRIBUTION
    # ===========================================================
    # For p,q on cluster i, the bra and ket fock sectors at cluster i differ:
    #   γ_ab: fock_bra[i] = (na_i+1, nb_i-1),  fock_ket[i] = (na_i, nb_i)
    #   γ_ba: fock_bra[i] = (na_i-1, nb_i+1),  fock_ket[i] = (na_i, nb_i)
    # All other clusters must share the same fock sector.

    for (fock_ket, configs_ket) in ket.data
        for ci in clusters
            ci_idx = ci.idx
            norb_i = length(ci)
            off_i  = orb_offsets[ci_idx]
            na_i, nb_i = fock_ket[ci_idx]

            # ---- α→β on-cluster: create α, annihilate β at cluster i ----
            if nb_i >= 1 && na_i < norb_i
                fock_bra = replace(fock_ket, [ci_idx], [(na_i+1, nb_i-1)])

                if haskey(bra.data, fock_bra)
                    configs_bra = bra.data[fock_bra]
                    ftrans = (fock_bra[ci_idx], fock_ket[ci_idx])  # ((na+1,nb-1),(na,nb))

                    if haskey(cluster_ops[ci_idx], "Ab") &&
                       haskey(cluster_ops[ci_idx]["Ab"], ftrans)

                        # "Ab" stored as (norb_i^2, n_s, n_t) after global reshape
                        Ab_i = cluster_ops[ci_idx]["Ab"][ftrans]
                        n_s  = size(Ab_i, 2)
                        n_t  = size(Ab_i, 3)
                        Ab_r = reshape(Ab_i, norb_i, norb_i, n_s, n_t)

                        ρ = zeros(T, n_s, n_t, R1, R2)

                        bra_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R1,T}}}}()
                        for (config, coeff) in configs_bra
                            key = [config[k] for k in 1:N if k != ci_idx]
                            push!(get!(bra_groups, key, Pair{Int16,MVector{R1,T}}[]),
                                  config[ci_idx] => coeff)
                        end

                        ket_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R2,T}}}}()
                        for (config, coeff) in configs_ket
                            key = [config[k] for k in 1:N if k != ci_idx]
                            push!(get!(ket_groups, key, Pair{Int16,MVector{R2,T}}[]),
                                  config[ci_idx] => coeff)
                        end

                        for (key, bra_list) in bra_groups
                            haskey(ket_groups, key) || continue
                            for (s_i, c_bra) in bra_list
                                for (t_i, c_ket) in ket_groups[key]
                                    for r2 in 1:R2, r1 in 1:R1
                                        ρ[s_i, t_i, r1, r2] += c_bra[r1] * c_ket[r2]
                                    end
                                end
                            end
                        end

                        for r2 in 1:R2, r1 in 1:R1
                            for t_i in 1:n_t, s_i in 1:n_s
                                ρval = ρ[s_i, t_i, r1, r2]
                                iszero(ρval) && continue
                                for q in 1:norb_i, p in 1:norb_i
                                    γ_ab[off_i+p, off_i+q, r1, r2] +=
                                        Ab_r[p, q, s_i, t_i] * ρval
                                end
                            end
                        end

                    end  # haskey Ab
                end  # haskey fock_bra
            end  # nb_i >= 1 && na_i < norb_i

            # ---- β→α on-cluster: create β, annihilate α at cluster i ----
            if na_i >= 1 && nb_i < norb_i
                fock_bra = replace(fock_ket, [ci_idx], [(na_i-1, nb_i+1)])

                if haskey(bra.data, fock_bra)
                    configs_bra = bra.data[fock_bra]
                    ftrans = (fock_bra[ci_idx], fock_ket[ci_idx])  # ((na-1,nb+1),(na,nb))

                    if haskey(cluster_ops[ci_idx], "Ba") &&
                       haskey(cluster_ops[ci_idx]["Ba"], ftrans)

                        # "Ba" stored as (norb_i^2, n_s, n_t) after global reshape
                        Ba_i = cluster_ops[ci_idx]["Ba"][ftrans]
                        n_s  = size(Ba_i, 2)
                        n_t  = size(Ba_i, 3)
                        Ba_r = reshape(Ba_i, norb_i, norb_i, n_s, n_t)

                        ρ = zeros(T, n_s, n_t, R1, R2)

                        bra_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R1,T}}}}()
                        for (config, coeff) in configs_bra
                            key = [config[k] for k in 1:N if k != ci_idx]
                            push!(get!(bra_groups, key, Pair{Int16,MVector{R1,T}}[]),
                                  config[ci_idx] => coeff)
                        end

                        ket_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R2,T}}}}()
                        for (config, coeff) in configs_ket
                            key = [config[k] for k in 1:N if k != ci_idx]
                            push!(get!(ket_groups, key, Pair{Int16,MVector{R2,T}}[]),
                                  config[ci_idx] => coeff)
                        end

                        for (key, bra_list) in bra_groups
                            haskey(ket_groups, key) || continue
                            for (s_i, c_bra) in bra_list
                                for (t_i, c_ket) in ket_groups[key]
                                    for r2 in 1:R2, r1 in 1:R1
                                        ρ[s_i, t_i, r1, r2] += c_bra[r1] * c_ket[r2]
                                    end
                                end
                            end
                        end

                        for r2 in 1:R2, r1 in 1:R1
                            for t_i in 1:n_t, s_i in 1:n_s
                                ρval = ρ[s_i, t_i, r1, r2]
                                iszero(ρval) && continue
                                for q in 1:norb_i, p in 1:norb_i
                                    γ_ba[off_i+p, off_i+q, r1, r2] +=
                                        Ba_r[p, q, s_i, t_i] * ρval
                                end
                            end
                        end

                    end  # haskey Ba
                end  # haskey fock_bra
            end  # na_i >= 1 && nb_i < norb_i

        end  # ci loop
    end  # fock_ket loop (on-cluster)


    # ===========================================================
    # 2. INTER-CLUSTER SPIN-FLIP CONTRIBUTION
    # ===========================================================
    # α→β: p'_α at i, q_β at j  →  fock_bra[i]=(na_i+1,nb_i), fock_bra[j]=(na_j,nb_j-1)
    #       uses "A"[i] × "b"[j]
    # β→α: p'_β at i, q_α at j  →  fock_bra[i]=(na_i,nb_i+1), fock_bra[j]=(na_j-1,nb_j)
    #       uses "B"[i] × "a"[j]
    # Fermionic sign: same _rdm_sign convention as for αα/ββ.

    for ci in clusters
        ci_idx = ci.idx
        norb_i = length(ci)
        off_i  = orb_offsets[ci_idx]

        for cj in clusters
            ci_idx == cj.idx && continue
            cj_idx = cj.idx
            norb_j = length(cj)
            off_j  = orb_offsets[cj_idx]

            other_k = [k for k in 1:N if k != ci_idx && k != cj_idx]

            for (fock_ket, configs_ket) in ket.data

                na_i, nb_i = fock_ket[ci_idx]
                na_j, nb_j = fock_ket[cj_idx]

                # ---- α→β inter-cluster: create α at i, annihilate β at j ----
                if na_i < length(clusters[ci_idx]) && nb_j >= 1
                    fock_bra = replace(fock_ket,
                                       [ci_idx, cj_idx],
                                       [(na_i+1, nb_i), (na_j, nb_j-1)])

                    if haskey(bra.data, fock_bra)
                        configs_bra = bra.data[fock_bra]
                        ftrans_i = (fock_bra[ci_idx], fock_ket[ci_idx])  # ((na+1,nb),(na,nb)) → "A"
                        ftrans_j = (fock_bra[cj_idx], fock_ket[cj_idx])  # ((na,nb-1),(na,nb)) → "b"

                        if haskey(cluster_ops[ci_idx], "A") &&
                           haskey(cluster_ops[ci_idx]["A"], ftrans_i) &&
                           haskey(cluster_ops[cj_idx], "b") &&
                           haskey(cluster_ops[cj_idx]["b"], ftrans_j)

                            A_i = cluster_ops[ci_idx]["A"][ftrans_i]   # (norb_i, n_s, n_t)
                            b_j = cluster_ops[cj_idx]["b"][ftrans_j]   # (norb_j, n_s, n_t)
                            state_sign = _rdm_sign(fock_ket, ci_idx, cj_idx)

                            bra_by_other = Dict{Vector{Int16},
                                Vector{Tuple{Int16,Int16,MVector{R1,T}}}}()
                            for (config, coeff) in configs_bra
                                key = [config[k] for k in other_k]
                                push!(get!(bra_by_other, key,
                                           Tuple{Int16,Int16,MVector{R1,T}}[]),
                                      (config[ci_idx], config[cj_idx], coeff))
                            end

                            ket_by_other = Dict{Vector{Int16},
                                Vector{Tuple{Int16,Int16,MVector{R2,T}}}}()
                            for (config, coeff) in configs_ket
                                key = [config[k] for k in other_k]
                                push!(get!(ket_by_other, key,
                                           Tuple{Int16,Int16,MVector{R2,T}}[]),
                                      (config[ci_idx], config[cj_idx], coeff))
                            end

                            for (key, bra_list) in bra_by_other
                                haskey(ket_by_other, key) || continue
                                for (s_i, s_j, c_bra) in bra_list
                                    for (t_i, t_j, c_ket) in ket_by_other[key]
                                        for r2 in 1:R2, r1 in 1:R1
                                            coeff_prod = state_sign * c_bra[r1] * c_ket[r2]
                                            iszero(coeff_prod) && continue
                                            for q in 1:norb_j
                                                bj_val = b_j[q, s_j, t_j]
                                                iszero(bj_val) && continue
                                                for p in 1:norb_i
                                                    γ_ab[off_i+p, off_j+q, r1, r2] +=
                                                        A_i[p, s_i, t_i] * bj_val * coeff_prod
                                                end
                                            end
                                        end
                                    end
                                end
                            end

                        end  # haskey checks
                    end  # haskey fock_bra
                end  # α→β inter-cluster

                # ---- β→α inter-cluster: create β at i, annihilate α at j ----
                if nb_i < length(clusters[ci_idx]) && na_j >= 1
                    fock_bra = replace(fock_ket,
                                       [ci_idx, cj_idx],
                                       [(na_i, nb_i+1), (na_j-1, nb_j)])

                    if haskey(bra.data, fock_bra)
                        configs_bra = bra.data[fock_bra]
                        ftrans_i = (fock_bra[ci_idx], fock_ket[ci_idx])  # ((na,nb+1),(na,nb)) → "B"
                        ftrans_j = (fock_bra[cj_idx], fock_ket[cj_idx])  # ((na-1,nb),(na,nb)) → "a"

                        if haskey(cluster_ops[ci_idx], "B") &&
                           haskey(cluster_ops[ci_idx]["B"], ftrans_i) &&
                           haskey(cluster_ops[cj_idx], "a") &&
                           haskey(cluster_ops[cj_idx]["a"], ftrans_j)

                            B_i = cluster_ops[ci_idx]["B"][ftrans_i]   # (norb_i, n_s, n_t)
                            a_j = cluster_ops[cj_idx]["a"][ftrans_j]   # (norb_j, n_s, n_t)
                            state_sign = _rdm_sign(fock_ket, ci_idx, cj_idx)

                            bra_by_other = Dict{Vector{Int16},
                                Vector{Tuple{Int16,Int16,MVector{R1,T}}}}()
                            for (config, coeff) in configs_bra
                                key = [config[k] for k in other_k]
                                push!(get!(bra_by_other, key,
                                           Tuple{Int16,Int16,MVector{R1,T}}[]),
                                      (config[ci_idx], config[cj_idx], coeff))
                            end

                            ket_by_other = Dict{Vector{Int16},
                                Vector{Tuple{Int16,Int16,MVector{R2,T}}}}()
                            for (config, coeff) in configs_ket
                                key = [config[k] for k in other_k]
                                push!(get!(ket_by_other, key,
                                           Tuple{Int16,Int16,MVector{R2,T}}[]),
                                      (config[ci_idx], config[cj_idx], coeff))
                            end

                            for (key, bra_list) in bra_by_other
                                haskey(ket_by_other, key) || continue
                                for (s_i, s_j, c_bra) in bra_list
                                    for (t_i, t_j, c_ket) in ket_by_other[key]
                                        for r2 in 1:R2, r1 in 1:R1
                                            coeff_prod = state_sign * c_bra[r1] * c_ket[r2]
                                            iszero(coeff_prod) && continue
                                            for q in 1:norb_j
                                                aj_val = a_j[q, s_j, t_j]
                                                iszero(aj_val) && continue
                                                for p in 1:norb_i
                                                    γ_ba[off_i+p, off_j+q, r1, r2] +=
                                                        B_i[p, s_i, t_i] * aj_val * coeff_prod
                                                end
                                            end
                                        end
                                    end
                                end
                            end

                        end  # haskey checks
                    end  # haskey fock_bra
                end  # β→α inter-cluster

            end  # fock_ket loop
        end  # cj loop
    end  # ci loop (inter-cluster)

    return γ_ab, γ_ba
end


"""
    compute_1rdm_sf(psi::TPSCIstate{T,N,R}, cluster_ops)

Single-state spin-flip 1-RDM. Returns (γ_ab, γ_ba) of shape (norb, norb, R, R).
Diagonal elements [p,q,r,r] give the per-root spin-density matrix;
off-diagonal [p,q,r1,r2] are spin-flip transition 1-RDMs.
"""
function compute_1rdm_sf(psi::TPSCIstate{T,N,R},
                         cluster_ops::Vector{ClusterOps{T}}) where {T,N,R}
    return compute_1rdm_sf(psi, psi, cluster_ops)
end


"""
    compute_1rdm(psi::TPSCIstate{T,N,R}, cluster_ops)

Single-state version: returns (γ_aa, γ_bb) of shape (norb, norb, R, R).
Diagonal elements [p,q,r,r] are the per-root expectation values;
off-diagonal elements [p,q,r1,r2] are the inter-root transition 1-RDMs.
"""
function compute_1rdm(psi::TPSCIstate{T,N,R},
                      cluster_ops::Vector{ClusterOps{T}}) where {T,N,R}
    return compute_1rdm(psi, psi, cluster_ops)
end


# ---------------------------------------------------------------------------
# Property contraction
# ---------------------------------------------------------------------------
"""
    contract_1rdm_property(γ_aa, γ_bb, prop_ints::AbstractMatrix{T})

Contract the transition 1-RDM with a one-electron property integral matrix.

    P[r1,r2] = Σ_{pq} prop_ints[p,q] * (γ_aa[p,q,r1,r2] + γ_bb[p,q,r1,r2])

Returns a matrix of shape (R1, R2).

`prop_ints` must have size (norb, norb) in the same orbital ordering as the
cluster_ops (i.e. clusters concatenated in index order).
"""
function contract_1rdm_property(γ_aa::Array{T,4}, γ_bb::Array{T,4},
                                 prop_ints::AbstractMatrix{T}) where T
    norb, _, R1, R2 = size(γ_aa)
    size(prop_ints) == (norb, norb) || throw(DimensionMismatch(
        "prop_ints size $(size(prop_ints)) does not match 1-RDM norb=$norb"))

    P = zeros(T, R1, R2)
    γ_total = γ_aa .+ γ_bb
    for r2 in 1:R2, r1 in 1:R1
        @views P[r1, r2] = dot(vec(prop_ints), vec(γ_total[:, :, r1, r2]))
    end
    return P
end


"""
    contract_1rdm_property(γ_aa, γ_bb, prop_ints_list::Vector{<:AbstractMatrix{T}})

Contract with multiple property integral matrices at once.
Returns a Vector of (R1×R2) matrices, one per property.
"""
function contract_1rdm_property(γ_aa::Array{T,4}, γ_bb::Array{T,4},
                                 prop_ints_list::Vector{<:AbstractMatrix{T}}) where T
    return [contract_1rdm_property(γ_aa, γ_bb, p) for p in prop_ints_list]
end


# ---------------------------------------------------------------------------
# Direct 1-electron property (without forming the full 1-RDM)
# ---------------------------------------------------------------------------
"""
    compute_1e_property_direct(bra, ket, cluster_ops, h_prop)

Compute the one-electron property matrix

    P[r1,r2] = <bra_{r1}|Σ_{pq} h_prop[p,q] (a†_{p,α} a_{q,α} + a†_{p,β} a_{q,β})|ket_{r2}>

without forming the full (norb,norb,R1,R2) 1-RDM.  Instead the orbital-index
contraction with `h_prop` is performed on-the-fly:

  • On-cluster I:
        M_I[s,t] = Σ_{p,q∈I} h_prop_I[p,q] (Aa_I[p,q,s,t] + Bb_I[p,q,s,t])
        P[r1,r2] += Σ_{s,t} M_I[s,t] ρ_I[s,t,r1,r2]

  • Inter-cluster CT (α): p∈I creates, q∈J annihilates
        coupling_IJ[s_I,t_I,s_J,t_J] = Σ_{p∈I,q∈J} h_prop[off_I+p, off_J+q]
                                          × A_I[p,s_I,t_I] × a_J[q,s_J,t_J]
        P[r1,r2] += sign × coupling_IJ × c_bra[r1] × c_ket[r2]   (+ h.c.)

  • Same for β (B/b operators).

Returns `P` of shape (R1, R2).

Compared with `contract_1rdm_property(compute_1rdm(...), h_prop)` this avoids
allocating a large intermediate RDM but should give identical results (use both
as a correctness check).
"""
function compute_1e_property_direct(bra::TPSCIstate{T,N,R1},
                                     ket::TPSCIstate{T,N,R2},
                                     cluster_ops::Vector{ClusterOps{T}},
                                     h_prop::AbstractMatrix{T}) where {T,N,R1,R2}

    clusters = bra.clusters
    norb     = sum(length(c) for c in clusters)
    size(h_prop) == (norb, norb) || throw(DimensionMismatch(
        "h_prop size $(size(h_prop)) ≠ (norb=$norb, norb=$norb)"))

    P = zeros(T, R1, R2)

    orb_offsets = zeros(Int, N)
    for i in 2:N
        orb_offsets[i] = orb_offsets[i-1] + length(clusters[i-1])
    end

    # =========================================================
    # 1. ON-CLUSTER CONTRIBUTION
    # =========================================================
    for (fock, configs_bra) in bra.data
        haskey(ket.data, fock) || continue
        configs_ket = ket.data[fock]

        for ci in clusters
            ci_idx = ci.idx
            norb_i = length(ci)
            off_i  = orb_offsets[ci_idx]
            ftrans = (fock[ci_idx], fock[ci_idx])

            haskey(cluster_ops[ci_idx], "Aa") || continue
            haskey(cluster_ops[ci_idx]["Aa"], ftrans) || continue
            haskey(cluster_ops[ci_idx], "Bb") || continue
            haskey(cluster_ops[ci_idx]["Bb"], ftrans) || continue

            Aa_i = cluster_ops[ci_idx]["Aa"][ftrans]   # (norb_i^2, n_s, n_t)
            Bb_i = cluster_ops[ci_idx]["Bb"][ftrans]
            n_s  = size(Aa_i, 2)
            n_t  = size(Aa_i, 3)
            Aa_r = reshape(Aa_i, norb_i, norb_i, n_s, n_t)
            Bb_r = reshape(Bb_i, norb_i, norb_i, n_s, n_t)

            # on-cluster block of h_prop
            h_I = view(h_prop, off_i+1:off_i+norb_i, off_i+1:off_i+norb_i)

            # Precompute M_I[s,t] = Σ_{pq} h_I[p,q] (Aa[p,q,s,t] + Bb[p,q,s,t])
            M_I = zeros(T, n_s, n_t)
            for t_i in 1:n_t, s_i in 1:n_s
                acc = zero(T)
                for q in 1:norb_i, p in 1:norb_i
                    acc += h_I[p, q] * (Aa_r[p, q, s_i, t_i] + Bb_r[p, q, s_i, t_i])
                end
                M_I[s_i, t_i] = acc
            end

            # Build cluster RDM ρ_I (same grouping as compute_1rdm)
            ρ = zeros(T, n_s, n_t, R1, R2)
            bra_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R1,T}}}}()
            for (config, coeff) in configs_bra
                key = [config[k] for k in 1:N if k != ci_idx]
                push!(get!(bra_groups, key, Pair{Int16,MVector{R1,T}}[]),
                      config[ci_idx] => coeff)
            end
            ket_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R2,T}}}}()
            for (config, coeff) in configs_ket
                key = [config[k] for k in 1:N if k != ci_idx]
                push!(get!(ket_groups, key, Pair{Int16,MVector{R2,T}}[]),
                      config[ci_idx] => coeff)
            end
            for (key, bra_list) in bra_groups
                haskey(ket_groups, key) || continue
                for (s_i, c_bra) in bra_list
                    for (t_i, c_ket) in ket_groups[key]
                        for r2 in 1:R2, r1 in 1:R1
                            ρ[s_i, t_i, r1, r2] += c_bra[r1] * c_ket[r2]
                        end
                    end
                end
            end

            # P[r1,r2] += Σ_{st} M_I[s,t] * ρ[s,t,r1,r2]
            for r2 in 1:R2, r1 in 1:R1
                acc = zero(T)
                for t_i in 1:n_t, s_i in 1:n_s
                    acc += M_I[s_i, t_i] * ρ[s_i, t_i, r1, r2]
                end
                P[r1, r2] += acc
            end

        end  # ci loop
    end  # fock loop


    # =========================================================
    # 2. INTER-CLUSTER CT CONTRIBUTION
    # =========================================================
    for ci in clusters
        ci_idx = ci.idx
        norb_i = length(ci)
        off_i  = orb_offsets[ci_idx]

        for cj in clusters
            ci_idx == cj.idx && continue
            cj_idx = cj.idx
            norb_j = length(cj)
            off_j  = orb_offsets[cj_idx]

            # Off-diagonal block of h_prop: p∈I creates, q∈J annihilates
            h_IJ = view(h_prop, off_i+1:off_i+norb_i, off_j+1:off_j+norb_j)
            all(iszero, h_IJ) && continue   # skip if no coupling

            other_k = [k for k in 1:N if k != ci_idx && k != cj_idx]

            for (fock_ket, configs_ket) in ket.data
                na_i, nb_i = fock_ket[ci_idx]
                na_j, nb_j = fock_ket[cj_idx]

                # ---- Alpha: +1α at I, -1α at J ----
                if na_j >= 1 && na_i < length(clusters[ci_idx])
                    fock_bra = replace(fock_ket,
                                       [ci_idx, cj_idx],
                                       [(na_i+1, nb_i), (na_j-1, nb_j)])
                    haskey(bra.data, fock_bra) || @goto skip_aa_direct
                    configs_bra = bra.data[fock_bra]
                    ftrans_i = (fock_bra[ci_idx], fock_ket[ci_idx])
                    ftrans_j = (fock_bra[cj_idx], fock_ket[cj_idx])

                    if haskey(cluster_ops[ci_idx], "A") &&
                       haskey(cluster_ops[ci_idx]["A"], ftrans_i) &&
                       haskey(cluster_ops[cj_idx], "a") &&
                       haskey(cluster_ops[cj_idx]["a"], ftrans_j)

                        A_i = cluster_ops[ci_idx]["A"][ftrans_i]   # (norb_i, n_si, n_ti)
                        a_j = cluster_ops[cj_idx]["a"][ftrans_j]   # (norb_j, n_sj, n_tj)
                        state_sign = _rdm_sign(fock_ket, ci_idx, cj_idx)

                        # Precompute coupling[s_i,t_i,s_j,t_j] = Σ_{pq} h_IJ[p,q] A_i[p,si,ti] a_j[q,sj,tj]
                        n_si = size(A_i, 2); n_ti = size(A_i, 3)
                        n_sj = size(a_j, 2); n_tj = size(a_j, 3)
                        coupling = zeros(T, n_si, n_ti, n_sj, n_tj)
                        for tj in 1:n_tj, sj in 1:n_sj, ti in 1:n_ti, si in 1:n_si
                            acc = zero(T)
                            for q in 1:norb_j, p in 1:norb_i
                                acc += h_IJ[p, q] * A_i[p, si, ti] * a_j[q, sj, tj]
                            end
                            coupling[si, ti, sj, tj] = acc
                        end

                        bra_by_other = Dict{Vector{Int16},
                                            Vector{Tuple{Int16,Int16,MVector{R1,T}}}}()
                        for (config, coeff) in configs_bra
                            key = [config[k] for k in other_k]
                            push!(get!(bra_by_other, key, Tuple{Int16,Int16,MVector{R1,T}}[]),
                                  (config[ci_idx], config[cj_idx], coeff))
                        end
                        ket_by_other = Dict{Vector{Int16},
                                            Vector{Tuple{Int16,Int16,MVector{R2,T}}}}()
                        for (config, coeff) in configs_ket
                            key = [config[k] for k in other_k]
                            push!(get!(ket_by_other, key, Tuple{Int16,Int16,MVector{R2,T}}[]),
                                  (config[ci_idx], config[cj_idx], coeff))
                        end

                        for (key, bra_list) in bra_by_other
                            haskey(ket_by_other, key) || continue
                            for (s_i, s_j, c_bra) in bra_list
                                for (t_i, t_j, c_ket) in ket_by_other[key]
                                    coup = state_sign * coupling[s_i, t_i, s_j, t_j]
                                    iszero(coup) && continue
                                    for r2 in 1:R2, r1 in 1:R1
                                        P[r1, r2] += coup * c_bra[r1] * c_ket[r2]
                                    end
                                end
                            end
                        end
                    end
                    @label skip_aa_direct
                end  # alpha-alpha

                # ---- Beta: +1β at I, -1β at J ----
                if nb_j >= 1 && nb_i < length(clusters[ci_idx])
                    fock_bra = replace(fock_ket,
                                       [ci_idx, cj_idx],
                                       [(na_i, nb_i+1), (na_j, nb_j-1)])
                    haskey(bra.data, fock_bra) || @goto skip_bb_direct
                    configs_bra = bra.data[fock_bra]
                    ftrans_i = (fock_bra[ci_idx], fock_ket[ci_idx])
                    ftrans_j = (fock_bra[cj_idx], fock_ket[cj_idx])

                    if haskey(cluster_ops[ci_idx], "B") &&
                       haskey(cluster_ops[ci_idx]["B"], ftrans_i) &&
                       haskey(cluster_ops[cj_idx], "b") &&
                       haskey(cluster_ops[cj_idx]["b"], ftrans_j)

                        B_i = cluster_ops[ci_idx]["B"][ftrans_i]
                        b_j = cluster_ops[cj_idx]["b"][ftrans_j]
                        state_sign = _rdm_sign(fock_ket, ci_idx, cj_idx)

                        n_si = size(B_i, 2); n_ti = size(B_i, 3)
                        n_sj = size(b_j, 2); n_tj = size(b_j, 3)
                        coupling = zeros(T, n_si, n_ti, n_sj, n_tj)
                        for tj in 1:n_tj, sj in 1:n_sj, ti in 1:n_ti, si in 1:n_si
                            acc = zero(T)
                            for q in 1:norb_j, p in 1:norb_i
                                acc += h_IJ[p, q] * B_i[p, si, ti] * b_j[q, sj, tj]
                            end
                            coupling[si, ti, sj, tj] = acc
                        end

                        bra_by_other = Dict{Vector{Int16},
                                            Vector{Tuple{Int16,Int16,MVector{R1,T}}}}()
                        for (config, coeff) in configs_bra
                            key = [config[k] for k in other_k]
                            push!(get!(bra_by_other, key, Tuple{Int16,Int16,MVector{R1,T}}[]),
                                  (config[ci_idx], config[cj_idx], coeff))
                        end
                        ket_by_other = Dict{Vector{Int16},
                                            Vector{Tuple{Int16,Int16,MVector{R2,T}}}}()
                        for (config, coeff) in configs_ket
                            key = [config[k] for k in other_k]
                            push!(get!(ket_by_other, key, Tuple{Int16,Int16,MVector{R2,T}}[]),
                                  (config[ci_idx], config[cj_idx], coeff))
                        end

                        for (key, bra_list) in bra_by_other
                            haskey(ket_by_other, key) || continue
                            for (s_i, s_j, c_bra) in bra_list
                                for (t_i, t_j, c_ket) in ket_by_other[key]
                                    coup = state_sign * coupling[s_i, t_i, s_j, t_j]
                                    iszero(coup) && continue
                                    for r2 in 1:R2, r1 in 1:R1
                                        P[r1, r2] += coup * c_bra[r1] * c_ket[r2]
                                    end
                                end
                            end
                        end
                    end
                    @label skip_bb_direct
                end  # beta-beta

            end  # fock_ket loop
        end  # cj loop
    end  # ci loop

    return P
end

"""
    compute_1e_property_direct(psi, cluster_ops, h_prop)

Single-state version: returns P of shape (R, R).
"""
function compute_1e_property_direct(psi::TPSCIstate{T,N,R},
                                     cluster_ops::Vector{ClusterOps{T}},
                                     h_prop::AbstractMatrix{T}) where {T,N,R}
    return compute_1e_property_direct(psi, psi, cluster_ops, h_prop)
end
"""
tpsci_property_threaded.jl

Parallel (multi-threaded) versions of compute_1rdm and compute_1rdm_sf.

Strategy
--------
Both functions have two independent loops:
  1. On-cluster  : pairs (fock, ci)   — flattened and distributed with @threads
  2. Inter-cluster: pairs (ci, cj)    — flattened and distributed with @threads

Each thread accumulates into its own local γ array (γ_xx_loc[threadid()]).
These are summed serially at the end, so there are no data races and no locks.

Usage
-----
    γ_aa, γ_bb = compute_1rdm_threaded(bra, ket, cluster_ops)
    γ_ab, γ_ba = compute_1rdm_sf_threaded(bra, ket, cluster_ops)

The single-state wrappers (bra == ket) are also provided.
"""

using Base.Threads


# ---------------------------------------------------------------------------
# Parallel 1-RDM
# ---------------------------------------------------------------------------
"""
    compute_1rdm_threaded(bra, ket, cluster_ops)

Multi-threaded version of `compute_1rdm`. Returns (γ_aa, γ_bb), each of
shape (norb, norb, R1, R2). Results are numerically identical to the serial
version within floating-point rounding.
"""
function compute_1rdm_threaded(bra::TPSCIstate{T,N,R1},
                                ket::TPSCIstate{T,N,R2},
                                cluster_ops::Vector{ClusterOps{T}}) where {T,N,R1,R2}

    clusters = bra.clusters
    norb     = sum(length(c) for c in clusters)

    orb_offsets = zeros(Int, N)
    for i in 2:N
        orb_offsets[i] = orb_offsets[i-1] + length(clusters[i-1])
    end

    nthreads  = Threads.nthreads()
    γ_aa_loc  = [zeros(T, norb, norb, R1, R2) for _ in 1:nthreads]
    γ_bb_loc  = [zeros(T, norb, norb, R1, R2) for _ in 1:nthreads]

    # ==========================================================
    # 1. ON-CLUSTER — @threads over (fock, ci) pairs
    # ==========================================================
    fock_list  = [fock for (fock, _) in bra.data if haskey(ket.data, fock)]
    on_cluster_pairs = [(fock, ci) for fock in fock_list for ci in clusters]

    @threads for idx in eachindex(on_cluster_pairs)
        tid  = Threads.threadid()
        fock, ci = on_cluster_pairs[idx]

        configs_bra = bra.data[fock]
        configs_ket = ket.data[fock]
        ci_idx = ci.idx
        norb_i = length(ci)
        off_i  = orb_offsets[ci_idx]
        ftrans = (fock[ci_idx], fock[ci_idx])

        haskey(cluster_ops[ci_idx], "Aa")          || continue
        haskey(cluster_ops[ci_idx]["Aa"], ftrans)   || continue
        haskey(cluster_ops[ci_idx], "Bb")           || continue
        haskey(cluster_ops[ci_idx]["Bb"], ftrans)   || continue

        Aa_i = cluster_ops[ci_idx]["Aa"][ftrans]
        Bb_i = cluster_ops[ci_idx]["Bb"][ftrans]
        n_s  = size(Aa_i, 2)
        n_t  = size(Aa_i, 3)

        ρ = zeros(T, n_s, n_t, R1, R2)

        bra_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R1,T}}}}()
        for (config, coeff) in configs_bra
            key = [config[k] for k in 1:N if k != ci_idx]
            push!(get!(bra_groups, key, Pair{Int16,MVector{R1,T}}[]),
                  config[ci_idx] => coeff)
        end

        ket_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R2,T}}}}()
        for (config, coeff) in configs_ket
            key = [config[k] for k in 1:N if k != ci_idx]
            push!(get!(ket_groups, key, Pair{Int16,MVector{R2,T}}[]),
                  config[ci_idx] => coeff)
        end

        for (key, bra_list) in bra_groups
            haskey(ket_groups, key) || continue
            for (s_i, c_bra) in bra_list
                for (t_i, c_ket) in ket_groups[key]
                    for r2 in 1:R2, r1 in 1:R1
                        ρ[s_i, t_i, r1, r2] += c_bra[r1] * c_ket[r2]
                    end
                end
            end
        end

        Aa_r = reshape(Aa_i, norb_i, norb_i, n_s, n_t)
        Bb_r = reshape(Bb_i, norb_i, norb_i, n_s, n_t)

        for r2 in 1:R2, r1 in 1:R1
            for t_i in 1:n_t, s_i in 1:n_s
                ρval = ρ[s_i, t_i, r1, r2]
                iszero(ρval) && continue
                for q in 1:norb_i, p in 1:norb_i
                    γ_aa_loc[tid][off_i+p, off_i+q, r1, r2] += Aa_r[p, q, s_i, t_i] * ρval
                    γ_bb_loc[tid][off_i+p, off_i+q, r1, r2] += Bb_r[p, q, s_i, t_i] * ρval
                end
            end
        end
    end  # on-cluster @threads

    # ==========================================================
    # 2. INTER-CLUSTER — @threads over (ci, cj) pairs
    # ==========================================================
    ci_cj_pairs = [(ci, cj) for ci in clusters for cj in clusters
                   if ci.idx != cj.idx]

    @threads for idx in eachindex(ci_cj_pairs)
        tid    = Threads.threadid()
        ci, cj = ci_cj_pairs[idx]
        ci_idx = ci.idx;  norb_i = length(ci);  off_i = orb_offsets[ci_idx]
        cj_idx = cj.idx;  norb_j = length(cj);  off_j = orb_offsets[cj_idx]
        other_k = [k for k in 1:N if k != ci_idx && k != cj_idx]

        for (fock_ket, configs_ket) in ket.data
            na_i, nb_i = fock_ket[ci_idx]
            na_j, nb_j = fock_ket[cj_idx]

            # ---- Alpha-alpha: create α at i, annihilate α at j ----
            if na_j >= 1 && na_i < length(clusters[ci_idx])
                fock_bra = replace(fock_ket, [ci_idx, cj_idx],
                                   [(na_i+1, nb_i), (na_j-1, nb_j)])

                if haskey(bra.data, fock_bra)
                    configs_bra = bra.data[fock_bra]
                    ftrans_i = (fock_bra[ci_idx], fock_ket[ci_idx])
                    ftrans_j = (fock_bra[cj_idx], fock_ket[cj_idx])

                    if haskey(cluster_ops[ci_idx], "A") &&
                       haskey(cluster_ops[ci_idx]["A"], ftrans_i) &&
                       haskey(cluster_ops[cj_idx], "a") &&
                       haskey(cluster_ops[cj_idx]["a"], ftrans_j)

                        A_i        = cluster_ops[ci_idx]["A"][ftrans_i]
                        a_j        = cluster_ops[cj_idx]["a"][ftrans_j]
                        state_sign = _rdm_sign(fock_ket, ci_idx, cj_idx)

                        bra_by_other = Dict{Vector{Int16},
                            Vector{Tuple{Int16,Int16,MVector{R1,T}}}}()
                        for (config, coeff) in configs_bra
                            key = [config[k] for k in other_k]
                            push!(get!(bra_by_other, key,
                                       Tuple{Int16,Int16,MVector{R1,T}}[]),
                                  (config[ci_idx], config[cj_idx], coeff))
                        end

                        ket_by_other = Dict{Vector{Int16},
                            Vector{Tuple{Int16,Int16,MVector{R2,T}}}}()
                        for (config, coeff) in configs_ket
                            key = [config[k] for k in other_k]
                            push!(get!(ket_by_other, key,
                                       Tuple{Int16,Int16,MVector{R2,T}}[]),
                                  (config[ci_idx], config[cj_idx], coeff))
                        end

                        for (key, bra_list) in bra_by_other
                            haskey(ket_by_other, key) || continue
                            for (s_i, s_j, c_bra) in bra_list
                                for (t_i, t_j, c_ket) in ket_by_other[key]
                                    for r2 in 1:R2, r1 in 1:R1
                                        cp = state_sign * c_bra[r1] * c_ket[r2]
                                        iszero(cp) && continue
                                        for q in 1:norb_j
                                            aj = a_j[q, s_j, t_j]
                                            iszero(aj) && continue
                                            for p in 1:norb_i
                                                γ_aa_loc[tid][off_i+p, off_j+q, r1, r2] +=
                                                    A_i[p, s_i, t_i] * aj * cp
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end  # alpha-alpha

            # ---- Beta-beta: create β at i, annihilate β at j ----
            if nb_j >= 1 && nb_i < length(clusters[ci_idx])
                fock_bra = replace(fock_ket, [ci_idx, cj_idx],
                                   [(na_i, nb_i+1), (na_j, nb_j-1)])

                if haskey(bra.data, fock_bra)
                    configs_bra = bra.data[fock_bra]
                    ftrans_i = (fock_bra[ci_idx], fock_ket[ci_idx])
                    ftrans_j = (fock_bra[cj_idx], fock_ket[cj_idx])

                    if haskey(cluster_ops[ci_idx], "B") &&
                       haskey(cluster_ops[ci_idx]["B"], ftrans_i) &&
                       haskey(cluster_ops[cj_idx], "b") &&
                       haskey(cluster_ops[cj_idx]["b"], ftrans_j)

                        B_i        = cluster_ops[ci_idx]["B"][ftrans_i]
                        b_j        = cluster_ops[cj_idx]["b"][ftrans_j]
                        state_sign = _rdm_sign(fock_ket, ci_idx, cj_idx)

                        bra_by_other = Dict{Vector{Int16},
                            Vector{Tuple{Int16,Int16,MVector{R1,T}}}}()
                        for (config, coeff) in configs_bra
                            key = [config[k] for k in other_k]
                            push!(get!(bra_by_other, key,
                                       Tuple{Int16,Int16,MVector{R1,T}}[]),
                                  (config[ci_idx], config[cj_idx], coeff))
                        end

                        ket_by_other = Dict{Vector{Int16},
                            Vector{Tuple{Int16,Int16,MVector{R2,T}}}}()
                        for (config, coeff) in configs_ket
                            key = [config[k] for k in other_k]
                            push!(get!(ket_by_other, key,
                                       Tuple{Int16,Int16,MVector{R2,T}}[]),
                                  (config[ci_idx], config[cj_idx], coeff))
                        end

                        for (key, bra_list) in bra_by_other
                            haskey(ket_by_other, key) || continue
                            for (s_i, s_j, c_bra) in bra_list
                                for (t_i, t_j, c_ket) in ket_by_other[key]
                                    for r2 in 1:R2, r1 in 1:R1
                                        cp = state_sign * c_bra[r1] * c_ket[r2]
                                        iszero(cp) && continue
                                        for q in 1:norb_j
                                            bj = b_j[q, s_j, t_j]
                                            iszero(bj) && continue
                                            for p in 1:norb_i
                                                γ_bb_loc[tid][off_i+p, off_j+q, r1, r2] +=
                                                    B_i[p, s_i, t_i] * bj * cp
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end  # beta-beta

        end  # fock_ket loop
    end  # inter-cluster @threads

    # Sum thread-local arrays
    γ_aa = sum(γ_aa_loc)
    γ_bb = sum(γ_bb_loc)
    return γ_aa, γ_bb
end


"""
    compute_1rdm_threaded(psi, cluster_ops)

Single-state parallel 1-RDM. Returns (γ_aa, γ_bb) of shape (norb, norb, R, R).
"""
function compute_1rdm_threaded(psi::TPSCIstate{T,N,R},
                                cluster_ops::Vector{ClusterOps{T}}) where {T,N,R}
    return compute_1rdm_threaded(psi, psi, cluster_ops)
end


# ---------------------------------------------------------------------------
# Parallel spin-flip 1-RDM
# ---------------------------------------------------------------------------
"""
    compute_1rdm_sf_threaded(bra, ket, cluster_ops)

Multi-threaded version of `compute_1rdm_sf`. Returns (γ_ab, γ_ba), each of
shape (norb, norb, R1, R2). Results are numerically identical to the serial
version within floating-point rounding.
"""
function compute_1rdm_sf_threaded(bra::TPSCIstate{T,N,R1},
                                   ket::TPSCIstate{T,N,R2},
                                   cluster_ops::Vector{ClusterOps{T}}) where {T,N,R1,R2}

    clusters = bra.clusters
    norb     = sum(length(c) for c in clusters)

    orb_offsets = zeros(Int, N)
    for i in 2:N
        orb_offsets[i] = orb_offsets[i-1] + length(clusters[i-1])
    end

    nthreads  = Threads.nthreads()
    γ_ab_loc  = [zeros(T, norb, norb, R1, R2) for _ in 1:nthreads]
    γ_ba_loc  = [zeros(T, norb, norb, R1, R2) for _ in 1:nthreads]

    # ==========================================================
    # 1. ON-CLUSTER SPIN-FLIP — @threads over (fock_ket, ci) pairs
    # ==========================================================
    fock_ket_list    = collect(keys(ket.data))
    on_cluster_pairs = [(fock_ket, ci) for fock_ket in fock_ket_list for ci in clusters]

    @threads for idx in eachindex(on_cluster_pairs)
        tid           = Threads.threadid()
        fock_ket, ci  = on_cluster_pairs[idx]
        configs_ket   = ket.data[fock_ket]
        ci_idx        = ci.idx
        norb_i        = length(ci)
        off_i         = orb_offsets[ci_idx]
        na_i, nb_i    = fock_ket[ci_idx]

        # ---- α→β on-cluster: create α, annihilate β at cluster i ----
        if nb_i >= 1 && na_i < norb_i
            fock_bra = replace(fock_ket, [ci_idx], [(na_i+1, nb_i-1)])

            if haskey(bra.data, fock_bra)
                configs_bra = bra.data[fock_bra]
                ftrans = (fock_bra[ci_idx], fock_ket[ci_idx])

                if haskey(cluster_ops[ci_idx], "Ab") &&
                   haskey(cluster_ops[ci_idx]["Ab"], ftrans)

                    Ab_i = cluster_ops[ci_idx]["Ab"][ftrans]
                    n_s  = size(Ab_i, 2)
                    n_t  = size(Ab_i, 3)
                    Ab_r = reshape(Ab_i, norb_i, norb_i, n_s, n_t)
                    ρ    = zeros(T, n_s, n_t, R1, R2)

                    bra_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R1,T}}}}()
                    for (config, coeff) in configs_bra
                        key = [config[k] for k in 1:N if k != ci_idx]
                        push!(get!(bra_groups, key, Pair{Int16,MVector{R1,T}}[]),
                              config[ci_idx] => coeff)
                    end
                    ket_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R2,T}}}}()
                    for (config, coeff) in configs_ket
                        key = [config[k] for k in 1:N if k != ci_idx]
                        push!(get!(ket_groups, key, Pair{Int16,MVector{R2,T}}[]),
                              config[ci_idx] => coeff)
                    end

                    for (key, bra_list) in bra_groups
                        haskey(ket_groups, key) || continue
                        for (s_i, c_bra) in bra_list
                            for (t_i, c_ket) in ket_groups[key]
                                for r2 in 1:R2, r1 in 1:R1
                                    ρ[s_i, t_i, r1, r2] += c_bra[r1] * c_ket[r2]
                                end
                            end
                        end
                    end

                    for r2 in 1:R2, r1 in 1:R1
                        for t_i in 1:n_t, s_i in 1:n_s
                            ρval = ρ[s_i, t_i, r1, r2]
                            iszero(ρval) && continue
                            for q in 1:norb_i, p in 1:norb_i
                                γ_ab_loc[tid][off_i+p, off_i+q, r1, r2] +=
                                    Ab_r[p, q, s_i, t_i] * ρval
                            end
                        end
                    end
                end
            end
        end  # α→β on-cluster

        # ---- β→α on-cluster: create β, annihilate α at cluster i ----
        if na_i >= 1 && nb_i < norb_i
            fock_bra = replace(fock_ket, [ci_idx], [(na_i-1, nb_i+1)])

            if haskey(bra.data, fock_bra)
                configs_bra = bra.data[fock_bra]
                ftrans = (fock_bra[ci_idx], fock_ket[ci_idx])

                if haskey(cluster_ops[ci_idx], "Ba") &&
                   haskey(cluster_ops[ci_idx]["Ba"], ftrans)

                    Ba_i = cluster_ops[ci_idx]["Ba"][ftrans]
                    n_s  = size(Ba_i, 2)
                    n_t  = size(Ba_i, 3)
                    Ba_r = reshape(Ba_i, norb_i, norb_i, n_s, n_t)
                    ρ    = zeros(T, n_s, n_t, R1, R2)

                    bra_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R1,T}}}}()
                    for (config, coeff) in configs_bra
                        key = [config[k] for k in 1:N if k != ci_idx]
                        push!(get!(bra_groups, key, Pair{Int16,MVector{R1,T}}[]),
                              config[ci_idx] => coeff)
                    end
                    ket_groups = Dict{Vector{Int16}, Vector{Pair{Int16, MVector{R2,T}}}}()
                    for (config, coeff) in configs_ket
                        key = [config[k] for k in 1:N if k != ci_idx]
                        push!(get!(ket_groups, key, Pair{Int16,MVector{R2,T}}[]),
                              config[ci_idx] => coeff)
                    end

                    for (key, bra_list) in bra_groups
                        haskey(ket_groups, key) || continue
                        for (s_i, c_bra) in bra_list
                            for (t_i, c_ket) in ket_groups[key]
                                for r2 in 1:R2, r1 in 1:R1
                                    ρ[s_i, t_i, r1, r2] += c_bra[r1] * c_ket[r2]
                                end
                            end
                        end
                    end

                    for r2 in 1:R2, r1 in 1:R1
                        for t_i in 1:n_t, s_i in 1:n_s
                            ρval = ρ[s_i, t_i, r1, r2]
                            iszero(ρval) && continue
                            for q in 1:norb_i, p in 1:norb_i
                                γ_ba_loc[tid][off_i+p, off_i+q, r1, r2] +=
                                    Ba_r[p, q, s_i, t_i] * ρval
                            end
                        end
                    end
                end
            end
        end  # β→α on-cluster

    end  # on-cluster @threads

    # ==========================================================
    # 2. INTER-CLUSTER SPIN-FLIP — @threads over (ci, cj) pairs
    # ==========================================================
    ci_cj_pairs = [(ci, cj) for ci in clusters for cj in clusters
                   if ci.idx != cj.idx]

    @threads for idx in eachindex(ci_cj_pairs)
        tid    = Threads.threadid()
        ci, cj = ci_cj_pairs[idx]
        ci_idx = ci.idx;  norb_i = length(ci);  off_i = orb_offsets[ci_idx]
        cj_idx = cj.idx;  norb_j = length(cj);  off_j = orb_offsets[cj_idx]
        other_k = [k for k in 1:N if k != ci_idx && k != cj_idx]

        for (fock_ket, configs_ket) in ket.data
            na_i, nb_i = fock_ket[ci_idx]
            na_j, nb_j = fock_ket[cj_idx]

            # ---- α→β inter-cluster: create α at i, annihilate β at j ----
            if na_i < length(clusters[ci_idx]) && nb_j >= 1
                fock_bra = replace(fock_ket, [ci_idx, cj_idx],
                                   [(na_i+1, nb_i), (na_j, nb_j-1)])

                if haskey(bra.data, fock_bra)
                    configs_bra = bra.data[fock_bra]
                    ftrans_i = (fock_bra[ci_idx], fock_ket[ci_idx])
                    ftrans_j = (fock_bra[cj_idx], fock_ket[cj_idx])

                    if haskey(cluster_ops[ci_idx], "A") &&
                       haskey(cluster_ops[ci_idx]["A"], ftrans_i) &&
                       haskey(cluster_ops[cj_idx], "b") &&
                       haskey(cluster_ops[cj_idx]["b"], ftrans_j)

                        A_i        = cluster_ops[ci_idx]["A"][ftrans_i]
                        b_j        = cluster_ops[cj_idx]["b"][ftrans_j]
                        state_sign = _rdm_sign(fock_ket, ci_idx, cj_idx)

                        bra_by_other = Dict{Vector{Int16},
                            Vector{Tuple{Int16,Int16,MVector{R1,T}}}}()
                        for (config, coeff) in configs_bra
                            key = [config[k] for k in other_k]
                            push!(get!(bra_by_other, key,
                                       Tuple{Int16,Int16,MVector{R1,T}}[]),
                                  (config[ci_idx], config[cj_idx], coeff))
                        end
                        ket_by_other = Dict{Vector{Int16},
                            Vector{Tuple{Int16,Int16,MVector{R2,T}}}}()
                        for (config, coeff) in configs_ket
                            key = [config[k] for k in other_k]
                            push!(get!(ket_by_other, key,
                                       Tuple{Int16,Int16,MVector{R2,T}}[]),
                                  (config[ci_idx], config[cj_idx], coeff))
                        end

                        for (key, bra_list) in bra_by_other
                            haskey(ket_by_other, key) || continue
                            for (s_i, s_j, c_bra) in bra_list
                                for (t_i, t_j, c_ket) in ket_by_other[key]
                                    for r2 in 1:R2, r1 in 1:R1
                                        cp = state_sign * c_bra[r1] * c_ket[r2]
                                        iszero(cp) && continue
                                        for q in 1:norb_j
                                            bj = b_j[q, s_j, t_j]
                                            iszero(bj) && continue
                                            for p in 1:norb_i
                                                γ_ab_loc[tid][off_i+p, off_j+q, r1, r2] +=
                                                    A_i[p, s_i, t_i] * bj * cp
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end  # α→β inter

            # ---- β→α inter-cluster: create β at i, annihilate α at j ----
            if nb_i < length(clusters[ci_idx]) && na_j >= 1
                fock_bra = replace(fock_ket, [ci_idx, cj_idx],
                                   [(na_i, nb_i+1), (na_j-1, nb_j)])

                if haskey(bra.data, fock_bra)
                    configs_bra = bra.data[fock_bra]
                    ftrans_i = (fock_bra[ci_idx], fock_ket[ci_idx])
                    ftrans_j = (fock_bra[cj_idx], fock_ket[cj_idx])

                    if haskey(cluster_ops[ci_idx], "B") &&
                       haskey(cluster_ops[ci_idx]["B"], ftrans_i) &&
                       haskey(cluster_ops[cj_idx], "a") &&
                       haskey(cluster_ops[cj_idx]["a"], ftrans_j)

                        B_i        = cluster_ops[ci_idx]["B"][ftrans_i]
                        a_j        = cluster_ops[cj_idx]["a"][ftrans_j]
                        state_sign = _rdm_sign(fock_ket, ci_idx, cj_idx)

                        bra_by_other = Dict{Vector{Int16},
                            Vector{Tuple{Int16,Int16,MVector{R1,T}}}}()
                        for (config, coeff) in configs_bra
                            key = [config[k] for k in other_k]
                            push!(get!(bra_by_other, key,
                                       Tuple{Int16,Int16,MVector{R1,T}}[]),
                                  (config[ci_idx], config[cj_idx], coeff))
                        end
                        ket_by_other = Dict{Vector{Int16},
                            Vector{Tuple{Int16,Int16,MVector{R2,T}}}}()
                        for (config, coeff) in configs_ket
                            key = [config[k] for k in other_k]
                            push!(get!(ket_by_other, key,
                                       Tuple{Int16,Int16,MVector{R2,T}}[]),
                                  (config[ci_idx], config[cj_idx], coeff))
                        end

                        for (key, bra_list) in bra_by_other
                            haskey(ket_by_other, key) || continue
                            for (s_i, s_j, c_bra) in bra_list
                                for (t_i, t_j, c_ket) in ket_by_other[key]
                                    for r2 in 1:R2, r1 in 1:R1
                                        cp = state_sign * c_bra[r1] * c_ket[r2]
                                        iszero(cp) && continue
                                        for q in 1:norb_j
                                            aj = a_j[q, s_j, t_j]
                                            iszero(aj) && continue
                                            for p in 1:norb_i
                                                γ_ba_loc[tid][off_i+p, off_j+q, r1, r2] +=
                                                    B_i[p, s_i, t_i] * aj * cp
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end  # β→α inter

        end  # fock_ket loop
    end  # inter-cluster @threads

    γ_ab = sum(γ_ab_loc)
    γ_ba = sum(γ_ba_loc)
    return γ_ab, γ_ba
end


"""
    compute_1rdm_sf_threaded(psi, cluster_ops)

Single-state parallel spin-flip 1-RDM.
"""
function compute_1rdm_sf_threaded(psi::TPSCIstate{T,N,R},
                                   cluster_ops::Vector{ClusterOps{T}}) where {T,N,R}
    return compute_1rdm_sf_threaded(psi, psi, cluster_ops)
end

