using ActiveSpaceSolvers
using BlockDavidson


"""
    get_ortho_compliment(tss::ClusterSubspace, cb::ClusterBasis)

For a given `ClusterSubspace`, `tss`, return the subspace remaining
"""
function get_ortho_compliment(tss::ClusterSubspace, cb::ClusterBasis)
    #={{{=#
    data = OrderedDict{Tuple{UInt8,UInt8},UnitRange{Int}}()
    for (fock, basis) in cb

        if haskey(tss.data, fock)
            first(tss.data[fock]) == 1 || error(" p-space doesn't include ground state?")
            newrange = last(tss[fock])+1:size(cb[fock], 2)
            if length(newrange) > 0
                data[fock] = newrange
            end
        else
            newrange = 1:size(cb[fock], 2)
            if length(newrange) > 0
                data[fock] = newrange
            end
        end
    end

    return ClusterSubspace(tss.cluster, data)
    #=}}}=#
end



"""
    compute_cluster_ops(cluster_bases::Vector{ClusterBasis})
"""
function compute_cluster_ops(cluster_bases, ints::InCoreInts{T}; verbose=0) where {T}
    #={{{=#
    clusters = Vector{MOCluster}()
    for ci in cluster_bases
        push!(clusters, ci.cluster)
    end

    cluster_ops = Vector{ClusterOps{T}}()
    for ci in clusters
        push!(cluster_ops, ClusterOps(ci, T=T))
    end


    for ci in clusters

        verbose > 0 && display(ci)
        verbose > 0 && flush(stdout)

        cb = cluster_bases[ci.idx]

        cluster_ops[ci.idx]["H"] = FermiCG.tdm_H(cb, subset(ints, ci.orb_list), verbose=verbose > 1 ? 1 : 0)
        cluster_ops[ci.idx]["A"], cluster_ops[ci.idx]["a"] = FermiCG.tdm_A(cb, "alpha")
        cluster_ops[ci.idx]["B"], cluster_ops[ci.idx]["b"] = FermiCG.tdm_A(cb, "beta")
        cluster_ops[ci.idx]["AA"], cluster_ops[ci.idx]["aa"] = FermiCG.tdm_AA(cb, "alpha")
        cluster_ops[ci.idx]["BB"], cluster_ops[ci.idx]["bb"] = FermiCG.tdm_AA(cb, "beta")
        cluster_ops[ci.idx]["Aa"] = FermiCG.tdm_Aa(cb, "alpha")
        cluster_ops[ci.idx]["Bb"] = FermiCG.tdm_Aa(cb, "beta")
        cluster_ops[ci.idx]["Ab"], cluster_ops[ci.idx]["Ba"] = FermiCG.tdm_Ab(cb)
        # remove BA and ba account for these terms 
        cluster_ops[ci.idx]["AB"], cluster_ops[ci.idx]["ba"], cluster_ops[ci.idx]["BA"], cluster_ops[ci.idx]["ab"] = FermiCG.tdm_AB(cb)
        cluster_ops[ci.idx]["AAa"], cluster_ops[ci.idx]["Aaa"] = FermiCG.tdm_AAa(cb, "alpha")
        cluster_ops[ci.idx]["BBb"], cluster_ops[ci.idx]["Bbb"] = FermiCG.tdm_AAa(cb, "beta")
        cluster_ops[ci.idx]["ABa"], cluster_ops[ci.idx]["Aba"] = FermiCG.tdm_ABa(cb, "alpha")
        cluster_ops[ci.idx]["ABb"], cluster_ops[ci.idx]["Bba"] = FermiCG.tdm_ABa(cb, "beta")
        #cluster_ops[ci.idx]["ABa"], cluster_ops[ci.idx]["Aba"], cluster_ops[ci.idx]["BAa"], cluster_ops[ci.idx]["Aab"] = FermiCG.tdm_ABa(cb,"alpha")
        #cluster_ops[ci.idx]["ABb"], cluster_ops[ci.idx]["Bba"], cluster_ops[ci.idx]["BAb"], cluster_ops[ci.idx]["Bab"] = FermiCG.tdm_ABa(cb,"beta")

        # spin operators

        #
        # S+
        op = Dict{Tuple,Array}()
        for (fock, mat) in cluster_ops[ci.idx]["Ab"]
            dims = size(mat)
            op[fock] = zeros(dims[2:4]...)
            for j in 1:dims[4]
                for i in 1:dims[3]
                    for p in 1:dims[2]
                        op[fock][p, i, j] = mat[p, p, i, j]
                    end
                end
            end
        end
        cluster_ops[ci.idx]["S+"] = op

        #
        # S-
        op = Dict{Tuple,Array}()
        for (fock, mat) in cluster_ops[ci.idx]["Ba"]
            dims = size(mat)
            op[fock] = zeros(dims[2:4]...)
            for j in 1:dims[4]
                for i in 1:dims[3]
                    for p in 1:dims[2]
                        op[fock][p, i, j] = mat[p, p, i, j]
                    end
                end
            end
        end
        cluster_ops[ci.idx]["S-"] = op

        #
        # Sz
        op = Dict{Tuple,Array}()
        #
        # loop over fock-space transitions
        for (fock, basis) in cb
            focktrans = (fock, fock)

            sz = (fock[1] - fock[2]) / 2.0
            op[focktrans] = sz * Matrix(1.0I, size(cb[fock], 2), size(cb[fock], 2))
            op[focktrans] = reshape(op[focktrans], 1, size(op[focktrans], 1), size(op[focktrans], 2))

        end
        cluster_ops[ci.idx]["Sz"] = op


        #
        # S2
        cluster_ops[ci.idx]["S2"] = FermiCG.tdm_S2(cb, subset(ints, ci.orb_list), verbose=0)


        to_delete = [
        #"AAa",
        #"Aaa",
        #"BBb",
        #"Bbb",
        #
        #"ABa",
        #"Aba",
        ##"BAa",
        ##"Aab",
        #
        #"ABb",
        #"Bba",
        ##"BAb",
        ##"Bab",
        #"Aa",
        #"Bb",
        #"Ab",
        #"Ba",
        #"AB",
        #"ba",
        #"BA",
        #"ab",
        #"AA",
        #"BB",
        #"aa",
        #"bb"
        ]
        for op in to_delete
            for (ftran, array) in cluster_ops[ci.idx][op]
                cluster_ops[ci.idx][op][ftran] .*= 0
            end
        end


        # Compute single excitation operator
        tmp = Dict{Tuple,Array}()
        for (fock, basis) in cb
            tmp[(fock, fock)] = (cluster_ops[ci.idx]["Aa"][(fock, fock)] + cluster_ops[ci.idx]["Bb"][(fock, fock)])
        end
        cluster_ops[ci.idx]["E1"] = tmp



        #
        # reshape data into 3index quantities: e.g., (pqr, I, J)
        for opstring in keys(cluster_ops[ci.idx])
            opstring != "H" || continue
            opstring != "S2" || continue
            for ftrans in keys(cluster_ops[ci.idx][opstring])
                data = cluster_ops[ci.idx][opstring][ftrans]
                dim1 = prod(size(data)[1:(length(size(data))-2)])
                dim2 = size(data)[length(size(data))-1]
                dim3 = size(data)[length(size(data))-0]
                cluster_ops[ci.idx][opstring][ftrans] = copy(reshape(data, (dim1, dim2, dim3)))
            end
        end
    end
    return cluster_ops
end
#=}}}=#


"""
    tdm_H(cb::ClusterBasis; verbose=0)

Compute local Hamiltonian `<s|H|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_H(cb::ClusterBasis, ints; verbose=0)
    #={{{=#
    verbose > 1 && println("")
    verbose > 1 && display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for (fock, basis) in cb
        focktrans = (fock, fock)
        verbose > 1 && display(basis.ansatz)
        Hmap = LinearMap(ints, basis.ansatz)
        dicti[focktrans] = cb[fock]' * Matrix((Hmap * cb[fock]))
        if verbose > 0
            for e in 1:size(cb[fock], 2)
                @printf(" %4i %12.8f\n", e, dicti[focktrans][e, e])
            end
        end
    end
    return dicti
    #=}}}=#
end


"""
"""
function tdm_S2(cb::ClusterBasis, ints; verbose=0)
    #={{{=#
    verbose > 1 && println("")
    verbose > 1 && display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for (fock, basis) in cb
        focktrans = (fock, fock)
        verbose > 1 && display(basis.ansatz)

        dicti[focktrans] = cb[fock]' * apply_S2_matrix(basis.ansatz, cb[fock].vectors)

        if verbose > 0
            for e in 1:size(cb[fock], 2)
                @printf(" %4i %12.8f\n", e, dicti[focktrans][e, e])
            end
        end
    end
    return dicti
    #=}}}=#
end


"""
    tdm_A(cb::ClusterBasis; verbose=0)

Compute `<s|p'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_A(cb::ClusterBasis, spin_case; verbose=0)
    #={{{=#
    verbose > 1 && println("")
    verbose > 1 && display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na + 1, nb)
            elseif spin_case == "beta"
                fockbra = (na, nb + 1)
            else
                throw(DomainError(spin_case))
            end
            fockket = (na, nb)
            focktrans = (fockbra, fockket)
            focktrans_adj = (fockket, fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                if spin_case == "alpha"
                    dicti[focktrans] = compute_operator_c_a(basis_bra, basis_ket)
                else
                    dicti[focktrans] = compute_operator_c_b(basis_bra, basis_ket)
                end
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] = permutedims(dicti[focktrans], [1, 3, 2])
            end
        end
    end
    return dicti, dicti_adj
    #=}}}=#
end


"""
    tdm_AA(cb::ClusterBasis; verbose=0)

Compute `<s|p'q'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AA(cb::ClusterBasis, spin_case; verbose=0)
    #={{{=#
    verbose > 1 && println("")
    verbose > 1 && display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na + 2, nb)
            elseif spin_case == "beta"
                fockbra = (na, nb + 2)
            else
                throw(DomainError(spin_case))
            end

            fockket = (na, nb)
            focktrans = (fockbra, fockket)
            focktrans_adj = (fockket, fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                if spin_case == "alpha"
                    dicti[focktrans] = compute_operator_cc_aa(basis_bra, basis_ket)
                else
                    dicti[focktrans] = compute_operator_cc_bb(basis_bra, basis_ket)
                end
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] = permutedims(dicti[focktrans], [2, 1, 4, 3])
            end
        end
    end
    return dicti, dicti_adj
    #=}}}=#
end


"""
    tdm_Aa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_Aa(cb::ClusterBasis, spin_case; verbose=0)
    #={{{=#
    verbose > 1 && println("")
    verbose > 1 && display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = (na, nb)

            fockket = (na, nb)
            focktrans = (fockbra, fockket)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                if spin_case == "alpha"
                    dicti[focktrans] = compute_operator_ca_aa(basis_bra, basis_ket)
                else
                    dicti[focktrans] = compute_operator_ca_bb(basis_bra, basis_ket)
                end
            end
        end
    end
    return dicti
    #=}}}=#
end


"""
    tdm_Ab(cb::ClusterBasis; verbose=0)

Compute `<s|p'q|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space, where
`p'` is alpha and `q` is beta.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_Ab(cb::ClusterBasis; verbose=0)
    #={{{=#
    verbose > 1 && println("")
    verbose > 1 && display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in -1:norbs+1
        for nb in -1:norbs+1
            fockbra = (na + 1, nb - 1)

            fockket = (na, nb)
            focktrans = (fockbra, fockket)
            focktrans_adj = (fockket, fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                dicti[focktrans] = compute_operator_ca_ab(basis_bra, basis_ket)

                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] = permutedims(dicti[focktrans], [2, 1, 4, 3])
            end
        end
    end
    return dicti, dicti_adj
    #=}}}=#
end


"""
    tdm_AB(cb::ClusterBasis; verbose=0)

Compute `<s|p'q'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space, where
`p'` is alpha and `q'` is beta.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AB(cb::ClusterBasis; verbose=0)
    #={{{=#
    verbose > 1 && println("")
    verbose > 1 && display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    dictj = Dict{Tuple,Array}()
    dictj_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in -2:norbs+2
        for nb in -2:norbs+2
            fockbra = (na + 1, nb + 1)

            fockket = (na, nb)
            focktrans = (fockbra, fockket)
            focktrans_adj = (fockket, fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                dicti[focktrans] = compute_operator_cc_ab(basis_bra, basis_ket)
                dictj[focktrans] = -permutedims(dicti[focktrans], [2, 1, 3, 4])

                # adjoint 
                dicti_adj[focktrans_adj] = permutedims(dicti[focktrans], [2, 1, 4, 3])
                dictj_adj[focktrans_adj] = permutedims(dictj[focktrans], [2, 1, 4, 3])
            end
        end
    end
    return dicti, dicti_adj, dictj, dictj_adj
    #=}}}=#
end


"""
    tdm_AAa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q'r|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AAa(cb::ClusterBasis, spin_case; verbose=0)
    #={{{=#
    verbose > 1 && println("")
    verbose > 1 && display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na + 1, nb)
            elseif spin_case == "beta"
                fockbra = (na, nb + 1)
            else
                throw(DomainError(spin_case))
            end

            fockket = (na, nb)
            focktrans = (fockbra, fockket)
            focktrans_adj = (fockket, fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                if spin_case == "alpha"
                    dicti[focktrans] = compute_operator_cca_aaa(basis_bra, basis_ket)
                else
                    dicti[focktrans] = compute_operator_cca_bbb(basis_bra, basis_ket)
                end
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] = permutedims(dicti[focktrans], [3, 2, 1, 5, 4])
            end
        end
    end
    return dicti, dicti_adj
    #=}}}=#
end


"""
    tdm_ABa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q'r|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_ABa(cb::ClusterBasis, spin_case; verbose=0)
    #={{{=#
    verbose > 1 && println("")
    verbose > 1 && display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    dictj = Dict{Tuple,Array}()
    dictj_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in -2:norbs+2
        for nb in -2:norbs+2
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na, nb + 1)
            elseif spin_case == "beta"
                fockbra = (na + 1, nb)
            else
                throw(DomainError(spin_case))
            end

            fockket = (na, nb)
            focktrans = (fockbra, fockket)
            focktrans_adj = (fockket, fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                if spin_case == "alpha"
                    dicti[focktrans] = compute_operator_cca_aba(basis_bra, basis_ket)
                    dictj[focktrans] = -permutedims(dicti[focktrans], [2, 1, 3, 4, 5])
                elseif spin_case == "beta"
                    dicti[focktrans] = compute_operator_cca_abb(basis_bra, basis_ket)
                    dictj[focktrans] = -permutedims(dicti[focktrans], [2, 1, 3, 4, 5])
                else
                    error("Wrong spin_case: ", spin_case)
                end

                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]

                dicti_adj[focktrans_adj] = permutedims(dicti[focktrans], [3, 2, 1, 5, 4])
                dictj_adj[focktrans_adj] = permutedims(dictj[focktrans], [3, 2, 1, 5, 4])
            end
        end
    end
    return dicti, dicti_adj, dictj, dictj_adj
    #=}}}=#
end



"""
    function add_cmf_operators!(ops::Vector{ClusterOps}, bases::Vector{ClusterBasis}, ints, Da, Db; verbose=0)

Add effective local hamiltonians (local CASCI) type hamiltonians to a `ClusterOps` type for each `Cluster'
"""
function add_cmf_operators!(ops, bases, ints, Da, Db; verbose=0)
    #={{{=#
    n_clusters = length(bases)
    for ci_idx in 1:n_clusters
        cb = bases[ci_idx]
        ci = cb.cluster
        verbose > 0 && println()
        verbose > 0 && display(ci)
        norbs = length(cb.cluster)

        ints_i = subset(ints, ci.orb_list, Da, Db)
        #ints_i = form_casci_ints(ints, ci, Da, Db)


        dicti = Dict{Tuple,Array}()

        #
        # loop over fock-space transitions
        for (fock, basis) in cb
            focktrans = (fock, fock)
            verbose > 1 && display(basis.ansatz)
            Hmap = LinearMap(ints_i, basis.ansatz)

            dicti[focktrans] = cb[fock]' * Matrix((Hmap * cb[fock]))

            if verbose > 0
                for e in 1:size(cb[fock], 2)
                    @printf(" %4i %12.8f\n", e, dicti[focktrans][e, e])
                end
            end
        end
        ops[ci.idx]["Hcmf"] = dicti
    end
    return
end
#=}}}=#






"""
    compute_cluster_eigenbasis_spin(   ints::InCoreInts{T}, 
                                       clusters::Vector{MOCluster}, 
                                       rdm1::RDM1{T},
                                       delta_elec::Vector,
                                       ref_fock::FockConfig; 
                                       verbose=0, 
                                       max_roots=10, 
                                       A::Type=FCIAnsatz) where T

Return a Vector of `ClusterBasis` for each `Cluster`.
For each number of electrons specified by ref_fock +- 1->delta_elec (for each cluster), 
we solve the CASCI problem, collecting `max_roots` of the lowest energy eigenvectors for the half-filled (or of odd number nalpha = nbeta+1) level. Then we apply S^- and S^+ to generate the higher/lower m_s blocks directly. 

# Arguments
#
- `ints`: InCoreInts integrals
- `clusters`: Clusters 
- `verbose`: Print level
- `ref_fock`:  reference space for defining target focksectors with `delta_elec`
- `delta_elec`: number of electrons different from reference (init_fspace) for each cluster
- `max_roots::Int`: Maximum number of vectors for each focksector basis
- `rdm1`: background density matrix for embedding local hamiltonian 
- `A`: the type of Ansatz object used to solve each cluster. Default is FCIAnsatz     
- `T`: Data type of the eigenvectors 
"""
function compute_cluster_eigenbasis_spin(ints::InCoreInts{T},
    clusters::Vector{MOCluster},
    rdm1::RDM1{T},
    delta_elec::Vector,
    ref_fock::FockConfig;
    verbose=0,
    max_roots=10,
    A::Type=FCIAnsatz) where {T}
    #={{{=#
    # initialize output
    #
    cluster_bases = Vector{ClusterBasis{A,T}}()

    length(delta_elec) == length(clusters) || error("length(delta_elec) != length(clusters)")
    for ci in clusters
        verbose > 1 && display(ci)

        ints_i = subset(ints, ci, rdm1)


        #
        # Verify that density matrix provided is consistent with reference fock sectors
        occs = diag(rdm1.a)
        occs[ci.orb_list] .= 0
        na_embed = sum(occs)
        occs = diag(rdm1.b)
        occs[ci.orb_list] .= 0
        nb_embed = sum(occs)
        verbose > 1 && @printf(" Number of embedded electrons a,b: %f %f
", na_embed, nb_embed)


        delta_e_i = delta_elec[ci.idx]

        #
        # Get list of Fock-space sectors for current cluster
        #
        ni = ref_fock[ci.idx][1] + ref_fock[ci.idx][2]  # number of electrons in ci
        sectors = []
        max_e = 2 * length(ci)
        min_e = 0
        for nj in ni-delta_e_i:ni+delta_e_i

            nj <= max_e || continue
            nj >= min_e || continue

            naj = nj ÷ 2 + nj % 2
            nbj = nj ÷ 2
            push!(sectors, (naj, nbj))
        end

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci, T=T)
        for sec in sectors

            #
            # prepare for FCI calculation for give sector of Fock space
            ansatz = FCIAnsatz(length(ci), sec[1], sec[2])
            verbose > 1 && @printf(" Preparing to compute : 
")
            verbose > 1 && display(ansatz)
            verbose > 1 && flush(stdout)

            nr = min(max_roots, ansatz.dim)

            if ansatz.dim < 500 || ansatz.dim == nr
                #
                # Build full Hamiltonian matrix in cluster's Slater Det basis
                Hmat = build_H_matrix(ints_i, ansatz)
                F = eigen(Hmat)

                basis_i[sec] = Solution(ansatz, F.values[1:nr], F.vectors[:, 1:nr])

                #display(e)
            else
                #
                # Do sparse build 
                basis_i[sec] = solve(ints_i, ansatz, SolverSettings(nroots=nr, package="arpack"))
            end

            #
            # Loop over spin-flips
            # 
            # s2 = s(s+1) 


            s2 = compute_s2(basis_i[sec])

            nr = length(basis_i[sec].energies)
            #for r in 1:nr
            #    S = (-1 + sqrt(1+4*s2[r]))/2
            #    gr = 2*S+1 # Degeneracy
            #end

            #
            #   S-
            #
            # find how many applications of S- we need to try

            verbose > 1 && println(" Compute higher and lower Ms components")
            n_sm = minimum((sec[1], ansatz.no - sec[2]))
            vi = deepcopy(basis_i[sec].vectors)
            ansatzi = deepcopy(basis_i[sec].ansatz)
            for smi in 1:n_sm
                vi, ansatzi = apply_sminus(vi, ansatzi)

                verbose > 1 && display(ansatzi)
                flush(stdout)

                if size(vi, 2) == 0
                    # we have killed all the spin states
                    continue
                end

                Hmapi = LinearMap(ints_i, ansatzi)
                ei = diag(vi' * Matrix(Hmapi * vi))
                #ei = compute_energy(vi, ansatzi)

                si = Solution(ansatzi, ei, vi)
                seci = (ansatzi.na, ansatzi.nb)
                basis_i[seci] = si
            end
            #
            #   S+
            #
            # find how many applications of S+ we need to try

            n_sp = minimum((sec[2], ansatz.no - sec[1]))
            vi = deepcopy(basis_i[sec].vectors)
            ansatzi = deepcopy(basis_i[sec].ansatz)
            for spi in 1:n_sp
                vi, ansatzi = apply_splus(vi, ansatzi)

                verbose > 1 && display(ansatzi)
                flush(stdout)

                if size(vi, 2) == 0
                    # we have killed all the spin states
                    continue
                end

                Hmapi = LinearMap(ints_i, ansatzi)
                ei = diag(vi' * Matrix(Hmapi * vi))
                #ei = compute_energy(vi, ansatzi)

                si = Solution(ansatzi, ei, vi)
                seci = (ansatzi.na, ansatzi.nb)
                basis_i[seci] = si
            end

        end

        flush(stdout)
        if verbose > 0
            println()
            for (sec, sol) in basis_i
                println()
                display(sol.ansatz)
                s2 = compute_s2(sol)
                for i in 1:length(sol.energies)
                    @printf("   State %4i Energy: %12.8f S2: %12.8f
", i, sol.energies[i], s2[i])
                end
                flush(stdout)
            end
        end

        push!(cluster_bases, basis_i)
    end
    return cluster_bases
end
#=}}}=#


"""
    compute_cluster_eigenbasis(ints::InCoreInts, clusters::Vector{MOCluster}; 
        init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
        rdm1a=nothing, rdm1b=nothing, T::Type=Float64)

Return a Vector of `ClusterBasis` for each `Cluster` 
- `ints::InCoreInts`: In-core integrals
- `clusters::Vector{MOCluster}`: Clusters 
- `verbose::Int`: Print level
- `init_fspace`: list of pairs of (nα,nβ) for each cluster for defining reference space
                 for selecting out only certain fock sectors
- `delta_elec`: number of electrons different from reference (init_fspace)
- `max_roots::Int`: Maximum number of vectors for each focksector basis
- `rdm1a`: background density matrix for embedding local hamiltonian (alpha)
- `rdm1b`: background density matrix for embedding local hamiltonian (beta)
- `ansatze`: should be a list of Ansatz objects so that we know how to solve each cluster. Default is FCIAnsatz     
- `T`: Data type of the eigenvectors 
"""
function compute_cluster_eigenbasis(ints::InCoreInts, clusters::Vector{MOCluster};
    init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10,
    rdm1a=nothing, rdm1b=nothing,
    ansatze=nothing,
    T::Type=Float64, A::Type=FCIAnsatz)
    #={{{=#
    # initialize output
    #
    cluster_bases = Vector{ClusterBasis{A,T}}()

    for ci in clusters
        verbose == 0 || display(ci)

        if (rdm1a !== nothing && init_fspace == nothing)
            error(" Cant embed without init_fspace")
        end

        #
        # Get subset of integrals living on cluster, ci
        if rdm1a === nothing && rdm1b === nothing
            ints_i = subset(ints, ci.orb_list)
        else
            ints_i = subset(ints, ci.orb_list, rdm1a, rdm1b)
        end


        if all((rdm1a, rdm1b, init_fspace) .!= nothing)
            # 
            # Verify that density matrix provided is consistent with reference fock sectors
            occs = diag(rdm1a)
            occs[ci.orb_list] .= 0
            na_embed = sum(occs)
            occs = diag(rdm1b)
            occs[ci.orb_list] .= 0
            nb_embed = sum(occs)
            verbose == 0 || @printf(" Number of embedded electrons a,b: %f %f", na_embed, nb_embed)
        end

        delta_e_i = ()
        if all((delta_elec, init_fspace) .!= nothing)
            delta_e_i = (init_fspace[ci.idx][1], init_fspace[ci.idx][2], delta_elec)
        end

        #
        # Get list of Fock-space sectors for current cluster
        #
        sectors = possible_focksectors(ci, delta_elec=delta_e_i)

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci, T=T)
        for sec in sectors

            #
            # prepare for FCI calculation for give sector of Fock space
            ansatz = FCIAnsatz(length(ci), sec[1], sec[2])
            verbose == 0 || display(ansatz)
            verbose == 0 || flush(stdout)

            nr = min(max_roots, ansatz.dim)

            if ansatz.dim < 500 || ansatz.dim == nr
                #
                # Build full Hamiltonian matrix in cluster's Slater Det basis
                Hmat = build_H_matrix(ints_i, ansatz)
                F = eigen(Hmat)

                basis_i[sec] = Solution(ansatz, Vector{T}(F.values[1:nr]), Matrix{T}(F.vectors[:, 1:nr]))
                #display(e)
            else
                #
                # Do sparse build 
                #if ansatz.dim > 3000
                #    display(norm(ints_i.h1))
                #    display(norm(ints_i.h2))
                #end
                basis_i[sec] = solve(ints_i, ansatz, SolverSettings(nroots=nr))
            end
            if verbose > 0
                state = 1
                for ei in basis_i[sec].energies
                    @printf("   State %4i Energy: %12.8f %12.8f
", state, ei, ei + ints.h0)
                    state += 1
                end
                flush(stdout)
            end
        end
        push!(cluster_bases, basis_i)
    end
    return cluster_bases
end

"""
    extend_fock_sectors(sectors, delta_elec, max_alpha, max_beta)
Given a list of fock sectors (n_alpha, n_beta), extend the list by adding all sectors
that are within delta_elec of each sector, while ensuring that the new sectors
are within the bounds of (0, max_alpha) and (0, max_beta).
"""
function extend_fock_sectors(sectors, delta_elec, max_alpha, max_beta)
    extended_sectors = Set{Tuple{Int, Int}}()
    for (n_alpha, n_beta) in sectors
        push!(extended_sectors, (n_alpha, n_beta))
        for d_alpha in -delta_elec:delta_elec
            for d_beta in -delta_elec:delta_elec
                new_alpha = n_alpha + d_alpha
                new_beta = n_beta + d_beta
                if 0 <= new_alpha <= max_alpha && 0 <= new_beta <= max_beta
                    push!(extended_sectors, (new_alpha, new_beta))
                end
            end
        end
    end
    return collect(extended_sectors)
end

function form_casci_eff_ints(ints2, active_orbitals::Vector{Int})
    # Extract effective integrals on active orbital subset
    h0_eff = ints2.h0
    h1_eff = ints2.h1[active_orbitals, active_orbitals]
    h2_eff = ints2.h2[active_orbitals, active_orbitals, active_orbitals, active_orbitals]

    # Return a new IntegralData struct with effective integrals
    return InCoreInts(h0_eff, h1_eff, h2_eff)
end


"""
    _diag_init_guess(ints, ansatz, nr)

Build a Davidson initial guess using the Fock diagonal: unit vectors for
the `nr` determinants with lowest sum of h1 orbital energies.  This is
O(dim·ne) — much cheaper than sampling full H diagonal elements via matvecs.
For CMF-optimized orbitals the h1 diagonal are the CMF orbital energies, so
the lowest-Fock determinant is typically very close to the CMF reference.
"""
function _diag_init_guess(ints::InCoreInts, ansatz::FCIAnsatz, nr::Int)
    n = ansatz.dim
    orb_energies = diag(ints.h1)
    d = ActiveSpaceSolvers.FCI.compute_fock_diagonal(ansatz, orb_energies, 0.0)
    idx = sortperm(d)[1:min(nr, n)]
    v0 = zeros(n, nr)
    for (col, row) in enumerate(idx)
        v0[row, col] = 1.0
    end
    return v0
end


"""
    _merge_svd_basis!(basis, sol, norbs_frag, norbs_bath, thresh, verbose)

Helper: SVD-decompose `sol` (a Solution in fragment+bath space), project to fragment,
accumulate columns into `basis` dict, and orthogonalize per Fock sector.
"""
function _merge_svd_basis!(basis::OrderedDict, sol, norbs_frag::Int, norbs_bath::Int,
                           thresh::Real, verbose::Int)
    new_b = ActiveSpaceSolvers.svd_state_project_S2(sol, norbs_frag, norbs_bath, thresh, verbose=verbose)
    for (fock, vecs) in new_b
        if haskey(basis, fock)
            combined = hcat(basis[fock], vecs)
            F = svd(combined, full=false)
            keep = F.S .> 1e-8
            basis[fock] = F.U[:, keep]
        else
            # Orthonormalize the block itself (it should already be, but be safe)
            F = svd(vecs, full=false)
            keep = F.S .> 1e-8
            basis[fock] = F.U[:, keep]
        end
    end
end


"""
    form_est_spinbasis(ints, ci, Da, Db; kwargs...)

Generate spin-pure cluster basis states by applying
S+/S- ladder operators in the fragment+bath space BEFORE the Schmidt (SVD) decomposition.
This eliminates spin contamination while retaining the entangled-bath quality of EST states.

Two paths depending on fragment+bath FCI dimension vs `max_fci_dim`:

- **Main path** (dim ≤ max_fci_dim): solves FCI in fragment+bath via Davidson, applies S-
  and S+ chains, then SVD-decomposes each spin-pure state to extract fragment basis vectors.

- **Fallback** (dim > max_fci_dim): contracts bath orbitals back into the fragment via a
  second mean-field embedding (using the CMF density in the Schmidt-rotated basis), then
  solves fragment-only FCI + S+/S-. Less entanglement than the main path but exact spin
  eigenstates and tractable for large clusters (≤ 12 fragment orbitals).

# Arguments
- `ints`: InCoreInts integrals
- `ci`: MOCluster
- `Da`, `Db`: CMF alpha/beta 1RDMs
- `thresh_schmidt`: SVD threshold for keeping Schmidt vectors
- `thresh_orb`: threshold for counting bath orbitals from exchange-matrix SVD
- `max_bath`: maximum number of bath orbitals to keep. Default (`nothing`) caps bath at cluster size
             (fragment+bath ≤ 2×cluster). Pass an integer to override, e.g. `max_bath=10`.
- `target_norb`: if set, caps bath so total fragment+bath = target_norb orbitals.
                 Takes precedence over max_bath.
- `do_embedding`: whether to include mean-field environment embedding
- `verbose`: 0 = silent, 1 = summary, 2 = matrices
- `eig_nr`: number of Davidson roots in fragment+bath FCI per sector
- `max_iter_davidson`: Davidson max iterations
- `max_fci_dim`: per-sector FCI dim cap. When a sector exceeds this, bath orbitals are
                 peeled off one at a time (weakest first) until the dim fits. Never falls
                 back to fragment-only FCI.
- `per_sector`: if true, run a separate FCI+S±+SVD for each target cluster Fock sector
               (same coverage as compute_cluster_eigenbasis_spin but with bath orbitals).
               Requires `target_sectors` to be provided.
- `target_sectors`: list of (na_frag, nb_frag) cluster Fock sectors to process when `per_sector=true`
- `apply_spin_ladder`: if true (default), apply S+/S- chains after FCI to generate full spin multiplet.
                       Set false to skip ladder operators (FCI+SVD only, no spin-sector extrapolation).
"""
function form_est_spinbasis(ints::InCoreInts{T},
                            ci::MOCluster,
                            Da,
                            Db;
                            thresh_schmidt=1e-3,
                            thresh_orb=1e-8,
                            max_bath=nothing,
                            target_norb=nothing,
                            thresh_ci=1e-6,
                            do_embedding=true,
                            verbose=0,
                            eig_nr=1,
                            max_iter_davidson=200,
                            max_fci_dim=50_000_000,
                            per_sector=false,
                            target_sectors=nothing,
                            apply_spin_ladder=true,
                            A::Type=FCIAnsatz) where {T}
    #={{{=#

    # ---- embedding setup: exchange matrix → bath orbitals → effective integrals ----
    if verbose > 0
        println()
        println("------------------------------------------------------------")
        @printf("Form EST Spin Basis for Cluster %4i\n", ci.idx)
    end
    D = Da + Db

    K = zeros(size(ints.h1))
    @tensor begin
        K[q, r] = ints.h2[p, q, r, s] * D[p, s]
    end

    no = size(ints.h1, 1)
    ci_no = length(ci.orb_list)

    na_tot = Int(round(tr(Da)))
    nb_tot = Int(round(tr(Db)))
    if verbose > 1
        println(" Number of electrons in full system:")
        @printf(" α: %12.8f  β:%12.8f \n ", na_tot, nb_tot)
    end

    active = ci.orb_list
    backgr = Vector{Int}()
    for i in 1:no
        if !(i in active)
            append!(backgr, i)
        end
    end

    K2 = zeros((ci_no, no - ci_no))
    for (pi, p) in enumerate(active)
        for (qi, q) in enumerate(backgr)
            K2[pi, qi] = K[p, q]
        end
    end

    F = svd(K2, full=true)
    verbose > 1 && @printf("\nSing. Val.\n")
    nkeep = 0
    for si in F.S
        verbose > 1 && @printf("%16.12f\n", si)
        if si > thresh_orb
            nkeep += 1
        end
    end
    if target_norb !== nothing
        nkeep = min(nkeep, target_norb - ci_no)
    elseif max_bath !== nothing
        nkeep = min(nkeep, max_bath)
    else
        nkeep = min(nkeep, ci_no)   # default: bath ≤ cluster size (fragment+bath ≤ 2×cluster)
    end
    nkeep = max(nkeep, 0)
    verbose > 0 && @printf(" Bath orbitals kept: %i (norb_frag=%i, norb_fb=%i)\n", nkeep, ci_no, ci_no + nkeep)

    C = zeros(size(ints.h1))
    for (pi, p) in enumerate(active)
        for (qi, q) in enumerate(active)
            if pi == qi
                C[p, qi] = 1
            end
        end
    end
    for (pi, p) in enumerate(backgr)
        for (qi, q) in enumerate(backgr)
            C[p, qi+length(active)] = F.Vt[qi, pi]
        end
    end

    Cfrag = C[:, 1:ci_no]
    Cbath = C[:, ci_no+1:ci_no+nkeep]
    Cenvt = C[:, ci_no+nkeep+1:end]

    K2  = C' * K * C
    Da2 = C' * Da * C
    Db2 = C' * Db * C

    na = tr(Da2[1:ci_no+nkeep, 1:ci_no+nkeep])
    nb = tr(Db2[1:ci_no+nkeep, 1:ci_no+nkeep])
    if verbose > 1
        println(" Number of electrons in Fragment+Bath system:")
        @printf("  α: %12.8f  β:%12.8f \n ", na, nb)
    end

    denvt_a = Cenvt * Cenvt' * Da * Cenvt * Cenvt'
    denvt_b = Cenvt * Cenvt' * Db * Cenvt * Cenvt'

    na_envt = Int(round(tr(Cenvt' * Da * Cenvt)))
    nb_envt = Int(round(tr(Cenvt' * Db * Cenvt)))
    if verbose > 1
        println(" Number of electrons in Environment system (rounded):")
        @printf("  α: %12.8f  β:%12.8f \n ", na_envt, nb_envt)
    end

    denvt_a = C' * denvt_a * C
    denvt_b = C' * denvt_b * C
    ints2 = orbital_rotation(ints, C)

    denvt_a[abs.(denvt_a).<1e-15] .= 0
    denvt_b[abs.(denvt_b).<1e-15] .= 0

    if do_embedding && size(Cenvt, 2) > 0
        EIG = eigen(denvt_a)
        U   = EIG.vectors[:, sortperm(EIG.values, rev=true)]
        denvt_a = U[:, 1:na_envt] * U[:, 1:na_envt]'

        EIG = eigen(denvt_b)
        U   = EIG.vectors[:, sortperm(EIG.values, rev=true)]
        denvt_b = U[:, 1:nb_envt] * U[:, 1:nb_envt]'
    elseif !do_embedding
        denvt_a *= 0
        denvt_b *= 0
    end

    no_range = collect(1:size(Cfrag, 2)+size(Cbath, 2))

    ints_f = if do_embedding
        subset(ints2, no_range, denvt_a, denvt_b)
    else
        form_casci_eff_ints(ints2, no_range)
    end

    na_actv = na_tot - na_envt
    nb_actv = nb_tot - nb_envt
    if verbose > 1
        println(" Number of active electrons in Fragment+Bath:")
        @printf("  α: %12.8f  β:%12.8f \n ", na_actv, nb_actv)
    end

    norb2 = size(ints_f.h1, 1)

    # Fragment electron counts from CMF density (needed for per_sector bath adjustment)
    na_frag_ref = Int(round(tr(Da2[1:ci_no, 1:ci_no])))
    nb_frag_ref = Int(round(tr(Db2[1:ci_no, 1:ci_no])))
    na_bath = na_actv - na_frag_ref   # bath electrons at CMF reference
    nb_bath = nb_actv - nb_frag_ref

    ansatz_ref = FCIAnsatz(norb2, na_actv, nb_actv)

    # Helper: FCI in fragment+bath at a given (na_fb, nb_fb) electron count,
    # then S±+SVD, accumulating into basis dict.
    # If the FCI dim exceeds max_fci_dim, peels off the weakest bath orbitals one by one
    # (folded into mean-field via subset) until the dim fits. Never falls back to fragment-only.
    function _fci_spin_svd!(basis, na_fb, nb_fb)
        (0 <= na_fb <= norb2 && 0 <= nb_fb <= norb2) || return

        # Adaptively reduce bath if dim too large
        ints_fb = ints_f
        norb_fb = norb2
        na_fb_eff = na_fb
        nb_fb_eff = nb_fb
        nkeep_eff = nkeep
        while true
            ansatz_try = FCIAnsatz(norb_fb, na_fb_eff, nb_fb_eff)
            ansatz_try.dim == 0 && return
            ansatz_try.dim <= max_fci_dim && break
            nkeep_eff == 0 && break   # can't reduce further; proceed with fragment-only FCI
            # fold the weakest bath orbital (last in ints_fb) into mean-field
            drop_idx = nkeep_eff  # 1-based index of the bath orbital to drop (last = weakest)
            # drop the last (weakest) bath orbital: fold it into mean-field
            drop_na = Int(round(Da2[ci_no + drop_idx, ci_no + drop_idx]))
            drop_nb = Int(round(Db2[ci_no + drop_idx, ci_no + drop_idx]))
            drop_rdm_a = zeros(T, norb_fb, norb_fb)
            drop_rdm_b = zeros(T, norb_fb, norb_fb)
            drop_rdm_a[ci_no + drop_idx, ci_no + drop_idx] = Da2[ci_no + drop_idx, ci_no + drop_idx]
            drop_rdm_b[ci_no + drop_idx, ci_no + drop_idx] = Db2[ci_no + drop_idx, ci_no + drop_idx]
            keep_orbs = collect(vcat(1:ci_no, ci_no+1:ci_no+nkeep_eff-1))
            ints_fb = subset(ints_fb, keep_orbs, drop_rdm_a, drop_rdm_b)
            na_fb_eff -= drop_na
            nb_fb_eff -= drop_nb
            nkeep_eff -= 1
            norb_fb = ci_no + nkeep_eff
        end

        ansatz_i = FCIAnsatz(norb_fb, na_fb_eff, nb_fb_eff)
        ansatz_i.dim == 0 && return
        nkeep_used = nkeep_eff
        if nkeep_used < nkeep
            verbose > 0 && @printf(" Reduced bath %i→%i for sector (na=%i,nb=%i), dim=%i\n",
                                    nkeep, nkeep_used, na_fb_eff, nb_fb_eff, ansatz_i.dim)
        end
        nr_i = min(eig_nr, ansatz_i.dim)
        e_i, v_i = if ansatz_i.dim < 2000 || nr_i * 4 > ansatz_i.dim
            verbose > 0 && @printf(" FCI sector (na_fb=%i, nb_fb=%i, dim=%i, nr=%i) → direct\n",
                                    na_fb_eff, nb_fb_eff, ansatz_i.dim, nr_i)
            Hmat = build_H_matrix(ints_fb, ansatz_i)
            F = eigen(Hmat)
            F.values[1:nr_i], F.vectors[:, 1:nr_i]
        else
            verbose > 0 && @printf(" FCI sector (na_fb=%i, nb_fb=%i, dim=%i, nr=%i) → Davidson\n",
                                    na_fb_eff, nb_fb_eff, ansatz_i.dim, nr_i)
            Hmap_i = LinearMap(ints_fb, ansatz_i)
            v0 = _diag_init_guess(ints_fb, ansatz_i, nr_i)
            dav = FermiCG.Davidson(Hmap_i, v0=v0, max_iter=max_iter_davidson,
                                   max_ss_vecs=max(20, 4*nr_i), nroots=nr_i, tol=1e-8)
            e_dav, v_dav = BlockDavidson.eigs(dav, iprint=verbose)
            real.(e_dav), v_dav
        end
        _merge_svd_basis!(basis, Solution(ansatz_i, e_i, v_i), length(active), nkeep_used, thresh_schmidt, verbose)
        if apply_spin_ladder
            # S- chain: guard against nb+1 > no (apply_sminus condition)
            vi = v_i; ansatzi = ansatz_i; smi = 0
            while ansatzi.na > 0 && ansatzi.nb + 1 <= ansatzi.no
                vi, ansatzi = apply_sminus(vi, ansatzi)
                size(vi, 2) == 0 && break
                smi += 1
                verbose > 0 && @printf("  S- applied (%i): fb sector (%i,%i)\n", smi, ansatzi.na, ansatzi.nb)
                Hmapi = LinearMap(ints_fb, ansatzi)
                ei = diag(vi' * Matrix(Hmapi * vi))
                _merge_svd_basis!(basis, Solution(ansatzi, ei, vi), length(active), nkeep_used, thresh_schmidt, verbose)
            end
            # S+ chain: guard against na+1 > no (apply_splus condition)
            vi = v_i; ansatzi = ansatz_i; spi = 0
            while ansatzi.nb > 0 && ansatzi.na + 1 <= ansatzi.no
                vi, ansatzi = apply_splus(vi, ansatzi)
                size(vi, 2) == 0 && break
                spi += 1
                verbose > 0 && @printf("  S+ applied (%i): fb sector (%i,%i)\n", spi, ansatzi.na, ansatzi.nb)
                Hmapi = LinearMap(ints_fb, ansatzi)
                ei = diag(vi' * Matrix(Hmapi * vi))
                _merge_svd_basis!(basis, Solution(ansatzi, ei, vi), length(active), nkeep_used, thresh_schmidt, verbose)
            end
        end
    end

    # ---- always use fragment+bath path (no fragment-only fallback) ----
    basis = OrderedDict()

    if per_sector && target_sectors !== nothing
        #
        # PER-SECTOR PATH: separate FCI for each target cluster Fock sector,
        # with bath at CMF occupancy. Closes the charge-sector coverage gap vs
        # compute_cluster_eigenbasis_spin.
        #
        verbose > 0 && @printf(" Per-sector path: FCI in %i orbs per target sector (na_bath=%i, nb_bath=%i)\n",
                                norb2, na_bath, nb_bath)
        for (na_sec, nb_sec) in target_sectors
            _fci_spin_svd!(basis, na_sec + na_bath, nb_sec + nb_bath)
        end
    else
        #
        # SINGLE-SECTOR PATH: FCI only at reference (na_actv, nb_actv)
        #
        verbose > 0 && @printf(" Main path: FCI in %i orbs (dim=%i), then S±+SVD\n",
                                norb2, ansatz_ref.dim)
        _fci_spin_svd!(basis, na_actv, nb_actv)
    end

    return basis
    #=}}}=#
end


"""
    compute_cluster_est_spinbasis(ints, clusters, Da, Db; kwargs...)

Uses `form_est_spinbasis` per cluster, which applies
S+/S- ladder operators in the fragment+bath space BEFORE the SVD decomposition. This gives
spin-pure cluster basis states while retaining the entangled-bath quality of the EST ansatz.

For large clusters where the fragment+bath FCI dimension exceeds `max_fci_dim`, automatically
falls back to fragment-only FCI with bath orbitals folded in via mean-field.

# Key keyword arguments
- `verbose`: 0 = silent, 1 = per-cluster summary, 2 = full matrices
- `thresh_schmidt`: SVD threshold for Schmidt vectors (default 1e-3)
- `init_fspace`: reference FockConfig, needed to determine sectors
- `delta_elec`: electron fluctuation range per cluster
- `est_max_cycles`: Davidson max iterations (default 200)
- `max_fci_dim`: dimension above which fragment-only fallback is used (default 5_000_000)
"""
function compute_cluster_est_spinbasis(ints::InCoreInts{T}, clusters::Vector{MOCluster}, Da, Db;
                            thresh_schmidt=1e-3,
                            thresh_orb=1e-8,
                            thresh_ci=1e-6,
                            do_embedding=true,
                            verbose=0,
                            init_fspace=nothing,
                            delta_elec=nothing,
                            est_nr=1,
                            est_max_cycles=200,
                            est_thresh=1e-6,
                            max_fci_dim=50_000_000,
                            per_sector=false,
                            apply_spin_ladder=true,
                            max_bath=nothing,
                            target_norb=nothing,
                            A::Type=FCIAnsatz) where {T}
    #={{{=#
    cluster_bases = Vector{ClusterBasis{A,T}}()

    for ci in clusters
        verbose > 1 && display(ci)

        # Determine fock sectors before calling form_est_spinbasis so we can
        # pass them as target_sectors for the per_sector path.
        if init_fspace[ci.idx][1] == init_fspace[ci.idx][2]
            sectors = [(init_fspace[ci.idx][1], init_fspace[ci.idx][2])]
        else
            sectors = possible_spin_focksectors_percluster(ci, init_fspace)
        end
        max_e = 2 * length(ci)
        if delta_elec === nothing
            delta_e_i = init_fspace[ci.idx][1] + init_fspace[ci.idx][2]
        else
            delta_e_i = delta_elec[ci.idx]
        end
        sectors = extend_fock_sectors(sectors, delta_e_i, max_e, max_e)
        verbose > 0 && println(" Fock sectors: ", sectors)

        basis = FermiCG.form_est_spinbasis(ints, ci, Da, Db,
                    thresh_schmidt=thresh_schmidt, thresh_orb=thresh_orb,
                    max_bath=max_bath,
                    target_norb=target_norb,
                    do_embedding=do_embedding, verbose=verbose,
                    eig_nr=est_nr, max_iter_davidson=est_max_cycles,
                    max_fci_dim=max_fci_dim,
                    per_sector=per_sector,
                    target_sectors=per_sector ? sectors : nothing,
                    apply_spin_ladder=apply_spin_ladder)

        basis_i = ClusterBasis(ci)
        for sec in sectors
            if sec in keys(basis)
                if verbose > 0
                    println(" Cluster: ", ci.idx, " Fock Sector: ", sec)
                    println(" Number of states in basis: ", size(basis[sec], 2))
                end
                basis_i[sec] = Solution(FCIAnsatz(length(ci), sec[1], sec[2]),
                                        zeros(size(basis[sec], 2)), basis[sec])
            end
        end
        push!(cluster_bases, basis_i)
    end
    return cluster_bases
    #=}}}=#
end


# ──────────────────────────────────────────────────────────────────────────────
# Legacy EST (non-spin-preserving): form_schmidt_basis + compute_cluster_est_basis
# Uses svd_state (no S2 projection). Kept for comparison with est_spinbasis.
# ──────────────────────────────────────────────────────────────────────────────

"""
    form_schmidt_basis(ints, ci, Da, Db; kwargs...)

Legacy EST basis builder: FCI in fragment+bath space, then SVD-project to
fragment using `svd_state` (no S2 projection). Kept for comparison purposes.

If `target_sectors` is provided, does a dedicated FCI per sector (like
`form_est_spinbasis`) instead of a single FCI at the reference electron count.
This ensures every target sector has states even when charge-transfer sectors
have negligible weight in the reference-sector FCI.
"""
function form_schmidt_basis(ints::InCoreInts, ci::MOCluster, Da, Db;
        thresh_schmidt=1e-3, thresh_orb=1e-8, thresh_ci=1e-6, do_embedding=true,
        verbose=0, eig_nr=1, eig_max_cycles=200, A::Type=FCIAnsatz,
        target_sectors=nothing)

    verbose > 0 && println()
    verbose > 0 && println("------------------------------------------------------------")
    verbose > 0 && @printf("Form EST basis (legacy) for Cluster %4i\n", ci.idx)

    D = Da + Db
    K = zeros(size(ints.h1))
    @tensor begin
        K[q,r] = ints.h2[p,q,r,s] * D[p,s]
    end

    no = size(ints.h1, 1)
    ci_no = length(ci.orb_list)
    na_tot = Int(round(tr(Da)))
    nb_tot = Int(round(tr(Db)))

    active = ci.orb_list
    backgr = [i for i in 1:no if !(i in active)]

    K2 = zeros(ci_no, no - ci_no)
    for (pi, p) in enumerate(active), (qi, q) in enumerate(backgr)
        K2[pi, qi] = K[p, q]
    end

    F = svd(K2, full=true)
    nkeep = sum(F.S .> thresh_orb)

    C = zeros(size(ints.h1))
    for (pi, p) in enumerate(active), (qi, q) in enumerate(active)
        pi == qi && (C[p, qi] = 1)
    end
    for (pi, p) in enumerate(backgr), (qi, q) in enumerate(backgr)
        C[p, qi+length(active)] = F.Vt[qi, pi]
    end

    Cfrag = C[:, 1:ci_no]
    Cbath = C[:, ci_no+1:ci_no+nkeep]
    Cenvt = C[:, ci_no+nkeep+1:end]

    K2  = C' * K  * C
    Da2 = C' * Da * C
    Db2 = C' * Db * C

    na = tr(Da2[1:ci_no+nkeep, 1:ci_no+nkeep])
    nb = tr(Db2[1:ci_no+nkeep, 1:ci_no+nkeep])

    denvt_a = Cenvt * Cenvt' * Da * Cenvt * Cenvt'
    denvt_b = Cenvt * Cenvt' * Db * Cenvt * Cenvt'

    na_envt = Int(round(tr(Cenvt' * Da * Cenvt)))
    nb_envt = Int(round(tr(Cenvt' * Db * Cenvt)))

    denvt_a = C' * denvt_a * C
    denvt_b = C' * denvt_b * C
    ints2 = orbital_rotation(ints, C)
    denvt_a[abs.(denvt_a) .< 1e-15] .= 0
    denvt_b[abs.(denvt_b) .< 1e-15] .= 0

    if do_embedding && size(Cenvt, 2) > 0
        EIG = eigen(denvt_a); U = EIG.vectors[:, sortperm(EIG.values, rev=true)]
        denvt_a = U[:, 1:na_envt] * U[:, 1:na_envt]'
        EIG = eigen(denvt_b); U = EIG.vectors[:, sortperm(EIG.values, rev=true)]
        denvt_b = U[:, 1:nb_envt] * U[:, 1:nb_envt]'
    else
        denvt_a .*= 0; denvt_b .*= 0
    end

    no_range = collect(1:size(Cfrag,2) + size(Cbath,2))
    ints_f = subset(ints2, no_range, denvt_a, denvt_b)

    na_actv = na_tot - na_envt
    nb_actv = nb_tot - nb_envt

    norb2 = size(ints_f.h1, 1)

    # Fragment reference electron counts from CMF density (needed for per-sector bath offset)
    na_frag_ref = Int(round(tr(Da2[1:ci_no, 1:ci_no])))
    nb_frag_ref = Int(round(tr(Db2[1:ci_no, 1:ci_no])))
    na_bath_ref = na_actv - na_frag_ref
    nb_bath_ref = nb_actv - nb_frag_ref

    # Helper: run FCI at (na_fb, nb_fb) and return eigenpairs
    function _run_fci(na_fb, nb_fb, nr)
        ansatz_s = FCIAnsatz(norb2, na_fb, nb_fb)
        nr_i = min(nr, ansatz_s.dim)
        if ansatz_s.dim < 2000 || nr_i * 4 > ansatz_s.dim
            verbose > 0 && @printf(" FCI sector (na_fb=%i, nb_fb=%i, dim=%i, nr=%i) → direct\n",
                                    na_fb, nb_fb, ansatz_s.dim, nr_i)
            Hmat = build_H_matrix(ints_f, ansatz_s)
            F2 = eigen(Hmat)
            return ansatz_s, F2.values[1:nr_i], F2.vectors[:, 1:nr_i]
        else
            verbose > 0 && @printf(" FCI sector (na_fb=%i, nb_fb=%i, dim=%i, nr=%i) → Davidson\n",
                                    na_fb, nb_fb, ansatz_s.dim, nr_i)
            Hmap = LinearMap(ints_f, ansatz_s)
            v0 = _diag_init_guess(ints_f, ansatz_s, nr_i)
            dav = FermiCG.Davidson(Hmap, v0=v0, max_iter=eig_max_cycles,
                                   max_ss_vecs=max(40, 8*nr_i), nroots=nr_i, tol=1e-6)
            e_dav, v_dav = BlockDavidson.eigs(dav, iprint=verbose)
            return ansatz_s, real.(e_dav), v_dav
        end
    end

    if isnothing(target_sectors)
        # Original: single FCI at reference electron count, SVD over all sectors
        ansatz_ref, e_i, v_i = _run_fci(na_actv, nb_actv, eig_nr)
        sol = Solution(ansatz_ref, e_i, v_i)
        return ActiveSpaceSolvers.svd_state(sol, length(active), nkeep, thresh_schmidt, verbose=verbose)
    else
        # Per-sector: dedicated FCI for each target sector + svd_state (no S2 projection).
        # Mirrors form_est_spinbasis but without S2 projection.
        basis = OrderedDict()
        for sec in target_sectors
            na_fb = na_bath_ref + sec[1]
            nb_fb = nb_bath_ref + sec[2]
            (0 <= na_fb <= norb2 && 0 <= nb_fb <= norb2) || continue

            ansatz_sec, e_i, v_i = _run_fci(na_fb, nb_fb, eig_nr)
            ansatz_sec.dim == 0 && continue

            sol = Solution(ansatz_sec, e_i, v_i)
            new_b = ActiveSpaceSolvers.svd_state(sol, length(active), nkeep, thresh_schmidt, verbose=verbose)
            haskey(new_b, sec) || continue

            if haskey(basis, sec)
                combined = hcat(basis[sec], new_b[sec])
                F_qr = qr(combined)
                rk = sum(abs.(diag(Matrix(F_qr.R))) .> 1e-10)
                basis[sec] = Matrix(F_qr.Q)[:, 1:rk]
            else
                basis[sec] = new_b[sec]
            end
        end
        return basis
    end
end


"""
    compute_cluster_est_basis(ints, clusters, Da, Db; kwargs...)

Legacy EST driver: calls `form_schmidt_basis` (no S2 projection) for each cluster.
Kept for comparison with `compute_cluster_est_spinbasis`.
"""
function compute_cluster_est_basis(ints::InCoreInts{T}, clusters::Vector{MOCluster}, Da, Db;
                thresh_schmidt=1e-3, thresh_orb=1e-8, thresh_ci=1e-6,
                do_embedding=true, verbose=0, init_fspace=nothing, delta_elec=nothing,
                est_nr=1, est_max_cycles=200, est_thresh=1e-6,
                A::Type=FCIAnsatz) where T
    cluster_bases = Vector{ClusterBasis{A,T}}()

    for ci in clusters
        verbose > 1 && display(ci)

        delta_e_i = ()
        if all((delta_elec, init_fspace) .!= nothing)
            delta_e_i = (init_fspace[ci.idx][1], init_fspace[ci.idx][2], delta_elec[ci.idx])
        end
        sectors = possible_focksectors(ci, delta_elec=delta_e_i)

        basis = form_schmidt_basis(ints, ci, Da, Db,
                    thresh_schmidt=thresh_schmidt, thresh_orb=thresh_orb,
                    do_embedding=do_embedding, verbose=verbose,
                    eig_nr=est_nr, eig_max_cycles=est_max_cycles,
                    target_sectors=sectors)

        basis_i = ClusterBasis(ci)
        for sec in sectors
            if sec in keys(basis)
                verbose > 0 && println(" Cluster: ", ci.idx, " Fock Sector: ", sec,
                                       " states: ", size(basis[sec], 2))
                basis_i[sec] = Solution(FCIAnsatz(length(ci), sec[1], sec[2]),
                                        zeros(size(basis[sec], 2)), basis[sec])
            end
        end
        push!(cluster_bases, basis_i)
    end
    return cluster_bases
end