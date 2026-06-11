using ActiveSpaceSolvers

"""
    cluster::MOCluster                            # Cluster to which basis belongs
    basis::Dict{Tuple,Matrix{T}}                # Basis vectors (nα, nβ)=>[I,s]

These basis coefficients map local slater determinants to local vectors
`(nα, nβ): 
V[αstring*βstring, cluster_state]`
"""
struct ClusterBasis{A,T}
    cluster::MOCluster
    basis::Dict{Tuple{Int16,Int16},Solution{A,T}}
end
ClusterBasis(ci::MOCluster; T::Type = Float64, A=FCIAnsatz) = ClusterBasis(ci, Dict{Tuple{Int16,Int16},Solution{A,T}}())

"""
    ClusterBasis(cb::ClusterBasis, T::Type)

Convert from one data type to another
"""
function ClusterBasis(cb::ClusterBasis{A,TT}, T::Type) where {A,TT}
    out = ClusterBasis{A,T}(cb.cluster, Dict{Tuple{Int16,Int16},Matrix{T}}())
    for (fspace,basis) in cb.basis
        out.basis[fspace] = Solution(basis.ansatz, Vector{T}(basis.energies), Matrix{T}(basis.vectors))
    end
    return out
end

Base.iterate(cb::ClusterBasis, state=1) = iterate(cb.basis, state)
Base.length(cb::ClusterBasis) = length(cb.basis)
Base.getindex(cb::ClusterBasis,i) = cb.basis[i] 
Base.setindex!(cb::ClusterBasis,val,key) = cb.basis[key] = val
Base.haskey(cb::ClusterBasis,key) = haskey(cb.basis, key)
function Base.display(cb::ClusterBasis) 
    @printf(" ClusterBasis for Cluster: %4i\n",cb.cluster.idx)
    norb = length(cb.cluster)
    sum_total_dim = 0
    sum_dim = 0
    
    T = eltype(norb)
    for (sector, vecs) in cb.basis
        na = T(sector[1])
        nb = T(sector[2])
        dim = size(vecs,2)
        total_dim = binomial(norb,na) * binomial(norb,nb) 
        sum_dim += dim
        sum_total_dim += total_dim
        
        @printf("   FockSector = (%2iα, %2iβ): Total Dim = %5i: Dim = %4i\n", sector[1],sector[2],total_dim, dim)
    end
       
    @printf("   -----------------------------\n")
    @printf("   Total Dim = %5i: Dim = %4i\n", sum_total_dim, sum_dim)
end



"""
    rotate!(cb::ClusterBasis, U::Dict{Tuple,Matrix{T}}) where {T} 

Rotate `cb` by unitary matrices in `U`
"""
function rotate!(cb::ClusterBasis,U::Dict{Tuple,Matrix{T}}) where {T} 
#={{{=#
    for (fspace,mat) in U
        cb[fspace].vectors .= cb[fspace] * mat
    end
end
#=}}}=#

    
function check_basis_orthogonality(basis::ClusterBasis; thresh=1e-12)
    for (fspace,mat) in basis
        if check_orthogonality(mat,thresh=thresh) == false
            println(" Cluster:", basis.cluster)
            println(" Fockspace:", fspace)
        end
    end
end

using LinearAlgebra

"""
Merge two Vector{ClusterBasis}: for each cluster, union the Fock sectors.
If a sector exists in both, keep the one with more roots (columns in vectors).
If you want to combine states from both (e.g. augment ground with CT states),
pass augment=true — this hcats vectors and re-orthogonalizes via QR.
"""
function merge_cluster_bases(cb1::Vector{T}, cb2::Vector{T}; augment=false) where T
    @assert length(cb1) == length(cb2)
    result = Vector{T}(undef, length(cb1))
    for i in eachindex(cb1)
        new_cb = FermiCG.ClusterBasis(cb1[i].cluster)
        # seed with all sectors from cb1
        for (fspace, sol) in cb1[i]
            new_cb[fspace] = sol
        end
        # merge sectors from cb2
        for (fspace, sol2) in cb2[i]
            if !haskey(new_cb, fspace)
                new_cb[fspace] = sol2
            elseif augment
                sol1 = new_cb[fspace]
                # concatenate vectors and re-orthogonalize
                combined = hcat(sol1.vectors, sol2.vectors)
                Q, _ = qr(combined)
                n = size(combined, 2)
                vecs = Matrix(Q)[:, 1:n]
                energies = vcat(sol1.energies, sol2.energies)[1:n]
                new_cb[fspace] = ActiveSpaceSolvers.Solution(sol1.ansatz, energies, vecs)
            else
                # keep whichever has more roots
                if size(sol2.vectors, 2) > size(new_cb[fspace].vectors, 2)
                    new_cb[fspace] = sol2
                end
            end
        end
        result[i] = new_cb
    end
    return result
end
