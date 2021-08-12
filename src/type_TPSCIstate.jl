using StaticArrays

"""
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, Vector{T}}}

This represents an arbitrarily sparse state. E.g., used in TPSCI
"""
struct TPSCIstate{T,N,R} <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, MVector{R,T}}}
end
Base.haskey(ts::TPSCIstate, i) = return haskey(ts.data,i)
#Base.iterate(ts::TPSCIstate, state=1) = iterate(ts.data, state)
#Base.eltype(::Type{TPSCIstate{T,N,R}}) where {T,N,R} = OrderedDict{ClusterConfig{N}, MVector{R,T}} 

"""
    TPSCIstate(clusters; T=Float64, R=1)

Constructor creating an empty vector
# Arguments
- `clusters::Vector{Cluster}`
- `T`:  Type of data for coefficients
- `R`:  Number of roots
# Returns
- `TPSCIstate`
"""
function TPSCIstate(clusters; T=Float64, R=1)
    N = length(clusters)
    return TPSCIstate{T,N,R}(clusters,OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, MVector{R,T}}}())
end

"""
    function TPSCIstate(v::TPSCIstate{T,N,R}; T=T, R=R) where {T,N,R}

Constructor creating a `TPSCIstate` with the same basis as `v`, but with potentially different `R` and `T`. 
Coefficients of new vector are 0.0

# Arguments
- `T`:  Type of data for coefficients
- `R`:  Number of roots
# Returns
- `TPSCIstate`
"""
function TPSCIstate(v::TPSCIstate{TT,NN,RR}; T=TT, R=RR) where {TT,NN,RR}
    out = TPSCIstate(v.clusters,T=T,R=R)
    for (fock, configs) in v.data
        add_fockconfig!(out,fock)
        for (config, coeffs) in configs
            out[fock][config] = zeros(T,R)
        end
    end
    return out
end

"""
    TPSCIstate(clusters::Vector{Cluster}, fconfig::FockConfig{N}; T=Float64, R=1) where {N}

Constructor using only a single FockConfig. This allows us to turn the CMF state into a TPSCIstate.
# Arguments
- `clusters`: vector of clusters types
- `fconfig`: starting FockConfig
- `T`:  Type of data for coefficients
- `R`:  Number of roots
# Returns
- `TPSCIstate`
"""
function TPSCIstate(clusters::Vector{Cluster}, fconfig::FockConfig{N}; T=Float64, R=1) where {N}
    #={{{=#

    state = TPSCIstate(clusters, T=T, R=R)
    add_fockconfig!(state, fconfig)
    conf = ClusterConfig([1 for i in 1:length(clusters)])
    state[fconfig][conf] = zeros(T,R) 
    return state
#=}}}=#
end









"""
    add_fockconfig!(s::TPSCIstate, fock::FockConfig)
"""
function add_fockconfig!(s::TPSCIstate{T,N,R}, fock::FockConfig{N}) where {T<:Number,N,R}
    s.data[fock] = OrderedDict{ClusterConfig{N}, MVector{R,T}}()
    #s.data[fock] = OrderedDict{ClusterConfig{N}, MVector{R,T}}(ClusterConfig([1 for i in 1:N]) => zeros(MVector{R,T}))
end

"""
    getindex(s::TPSCIstate, fock::Vector{Tuple{T,T}}) where T<:Integer
"""
#Base.getindex(s::TPSCIstate, fock::Vector{Tuple{T,T}}) where T<:Integer = s.data[fock]
@inline Base.getindex(s::TPSCIstate, fock) = s.data[fock]
@inline Base.setindex!(s::TPSCIstate, a, b) = s.data[b] = a


function Base.size(s::TPSCIstate{T,N,R}) where {T,N,R}
    return length(s),R
end
function Base.length(s::TPSCIstate)
    l = 0
    for (fock,configs) in s.data 
        l += length(keys(configs))
    end
    return l
end
"""
    get_vector(s::TPSCIstate; root=1)
"""
function get_vector(s::TPSCIstate; root=1)
    v = zeros(length(s))
    idx = 1
    for (fock, configs) in s.data
        for (config, coeff) in configs
            v[idx] = coeff[root]
            idx += 1
        end
    end
    return v
end
"""
    get_vectors(s::TPSCIstate)
"""
function get_vectors(s::TPSCIstate{T,N,R}) where {T,N,R}
    v = zeros(T,length(s), R)
    idx = 1
    for (fock, configs) in s.data
        for (config, coeff) in configs
            v[idx,:] .= coeff[:]
            idx += 1
        end
    end
    return v
end

"""
    get_vectors!(v, s::TPSCIstate)
"""
function get_vectors!(v, s::TPSCIstate{T,N,R}) where {T,N,R}
    idx = 1
    for (fock, configs) in s.data
        for (config, coeff) in configs
            v[idx,:] .= coeff[:]
            idx += 1
        end
    end
    return
end

"""
    function set_vector!(ts::TPSCIstate{T,N,R}, v::Matrix{T}) where {T,N,R}

Fill the coefficients of `ts` with the values in `v`
"""
function set_vector!(ts::TPSCIstate{T,N,R}, v::Matrix{T}) where {T,N,R}

    nbasis = size(v,1)
    nroots = size(v,2)

    length(ts) == nbasis || throw(DimensionMismatch)
    R == nroots || throw(DimensionMismatch)

    idx = 1
    for (fock, tconfigs) in ts.data
        for (tconfig, coeffs) in tconfigs
            #ts[fock][tconfig] = MVector{R}(v[idx,:])
            @views coeffs .= v[idx,:]
            idx += 1
        end
    end
    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
    return
end

#"""
#    function set_vector!(ts::TPSCIstate{T,N,R}, v) where {T,N,R}
#
#Fill the coefficients of `ts` with the values in `v`
#"""
#function set_vector!(ts::TPSCIstate{T,N,R}, v) where {T,N,R}
#
#    nbasis = size(v,1)
#    nroots = size(v,2)
#
#    length(ts) == nbasis || throw(DimensionMismatch)
#    R == nroots || throw(DimensionMismatch)
#
#    idx = 1
#    for (fock, tconfigs) in ts.data
#        for (tconfig, coeffs) in tconfigs
#            #ts[fock][tconfig] = MVector{R}(v[idx,:])
#            @views coeffs .= v[idx,:]
#            idx += 1
#        end
#    end
#    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
#    return
#end


"""
    Base.display(s::TPSCIstate; thresh=1e-3, root=1)

Pretty print
"""
function Base.display(s::TPSCIstate; thresh=1e-3, root=1)
    @printf(" --------------------------------------------------\n")
    @printf(" ---------- Fockspaces in state ------: Dim = %5i  \n",length(s))
    @printf(" ----------                root ------:     = %5i  \n",root)
    @printf(" --------------------------------------------------\n")
    @printf(" Printing contributions greater than: %f", thresh)
    @printf("\n")
    @printf(" %-20s%-20s%-20s\n", "Weight", "# Configs", "Fock space(α,β)...") 
    @printf(" %-20s%-20s%-20s\n", "-------", "---------", "----------")
    for (fock,configs) in s.data
        prob = 0
        for (config, coeff) in configs 
            prob += coeff[root]*coeff[root] 
        end
        if prob > thresh
            @printf(" %-20.3f%-20i", prob,length(s.data[fock]))
            for sector in fock 
                @printf("(%2i,%-2i)", sector[1],sector[2])
            end
            println()
        end
    end
    print(" --------------------------------------------------\n")
end

"""
    print_configs(s::ClusterState; thresh=1e-3)

Pretty print
"""
function print_configs(s::TPSCIstate; thresh=1e-3, root=1)
    #display(keys(s.data))
    idx = 1
    for (fock,configs) in s.data
        length(s.clusters) == length(fock) || throw(Exception)
        length(s.data[fock]) > 0 || continue
        @printf(" Dim %4i fock_space: ",length(s.data[fock]))
        [@printf(" %-2i(%i:%i) ",fii,fi[1],fi[2]) for (fii,fi) in enumerate(fock)] 
        println()
        for (config, value) in s.data[fock]
            @printf(" %5i",idx)
            for c in config
                @printf("%3i",c)
            end
            @printf(":%12.8f\n",value[1])
            idx += 1
        end
    end
end

"""
    norm(s::TPSCIstate, root)
"""
function LinearAlgebra.norm(s::TPSCIstate, root)
    norm = 0
    for (fock,configs) in s.data
        for (config,coeff) in configs
            norm += coeff[root]*coeff[root]
        end
    end
    return sqrt(norm)
end

"""
    norm(s::TPSCIstate{T,N,R}) where {T,N,R}
"""
function LinearAlgebra.norm(s::TPSCIstate{T,N,R}) where {T,N,R}
    norms = zeros(T,R)
    for (fock,configs) in s.data
        for (config,coeff) in configs
            for r in 1:R
                norms[r] += coeff[r]*coeff[r]
            end
        end
    end
    for r in 1:R
        norms[r] = sqrt(norms[r])
    end
    return norms
end

"""
    normalize!(s::AbstractState)
"""
function normalize!(s::AbstractState)
    scale!(s,1/sqrt(dot(s,s))) 
end

"""
    scale!(s::TPSCIstate,c)
"""
function scale!(s::TPSCIstate{T,N,R},c;root=nothing) where {T,N,R}
    if root == nothing
        for (fock,configs) in s.data
            for (config,coeff) in configs
                s[fock][config] .= coeff.*c
            end
        end
    else
        root <= R || error("root>R")
        for (fock,configs) in s.data
            for (config,coeff) in configs
                s[fock][config][root] = coeff[root]*c
            end
        end
    end
end
    
"""
    dot(v1::TPSCIstate,v2::TPSCIstate; r1=1, r2=1)
"""
function dot(v1::TPSCIstate{T,N,1},v2::TPSCIstate{T,N,1}) where {T,N}
    d = T(0)
    for (fock,configs) in v1.data
        haskey(v2.data, fock) || continue
        for (config,coeff) in configs
            haskey(v2.data[fock], config) || continue
            d += coeff[1] * v2.data[fock][config][1]
        end
    end
    return d
end
    
"""
    dot(v1::TPSCIstate,v2::TPSCIstate; r1=1, r2=1)
"""
function dot(v1::TPSCIstate{T,N,R}, v2::TPSCIstate{T,N,R}, r1, r2) where {T,N,R}
    d = T(0)
    for (fock,configs) in v1.data
        haskey(v2.data, fock) || continue
        for (config,coeff) in configs
            haskey(v2.data[fock], config) || continue
            d += coeff[r1] * v2.data[fock][config][r2]
        end
    end
    return d
end
    
"""
    dot(v1::TPSCIstate,v2::TPSCIstate; r1=1, r2=1)
"""
function orth!(v1::TPSCIstate{T,N,R}) where {T,N,R}
    d = T(0)
    F = svd(get_vectors(v1))

    set_vector!(v1, F.U*F.Vt)
    return 
end
    
"""
    prune_empty_fock_spaces!(s::TPSCIstate)
        
remove fock_spaces that don't have any configurations 
"""
function prune_empty_fock_spaces!(s::TPSCIstate)
    keylist = [keys(s.data)...]
    for fock in keylist
        if length(s[fock]) == 0
            delete!(s.data, fock)
        end
    end
#    # I'm not sure why this is necessary
#    idx = 0
#    for (fock,configs) in s.data
#        for (config, coeffs) in s.data[fock]
#            idx += 1
#        end
#    end
    return 
end

"""
    zero!(s::TPSCIstate)

set all elements to zero
"""
function zero!(s::TPSCIstate{T,N,R}) where {T,N,R}
    for (fock,configs) in s.data
        for (config,coeffs) in configs                
            s.data[fock][config] = zeros(size(MVector{R,T}))
            #s.data[fock][config] = zeros(MVector{R,T})
        end
    end
end


"""
    function randomize!(s::TPSCIstate{T,N,R}) where {T,N,R}

set all elements to random values, and orthogonalize
"""
function randomize!(s::TPSCIstate{T,N,R}) where {T,N,R}
    #={{{=#
    v0 = rand(T,size(s)) .- .5 
    set_vector!(s,v0)
    orthonormalize!(s)
end
#=}}}=#


"""
    function orthonormalize!(s::TPSCIstate{T,N,R}) where {T,N,R}

orthonormalize
"""
function orthonormalize!(s::TPSCIstate{T,N,R}) where {T,N,R}
    #={{{=#
    v0 = get_vectors(s) 
    v0[:,1] .= v0[:,1]./norm(v0[:,1])
    for r in 2:R
        #|vr> = |vr> - |v1><v1|vr> - |v2><v2|vr> - ... 
        for r0 in 1:r-1 
            v0[:,r] .-= v0[:,r0] .* (v0[:,r0]'*v0[:,r])
        end
        v0[:,r] .= v0[:,r]./norm(v0[:,r])
    end
    isapprox(det(v0'*v0), 1.0, atol=1e-14) || @warn "initial guess det(v0'v0) = ", det(v0'v0) 
    set_vector!(s,v0)
end
#=}}}=#


"""
    clip!(s::TPSCIstate; thresh=1e-5)
"""
function clip!(s::TPSCIstate; thresh=1e-5)
#={{{=#
    for (fock,configs) in s.data
        for (config,coeff) in configs      
            if all(abs(c) < thresh for c in coeff)
                delete!(s.data[fock], config)
            end
        end
    end
    prune_empty_fock_spaces!(s)
end
#=}}}=#


"""
    add!(s1::TPSCIstate, s2::TPSCIstate)

Add coeffs in `s2` to `s1`
"""
function add!(s1::TPSCIstate, s2::TPSCIstate)
    #={{{=#
    for (fock,configs) in s2.data
        if haskey(s1, fock)
            for (config,coeffs) in configs
                if haskey(s1[fock], config)
                    s1[fock][config] .+= s2[fock][config]
                else
                    s1[fock][config] = deepcopy(s2[fock][config])
                end
            end
        else
            s1[fock] = deepcopy(s2[fock])
        end
    end
    #=}}}=#
end


"""
    function extract_roots(v::TPSCIstate{T,N,R}, roots)

Extract roots to give new `TPSCIstate` 
"""
function extract_roots(v::TPSCIstate{T,N,R}, roots) where {T,N,R}
    vecs = get_vectors(v)[:,roots]

    out = TPSCIstate(v.clusters, T=T, R=length(roots))
    for (fock,configs) in v.data
        add_fockconfig!(out,fock)
        for (config,coeffs) in configs
            out[fock][config] = deepcopy(v[fock][config][roots])
        end
    end

    return out
end

