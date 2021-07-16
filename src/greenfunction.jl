
"""
Abstract type for Green functions
    A : type of the axis
    D : Array like container describing the dirac part
    R : Array like container describing the regular part
    
    All T<:AbsractGreenFunction should implement the following constructor:
    T(axis, dirac, regular, blocksize)
    """
abstract type AbstractGreenFunction{A,D <: AbstractArray,R <: AbstractArray} end
abstract type AbstractRetardedGreenFunction{A,D <: AbstractArray,R <: AbstractArray} <: AbstractGreenFunction{A,D,R} end
abstract type AbstractAdvancedGreenFunction{A,D <: AbstractArray,R <: AbstractArray} <: AbstractGreenFunction{A,D,R} end

struct RetardedGreenFunction{A,D,R} <: AbstractRetardedGreenFunction{A,D,R}
    axis::A
    dirac::D
    retarded::R
    blocksize::Int64
end
struct AdvancedGreenFunction{A,D,R} <: AbstractAdvancedGreenFunction{A,D,R}
    axis::A
    dirac::D
    advanced::R
    blocksize::Int64
end
struct GreenFunction{A,D,R} <: AbstractGreenFunction{A,D,R}
    axis::A
    dirac::D
    regular::R
    blocksize::Int64
end

RetardedOrAdvanced = Union{RetardedGreenFunction,AdvancedGreenFunction}

#Construction utilities

for G in (:GreenFunction,:RetardedGreenFunction,:AdvancedGreenFunction)
    @eval begin
        function $G(axis,regular::AbstractArray{T,2},blocksize) where {T}
            $G(axis, zeros(T,blocksize,blocksize,length(axis)), regular, blocksize)
        end
    end
end

for (G,croped) in [
        (:GreenFunction, :(freg)),
        (:RetardedGreenFunction, :((x,y) -> x>=y ? freg(x,y) : zero(f00)  )),
        (:AdvancedGreenFunction,:( (x,y) -> x<=y ? freg(x,y) : zero(f00) ))
         ]
    @eval begin
        function $G(axis,fδ,freg; compressor = HssCompressor())
            #-For Retarded / Advanced Green functions, we enforce that the relevant part
            #of the Green function is zero.
            #-For GreenFunction the quadrature rule does not require the value at (t,t)
            #that can be poorly defined
            f00 = freg(axis[1],axis[1])
            bs = size(f00,1)
            r = compressor(axis, $croped)
            δ = zeros(eltype(f00),bs,bs,length(axis))
            Threads.@threads for i = 1:length(axis)
                δ[:,:,i] .= fδ(axis[i]) 
            end
            $G(axis, δ, r, bs)
        end
    end
end



size(g::AbstractGreenFunction) = ( length(axis(g)), length(axis(g)) )
blocksize(g::AbstractGreenFunction) = g.blocksize
axis(g::AbstractGreenFunction) = g.axis
eltype(g::AbstractGreenFunction) = eltype(regular(g))
function iscompatible(g,k)
     axis(g) == axis(k) || blocksize(g) == blocksize(k)
end

advanced(g::AdvancedGreenFunction) = g.advanced
retarded(g::RetardedGreenFunction) = g.retarded

regular(g::GreenFunction) = g.regular
regular(g::RetardedGreenFunction) = g.retarded
regular(g::AdvancedGreenFunction) = g.advanced

dirac(g::AbstractGreenFunction) = g.dirac

for op in (:+,:-)
    @eval begin 
        $op(g::G) where {G<:AbstractGreenFunction} =  G(axis(g),$op(dirac(g)), $op(regular(g)),blocksize(g))

        function $op(g::G,k::G) where {G<:AbstractGreenFunction}  
            @assert axis(g) == axis(k)
            G(axis(g),$op(dirac(g),dirac(k)), $op(regular(g),regular(k)), blocksize(g))
        end
        function $op(g::AbstractGreenFunction,k::AbstractGreenFunction)
            #Be carefull here !
            @assert axis(g) == axis(k)
            reg_part = $op(regular(g),regular(k))
            reg_part -= 0.5*extract_blockdiag(reg_part,blocksize(g))
            GreenFunction(axis(g),$op(dirac(g),dirac(k)), reg_part, blocksize(g))
        end
    end
end


for op in (:*,:\)
    @eval begin
        function $op(λ::T,g::G) where {T<: Number,G<:AbstractGreenFunction}
            return G(axis(g), $op(λ, dirac(g)), $op(λ, regular(g)), blocksize(g))
        end
    end
end
#=
for op in (:*,:/)
    @eval begin
        function $op(g::G,λ::T) where {T<: Number,G<:AbstractGreenFunction}
            return G(axis(g), $op(dirac(g), λ), $op(regular(g),λ), blocksize(g) )
        end
    end
end
=#
function _integral_operator(a::AbstractMatrix{T},bs; compressor = HssCompressor()) where T
    #TODO improve for integral of the type R/A*K ou K*R/A
    bd_a = extract_blockdiag(a,bs, compressor = compressor)
    A = a-T(0.5)*bd_a
    return A,bd_a
end


function _prod(δ::AbstractArray{T,3},r::AbstractArray{T,2}; compressor = HssCompressor()) where T
    matrix_δ = blockdiag(δ, compressor = compressor)
    δr = matrix_δ * r
    return δr
end
function _prod(r::AbstractArray{T,2}, δ::AbstractArray{T,3};compressor = HssCompressor()) where T
    matrix_δ = blockdiag(δ, compressor = compressor)
    rδ = r*matrix_δ 
    return rδ
end

function _prod(δl::AbstractArray{T,3},δr::AbstractArray{T,3}) where T
    return batched_mul(δl, δr)
end

function cc_prod(g::AbstractArray{T,2},k::AbstractArray{T,2},axis; compressor = HssCompressor()) where T
    bs = div(size(g,1),length(axis))
    @assert size(g,1) % length(axis) == 0
    dg = extract_blockdiag(g,bs,compressor = compressor)
    dk = extract_blockdiag(k,bs,compressor = compressor)
    G = g - T(0.5)*dg
    K = k - T(0.5)*dk
    S = G*K
    s = S - extract_blockdiag(S,bs, compressor = compressor) # Diagonal is zero
    return T(step(axis))*s
end

function (*)(g::RA,k::RA,; compressor = HssCompressor()) where {RA <: RetardedOrAdvanced}
    @assert iscompatible(g,k)
    δ = _prod(dirac(g),dirac(k))
    reg = cc_prod(regular(g),regular(k), axis(g), compressor = compressor)
    #reg = eltype(g)(step(axis(g)))*regular(g)*regular(k)
    reg += _prod(dirac(g),regular(k),compressor = compressor)
    reg += _prod(regular(g),dirac(k),compressor = compressor)
    return RA(axis(g),δ,compressor(reg),blocksize(g))
end

function (*)(g::AbstractGreenFunction,k::AbstractGreenFunction; compressor = HssCompressor())
    @assert iscompatible(g,k)
    δ = _prod(dirac(g),dirac(k))
    reg = _prod(g, k, regular(g),regular(k), axis(g), compressor = compressor)
    #reg = eltype(g)(step(axis(g)))*regular(g)*regular(k)
    reg += _prod(dirac(g),regular(k),compressor = compressor)
    reg += _prod(regular(g),dirac(k),compressor = compressor)
    return GreenFunction(axis(g),δ,compressor(reg),blocksize(g))
end


function _prod(::GreenFunction,::RetardedGreenFunction, g::AbstractArray{T,2},k::AbstractArray{T,2},axis; compressor = HssCompressor()) where T
    #First we dress g with integration weights
    bs = div(size(g,1),length(axis))
    g_op = g - T(0.5)*col(g,length(axis),bs)
    g_op *= step(axis)
    k_op = k - T(0.5)*extract_blockdiag(k,bs,compressor = compressor)
    g_op,k_op = compressor.( (g_op,k_op) )
    gk = g_op * k_op
    gk = gk - col(gk,length(axis),bs)
    return compressor(gk)
end

#Here
function _prod(::GreenFunction,::AdvancedGreenFunction,g::AbstractArray{T,2},k::AbstractArray{T,2},axis; compressor = HssCompressor()) where T
    #First we dress g with integration weights
    bs = div(size(g,1),length(axis))
    g_op = g - T(0.5)*col(g,1,bs)
    g_op -= T(0.5)*col(g,length(axis),bs)
    g_op += T(1+1/2 -1)*extract_blockdiag(g_op,bs,[-1,1], compressor = compressor) + 
    T(0-1)*extract_blockdiag(g_op,bs,0, compressor = compressor)
    g_op = step(axis)*g_op
    k_op = k - T(0.5)*extract_blockdiag(k,bs,compressor = compressor)
    g_op,k_op = compressor.( (g_op,k_op) )
    gk = g_op* k_op
    #Apply corrections
    gk = gk - col(gk,1,bs)
    bd_g = extract_blockdiag(g,bs,[-1,0,1])
    bd_k = extract_blockdiag(k,bs,[-1,0,1])
    c_m = [(3/4-1/2)step(axis)*bd_g[blockrange(t,bs),blockrange(t+1,bs)]*bd_k[blockrange(t+1,bs),blockrange(t+1,bs)]
            for t = 1:length(axis)-1]
    c_p = [(3/4-1)*step(axis)*bd_g[blockrange(t,bs),blockrange(t-1,bs)]*bd_k[blockrange(t-1,bs),blockrange(t-1,bs)] 
            for t = 2:length(axis)]
    correction = blockdiag(c_m,-1,compressor = compressor)+blockdiag(c_p,1,compressor = compressor)
    gk -= correction
    return compressor(gk)
end

#Here
function _prod(::RetardedGreenFunction, ::GreenFunction, k::AbstractArray{T,2},g::AbstractArray{T,2},axis; compressor = HssCompressor()) where T
    bs = div(size(g,1),length(axis))
    g_op = g - T(0.5)*row(g,1,bs)
    g_op -= T(0.5)*row(g,length(axis),bs)
    # #=
    g_op += T(1+1/2 -1)*extract_blockdiag(g_op,bs,[-1,1], compressor = compressor) + 
    T(0-1)*extract_blockdiag(g_op,bs,0, compressor = compressor) # =#!!!
    g_op = step(axis) * g_op
    k_op = k - T(0.5)*extract_blockdiag(k,bs,compressor = compressor)
    g_op,k_op = compressor.( (g_op,k_op) )
    kg = k_op*g_op
    #Correction
    bd_g = extract_blockdiag(g,bs,[-1,0,1])
    bd_k = extract_blockdiag(k,bs,[-1,0,1])
    c_m = [(3/4-1)step(axis)*bd_g[blockrange(t,bs),blockrange(t+1,bs)]*bd_k[blockrange(t+1,bs),blockrange(t+1,bs)]
            for t = 1:length(axis)-1]
    c_p = [(3/4-1/2)*step(axis)*bd_g[blockrange(t,bs),blockrange(t-1,bs)]*bd_k[blockrange(t-1,bs),blockrange(t-1,bs)] 
            for t = 2:length(axis) ]
    correction = blockdiag(c_m,-1,compressor = compressor)+blockdiag(c_p,1,compressor = compressor)
    kg -= correction
    kg = kg - row(kg,1,bs)
    return compressor(kg)
end

function _prod(::AdvancedGreenFunction, ::GreenFunction, k::AbstractArray{T,2},g::AbstractArray{T,2},axis; compressor = HssCompressor()) where T
    bs = div(size(g,1),length(axis))
    g_op = g - T(0.5)*row(g,length(axis),bs)
    g_op = step(axis) * g_op
    k_op = k - T(0.5)*extract_blockdiag(k,bs,compressor = compressor)
    g_op,k_op = compressor.( (g_op,k_op) )
    kg = k_op*g_op
    kg = kg - row(kg,length(axis),bs)
    return compressor(kg)
end

function _prod(::GreenFunction, ::GreenFunction, g::AbstractArray{T,2},k::AbstractArray{T,2},axis; compressor = HssCompressor()) where T
    bs = div(size(g,1),length(axis))
    g_op = g - T(0.5)*row(g,length(axis),bs)
    g_op = step(axis) * g_op
    g_op = compressor(g_op)
    gk = g_op*k
    return compressor(gk)
end

#=
function vpc_prod(g::AbstractArray{T,2},k::AbstractArray{T,2},axis; compressor = HssCompressor()) where T
    #First we dress g with integration weights
    bs = div(size(g,1),length(axis))
    correction = T(1-(1/2-2/9))*extract_blockdiag(g,bs,[-2,2],compressor = compressor) +
    T(1-11/9)*extract_blockdiag(g,bs,[-1,1],compressor = compressor) - extract_blockdiag(g,bs,0,compressor = compressor)
    g_op = g + correction
    compressor(g_op) #recompress inplace g
    g_op *= step(axis)
    return g_op * k
end

function cvp_prod(k::AbstractArray{T,2},g::AbstractArray{T,2},axis; compressor = HssCompressor()) where T
    #First we dress g with integration weights
    bs = div(size(g,1),length(axis))
    correction = T(1-(1/2-2/9))*extract_blockdiag(g,bs,[-2,2],compressor = compressor) +
    T(1-11/9)*extract_blockdiag(g,bs,[-1,1],compressor  = compressor) - extract_blockdiag(g,bs,0,compressor  = compressor)
    g_op = g + correction
    compressor(g_op) #recompress inplace g
    g_op *= step(axis)
    return k*g_op
end
=#

function adjoint(G::RetardedGreenFunction)
    return AdvancedGreenFunction(axis(G), permutedims( dirac(G),(2,1,3)) .|> conj , 
    copy(adjoint(retarded(G)) ), blocksize(G) )
end

function adjoint(G::AdvancedGreenFunction)
    return RetardedGreenFunction(axis(G),permutedims( dirac(G),(2,1,3)) .|> conj, 
    copy(adjoint(advanced(G)) ), blocksize(G) )
end

function adjoint(G::GreenFunction)
    return GreenFunction(axis(G),permutedims( dirac(G),(2,1,3)) .|> conj, copy( adjoint(regular(G))), blocksize(G) )
end

function one(g::G) where {G<:AbstractGreenFunction}
    T = eltype(g)
    bs = blocksize(g)
    ax = axis(g)
    δ = zeros(bs,bs,length(ax))
    Threads.@threads for i = 1:length(ax)
        δ[:,:,i] .= diagm(0=> [T(1) for j = 1:bs]) 
    end
    reg = spdiagm(0=>[T(0) for j = 1:(bs*length(ax))])
    return G(ax,δ,_adapt(regular(g),reg), bs)
end
