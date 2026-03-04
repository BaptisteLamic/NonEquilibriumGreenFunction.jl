abstract type LowRankBlock{T} end
function eltype(::Type{<:LowRankBlock{T}}) where T
    return T
end
function size(A::LowRankBlock)::Tuple{Int,Int}
    return (size(A, 1), size(A, 2))
end
#Subtypes are expectd to implement size(A, i) for i in 1:2
#Subtypes are expectd to * operator for Array on both sides
#Subtypes are expected to implement full(A)
#Subtypes are expected to implement view(A, i, j)

struct SvdBlock{T,L,D,R} <: LowRankBlock{T}
    u::L
    s::D
    v::R
end
function SvdBlock(u::L, s::D, v::R) where {L,D,R}
    if eltype(u) != eltype(v)
        throw(ArgumentError("Incompatible element types, u has eltype $(eltype(s)) and v has eltype $(eltype(v))."))
    end
    if real(eltype(u)) != real(eltype(s))
        throw(ArgumentError("Incompatible element types, u has eltype $(eltype(s)) and s has eltype $(eltype(s))."))
    end
    return SvdBlock{eltype(u),L,D,R}(u, s, v)
end

function SvdBlock(kf::Union{KernelFunction,AbstractMatrix}, ctx::HodlrContext)
    u, s, vt = _compute_lowrank_factorization(kf, ctx)
    SvdBlock(u, s, vt)
end
function SvdBlock(kf::Union{KernelFunction,AbstractMatrix}, tol::Real)
    return SvdBlock(kf, HodlrContext(tol=tol))
end

function size(A::SvdBlock, i)::Int
    if i == 1
        s = size(A.u, 1)
    elseif i == 2
        s = size(A.v, 2)
    else
        s = 1
    end
    return s
end
function full(A::SvdBlock)
    return A.u * A.s * A.v
end

function (*)(A::SvdBlock, B::AbstractArray)
    return A.u * (A.s * (A.v * B))
end
function (*)(A::AbstractArray, B::SvdBlock)
    return (A * B.u) * B.s * B.v
end
function (*)(A::SvdBlock, B::SvdBlock)
    #TODO: naive optimization, find a reference paper / implementation 
    #OPTION: lazy optimization by just storing the sets of low rank matrices
    core = (A.s * A.v) * (B.u * B.s)
    #TODO propagate error context here
    u_core, s, vt_core = _compute_lowrank_factorization(core, HodlrContext())
    u = A.u * u_core
    v = vt_core * B.v
    return SvdBlock(u, s, v)
end
function view(A::SvdBlock{M,D}, i, j) where {M,D}
    view_on_u = view(A.u, i, :)
    if ndims(view_on_u) == 1
        view_on_u = transpose(view_on_u)
    end
    return SvdBlock(view_on_u, A.s, view(A.v, :, j))
end

function _compute_lowrank_factorization(kf::Union{KernelFunction,AbstractMatrix}, ctx::HodlrContext)
    m = LinearOperator(kf)
    #TODO : tune the algorithm selection method. 
    #ISSUE: when using LinearOperator, LowRankApprox enforce to use sketch=:randn
    #TODO: we need to find a way around that to preserve complexity
    if prod(size(kf)) > ctx.sampling_threshold
        F = psvdfact(m, rtol=ctx.tol, sketch=:randn)
    else
        F = psvdfact(m, rtol=ctx.tol)
    end
    U = F[:U]
    S = Diagonal(F[:S])
    V = F[:Vt]
    return U, S, V
end