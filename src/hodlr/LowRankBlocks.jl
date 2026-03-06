import Base: view, size, eltype, *

abstract type LowRankBlock{T} end
function eltype(::Type{<:LowRankBlock{T}}) where T
    return T
end
function size(A::LowRankBlock)::Tuple{Int,Int}
    return (size(A, 1), size(A, 2))
end
function (-)(A::LowRankBlock)
    return -1 * A
end
#Subtypes are expectd to implement size(A, i) for i in 1:2
#Subtypes are expected to implement full(A)
#Subtypes are expected to implement view(A, i, j)
#Subtypes are expectd to * operator for Array on both sides
#Subtypes are expectd to * operator for scalar on left
#Subtypes are expectd to * operator for I operator on left and right

struct SumBlock{T,L,R} <: LowRankBlock{T}
    left::LowRankBlock{T}
    right::LowRankBlock{T}
end
function SumBlock(left::L, right::R) where {L,R}
    if eltype(left) != eltype(right)
        throw(ArgumentError("Incompatible element types, left has eltype $(eltype(left)) and right has eltype $(eltype(right))."))
    end
    if size(left) != size(right)
        throw(ArgumentError("Incompatible sizes, left has size $(size(left)) and right has size $(size(right))."))
    end
    return SumBlock{eltype(left),L,R}(left, right)
end
function (+)(A::LowRankBlock{T}, B::LowRankBlock{T}) where T
    return SumBlock(A, B)
end
function size(A::SumBlock, i)::Int
    return size(A.left, i)
end
function full(A::SumBlock)
    return full(A.left) + full(A.right)
end


function view(A::SumBlock, i, j)
    return SumBlock(view(A.left, i, j), view(A.right, i, j))
end
function (*)(A::SumBlock, B::Union{AbstractMatrix,HodlrTree})
    return A.left * B + A.right * B
end
function (*)(A::Union{AbstractMatrix,HodlrTree}, B::SumBlock)
    return A * B.left + A * B.right
end
function (*)(A::SumBlock, B::SumBlock)
    full(A.left) * full(B.left) + full(A.left) * full(B.right) + full(A.right) * full(B.left) + full(A.right) * full(B.right)
end
function (*)(a::Number, A::SumBlock)
    return SumBlock(a * A.left, a * A.right)
end
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

function (*)(left::SvdBlock, right::Union{AbstractMatrix,HodlrTree})
    applied_v = left.v * right
    #TODO: add reorthogonalization step
    return SvdBlock(left.u, left.s, applied_v)
end
function (*)(left::SvdBlock, right::AbstractVector)
    return left.u * (left.s * (left.v * right))
end
function (*)(left::Union{AbstractMatrix,HodlrTree}, right::SvdBlock)
    #TODO: add reorthogonalization step
    new_u = left * right.u
    return SvdBlock(new_u, right.s, right.v)
end
function (*)(left::AbstractVector, right::SvdBlock)
    return ((left * right.u) * right.s) * right.v
end
function (*)(A::SvdBlock, B::SvdBlock)
    new_s = (A.s * A.v) * (B.u * B.s)
    return SvdBlock(A.u, new_s, B.v)
end
function (*)(a::Number, A::SvdBlock)
    return SvdBlock(A.u, a * A.s, A.v)
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
