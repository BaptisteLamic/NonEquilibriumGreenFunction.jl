
using StaticArrays
using LowRankApprox


include("PartitionTrees.jl")

export build_hodlr
export HodlrContext
export full

@kwdef struct HodlrContext
    tol::Real = 1e-6
    leafsize::Int = 64
end

struct LowRankBlock{M,D}
    u::M
    s::D
    v::M
end

function LowRankBlock(kf::Union{KernelFunction,AbstractMatrix}, ctx::HodlrContext)
    LowRankBlock(kf, ctx.tol)
end
function LowRankBlock(kf::Union{KernelFunction,AbstractMatrix}, tol::Real)
    #TODO: use randomized SVD for large blocks, but for now we can just use a dense SVD since the blocks are small.
    u, s, vt = _compute_lowrank_factorization(kf, tol)
    LowRankBlock(u, s, vt)
end
function _compute_lowrank_factorization(kf::Union{KernelFunction,AbstractMatrix}, tol::Real)
    m = kf[:, :] #TODO: not build the full matrix
    F = psvdfact(m, rtol=tol)
    U = F[:U]
    S = Diagonal(F[:S])
    V = F[:Vt]
    return U, S, V
end
function rank(A::LowRankBlock)
    return size(A.u, 2)
end
function size(A::LowRankBlock, i)::Int
    if i == 1
        s = size(A.u, 1)
    elseif i == 2
        s = size(A.v, 2)
    else
        s = 1
    end
    return s
end
function size(A::LowRankBlock)::Tuple{Int,Int}
    return (size(A, 1), size(A, 2))
end
function eltype(::LowRankBlock{M}) where M
    return eltype(M)
end

function full(A::LowRankBlock)
    result = A.u * A.s * A.v
    return result
end

function (*)(A::LowRankBlock, B::AbstractArray)
    return A.u * (A.s * (A.v * B))
end
function (*)(A::AbstractArray, B::LowRankBlock)
    return (A * B.u) * B.s * B.v
end
function (*)(A::LowRankBlock, B::LowRankBlock)
    #TODO: naive optimization, find a reference paper / implementation 
    #OPTION: lazy optimization by just storing the sets of low rank matrices
    core = (A.s * A.v) * (B.u * B.s)
    u_core, s, vt_core = _compute_lowrank_factorization(core, 1e-12)
    u = A.u * u_core
    v = vt_core * B.v
    return LowRankBlock(u, s, v)
end
abstract type Holdr{M} end

struct LeafHoldr{M} <: Holdr{M}
    data::M
end

struct NodeHoldr{M} <: Holdr{M}
    A::Holdr{M}
    B::Holdr{M}
    upper_offdiag::LowRankBlock{M}
    lower_offdiag::LowRankBlock{M}
end

# Helper functions to replace @match patterns
function isleaf(holdr::Holdr)
    return isa(holdr, LeafHoldr)
end

function isnode(holdr::Holdr)
    return isa(holdr, NodeHoldr)
end

function get_children(holdr::LeafHoldr)
    return holdr.data
end

function get_children(holdr::NodeHoldr)
    return (holdr.A, holdr.B, holdr.upper_offdiag, holdr.lower_offdiag)
end

function _convert_matrix_partition_to_domain(partition_row, partition_col, domain, bs)
    range = get_range(partition_row)
    converted_partition_row = div(range[1] - 1, bs)+1:div(range[end] - 1, bs)+1
    range = get_range(partition_col)
    converted_partition_col = div(range[1] - 1, bs)+1:div(range[end] - 1, bs)+1
    raw_domain_row = xaxis(domain)[converted_partition_row]
    raw_domain_col = yaxis(domain)[converted_partition_col]
    return KernelDomain((raw_domain_row[1], raw_domain_row[end]), (raw_domain_col[1], raw_domain_col[end]), x_steps=length(raw_domain_row), y_steps=length(raw_domain_col))
end

function build_hodlr(kf::KernelFunction, row_partition::PartitionTree, col_partition::PartitionTree, ctx::HodlrContext)
    if isleaf(row_partition) && isleaf(col_partition)
        return _construct_leaf(kf, row_partition, col_partition)
    else
        _xaxis = xaxis(kf.domain)
        _yaxis = yaxis(kf.domain)
        upper_rows, lower_row = split_partition(row_partition)
        left_cols, right_cols = split_partition(col_partition)
        upper_offdiag_kernel = restrict_domain(kf,
            _convert_matrix_partition_to_domain(upper_rows, right_cols, kf.domain, blocksize(kf)))
        lower_offdiag_kernel = restrict_domain(kf,
            _convert_matrix_partition_to_domain(lower_row, left_cols, kf.domain, blocksize(kf)))

        A = build_hodlr(kf, upper_rows, left_cols, ctx)
        B = build_hodlr(kf, lower_row, right_cols, ctx)
        upper_offdiag = LowRankBlock(upper_offdiag_kernel, ctx)
        lower_offdiag = LowRankBlock(lower_offdiag_kernel, ctx)
        return NodeHoldr(A, B, upper_offdiag, lower_offdiag)
    end
end

function build_hodlr(kf::KernelFunction, ctx::HodlrContext)
    row_partition = build_partition(1:size(kf, 1), ctx.leafsize)
    col_partition = build_partition(1:size(kf, 2), ctx.leafsize)
    return build_hodlr(kf, row_partition, col_partition, ctx)
end

function _construct_leaf(kf, row_partition, col_partition)
    restricted_kf = restrict_domain(kf, _convert_matrix_partition_to_domain(row_partition, col_partition, kf.domain, blocksize(kf)))
    M = zeros(eltype(restricted_kf), size(restricted_kf, 1), size(restricted_kf, 1))
    fill_with_kernel!(M, restricted_kf)
    return LeafHoldr(M)
end

function size(holdr::LeafHoldr{M}) where M
    return size(holdr.data)
end

function size(holdr::NodeHoldr{M}) where M
    return (size(holdr.A, 1) + size(holdr.B, 1), size(holdr.A, 2) + size(holdr.B, 2))
end

function size(holdr::Holdr{M}, i) where M
    s = size(holdr)
    if i < 1 || i > 2
        return 1
    else
        return s[i]
    end
end

function eltype(::Holdr{M}) where M
    return eltype(M)
end

function full(holdr::LeafHoldr{M}) where M
    return holdr.data
end

function full(holdr::NodeHoldr{M}) where M
    dense_matrix = zeros(eltype(holdr), size(holdr)...)
    _full!(dense_matrix, holdr)
    return dense_matrix
end

function _full!(out, holdr::LeafHoldr)
    M = holdr.data
    out[:, :] .= M
end

function _full!(out, holdr::NodeHoldr)
    A, B, upper_offdiag, lower_offdiag = holdr.A, holdr.B, holdr.upper_offdiag, holdr.lower_offdiag
    nA1, nA2 = size(A)
    nB1, nB2 = size(B)
    out_up = view(out, 1:nA1, 1:nA2)
    out_down = view(out, nA1+1:nA1+nB1, nA2+1:nA2+nB2)
    _full!(out_up, A)
    _full!(out_down, B)
    out[1:nA1, nA2+1:end] .= full(upper_offdiag)
    out[nA1+1:end, 1:nA2] .= full(lower_offdiag)
end

function (*)(holdr::Holdr, x::AbstractArray)
    if size(x, 1) != size(holdr, 2)
        throw(DimensionMismatch("The number of rows of x must match the number of columns of the Holdr."))
    end
    out = zeros(eltype(holdr), size(holdr, 1), size(x, 2))
    _apply_right_mul!(out, holdr, x)
    return out
end
function _apply_right_mul!(out, holdr::LeafHoldr, x::AbstractArray)
    M = holdr.data
    out[:, :] .= M * x
end
function _apply_right_mul!(out, holdr::NodeHoldr, x::AbstractArray)
    A, B, upper_offdiag, lower_offdiag = holdr.A, holdr.B, holdr.upper_offdiag, holdr.lower_offdiag
    nA1, nA2 = size(A)
    nB1, nB2 = size(B)
    x_up = view(x, 1:nA2, :)
    x_down = view(x, nA2+1:nA2+nB2, :)
    out_up = view(out, 1:nA1, :)
    out_down = view(out, nA1+1:nA1+nB1, :)
    _apply_right_mul!(out_up, A, x_up)
    _apply_right_mul!(out_down, B, x_down)
    out_up[:, :] .+= upper_offdiag * x_down
    out_down[:, :] .+= lower_offdiag * x_up
end

function (*)(x::AbstractArray, holdr::Holdr)
    if size(x, 2) != size(holdr, 1)
        throw(DimensionMismatch("The length of the vector must match the number of rows of the Holdr."))
    end
    out = zeros(eltype(holdr), size(x, 1), size(holdr, 2))
    _apply_left_mul_vector_1D!(out, x, holdr)
    return out
end
function _apply_left_mul_vector_1D!(out, x, holdr::LeafHoldr)
    M = holdr.data
    out[:, :] .= x * M
end
function _apply_left_mul_vector_1D!(out, x, holdr::NodeHoldr)
    A, B, upper_offdiag, lower_offdiag = holdr.A, holdr.B, holdr.upper_offdiag, holdr.lower_offdiag
    nA1, nA2 = size(A)
    nB1, nB2 = size(B)
    x_up = view(x, :, 1:nA1)
    x_down = view(x, :, nA1+1:nA1+nB1)
    out_up = view(out, :, 1:nA2)
    out_down = view(out, :, nA2+1:nA2+nB2)
    _apply_left_mul_vector_1D!(out_up, x_up, A)
    _apply_left_mul_vector_1D!(out_down, x_down, B)
    out_up[:, :] .+= x_down * lower_offdiag
    out_down[:, :] .+= x_up * upper_offdiag
end