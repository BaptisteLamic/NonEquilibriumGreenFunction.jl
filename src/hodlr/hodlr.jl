
using StaticArrays
using LowRankApprox


include("PartitionTrees.jl")

export build_hodlr
export HodlrContext
export full

import Base: size, eltype, *, view

@kwdef struct HodlrContext
    tol::Real = 1e-6
    leafsize::Int = 64
    sampling_threshold::Int = 100^2
end
include("LowRankBlocks.jl")
# T is the scalar type
abstract type Hodlr{T} end
struct LeafHodlr{T} <: Hodlr{T}
    data::Matrix{T}
end
function LeafHodlr(data)
    return LeafHodlr{eltype(data)}(data)
end
struct NodeHodlr{T} <: Hodlr{T}
    A::Hodlr{T}
    B::Hodlr{T}
    upper_offdiag::LowRankBlock{T}
    lower_offdiag::LowRankBlock{T}
end

# Helper functions to replace @match patterns
function isleaf(matrix::Hodlr)
    return isa(matrix, LeafHodlr)
end

function isnode(matrix::Hodlr)
    return isa(matrix, NodeHodlr)
end

function get_children(matrix::Hodlr)
    if isleaf(matrix)
        return matrix.data
    elseif isnode(matrix)
        return (matrix.A, matrix.B, matrix.upper_offdiag, matrix.lower_offdiag)
    else
        throw(ArgumentError("Matrix is neither a leaf nor a node"))
    end
end

function _convert_matrix_partition_to_domain(partition_row::PartitionTree, partition_col::PartitionTree, domain, bs)
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
        upper_offdiag = SvdBlock(upper_offdiag_kernel, ctx)
        lower_offdiag = SvdBlock(lower_offdiag_kernel, ctx)
        return NodeHodlr(A, B, upper_offdiag, lower_offdiag)
    end
end
function _construct_leaf(kf, row_partition::PartitionTree, col_partition::PartitionTree)
    restricted_kf = restrict_domain(kf, _convert_matrix_partition_to_domain(row_partition, col_partition, kf.domain, blocksize(kf)))
    M = zeros(eltype(restricted_kf), size(restricted_kf, 1), size(restricted_kf, 1))
    fill_with_kernel!(M, restricted_kf)
    return LeafHodlr(M)
end
function build_hodlr(kf::KernelFunction, ctx::HodlrContext)
    row_partition = build_partition(1:size(kf, 1), ctx.leafsize)
    col_partition = build_partition(1:size(kf, 2), ctx.leafsize)
    return build_hodlr(kf, row_partition, col_partition, ctx)
end

function build_hodlr(matrix::LowRankBlock, ctx::HodlrContext)
    row_partition = build_partition(1:size(matrix, 1), ctx.leafsize)
    col_partition = build_partition(1:size(matrix, 2), ctx.leafsize)
    return _build_hodlr_from_lowrank(matrix, row_partition, col_partition, ctx)
end

function _build_hodlr_from_lowrank(matrix::LowRankBlock, row_partition, col_partition, ctx::HodlrContext)
    if isleaf(row_partition) && isleaf(col_partition)
        return _construct_leaf(matrix, row_partition, col_partition)
    end
    upper_rows, lower_row = split_partition(row_partition)
    left_cols, right_cols = split_partition(col_partition)
    A = _build_hodlr_from_lowrank(matrix, upper_rows, left_cols, ctx)
    B = _build_hodlr_from_lowrank(matrix, lower_row, right_cols, ctx)
    upper_offdiag = view(matrix, get_range(upper_rows), get_range(right_cols))
    lower_offdiag = view(matrix, get_range(lower_row), get_range(left_cols))
    return NodeHodlr(A, B, upper_offdiag, lower_offdiag)
end
function _construct_leaf(matrix::LowRankBlock, row_partition::PartitionTree, col_partition::PartitionTree)
    sub_matrix = view(matrix, get_range(row_partition), get_range(col_partition))
    return LeafHodlr(full(sub_matrix))
end

function size(Hodlr::LeafHodlr{M}) where M
    return size(Hodlr.data)
end

function size(Hodlr::NodeHodlr{M}) where M
    return (size(Hodlr.A, 1) + size(Hodlr.B, 1), size(Hodlr.A, 2) + size(Hodlr.B, 2))
end

function size(Hodlr::Hodlr{M}, i) where M
    s = size(Hodlr)
    if i < 1 || i > 2
        return 1
    else
        return s[i]
    end
end

function eltype(::Hodlr{M}) where M
    return eltype(M)
end

function full(Hodlr::LeafHodlr{M}) where M
    return Hodlr.data
end

function full(Hodlr::NodeHodlr{M}) where M
    dense_matrix = zeros(eltype(Hodlr), size(Hodlr)...)
    _full!(dense_matrix, Hodlr)
    return dense_matrix
end

function _full!(out, Hodlr::LeafHodlr)
    M = Hodlr.data
    out[:, :] .= M
end

function _full!(out, Hodlr::NodeHodlr)
    A, B, upper_offdiag, lower_offdiag = Hodlr.A, Hodlr.B, Hodlr.upper_offdiag, Hodlr.lower_offdiag
    nA1, nA2 = size(A)
    nB1, nB2 = size(B)
    out_up = view(out, 1:nA1, 1:nA2)
    out_down = view(out, nA1+1:nA1+nB1, nA2+1:nA2+nB2)
    _full!(out_up, A)
    _full!(out_down, B)
    out[1:nA1, nA2+1:end] .= full(upper_offdiag)
    out[nA1+1:end, 1:nA2] .= full(lower_offdiag)
end

function (*)(left::T, x::M) where {T<:Hodlr,M<:AbstractArray}
    _throw_error_if_incompatible_size(left, x)
    out = zeros(eltype(left), size(left, 1), size(x, 2))
    _apply_right_mul!(out, left, x)
    return out
end
function _apply_right_mul!(out, Hodlr::LeafHodlr, x)
    M = Hodlr.data
    out[:, :] .= M * x
end
function _apply_right_mul!(out, Hodlr::NodeHodlr, x)
    A, B, upper_offdiag, lower_offdiag = Hodlr.A, Hodlr.B, Hodlr.upper_offdiag, Hodlr.lower_offdiag
    nA1, nA2 = size(A)
    nB1, nB2 = size(B)
    x_up = view(x, 1:nA2, :)
    x_down = view(x, nA2+1:nA2+nB2, :)
    out_up = view(out, 1:nA1, :)
    out_down = view(out, nA1+1:nA1+nB1, :)
    _apply_right_mul!(out_up, A, x_up)
    _apply_right_mul!(out_down, B, x_down)
    #TODO : optimize by avoiding forming the full matrix
    out_up[:, :] .+= full(upper_offdiag * x_down)
    out_down[:, :] .+= full(lower_offdiag * x_up)
end

function (*)(x::M, right::T) where {T<:Hodlr,M<:AbstractArray}
    _throw_error_if_incompatible_size(x, right)
    out = zeros(eltype(right), size(x, 1), size(right, 2))
    _apply_left_mul_vector_1D!(out, x, right)
    return out
end
function _apply_left_mul_vector_1D!(out, x, Hodlr::LeafHodlr)
    M = Hodlr.data
    out[:, :] .= x * M
end
function _apply_left_mul_vector_1D!(out, x, Hodlr::NodeHodlr)
    A, B, upper_offdiag, lower_offdiag = Hodlr.A, Hodlr.B, Hodlr.upper_offdiag, Hodlr.lower_offdiag
    nA1, nA2 = size(A)
    nB1, nB2 = size(B)
    x_up = view(x, :, 1:nA1)
    x_down = view(x, :, nA1+1:nA1+nB1)
    out_up = view(out, :, 1:nA2)
    out_down = view(out, :, nA2+1:nA2+nB2)
    _apply_left_mul_vector_1D!(out_up, x_up, A)
    _apply_left_mul_vector_1D!(out_down, x_down, B)
    #TODO : optimize by avoiding forming the full matrix
    out_up[:, :] .+= full(x_down * lower_offdiag)
    out_down[:, :] .+= full(x_up * upper_offdiag)
end

function _throw_error_if_incompatible_size(left, right)
    if size(left, 2) != size(right, 1)
        throw(DimensionMismatch("Non compatible dimensions for multiplication : left has size $(size(left)) and right has size $(size(right))."))
    end
end