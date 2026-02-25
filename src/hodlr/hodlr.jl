#TODO: get ride of Moshi dependencies, can produce subtile bugs
using Moshi.Match: @match
using Moshi.Data: @data
using Moshi.Derive: @derive
using StaticArrays
using ACAFact

include("PartitionTrees.jl")

export build_hodlr
export HodlrContext
export full

@kwdef struct HodlrContext
    tol::Real
    maxrank::Int
    rankstart::Int
    leafsize::Int = 64
end

struct LowRankBlock{M}
    u::M
    v::M
end

function LowRankBlock(kf::Union{KernelFunction,AbstractMatrix}, ctx::HodlrContext)
    LowRankBlock(kf, ctx.tol; maxrank=ctx.maxrank, rankstart=ctx.rankstart)
end
function LowRankBlock(kf::Union{KernelFunction,AbstractMatrix}, tol::Real; maxrank::Int, rankstart::Int=div(maxrank, 4))
    U, V = aca(kf, tol, maxrank=maxrank, rankstart=rankstart)
    LowRankBlock(U, V)
end
function rank(A::LowRankBlock)
    return size(A.u, 2)
end
function size(A::LowRankBlock, i)::Int
    if i == 1
        s = size(A.u, 1)
    elseif i == 2
        s = size(A.v, 1)
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
    result = A.u * A.v'
    return result
end


function (*)(A::LowRankBlock, B::AbstractArray)
    return A.u * (A.v' * B)
end
function (*)(A::AbstractArray, B::LowRankBlock)
    return (A * B.u) * B.v'
end
function (*)(A::LowRankBlock, B::LowRankBlock)
    #TODO: naive optimization, find a reference paper / implementation 
    #OPTION: lazy optimization by just storting the sets of low ranks matrices
    core = A.v' * B.u
    return LowRankBlock(A.u * core, B.v)
end

@data Holdr{M} begin
    Leaf(M)
    struct Node
        A::Holdr{M}
        B::Holdr{M}
        upper_offdiag::LowRankBlock{M}
        lower_offdiag::LowRankBlock{M}
    end
end
@derive Holdr[Hash, Eq, Show]

function isleaf(holdr::Holdr.Type)
    @match holdr begin
        Holdr.Leaf(_) => true
        Holdr.Node(_, _, _, _) => false
    end
end
function isnode(holdr::Holdr.Type)
    return !isleaf(holdr)
end
function get_children(holdr::Holdr.Type)
    @match holdr begin
        Holdr.Leaf(M) => M
        Holdr.Node(A, B, upper_offdiag, lower_offdiag) => (A, B, upper_offdiag, lower_offdiag)
    end
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
            _convert_matrix_partition_to_domain(upper_rows, right_cols, kf.domain, kf.blocksize))
        lower_offdiag_kernel = restrict_domain(kf,
            _convert_matrix_partition_to_domain(lower_row, left_cols, kf.domain, kf.blocksize))

        A = build_hodlr(kf, upper_rows, left_cols, ctx)
        B = build_hodlr(kf, lower_row, right_cols, ctx)
        upper_offdiag = LowRankBlock(upper_offdiag_kernel, ctx)
        lower_offdiag = LowRankBlock(lower_offdiag_kernel, ctx)
        return Holdr.Node(A, B, upper_offdiag, lower_offdiag)
    end
end

function build_hodlr(kf::KernelFunction, ctx::HodlrContext)
    row_partition = build_partition(1:size(kf, 1), ctx.leafsize)
    col_partition = build_partition(1:size(kf, 2), ctx.leafsize)
    return build_hodlr(kf, row_partition, col_partition, ctx)
end

function _construct_leaf(kf, row_partition, col_partition)
    restricted_kf = restrict_domain(kf, _convert_matrix_partition_to_domain(row_partition, col_partition, kf.domain, kf.blocksize))
    M = zeros(eltype(restricted_kf), size(restricted_kf, 1), size(restricted_kf, 1))
    fill_with_kernel!(M, restricted_kf)
    return Holdr.Leaf(M)
end

function size(holdr::Holdr.Type{M}) where M
    @match holdr begin
        Holdr.Leaf(M) => size(M)
        Holdr.Node(A, B, _, _) => (size(A, 1) + size(B, 1), size(A, 2) + size(B, 2))
    end
end
function size(holdr::Holdr.Type{M}, i) where M
    s = size(holdr)
    if i < 1 || i > 2
        return 1
    else
        return s[i]
    end
end

function eltype(::Holdr.Type{M}) where M
    return eltype(M)
end

function full(holdr::Holdr.Type{M}) where M
    dense_matrix = zeros(eltype(holdr), size(holdr)...)
    _full!(dense_matrix, holdr)
    return dense_matrix
end
function _full!(out, holdr::Holdr.Type)
    if isleaf(holdr)
        M = get_children(holdr)
        out[:, :] .= M
    else
        A, B, upper_offdiag, lower_offdiag = get_children(holdr)
        nA1, nA2 = size(A)
        nB1, nB2 = size(B)
        out_up = view(out, 1:nA1, 1:nA2)
        out_down = view(out, nA1+1:nA1+nB1, nA2+1:nA2+nB2)
        _full!(out_up, A)
        _full!(out_down, B)
        out[1:nA1, nA2+1:end] .= full(upper_offdiag)
        out[nA1+1:end, 1:nA2] .= full(lower_offdiag)
    end
end


