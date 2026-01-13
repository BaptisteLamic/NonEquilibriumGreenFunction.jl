using Moshi.Match: @match
using Moshi.Data: @data
using Moshi.Derive: @derive
using StaticArrays
using ACAFact

include("PartitionTrees.jl")

export Hodlr
export HodlrContext

@kwdef struct HodlrContext
    tol::Real
    maxrank::Int
end

struct LowRankBlock{M}
    u::M
    v::M
end

function LowRankBlock(kf::KernelFunction, ctx::HodlrContext)
    LowRankBlock(kf, ctx.tol; maxrank=ctx.maxrank)
end
function LowRankBlock(kf::KernelFunction, tol::Real; maxrank::Int)
    (U, V) = aca(kf, tol, maxrank=maxrank)
    LowRankBlock(U, V)
end


@testitem "Test LowRankBlock" begin
    using LinearAlgebra
    using ACAFact
    using StaticArrays
    dom = range(0.0, 1.0, length=100)
    m = @SMatrix [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.LowRankBlock(kf, 1e-6, maxrank=10)
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

function Hodlr(kf::KernelFunction, row_partition::PartitionTree, col_partition::PartitionTree, ctx::HodlrContext)
    if is_leaf(row_partition) && is_leaf(col_partition)
        return _construct_leaf(kf)
    else
        upper_rows,lower_row = split_partition(row_partition)
        left_cols, right_cols = split_partition(col_partition)

        A_kernel = update_domain(kf,upper_rows |> get_range , left_cols |> get_range)
        B_kernel = update_domain(kf,lower_row |> get_range , right_cols |> get_range )
        upper_offdiag_kernel = update_domain(kf,upper_rows |> get_range, right_cols |> get_range)
        lower_offdiag_kernel = update_domain(kf,lower_row |> get_range, left_cols |> get_range)

        A = Hodlr(A_kernel, upper_rows, left_cols, ctx)
        B = Hodlr(B_kernel, lower_row, right_cols, ctx)
        upper_offdiag = LowRankBlock(upper_offdiag_kernel, ctx)
        lower_offdiag = LowRankBlock(lower_offdiag_kernel, ctx)
        return Holdr.Node(A,B,upper_offdiag,lower_offdiag)
    end

end

function Hodlr(kf::KernelFunction,ctx::HodlrContext)
    row_partition = PartitionTree(1:size(kf,1))
    col_partition = PartitionTree(1:size(kf,2))
    return Hodlr(kf, row_partition, col_partition, ctx)
end

function _construct_leaf(kf)
    M = Array{eltype(kf)}(undef, size(kf)...)
    fill_with_kernel!(M, kf)
    return Holdr.Leaf(M)
end

@testitem "Test Hodlr construction" begin
    using LinearAlgebra
    using ACAFact
    using StaticArrays
    dom = range(0.0, 1.0, length=100)
    m = @SMatrix [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    holdr = Hodlr(kf, HodlrContext(tol = 1e-6, maxrank = 20))
end