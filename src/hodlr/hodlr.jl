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

function LowRankBlock(kf, ctx::HodlrContext)
    LowRankBlock(kf, ctx.tol; maxrank=ctx.maxrank)
end
function LowRankBlock(kf, tol::Real; maxrank::Int)
    (U, V) = aca(kf, tol, maxrank=maxrank)
    LowRankBlock(U, V)
end
function rank(A::LowRankBlock)
    return size(A.u,2)
end
function size(A::LowRankBlock, i) :: Int
    if i == 1
        s = size(A.u,1)
    elseif i==2
        s = size(A.v,1)
    else
        s = 1
    end
    return s
end
function size(A::LowRankBlock) :: Tuple{Int,Int}
    return (size(A,1),size(A,2))
end

@testitem "Test LowRankBlock creation" begin
    using LinearAlgebra
    using ACAFact
    using StaticArrays
    dom = range(0.0, 1.0, length=100)
    m = @SMatrix [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.LowRankBlock(kf, 1e-6, maxrank=10)
end

function full(A::LowRankBlock)
    return A.u * A.v'
end

@testitem "Test LowRankBlock arithmetic" begin
    using LinearAlgebra
    k = 2
    n = 5
    m = 6
    u = randn(ComplexF64,n,k)
    v = randn(ComplexF64,k,m)
    A = u*v
    @test size(A) == (n,m)
    @test k == rank(A)
    aca_A = NonEquilibriumGreenFunction.LowRankBlock(A,1e-9, maxrank=10)
    @test size(aca_A) == size(A)
    @show size(aca_A.u)
    @show size(aca_A.v)
    @test NonEquilibriumGreenFunction.rank(aca_A) >= k
    @test norm(A - NonEquilibriumGreenFunction.full(aca_A))/norm(A) < 1E-8
end

function (*)(A::LowRankBlock ,B)
    return A.u*(A.v'*B)
end
function (*)(A ,B::LowRankBlock)
    return (A*B.u)*B.v'
end
function (*)(A::LowRankBlock ,B::LowRankBlock)
    #TODO: naive optimization, find a reference paper / implementation 
    #OPTION: lazy optimization by just storting the sets of low ranks matrices
    core = A.v'*B.u
    return LowRankBlock(A.u*core, B.v)
end

@testitem "Test LowRankBlock x Dense" begin
    using LinearAlgebra
    dom = range(0.0, 1.0, length=100)
    m =  [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.LowRankBlock(kf, 1e-6, maxrank=10)
    x = randn(ComplexF32, size(block,2))
    y = block*x
    y_full = NonEquilibriumGreenFunction.full(block)*x
    @test norm(y - y_full)/norm(y_full) < 1E-8
end

@testitem "Test Dense x LowRankBlock" begin
    using LinearAlgebra
    dom = range(0.0, 1.0, length=100)
    m =  [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.LowRankBlock(kf, 1e-6, maxrank=10)
    x = randn(ComplexF32, size(block,1))
    y = x'*block
    y_full = x'*NonEquilibriumGreenFunction.full(block)
    @test norm(y - y_full)/norm(y_full) < 1E-8
end

@testitem "Test LowRankBlock x LowRankBlock" begin
    using LinearAlgebra
    n,k1,m,k2,l = 100,12,80,10,100
    block1 = NonEquilibriumGreenFunction.LowRankBlock(randn(ComplexF64,n,k1), randn(ComplexF64,k1,m))
    block2 = NonEquilibriumGreenFunction.LowRankBlock(randn(ComplexF64,m,k2), randn(ComplexF64,k2,l))
    full_product = NonEquilibriumGreenFunction.full(block1)*NonEquilibriumGreenFunction.full(block2)
    block_product = block1*block2
    @test norm(full_product - NonEquilibriumGreenFunction.full(block_product))/norm(full_product) < 1E-8
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