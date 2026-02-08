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
end

struct LowRankBlock{M}
    u::M
    v::M
end

function LowRankBlock(kf, ctx::HodlrContext)
    LowRankBlock(kf, ctx.tol; maxrank=ctx.maxrank, rankstart=ctx.rankstart)
end
function LowRankBlock(kf, tol::Real; maxrank::Int, rankstart::Int = div(maxrank,4))
    U, V = aca(kf, maxrank)
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
function eltype(::LowRankBlock{M}) where M
    return eltype(M)
end

@testitem "Test LowRankBlock creation" begin
    using LinearAlgebra
    dom = range(0.0, 1.0, length=100)
    m = [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.LowRankBlock(kf, 1e-6, maxrank=10)
end

function full(A::LowRankBlock)
    result = A.u * A.v'
    return result
end

@testitem "Test LowRankBlock creation from vectors" begin
    using LinearAlgebra
    k = 5
    n = 128
    m = 243
    u0 = randn(ComplexF64,n,k)
    v0 = randn(ComplexF64,k,m)
    A = u0*v0
    @test size(A) == (n,m)
    @test k == rank(A)
    const tol = 1E-8
    aca_A = NonEquilibriumGreenFunction.LowRankBlock(A,0.1*tol, maxrank=10)
    @test size(aca_A) == size(A)
    @test NonEquilibriumGreenFunction.rank(aca_A) >= k
    @test norm(A - NonEquilibriumGreenFunction.full(aca_A))/norm(A) < tol
    @test norm(A - NonEquilibriumGreenFunction.full(aca_A)) < tol
end

@testitem "Test LowRankBlock creation from KernelFunction" begin
    using LinearAlgebra
    using ACAFact
    import NonEquilibriumGreenFunction.LowRankBlock
    dom = range(0.0, 1.0, length=3)
    m =  [1 2; 1 1]
    m = m + m'
    const tol = 1E-9
    kf = NonEquilibriumGreenFunction.KernelFunction((x, y) -> m .* exp(1im*(x - y)), dom)
    aca_block = NonEquilibriumGreenFunction.LowRankBlock(kf, 0.01*tol, maxrank = 60)
    full_block = zeros(eltype(aca_block),size(aca_block)...)
    NonEquilibriumGreenFunction.fill_with_kernel!(full_block,kf)
    @test eltype(full_block) == eltype(aca_block)
    @test eltype(full_block) == eltype(kf)
    @test norm(full_block - full(aca_block)) < tol
    @test rank(full_block) == rank(full(aca_block))
    @test norm(full(LowRankBlock(full_block, 1E-2 * tol, maxrank = 20, rankstart = 12)) - full_block ) < tol 
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

@testitem "Test LowRankBlock x Dense Vector" begin
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

@testitem "Test LowRankBlock x Dense Matrix" begin
    using LinearAlgebra
    dom = range(0.0, 1.0, length=100)
    m =  [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.LowRankBlock(kf, 1e-6, maxrank=10)
    x = randn(ComplexF32, (size(block,2),4))
    y = block*x
    y_full = NonEquilibriumGreenFunction.full(block)*x
    @test norm(y - y_full)/norm(y_full) < 1E-8
end

@testitem "Test Dense Vector x LowRankBlock" begin
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

@testitem "Test Dense Matrix x LowRankBlock" begin
    using LinearAlgebra
    dom = range(0.0, 1.0, length=100)
    m =  [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.LowRankBlock(kf, 1e-6, maxrank=10)
    x = randn(ComplexF32, (12,size(block,1)))
    y = x*block
    y_full = x*NonEquilibriumGreenFunction.full(block)
    @test norm(y - y_full)/norm(y_full) < 1E-8
end


@testitem "Test LowRankBlock x LowRankBlock" begin
    using LinearAlgebra
    n,k1,m,k2,l = 100,12,80,10,100
    block1 = NonEquilibriumGreenFunction.LowRankBlock(randn(ComplexF64,n,k1), randn(ComplexF64,m,k1))
    block2 = NonEquilibriumGreenFunction.LowRankBlock(randn(ComplexF64,m,k2), randn(ComplexF64,l,k2))
    full_block1 = NonEquilibriumGreenFunction.full(block1)
    full_block2 = NonEquilibriumGreenFunction.full(block2)
    full_product = full_block1*full_block2
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

function isleaf(holdr::Holdr.Type)
    @match holdr begin
        Holdr.Leaf(_) => true
        Holdr.Node(_,_,_,_) => false
    end
end
function isnode(holdr::Holdr.Type)
    return !isleaf(holdr)
end
function get_children(holdr::Holdr.Type)
    @match holdr begin
        Holdr.Leaf(M) => M
        Holdr.Node(A,B,upper_offdiag,lower_offdiag) => (A,B,upper_offdiag,lower_offdiag)
    end
end

function build_hodlr(kf::KernelFunction, row_partition::PartitionTree, col_partition::PartitionTree, ctx::HodlrContext)
    if isleaf(row_partition) && isleaf(col_partition)
        return _construct_leaf(kf)
    else
        upper_rows,lower_row = split_partition(row_partition)
        left_cols, right_cols = split_partition(col_partition)

        A_kernel = update_domain(kf,upper_rows |> get_range , left_cols |> get_range)
        B_kernel = update_domain(kf,lower_row |> get_range , right_cols |> get_range )
        upper_offdiag_kernel = update_domain(kf,upper_rows |> get_range, right_cols |> get_range)
        lower_offdiag_kernel = update_domain(kf,lower_row |> get_range, left_cols |> get_range)

        A = build_hodlr(A_kernel, upper_rows, left_cols, ctx)
        B = build_hodlr(B_kernel, lower_row, right_cols, ctx)
        upper_offdiag = LowRankBlock(upper_offdiag_kernel, ctx)
        lower_offdiag = LowRankBlock(lower_offdiag_kernel, ctx)
        return Holdr.Node(A,B,upper_offdiag,lower_offdiag)
    end
end

function build_hodlr(kf::KernelFunction,ctx::HodlrContext)
    row_partition = PartitionTree(1:size(kf,1))
    col_partition = PartitionTree(1:size(kf,2))
    return build_hodlr(kf, row_partition, col_partition, ctx)
end

function _construct_leaf(kf)
    M = zeros(eltype(kf),size(kf)...)
    M = Array{eltype(kf)}(0, size(kf)...)
    fill_with_kernel!(M, kf)
    return Holdr.Leaf(M)
end

function size(holdr::Holdr.Type{M}) where M
    @match holdr begin
        Holdr.Leaf(M) => size(M)
        Holdr.Node(A,B,_,_) => (size(A,1)+size(B,1), size(A,2)+size(B,2))
    end
end
function size(holdr::Holdr.Type{M},i) where M
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

@testitem "Test Hodlr construction" begin
    using LinearAlgebra
    dom = range(0.0, 1.0, length=100)
    m = [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    holdr = build_hodlr(kf, HodlrContext(tol = 1e-6, maxrank = 60, rankstart = 20))
    @test size(holdr) == size(kf) .* size(m) 
end

function full(holdr::Holdr.Type{M}) where M
    dense_matrix = zeros(eltype(holdr),size(holdr)...)
    _full!(dense_matrix, holdr)
    return dense_matrix
end
function _full!(out, holdr::Holdr.Type)
    if isleaf(holdr)
            M = get_children(holdr)
            out[:,:] .= M
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

@testitem "Test Hodlr full" begin
    using LinearAlgebra
    dom = range(0.0, 1.0, length=512)
    m =  [1 1; 1 1]
    const tol = 1E-9
    kf = KernelFunction((x, y) -> m .* exp(1im*(x - y)), dom)
    holdr = build_hodlr(kf, HodlrContext(tol = 0.01*tol, maxrank = 4, rankstart = 20))
    full_hodlr = full(holdr)
    dense = zeros(eltype(holdr),size(holdr)...)
    NonEquilibriumGreenFunction.fill_with_kernel!(dense,kf)
    @test norm(dense - full_hodlr)/norm(dense) < tol
    @test norm(dense - full_hodlr) < tol
end
