

@testitem "Test LowRankBlock creation" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=100)
    m = [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.LowRankBlock(kf, 1e-6)
end

@testitem "Test LowRankBlock creation from vectors" begin
    using LinearAlgebra
    k = 5
    n = 128
    m = 243
    u0 = randn(ComplexF64, n, k)
    v0 = randn(ComplexF64, k, m)
    A = u0 * v0
    @test size(A) == (n, m)
    @test k == rank(A)
    tol = 1E-8
    aca_A = NonEquilibriumGreenFunction.LowRankBlock(A, 0.1 * tol)
    @test size(aca_A) == size(A)
    @test NonEquilibriumGreenFunction.rank(aca_A) >= k
    @test norm(A - NonEquilibriumGreenFunction.full(aca_A)) / norm(A) < tol
    @test norm(A - NonEquilibriumGreenFunction.full(aca_A)) < tol
end

@testitem "Test LowRankBlock creation from KernelFunction" begin
    # NOTE: ACA approximation of block-structured kernels may fail due to discontinuities at block boundaries.
    # The kernel's piecewise structure violates smoothness assumptions underlying the adaptive cross approximation.
    # Hence this test fail
    using LinearAlgebra
    import NonEquilibriumGreenFunction.LowRankBlock
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    m = [1 2; 1 1]
    tol = 1E-8
    kf = NonEquilibriumGreenFunction.KernelFunction((x, y) -> m .* exp(1im * (x - y)), dom)
    aca_block = NonEquilibriumGreenFunction.LowRankBlock(kf, 0.01 * tol)
    full_block = zeros(eltype(aca_block), size(aca_block)...)
    NonEquilibriumGreenFunction.fill_with_kernel!(full_block, kf)
    @test eltype(full_block) == eltype(aca_block)
    @test eltype(full_block) == eltype(kf)
    #TODO fix ACA for non block-structured kernels.
    @test norm(full_block - full(aca_block)) < tol
    @test rank(full_block) == rank(full(aca_block))
    @test norm(full(LowRankBlock(full_block, 1E-2 * tol)) - full_block) < tol
end

@testitem "LowRankBlock view" begin
    using LinearAlgebra
    n, k, m = 128, 5, 64
    block = NonEquilibriumGreenFunction.LowRankBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    @test size(view(block, :, 1:2)) == (size(block, 1), 2)
    @test size(view(block, 1, :)) == (1, size(block, 2))
    @test size(view(block, :, 1)) == (size(block, 1), 1)
    @test size(view(block, 1:2, :)) == (2, size(block, 2))
    @test size(view(block, 1:2, 1:2)) == (2, 2)
end

@testitem "Test LowRankBlock x Dense Vector" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=100)
    m = [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.LowRankBlock(kf, 1e-6)
    x = randn(ComplexF32, size(block, 2))
    y = block * x
    y_full = NonEquilibriumGreenFunction.full(block) * x
    @test norm(y - y_full) / norm(y_full) < 1E-8
end

@testitem "Test LowRankBlock x Dense Matrix" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=100)
    m = [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.LowRankBlock(kf, 1e-6)
    x = randn(ComplexF32, (size(block, 2), 4))
    y = block * x
    y_full = NonEquilibriumGreenFunction.full(block) * x
    @test norm(y - y_full) / norm(y_full) < 1E-8
end

@testitem "Test Dense Vector x LowRankBlock" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=100)
    m = [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.LowRankBlock(kf, 1e-6)
    x = randn(ComplexF32, size(block, 1))
    y = x' * block
    y_full = x' * NonEquilibriumGreenFunction.full(block)
    @test norm(y - y_full) / norm(y_full) < 1E-8
end

@testitem "Test Dense Matrix x LowRankBlock" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=100)
    m = [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.LowRankBlock(kf, 1e-6)
    x = randn(ComplexF32, (12, size(block, 1)))
    y = x * block
    y_full = x * NonEquilibriumGreenFunction.full(block)
    @test norm(y - y_full) / norm(y_full) < 1E-8
end


@testitem "Test LowRankBlock x LowRankBlock" begin
    using LinearAlgebra
    n, k1, m, k2, l = 100, 12, 80, 10, 100
    block1 = NonEquilibriumGreenFunction.LowRankBlock(randn(ComplexF64, n, k1), Diagonal(randn(Float64, k1)), randn(ComplexF64, k1, m))
    block2 = NonEquilibriumGreenFunction.LowRankBlock(randn(ComplexF64, m, k2), Diagonal(randn(Float64, k2)), randn(ComplexF64, k2, l))
    full_block1 = NonEquilibriumGreenFunction.full(block1)
    full_block2 = NonEquilibriumGreenFunction.full(block2)
    full_product = full_block1 * full_block2
    block_product = block1 * block2
    @test norm(full_product - NonEquilibriumGreenFunction.full(block_product)) / norm(full_product) < 1E-8
end



@testitem "Test Hodlr construction" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    m = ones(2, 2)
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    holdr = build_hodlr(kf, HodlrContext(tol=1e-6, leafsize=size(kf, 1) ÷ 2))
    @test size(holdr) == size(kf)
end


@testitem "Test Hodlr full" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    m = [1 2; 1 1]
    const tol = 1E-9
    kf = KernelFunction((x, y) -> m .* exp(1im * (x - y)), dom)
    holdr = build_hodlr(kf, HodlrContext(tol=0.01 * tol, leafsize=64))
    full_hodlr = full(holdr)
    dense = zeros(eltype(holdr), size(holdr)...)
    NonEquilibriumGreenFunction.fill_with_kernel!(dense, kf)
    @test norm(dense - full_hodlr) / norm(dense) < tol
    @test norm(dense - full_hodlr) < tol
end

@testitem "Hodlr product with vector" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    m = [1 2; 1 1]
    kf = KernelFunction((x, y) -> m .* exp(1im * (x - y)), dom)
    ctx = HodlrContext()
    holdr = build_hodlr(kf, ctx)
    full_hodlr = full(holdr)
    x = randn(eltype(holdr), size(holdr, 2))
    y_hodlr = holdr * x
    y_full = full_hodlr * x
    @test norm(y_hodlr - y_full) / norm(y_full) < 10 * ctx.tol
    y_full = x' * full_hodlr
    @test norm(x' * holdr - y_full) / norm(y_full) < 10 * ctx.tol
end

@testitem "Hodlr product with array" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    m = [1 2; 1 1]
    kf = KernelFunction((x, y) -> m .* exp(1im * (x - y)), dom)
    ctx = HodlrContext()
    holdr = build_hodlr(kf, ctx)
    full_hodlr = full(holdr)
    x = randn(eltype(holdr), size(holdr, 2), 12)
    y_full = full_hodlr * x
    y_hodlr = holdr * x
    @test size(y_hodlr) == size(y_full)
    @test norm(y_hodlr - y_full) / norm(y_full) < 10 * ctx.tol
    y_full = x' * full_hodlr
    @test norm(x' * holdr - y_full) / norm(y_full) < 10 * ctx.tol
end

@testitem "Hodlr product with LowRankBlock" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    kf = KernelFunction((x, y) -> [1 2; 1 1] .* exp(1im * (x - y)), dom)
    ctx = HodlrContext()
    holdr = build_hodlr(kf, ctx)
    k, m = 12, 80
    low_rank_block = NonEquilibriumGreenFunction.LowRankBlock(randn(ComplexF64, size(holdr, 2), k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    holder_product = holdr * low_rank_block
    full_product = full(holdr) * NonEquilibriumGreenFunction.full(low_rank_block)
    @test size(holder_product) == size(full_product)
    @test norm(full_product - NonEquilibriumGreenFunction.full(holder_product)) / norm(full_product) < 10 * ctx.tol
end
@testitem "LowRankBlock product with Hodlr" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    kf = KernelFunction((x, y) -> [1 2; 1 1] .* exp(1im * (x - y)), dom)
    ctx = HodlrContext()
    holdr = build_hodlr(kf, ctx)
    k, m = 12, 80
    low_rank_block = NonEquilibriumGreenFunction.LowRankBlock(randn(ComplexF64, m, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, size(holdr, 1)))
    holder_product = low_rank_block * holdr
    full_product = NonEquilibriumGreenFunction.full(low_rank_block) * full(holdr)
    @test size(holder_product) == size(full_product)
    @test norm(full_product - NonEquilibriumGreenFunction.full(holder_product)) / norm(full_product) < 10 * ctx.tol
end
@testitem "construct Hodlr from LowRankBlock" begin
    using LinearAlgebra
    n, k, m = 512, 12, 512
    low_rank_block = NonEquilibriumGreenFunction.LowRankBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    holdr = build_hodlr(low_rank_block, HodlrContext(tol=1E-9, leafsize=64))
    @test size(holdr) == size(low_rank_block)
    @test norm(full(holdr) * full(low_rank_block) - full(holdr) * full(low_rank_block)) / norm(full(holdr) * full(low_rank_block)) < 10 * ctx.tol
end