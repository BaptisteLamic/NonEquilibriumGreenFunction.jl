

@testitem "Test SvdBlock creation" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=100)
    m = [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.SvdBlock(kf, 1e-6)
end

@testitem "Test SvdBlock creation from vectors" begin
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
    block = NonEquilibriumGreenFunction.SvdBlock(A, 0.1 * tol)
    @test size(block) == size(A)
    @test norm(A - NonEquilibriumGreenFunction.full(block)) / norm(A) < tol
    @test norm(A - NonEquilibriumGreenFunction.full(block)) < tol
end

@testitem "Test SvdBlock creation from KernelFunction" begin
    # NOTE: ACA approximation of block-structured kernels may fail due to discontinuities at block boundaries.
    # The kernel's piecewise structure violates smoothness assumptions underlying the adaptive cross approximation.
    # Hence this test fail
    using LinearAlgebra
    import NonEquilibriumGreenFunction.SvdBlock
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    m = [1 2; 1 1]
    tol = 1E-8
    kf = NonEquilibriumGreenFunction.KernelFunction((x, y) -> m .* exp(1im * (x - y)), dom)
    svd_block = NonEquilibriumGreenFunction.SvdBlock(kf, 0.01 * tol)
    full_block = zeros(eltype(svd_block), size(svd_block)...)
    NonEquilibriumGreenFunction.fill_with_kernel!(full_block, kf)
    @test eltype(full_block) == eltype(svd_block)
    @test eltype(full_block) == eltype(kf)
    #TODO fix ACA for non block-structured kernels.
    @test norm(full_block - full(svd_block)) < tol
    @test rank(full_block) == rank(full(svd_block))
    @test norm(full(SvdBlock(full_block, 1E-2 * tol)) - full_block) < tol
end

@testitem "Test SvdBlock creation from  sparse matrix" begin
    using LinearAlgebra
    using SparseArrays
    tol = 1E-8
    sparse_matrix = spdiagm(12 => randn(128))
    svd_block = NonEquilibriumGreenFunction.SvdBlock(sparse_matrix, 0.01 * tol)
    @show typeof(svd_block)
end

@testitem "SvdBlock view" begin
    using LinearAlgebra
    n, k, m = 128, 5, 64
    block = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    @test size(view(block, :, 1:2)) == (size(block, 1), 2)
    @test size(view(block, 1, :)) == (1, size(block, 2))
    @test size(view(block, :, 1)) == (size(block, 1), 1)
    @test size(view(block, 1:2, :)) == (2, size(block, 2))
    @test size(view(block, 1:2, 1:2)) == (2, 2)
end

@testitem "Test SvdBlock x Dense Vector" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=100)
    m = [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.SvdBlock(kf, 1e-6)
    x = randn(ComplexF32, size(block, 2))
    y = block * x
    y_full = NonEquilibriumGreenFunction.full(block) * x
    @test norm(y - y_full) / norm(y_full) < 1E-8
end

@testitem "Test SvdBlock x Dense Matrix" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=100)
    m = [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.SvdBlock(kf, 1e-6)
    x = randn((size(block, 2), 4))
    y = block * x
    y_full = NonEquilibriumGreenFunction.full(block) * x
    @test norm(full(y) - y_full) / norm(y_full) < 1E-8
end

@testitem "Test Dense Vector x SvdBlock" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=100)
    m = [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.SvdBlock(kf, 1e-6)
    x = randn(size(block, 1))
    y = x' * block
    y_full = x' * NonEquilibriumGreenFunction.full(block)
    @test norm(full(y) - y_full) / norm(y_full) < 1E-8
end

@testitem "Test Dense Matrix x SvdBlock" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=100)
    m = [1 2; 3 4]
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    block = NonEquilibriumGreenFunction.SvdBlock(kf, 1e-6)
    x = randn((12, size(block, 1)))
    y = x * block
    y_full = x * NonEquilibriumGreenFunction.full(block)
    @test norm(full(y) - y_full) / norm(y_full) < 1E-8
end

@testitem "Test SvdBlock x SvdBlock" begin
    using LinearAlgebra
    n, k1, m, k2, l = 100, 12, 80, 10, 100
    block1 = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k1), Diagonal(randn(Float64, k1)), randn(ComplexF64, k1, m))
    block2 = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, m, k2), Diagonal(randn(Float64, k2)), randn(ComplexF64, k2, l))
    full_block1 = NonEquilibriumGreenFunction.full(block1)
    full_block2 = NonEquilibriumGreenFunction.full(block2)
    full_product = full_block1 * full_block2
    block_product = block1 * block2
    @test norm(full_product - NonEquilibriumGreenFunction.full(block_product)) / norm(full_product) < 1E-8
end

@testitem "Test SvdBlock + SvdBlock" begin
    using LinearAlgebra
    n, k, m = 100, 12, 80, 10, 100
    block1 = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    block2 = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    full_block1 = NonEquilibriumGreenFunction.full(block1)
    full_block2 = NonEquilibriumGreenFunction.full(block2)
    full_sum = full_block1 + full_block2
    block_sum = block1 + block2
    @test norm(full_sum - NonEquilibriumGreenFunction.full(block_sum)) / norm(full_sum) < 1E-8
end

@testitem "scalar * SvdBlock" begin
    using LinearAlgebra
    n, k, m = 100, 12, 80, 10, 100
    block = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    minus_block = -1 * block
    @test norm(NonEquilibriumGreenFunction.full(block + minus_block)) < 1E-12
end

@testitem "Test Hodlr construction from kernel" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    m = ones(2, 2)
    kf = KernelFunction((x, y) -> m .* exp(-abs2(x - y)), dom)
    Hodlr = build_hodlr(kf, HodlrSettings(tol=1e-6, leafsize=size(kf, 1) ÷ 4 ))
    @test size(Hodlr) == size(kf)
end

@testitem "Test Hodlr construction from sparse matrix" begin
    using LinearAlgebra
    using SparseArrays
    sparse_matrix = spdiagm(12 => randn(128))
    Hodlr = build_hodlr(sparse_matrix, HodlrSettings(tol=1e-6, leafsize=size(sparse_matrix, 1) ÷ 4))
    @test size(Hodlr) == size(sparse_matrix)
    @test norm(full(Hodlr) - Array(sparse_matrix)) < 1e-10
end



@testitem "Test Hodlr full" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    m = [1 2; 1 1]
    const tol = 1E-9
    kf = KernelFunction((x, y) -> m .* exp(1im * (x - y)), dom)
    Hodlr = build_hodlr(kf, HodlrSettings(tol=0.01 * tol, leafsize=64))
    full_hodlr = full(Hodlr)
    dense = zeros(eltype(Hodlr), size(Hodlr)...)
    NonEquilibriumGreenFunction.fill_with_kernel!(dense, kf)
    @test norm(dense - full_hodlr) / norm(dense) < tol
    @test norm(dense - full_hodlr) < tol
end

@testitem "Hodlr product with vector" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    m = [1 2; 1 1]
    kf = KernelFunction((x, y) -> m .* exp(1im * (x - y)), dom)
    ctx = HodlrSettings()
    Hodlr = build_hodlr(kf, ctx)
    full_hodlr = full(Hodlr)
    x = randn(eltype(Hodlr), size(Hodlr, 2))
    y_hodlr = Hodlr * x
    y_full = full_hodlr * x
    @test norm(y_hodlr - y_full) / norm(y_full) < 10 * ctx.tol
    y_full = x' * full_hodlr
    @test norm(x' * Hodlr - y_full) / norm(y_full) < 10 * ctx.tol
end

@testitem "Hodlr product with array" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    m = [1 2; 1 1]
    kf = KernelFunction((x, y) -> m .* exp(1im * (x - y)), dom)
    ctx = HodlrSettings()
    Hodlr = build_hodlr(kf, ctx)
    full_hodlr = full(Hodlr)
    x = randn(eltype(Hodlr), size(Hodlr, 2), 12)
    y_full = full_hodlr * x
    y_hodlr = Hodlr * x
    @test size(y_hodlr) == size(y_full)
    @test norm(y_hodlr - y_full) / norm(y_full) < 10 * ctx.tol
    y_full = x' * full_hodlr
    @test norm(x' * Hodlr - y_full) / norm(y_full) < 10 * ctx.tol
end

@testitem "Hodlr product with SvdBlock" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    kf = KernelFunction((x, y) -> [1 2; 1 1] .* exp(1im * (x - y)), dom)
    ctx = HodlrSettings()
    Hodlr = build_hodlr(kf, ctx)
    k, m = 12, 80
    low_rank_block = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, size(Hodlr, 2), k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    holder_product = Hodlr * low_rank_block
    full_product = full(Hodlr) * NonEquilibriumGreenFunction.full(low_rank_block)
    @test size(holder_product) == size(full_product)
    @test norm(full_product - NonEquilibriumGreenFunction.full(holder_product)) / norm(full_product) < 10 * ctx.tol
end
@testitem "SvdBlock product with Hodlr" begin
    using LinearAlgebra
    dom = KernelDomain((0.0, 1.0), n_steps=512)
    kf = KernelFunction((x, y) -> [1 2; 1 1] .* exp(1im * (x - y)), dom)
    ctx = HodlrSettings()
    Hodlr = build_hodlr(kf, ctx)
    k, m = 12, 80
    low_rank_block = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, m, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, size(Hodlr, 1)))
    holder_product = low_rank_block * Hodlr
    full_product = NonEquilibriumGreenFunction.full(low_rank_block) * full(Hodlr)
    @test size(holder_product) == size(full_product)
    @test norm(full_product - NonEquilibriumGreenFunction.full(holder_product)) / norm(full_product) < 10 * ctx.tol
end
@testitem "construct Hodlr from SvdBlock" begin
    using LinearAlgebra
    n, k, m = 512, 12, 512
    ctx = HodlrSettings(tol=1E-8, leafsize=64)
    low_rank_block = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    Hodlr = build_hodlr(low_rank_block, ctx)
    @test size(Hodlr) == size(low_rank_block)
    @test norm(full(Hodlr) * full(low_rank_block) - full(Hodlr) * full(low_rank_block)) / norm(full(Hodlr) * full(low_rank_block)) < 10 * ctx.tol
end

@testitem "hodlr + LowRankBlock" begin
    using LinearAlgebra
    n, k, m = 512, 12, 512
    ctx = HodlrSettings(tol=1E-8, leafsize=64)
    low_rank_block_1 = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    hodlr_1 = build_hodlr(low_rank_block_1, ctx)
    low_rank_block_2 = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    full_sum = full(hodlr_1) + full(low_rank_block_2)
    hodlr_sum = full(hodlr_1 + low_rank_block_2)
    @test norm(full_sum - hodlr_sum) / norm(full_sum) < 10 * ctx.tol
end

@testitem "hodlr + hodlr" begin
    using LinearAlgebra
    n, k, m = 512, 12, 512
    ctx = HodlrSettings(tol=1E-8, leafsize=64)
    low_rank_block_1 = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    hodlr_1 = build_hodlr(low_rank_block_1, ctx)
    low_rank_block_2 = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    hodlr_2 = build_hodlr(low_rank_block_2, ctx)
    full_sum = full(hodlr_1) + full(hodlr_2)
    hodlr_sum = full(hodlr_1 + hodlr_2)
    @test norm(full_sum - hodlr_sum) / norm(full_sum) < 10 * ctx.tol
end

@testitem "hodlr x hodlr" begin
    using LinearAlgebra
    n, k, m = 512, 12, 512
    ctx = HodlrSettings(tol=1E-8, leafsize=64)
    low_rank_block_1 = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    hodlr_1 = build_hodlr(low_rank_block_1, ctx)
    low_rank_block_2 = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    hodlr_2 = build_hodlr(low_rank_block_2, ctx)
    full_product = full(hodlr_1) * full(hodlr_2)
    hodlr_product = full(hodlr_1 * hodlr_2)
    @test norm(full_product - hodlr_product) / norm(full_product) < 10 * ctx.tol
end

@testitem "build upper triangular hodlr" begin
    using LinearAlgebra
    n, k, m = 512, 12, 512
    ctx = HodlrSettings(tol=1E-8, leafsize=64)
    low_rank_block = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    hodlr_acausal = build_hodlr(low_rank_block, ctx)
    hodlr_masked = NonEquilibriumGreenFunction.drop_lower_block_offdiagonal(hodlr_acausal)
    hodlr_causal = NonEquilibriumGreenFunction.build_upper_triangular_hodlr(low_rank_block, ctx)
     @test norm(full(hodlr_masked) - full(hodlr_causal)) / norm(full(hodlr_masked)) < 10 * ctx.tol
end

@testitem "inv(hodlr)" begin
    using LinearAlgebra
    n, k, m = 512, 12, 512
    ctx = HodlrSettings(tol=1E-8, leafsize=64)
    low_rank_block = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    hodlr = NonEquilibriumGreenFunction.build_upper_triangular_hodlr(low_rank_block, ctx)
    full_inv = inv(full(hodlr))
    norm(full(inv(hodlr)) - full_inv) / norm(full_inv) < 10 * ctx.tol
end

@testitem "holdr Arithmetics" begin
    using LinearAlgebra
    n, k, m = 512, 12, 512
    ctx = HodlrSettings(tol=1E-8, leafsize=64)
    low_rank_block = NonEquilibriumGreenFunction.SvdBlock(randn(ComplexF64, n, k), Diagonal(randn(Float64, k)), randn(ComplexF64, k, m))
    hodlr_tree = NonEquilibriumGreenFunction.build_upper_triangular_hodlr(low_rank_block, ctx)
    hodlr = Hodlr(hodlr_tree)
    norm(full(hodlr - hodlr)) < 1E-12 
    norm(full(hodlr + hodlr - 2*hodlr) ) < 1E-12 
    norm(full(2*hodlr  - hodlr - hodlr) ) < 1E-12 
    norm(full(4*hodlr*hodlr  - (2*hodlr)*(2*hodlr)) ) < 1E-12 
end