
@testitem "Test ACA on block trivial structured kernel" begin
    using ACAFact
    using LinearAlgebra
    dom = range(0.0, 1.0, length=512)
    m =  [1 1; 1 1]
    const tol = 1E-9
    kf = NonEquilibriumGreenFunction.KernelFunction((x, y) -> m .* exp(1im*(x - y)), dom)
    U, V = aca(kf, 2)
    full_block = zeros(eltype(U),size(kf)...)
    NonEquilibriumGreenFunction.fill_with_kernel!(full_block,kf)
    aca_block = U*V'
    @test norm(full_block - aca_block) < tol
end

@testitem "Test ACA on block structured kernel" begin
    using ACAFact
    using LinearAlgebra
    dom = range(0.0, 1.0, length=512)
    m =  [1 2; 1 1]
    const tol = 1E-9
    kf = NonEquilibriumGreenFunction.KernelFunction((x, y) -> m .* exp(1im*(x - y)), dom)
    U, V = aca(kf,12)
    full_block = zeros(eltype(U),size(kf)...)
    NonEquilibriumGreenFunction.fill_with_kernel!(full_block,kf)
    aca_block = U*V'
    @test_broken norm(full_block - aca_block) < tol
end