using TensorCrossInterpolation

@testitem "Test TensorCrossInterpolation" begin
    import TensorCrossInterpolation as TCI
    using LinearAlgebra
    n_steps = 512
    dom = KernelDomain((0.0, 1.0), n_steps=n_steps)
    m = [1 1; 1 1]
    const tol = 1E-9
    kf = NonEquilibriumGreenFunction.KernelFunction((x, y) -> m .* exp(1im * (x - y)), dom)
    bs = kf.blocksize
    kf(1, 1, 1, 1)
    TCI.crossinterpolate2(ComplexF64, (v) -> kf(v[1], v[2], v[3], v[4]), (n_steps, n_steps, bs, bs); tolerance=tol)
end
