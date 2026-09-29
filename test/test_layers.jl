@testitem "Layered module structure" begin
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: Kernels, Physics

    @test NonEquilibriumGreenFunction.Kernel === Kernels.Kernel
    @test NonEquilibriumGreenFunction.solve_dyson === Kernels.solve_dyson
    @test NonEquilibriumGreenFunction.causality === Kernels.causality
    @test NonEquilibriumGreenFunction.thermal_kernel === Physics.thermal_kernel
    @test NonEquilibriumGreenFunction.solve_keldysh === Physics.solve_keldysh
    @test NonEquilibriumGreenFunction.lead_current === Physics.lead_current
    @test Base.isdefined(NonEquilibriumGreenFunction, :AdaptiveRichardson)
end

@testitem "solve_keldysh and lead_current" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: Physics

    function notebook_current(G_R, G_K, Σ_R, Σ_K)
        Σ_R_time_G_R = Σ_R * G_R
        G_R_time_Σ_R = G_R * Σ_R
        G_R_time_Σ_K = G_R * Σ_K
        G_K_time_adjoint_Σ_R = G_K * adjoint(Σ_R)
        Σ_K_time_adjoint_G_R = Σ_K * adjoint(G_R)
        Σ_R_time_G_K = Σ_R * G_K
        return -((1 // 2) * (adjoint(Σ_R_time_G_R) - Σ_R_time_G_R) + G_R_time_Σ_K +
                 G_K_time_adjoint_Σ_R - (Σ_K_time_adjoint_G_R + Σ_R_time_G_K) +
                 (3 // 2) * (G_R_time_Σ_R - adjoint(G_R_time_Σ_R)))
    end

    ax = 0:0.1:2
    g = RetardedKernel(ax, Stationary(tau -> ComplexF64(-2im)); compression=HssCompression())
    Σ_R = InstantaneousKernel(ax, t -> ComplexF64(-1im * 0.5); compression=HssCompression())
    ρ = AcausalKernel(ax, Stationary(tau -> thermal_kernel(tau, 100) .|> ComplexF64); compression=HssCompression())
    W = InstantaneousKernel(ax, t -> ComplexF64(sqrt(0.5)); compression=HssCompression())
    Σ_K = -2im * W' * ρ * W

    (; G_R, G_K) = Physics.solve_keldysh(g, Σ_R, Σ_K)
    @test isretarded(G_R)
    @test isacausal(G_K)

    G_R_ref = solve_dyson(g, g * Σ_R)
    G_K_ref = G_R_ref * Σ_K * G_R_ref'
    @test matrix(G_R) ≈ matrix(G_R_ref)
    @test matrix(G_K) ≈ matrix(G_K_ref)

    I_new = Physics.lead_current(G_R, G_K, Σ_R, Σ_K)
    I_ref = notebook_current(G_R, G_K, Σ_R, Σ_K)
    @test matrix(I_new) ≈ matrix(I_ref)

    @test_throws AssertionError Physics.solve_keldysh(g, adjoint(g), Σ_K)
end

@testitem "lead_current physical sanity checks" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: Physics

    ax = 0:0.1:1
    g = RetardedKernel(ax, Stationary(tau -> ComplexF64(-2im)); compression=NONCompression())
    Σ_R = InstantaneousKernel(ax, t -> ComplexF64(-1im * 0.5); compression=NONCompression())
    ρ = AcausalKernel(ax, Stationary(tau -> ComplexF64(1.0)); compression=NONCompression())
    W = InstantaneousKernel(ax, t -> ComplexF64(sqrt(0.5)); compression=NONCompression())
    Σ_K = -2im * W' * ρ * W

    (; G_R, G_K) = Physics.solve_keldysh(g, Σ_R, Σ_K)

    zero_dirac = InstantaneousKernel(ax, t -> ComplexF64(0); compression=NONCompression())
    I_zero = Physics.lead_current(G_R, G_K, zero_dirac, zero_dirac)
    @test matrix(I_zero) ≈ zeros(ComplexF64, size(matrix(G_R))) atol = 1e-12

    ρ2 = AcausalKernel(ax, Stationary(tau -> thermal_kernel(tau, 50) .|> ComplexF64); compression=NONCompression())
    Σ_K2 = -2im * W' * ρ2 * W
    I1 = Physics.lead_current(G_R, G_K, Σ_R, Σ_K)
    I2 = Physics.lead_current(G_R, G_K, Σ_R, Σ_K2)
    I12 = Physics.lead_current(G_R, G_K, Σ_R, Σ_K + Σ_K2)
    I0 = Physics.lead_current(G_R, G_K, Σ_R, zero_dirac)
    @test matrix(I12) ≈ matrix(I1) + matrix(I2) - matrix(I0) atol = 1e-12

    I3 = Physics.lead_current(G_R, G_K, Σ_R, 3 * Σ_K)
    @test matrix(I3) ≈ 3 * matrix(I1) - 2 * matrix(I0) atol = 1e-12

    @test_throws AssertionError Physics.lead_current(adjoint(g), G_K, Σ_R, Σ_K)
    @test_throws AssertionError Physics.lead_current(G_R, G_K, Σ_K, Σ_R)

    sig = Physics.current_signal(Physics.lead_current(G_R, G_K, Σ_R, Σ_K))
    @test length(sig) == length(ax)
    @test sig ≈ diag(matrix(Physics.lead_current(G_R, G_K, Σ_R, Σ_K))) atol = 1e-12

    I_sum = Physics.lead_current(G_R, G_K, Σ_R, Σ_K + Σ_K2)
    @test Physics.current_signal(I_sum) ≈
          Physics.current_signal(Physics.lead_current(G_R, G_K, Σ_R, Σ_K)) +
          Physics.current_signal(Physics.lead_current(G_R, G_K, Σ_R, Σ_K2)) -
          Physics.current_signal(Physics.lead_current(G_R, G_K, Σ_R, zero_dirac)) atol = 1e-12
end

@testitem "current_signal with Keldysh blocksize > 1" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: Physics

    ax = 0:0.25:1
    g = RetardedKernel(ax, Stationary(tau -> ComplexF64[1.0 2.0; 3.0 4.0] * (tau + 1.0)); compression=NONCompression())
    Σ_R = InstantaneousKernel(ax, t -> ComplexF64.([-0.5im 0; 0 -0.25im]); compression=NONCompression())
    ρ = AcausalKernel(ax, Stationary(tau -> ComplexF64[1 0; 0 1] * exp(-tau^2)); compression=NONCompression())
    W = InstantaneousKernel(ax, t -> ComplexF64.([sqrt(0.5) 0; 0 sqrt(0.3)]); compression=NONCompression())
    Σ_K = -2im * W' * ρ * W

    (; G_R, G_K) = Physics.solve_keldysh(g, Σ_R, Σ_K)
    Iop = Physics.lead_current(G_R, G_K, Σ_R, Σ_K)
    sig = Physics.current_signal(Iop)
    mat = matrix(Iop)
    manual = [tr(mat[2*i-1:2*i, 2*i-1:2*i]) for i in 1:div(size(mat, 1), 2)]
    @test length(sig) == length(ax)
    @test sig ≈ manual atol = 1e-12
    @test sig ≠ diag(mat)[1:2:end]

    z = InstantaneousKernel(ax, t -> zeros(ComplexF64, 2, 2); compression=NONCompression())
    @test Physics.current_signal(Physics.lead_current(G_R, G_K, z, z)) ≈ zeros(ComplexF64, length(ax)) atol = 1e-12
end

@testitem "Causality predicates on all operators" begin
    using NonEquilibriumGreenFunction
    ax = 0:0.1:1
    δ = InstantaneousKernel(ax, t -> ComplexF64(-1im); compression=HssCompression())
    @test !isretarded(δ)
    @test !isadvanced(δ)
    @test !isacausal(δ)
    @test causality(δ) == Instantaneous()
end

@testitem "theq_lesser_time_kernel callable from Physics layer" begin
    using NonEquilibriumGreenFunction
    f_δ, f_reg = theq_lesser_time_kernel(100, 2, 0.1)
    @test f_δ(0.) == [-1.0 0.0; 0.0 -1.0]
    @test size(f_reg(0.2, 0.1)) == (2, 2)
    @test all(isfinite, f_reg(0.2, 0.1))
    @test f_reg(0.1, 0.1) == f_reg(0.2, 0.2)
end
