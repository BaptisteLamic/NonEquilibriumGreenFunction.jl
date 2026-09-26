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
    g = discretize_retardedkernel(ax, (t, tp) -> ComplexF64(-2im); compression=HssCompression(), stationary=true)
    Σ_R = discretize_dirac(ax, t -> ComplexF64(-1im * 0.5); compression=HssCompression())
    ρ = discretize_acausalkernel(ax, (t, tp) -> thermal_kernel(t - tp, 100) .|> ComplexF64; stationary=true, compression=HssCompression())
    W = discretize_dirac(ax, t -> ComplexF64(sqrt(0.5)); compression=HssCompression())
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

@testitem "Causality predicates on all operators" begin
    using NonEquilibriumGreenFunction
    ax = 0:0.1:1
    δ = discretize_dirac(ax, t -> ComplexF64(-1im); compression=HssCompression())
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
