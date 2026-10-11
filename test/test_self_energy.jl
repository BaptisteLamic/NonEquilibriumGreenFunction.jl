@testitem "second_born_self_energy: blockwise reference" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: lesser_greater, blockrange

    bs, N = 2, 8
    ax = LinRange(0.0, 1.0, N)
    T = ComplexF64
    AR = randn(T, bs * N, bs * N)
    AK = randn(T, bs * N, bs * N)
    G_R = RetardedKernel(ax, AR, bs, NONCompression())
    G_K = AcausalKernel(ax, AK, bs, NONCompression())

    V = randn(bs, bs)
    result = second_born_self_energy(G_R, G_K, V)

    @test isretarded(result.Σ_R)
    @test isacausal(result.Σ_K)

    G⁼, G⁽ = lesser_greater(G_R, G_K)
    @test G⁼ ≈ (Matrix(matrix(G_K)) - Matrix(matrix(G_R)) + Matrix(matrix(adjoint(G_R)))) / 2
    @test G⁽ ≈ (Matrix(matrix(G_K)) + Matrix(matrix(G_R)) - Matrix(matrix(adjoint(G_R)))) / 2

    xi = 2
    function ref_blocks(g, gt, sign)
        out = zeros(T, bs, bs)
        for i in 1:bs, j in 1:bs
            acc = zero(T)
            for k in 1:bs, l in 1:bs
                acc += V[i, k] * V[j, l] * (xi * g[i, j] * gt[l, k] * g[k, l] - g[i, l] * gt[l, k] * g[k, j])
            end
            out[i, j] = sign * im * acc
        end
        return out
    end

    ΣK = Matrix(matrix(result.Σ_K))
    ΣR = Matrix(matrix(result.Σ_R))
    for I in 1:N, J in 1:N
        rI = blockrange(I, bs)
        rJ = blockrange(J, bs)
        S⁼ = ref_blocks(G⁼[rI, rJ], G⁽[rJ, rI], +1)
        S⁽ = ref_blocks(G⁽[rI, rJ], G⁼[rJ, rI], -1)
        @test ΣK[rI, rJ] ≈ S⁼ + S⁽
        expected_R = I >= J ? S⁽ - S⁼ : zeros(T, bs, bs)
        @test ΣR[rI, rJ] ≈ expected_R
    end
end

@testitem "second_born_self_energy: single-level limit" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction
    bs, N = 1, 6
    ax = LinRange(0.0, 1.0, N)
    T = ComplexF64
    U = 0.7
    G_R = RetardedKernel(ax, randn(T, N, N), bs, NONCompression())
    G_K = AcausalKernel(ax, randn(T, N, N), bs, NONCompression())
    result = second_born_self_energy(G_R, G_K, U)
    G⁼, G⁽ = lesser_greater(G_R, G_K)
    # bubble and exchange coincide at bs = 1, giving the textbook i U^2 G⁼ĜG⁼
    S⁼ref = [im * U^2 * (2 - 1) * G⁼[i, j] * G⁽[j, i] * G⁼[i, j] for i in 1:N, j in 1:N]
    S⁽ref = [-im * U^2 * (2 - 1) * G⁽[i, j] * G⁼[j, i] * G⁽[i, j] for i in 1:N, j in 1:N]
    ΣK = Matrix(matrix(result.Σ_K))
    @test ΣK ≈ S⁼ref .+ S⁽ref
end

@testitem "second_born_self_energy: causality and size checks" begin
    using NonEquilibriumGreenFunction
    ax = LinRange(0.0, 1.0, 4)
    G_R = RetardedKernel(ax, randn(ComplexF64, 4, 4), 1, NONCompression())
    G_K = AcausalKernel(ax, randn(ComplexF64, 4, 4), 1, NONCompression())
    @test_throws ArgumentError second_born_self_energy(G_K, G_K, 1.0)
    @test_throws ArgumentError second_born_self_energy(G_R, G_R, 1.0)
    @test_throws ArgumentError second_born_self_energy(G_R, G_K, [1 0; 0 1])
end

@testitem "hartree_fock_self_energy: block-diagonal reference" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction
    bs, N = 2, 6
    ax = LinRange(0.0, 1.0, N)
    T = ComplexF64
    G_R = RetardedKernel(ax, randn(T, bs * N, bs * N), bs, NONCompression())
    G_K = AcausalKernel(ax, randn(T, bs * N, bs * N), bs, NONCompression())
    V = randn(bs, bs)
    Σ_HF = hartree_fock_self_energy(G_R, G_K, V)
    @test islocal(Σ_HF)
    G⁼, _ = lesser_greater(G_R, G_K)
    M = Matrix(matrix(Σ_HF))
    for I in 1:N
        rI = blockrange(I, bs)
        rho = -im .* G⁼[rI, rI]
        @test M[rI, rI] ≈ 2 .* Diagonal(V * diag(rho)) .- V .* transpose(rho)
        for J in 1:N
            J == I && continue
            @test norm(M[rI, blockrange(J, bs)]) == 0
        end
    end
end

@testitem "self energies feed solve_keldysh (interaction quench)" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction
    # single spin-degenerate level coupled to wide-band leads, then a
    # second-Born + Hartree-Fock interaction correction on top.
    δt, T_end = 0.05, 1.0
    ax = 0.0:δt:T_end
    N = length(ax)
    Γl, Γr, β, U = 0.5, 0.5, 100.0, 0.4
    cpr = NONCompression()

    Σ_R_leads = LocalKernel(ax, t -> -1im * (Γl + Γr), compression=cpr)
    g = RetardedKernel(ax, Stationary(τ -> ComplexF64(-2im)), compression=cpr)
    G_R = solve_dyson(g, g * Σ_R_leads)

    ρ = AcausalKernel(ax, Stationary(τ -> thermal_kernel(τ, β) .* ComplexF64(1)), compression=cpr)
    c_l = LocalKernel(ax, t -> sqrt(Γl), compression=cpr)
    c_r = LocalKernel(ax, t -> sqrt(Γr), compression=cpr)
    Σ_K_leads = -2im * (c_l' * ρ * c_l + c_r' * ρ * c_r)
    G_K = G_R * Σ_K_leads * G_R'

    # interacting correction; the instantaneous HF term is folded into Σ_R at
    # the matrix level (a SumOperator with a LocalKernel would carry Acausal
    # causality, which solve_keldysh's checks reject)
    born = second_born_self_energy(G_R, G_K, U)
    hf = hartree_fock_self_energy(G_R, G_K, U)
    Σ_R_int = make_similar(born.Σ_R, Matrix(matrix(born.Σ_R)) + Matrix(matrix(hf)))
    Σ_K_int = born.Σ_K

    # the total self-energies must satisfy the solve_keldysh causality contract
    @test isretarded(born.Σ_R)
    @test isacausal(born.Σ_K)
    @test islocal(hf)

    # assemble totals at matrix level: mixing LocalKernel (Acausal) and
    # Retarded kernels in a SumOperator yields Acausal causality, which
    # solve_keldysh's checks reject; the physics is fine, so fold explicitly
    Σ_R_tot = make_similar(Σ_R_int,
        Matrix(matrix(Σ_R_leads)) + Matrix(matrix(Σ_R_int)))
    Σ_K_tot = make_similar(Σ_K_int,
        Matrix(matrix(Σ_K_leads)) + Matrix(matrix(Σ_K_int)))
    dressed = solve_keldysh(g, Σ_R_tot, Σ_K_tot)
    @test isretarded(dressed.G_R)
    @test isacausal(dressed.G_K)
    # finite and of sane magnitude
    @test all(isfinite, matrix(dressed.G_R))
    @test all(isfinite, matrix(dressed.G_K))
    # the interacting G must differ from the noninteracting one
    @test norm(Matrix(matrix(dressed.G_R)) - Matrix(matrix(G_R))) > 0
end
