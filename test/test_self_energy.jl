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

    # dense references, assembled from the maps on grid points (tests only)
    G⁼ref = (Matrix(matrix(G_K)) - Matrix(matrix(G_R)) + Matrix(matrix(adjoint(G_R)))) / 2
    G⁽ref = (Matrix(matrix(G_K)) + Matrix(matrix(G_R)) - Matrix(matrix(adjoint(G_R)))) / 2
    G⁼map, G⁽map = lesser_greater(G_R, G_K)
    for I in 1:N, J in 1:N
        @test G⁼map(ax[I], ax[J]) ≈ G⁼ref[blockrange(I, bs), blockrange(J, bs)]
        @test G⁽map(ax[I], ax[J]) ≈ G⁽ref[blockrange(I, bs), blockrange(J, bs)]
    end

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
        S⁼ = ref_blocks(G⁼ref[rI, rJ], G⁽ref[rJ, rI], +1)
        S⁽ = ref_blocks(G⁽ref[rI, rJ], G⁼ref[rJ, rI], -1)
        @test ΣK[rI, rJ] ≈ S⁼ + S⁽
        expected_R = I >= J ? S⁽ - S⁼ : zeros(T, bs, bs)
        @test ΣR[rI, rJ] ≈ expected_R
    end
end

@testitem "second_born_self_energy: single-level limit" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: lesser_greater, blockrange
    bs, N = 1, 6
    ax = LinRange(0.0, 1.0, N)
    T = ComplexF64
    U = 0.7
    G_R = RetardedKernel(ax, randn(T, N, N), bs, NONCompression())
    G_K = AcausalKernel(ax, randn(T, N, N), bs, NONCompression())
    result = second_born_self_energy(G_R, G_K, U)
    G⁼map, G⁽map = lesser_greater(G_R, G_K)
    # bubble and exchange coincide at bs = 1, giving the textbook i U^2 G⁼ĜG⁼
    ΣK = Matrix(matrix(result.Σ_K))
    for I in 1:N, J in 1:N
        g = G⁼map(ax[I], ax[J])[1, 1]
        gt = G⁽map(ax[J], ax[I])[1, 1]
        @test ΣK[I, J] ≈ im * U^2 * g * gt * g - im * U^2 * (G⁽map(ax[I], ax[J])[1, 1] * G⁼map(ax[J], ax[I])[1, 1] * G⁽map(ax[I], ax[J])[1, 1])
    end
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
    using NonEquilibriumGreenFunction: lesser_greater, blockrange
    bs, N = 2, 6
    ax = LinRange(0.0, 1.0, N)
    T = ComplexF64
    G_R = RetardedKernel(ax, randn(T, bs * N, bs * N), bs, NONCompression())
    G_K = AcausalKernel(ax, randn(T, bs * N, bs * N), bs, NONCompression())
    V = randn(bs, bs)
    Σ_HF = hartree_fock_self_energy(G_R, G_K, V)
    @test islocal(Σ_HF)
    G⁼ref = (Matrix(matrix(G_K)) - Matrix(matrix(G_R)) + Matrix(matrix(adjoint(G_R)))) / 2
    M = Matrix(matrix(Σ_HF))
    for I in 1:N
        rI = blockrange(I, bs)
        rho = -im .* G⁼ref[rI, rI]
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

    # interacting correction
    born = second_born_self_energy(G_R, G_K, U)
    hf = hartree_fock_self_energy(G_R, G_K, U)

    # the total self-energies must satisfy the solve_keldysh causality contract
    @test isretarded(born.Σ_R)
    @test isacausal(born.Σ_K)
    @test islocal(hf)

    # assemble totals at matrix level: mixing LocalKernel (Acausal) and
    # Retarded kernels in a SumOperator yields Acausal causality, which
    # solve_keldysh's checks reject; the physics is fine, so fold explicitly
    Σ_R_tot = make_similar(born.Σ_R, Matrix(matrix(born.Σ_R)) + Matrix(matrix(Σ_R_leads)) + Matrix(matrix(hf)))
    Σ_K_tot = make_similar(born.Σ_K, Matrix(matrix(born.Σ_K)) + Matrix(matrix(Σ_K_leads)))
    dressed = solve_keldysh(g, Σ_R_tot, Σ_K_tot)
    @test isretarded(dressed.G_R)
    @test isacausal(dressed.G_K)
    @test all(isfinite, matrix(dressed.G_R))
    @test all(isfinite, matrix(dressed.G_K))
    @test norm(Matrix(matrix(dressed.G_R)) - Matrix(matrix(G_R))) > 0
end

@testitem "second_born_self_energy: HSS path never materializes dense G" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction
    using HssMatrices: HssMatrix
    using NonEquilibriumGreenFunction: lesser_greater, blockrange
    # same physics as the blockwise reference, but through HSS-compressed
    # kernels: the maps must read blocks from the HSS storage on demand
    bs, N = 2, 32
    ax = LinRange(0.0, 1.0, N)
    T = ComplexF64
    cpr = HssCompression(atol=1e-12, rtol=1e-12, leafsize=8)
    G_R = RetardedKernel(ax, Stationary(τ -> [exp(-abs(τ)) 0.2im*exp(-2abs(τ)); 0 0.5exp(-abs(τ))]), compression=cpr)
    # consistent Keldysh component built from the same axis
    G_K = AcausalKernel(ax, Stationary(τ -> [thermal_kernel(τ, 10.0) 0; 0 thermal_kernel(τ, 10.0)] .* 1im), compression=cpr)
    V = [0.4 0.1; 0.1 0.3]
    result = second_born_self_energy(G_R, G_K, V)
    @test matrix(result.Σ_K) isa HssMatrix
    @test matrix(result.Σ_R) isa HssMatrix
    @test isretarded(result.Σ_R)
    @test isacausal(result.Σ_K)

    # cross-check a few blocks against the map evaluated on grid points
    G⁼map, G⁽map = lesser_greater(G_R, G_K)
    born_ref(g, gt, V; xi=2) = begin
        bs = size(g, 1)
        T = eltype(g)
        S = similar(g)
        for i in 1:bs, j in 1:bs
            acc = zero(T)
            for k in 1:bs, l in 1:bs
                acc += V[i, k] * V[j, l] * (xi * g[i, j] * gt[l, k] * g[k, l] - g[i, l] * gt[l, k] * g[k, j])
            end
            S[i, j] = acc
        end
        return S
    end
    ΣKmap(t, tp) = im .* born_ref(G⁼map(t, tp), G⁽map(tp, t), V) .- im .* born_ref(G⁽map(t, tp), G⁼map(tp, t), V)
    for (I, J) in zip([1, 5, 12, N], [1, 3, 20, N-2])
        blk = matrix(result.Σ_K)[collect(blockrange(I, bs)), collect(blockrange(J, bs))]
        @test isapprox(blk, ΣKmap(ax[I], ax[J]); atol=1e-10)
    end
end

@testitem "second_born_self_energy: stationary fast path matches general path" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: blockrange
    # stationary G: the circulant (separation-indexed) path must agree with
    # the general two-time path block by block
    bs, N = 2, 24
    ax = LinRange(0.0, 1.0, N)
    cpr = NONCompression()
    f_R = τ -> [exp(-abs(τ)) 0.2im*exp(-2abs(τ)); 0 0.5exp(-abs(τ))]
    f_K = τ -> [thermal_kernel(τ, 10.0) 0; 0 thermal_kernel(τ, 10.0)] .* 1im
    G_R = RetardedKernel(ax, Stationary(f_R), compression=cpr)
    G_K = AcausalKernel(ax, Stationary(f_K), compression=cpr)
    V = [0.4 0.1; 0.1 0.3]

    gen = second_born_self_energy(G_R, G_K, V)
    sta = second_born_self_energy(G_R, G_K, V; stationary=true)

    MK = Matrix(matrix(gen.Σ_K))
    MKs = Matrix(matrix(sta.Σ_K))
    MR = Matrix(matrix(gen.Σ_R))
    MRs = Matrix(matrix(sta.Σ_R))
    @test MK ≈ MKs atol=1e-12
    @test MR ≈ MRs atol=1e-12
    @test isretarded(sta.Σ_R)
    @test isacausal(sta.Σ_K)
end
