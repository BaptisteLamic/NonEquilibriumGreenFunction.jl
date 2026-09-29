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
    Σ_R = LocalKernel(ax, t -> ComplexF64(-1im * 0.5); compression=HssCompression())
    ρ = AcausalKernel(ax, Stationary(tau -> thermal_kernel(tau, 100) .|> ComplexF64); compression=HssCompression())
    W = LocalKernel(ax, t -> ComplexF64(sqrt(0.5)); compression=HssCompression())
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
    Σ_R = LocalKernel(ax, t -> ComplexF64(-1im * 0.5); compression=NONCompression())
    ρ = AcausalKernel(ax, Stationary(tau -> ComplexF64(1.0)); compression=NONCompression())
    W = LocalKernel(ax, t -> ComplexF64(sqrt(0.5)); compression=NONCompression())
    Σ_K = -2im * W' * ρ * W

    (; G_R, G_K) = Physics.solve_keldysh(g, Σ_R, Σ_K)

    zero_dirac = LocalKernel(ax, t -> ComplexF64(0); compression=NONCompression())
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
    Σ_R = LocalKernel(ax, t -> ComplexF64.([-0.5im 0; 0 -0.25im]); compression=NONCompression())
    ρ = AcausalKernel(ax, Stationary(tau -> ComplexF64[1 0; 0 1] * exp(-tau^2)); compression=NONCompression())
    W = LocalKernel(ax, t -> ComplexF64.([sqrt(0.5) 0; 0 sqrt(0.3)]); compression=NONCompression())
    Σ_K = -2im * W' * ρ * W

    (; G_R, G_K) = Physics.solve_keldysh(g, Σ_R, Σ_K)
    Iop = Physics.lead_current(G_R, G_K, Σ_R, Σ_K)
    sig = Physics.current_signal(Iop)
    mat = matrix(Iop)
    manual = [tr(mat[2*i-1:2*i, 2*i-1:2*i]) for i in 1:div(size(mat, 1), 2)]
    @test length(sig) == length(ax)
    @test sig ≈ manual atol = 1e-12
    @test sig ≠ diag(mat)[1:2:end]

    z = LocalKernel(ax, t -> zeros(ComplexF64, 2, 2); compression=NONCompression())
    @test Physics.current_signal(Physics.lead_current(G_R, G_K, z, z)) ≈ zeros(ComplexF64, length(ax)) atol = 1e-12
end

@testitem "Causality predicates on all operators" begin
    using NonEquilibriumGreenFunction
    ax = 0:0.1:1
    δ = LocalKernel(ax, t -> ComplexF64(-1im); compression=HssCompression())
    # A Local contact term satisfies every support constraint: it is
    # simultaneously retarded, advanced and acausal; the canonical
    # representative is Acausal, and its locality is Local.
    @test isretarded(δ)
    @test isadvanced(δ)
    @test isacausal(δ)
    @test causality(δ) == Acausal()
    @test locality(δ) == Local()
    @test islocal(δ)
end

@testitem "Locality axis: Local is the algebra unit" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction

    ax = 0:0.1:1
    g = RetardedKernel(ax, TwoTime((t, tp) -> ComplexF64(t - tp + 1.0)); compression=NONCompression())
    ρ = AcausalKernel(ax, Stationary(tau -> ComplexF64(exp(-tau^2))); compression=NONCompression())
    δ1 = LocalKernel(ax, t -> ComplexF64(2); compression=NONCompression())
    δ2 = LocalKernel(ax, t -> ComplexF64(3); compression=NONCompression())

    # locality composition
    @test locality(g) == Smooth()
    @test locality(ρ) == Smooth()
    @test locality(δ1) == Local()
    @test locality(δ1 * δ2) == Local()
    @test locality(δ1 * g) == Smooth()
    @test locality(g * δ1) == Smooth()
    @test locality(δ1 + δ2) == Local()
    @test locality(g + δ1) == Smooth()
    @test locality(g + ρ) == Smooth()

    # causality is preserved when composing with Local (unit of the algebra)
    @test causality(δ1 * g) == causality(g)
    @test causality(g * δ1) == causality(g)
    @test isretarded(δ1 * g)
    @test isacausal(g * δ1 * ρ)

    # Local operators are applied exactly: δ1 * δ2 is the pointwise product
    @test matrix(δ1 * δ2) ≈ matrix(δ1) * matrix(δ2)
    @test [m[1, 1] for m in same_time(δ1 * δ2)] ≈ 6 .* ones(length(ax))
    # and δ acting on a smooth kernel never adds a quadrature weight
    @test matrix(g * δ1) ≈ matrix(g) * matrix(δ1)
    @test matrix(δ1 * g) ≈ matrix(δ1) * matrix(g)
end

@testitem "theq_lesser_time_kernel callable from Physics layer" begin
    using NonEquilibriumGreenFunction
    f_δ, f_reg = theq_lesser_time_kernel(100, 2, 0.1)
    @test f_δ(0.) == [-1.0 0.0; 0.0 -1.0]
    @test size(f_reg(0.2, 0.1)) == (2, 2)
    @test all(isfinite, f_reg(0.2, 0.1))
    @test f_reg(0.1, 0.1) == f_reg(0.2, 0.2)
end

@testitem "Singular kernel map: product-integration restores second order" begin
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction.Kernels: Singular

    β = 10.0
    s(τ) = abs(τ) > 1e-64 ? -1im / β * csch(π * τ / β) : 0.0im
    T = 2.0
    g(τ) = ComplexF64(exp(-((τ - 0.3)^2) / 0.5) * (1.0 + 0.5im))

    # reference: fine-grid singular rule (converges O(dt^2))
    Nref = 4001
    axref = range(-T, T; length=Nref)
    Lref = AcausalKernel(axref, Stationary(g); compression=NONCompression())
    Kref = AcausalKernel(axref, Singular(s); compression=NONCompression())
    mid = (Nref + 1) ÷ 2
    ref = matrix(Kref * Lref)[mid, mid]

    function midval(N, maptype)
        ax = range(-T, T; length=N)
        K = AcausalKernel(ax, maptype; compression=NONCompression())
        L = AcausalKernel(ax, Stationary(g); compression=NONCompression())
        return matrix(K * L)[(N + 1) ÷ 2, (N + 1) ÷ 2]
    end

    # diagonal weight vanishes (principal-value prescription for the odd core)
    ax = range(-T, T; length=51)
    W = singular_weights(s, step(ax); N=51)
    @test W[0 + 51] == 0.0
    K = AcausalKernel(ax, Singular(s); compression=NONCompression())
    @test matrix(K)[(51 + 1) ÷ 2, (51 + 1) ÷ 2] == 0.0

    # second-order convergence: error drops ~4x per refinement
    e1 = abs(midval(101, Singular(s)) - ref)
    e2 = abs(midval(201, Singular(s)) - ref)
    @test 3.0 < e1 / e2 < 5.5

    # naive sampling of the singular core is strictly worse at matched N
    @test abs(midval(101, Stationary(s)) - ref) > 2 * e1

    # HSS compression path produces a finite product with a smooth partner
    axh = range(-T, T; length=201)
    Kh = AcausalKernel(axh, Singular(s); compression=HssCompression(leafsize=32))
    Lh = AcausalKernel(axh, Stationary(g); compression=HssCompression(leafsize=32))
    Ph = Kh * Lh
    @test all(isfinite, real(matrix(Ph)[100, 100]))

    # retarded masking: no support at negative lags
    Kr = RetardedKernel(ax, Singular(s); compression=NONCompression())
    Mr = matrix(Kr)
    @test Mr[10, 20] == 0.0
end

@testitem "Singular kernel map: finite temperature, matrix-valued core" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction.Kernels: Singular

    # matrix-valued thermal core: distinct β-scaled levels, bs = 2
    β = 10.0
    A = [1.0 2.0; 0.5 1.5]
    f(τ) = abs(τ) > 1e-64 ?
        (-1im / β) .* A .* csch(π * τ / β) :
        zeros(ComplexF64, 2, 2)
    T = 2.0
    g(τ) = ComplexF64(exp(-((τ - 0.3)^2) / 0.5) * (1.0 + 0.5im))

    g2(τ) = ComplexF64(1.0) .* A .+ g(τ) .* [1.0 0.3; 0.2 0.9]

    Nref = 2001
    axref = range(-T, T; length=Nref)
    Lref = AcausalKernel(axref, Stationary(g2); compression=NONCompression())
    Kref = AcausalKernel(axref, Singular(f); compression=NONCompression())
    mid = (Nref + 1) ÷ 2
    ref = matrix(Kref * Lref)[mid, mid]

    # weights are per-entry, diagonal block exactly zero
    W = singular_weights(f, 0.02; N=201)
    @test size(W) == (2, 2, 401)
    @test all(iszero, W[:, :, 0 + 201])

    function midval(N, mt)
        ax = range(-T, T; length=N)
        K = AcausalKernel(ax, mt; compression=NONCompression())
        L = AcausalKernel(ax, Stationary(g2); compression=NONCompression())
        return matrix(K * L)[(N + 1) ÷ 2, (N + 1) ÷ 2]
    end
    # product-integration beats naive sampling of the singular core at
    # matched N (note: bs > 1 acausal products are first-order in the
    # current quadrature, so we compare errors at fixed N)
    @test abs(midval(201, Singular(f)) - ref) < abs(midval(201, Stationary(f)) - ref)
    @test abs(midval(201, Singular(f)) - ref) < abs(midval(101, Singular(f)) - ref)
end

@testitem "Singular kernel map: finite-temperature convergence across β" begin
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction.Kernels: Singular

    T = 2.0
    g(τ) = ComplexF64(exp(-((τ - 0.3)^2) / 0.5) * (1.0 + 0.5im))
    for β in (2.0, 50.0)
        s(τ) = abs(τ) > 1e-64 ? -1im / β * csch(π * τ / β) : 0.0im
        Nref = 2001
        axref = range(-T, T; length=Nref)
        Lref = AcausalKernel(axref, Stationary(g); compression=NONCompression())
        Kref = AcausalKernel(axref, Singular(s); compression=NONCompression())
        mid = (Nref + 1) ÷ 2
        ref = matrix(Kref * Lref)[mid, mid]
        midval(N) = begin
            ax = range(-T, T; length=N)
            K = AcausalKernel(ax, Singular(s); compression=NONCompression())
            L = AcausalKernel(ax, Stationary(g); compression=NONCompression())
            matrix(K * L)[(N + 1) ÷ 2, (N + 1) ÷ 2]
        end
        e1 = abs(midval(101) - ref)
        e2 = abs(midval(201) - ref)
        @test 3.0 < e1 / e2 < 5.5
    end
end

@testitem "thermal_kernel zero-temperature limit" begin
    using NonEquilibriumGreenFunction
    @test thermal_kernel(0.5, Inf) ≈ -1im / (π * 0.5)
    # large-β consistency: csch form approaches the T=0 limit
    @test isapprox(thermal_kernel(0.05, 1e10), -1im / (π * 0.05); rtol=1e-6)
end

@testitem "Quadrature axis: rules are structural and switchable" begin
    using NonEquilibriumGreenFunction

    # default is the second-order trapezoid rule
    ax = range(-2.0, 2.0; length=51)
    K = AcausalKernel(ax, Stationary(τ -> ComplexF64(exp(-τ^2))); compression=NONCompression())
    @test quadrature(discretization(K)) == TrapezoidQuadrature()
    Kr = AcausalKernel(ax, Stationary(τ -> ComplexF64(exp(-τ^2))); compression=NONCompression(),
        quadrature=RectangleQuadrature())
    @test quadrature(discretization(Kr)) == RectangleQuadrature()
    # rectangle rule reproduces the historical product bit-for-bit
    Lr = AcausalKernel(ax, Stationary(τ -> ComplexF64(exp(-(τ - 0.3)^2))); compression=NONCompression(),
        quadrature=RectangleQuadrature())
    dt = step(ax)
    @test matrix(Kr * Lr) ≈ dt * matrix(Kr) * matrix(Lr)
    # trapezoid rule: half weights on the two domain-edge row-blocks
    L = AcausalKernel(ax, Stationary(τ -> ComplexF64(exp(-(τ - 0.3)^2))); compression=NONCompression())
    Mr = copy(matrix(L))
    Mr[1, :] .*= 0.5
    Mr[size(Mr, 1), :] .*= 0.5
    @test matrix(K * L) ≈ dt * matrix(K) * Mr
end

@testitem "Quadrature axis: trapezoid restores second order, bs-independent" begin
    using NonEquilibriumGreenFunction

    # smooth matrix-valued kernels, bs = 2, non-decaying at the boundary
    T = 2.0
    k1(τ) = ComplexF64.([exp(-τ^2) 0.3exp(-(τ-0.2)^2); 0.2exp(-(τ+0.1)^2) 0.5exp(-τ^2)])
    g(τ) = ComplexF64(exp(-((τ - 0.3)^2) / 0.5) * (1.0 + 0.5im))
    g2(τ) = [1.0 0.3; 0.2 0.9] .* g(τ) .+ ComplexF64.([0.5 0.1; 0.1 0.25])

    function midval(N, q)
        ax = range(-T, T; length=N)
        K = AcausalKernel(ax, Stationary(k1); compression=NONCompression(), quadrature=q)
        L = AcausalKernel(ax, Stationary(g2); compression=NONCompression(), quadrature=q)
        M = matrix(K * L)
        n0 = (N + 1) ÷ 2
        return M[2*(n0-1)+1, 2*(n0-1)+1]
    end

    # reference: fine-grid trapezoid (converges O(dt^2)); ref error is
    # ~1e-9, far below the compared errors
    ref = midval(1601, TrapezoidQuadrature())

    # trapezoid: second order (ratio ~ 4), independent of blocksize
    e1 = abs(midval(101, TrapezoidQuadrature()) - ref)
    e2 = abs(midval(201, TrapezoidQuadrature()) - ref)
    @test 3.0 < e1 / e2 < 5.5

    # rectangle: first order (ratio ~ 2) at the same bs
    r1 = abs(midval(101, RectangleQuadrature()) - ref)
    r2 = abs(midval(201, RectangleQuadrature()) - ref)
    @test 1.5 < r1 / r2 < 2.7

    # trapezoid strictly dominates rectangle at matched N
    @test e1 < r1
end

@testitem "Quadrature axis: causal products under the trapezoid rule" begin
    using NonEquilibriumGreenFunction

    T = 2.0
    k1(τ) = ComplexF64(exp(-τ^2))
    k2(τ) = ComplexF64(exp(-((τ - 0.2)^2) / 0.3) * (1.0 - 0.3im))
    g(τ) = ComplexF64(exp(-((τ - 0.3)^2) / 0.5) * (1.0 + 0.5im))
    t, tp = 0.6, 0.2

    function entry(N, q, cons)
        ax = range(0.0, T; length=N)
        dt = step(ax)
        i = round(Int, t / dt) + 1
        j = round(Int, tp / dt) + 1
        K, L = cons(ax, q)
        return matrix(K * L)[i, j]
    end

    # retarded x acausal: trapezoid restores second order
    rac = (ax, q) -> (
        RetardedKernel(ax, Stationary(k1); compression=NONCompression(), quadrature=q),
        AcausalKernel(ax, Stationary(g); compression=NONCompression(), quadrature=q))
    ref = entry(1601, TrapezoidQuadrature(), rac)
    e1 = abs(entry(101, TrapezoidQuadrature(), rac) - ref)
    e2 = abs(entry(201, TrapezoidQuadrature(), rac) - ref)
    @test 3.0 < e1 / e2 < 5.5
    r1 = abs(entry(101, RectangleQuadrature(), rac) - ref)
    @test r1 > e1

    # degenerate boundary row: exactly zero under the trapezoid rule
    # (the collapse point carries a zero panel), nonzero under the
    # historical rectangle rule
    ax = range(0.0, T; length=101)
    K, L = rac(ax, TrapezoidQuadrature())
    @test maximum(abs.(matrix(K * L)[1, :])) == 0.0
    K, L = rac(ax, RectangleQuadrature())
    @test maximum(abs.(matrix(K * L)[1, :])) > 0.0

    # retarded x retarded: already second order under both rules (dressing)
    rr = (ax, q) -> (
        RetardedKernel(ax, Stationary(k1); compression=NONCompression(), quadrature=q),
        RetardedKernel(ax, Stationary(k2); compression=NONCompression(), quadrature=q))
    refrr = entry(1601, RectangleQuadrature(), rr)
    a1 = abs(entry(101, RectangleQuadrature(), rr) - refrr)
    a2 = abs(entry(201, RectangleQuadrature(), rr) - refrr)
    @test 3.0 < a1 / a2 < 5.5
end

@testitem "Quadrature axis: acausal x advanced under the trapezoid rule" begin
    T = 1.0
    g2(τ) = ComplexF64(exp(-((τ - 0.3)^2) / 0.5) * (1.0 + 0.5im))
    k3(τ) = ComplexF64(exp(-(τ + 0.1)^2))
    t, tp = 0.6, 0.2
    function entry(N, q)
        ax = range(0.0, T; length=N)
        dt = step(ax)
        i = round(Int, t / dt) + 1
        j = round(Int, tp / dt) + 1
        A = AcausalKernel(ax, Stationary(g2); compression=NONCompression(), quadrature=q)
        B = AdvancedKernel(ax, Stationary(k3); compression=NONCompression(), quadrature=q)
        return matrix(A * B)[i, j]
    end
    ref = entry(1601, TrapezoidQuadrature())
    e1 = abs(entry(101, TrapezoidQuadrature()) - ref)
    e2 = abs(entry(201, TrapezoidQuadrature()) - ref)
    @test 3.0 < e1 / e2 < 5.5
    r1 = abs(entry(101, RectangleQuadrature()) - ref)
    r2 = abs(entry(201, RectangleQuadrature()) - ref)
    @test 1.5 < r1 / r2 < 2.7
    @test e1 < r1
    # degenerate boundary column (t' = t₀): exactly zero under trapezoid
    ax = range(0.0, T; length=101)
    A = AcausalKernel(ax, Stationary(g2); compression=NONCompression(), quadrature=TrapezoidQuadrature())
    B = AdvancedKernel(ax, Stationary(k3); compression=NONCompression(), quadrature=TrapezoidQuadrature())
    @test maximum(abs.(matrix(A * B)[:, 1])) == 0.0
    Ar = AcausalKernel(ax, Stationary(g2); compression=NONCompression(), quadrature=RectangleQuadrature())
    Br = AdvancedKernel(ax, Stationary(k3); compression=NONCompression(), quadrature=RectangleQuadrature())
    @test maximum(abs.(matrix(Ar * Br)[:, 1])) > 0.0
end
