@testitem "discretization error estimate: oscillatory analytic solution" begin
    using LinearAlgebra
    T = Float64
    gfun(x) = T(sin(9x))
    kfun(x) = T(-cos(9x))
    sol(x) = T((18 * exp(-x / 2) * sin(sqrt(323) * x / 2)) / sqrt(323))
    t0, t1 = 0.0, 10.0
    for N in (2^6 + 1, 2^7 + 1)
        ax = LinRange(t0, t1, N)
        for cpr in (NONCompression(), HssCompression(atol=1E-10, rtol=1E-10, kest=20))
            G0 = RetardedKernel(ax, TwoTime((x, y) -> gfun(x - y)); compression=cpr)
            K = RetardedKernel(ax, TwoTime((x, y) -> kfun(x - y)); compression=cpr)
            G = solve_dyson(G0, K)
            est = estimate_discretization_error(G0, K, G)
            Gtrue = [sol(x - y) * (x >= y) for x in ax, y in ax]
            err = maximum(abs.(Matrix(matrix(G)) .- Gtrue))
            @test est.norm_estimate > 0
            @test abs(est.norm_estimate - err) < 0.5 * err          # asymptotic exactness
            @test est.norm_bound >= err                              # rigorous bound
        end
    end
end

@testitem "discretization error estimate: constant and polynomial kernels" begin
    using LinearAlgebra
    ax = LinRange(0.0, 10.0, 2^7 + 1)
    # G = 1 - ∫G  ⇒  G = e^{-t}
    let g = RetardedKernel(ax, TwoTime((x, y) -> 1.0)),
        K = RetardedKernel(ax, TwoTime((x, y) -> -1.0))
        G = solve_dyson(g, K)
        est = estimate_discretization_error(g, K, G)
        Gtrue = [x >= y ? exp(-(x - y)) : 0.0 for x in ax, y in ax]
        err = maximum(abs.(Matrix(matrix(G)) .- Gtrue))
        @test abs(est.norm_estimate - err) < 0.1 * err
        @test est.norm_bound >= err
    end
    # G = 1 - ∫(t-s) G(s) ds  ⇒  G = cos(t)
    let g = RetardedKernel(ax, TwoTime((x, y) -> 1.0)),
        K = RetardedKernel(ax, TwoTime((x, y) -> -(x - y)))
        G = solve_dyson(g, K)
        est = estimate_discretization_error(g, K, G)
        Gtrue = [x >= y ? cos(x - y) : 0.0 for x in ax, y in ax]
        err = maximum(abs.(Matrix(matrix(G)) .- Gtrue))
        @test abs(est.norm_estimate - err) < 0.1 * err
        @test est.norm_bound >= err
    end
end

@testitem "discretization error estimate: complex and matrix-valued kernels" begin
    using LinearAlgebra
    ax = LinRange(0.0, 10.0, 2^7 + 1)
    # complex scalar kernel
    let g = RetardedKernel(ax, TwoTime((x, y) -> ComplexF64(sin(3(x - y))));
                           compression=NONCompression()),
        K = RetardedKernel(ax, TwoTime((x, y) -> ComplexF64(-cos(3(x - y))));
                           compression=NONCompression())
        G = solve_dyson(g, K)
        est = estimate_discretization_error(g, K, G)
        sol(x) = 6 / sqrt(35) * exp(-x / 2) * sin(sqrt(35) * x / 2)
        Gtrue = [x >= y ? sol(x - y) : 0.0im for x in ax, y in ax]
        err = maximum(abs.(Matrix(matrix(G)) .- Gtrue))
        @test abs(est.norm_estimate - err) < 0.2 * err
        @test est.norm_bound >= err
    end
    # 2×2 matrix kernel: G = I - A∫G  ⇒  G = exp(-A t)
    A = [0.5 0.2; 0.1 0.7]
    let g = RetardedKernel(ax, TwoTime((x, y) -> Matrix{Float64}(I, 2, 2));
                           compression=NONCompression()),
        K = RetardedKernel(ax, TwoTime((x, y) -> -A); compression=NONCompression())
        G = solve_dyson(g, K)
        est = estimate_discretization_error(g, K, G)
        Gtrue = [x >= y ? exp(-A * (x - y)) : zeros(2, 2) for x in ax, y in ax]
        Gtrue_mat = reduce(vcat, [reduce(hcat, Gtrue[i, :]) for i in axes(Gtrue, 1)])
        err = maximum(abs.(Matrix(matrix(G)) .- Gtrue_mat))
        @test abs(est.norm_estimate - err) < 0.1 * err
        @test est.norm_bound >= err
    end
end

@testitem "discretization error estimate: corrected solution improves" begin
    using LinearAlgebra
    ax = LinRange(0.0, 10.0, 2^8 + 1)
    cpr = NONCompression()
    g = RetardedKernel(ax, TwoTime((x, y) -> sin(9(x - y))); compression=cpr)
    K = RetardedKernel(ax, TwoTime((x, y) -> -cos(9(x - y))); compression=cpr)
    G = solve_dyson(g, K)
    est = estimate_discretization_error(g, K, G)
    sol(x) = (18 * exp(-x / 2) * sin(sqrt(323) * x / 2)) / sqrt(323)
    Gtrue = [x >= y ? sol(x - y) : 0.0 for x in ax, y in ax]
    err = maximum(abs.(Matrix(matrix(G)) .- Gtrue))
    corrected = Matrix(matrix(G)) .+ Matrix(matrix(est.estimate))
    err_corr = maximum(abs.(corrected .- Gtrue))
    @test err_corr < 0.5 * err
end

@testitem "discretization error estimate: defect is O(dt^2)" begin
    using LinearAlgebra
    sol(x) = (18 * exp(-x / 2) * sin(sqrt(323) * x / 2)) / sqrt(323)
    norms = Float64[]
    for N in (2^6 + 1, 2^7 + 1, 2^8 + 1)
        ax = LinRange(0.0, 10.0, N)
        g = RetardedKernel(ax, TwoTime((x, y) -> sin(9(x - y))); compression=NONCompression())
        K = RetardedKernel(ax, TwoTime((x, y) -> -cos(9(x - y))); compression=NONCompression())
        G = solve_dyson(g, K)
        est = estimate_discretization_error(g, K, G)
        push!(norms, est.norm_estimate)
    end
    order = log(norms[1] / norms[3]) / log(4)
    @test order > 1.8   # second-order convergence of the estimated error
end

@testitem "keldysh full-flow error estimate: retarded branch" begin
    using LinearAlgebra
    mk(ax) = (RetardedKernel(ax, TwoTime((t, tp) -> exp(-1im * (t - tp))); compression=NONCompression()),
              RetardedKernel(ax, TwoTime((t, tp) -> -0.5im * exp(-2 * (t - tp))); compression=NONCompression()),
              AcausalKernel(ax, TwoTime((t, tp) -> 1.0im * exp(-(t - tp)^2)); compression=NONCompression()))
    for (N1, N2) in ((101, 401), (201, 801))
        ax = range(0.0, 5.0; length=N1)
        ax2 = range(0.0, 5.0; length=N2)
        g, S, K = mk(ax)
        g2, S2, K2 = mk(ax2)
        sol = solve_keldysh(g, S, K)
        sol2 = solve_keldysh(g2, S2, K2)
        est = estimate_keldysh_error(g, S, K, sol.G_R, sol.G_K)
        idx = [round(Int, 1 + (i - 1) * (N2 - 1) / (N1 - 1)) for i in 1:N1]
        ref = Matrix(matrix(sol2.G_R))[idx, idx]
        err_raw = maximum(abs.(Matrix(matrix(sol.G_R)) .- ref))
        err_corr = maximum(abs.(Matrix(matrix(est.corrected.G_R)) .- ref))
        @test est.norm_estimate > 0
        @test abs(est.G_R.norm_estimate - err_raw) < 0.3 * err_raw    # asymptotic exactness
        @test est.G_R.norm_bound >= err_raw                          # rigorous bound
        @test err_corr < 0.5 * err_raw                                # correction improves
        @test est.norm_bound >= est.norm_estimate
    end
end

@testitem "keldysh full-flow error estimate: kinetic branch and correction" begin
    using LinearAlgebra
    mk(ax) = (RetardedKernel(ax, TwoTime((t, tp) -> exp(-1im * (t - tp))); compression=NONCompression()),
              RetardedKernel(ax, TwoTime((t, tp) -> -0.5im * exp(-2 * (t - tp))); compression=NONCompression()),
              AcausalKernel(ax, TwoTime((t, tp) -> 1.0im * exp(-(t - tp)^2)); compression=NONCompression()))
    N1, N2 = 101, 401
    ax = range(0.0, 5.0; length=N1)
    ax2 = range(0.0, 5.0; length=N2)
    g, S, K = mk(ax)
    g2, S2, K2 = mk(ax2)
    sol = solve_keldysh(g, S, K)
    sol2 = solve_keldysh(g2, S2, K2)
    est = estimate_keldysh_error(g, S, K, sol.G_R, sol.G_K)
    idx = [round(Int, 1 + (i - 1) * (N2 - 1) / (N1 - 1)) for i in 1:N1]
    ref = Matrix(matrix(sol2.G_K))[idx, idx]
    err_raw = maximum(abs.(Matrix(matrix(sol.G_K)) .- ref))
    err_corr = maximum(abs.(Matrix(matrix(est.corrected.G_K)) .- ref))
    @test est.G_K.norm_estimate > 0
    @test abs(est.G_K.norm_estimate - err_raw) < 0.3 * err_raw    # leading-order estimate
    @test est.G_K.norm_bound >= err_raw                           # rigorous bound
    @test err_corr < 0.8 * err_raw                                # correction improves
    # the corrected Green functions are valid kernels usable downstream
    I_corr = current_signal(lead_current(est.corrected.G_R, est.corrected.G_K, S, K))
    I_raw = current_signal(lead_current(sol.G_R, sol.G_K, S, K))
    @test length(I_corr) == length(I_raw) == N1
end

@testitem "solve_keldysh_with_error_estimate consistency" begin
    using LinearAlgebra
    ax = range(0.0, 5.0; length=101)
    g = RetardedKernel(ax, TwoTime((t, tp) -> exp(-1im * (t - tp))); compression=NONCompression())
    S = RetardedKernel(ax, TwoTime((t, tp) -> -0.5im * exp(-2 * (t - tp))); compression=NONCompression())
    K = AcausalKernel(ax, TwoTime((t, tp) -> 1.0im * exp(-(t - tp)^2)); compression=NONCompression())
    sol = solve_keldysh(g, S, K)
    sol_est = solve_keldysh_with_error_estimate(g, S, K)
    @test Matrix(matrix(sol_est.G_R)) == Matrix(matrix(sol.G_R))
    @test Matrix(matrix(sol_est.G_K)) == Matrix(matrix(sol.G_K))
    @test sol_est.error isa KeldyshErrorEstimate
    est = estimate_keldysh_error(g, S, K, sol.G_R, sol.G_K)
    @test Matrix(matrix(sol_est.error.corrected.G_R)) ≈ Matrix(matrix(est.corrected.G_R))
    @test Matrix(matrix(sol_est.error.corrected.G_K)) ≈ Matrix(matrix(est.corrected.G_K))
    @test sol_est.error.norm_bound >= sol_est.error.norm_estimate
end

@testitem "keldysh full-flow estimate: O(dt^2) scaling" begin
    using LinearAlgebra
    mk(ax) = (RetardedKernel(ax, TwoTime((t, tp) -> exp(-1im * (t - tp))); compression=NONCompression()),
              RetardedKernel(ax, TwoTime((t, tp) -> -0.5im * exp(-2 * (t - tp))); compression=NONCompression()),
              AcausalKernel(ax, TwoTime((t, tp) -> 1.0im * exp(-(t - tp)^2)); compression=NONCompression()))
    norms = Float64[]
    for N in (2^6 + 1, 2^7 + 1, 2^8 + 1)
        ax = range(0.0, 5.0; length=N)
        g, S, K = mk(ax)
        sol = solve_keldysh(g, S, K)
        est = estimate_keldysh_error(g, S, K, sol.G_R, sol.G_K)
        push!(norms, est.G_R.norm_estimate)
    end
    order = log(norms[1] / norms[3]) / log(4)
    @test order > 1.8   # second-order convergence of the estimated retarded error
end

@testitem "keldysh estimate with contact (local) self-energies" begin
    using LinearAlgebra
    ax = range(0.0, 12.0; length=121)
    cpr = NONCompression()
    g = RetardedKernel(ax, TwoTime((t, tp) -> -1.0im * exp(-1im * 0.0 * (t - tp))); compression=cpr)
    Σ_R = LocalKernel(ax, t -> ComplexF64(-0.5im); compression=cpr)
    ρ = AcausalKernel(ax, TwoTime((t, tp) -> ComplexF64(exp(-((t - tp) / 0.8)^2))); compression=cpr)
    cl = LocalKernel(ax, t -> ComplexF64(sqrt(0.5)); compression=cpr)
    cr = LocalKernel(ax, t -> sqrt(0.5) * exp(1im * 2.0 * t); compression=cpr)
    Σ_K = -2im * (cl' * ρ * cl + cr' * ρ * cr)
    sol = solve_keldysh_with_error_estimate(g, Σ_R, Σ_K)
    @test sol.error isa KeldyshErrorEstimate
    @test sol.error.norm_estimate > 0
    @test sol.error.norm_bound >= sol.error.norm_estimate
    # contact Σ_R has zero formation defect: the retarded estimate must match
    # estimate_discretization_error on the same solve
    K = g * Σ_R
    est = estimate_discretization_error(g, K, sol.G_R)
    @test abs(sol.error.G_R.norm_estimate - est.norm_estimate) < 1e-12
    # correction improves the kinetic branch against a refined reference
    ax2 = range(0.0, 12.0; length=481)
    g2 = RetardedKernel(ax2, TwoTime((t, tp) -> -1.0im); compression=cpr)
    Σ_R2 = LocalKernel(ax2, t -> ComplexF64(-0.5im); compression=cpr)
    ρ2 = AcausalKernel(ax2, TwoTime((t, tp) -> ComplexF64(exp(-((t - tp) / 0.8)^2))); compression=cpr)
    cl2 = LocalKernel(ax2, t -> ComplexF64(sqrt(0.5)); compression=cpr)
    cr2 = LocalKernel(ax2, t -> sqrt(0.5) * exp(1im * 2.0 * t); compression=cpr)
    Σ_K2 = -2im * (cl2' * ρ2 * cl2 + cr2' * ρ2 * cr2)
    ref = solve_keldysh(g2, Σ_R2, Σ_K2)
    idx = [round(Int, 1 + (i - 1) * (481 - 1) ÷ (121 - 1)) for i in 1:121]
    refK = Matrix(matrix(ref.G_K))[idx, idx]
    err_raw = maximum(abs.(Matrix(matrix(sol.G_K)) .- refK))
    err_corr = maximum(abs.(Matrix(matrix(sol.error.corrected.G_K)) .- refK))
    @test err_corr < err_raw
    @test sol.error.G_K.norm_bound >= err_raw
end

@testitem "keldysh full-flow estimate: kinetic branch is O(dt^2)" begin
    using LinearAlgebra
    mk(ax) = (RetardedKernel(ax, TwoTime((t, tp) -> exp(-1im * (t - tp))); compression=NONCompression()),
              RetardedKernel(ax, TwoTime((t, tp) -> -0.5im * exp(-2 * (t - tp))); compression=NONCompression()),
              AcausalKernel(ax, TwoTime((t, tp) -> 1.0im * exp(-(t - tp)^2)); compression=NONCompression()))
    norms = Float64[]
    for N in (2^6 + 1, 2^7 + 1, 2^8 + 1)
        ax = range(0.0, 5.0; length=N)
        g, S, K = mk(ax)
        sol = solve_keldysh(g, S, K)
        est = estimate_keldysh_error(g, S, K, sol.G_R, sol.G_K)
        push!(norms, est.G_K.norm_estimate)
    end
    order = log(norms[1] / norms[3]) / log(4)
    # the kinetic estimate is a leading-order propagation of the O(dt^2)
    # retarded error plus the dressing defects: the measured order is
    # slightly below the clean retarded value 2 pre-asymptotically
    @test order > 1.5   # second-order convergence of the estimated kinetic error
end

@testitem "keldysh estimate with Singular thermal core in Sigma_K" begin
    using LinearAlgebra
    # thermal Keldysh core -i/beta * csch(pi*tau/beta): principal-value singular
    # at tau = 0, discretized with product-integration weights via Singular.
    # The Euler-Maclaurin estimate does not strictly apply to the
    # product-integration rule; this test pins its leading-order behavior:
    # the estimate must remain a positive, same-order diagnostic and the
    # correction must not degrade the kinetic solution.
    beta = 2.0
    ax = range(0.0, 8.0; length=129)
    cpr = NONCompression()
    g = RetardedKernel(ax, TwoTime((t, tp) -> -1.0im); compression=cpr)
    Σ_R = LocalKernel(ax, t -> ComplexF64(-0.5im); compression=cpr)
    ρ_core = Singular(τ -> -1.0im / beta * csch(pi * τ / beta))
    ρ = AcausalKernel(ax, ρ_core; compression=cpr)
    cl = LocalKernel(ax, t -> ComplexF64(sqrt(0.5)); compression=cpr)
    Σ_K = -2im * (cl' * ρ * cl)
    sol = solve_keldysh_with_error_estimate(g, Σ_R, Σ_K)
    @test sol.error isa KeldyshErrorEstimate
    @test sol.error.norm_estimate > 0
    @test sol.error.norm_bound >= sol.error.norm_estimate
    # correction must not degrade the kinetic branch against a refined reference
    ax2 = range(0.0, 8.0; length=513)
    g2 = RetardedKernel(ax2, TwoTime((t, tp) -> -1.0im); compression=cpr)
    Σ_R2 = LocalKernel(ax2, t -> ComplexF64(-0.5im); compression=cpr)
    ρ2 = AcausalKernel(ax2, ρ_core; compression=cpr)
    cl2 = LocalKernel(ax2, t -> ComplexF64(sqrt(0.5)); compression=cpr)
    Σ_K2 = -2im * (cl2' * ρ2 * cl2)
    ref = solve_keldysh(g2, Σ_R2, Σ_K2)
    m = 4
    idx = [1 + (i - 1) * m for i in 1:129]
    refK = Matrix(matrix(ref.G_K))[idx, idx]
    err_raw = maximum(abs.(Matrix(matrix(sol.G_K)) .- refK))
    err_corr = maximum(abs.(Matrix(matrix(sol.error.corrected.G_K)) .- refK))
    @test err_corr < err_raw
    # the estimate is a same-order diagnostic of the true kinetic error
    @test sol.error.G_K.norm_estimate < 10 * err_raw
    @test sol.error.G_K.norm_estimate > 0.1 * err_raw
end

@testitem "keldysh estimate with sum self-energies (nested sums distribute)" begin
    using LinearAlgebra
    # Σ_R as a nested sum of contact + smooth retarded terms, as in the
    # superconductor example (g_lead = local delta + smooth BCS kernel)
    ax = range(0.0, 6.0; length=81)
    cpr = NONCompression()
    g = RetardedKernel(ax, TwoTime((t, tp) -> -1.0im * exp(-1im * 0.5 * (t - tp))); compression=cpr)
    δ_part = LocalKernel(ax, t -> ComplexF64(-0.4im); compression=cpr)
    smooth_part = RetardedKernel(ax, TwoTime((t, tp) -> ComplexF64(-0.3im * exp(-2(t - tp)))); compression=cpr)
    Σ_R_lead = δ_part + smooth_part
    cl = LocalKernel(ax, t -> ComplexF64(0.5); compression=cpr)
    cr = LocalKernel(ax, t -> ComplexF64(0.5); compression=cpr)
    Σ_R = cl' * Σ_R_lead * cl + cr' * Σ_R_lead * cr
    ρ = AcausalKernel(ax, TwoTime((t, tp) -> ComplexF64(exp(-(t - tp)^2))); compression=cpr)
    Σ_K = -2im * (cl' * ρ * cl + cr' * ρ * cr)
    sol = solve_keldysh_with_error_estimate(g, Σ_R, Σ_K)
    @test sol.error isa KeldyshErrorEstimate
    @test sol.error.norm_estimate > 0
    @test sol.error.norm_bound >= sol.error.norm_estimate
    # correction improves the retarded branch against a refined reference
    ax2 = range(0.0, 6.0; length=321)
    g2 = RetardedKernel(ax2, TwoTime((t, tp) -> -1.0im * exp(-1im * 0.5 * (t - tp))); compression=cpr)
    δ2 = LocalKernel(ax2, t -> ComplexF64(-0.4im); compression=cpr)
    sp2 = RetardedKernel(ax2, TwoTime((t, tp) -> ComplexF64(-0.3im * exp(-2(t - tp)))); compression=cpr)
    ΣRl2 = δ2 + sp2
    cl2 = LocalKernel(ax2, t -> ComplexF64(0.5); compression=cpr)
    cr2 = LocalKernel(ax2, t -> ComplexF64(0.5); compression=cpr)
    Σ_R2 = cl2' * ΣRl2 * cl2 + cr2' * ΣRl2 * cr2
    ρ2 = AcausalKernel(ax2, TwoTime((t, tp) -> ComplexF64(exp(-(t - tp)^2))); compression=cpr)
    Σ_K2 = -2im * (cl2' * ρ2 * cl2 + cr2' * ρ2 * cr2)
    ref = solve_keldysh(g2, Σ_R2, Σ_K2)
    m = 4
    idx = [1 + (i - 1) * m for i in 1:81]
    refR = Matrix(matrix(ref.G_R))[idx, idx]
    err_raw = maximum(abs.(Matrix(matrix(sol.G_R)) .- refR))
    err_corr = maximum(abs.(Matrix(matrix(sol.error.corrected.G_R)) .- refR))
    # the correction is same-order as the raw error (leading-order defect
    # propagation); the estimate must cover the true error
    @test err_corr < 2 * err_raw
    @test sol.error.G_R.norm_estimate >= 0.3 * err_raw
    @test sol.error.G_R.norm_bound >= err_raw
end

@testitem "keldysh estimate under RectangleQuadrature (regression: wrong defect model)" begin
    using LinearAlgebra
    # Regression test: the rectangle-rule dressing-defect model must match
    # the actual quadrature of the kernel algebra. The previous model treated
    # rectangle products as plain left-rule sums, which inflated the retarded
    # effectivity to ~60 and made `corrected` 60x WORSE than the raw solution.
    # Same-causality products are dressed identically to trapezoid (pure EM
    # defect); acausal rectangle paths carry an additional O(dt) domain-edge
    # bias term.
    mk(ax, q) = (RetardedKernel(ax, TwoTime((t, tp) -> exp(-1im * (t - tp))); compression=NONCompression(), quadrature=q),
                 RetardedKernel(ax, TwoTime((t, tp) -> -0.5im * exp(-2 * (t - tp))); compression=NONCompression(), quadrature=q),
                 AcausalKernel(ax, TwoTime((t, tp) -> 1.0im * exp(-(t - tp)^2)); compression=NONCompression(), quadrature=q))
    for (N1, N2) in ((51, 201), (101, 401))
        ax = range(0.0, 5.0; length=N1)
        ax2 = range(0.0, 5.0; length=N2)
        g, S, K = mk(ax, RectangleQuadrature())
        g2, S2, K2 = mk(ax2, RectangleQuadrature())
        sol = solve_keldysh(g, S, K)
        sol2 = solve_keldysh(g2, S2, K2)
        est = estimate_keldysh_error(g, S, K, sol.G_R, sol.G_K)
        m = (N2 - 1) ÷ (N1 - 1)
        idx = [1 + (i - 1) * m for i in 1:N1]
        # retarded branch: the estimate must be same-order (was ~60x too large)
        refR = Matrix(matrix(sol2.G_R))[idx, idx]
        err_raw = maximum(abs.(Matrix(matrix(sol.G_R)) .- refR))
        err_corr = maximum(abs.(Matrix(matrix(est.corrected.G_R)) .- refR))
        @test abs(est.G_R.norm_estimate - err_raw) < 0.5 * err_raw
        @test est.G_R.norm_bound >= err_raw
        @test err_corr < 0.8 * err_raw          # correction improves (was ~60x worse)
        # kinetic branch
        refK = Matrix(matrix(sol2.G_K))[idx, idx]
        errK_raw = maximum(abs.(Matrix(matrix(sol.G_K)) .- refK))
        errK_corr = maximum(abs.(Matrix(matrix(est.corrected.G_K)) .- refK))
        @test est.G_K.norm_estimate < 3 * errK_raw
        @test est.G_K.norm_bound >= errK_raw
        @test errK_corr < errK_raw              # correction improves (was ~2.4x worse)
    end
end

@testitem "bound conservatism for unitary kernels and coarse-grid effectivity" begin
    using LinearAlgebra
    cpr = NONCompression()
    # The Gronwall constant exp(||K||_T) ignores phase cancellation: for a
    # pure-phase kernel the bound is formally valid but astronomically
    # conservative. This pins that behavior (documented in the docstring)
    # so a future tightening can flip the assertion.
    ax = range(0.0, 5.0; length=101)
    g = RetardedKernel(ax, TwoTime((t, tp) -> 1.0 + 0im); compression=cpr)
    K = RetardedKernel(ax, TwoTime((t, tp) -> -9.0im); compression=cpr)
    G = solve_dyson(g, K)
    est = estimate_discretization_error(g, K, G)
    Gtrue = [x >= y ? exp(-9im * (x - y)) : 0.0im for x in ax, y in ax]
    err = maximum(abs.(Matrix(matrix(G)) .- Gtrue))
    @test est.norm_bound >= err                    # still rigorous
    @test est.norm_bound / err > 1e10              # but vacuous (~1e18)
    # coarse under-resolved grid: the leading-order estimate under-reports
    # (documented caveat, not a bug)
    axc = LinRange(0.0, 10.0, 11)
    gc = RetardedKernel(axc, TwoTime((t, tp) -> sin(9(t - tp))); compression=cpr)
    Kc = RetardedKernel(axc, TwoTime((t, tp) -> -cos(9(t - tp))); compression=cpr)
    Gc = solve_dyson(gc, Kc)
    estc = estimate_discretization_error(gc, Kc, Gc)
    solc(x) = (18 * exp(-x / 2) * sin(sqrt(323) * x / 2)) / sqrt(323)
    Gtc = [x >= y ? solc(x - y) : 0.0 for x in axc, y in axc]
    errc = maximum(abs.(Matrix(matrix(Gc)) .- Gtc))
    @test estc.norm_bound >= errc                  # bound holds
    @test estc.norm_estimate / errc < 0.5          # estimate under-reports (was ~0.12)
end
