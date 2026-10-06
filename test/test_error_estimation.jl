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
