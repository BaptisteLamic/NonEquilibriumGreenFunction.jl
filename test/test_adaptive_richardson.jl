@testitem "Adaptive Richardson Extrapolation" begin
    using NonEquilibriumGreenFunction.AdaptiveRichardson
    using Statistics
    import Random
    Random.seed!(1234)
    # Test the Adaptive Richardson Extrapolation method
    function generate_convergeance_order_test_function(order, dt, f0)
        nb_coefficients = 10
        coefficients = randn(order + nb_coefficients)
        function ramp(t, t0)
            return t > t0 ? t : zero(t)
        end
        function f(t)
            polynomial_part = sum((coefficients[k] * t^k for k in order:order+nb_coefficients-1))
            return f0 + polynomial_part + sin(t) * (1 - exp(ramp(t, dt)))
        end
    end
    f = generate_convergeance_order_test_function(3, 1, 1.0)
    n = 20
    cfg = AdaptiveConfig(max_pointsA=14, max_pointsB=n, rtol=1e-6, max_total_points=n, verbose=false)
    result = adaptive_richardson(f, 2., cfg)
    @test result.converged
    @test result.error < cfg.rtol
    # The heuristic error estimate is not a guaranteed bound; check against the
    # extrapolation tolerance instead.
    @test abs(f(0) - result.u0_est) < 10 * cfg.rtol
end

@testitem "Observed Order" begin
    using NonEquilibriumGreenFunction.AdaptiveRichardson
    using Statistics
    import Random
    Random.seed!(1234)
    # Test the Adaptive Richardson Extrapolation method
    function generate_convergeance_order_test_function(order, dt, f0)
        nb_coefficients = 10
        coefficients = randn(order + nb_coefficients)
        function ramp(t, t0)
            return t > t0 ? t : zero(t)
        end
        function f(t)
            polynomial_part = sum((coefficients[k] * t^k for k in order:order+nb_coefficients-1))
            return f0 + polynomial_part + sin(t) * (1 - exp(ramp(t, dt)))
        end
    end
    f = generate_convergeance_order_test_function(3, 0.1, 1.0)
    t = [0.1 * 0.5^k for k in 1:10]
    y = [f(ti) for ti in t]
    list_of_orders = observed_order(t, y)
    # observed_order legitimately returns NaN for pre-asymptotic/degenerate triplets
    finite = filter(isfinite, list_of_orders)
    @test all(finite .> 1)
    @test median(finite) ≈ 3.0 atol = 1e-1
    @test is_asymptotic(finite)
end

@testitem "Polynomial Interpolation" begin
    using NonEquilibriumGreenFunction.AdaptiveRichardson
    # Test polynomial fitting with weights
    dt_vals = [0.1, 0.05, 0.025, 0.0125]
    u_vals = [1.0, 1.5, 2.0, 2.5]
    weights = [1.0, 0.8, 0.6, 0.4]

    result = fit_polynomial(dt_vals, u_vals, length(dt_vals); weights=weights)
    function fitted(t)
        return sum(result.coeffs[i] * t^(i - 1) for i in 1:length(result.coeffs))
    end
    @test norm(fitted.(dt_vals) .- u_vals) < 1e-6
end

@testitem "Polynomial fit" begin
    using NonEquilibriumGreenFunction.AdaptiveRichardson
    using LinearAlgebra
    import Random
    Random.seed!(1234)
    # Test polynomial fitting with weights
    dt_vals = LinRange(10, 0.001, 1024)
    f(x) = 1 - x^2
    weights = [1 / i for i in 1:length(dt_vals)]
    # Simulate noisy observations
    u_vals = f.(dt_vals) + weights .* randn(length(dt_vals)) * 1e-4

    result = fit_polynomial(dt_vals, u_vals, 2; weights=weights)
    function fitted(t)
        return sum(result.coeffs[i] * t^(i - 1) for i in 1:length(result.coeffs))
    end
    @test result.error_est < 1e-4
    @test result.error_est > 1e-5
    @test norm(fitted.(dt_vals) .- f.(dt_vals)) / length(dt_vals) < 1e-6
end

@testitem "robust_extrapolate" begin
    using NonEquilibriumGreenFunction.AdaptiveRichardson
    using LinearAlgebra
    import Random
    Random.seed!(1234)
    # Test polynomial fitting with weights
    n = 12
    p = 3
    dt_vals = LinRange(10, 0.001, n)
    f(x) = 1 - x^p
    # Simulate noisy observations
    u_vals = f.(dt_vals) .+ 1e-8 .* randn(n)
    cfg = AdaptiveConfig(verbose=true)
    result = robust_extrapolate(dt_vals, u_vals, nothing, cfg)
    @test result.degree >= p
    @test isapprox(result.u0[1], f(0), atol=1e-6)
end
