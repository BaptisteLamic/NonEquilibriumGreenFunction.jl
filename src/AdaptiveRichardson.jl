module AdaptativeRichardson
using LinearAlgebra
using Statistics

export AdaptativeConfig
export observed_order, is_asymptotic, fit_polynomial, robust_extrapolate, adaptative_richardson

"""
Configuration for adaptive Richardson extrapolation.
"""
Base.@kwdef struct AdaptativeConfig
    # Refinement ratios
    r_large::Float64 = 2.0
    r_small::Float64 = sqrt(2.0)

    # Method properties
    p_theory::Float64 = NaN  # Theoretical order of convergence, NaN for automatic detection

    # Point limits
    max_pointsA::Int = 5
    max_pointsB::Int = 5
    max_total_points::Int = 10

    # Convergence criteria
    rtol::Float64 = 1e-6
    atol::Float64 = 1e-6

    # Asymptotic detection
    window::Int = 3
    p_tol::Float64 = 0.30
    p_variance_tol::Float64 = 0.2 # p_tol/2

    # Weighting
    w_decay::Float64 = 0.6
    pre_asymptotic_penalty::Float64 = 0.0

    # Robustness
    min_dt_ratio::Float64 = 1e-12  # relative to dt0

    # Numerical stability
    min_error_ratio::Float64 = 1e-14
    condition_threshold::Float64 = 1e12
    # Here the conduition number seem to be not very usefull to actally detect numerical issues.
    # Hence the crazy value.

    # Control
    max_time::Union{Float64,Nothing} = nothing
    verbose::Bool = false

end

"""
Result structure for Richardson extrapolation.
"""
Base.@kwdef struct RichardsonResult{T}
    u0_est::T
    dt_list::Vector{Float64}
    u_list::Vector{T}
    p_obs::Vector{Float64}
    error::Float64
    converged::Bool
    total_time::Float64
    phase_switch_index::Union{Int,Nothing} = nothing  # Index where asymptotic behavior is detected
    polynomial_degree::Int = -1  # Degree of the polynomial used in extrapolation
end

function Base.show(io::IO, r::RichardsonResult)
    println(io, "RichardsonResult:")
    println(io, "  Extrapolated u(0): $(r.u0_est)")
    println(io, "  Points used: $(length(r.dt_list))")
    println(io, "  Median observed order: $(median(filter(!isnan,r.p_obs)))")
    println(io, "  Phase switch at: $(r.phase_switch_index)")
    println(io, "  Converged: $(r.converged)")
    println(io, "  Polynomial degree: $(r.polynomial_degree)")
    println(io, "  Total time: $(r.total_time)s")
    println(io, "  Error estimate: $(r.error)")
end

"""
    adaptative_richardson(f, dt0; config=AdaptiveConfig(), normfun=default_norm)

Advanced adaptive Richardson extrapolation with two-phase refinement.

# Arguments
- `f(dt)`: simulation function returning observable at time step dt
- `dt0`: initial time step

# Keyword Arguments  
- `config::AdaptiveConfig`: configuration parameters

# Returns
- `RichardsonResult`: comprehensive results structure
"""
function adaptative_richardson(f, dt0, cfg::AdaptativeConfig=AdaptativeConfig())
    t_start = time()
    function time_budget_ok()
        cfg.max_time === nothing && return true
        return (time() - t_start) < cfg.max_time
    end
    dt_list = [dt0]
    u_list = [f(dt0)]
    p_list = Float64[]
    results = eltype(u_list)[]
    phase_switch_index = nothing
    converged = false

    function compute_error_estimate(results)
        if length(results) < 2
            error_estimate = NaN
        else
            error_estimate = 10 * abs(results[end] - results[end-1])
        end
        cfg.verbose && @info "Error estimate: $error_estimate"
        return error_estimate
    end

    function extrapolate(dt_list, u_list, phase_switch_index, cfg::AdaptativeConfig)
        return robust_extrapolate(dt_list, u_list, phase_switch_index, cfg)
    end

    function compute_step!(dt)
        push!(dt_list, dt)
        push!(u_list, f(dt))
        @assert length(u_list) >= 2 "At least 2 points are required for extrapolation"
        new_estimate = extrapolate(dt_list, u_list, phase_switch_index, cfg)
        push!(results, new_estimate.coeffs[1])
        if length(dt_list) >= 3
            p = observed_order(dt_list[end-2:end], u_list[end-2:end], cfg)
            push!(p_list, last(p))
        end
        return new_estimate
    end

    function isConverged()
        if length(results) < 2
            return false
        end
        error_estimate = compute_error_estimate(results)
        return (error_estimate / abs(results[end]) < cfg.rtol) || (error_estimate < cfg.atol)
    end
    function build_result(extrapolation)
        return RichardsonResult(
            u0_est=extrapolation.u0[1],
            dt_list=dt_list,
            u_list=u_list,
            p_obs=p_list,
            error=compute_error_estimate(results),
            converged=isConverged(),
            total_time=time() - t_start,
            polynomial_degree=extrapolation.degree,
            phase_switch_index=phase_switch_index
        )
    end
    for i in 2:cfg.max_pointsA
        if !time_budget_ok()
            break
        end
        dt = dt_list[end] / cfg.r_large
        compute_step!(dt)
        if isConverged()
            return build_result(extrapolate(dt_list, u_list, phase_switch_index, cfg))
        end
        if length(dt_list) >= 3
            cfg.verbose && @info "Phase A[$i]: dt=$(dt), p_obs=$(last(p_list))"
            if is_asymptotic(p_list, cfg)
                phase_switch_index = length(dt_list) - cfg.window + 1
                cfg.verbose && @info "Asymptotic regime detected at point $phase_switch_index"
                break
            end
        end
    end
    if !isnothing(phase_switch_index)
        # Recompute the last estimations, taking into account that we are in the asymptotic regime. 
        for i in cfg.window-1:-1:0
            results[end-i] = extrapolate(dt_list[1:end-i], u_list[1:end-i], phase_switch_index, cfg::AdaptativeConfig).coeffs[1]
        end
        #Phase B: refine the last point
        for i in 1:cfg.max_pointsB
            length(dt_list) >= cfg.max_total_points && break
            !time_budget_ok() && break

            dt = dt_list[end] / cfg.r_small
            current_fit = compute_step!(dt)
            cfg.verbose && @info "Phase B[$i]: dt=$(dt), p_obs=$(last(p_list))"
            if length(results) >= 2
                error_estimate = compute_error_estimate(results)
                if isConverged()
                     return build_result(current_fit)
                end
            end
        end
    end
    cfg.verbose && @warn "Adaptative Richardson did not converge within the limits"
    return build_result(robust_extrapolate(dt_list, u_list, phase_switch_index, cfg))
end


function is_asymptotic(order_history, cfg::AdaptativeConfig=AdaptativeConfig())
    if length(order_history) < cfg.window
        return false
    end
    recent = order_history[end-cfg.window+1:end]
    any(!isfinite, recent) && return false

    deviations = abs.(recent .- cfg.p_theory)
    mean_dev = mean(deviations)
    var_dev = var(recent)
    if isnan(cfg.p_theory)
        return var_dev < cfg.p_variance_tol
    else
        #TODO: review this condition
        return mean_dev < cfg.p_tol && sqrt(var_dev) < cfg.p_variance_tol
    end
end

function observed_order(u1::U, u2::U, u3::U, h1::T, h2::T, h3::T, cfg::AdaptativeConfig=AdaptativeConfig()) where {U,T}
    u_ratio = (u2 - u1) / (u3 - u2)
    if u_ratio <= cfg.min_error_ratio
        cfg.verbose && @warn "Observed order is NaN due to small ratio u_ratio = $(u_ratio) at h1=$(h1), h2=$(h2), h3=$(h3)"
        return NaN
    end
    h_ratio = (h2 - h1) / (h3 - h2)
    if h_ratio <= cfg.min_dt_ratio
        cfg.verbose && @warn "Observed order is NaN due to small dt ratio h1=$(h1), h2=$(h2), h3=$(h3)"
        return NaN
    end
    return log(u_ratio) / log(h_ratio)

end

function observed_order(h, u, cfg::AdaptativeConfig=AdaptativeConfig())
    if length(h) != length(u)
        throw(DomainError((h, u), "x and y must have the same length"))
    end
    if length(h) <= 2
        throw(DomainError((h, u), "x and y must have more than 2 elements"))
    end
    #Use successive triplets to evaluate a sequence of convergence orders
    return [observed_order(u[i], u[i+1], u[i+2], h[i], h[i+1], h[i+2], cfg) for i in 1:length(h)-2]
end

function robust_extrapolate(dt_vals, u_vals, switch_idx::Union{Int,Nothing}, cfg; minimum_degree=0)
    #TODO: improve fit selection logic
    n = length(dt_vals)
    if isnothing(switch_idx)
        max_degree = n - 1
    else
        max_degree = n - switch_idx + 1
    end
    best_result = nothing
    best_condition = Inf
    weights = compute_weights(n::Int, switch_idx::Union{Int,Nothing}, cfg)
    # Try different polynomial degrees
    for degree in minimum_degree:max_degree
        result = fit_polynomial(dt_vals, u_vals, degree, weights=weights)
        if isnothing(result)
            cfg.verbose && @warn "Degree $degree failed: $e"
            continue
        end
        if isnothing(best_result) || (result.condition < cfg.condition_threshold && result.error_est < best_result.error_est)
            best_result = result
            best_condition = result.condition
        end
    end

    if best_result === nothing
        # Fallback to unweighted linear fit
        cfg.verbose && @warn "All weighted fits failed, using fallback"
        result = fit_polynomial(dt_vals, u_vals, 1)
    end
    return best_result
end

function fit_polynomial(dt_vals, u_vals, degree; weights=ones(length(dt_vals)))
    n = length(dt_vals)

    # Build Vandermonde matrix
    Φ = [dt_vals[i]^j for i in 1:n, j in 0:degree]
    # Weight matrix
    W = Diagonal(sqrt.(weights))
    WΦ = W * Φ

    # Convert u_vals to matrix form
    sample = u_vals[1]
    if sample isa Number
        Wu = W * Float64.(u_vals)
        β = WΦ \ Wu
        u0_est = β[1]
        residual = norm(WΦ * β - Wu)
    else
        error("u_vals must be a vector of numbers")
    end

    # Condition number and error estimate
    cond_num = cond(WΦ)
    error_est = residual / sqrt(sum(weights))

    return (u0=u0_est, coeffs=β, condition=cond_num,
        residual=residual, error_est=error_est, degree=degree)
end

function compute_weights(n::Int, switch_idx::Union{Int,Nothing}, cfg::AdaptativeConfig)
    w = [cfg.w_decay .^ (n .- i) for i in 1:n]
    # Penalty for pre-asymptotic points
    if !isnothing(switch_idx)
        w[1:switch_idx-1] .*= cfg.pre_asymptotic_penalty
    end
    # Normalize weights
    w ./= sum(w)
    return w
end
end # module AdaptativeRichardson

@testitem "Adaptative Richardson Extrapolation" begin
    using NonEquilibriumGreenFunction.AdaptativeRichardson
    using Statistics
    # Test the Adaptative Richardson Extrapolation method
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
    cfg = AdaptativeConfig(max_pointsA=14, max_pointsB=n, rtol=1e-6, max_total_points=n, verbose=false)
    result = adaptative_richardson(f, 2., cfg)
    @test result.converged
    @test result.error < cfg.rtol
    @test abs(f(0) - result.u0_est) < result.error
end

@testitem "Observed Order" begin
    using NonEquilibriumGreenFunction.AdaptativeRichardson
    using Statistics
    # Test the Adaptative Richardson Extrapolation method
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
    @test all(list_of_orders .> 1)
    @test median(list_of_orders) ≈ 3.0 atol = 1e-1
    @test is_asymptotic(list_of_orders)
end

@testitem "Polynomial Interpolation" begin
    using NonEquilibriumGreenFunction.AdaptativeRichardson
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
    using NonEquilibriumGreenFunction.AdaptativeRichardson
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
    using NonEquilibriumGreenFunction.AdaptativeRichardson
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
    cfg = AdaptativeConfig(verbose=true)
    result = robust_extrapolate(dt_vals, u_vals, nothing, cfg)
    @test result.degree >= p
    @test isapprox(result.u0[1], f(0), atol=1e-6)
end