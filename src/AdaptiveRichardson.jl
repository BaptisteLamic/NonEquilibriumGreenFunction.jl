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
    max_pointsA::Int = 4
    max_pointsB::Int = 3
    max_total_points::Int = 7

    # Convergence criteria
    tol_rel::Float64 = 1e-6
    tol_abs::Float64 = 0.0

    # Asymptotic detection
    window::Int = 3
    p_tol::Float64 = 0.15
    p_variance_tol::Float64 = 0.075  # p_tol/2

    # Weighting
    w_decay::Float64 = 0.5
    pre_asymptotic_penalty::Float64 = 0.1

    # Robustness
    max_restarts::Int = 1
    restart_factor::Float64 = 2.0
    min_dt_ratio::Float64 = 1e-12  # relative to dt0

    # Numerical stability
    min_error_ratio::Float64 = 1e-14
    condition_threshold::Float64 = 1e6

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
    errors
    converged::Bool
    total_time::Float64
end

function Base.show(io::IO, r::RichardsonResult)
    println(io, "RichardsonResult:")
    println(io, "  Extrapolated u(0): $(r.u0_est)")
    println(io, "  Points used: $(length(r.dt_list))")
    println(io, "  Phase switch at: $(r.phase_switch_index)")
    println(io, "  Converged: $(r.converged)")
    println(io, "  Polynomial degree: $(r.polynomial_degree)")
    println(io, "  Condition number: $(r.condition_number)")
    println(io, "  Error estimate: $(r.extrapolation_error_est)")
    println(io, "  Total time: $(r.total_time)s")
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
function adaptative_richardson(f, dt0; cfg::AdaptativeConfig=AdaptativeConfig())
    t_start = time()
    function time_budget_ok()
        cfg.max_time === nothing && return true
        return (time() - t_start) < cfg.max_time
    end
    error("adaptative_richardson is not yet implemented.")
    dt_list = [dt0]
    u_list = [f(dt0)]
    p_list = Float64[]
    results = eltype(u_list)[]
    phase_switch_index = nothing
    for i in 1:cfg.max_pointsA
        if !time_budget_ok()
            break
        end
        dt = dt_list[end] / cfg.r_large
        push!(dt_list, dt)
        push!(u_list, f(dt))
        extrapolation = robust_extrapolate(dt_list, u_list, nothing, cfg)
        push!(results, extrapolation.coeffs[1])
        if length(dt_list) >= 3
            p = observed_order(dt_list[end-3:end], u_list[end-3:end], cfg)
            cfg.verbose && @info "Phase A[$i]: dt=$(dt), p_obs=$p"
            push!(p_list, p)
            if is_asymptotic(p_list, cfg)
                phase_switch_index = length(dt_list)
                cfg.verbose && @info "Asymptotic regime detected at point $phase_switch_index"
                break
            end
        end
    end
    #Phase B: refine the last point
    converged = false
    condition_number = NaN
    polynomial_degree = -1
    extrapolation_error_est = NaN
    for i in 1:cfg.max_pointsB
        length(dt_list) >= cfg.max_total_points && break
        !time_budget_ok() && break

        dt = dt_list[end] / cfg.r_small
        push!(dt_list, dt)
        push!(u_list, f(dt))
        p = observed_order(dt_list[end-3:end], u_list[end-3:end], cfg)
        push!(p_list, p)
        cfg.verbose && @info "Phase B[$i]: dt=$(dt), p_obs=$p"
        current_fit = robust_extrapolate(dt_list, u_list, nothing, cfg)
        push!(results, current_fit.coeffs[1])
        condition_number = current_fit.condition
        polynomial_degree = current_fit.degree
        extrapolation_error_est = current_fit.error_est
        if length(results) >= 2
            error_estimate = abs(results[end] - results[end-1])
            if error_estimate / abs(results[end]) < cfg.rtol || error_estimate < cfg.atol
                converged = true
                break
            end
        end
    end
    if !converged
        cfg.verbose && @warn "Adaptative Richardson did not converge within the limits"
        current_fit = robust_extrapolate(dt_list, u_list, nothing, cfg)
        condition_number = current_fit.condition
        polynomial_degree = current_fit.degree
        extrapolation_error_est = current_fit.error_est
    end
    
    total_time = time() - t_start
    
    return RichardsonResult(
        u0_est=final_fit.u0[1],
        dt_list=dt_list,
        u_list=u_list,
        p_obs=p_list,
        errors=abs(results[end] - results[end-1]),
        converged=converged,
        total_time=total_time
    )
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
        return sqrt(var_dev) < cfg.p_variance_tol
    else
        #TODO: review this condition
        return mean_dev < cfg.p_tol && sqrt(var_dev) < cfg.p_variance_tol
    end
end

function observed_order(x, y, cfg::AdaptativeConfig=AdaptativeConfig())
    if length(x) != length(y)
        throw(DomainError((x, y), "x and y must have the same length"))
    end
    if length(x) <= 2
        throw(DomainError((x, y), "x and y must have more than 2 elements"))
    end
    #Use successive triplets to evaluate a sequence of convergence orders
    successives_differences = norm.(y[2:end] .- y[1:end-1])
    ratio = (x[2:end-1] - x[1:end-2]) ./ (x[3:end] - x[2:end-1])
    return map(zip(successives_differences[2:end], successives_differences[1:end-1], y[1:end-2], y[2:end-1], y[3:end], ratio)) do (e2, e1, u1, u2, u3, r)
        if e1 <= cfg.min_error_ratio * max(norm(u1), norm(u2)) || e2 <= cfg.min_error_ratio * max(norm(u2), norm(u3)) || r <= 1.0
            return NaN
        end

        ratio = e1 / e2
        if ratio <= cfg.min_error_ratio || ratio >= 1 / cfg.min_error_ratio
            return NaN
        end
        return log(ratio) / log(r)
    end
end

function robust_extrapolate(dt_vals, u_vals, switch_idx::Union{Int,Nothing}, cfg)
    #TODO: improve fit selection logic
    n = length(dt_vals)
    max_degree = min(3, n - 1)

    best_result = nothing
    best_condition = Inf
    weights = compute_weights(n::Int, switch_idx::Union{Int,Nothing},cfg)
    # Try different polynomial degrees
    for degree in 1:max_degree
        result = fit_polynomial(dt_vals, u_vals, degree, weights=weights)
        if isnothing(result)
            cfg.verbose && @warn "Degree $degree failed: $e"
            continue
        end
        if isnothing(best_result) || (result.condition < cfg.condition_threshold && result.error_est < 0.5 * best_result.error_est)
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
        # Handle array-valued observables
        m = length(vec(sample))
        Wu = zeros(n, m)
        for i in 1:n
            Wu[i, :] = vec(u_vals[i])
        end
        Wu = W * Wu

        β = WΦ \ Wu
        u0_est = reshape(β[1, :], size(vec(sample)))
        residual = norm(WΦ * β - Wu)
    end

    # Condition number and error estimate
    cond_num = cond(WΦ)
    error_est = residual / sqrt(sum(weights))

    return (u0=u0_est, coeffs=β, condition=cond_num,
        residual=residual, error_est=error_est, degree=degree)
end

function compute_weights(n::Int, switch_idx::Union{Int,Nothing},cfg::AdaptativeConfig)
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
        function f(t)
            if t > dt
                return sin(t) * t^(order + 2)
            else
                return f0 + sum((coefficients[k] * t^k for k in order:order+nb_coefficients-1))
            end
        end
    end
    f = generate_convergeance_order_test_function(3, 0.1, 1.0)
    t = [0.1 * 0.5^k for k in 1:10]
    y = [f(ti) for ti in t]
    list_of_orders = observed_order(t, y)
    @test mean(list_of_orders) ≈ 3.0 atol = 1e-1
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
    # Test polynomial fitting with weights
    n = 12
    p = 3
    dt_vals = LinRange(10, 0.001, n) 
    f(x) = 1 - x^3
    # Simulate noisy observations
    u_vals = f.(dt_vals) .+ 1e-6 .* randn(n)
    cfg = AdaptativeConfig(verbose=true)
    result = robust_extrapolate(dt_vals, u_vals, nothing, cfg)
    @test result.degree == p
    @test isapprox(result.u0[1],f(0), atol=1e-6)
end