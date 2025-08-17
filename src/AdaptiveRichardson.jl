module AdaptativeRichardson
using LinearAlgebra
using Statistics

export AdaptativeConfig
export observed_order, is_asymptotic

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
    condition_threshold::Float64 = 1e12
    
    # Control
    max_time::Union{Float64, Nothing} = nothing
    verbose::Bool = false
    
end

function is_asymptotic(order_history, cfg::AdaptativeConfig=AdaptativeConfig())
    if length(order_history) < cfg.window
        return false
    end
    recent = order_history[end-cfg.window+1:end]
    any(!isfinite, recent) && return false

    #Check convergeance to theoritical order
    deviations = abs.(recent .- cfg.p_theory)
    mean_dev = mean(deviations)
    var_dev = var(recent)
    if  isnan(cfg.p_theory)
        return sqrt(var_dev) < cfg.p_variance_tol
    else
        return mean_dev < cfg.p_tol && sqrt(var_dev) < cfg.p_variance_tol
    end
end

function  observed_order(x,y, cfg::AdaptativeConfig=AdaptativeConfig())
    if length(x) != length(y)
        throw(DomainError((x, y), "x and y must have the same length"))
    end
    if length(x) <= 2
        throw(DomainError((x, y), "x and y must have more than 2 elements"))
    end
    #Use successive triplets to evaluate a sequence of convergence orders
    successives_differences = norm.(y[2:end] .- y[1:end-1])
    ratio = (x[2:end-1] - x[1:end-2]) ./ (x[3:end] - x[2:end-1])
    return map(zip(successives_differences[2:end], successives_differences[1:end-1], y[1:end-2],y[2:end-1], y[3:end], ratio)) do (e2, e1, u1, u2, u3, r)
        if e1 <= cfg.min_error_ratio * max(norm(u1), norm(u2)) || e2 <= cfg.min_error_ratio * max(norm(u2), norm(u3)) || r <= 1.0
            return NaN
        end
        
        ratio = e1 / e2
        if ratio <= cfg.min_error_ratio || ratio >= 1/cfg.min_error_ratio
            return NaN
        end
        return log(ratio) / log(r)
    end
end

end

@testitem "Adaptative Richardson Extrapolation" begin
    using NonEquilibriumGreenFunction.AdaptativeRichardson
    using Statistics
    # Test the Adaptative Richardson Extrapolation method
    function generate_convergeance_order_test_function(order, dt,f0)
        nb_coefficients = 10
        coefficients = randn(order + nb_coefficients)
        function f(t)
            if t > dt
                return sin(t)*t^(order+2)
            else
                return f0 + sum((coefficients[k]*t^k for k in order:order+nb_coefficients - 1 ))
            end
        end
    end
    f = generate_convergeance_order_test_function(3, 0.1, 1.0)
    t = [0.1 * 0.5^k for k in 1:10]
    y = [f(ti) for ti in t]
    list_of_orders = observed_order(t, y)
    @test mean(list_of_orders) ≈ 3.0 atol=1e-1
    @test is_asymptotic(list_of_orders)
end