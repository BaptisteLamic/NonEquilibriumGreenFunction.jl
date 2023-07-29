using NonEquilibriumGreenFunction
using HssMatrices
using LinearAlgebra
using SpecialFunctions
using Symbolics
#HssMatrices does not benefit from the use of multithreading BLAS.
BLAS.set_num_threads(1)

struct Parameters
    δt::Float64 #timestep
    T::Float64  #simulation length
    Γl::Float64 #left tunneling rate
    Γr::Float64 #right tunneling rate
    β::Float64  #inverse temperature
    Δ::Float64
    ϕl # left phase
    ϕr # right phase
    η
end

default_compression() = HssCompression(leafsize = 64)
function Parameters(; δt, T, Γl=1, Γr=1, β=1000, Δ=1.0, ϕl, ϕr,η = 0)
    return Parameters(δt, T, Γl, Γr, β, Δ, ϕl, ϕr,η)
end
axis(p::Parameters) = 0:p.δt:p.T
σ0() = [1.0 0.0; 0.0 1.0]
σx() = [0 1.0; 1.0 0.0]
σz() = [1.0 0.0; 0.0 -1.0]

# %%
function compute_retarded_lead_green_function(p::Parameters; cpr=default_compression())
    g_R_lead_delta = discretize_dirac(axis(p), t -> -1im * σ0(), compression=cpr)
    g_R_lead_continuous = discretize_retardedkernel(axis(p),
        (t, tp) ->  (p.Δ * besselj0(p.Δ*(t-tp)) * σx()+ 1im * p.Δ * besselj1(p.Δ*(t-tp)) * σ0()),
        compression=cpr, stationary=true)
    g_lead = g_R_lead_delta + g_R_lead_continuous
end

# %%
function compute_GR(p::Parameters;cpr=default_compression())
    g_R_lead = compute_retarded_lead_green_function(p::Parameters; cpr=cpr)
    apodisation(t) = (1-exp(-(t/2)^2))
    coupling_left = discretize_dirac(axis(p), t -> apodisation(t) * sqrt(p.Γl/2) * exp(1im * σz() * p.ϕl(t) / 2) * σz(), compression=cpr)
    coupling_right = discretize_dirac(axis(p), t -> apodisation(t) * sqrt(p.Γr/2) * exp(1im* σz() * p.ϕr(t) / 2) * σz(), compression=cpr)
    Σ_R_left = coupling_left' * g_R_lead * coupling_left
    Σ_R_right = coupling_right' * g_R_lead * coupling_right
    Σ_R = Σ_R_left + Σ_R_right
    g = discretize_retardedkernel(axis(p), (t, tp) -> ComplexF64(-1im)*exp(-(p.η * σ0() + 1im * σz() * 0) * (t-tp) ), compression=cpr, stationary=true)
    G_R = solve_dyson(g, g * Σ_R)
    return (; g_R_lead,  g, G_R, Σ_R_left, Σ_R, coupling_left, coupling_right)
end

# %%
function simulate_junction(p::Parameters; cpr=default_compression())
    results_GR = compute_GR(p,cpr = cpr)
    g_R_lead = results_GR[:g_R_lead]
    coupling_left = results_GR[:coupling_left]
    coupling_right = results_GR[:coupling_right]
    G_R = results_GR[:G_R]
    
    ρ = discretize_acausalkernel(axis(p), (t, tp) -> thermal_kernel(t - tp, p.β) * σ0() .|> ComplexF64,
    stationary=true, compression=cpr)
    g_lead_kinetic = g_R_lead *  ρ  - ρ * g_R_lead'
    g = results_GR[:g]
    g_dot_kinetic = g *  ρ  - ρ * g'
    compress!(g_dot_kinetic)

    Σ_K_left =  coupling_left' * g_lead_kinetic * coupling_left
    Σ_K_right = coupling_right' * g_lead_kinetic * coupling_right
    Σ_K = Σ_K_left + Σ_K_right
    G_K = G_R * Σ_K * G_R' 
    return (;results_GR..., G_K, Σ_K_left, Σ_K_right)
end

# %%
function compute_average_current(results)
    #First we build the expression
    @variables G_R::Kernel G_K::Kernel
    @variables Σ_R::Kernel Σ_K::Kernel
    τz = [1 2; 2 3] // 2
    G = [0 G_R'; G_R G_K]
    Σl = [Σ_K Σ_R; Σ_R' 0]
    expr = simplify_kernel(-tr(τz * (G * Σl - Σl * G)))
    #convert to a julia expression and compile
    f = build_function(expr, G_R, G_K, Σ_R, Σ_K, expression=Val{false})
    #evaluate it
    I_avr_op = f(results[:G_R], results[:G_K], results[:Σ_R_left], results[:Σ_K_left])
    #We have to take the trace on the Keldysh space by hand. 
    dg = diag(matrix(I_avr_op))
    (dg[1:2:end] .- dg[2:2:end])
end

# %%
V = 0.4
Γ = 5
δt = 0.05
Tmax = 300
δϕ(t) = V/Tmax * t^2/2
p =  Parameters(δt=δt, T = Tmax, ϕl=t -> -δϕ(t)/2, ϕr=t -> δϕ(t)/2, Δ = 1, Γl=Γ, Γr=0.5*Γ, β=1000,η = 0 );
@time results =  simulate_junction(p)
@profview simulate_junction(p)
@time Idc =  compute_average_current(results);




