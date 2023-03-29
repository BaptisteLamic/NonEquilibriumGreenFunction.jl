
# We group the problem definition in a module. 
using Revise
module SQDSJunction
using Revise
using NonEquilibriumGreenFunction
using DSP
using LinearAlgebra
using Symbolics

mutable struct SQDSParameters{T}
    t_l::T
    t_r::T
    β::T
    Δ::T #Must be real
    η::T
    ρ::T
end
function SQDSParameters(::Type{T}; args...) where {T}
    param = SQDSParameters{T}(
        1,                  # t_l
        1,                  # t_R
        100,                # β
        1,                  # Δ
        0,                   # η
        1,                  # ρ
    )
    for (key, value) in args
        setfield!(param, key, T(value))
    end
    param
end
SQDSParameters(; args...) = SQDSParameters(ComplexF64; args...)
function copy(opts::SQDSParameters{T}; args...) where {T}
    opts_ = SQDSParameters(T)
    for field in fieldnames(typeof(opts))
        setfield!(opts_, field, getfield(opts, field))
    end
    for (key, value) in args
        if key in fieldnames(typeof(opts))
            setfield!(opts_, key, value |> T)
        end
    end
    opts_
end

#Analytical expression of the bare Green functions and couplings
##Dot at the particle-hole symmetry
#Green function at the particle-hole symemtry
f_gRd(t::Number, param::SQDSParameters{T}) where {T} = -1im * pauli(0) .|> T
f_gRd(t::Number, tp::Number, param::SQDSParameters) = f_gRd(t - tp, param)

##Lead bare Green functions in the energy domain
f_gl_w_non_local(w::Number, p::SQDSParameters) = p.ρ * pi * ((-(w + 1im * p.η) * pauli(0) + p.Δ * pauli(1)) / sqrt(p.Δ^2 - (w + 1im * p.η)^2) + 1im * pauli(0))
f_gl_local(w::Number, p::SQDSParameters) = -p.ρ * 1im * pauli(0) * pi

##Coupling
function f_T(t::Number, ϕ, side::Symbol, param::SQDSParameters)
    if side == :l
        τ = param.t_l
    elseif side == :r
        τ = param.t_r
    else
        throw(ArgumentError("side must be :l or :r"))
    end
    return τ * pauli(3) * exp(1im * ϕ(t) * pauli(3) / 2)
end

function default_compression()
    HssCompression(; atol=5E-4, rtol=5E-4, kest=20, leafsize=256)
end

function retarded_correlator(ax, ϕ_l, ϕ_r, param::SQDSParameters=SQDSParameters(); cpr=default_ompression(), args...)
    param = isempty(args) ? param : copy(param; args...)

    gR_dot = RetardedKernel(ax, (t, tp) -> f_gRd(t, tp, param), stationary=true, compression=cpr)
    #The superconducting lead Green function contain a local and a non lcoal part
    oversampling = get(args, :oversampling, 1)
    @show oversampling
    non_local_gR_lead = energy2RetardedKernel(w -> f_gl_w_non_local(w, param), RetardedKernel,
        ax, compression=cpr, oversampling=oversampling)
    local_gR_lead = TimeLocalKernel(ax, t -> f_gl_local(t, param), compression=cpr)
    gR_lead = local_gR_lead + non_local_gR_lead

    #build the self-energy
    Tl = TimeLocalKernel(ax, t -> f_T(t, ϕ_l, :l, param), compression=cpr)
    Tr = TimeLocalKernel(ax, t -> f_T(t, ϕ_r, :r, param), compression=cpr)
    ΣR_l = adjoint(Tl) * gR_lead * Tl |> compress!
    ΣR_r = adjoint(Tr) * gR_lead * Tr |> compress!
    ΣR = ΣR_l + ΣR_r
    ΣR = ΣR |> compress!

    #Solve for the retarded Green function 
    GR = compress!(I - gR_dot * ΣR) \ gR_dot |> evaluate_expression |> compress!
    return (; GR, gR_dot, gR_lead, ΣR_l, ΣR_r, ΣR,
        Tl, Tr, param=param)
end

function kinetic_correlator(retarded::NamedTuple)
    ax = axis(retarded[:GR])
    param = retarded[:param]
    cpr = compression(retarded[:GR])
    T = scalartype(retarded[:GR])

    #Construction of the Keldysh thermal equilibrium function 
    thker = Kernel(ax, (t, tp) -> thermal_kernel(t - tp, param.β) * pauli(0) .|> T, stationary=true, compression=cpr)
    gK_dot = (retarded[:gR_dot] * thker - thker * retarded[:gR_dot]') |> compress!
    gK_lead = (retarded[:gR_lead] * thker - thker * retarded[:gR_lead]') |> compress!
    ΣK_l = adjoint(retarded[:Tl]) * gK_lead * retarded[:Tl] |> compress!
    ΣK_r = adjoint(retarded[:Tr]) * gK_lead * retarded[:Tr] |> compress!
    ΣK = ΣK_l + ΣK_r |> compress!
    #Compute the full Green function for an initially empty dot
    GK = retarded[:GR] * ΣK * retarded[:GR]' |> compress!
    return merge(retarded, (; GK, gK_dot, gK_lead, ΣK_l, ΣK_r, ΣK))
end
function kinetic_correlator(ax, ϕ_l, ϕ_r, param::SQDSParameters=SQDSParameters(); cpr=default_compression(), args...)
    return kinetic_correlator(
        retarded_correlator(ax, ϕ_l, ϕ_r, param; cpr=cpr, args...)
    )
end

function current(; G_R , G_K, Σ_R , Σ_K)
    function RAK_rotation()
    (1 .- 1im*pauli(2))#/sqrt(2)
    end
    function pauliRAK(k)
        RAK_rotation()*pauli(k)*RAK_rotation()'/2
    end
    τ_z = pauliRAK(3)
    G = [0 G_R'; G_R G_K]
    Σ = [Σ_K Σ_R; Σ_R' 0 ]
    step1 = tr(τ_z*(G*Σ - Σ*G))
    step2 = step1[1,1] + step1[2,2]
    #trace in nambu space + effect of σ_z
    step3 = diag(op_I)
    return step3[1:2:end] - step3[2:2:end]
end

end

##
using LinearAlgebra
#Better performance are observed when the BLAS multithreading is disactivated
BLAS.set_num_threads(1)

## Define problem paramaters 
param = SQDSJunction.SQDSParameters(Δ = 1.,t_l = 1, t_r = 1, η = 0.01)
γl = π*abs(param.t_l)^2 * param.ρ * 2 |> Float64 
#the 2 is for the spin degree of freedom
γr = π*abs(param.t_r)^2 * param.ρ * 2 |> Float64
Γ = γl + γr
dt  = 0.2 * min(1/Γ,2pi/param.Δ |> real)
@show dt

#define the compression 
cpr = HssCompression(; atol = 5E-6, rtol = 5E-6, kest = 20, leafsize = 32)
#Let's perform several simulations for various simulation length
N = [ 2^k for k in 6:2:12 ]
@time result = map(N) do n
    tmax = dt .* n 
    ax = (1:n) * dt
    ϕ_l = t->4pi * t / tmax
    ϕ_r = t->0
    println("$n started")
    timing = @elapsed sim_data = SQDSJunction.kinetic_correlator(ax,ϕ_l, ϕ_r,param; cpr = cpr, oversampling = 2)
    timing_current = @elapsed current = SQDSJunction.current(
        G_R = sim_data[:GR],
        G_K = sim_data[:GK],
        Σ_R = sim_data[:ΣR], 
        Σ_K = sim_data[:ΣK]  )
    println("$n done")
    (;ax, ϕ = ϕ_l.(ax), timing, timing_current, current,
        gr = sim_data[:gR_dot], gR_lead = sim_data[:gR_lead],
        gk = sim_data[:gK_dot], gK_lead = sim_data[:gK_lead],
        GR = sim_data[:GR], GK = sim_data[:GK] )
end;
