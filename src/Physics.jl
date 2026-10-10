module Physics

using LinearAlgebra: diagm, I
using ..NonEquilibriumGreenFunction: polygamma
import ..NonEquilibriumGreenFunction.Kernels: Kernel, solve_dyson, causality, isretarded, isadvanced, isacausal,
    adjoint, keldysh_trace, estimate_discretization_error, DiscretizationErrorEstimate, make_similar
import ..NonEquilibriumGreenFunction.Kernels: SumOperator
import ..NonEquilibriumGreenFunction: islocal, to_cpu, quadrature, TrapezoidQuadrature, RectangleQuadrature
import ..NonEquilibriumGreenFunction: matrix, axis, blocksize, compression, step, blockrange, scalartype
import ..NonEquilibriumGreenFunction: extract_blockdiag as _extract_blockdiag
import ..NonEquilibriumGreenFunction: Retarded, Advanced, Acausal
import Base: *

include("Physics/physics.jl")

# A Local (contact) factor is causality-neutral: a sum mixing a contact term
# with a smooth retarded (resp. acausal) operator is a valid retarded (resp.
# acausal) self-energy even though the summed causality of a mixed sum is
# Acausal. Validation therefore recurses into sums and checks only the smooth
# components.
function _smooth_causality_ok(x, ::Type{C}) where {C}
    x isa SumOperator && return (
        _smooth_causality_ok(x.left, C) && _smooth_causality_ok(x.right, C))
    return islocal(x) || causality(x) isa C
end

"""
    solve_keldysh(g, Σ_R, Σ_K; check=true)

Solve the Dyson equation for the retarded branch and dress the kinetic branch.

`g` is the bare retarded Green function, `Σ_R` the retarded self-energy and
`Σ_K` the Keldysh self-energy. Returns `(; G_R, G_K)` where
`G_R = solve_dyson(g, g * Σ_R)` and `G_K = G_R * Σ_K * G_R'`.

`Σ_R` and `Σ_K` are the *total* self-energies of the central region (sum over
all leads); pass the lead-only self-energies to `lead_current`, not here.

When `check` is true the causalities are validated: `g` and `Σ_R` must be
retarded or instantaneous and `Σ_K` acausal or instantaneous.
"""
function solve_keldysh(g::Kernel, Σ_R, Σ_K; check=true)
    if check
        retarded_ok(x) = _smooth_causality_ok(x, Retarded)
        acausal_ok(x) = _smooth_causality_ok(x, Acausal)
        @assert retarded_ok(g) "g must be retarded or instantaneous"
        @assert retarded_ok(Σ_R) "Σ_R must be retarded or instantaneous"
        @assert acausal_ok(Σ_K) "Σ_K must be acausal or instantaneous"
    end
    G_R = solve_dyson(g, g * Σ_R)
    G_K = G_R * Σ_K * G_R'
    return (; G_R, G_K)
end

"""
    lead_current(G_R, G_K, Σ_R, Σ_K; check=true)

Average current through a lead in terms of the full Green functions and the
*lead-only* self-energies, in the Keldysh-symmetric representation (see Jauho,
Haug & Wilkins, and the package examples for the derivation):

`-[ (1/2)((Σ_R G_R)' - Σ_R G_R) + G_R Σ_K + G_K Σ_R' - (Σ_K G_R' + Σ_R G_K) + (3/2)(G_R Σ_R - (G_R Σ_R)') ]`

Returns the operator whose block-diagonal encodes the time-resolved current;
extract the physical signal with `current_signal` (do not read `diag` of the raw
matrix for `bs > 1`).

Convention: positive current flows *into* the lead. `G_R`/`G_K` are the full
(retarded/kinetic) Green functions of the central region dressed with the
total self-energy (e.g. from `solve_keldysh`), while `Σ_R`/`Σ_K` here are the
self-energies of the considered lead only.

When `check` is true the causalities are validated: `G_R` must be retarded,
`G_K` and `Σ_K` acausal-or-instantaneous, and `Σ_R` retarded-or-instantaneous.
"""
function lead_current(G_R, G_K, Σ_R, Σ_K; check=true)
    if check
        retarded_ok(x) = isretarded(x) || islocal(x)
        acausal_ok(x) = isacausal(x) || islocal(x)
        @assert isretarded(G_R) "G_R must be retarded"
        @assert retarded_ok(Σ_R) "Σ_R must be retarded or instantaneous"
        @assert acausal_ok(G_K) "G_K must be acausal or instantaneous"
        @assert acausal_ok(Σ_K) "Σ_K must be acausal or instantaneous"
    end
    ΣRG_R = Σ_R * G_R
    G_RΣ_R = G_R * Σ_R
    return -(
        (1 // 2) * (adjoint(ΣRG_R) - ΣRG_R) +
        G_R * Σ_K +
        G_K * adjoint(Σ_R) -
        (Σ_K * adjoint(G_R) + Σ_R * G_K) +
        (3 // 2) * (G_RΣ_R - adjoint(G_RΣ_R))
    )
end

"""
    current_signal(op)

Time-resolved signal of a current/observable operator: the trace over the
Keldysh indices of each same-time block, one value per time step. For
`bs == 1` this is `diag(matrix(op))`.

`lead_current` returns an operator; use this function to extract the
physical signal from it.
"""
current_signal(op) = keldysh_trace(op)

"""
    KeldyshErrorEstimate

Full-flow discretization-error estimate of a Keldysh simulation
(`solve_keldysh` and everything built from its output).

Fields:

- `G_R`: retarded-branch estimate (`DiscretizationErrorEstimate` of the
  underlying `solve_dyson` solve);
- `G_K`: kinetic-branch estimate (`DiscretizationErrorEstimate` of
  `G_K = G_R Σ_K G_R'`);
- `corrected`: named tuple `(; G_R, G_K)` of *corrected* Green functions
  (`G + estimate`, the estimated error added back), ready to be passed to
  `lead_current` or any user observable;
- `norm_estimate`: estimated error of `G_R` and `G_K` (max-norm of the
  two branch estimates);
- `norm_bound`: upper bound of the same (rigorous given the true defect,
  conditionally rigorous given its leading-order approximation, see
  [`estimate_discretization_error`](@ref)).

The correction is deduced from the estimate, not from a refined solve:
adding it back improves the retarded branch (the scheme error is
asymptotically cancelled) and the kinetic branch at leading order.
"""
struct KeldyshErrorEstimate
    G_R::Any
    G_K::Any
    corrected::Any
    norm_estimate::Float64
    norm_bound::Float64
end

function Base.show(io::IO, r::KeldyshErrorEstimate)
    println(io, "KeldyshErrorEstimate:")
    println(io, "  Estimated error (‖E‖_max):   $(r.norm_estimate)")
    println(io, "  Error bound:                 $(r.norm_bound)")
end

"""
    solve_keldysh_with_error_estimate(g, Σ_R, Σ_K; check=true)

Full-flow variant of [`solve_keldysh`](@ref): solve the Keldysh flow and
simultaneously estimate its time-discretization error and deduce the
corresponding correction.

Returns `(; G_R, G_K, error)`, where `(G_R, G_K)` are the raw solution of
`solve_keldysh` and `error` is a [`KeldyshErrorEstimate`](@ref) whose
`corrected` field contains the improved `(; G_R, G_K)`.

The retarded-branch estimate is the a-posteriori estimate of the underlying
`solve_dyson` solve (see [`estimate_discretization_error`](@ref)); the
kinetic-branch error is obtained by propagating the retarded error through
the dressing `G_K = G_R Σ_K G_R'` and adding the quadrature defect of that
dressing itself, so the full flow — solve, dressing, and any downstream
observable built from `corrected` — carries a consistent error estimate and
correction at the cost of one extra triangular solve and O(N²) block
arithmetic.

When `check` is true the causalities are validated as in `solve_keldysh`.
"""
function solve_keldysh_with_error_estimate(g::Kernel, Σ_R, Σ_K; check=true)
    G_R, G_K = solve_keldysh(g, Σ_R, Σ_K; check=check)
    est = estimate_keldysh_error(g, Σ_R, Σ_K, G_R, G_K; check=check)
    return (; G_R, G_K, error=est)
end

"""
    estimate_keldysh_error(g, Σ_R, Σ_K, G_R, G_K; check=true)

Estimate the time-discretization error of a completed Keldysh simulation
`(G_R, G_K) = solve_keldysh(g, Σ_R, Σ_K)` a posteriori, and deduce the
corresponding corrected Green functions. No refined grid, second problem
or interpolation is involved.

Returns a [`KeldyshErrorEstimate`](@ref) with:

- `G_R`: the `DiscretizationErrorEstimate` of the retarded solve;
- `G_K`: the `DiscretizationErrorEstimate` of the kinetic branch, obtained
  by differentiating the dressing `G_K = G_R Σ_K G_R'` with respect to `G_R`
  and adding the quadrature defect of the two kernel products of the
  dressing (each product contributes an Euler–Maclaurin correction of its
  contraction integral);
- `corrected`: named tuple `(; G_R, G_K)` of corrected Green functions,
  `G + estimate`, directly usable in `lead_current` or any user
  observable: the correction is deduced from the estimate itself;
- `norm_estimate` / `norm_bound`: max over the two branches of the
  estimated error and of the upper bound (rigorous given the true defect,
  conditionally rigorous given its leading-order approximation, see
  [`estimate_discretization_error`](@ref)).

The retarded bound is the componentwise modulus bound of the underlying
`solve_dyson` (see [`estimate_discretization_error`](@ref)); the kinetic
bound propagates it through the dressing (submultiplicatively, via the
weighted column-sum operator norms) and adds the dressing and formation
defects. Both bounds are rigorous given the true defect and conditionally
rigorous given its leading-order approximation.

When `check` is true the causalities are validated as in `solve_keldysh`.
"""
function estimate_keldysh_error(g::Kernel, Σ_R, Σ_K, G_R, G_K; check=true)
    if check
        # a Local (contact) factor is causality-neutral: validate the smooth
        # components of sums recursively (a contact + retarded sum is a valid
        # retarded self-energy even though its summed causality is Acausal)
        retarded_ok(x) = _smooth_causality_ok(x, Retarded)
        acausal_ok(x) = _smooth_causality_ok(x, Acausal)
        @assert isretarded(G_R) "G_R must be retarded"
        @assert retarded_ok(Σ_R) "Σ_R must be retarded or instantaneous"
        @assert acausal_ok(G_K) "G_K must be acausal or instantaneous"
        @assert acausal_ok(Σ_K) "Σ_K must be acausal or instantaneous"
    end
    # the retarded flow is G_R = solve_dyson(g, K) with K = g * Σ_R: the
    # kernel K is itself a quadrature product whose formation defect enters
    # the retarded error to leading order; the estimate of the scheme is
    # built from the *exact* kernel values, so both defects are added.
    K = g * Σ_R
    Dform = _dressing_defect(g, Σ_R, K)
    est_R = estimate_discretization_error(g, K, G_R)
    E_R = est_R.estimate
    # first-order propagation of the formation defect through the solve:
    # A e = Dform * G_R at leading order
    bs = blocksize(G_R)
    n = length(axis(G_R))
    dt = step(G_R)
    MkK = to_cpu(matrix(K))
    T = eltype(MkK)
    diag_K = _extract_blockdiag(MkK, bs)
    left = Matrix{T}(I, bs * n, bs * n) .- dt .* (MkK .- 0.5 .* Matrix(diag_K))
    E_form = left \ to_cpu(matrix(Dform * G_R))
    for j in 1:n
        E_form[blockrange(j, bs), blockrange(j, bs)] .= 0
    end
    E_R_total = make_similar(G_R, compression(G_R)(to_cpu(matrix(E_R)) .+ E_form))
    # defect of the dressing itself: the two kernel products G_R * Σ_K and
    # (G_R * Σ_K) * G_R' are discretized by the quadrature rule; their
    # Euler–Maclaurin correction is estimated from the same coarse data.
    K1 = G_R * Σ_K
    D1 = _dressing_defect(G_R, Σ_K, K1)
    D2 = _dressing_defect(K1, G_R', G_K)
    # first-order propagation of the retarded error through the dressing:
    # δG_K = E_R Σ_K G_R' + G_R Σ_K E_R' (+ dressing defect)
    E_K = E_R_total * Σ_K * G_R' + G_R * Σ_K * E_R_total'
    # total defect of the kinetic branch, in matrix form
    Mk = zeros(eltype(to_cpu(matrix(G_K))), bs * n, bs * n)
    Mk .+= to_cpu(matrix(E_K))
    Mk .+= to_cpu(matrix(D1 * G_R'))
    Mk .+= to_cpu(matrix(D2))
    cp = compression(G_K)
    E_K_full = make_similar(G_K, cp(Mk))
    # rigorous bound: the componentwise retarded bound propagated through
    # the dressing, plus the dressing and formation defects. The formation
    # defect propagates through the same operator A (componentwise, no
    # Gronwall constant): |A⁻¹| (|Dform·G_R|) bounds its contribution to the
    # retarded error; the dressing propagation is submultiplicative via
    # the weighted column-sum operator norms.
    dform = maximum(abs, to_cpu(matrix(Dform * G_R)))
    bound_R = est_R.norm_bound + maximum(abs, inv(left) * (abs.(to_cpu(matrix(Dform * G_R)))))
    bound_K = bound_R * (
        _weighted_colsum_norm(Σ_K * G_R') + _weighted_colsum_norm(G_R * Σ_K)
    ) + maximum(abs, to_cpu(matrix(D1))) + maximum(abs, to_cpu(matrix(D2)))
    G_R_corrected = G_R + E_R_total
    G_K_corrected = G_K + E_K_full
    norm_estimate_R = maximum(abs, to_cpu(matrix(E_R_total)))
    norm_estimate_K = maximum(abs, to_cpu(matrix(E_K_full)))
    est_R_full = DiscretizationErrorEstimate(E_R_total, est_R.defect, norm_estimate_R, bound_R)
    return KeldyshErrorEstimate(
        est_R_full,
        DiscretizationErrorEstimate(
            E_K_full,
            cp(Mk),
            norm_estimate_K,
            bound_K,
        ),
        (; G_R = G_R_corrected, G_K = G_K_corrected),
        max(norm_estimate_R, norm_estimate_K),
        max(bound_R, bound_K),
    )
end

# Quadrature defect of one kernel product of the dressing, estimated at
# leading order from the coarse data. Sums distribute:
# defect(L·(A+B)) = defect(L·A) + defect(L·B), and products involving a
# Local (contact) factor are applied exactly by the operator algebra
# (block-diagonal multiplication, no quadrature), so their defect is zero.
# For Smooth factors the trapezoid rule on each contraction line has
# Euler–Maclaurin correction -(δt²/12)[F'(b) - F'(a)] with
# F(s) = left(t, s) right(s, t'); F' is estimated by one-sided three-point
# finite differences of the coarse matrices, as in the retarded-scheme
# defect. Under RectangleQuadrature the dressed same-causality products are
# identical to the trapezoid ones (only the acausal edge corrections of the
# algebra are quadrature-dependent), so they keep the pure EM correction;
# the acausal rectangle paths lose the half weight of the *domain-edge*
# endpoint, contributing an O(δt) bias term (calibrated and validated
# entry-by-entry against reference quadratures):
#   - Retarded×Acausal / Acausal×Retarded / Acausal×Advanced /
#     Advanced×Acausal: bias -(δt/2) F(lo) (the moving endpoint keeps its
#     half weight from the diagonal dressing, the domain edge does not);
#   - Acausal×Acausal (plain δt·M_L·M_R): bias -(δt/2)(F(lo) + F(hi))
#     (left-rule on both domain edges).
# Intervals shorter than two panels are skipped (the three-point
# differences do not resolve a single panel), leaving an unestimated
# O(δt²) near-edge defect, as for the first superdiagonal of the retarded
# scheme defect. For Singular (product-integration) factors the
# Euler–Maclaurin correction does not strictly apply; the estimate remains
# a useful leading-order diagnostic but carries no bound guarantee there.
function _dressing_defect(left::SumOperator, right, prod::SumOperator)
    return _dressing_defect(left.left, right, prod.left) +
           _dressing_defect(left.right, right, prod.right)
end

function _dressing_defect(left, right::SumOperator, prod::SumOperator)
    return _dressing_defect(left, right.left, prod.left) +
           _dressing_defect(left, right.right, prod.right)
end

function _dressing_defect(left::SumOperator, right::SumOperator, prod::SumOperator)
    # both sides are sums: distribute fully and recompute each sub-product
    return sum(_dressing_defect(li, rj, li * rj)
               for li in _sum_terms(left) for rj in _sum_terms(right))
end

# Flatten a (possibly nested) sum into its leaf terms.
_sum_terms(x) = (x,)
_sum_terms(x::SumOperator) = Iterators.flatten((_sum_terms(x.left), _sum_terms(x.right)))

function _dressing_defect(left::Kernel, right::Kernel, prod::Kernel)
    q = quadrature(left)
    if q == RectangleQuadrature()
        return _rectangle_defect(left, right, prod)
    end
    return _euler_maclaurin_defect(left, right, prod)
end

# Products involving a Local (contact) factor are applied exactly by the
# operator algebra (block-diagonal multiplication, no quadrature): their
# discretization defect is zero. The zero carries the product's causality.
function _dressing_defect(left, right, prod)
    bs = blocksize(prod)
    n = length(axis(prod))
    Z = zeros(scalartype(prod), bs * n, bs * n)
    return make_similar(prod, compression(prod)(Z))
end

function _euler_maclaurin_defect(left::Kernel, right::Kernel, prod::Kernel)
    bs = blocksize(prod)
    dt = step(prod)
    Ml = to_cpu(matrix(left))
    Mr = to_cpu(matrix(right))
    T = eltype(Ml)
    n = length(axis(prod))
    D = zeros(T, bs * n, bs * n)
    @inbounds for j in 1:n
        rj = blockrange(j, bs)
        for i in 1:n
            ri = blockrange(i, bs)
            # integration interval along the contraction variable of the
            # (i, j) entry depends on the causalities of the factors
            lo, hi = _interval(causality(left), causality(right), i, j, n)
            (isnothing(lo) || hi - lo < 2) && continue
            F(s) = Ml[ri, blockrange(s, bs)] * Mr[blockrange(s, bs), rj]
            dF_lo = (-3F(lo) + 4F(lo + 1) - F(lo + 2)) / (2dt)
            dF_hi = (3F(hi) - 4F(hi - 1) + F(hi - 2)) / (2dt)
            D[ri, rj] = -(dt^2 / 12) .* (dF_hi - dF_lo)
        end
    end
    return make_similar(prod, compression(prod)(D))
end

function _rectangle_defect(left::Kernel, right::Kernel, prod::Kernel)
    cl, cr = causality(left), causality(right)
    # same-causality products are dressed identically to the trapezoid rule
    if (cl isa Retarded && cr isa Retarded) || (cl isa Advanced && cr isa Advanced)
        return _euler_maclaurin_defect(left, right, prod)
    end
    bs = blocksize(prod)
    dt = step(prod)
    Ml = to_cpu(matrix(left))
    Mr = to_cpu(matrix(right))
    T = eltype(Ml)
    n = length(axis(prod))
    D = zeros(T, bs * n, bs * n)
    both_edges = cl isa Acausal && cr isa Acausal
    @inbounds for j in 1:n
        rj = blockrange(j, bs)
        for i in 1:n
            ri = blockrange(i, bs)
            lo, hi = _interval(cl, cr, i, j, n)
            (isnothing(lo) || hi - lo < 2) && continue
            F(s) = Ml[ri, blockrange(s, bs)] * Mr[blockrange(s, bs), rj]
            dF_lo = (-3F(lo) + 4F(lo + 1) - F(lo + 2)) / (2dt)
            dF_hi = (3F(hi) - 4F(hi - 1) + F(hi - 2)) / (2dt)
            D[ri, rj] = -(dt^2 / 12) .* (dF_hi - dF_lo)
            if both_edges
                D[ri, rj] .-= (dt / 2) .* (F(lo) + F(hi))
            else
                D[ri, rj] .-= (dt / 2) .* F(lo)
            end
        end
    end
    return make_similar(prod, compression(prod)(D))
end

# Integration interval [lo, hi] (grid indices, inclusive) of the contraction
# variable for the (i, j) entry of a product `left * right`, as discretized by
# the kernel algebra (validated against reference quadratures): the interval
# depends only on the causalities of the factors — retarded rows bound it
# above by `i`, advanced columns bound it below by `j`.
function _interval(c_left, c_right, i, j, n)
    if c_left isa Retarded
        if c_right isa Retarded
            return (j + 1, i - 1)
        elseif c_right isa Acausal
            return (1, i)
        elseif c_right isa Advanced
            return (j + 1, i)
        end
    elseif c_left isa Acausal
        if c_right isa Acausal
            return (1, n)
        elseif c_right isa Advanced
            return (1, j)
        elseif c_right isa Retarded
            return (1, i)
        end
    elseif c_left isa Advanced
        if c_right isa Advanced
            return (i, j - 1)
        elseif c_right isa Retarded
            return (i, j)
        elseif c_right isa Acausal
            return (i, n)
        end
    end
    return nothing
end

# Weighted column-sum style norm of an operator with the quadrature weights
# of its axis: the discrete analogue of ‖∫ K‖ used in the kinetic bound.
function _weighted_colsum_norm(op)
    M = to_cpu(matrix(op))
    bs = blocksize(op)
    n = length(axis(op))
    dt = step(op)
    w = ones(n)
    if quadrature(op) != RectangleQuadrature()
        w[1] *= 0.5
        w[end] *= 0.5
    end
    w .*= dt
    acc = 0.0
    for j in 1:n
        acc = max(acc, sum(abs, view(M, :, blockrange(j, bs))) * w[j])
    end
    return acc
end

export solve_keldysh, lead_current, current_signal
export solve_keldysh_with_error_estimate, estimate_keldysh_error, KeldyshErrorEstimate
end
