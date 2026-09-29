"""
    AbstractKernelMap

Abstract type describing the time dependence of a continuous kernel.

Subtypes:
- [`Stationary`](@ref): `f(τ)` depending only on the time difference
- [`TwoTime`](@ref): general `f(t, t')`
- [`Separable`](@ref): `f(t) * g(t')`
"""
abstract type AbstractKernelMap end

"""
    Stationary(f) <: AbstractKernelMap

Wraps a one-argument function `f(τ)` describing a time-translation
invariant kernel `K(t, t') = f(t - t')`. The single-argument signature
structurally guarantees stationarity; the discretization exploits it
with a circulant structure.
"""
struct Stationary{F} <: AbstractKernelMap
    f::F
end

(s::Stationary)(t, tp) = s.f(t - tp)

"""
    TwoTime(f) <: AbstractKernelMap

Wraps a general two-time function `f(t, t')`.
"""
struct TwoTime{F} <: AbstractKernelMap
    f::F
end

(tw::TwoTime)(t, tp) = tw.f(t, tp)

"""
    Separable(f, g) <: AbstractKernelMap

Wraps two one-argument functions describing the separable kernel
`K(t, t') = f(t) * g(t')`. This structure is exploited by the
low-rank compression path.
"""
struct Separable{F,G} <: AbstractKernelMap
    f::F
    g::G
end

function (sep::Separable)(t, tp)
    return sep.f(t) * sep.g(tp)
end

"""
    Singular(f) <: AbstractKernelMap

Wraps the one-argument stationary core `f(τ)` of a kernel that is
principal-value singular at `τ = 0` (e.g. the thermal Keldysh core
`-i/β csch(πτ/β)`). Sampling `f` on the time grid loses one order of
convergence, so a `Singular` map is discretized with product-integration
weights: the matrix entries are `W_k / δt` with
`W_k = PV∫ f(τ) ℓ_k(τ) dτ` the exact integrals against
the piecewise-linear hat functions of the grid. This restores second-order
accuracy for products with smooth kernels. The diagonal weight vanishes
(`W_0 = 0` for the odd thermal core), which is the principal-value
prescription.
"""
struct Singular <: AbstractKernelMap
    f
end

(s::Singular)(t, tp) = s.f(t - tp)

"""
    blocksize_and_eltype(m::AbstractKernelMap, axis)

Sample the map at the first axis point and return `(bs, T)` after
checking that the block is square.
"""
function blocksize_and_eltype(m::AbstractKernelMap, axis)
    f00 = m(axis[1], axis[1])
    size(f00, 1) == size(f00, 2) || throw(ArgumentError(
        "kernel map must return square matrices, got $(size(f00))"))
    return size(f00, 1), eltype(f00)
end
