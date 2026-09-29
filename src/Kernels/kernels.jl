"""
    Kernel{D,C}

A time-domain operator wrapping a discretization and a causality type.

# Fields

- `discretization::D`: the discretized matrix representation
- `causality::C`: one of `Retarded`, `Advanced` or `Acausal`
"""
struct Kernel{D<:AbstractDiscretisation, C<:AbstractCausality} <: SimpleOperator
    discretization::D
    causality::C
end

"""
    discretization(k::Kernel)

Returns the discretization of the kernel.
"""
discretization(k::Kernel) = k.discretization

"""
    causality(k::Kernel)

Returns the causality type of the kernel.
"""
causality(k::Kernel) = k.causality

"""
    isretarded(g)

Returns `true` if the operator is retarded.
"""
isretarded(g::Kernel) = causality(g) == Retarded()

"""
    isadvanced(g)

Returns `true` if the operator is advanced.
"""
isadvanced(g::Kernel) = causality(g) == Advanced()

"""
    isacausal(g)

Returns `true` if the operator is acausal.
"""
isacausal(g::Kernel) = causality(g) == Acausal()

# A Local contact term satisfies every support constraint, so it counts as
# retarded, advanced and acausal at once.
isretarded(g) = islocal(g) || causality(g) == Retarded()
isadvanced(g) = islocal(g) || causality(g) == Advanced()
isacausal(g) = islocal(g) || causality(g) == Acausal()

"""
    locality(::Kernel)

Regular kernels are `Smooth`: they act between distinct time steps and are
integrated by the quadrature rule.
"""
locality(::Kernel) = Smooth()

function make_similar(g::Kernel, new_discretization::AbstractDiscretisation )
    return Kernel(new_discretization, g |> causality)
end

# Add make_similar methods for Kernel to handle matrix and compression cases
function make_similar(g::Kernel, new_matrix::Union{AbstractMatrix,UniformScaling})
    new_dis = make_similar(discretization(g), new_matrix)
    return Kernel(new_dis, causality(g))
end

function make_similar(g::Kernel, new_compression::AbstractCompression)
    new_dis = make_similar(discretization(g), matrix(g), compression = new_compression)
    return Kernel(new_dis, causality(g))
end

function masked(map::AbstractKernelMap, ::Retarded, f00)
    return (t, tp) -> t >= tp ? map(t, tp) : zero(f00)
end

function masked(map::AbstractKernelMap, ::Advanced, f00)
    return (t, tp) -> t <= tp ? map(t, tp) : zero(f00)
end

function masked(map::AbstractKernelMap, ::Acausal, f00)
    return (t, tp) -> map(t, tp)
end

"""
    Kernel(::Type{C}, axis, map::AbstractKernelMap; compression=HssCompression()) where {C<:AbstractCausality}

Discretize the kernel described by `map` masked by the causality `C`
(`Retarded`, `Advanced` or `Acausal`) on `axis`.

Returns a `Kernel` with causality `C`.
"""
function Kernel(::Type{C}, axis, map::AbstractKernelMap;
    compression=HssCompression(), quadrature::AbstractQuadrature=TrapezoidQuadrature()) where {C<:AbstractCausality}
    causality = C()
    bs, _ = blocksize_and_eltype(map, axis)
    f_masked = masked(map, causality, map(axis[1], axis[1]))
    matrix = compression(axis, f_masked, stationary = map isa Stationary)
    discretization = TrapzDiscretisation(axis, matrix, bs, compression; quadrature=quadrature)
    return Kernel(discretization, causality)
end

"""
    Kernel(::Type{C}, axis, map::Singular; compression=HssCompression()) where {C<:AbstractCausality}

Discretize a singular stationary kernel (see [`Singular`](@ref)) with
product-integration weights: the block-circulant matrix stores the exact
hat-function integrals `W_k / δt` instead of the sampled values
`f(k·δt)`. This is the principal-value-aware discretization of the
Keldysh thermal core; it restores second-order convergence of kernel
products involving the singular branch. The core `f` may be scalar-valued
or matrix-valued (blocksize `> 1`, e.g. multi-level finite-temperature
systems).
"""
function Kernel(::Type{C}, axis, map::Singular;
    compression=HssCompression(), quadrature::AbstractQuadrature=TrapezoidQuadrature()) where {C<:AbstractCausality}
    causality = C()
    bs, _ = blocksize_and_eltype(map, axis)
    dt = step(axis)
    N = length(axis)
    W = singular_weights(map.f, dt; N=N)
    if W isa AbstractVector
        m = zeros(ComplexF64, 1, 1, 2N - 1)
        m[1, 1, :] .= W ./ dt
    else
        m = W ./ dt
    end
    tab = _masked_circulant(causality, m, N)
    matrix = compression(axis, tab)
    return Kernel(TrapzDiscretisation(axis, matrix, bs, compression; quadrature=quadrature), causality)
end

function _masked_circulant(::Acausal, m, N)
    return BlockCirculantMatrix(m)
end
function _masked_circulant(C, m, N)
    mm = copy(m)
    for k in -(N - 1):(N - 1)
        if (C isa Retarded && k < 0) || (C isa Advanced && k > 0)
            mm[:, :, k + N] .= 0
        end
    end
    return BlockCirculantMatrix(mm)
end

function Kernel(::Type{C}, axis, sep::Separable;
    compression=HssCompression(), quadrature::AbstractQuadrature=TrapezoidQuadrature()) where {C<:AbstractCausality}
    causality = C()
    bs, _ = blocksize_and_eltype(sep, axis)
    matrix = triangularLowRankCompression(compression, causality, axis, sep.f, sep.g)
    discretization = TrapzDiscretisation(axis, matrix, bs, compression; quadrature=quadrature)
    return Kernel(discretization, causality)
end

function Kernel{D,C}(axis, matrix, blocksize, compression) where {D<:AbstractDiscretisation, C<:AbstractCausality}
    causality = C()
    discretization = D(axis, matrix, blocksize, compression)
    Kernel(discretization, causality)
end

"""
    RetardedKernel(axis, map::AbstractKernelMap; compression=HssCompression())

Discretize a retarded kernel (zero for `t < t'`) described by `map`
(`Stationary`, `TwoTime` or `Separable`) on `axis`.

Returns a `Kernel` with `Retarded` causality.
"""
RetardedKernel(axis, map::AbstractKernelMap; kwargs...) =
    Kernel(Retarded, axis, map; kwargs...)

RetardedKernel(axis, matrix::AbstractMatrix, blocksize, compression) =
    Kernel{TrapzDiscretisation,Retarded}(axis, matrix, blocksize, compression)

"""
    AdvancedKernel(axis, map::AbstractKernelMap; compression=HssCompression())

Discretize an advanced kernel (zero for `t > t'`) described by `map`
(`Stationary`, `TwoTime` or `Separable`) on `axis`.

Returns a `Kernel` with `Advanced` causality.
"""
AdvancedKernel(axis, map::AbstractKernelMap; kwargs...) =
    Kernel(Advanced, axis, map; kwargs...)

AdvancedKernel(axis, matrix::AbstractMatrix, blocksize, compression) =
    Kernel{TrapzDiscretisation,Advanced}(axis, matrix, blocksize, compression)

"""
    AcausalKernel(axis, map::AbstractKernelMap; compression=HssCompression())

Discretize an acausal kernel (no time-ordering constraint, e.g. an
equilibrium occupation) described by `map` (`Stationary`, `TwoTime` or
`Separable`) on `axis`.

Returns a `Kernel` with `Acausal` causality.
"""
AcausalKernel(axis, map::AbstractKernelMap; kwargs...) =
    Kernel(Acausal, axis, map; kwargs...)

AcausalKernel(axis, matrix::AbstractMatrix, blocksize, compression) =
    Kernel{TrapzDiscretisation,Acausal}(axis, matrix, blocksize, compression)

include("kernel_algebra.jl")
include("kernel_solver.jl")
