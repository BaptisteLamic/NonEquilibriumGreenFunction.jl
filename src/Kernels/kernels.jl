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

function discretize_kernel(::Type{D},::Type{C},axis, f; compression=HssCompression(), stationary=false) where {D<:AbstractDiscretisation, C<:AbstractCausality}
    causality = C()
    f00 = f(axis[1],axis[1])
    @assert size(f00,1) == size(f00,2)
    bs = size(f00,1)
    _mask(::Retarded) = (x, y) -> x >= y ? f(x, y) : zero(f00)
    _mask(::Advanced) = (x, y) -> x <= y ? f(x, y) : zero(f00)
    _mask(::Acausal) = (x, y) -> f(x, y) 
    f_masked = _mask(causality)
    matrix = compression(axis, f_masked, stationary=stationary)
    discretization = D(axis, matrix, bs, compression)
    return Kernel(discretization, causality)
end

"""
    discretize_lowrank_kernel(D, C, axis, f, g; compression=HssCompression())

Discretize the separable kernel `f(t) * g(tp)` masked by the causality `C`
(`Retarded`, `Advanced` or `Acausal`), using the discretization type `D`.

Returns a `Kernel` with causality `C`.
"""
function discretize_lowrank_kernel(::Type{D},::Type{C}, axis, f,g ;compression=HssCompression())  where {D<:AbstractDiscretisation, C<:AbstractCausality}
    f00 = f(axis[1])
    g00 = g(axis[1])
    @assert size(f00,1) == size(f00,2)
    @assert size(f00) == size(g00)
    bs = size(f00,1)
    matrix = triangularLowRankCompression(compression,C(), axis, f, g)
    discretization = TrapzDiscretisation(axis, matrix, bs, compression)
    return Kernel(discretization, C())
end

function Kernel{D,C}(axis, matrix, blocksize, compression) where {D<:AbstractDiscretisation, C<:AbstractCausality}
    causality = C()
    discretization = D(axis, matrix, blocksize, compression)
    Kernel(discretization, causality)
end

"""
    discretize_retardedkernel(axis, f; compression=HssCompression(), stationary=false)

Discretize a retarded kernel `f(t, t')` (zero for `t < t'`) on `axis`.

The block size is inferred from `size(f(axis[1], axis[1]))`. With `stationary=true`
the kernel is assumed to depend only on `t - t'` and a circulant structure is used.

Returns a `Kernel` with `Retarded` causality.
"""
function discretize_retardedkernel(axis, f; compression=HssCompression(), stationary=false)
    discretize_kernel(TrapzDiscretisation,Retarded,
        axis, f;
        compression=compression, stationary=stationary
        )
end
function RetardedKernel(axis, matrix, blocksize, compression)
    Kernel{TrapzDiscretisation,Retarded}(axis, matrix, blocksize, compression)
end
"""
    discretize_advancedkernel(axis, f; compression=HssCompression(), stationary=false)

Discretize an advanced kernel `f(t, t')` (zero for `t > t'`) on `axis`.

Returns a `Kernel` with `Advanced` causality.
"""
function discretize_advancedkernel(axis, f; compression=HssCompression(), stationary=false)
    discretize_kernel(TrapzDiscretisation,Advanced,
        axis, f;
        compression=compression, stationary=stationary
        )
end
function AdvancedKernel(axis, matrix, blocksize, compression)
    Kernel{TrapzDiscretisation,Advanced}(axis, matrix, blocksize, compression)
end
"""
    discretize_acausalkernel(axis, f; compression=HssCompression(), stationary=false)

Discretize an acausal kernel `f(t, t')` on `axis`, e.g. an equilibrium occupation.

Returns a `Kernel` with `Acausal` causality.
"""
function discretize_acausalkernel(axis, f; compression=HssCompression(), stationary=false)
    discretize_kernel(TrapzDiscretisation,Acausal,
        axis, f;
        compression=compression, stationary=stationary
        )
end

function AcausalKernel(axis, matrix, blocksize, compression)
    Kernel{TrapzDiscretisation,Acausal}(axis, matrix, blocksize, compression)
end


include("kernel_algebra.jl")
include("kernel_solver.jl")
