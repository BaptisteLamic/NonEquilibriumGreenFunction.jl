module JLArraysExt

using NonEquilibriumGreenFunction: AbstractRepresentation, AbstractBackend, ldiv, to_cpu, same_time_blocks,
    blockrange
import NonEquilibriumGreenFunction: working_eltype, backend, compress, to_cpu, same_time_blocks
import NonEquilibriumGreenFunction
using JLArrays
using JLArrays: JLArray
using LinearAlgebra

"""
    JLArrayRepresentation() <: AbstractRepresentation

Dense representation storing kernel matrices as `JLArray` (GPU-style arrays
from JLArrays.jl). This is the `DenseRepresentation` counterpart for users
who want a plain dense representation backed by a GPU-style array backend:
no compression is performed, so a dense GPU-array family is accepted
anywhere a representation is.

`JLArrays` must be loaded (`using JLArrays`) for this representation to be
available. The linear solve (`ldiv`) and scalar indexing round-trip through
the CPU as JLArrays.jl does not provide a GPU `\\`.
"""
struct JLArrayRepresentation <: AbstractRepresentation end

"""
    JLArrayBackend{T} <: AbstractBackend

Backend of the `JLArray` dense family: the generic matrix-family
primitives (which stay in-family for `JLArray`) with construction,
materialization and block extraction specialized to `JLArray`.
"""
struct JLArrayBackend{T<:Number} <: AbstractBackend end

working_eltype(::JLArrayBackend{T}) where {T} = T

backend(::JLArrayRepresentation, ::Type{T}=ComplexF64) where {T<:Number} = JLArrayBackend{T}()

function compress(b::JLArrayBackend, axis, f; stationary=false)
    f00 = f(axis[1], axis[1])
    @assert size(f00, 1) == size(f00, 2) "Function f must return square matrices"
    bs = size(f00, 1)
    r = Matrix{eltype(f00)}(undef, bs * length(axis), bs * length(axis))
    for it in eachindex(axis)
        for itp in eachindex(axis)
            r[NonEquilibriumGreenFunction.blockrange(it, bs), NonEquilibriumGreenFunction.blockrange(itp, bs)] .= f(axis[it], axis[itp])
        end
    end
    return JLArray(r)
end

compress(::JLArrayBackend, m::AbstractMatrix) = JLArray(Array(m))

function compress(::JLArrayBackend, blocks::AbstractArray{T,3}) where {T}
    bs = size(blocks, 1)
    @assert size(blocks, 1) == size(blocks, 2) "Blocks must be square matrices"
    N = size(blocks, 3)
    r = zeros(T, N * bs, N * bs)
    for k in 1:N
        r[NonEquilibriumGreenFunction.blockrange(k, bs), NonEquilibriumGreenFunction.blockrange(k, bs)] .= blocks[:, :, k]
    end
    return JLArray(r)
end

NonEquilibriumGreenFunction.to_cpu(b::AbstractBackend, m::JLArray) = Array(m)

"""
    same_time_blocks(b, m::JLArray, bs)

Equal-time blocks of a JLArray. Extracted with ranged `getindex` (one ranged
copy per block) to avoid scalar indexing, then moved to the CPU.
"""
function NonEquilibriumGreenFunction.same_time_blocks(b::AbstractBackend, m::JLArray, bs)
    N = div(size(m, 1), bs)
    return [Array(m[blockrange(i, bs), blockrange(i, bs)]) for i in 1:N]
end

function NonEquilibriumGreenFunction.ldiv(left::JLArray, right::JLArray)
    return JLArray(Array(left) \ Array(right))
end

function NonEquilibriumGreenFunction.ldiv(left::AbstractMatrix, right::JLArray)
    return ldiv(left, Array(right)) |> JLArray
end

end # module
