module JLArraysExt

using NonEquilibriumGreenFunction: AbstractCompression, ldiv, to_cpu, same_time_blocks, blockrange
import NonEquilibriumGreenFunction
using JLArrays
using JLArrays: JLArray
using LinearAlgebra

"""
    JLArrayCompression() <: AbstractCompression

Dense compression storing kernel matrices as `JLArray` (GPU-style arrays from
JLArrays.jl). This is the `NONCompression` counterpart for users who want a
plain dense representation backed by a GPU-style array backend: no low-rank
compression is performed.

`JLArrays` must be loaded (`using JLArrays`) for this compression to be
available. The linear solve (`ldiv`) and scalar indexing round-trip through
the CPU as JLArrays.jl does not provide a GPU `\`.
"""
struct JLArrayCompression <: AbstractCompression end

function (c::JLArrayCompression)(axis, f; stationary=false)
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

function (c::JLArrayCompression)(m::AbstractMatrix)
    return JLArray(to_cpu(m))
end

function (c::JLArrayCompression)(axis, f, g)
    return c(axis, (t, tp) -> f(t) * g(tp))
end

NonEquilibriumGreenFunction.to_cpu(m::JLArray) = Array(m)

"""
    same_time_blocks(m::JLArray, bs)

Equal-time blocks of a JLArray. Extracted with ranged `getindex` (one ranged
copy per block) to avoid scalar indexing, then moved to the CPU.
"""
function NonEquilibriumGreenFunction.same_time_blocks(m::JLArray, bs)
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
