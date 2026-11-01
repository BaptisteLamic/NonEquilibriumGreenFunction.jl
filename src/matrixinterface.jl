# Matrix representation interface.
#
# This file defines the representation tag layer the rest of the package
# programs against, so that new matrix representations (dense CPU, HSS,
# GPU-style arrays, ...) can be plugged in without touching the kernel
# algebra or the solver. See docs/src/backend_contract.md.
#
# A representation tag is a pure parameter holder: the type describes the
# matrix family the kernels are stored in (dense CPU, HSS, GPU-style
# arrays, ...) and carries the construction parameters (e.g. HSS
# tolerances). The matrices themselves are built by the backend primitive
# `compress(backend(r), ...)` (see backend_interface.jl) - the word
# "compression" survives only there, where compression is literally what
# happens.
#
# The matrix interface (a family produced by a backend):
# - `size`, `eltype`, `getindex` (scalar or ranges; families forbidding scalar
#   indexing must extend the `same_time_blocks` primitive),
# - `+`, `-`, scalar `*`, `adjoint`, matrix-matrix `*`,
# - `LinearAlgebra.norm`,
# - `ldiv(left, right)` (solve; fallback `left \\ right`),
# - `to_cpu(m)` (fallback `Matrix(m)`).

"""
    AbstractRepresentation

Abstract type for matrix representation tags. A subtype is a pure
parameter holder describing the family the kernel matrices are stored in
and the construction parameters (e.g. HSS tolerances); construction goes
through the backend primitive `compress(backend(r), ...)`. See
`src/matrixinterface.jl` and the "Backend contract" documentation page
for the required interface.
"""
abstract type AbstractRepresentation end

"""
    ldiv(left, right)

Solve `left * X = right` for the matrix families used by representations.
Generic fallback: `left \\ right`. `HssMatrix` uses the in-place HSS
solver; GPU-style families typically round-trip through the CPU.
"""
ldiv(left, right) = left \ right

"""
    to_cpu(m)

Materialize a matrix of any family as a CPU `Matrix`.
Fallback: `Matrix(m)`.
"""
to_cpu(m::AbstractMatrix) = Matrix(m)

"""
    same_time_blocks(m, bs)

Equal-time blocks of a matrix: one `bs×bs` CPU matrix per time step, i.e.
`same_time_blocks(m, bs)[i] ≈ m[blockrange(i, bs), blockrange(i, bs)]`.
Generic fallback extracts the blocks with ranged `getindex`.
"""
function same_time_blocks(m::AbstractMatrix, bs)
    N = div(size(m, 1), bs)
    return [copy(@view m[blockrange(i, bs), blockrange(i, bs)]) for i in 1:N]
end

"""
    recompress_inplace!(r::AbstractRepresentation, m)

In-place recompression of the matrix `m` with the representation `r`.
The default is a no-op returning `m`; representations that support
in-place recompression (e.g. `HssRepresentation` on an `HssMatrix`)
should extend it.
"""
recompress_inplace!(::AbstractRepresentation, m) = m
