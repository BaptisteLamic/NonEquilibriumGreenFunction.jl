# Matrix and compression interface.
#
# This file defines the small interface the rest of the package programs
# against, so that new matrix representations (dense CPU, HSS, GPU-style
# arrays, ...) and new compression methods can be added without touching the
# kernel algebra or the solver. See docs/src/custom_compression.md.
#
# The compression interface (a callable subtype of `AbstractCompression`):
# - `c(axis, f; stationary=false)` builds the kernel matrix of `f(t, tp)`
#   (already causality-masked); `f` returns fixed-size square blocks.
# - `c(m::AbstractMatrix)` (re)compresses a matrix (dense or sparse) into the
#   family; must be pure.
# - optionally `c(axis, f, g)` for separable kernels (optimization hook).
# - optionally `c(blocks::AbstractArray{T,3})` for block-diagonal matrices
#   (optimization hook; default: compress the sparse form).
# - `Base.:(==)` between instances (free for plain immutables).
# - `recompress_inplace!(c, m)` for in-place recompression (default: no-op).
#
# The matrix interface (a family produced by a compression):
# - `size`, `eltype`, `getindex` (scalar or ranges; families forbidding scalar
#   indexing must extend `same_time_blocks`),
# - `+`, `-`, scalar `*`, `adjoint`, matrix-matrix `*`,
# - `LinearAlgebra.norm`,
# - `ldiv(left, right)` (solve; fallback `left \ right`),
# - `to_cpu(m)` (fallback `Matrix(m)`).

"""
    AbstractCompression

Abstract type for compression methods. See `src/matrixinterface.jl` (or the
"Custom compression" documentation page) for the required interface.
"""
abstract type AbstractCompression end

"""
    ldiv(left, right)

Solve `left * X = right` for the matrix families used by compression
methods. Generic fallback: `left \\ right`. `HssMatrix` uses the in-place HSS
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
    recompress_inplace!(c::AbstractCompression, m)

In-place recompression of the matrix `m` with the compression `c`.
The default is a no-op returning `m`; compressions that support in-place
recompression (e.g. `HssCompression` on an `HssMatrix`) should extend it.
"""
recompress_inplace!(::AbstractCompression, m) = m

"""
    (c::AbstractCompression)(blocks::AbstractArray{T,3})

Build the block-diagonal matrix with the `bs×bs` block `blocks[:, :, k]` on
diagonal block `k`, in the family of the compression `c`. Generic fallback:
build the sparse form and compress it. Compressions with an exact native
block-diagonal representation (e.g. `HssCompression` via `hss_blkdiag`)
should specialize this to avoid the lossy recompression round-trip.
"""
function (c::AbstractCompression)(blocks::AbstractArray{T,3}) where {T}
    bs = size(blocks, 1)
    @assert size(blocks, 1) == size(blocks, 2) "Blocks must be square matrices"
    N = size(blocks, 3)
    I = [(t - 1) * bs + i for i in 1:bs, j in 1:bs, t in 1:N]
    J = [(t - 1) * bs + j for i in 1:bs, j in 1:bs, t in 1:N]
    sp = SparseArrays.sparse(I[:], J[:], blocks[:], N * bs, N * bs)
    return c(sp)
end
