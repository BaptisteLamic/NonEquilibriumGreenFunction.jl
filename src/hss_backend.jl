# HSS reference backend for the backend interface (see
# docs/src/backend_interface_design.md). Adapts HssMatrices.jl to the
# numeric-tier contract and provides an optimized `evaluate` override:
# variadic `op_mul` chains are evaluated in balanced (tree) order instead
# of the default left fold - for HSS this bounds intermediate rank growth
# (log2(k) sequential products for a chain of length k instead of k-1)
# and balances the per-step error, which the default interpreter cannot
# express.

"""
    HssMatricesBackend <: AbstractCompressionBackend

Reference CPU HSS backend wrapping HssMatrices.jl. Carries the HSS
compression parameters (`atol`, `rtol`, `kest`, `leafsize`) and the
working scalar type. All primitives stay in the HSS family (dense
`Matrix` inputs are accepted at the boundary only, as the compressed
family's constructor input - there is no dense fallback for operations).
"""
struct HssMatricesBackend{T<:Number} <: AbstractCompressionBackend
    atol::Float64
    rtol::Float64
    kest::Int
    leafsize::Int
end

function HssMatricesBackend(; atol=1e-4, rtol=1e-4, kest=20, leafsize=32,
                            working_type::Type=ComplexF64)
    return HssMatricesBackend{working_type}(atol, rtol, kest, leafsize)
end

working_eltype(::HssMatricesBackend{T}) where {T} = T

# ---------------------------------------------------------------------------
# Mandatory primitives (all in the HSS family)
# ---------------------------------------------------------------------------

op_add(A::HssMatrix, B::HssMatrix) = A + B
op_sub(A::HssMatrix, B::HssMatrix) = A + (-1) * B
op_scale(alpha::Number, A::HssMatrix) = alpha * A
op_adjoint(A::HssMatrix) = adjoint(A)
op_norm(A::HssMatrix) = computeMatrixNorm(A)
apply(A::HssMatrix, x::AbstractMatrix) = A * x

function op_mul(A::HssMatrix, B::HssMatrix)
    return A * B
end

"""
    structure(m::HssMatrix)

The structure of an HSS matrix: its row/column cluster trees and the
rank profile per level. Exploited by `StructureOptimizingSolve` (tree
fixed, coefficients optimized) and by the balanced-chain evaluator.
"""
function structure(m::HssMatrix)
    return m
end

"""
    empty_like(m::HssMatrix)

Zero-coefficient HSS matrix with the same tree structure as `m`: leaf
diagonal blocks and coupling blocks are zeroed; the tree shape is kept
verbatim.
"""
function empty_like(m::HssMatrix)
    return _zero_like(m)
end

function _zero_like(m::HssMatrix)
    if m.leafnode
        # keep the leaf's basis structure (U/V); zero the diagonal block
        return HssMatrix(zeros(eltype(m), size(m.D)), m.U, m.V, m.rootnode)
    end
    # keep the full generator structure (R/W bases); zero the couplings
    return HssMatrix(_zero_like(m.A11), _zero_like(m.A22),
        zeros(eltype(m), size(m.B12)), zeros(eltype(m), size(m.B21)),
        m.R1, m.W1, m.R2, m.W2, m.rootnode)
end

function blockdiag_blocks(m::HssMatrix, bs)
    # same semantics as the generic `blockdiag_blocks` (utils.jl):
    # a `bs x bs x N` Array{T,3}, the input of `c(blocks)`
    blocks = same_time_blocks(m, bs)
    B = Array{eltype(m),3}(undef, bs, bs, length(blocks))
    for k in eachindex(blocks)
        B[:, :, k] .= blocks[k]
    end
    return B
end

function solve(b::HssMatricesBackend, left::HssMatrix, right::HssMatrix;
               strategy::AbstractSolveStrategy=DirectSolve())
    strategy isa DirectSolve || throw(ArgumentError(
        "strategy $(typeof(strategy)) not supported by HssMatricesBackend"))
    return ldiv!(left, right)
end

# ---------------------------------------------------------------------------
# Optimized DAG evaluation
# ---------------------------------------------------------------------------

"""
    evaluate(b::HssMatricesBackend, ex::AbstractOpNode)

Optimized whole-graph evaluation for the HSS family:

- Variadic `op_mul` chains are evaluated in **balanced (tree) order**:
  a chain `A*B*C*D` is evaluated as `(A*B)*(C*D)` rather than the default
  left fold. For HSS this halves the depth of sequential products
  (`log2(k)` vs `k-1` steps), which bounds intermediate rank growth and
  accumulates less compression error per step.
- All other nodes fall back to the default in-family interpreter, which
  is exact for the HSS family (every primitive is in-family).
"""
function evaluate(b::HssMatricesBackend, ex::AbstractOpNode)
    if ex isa MulNode
        return _balanced_chain(b, map(a -> evaluate(b, a), ex.args))
    end
    return invoke(evaluate, Tuple{AbstractCompressionBackend, AbstractOpNode}, b, ex)
end

function _balanced_chain(b::HssMatricesBackend, ops)
    n = length(ops)
    n == 1 && return ops[1]
    mid = (n + 1) ÷ 2
    left = _balanced_chain(b, ops[1:mid])
    right = _balanced_chain(b, ops[mid+1:end])
    return op_mul(left, right)
end

# fast path: already-compressed operators
function evaluate(b::HssMatricesBackend, m::HssMatrix)
    return m
end

# ---------------------------------------------------------------------------
# Default interpreter compatibility: HSS matrices are valid leaves
# ---------------------------------------------------------------------------

evaluate(b::AbstractCompressionBackend, m::HssMatrix) = m
