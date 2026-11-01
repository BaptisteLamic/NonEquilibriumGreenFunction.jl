# HSS reference backend for the backend interface (see
# docs/src/backend_interface_design.md). Adapts HssMatrices.jl to the
# numeric-tier contract and provides an optimized `evaluate` override:
# variadic `op_mul` chains are evaluated in balanced (tree) order instead
# of the default left fold - for HSS this bounds intermediate rank growth
# (log2(k) sequential products for a chain of length k instead of k-1)
# and balances the per-step error, which the default interpreter cannot
# express.

"""
    HssMatricesBackend <: AbstractBackend

Reference CPU HSS backend wrapping HssMatrices.jl. Carries the HSS
compression parameters (`atol`, `rtol`, `kest`, `leafsize`) and the
working scalar type. All primitives stay in the HSS family (dense
`Matrix` inputs are accepted at the boundary only, as the compressed
family's constructor input - there is no dense fallback for operations).
"""
struct HssMatricesBackend{T<:Number} <: AbstractBackend
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
op_sub(A::HssMatrix, B::HssMatrix) = A + op_scale(-1, B)
# promote through a scalar-mul that widens the matrix eltype when the
# scalar is more general (HssMatrices' own `alpha * A` cannot promote:
# Complex{Int} * HssMatrix{Float64} throws)
op_scale(alpha::Number, A::HssMatrix) = _scale_promote(alpha, A)

function _scale_promote(alpha::Number, A::HssMatrix{T}) where {T}
    R = typeof(zero(T) * alpha)
    R == T && return alpha * A
    return alpha * _widen_eltype(R, A)
end

# widen an HssMatrix to eltype R by rebuilding the tree with converted
# coefficient matrices (HssMatrices has no convert across eltypes)
function _widen_eltype(::Type{R}, m::HssMatrix{T}) where {R,T}
    if m.leafnode
        return HssMatrix(convert(Matrix{R}, m.D), convert(Matrix{R}, m.U), convert(Matrix{R}, m.V), m.rootnode)
    end
    return HssMatrix(_widen_eltype(R, m.A11), _widen_eltype(R, m.A22),
        convert(Matrix{R}, m.B12), convert(Matrix{R}, m.B21),
        convert(Matrix{R}, m.R1), convert(Matrix{R}, m.W1),
        convert(Matrix{R}, m.R2), convert(Matrix{R}, m.W2), m.rootnode)
end
op_adjoint(A::HssMatrix) = adjoint(A)
op_norm(A::HssMatrix) = computeMatrixNorm(A)
apply(A::HssMatrix, x::AbstractMatrix) = A * x

function op_mul(A::HssMatrix{TA}, B::HssMatrix{TB}) where {TA,TB}
    # widen to the promoted eltype: HssMatrices cannot multiply across
    # eltypes (e.g. ComplexF64 x Float64)
    R = promote_type(TA, TB)
    Aw = TA == R ? A : _widen_eltype(R, A)
    Bw = TB == R ? B : _widen_eltype(R, B)
    return Aw * Bw
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

function blockdiag_blocks(b::HssMatricesBackend, m::HssMatrix, bs)
    # same semantics as the generic `blockdiag_blocks`:
    # a `bs x bs x N` Array{T,3}, the input of `compress(b, blocks)`
    blocks = same_time_blocks(b, m, bs)
    B = Array{eltype(m),3}(undef, bs, bs, length(blocks))
    for k in eachindex(blocks)
        B[:, :, k] .= blocks[k]
    end
    return B
end

to_cpu(b::HssMatricesBackend, m::HssMatrix) = Matrix(m)
function same_time_blocks(b::HssMatricesBackend, m::HssMatrix, bs)
    T = eltype(m)
    N = div(size(m, 1), bs)
    return [T[m[p + j, p + k] for j in 1:bs, k in 1:bs] for p in (0:bs:(N - 1) * bs)]
end
recompress_inplace!(::HssMatricesBackend, rep::HssRepresentation, tab::HssMatrix) =
    HssMatrices.recompress!(tab, atol=rep.atol, rtol=rep.rtol, leafsize=rep.leafsize)

function compress(b::HssMatricesBackend, axis, f; stationary=false)
    f00 = f(axis[1], axis[1])
    @assert size(f00, 1) == size(f00, 2) "Function f must return square matrices"
    lm = stationary ? build_CirculantlinearMap(axis, f) : build_linearMap(axis, f)
    bs = size(f00, 1)
    cc = bisection_cluster(length(axis) * bs, leafsize=b.leafsize)
    return randcompress_adaptive(lm, cc, cc, atol=b.atol, rtol=b.rtol, kest=b.kest, leafsize=b.leafsize)
end
compress(b::HssMatricesBackend, tab::HssMatrix) = recompress_inplace!(b, HssRepresentation(atol=b.atol, rtol=b.rtol, kest=b.kest, leafsize=b.leafsize), tab)
compress(b::HssMatricesBackend, tab::AbstractMatrix) =
    hss(tab, atol=b.atol, rtol=b.rtol, leafsize=b.leafsize)
function compress(b::HssMatricesBackend, axis, tab::BlockCirculantMatrix)
    lm = build_CirculantlinearMap(tab)
    cc = bisection_cluster(size(tab, 1), leafsize=b.leafsize)
    return randcompress_adaptive(lm, cc, cc, atol=b.atol, rtol=b.rtol, kest=b.kest)
end
function compress(b::HssMatricesBackend, blocks::AbstractArray{T,3}) where {T}
    @assert size(blocks, 1) == size(blocks, 2) "Blocks must be square matrices"
    n = size(blocks, 1) * size(blocks, 3)
    cc = bisection_cluster(n, leafsize=b.leafsize)
    return hss_blkdiag(_blockdiag_sparse(blocks), cc, cc)
end

function solve(b::HssMatricesBackend, left::HssMatrix, right::HssMatrix;
               strategy::AbstractSolveStrategy=DirectSolve())
    supports_strategy(b, strategy) || throw(ArgumentError(
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
    # CSE first: repeated subgraphs (also inside a root MulNode chain)
    # are extracted into bindings and evaluated exactly once, through the
    # balanced path below.
    bindings, root = cse(ex)
    env = Dict{Symbol,Any}()
    for (name, node) in bindings
        env[name] = _eval_hss(b, node, env)
    end
    return _eval_hss(b, root, env)
end

# CSE-aware evaluation with the balanced-chain fast path for MulNodes
# (root and nested); all other nodes use the default in-family
# interpreter semantics with the binding values available.
function _eval_hss(b::HssMatricesBackend, ex::AbstractOpNode, env)
    ex isa CompressedLeaf && return ex.op
    ex isa BindingNode && return env[ex.name]
    if ex isa MulNode
        return _balanced_chain(b, map(a -> _eval_hss(b, a, env), ex.args))
    end
    return _eval_with(b, ex, env)
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

evaluate(b::AbstractBackend, m::HssMatrix) = m

# ---------------------------------------------------------------------------
# Bridge: HssRepresentation evaluates through HssMatricesBackend, so all
# kernel algebra built on HSS matrices gets the optimized (balanced
# chain) DAG evaluation. The working eltype is derived from the
# representation's kernels (task: working_eltype matches scalartype).
# ---------------------------------------------------------------------------

function backend(r::HssRepresentation, ::Type{T}=ComplexF64) where {T<:Number}
    return HssMatricesBackend{T}(r.atol, r.rtol, r.kest, r.leafsize)
end
