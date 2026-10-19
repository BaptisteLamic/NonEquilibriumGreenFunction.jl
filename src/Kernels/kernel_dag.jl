# Kernel-level DAG: physics constructs (lazy), separate from the concrete
# (numeric) DAG of the backend contract. See
# docs/src/backend_interface_design.md and docs/src/backend_contract.md.
#
# The kernel-level DAG encodes the *physics* of a composed operator:
# kernel products, sums, scalings and adjoints, with causality, locality,
# blocksize and eltype propagated at construction time (they must be
# known before evaluation: the decomposition of the *next* product
# depends on the causality of its operands). Concrete matrix manipulation
# lives in the backend DAG (`op_add`, `op_mul`, ...) and is produced by
# the lowering pass `lower` at the forcing points (matrix(), solve_dyson,
# observables). Backends never see physics.
#
# Forcing points (laziness ends here):
#   matrix(op), solve_dyson, same_time/keldysh_trace, compress!(op).

"""Operator whose value is a lazy kernel-level DAG node."""
# Lazy operators behave like SimpleOperators for every accessor
# (axis, step, blocksize, quadrature, compression, matrix, ...): the
# traits are carried by the node; `matrix` is the forcing point.
abstract type AbstractLazyOperator <: SimpleOperator end

"""
    KernelProd{L,R,Q,C} <: AbstractLazyOperator

Lazy kernel product `left * right` (kernel-level DAG node). The product's
traits (causality `C`, locality, blocksize, quadrature `Q`) are computed
at construction; the concrete matrix manipulation is deferred to the
lowering pass (`lower`), which emits backend-DAG nodes (dressing,
quadrature-weighted products, edge corrections) and is triggered at the
forcing points (`matrix`, `solve_dyson`, observables).
"""
mutable struct KernelProd{L<:AbstractOperator,R<:AbstractOperator,Q<:AbstractQuadrature,C<:AbstractCausality} <: AbstractLazyOperator
    left::L
    right::R
    quadrature::Q
    causality::C
    # materialization cache: a DAG is evaluated at most once. Shared
    # child nodes are cached once for every parent tree.
    materialized::Any
end

KernelProd(left::L, right::R, quadrature::Q, causality::C) where {L,R,Q,C} =
    KernelProd{L,R,Q,C}(left, right, quadrature, causality, nothing)

causality(op::KernelProd) = op.causality
locality(op::KernelProd) = locality_of_prod(locality(op.left), locality(op.right))
quadrature(op::KernelProd) = op.quadrature
discretization(op::KernelProd) = discretization(op.left)
axis(op::KernelProd) = axis(discretization(op))
blocksize(op::KernelProd) = blocksize(discretization(op))
scalartype(op::KernelProd) = promote_type(scalartype(op.left), scalartype(op.right))
compression(op::KernelProd) = compression(discretization(op))

"""
    lower(op::KernelProd)

Lowering pass: decompose the kernel product into concrete DAG nodes in
the compression family. Emits exactly the operations the backend
supports: block-diagonal extraction (`blockdiag_blocks`), dressing
(`op_sub` + the block-diagonal constructor), quadrature-weighted
products (`op_mul`/`op_scale`) and edge corrections. No evaluation
happens here; the returned concrete DAG is evaluated by the backend
(default interpreter or its `evaluate` override).
"""
function lower(op::KernelProd)
    if op.left isa LocalKernel || op.right isa LocalKernel
        return op_mul(CompressedLeaf(matrix(op.left)), CompressedLeaf(matrix(op.right)))
    end
    return _lower_prod(causality(op.left), causality(op.right), op.quadrature, op)
end

# Retarded/Advanced pairwise product: biased mul minus the blockwise
# correction (trapezoid dressing on both operands)
function _lower_prod(::C, ::C, q, op) where {C<:Union{Retarded,Advanced}}
    dis_l, dis_r = discretization(op.left), discretization(op.right)
    bs = blocksize(dis_l)
    dl = blockdiag_blocks(matrix(op.left), bs)
    dr = blockdiag_blocks(matrix(op.right), bs)
    cpr = compression(dis_l)
    weighted_L = _lower_dressing(dis_l, dl)
    weighted_R = _lower_dressing(dis_r, dr)
    biased = op_mul(weighted_L, weighted_R)
    corr = CompressedLeaf(_lower_blockwise(dl, dr, cpr))
    return op_scale(step(dis_l), op_sub(biased, corr))
end

# Acausal x Advanced: dress the right operand, edge-correct, single mul
function _lower_prod(::Acausal, ::Advanced, q, op)
    dis_l, dis_r = discretization(op.left), discretization(op.right)
    bs = blocksize(dis_r)
    dr = blockdiag_blocks(matrix(op.right), bs)
    weighted_R = _lower_dressing(dis_r, dr)
    weighted_R = _lower_acausal_advanced_edges(q, dis_r, weighted_R)
    return op_scale(step(dis_l), op_mul(CompressedLeaf(matrix(op.left)), weighted_R))
end

# Acausal x Acausal: quadrature-weighted product. No extra scale node:
# the weights carry the step (edge_weights * dt) inside
# _lower_quadrature_prod.
function _lower_prod(::Acausal, ::Acausal, q, op)
    return _lower_quadrature_prod(q, op)
end

# Retarded x Acausal: dress the left operand, edge-correct, single mul
function _lower_prod(::Retarded, ::Acausal, q, op)
    dis_l, dis_r = discretization(op.left), discretization(op.right)
    bs = blocksize(dis_l)
    dl = blockdiag_blocks(matrix(op.left), bs)
    weighted_L = _lower_dressing(dis_l, dl)
    weighted_L, MR = _lower_retarded_acausal_edges(q, dis_l, weighted_L, matrix(op.right))
    # weighted_L is already a DAG node; MR may be a bare matrix (leaf) or a
    # node after edge correction - normalize to nodes
    return op_scale(step(dis_l), op_mul(_as_node(weighted_L), _as_node(MR)))
end

_as_node(x::AbstractOpNode) = x
_as_node(m) = CompressedLeaf(m)

#  dressing as a concrete node: m - blocks(diag .* 1//2). Returns a DAG
# (op_sub node), not a matrix - the whole product chain stays lazy until
# `materialize` evaluates it once.
function _lower_dressing(dis, d::AbstractArray{T,3}) where {T}
    blocks = d .* T(1 // 2)
    cpr = compression(dis)
    return op_sub(CompressedLeaf(matrix(dis)), CompressedLeaf(cpr(blocks)))
end

# blockwise correction dl*dr/4 as an evaluated block-diagonal (it is a
# leaf constructor, not an algebra op)
_lower_blockwise(dl, dr, cpr) = cpr(_mul_blocks(dl, dr, 1 // 4))

# edge corrections: rectangle = identity, trapezoid = half/zer
# block-diagonal constructors (leaves) wrapping the weighted operand
function _lower_acausal_advanced_edges(::RectangleQuadrature, dis, weighted_R)
    return weighted_R
end
function _lower_acausal_advanced_edges(::TrapezoidQuadrature, dis, weighted_R)
    bs = blocksize(dis)
    half = _boundary_blockdiag(dis, 0.5 * Matrix{scalartype(dis)}(I, bs, bs))
    zer = _boundary_blockdiag(dis, zeros(scalartype(dis), bs, bs))
    return op_mul(CompressedLeaf(half), weighted_R, CompressedLeaf(zer))
end

function _lower_retarded_acausal_edges(::RectangleQuadrature, dis, weighted_L, MR)
    return weighted_L, MR
end
function _lower_retarded_acausal_edges(::TrapezoidQuadrature, dis, weighted_L, MR)
    bs = blocksize(dis)
    half = _boundary_blockdiag(dis, 0.5 * Matrix{scalartype(dis)}(I, bs, bs))
    zer = _boundary_blockdiag(dis, zeros(scalartype(dis), bs, bs))
    return op_mul(CompressedLeaf(zer), weighted_L), op_mul(CompressedLeaf(half), CompressedLeaf(MR))
end

# quadrature-weighted product as concrete nodes
function _lower_quadrature_prod(::RectangleQuadrature, op)
    return op_scale(step(discretization(op.left)),
        op_mul(CompressedLeaf(matrix(op.left)), CompressedLeaf(matrix(op.right))))
end
function _lower_quadrature_prod(::TrapezoidQuadrature, op)
    dis_l = discretization(op.left)
    bs = blocksize(dis_l)
    n = length(axis(dis_l))
    T = scalartype(dis_l)
    MR = matrix(op.right)
    if MR isa StridedMatrix
        w = step(dis_l) .* edge_weights(TrapezoidQuadrature(), n)
        return op_mul(CompressedLeaf(matrix(op.left)), CompressedLeaf(_scale_rowblocks(MR, w, bs, n)))
    end
    blocks = Array{T,3}(undef, bs, bs, n)
    for i in 1:n
        blocks[:, :, i] .= Matrix{T}(I, bs, bs)
    end
    blocks[:, :, 1] .= 0.5 * Matrix{T}(I, bs, bs)
    blocks[:, :, n] .= 0.5 * Matrix{T}(I, bs, bs)
    blocks .*= step(dis_l)
    D = compression(dis_l)(blocks)
    return op_mul(CompressedLeaf(matrix(op.left)), CompressedLeaf(D), CompressedLeaf(MR))
end

# forcing point: materialize a lazy operator to a concrete Kernel

"""
    materialize(op::AbstractLazyOperator)

Forcing point of the kernel-level DAG: lower `op` to a concrete DAG and
evaluate it in the compression family. Returns a concrete `Kernel` with
the lazily computed traits.
"""
function materialize(op::AbstractLazyOperator)
    isdefined(op, :materialized) && op.materialized !== nothing && return op.materialized::Kernel
    cpr = compression(op)
    m = evaluate(cpr, lower(op))
    dis = UniformDiscretisation(axis(op), m, blocksize(op), cpr;
        quadrature=quadrature(op))
    k = Kernel(dis, causality(op))
    op.materialized = k
    return k
end

matrix(op::AbstractLazyOperator) = matrix(materialize(op))

# lazy products compose: a lazy operand keeps the chain lazy (traits
# computed at construction). Local operands keep their concrete fast path.
function *(left::AbstractLazyOperator, right::AbstractLazyOperator)
    @assert quadrature(left) == quadrature(right) "kernel product requires matching quadrature rules"
    return KernelProd(left, right, quadrature(left),
        causality_of_prod(causality(left), causality(right)))
end
function *(left::AbstractLazyOperator, right::Kernel)
    @assert quadrature(left) == quadrature(right) "kernel product requires matching quadrature rules"
    return KernelProd(left, right, quadrature(left),
        causality_of_prod(causality(left), causality(right)))
end
function *(left::Kernel, right::AbstractLazyOperator)
    @assert quadrature(left) == quadrature(right) "kernel product requires matching quadrature rules"
    return KernelProd(left, right, quadrature(left),
        causality_of_prod(causality(left), causality(right)))
end
*(left::AbstractLazyOperator, right::LocalKernel) =
    materialize(KernelProd(left, right, quadrature(left), causality(left)))
*(left::LocalKernel, right::AbstractLazyOperator) =
    materialize(KernelProd(left, right, quadrature(right), causality(right)))

# adjoint of a lazy product: (A*B)' = B' * A' stays lazy
function adjoint(op::KernelProd)
    return KernelProd(adjoint(op.right), adjoint(op.left), op.quadrature, _adjoint_causality(op.causality))
end
_adjoint_causality(::Retarded) = Advanced()
_adjoint_causality(::Advanced) = Retarded()
_adjoint_causality(::Acausal) = Acausal()


# scalar scaling of a lazy product stays lazy: λ·(A*B) = (λA)·B
*(λ::Number, op::AbstractLazyOperator) =
    KernelProd(λ * op.left, op.right, quadrature(op), causality(op))
*(op::AbstractLazyOperator, λ::Number) =
    KernelProd(λ * op.left, op.right, quadrature(op), causality(op))
+(op::AbstractLazyOperator) = op
-(op::AbstractLazyOperator) = (-1) * op


# Composites containing lazy parts materialize through the DAG: the
# sum/difference lowers to op_add/op_sub nodes evaluated in the
# compression family (Kernel-specific +/- already produce concrete
# Kernels; this covers SumOperators with lazy parts).
# lowering is recursive over the whole kernel-level DAG: lazy sums lower
# to op_add over the lowered children (NOT per-part evaluation), so the
# backend evaluates one fused graph for the entire expression. A
# UniformScaling child (e.g. `I + K`) lowers to a scaled identity block
# built from the sibling's geometry.
function lower(op::SumOperator)
    ctx = _sum_context(op)
    return op_add(_sum_child(op.left, ctx), _sum_child(op.right, ctx))
end
function _sum_context(op::SumOperator)
    o = op.left isa SumOperator ? _sum_context(op.left) :
        op.right isa SumOperator ? _sum_context(op.right) :
        op.left isa UniformScaling ? op.right : op.left
    return o
end
# Children materialize through their caches: a node is evaluated at most
# once, and a shared child is reused (as a leaf) by every parent. The
# fusion boundary is therefore the node boundary - each node's own
# lowering is still one fused concrete DAG.
_sum_child(op::AbstractLazyOperator, ctx) = CompressedLeaf(matrix(materialize(op)))
_sum_child(op::SumOperator, ctx) = CompressedLeaf(matrix(materialize(op)))
_sum_child(op::SimpleOperator, ctx) = CompressedLeaf(matrix(op))
function _sum_child(op::UniformScaling, ctx)
    bs = blocksize(ctx)
    N = length(axis(ctx))
    T = scalartype(ctx)
    return CompressedLeaf(op.λ * (Matrix{T}(I, bs, bs) |> m -> begin
        M = zeros(T, bs * N, bs * N)
        for i in 1:N
            M[blockrange(i, bs), blockrange(i, bs)] .= m
        end
        M
    end))
end

function materialize(op::SumOperator)
    op.materialized !== nothing && return op.materialized::Kernel
    cpr = compression(_sum_context(op))
    m = evaluate(cpr, lower(op))
    ctx = _sum_context(op)
    dis = UniformDiscretisation(axis(ctx), m, blocksize(ctx), cpr; quadrature=quadrature(ctx))
    k = Kernel(dis, causality(op))
    op.materialized = k
    return k
end