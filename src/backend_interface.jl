# Backend interface (see docs/src/backend_interface_design.md and
# docs/src/backend_contract.md)
#
# Implemented and consumed: the expression DAG below is the only numeric
# evaluation path of the package. Kernel algebra, solve_dyson, observables,
# scalar/local-kernel operations and discretization adjoint/negation build
# DAGs and evaluate them via the representation's backend; representation
# tags reach the backend contract through backend(r) (see the bridge
# section).

# ---------------------------------------------------------------------------
# Thread budget facility (task-local, inherited by spawned children)
# ---------------------------------------------------------------------------

"""
    current_thread_budget()

Ambient thread budget for the current task. Backends must cap their
parallelism at `min(config.max_threads, current_thread_budget())`.
Defaults to `Threads.nthreads()` when no ambient budget is set.
"""
function current_thread_budget()
    tls = task_local_storage()
    n = haskey(tls, :negf_thread_budget) ? tls[:negf_thread_budget] : 0
    return (n isa Integer && n > 0) ? Int(n) : Threads.nthreads()
end

"""
    with_thread_budget(f, n)

Run `f()` with an ambient thread budget of `n`, inherited by tasks spawned
inside `f` (task-local storage propagates to `Threads.@spawn` children).
Callers running several backend calls concurrently (e.g. Negf.jl-style
`@spawn` fan-outs of kernel products) wrap the fan-out in this to bound
total parallelism.
"""
function with_thread_budget(f::Function, n::Integer)
    tls = task_local_storage()
    had = haskey(tls, :negf_thread_budget)
    old = had ? tls[:negf_thread_budget] : nothing
    task_local_storage(:negf_thread_budget, Int(n))
    try
        return f()
    finally
        if had
            task_local_storage(:negf_thread_budget, old)
        else
            pop!(tls, :negf_thread_budget)
        end
    end
end

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

"""
    DeviceConfig

Device configuration of a backend: `:cpu` or `:gpu` plus the array type
the backend stores. Part of the backend, never of the compressed operator.
"""
struct DeviceConfig{D,A}
    device::D
    arraytype::A
end

"""
    ThreadConfig

Threading configuration of a backend. `max_threads` is the backend's
ceiling; the effective width is `min(max_threads, current_thread_budget())`.
`blas_threads` bounds external thread pools (BLAS and equivalents) used
inside backend kernels. Backends must use task-based parallelism only
(`Threads.@spawn`, never `Threads.@threads`).
"""
struct ThreadConfig
    max_threads::Int
    blas_threads::Int
end

"""
    AbstractBackend

Backend contract (numeric tier). Subtypes own: device, threads, precision
(`sample_type`, `coeff_type`, `working_type`), and the matrix family they
produce. The mandatory primitive set (listed once in `CORE_PRIMITIVES`
below) and the optional `evaluate` override are specified in
`docs/src/backend_interface_design.md` and `docs/src/backend_contract.md`.
There is deliberately no dense fallback: every primitive must be
implemented in-family.
"""
abstract type AbstractBackend end

"""
    working_eltype(b::AbstractBackend)

The scalar type the backend works in: drives `scalartype` and the
`solve_dyson` assembly. Not derived from any stored matrix.
"""
function working_eltype end

"""
    MatrixFamilyBackend{T} <: AbstractBackend

Reference backend for any matrix family built on `AbstractMatrix` with
the standard Base operations (dense `Matrix`, HSS, GPU-style arrays
whose family implements Base ops). It is the default bridge for
representation tags: `DenseRepresentation` and any
`AbstractRepresentation` that has not been ported to a dedicated backend
evaluate through it. The primitives are the generic `AbstractMatrix`
methods defined below, so the family stays in-family (dense+dense stays
dense, HSS+HSS stays HSS); structured backends (`HssMatricesBackend`)
specialize the same generic functions for in-family wins (e.g. balanced
chains).
"""
struct MatrixFamilyBackend{T<:Number} <: AbstractBackend end

MatrixFamilyBackend(::Type{T}=ComplexF64) where {T<:Number} = MatrixFamilyBackend{T}()

working_eltype(::MatrixFamilyBackend{T}) where {T<:Number} = T


# ---------------------------------------------------------------------------
# Compressed operators (opaque; NOT an AbstractMatrix)
# ---------------------------------------------------------------------------

"""
    AbstractCompressedOperator

Opaque compressed representation produced by a backend. Deliberately not
an `AbstractMatrix`: no `getindex`, no Base operator obligations. All
access goes through the package-level primitives in this file.
"""
abstract type AbstractCompressedOperator end

"""
    structure(m::AbstractCompressedOperator)

The structure (tree/rank profile) of `m`, exploitable by
`StructureOptimizingSolve`. Returns `nothing` when the family carries no
exposed structure.
"""
function structure end

"""
    empty_like(st)

A zero-coefficient operator with the same structure as `st`.
"""
function empty_like end

# ---------------------------------------------------------------------------
# Mandatory in-family primitives (numeric tier)
# ---------------------------------------------------------------------------

"""In-family sum of two compressed operators of the same family."""
function op_add end

"""In-family difference of two compressed operators of the same family."""
function op_sub end

"""In-family scalar scaling of a compressed operator."""
function op_scale end

"""
    op_mul(A, B, C...)

Variadic in-family chain product. Semantically the left fold
`op_mul(op_mul(A, B), C)` within tolerance, but the backend may evaluate
in any order / fused form. This is the node through which multi-kernel
chains (`G_R * Σ_K * G_R'`) reach the backend as one object.
"""
function op_mul end

"""In-family adjoint of a compressed operator."""
function op_adjoint end

"""In-family norm of a compressed operator (optional, test-only)."""
function op_norm end

"""
    apply(m, x)

Operator action of `m` on `x` (used for residual closures and lazy
products). In-family; no dense fallback.
"""
function apply end

"""
    compress(b::AbstractBackend, axis, f; stationary=false)
    compress(b::AbstractBackend, m)
    compress(b::AbstractBackend, blocks::AbstractArray{T,3})

Build a matrix of the backend's family. `compress(b, axis, f;
stationary=...)` builds the kernel matrix of `f(t, tp)` (already
causality-masked by the caller; `f` returns fixed-size square blocks);
`compress(b, m)` (re)compresses a matrix into the family (must be pure);
`compress(b, blocks)` builds the block-diagonal matrix with block
`blocks[:, :, k]` on diagonal block `k`. This is the **only** construction
entry point of the numeric tier.
"""
function compress end

"""
    to_cpu(b::AbstractBackend, m)

Materialize a matrix of the backend's family as a CPU `Matrix`.
"""
function to_cpu end

"""
    same_time_blocks(b::AbstractBackend, m, bs)

Equal-time blocks of `m`: one `bs×bs` CPU matrix per time step, i.e.
`same_time_blocks(b, m, bs)[i] ≈ m[blockrange(i, bs), blockrange(i, bs)]`.
"""
function same_time_blocks end

"""
    recompress_inplace!(b::AbstractBackend, r, m)

In-place recompression of `m` in the family of backend `b`, using the
parameters of the representation tag `r`.
"""
function recompress_inplace! end

"""
    blockdiag_blocks(b::AbstractBackend, m, bs)

Equal-time blocks of `m` stacked as a `bs×bs×N` `Array{T,3}`: block `k`
along the third axis is `m[blockrange(k, bs), blockrange(k, bs)]`. This
is the natural input of `compress(b, blocks)`.
"""
function blockdiag_blocks end

# ---------------------------------------------------------------------------
# Solve strategies
# ---------------------------------------------------------------------------

"""Strategy tag for `solve`."""
abstract type AbstractSolveStrategy end

"""
    DirectSolve <: AbstractSolveStrategy

Standard in-family factorization solve (`left \\\\ right` semantics).
"""
struct DirectSolve <: AbstractSolveStrategy end

"""
    StructureOptimizingSolve <: AbstractSolveStrategy

Alternative strategy: the structure (e.g. HSS tree) of `left` is known
and fixed; the coefficients are optimized by the backend until the
residual meets `tol` (at most `maxiter` iterations). The iteration is
entirely backend-owned; the package only supplies `structure(left)`,
`empty_like`, and the residual closure.

**Defined but not yet implemented**: no backend in this package accepts
this strategy yet (every `solve` currently throws an `ArgumentError`
for it). It is part of the stage-1 contract so that backends can start
implementing it without a breaking change.
"""
struct StructureOptimizingSolve <: AbstractSolveStrategy
    maxiter::Int
    tol::Float64
end

"""
    solve(b, left, right; strategy=DirectSolve())

Solve `left * X = right` in the family of backend `b`. `DirectSolve` is
the factorization path; `StructureOptimizingSolve` keeps `structure(left)`
fixed and optimizes coefficients (backend-owned iteration; defined but
not yet implemented by any backend). Backends that do not support a
strategy reject it in the `SolveNode` constructor (fail fast) and in
`solve` itself.
"""
function solve end

"""
    supports_strategy(b::AbstractBackend, strategy::AbstractSolveStrategy) -> Bool

Whether backend `b` accepts `strategy` in `solve`/`SolveNode`. The
default accepts only `DirectSolve`.
"""
supports_strategy(::AbstractBackend, ::DirectSolve) = true
supports_strategy(::AbstractBackend, ::AbstractSolveStrategy) = false

# ---------------------------------------------------------------------------
# Generic matrix-family primitives (reference implementations)
#
# In-family for every AbstractMatrix-based family: operations between two
# members of the family produce a member of the family. These serve the
# MatrixFamilyBackend directly and act as the reference that structured
# backends specialize (e.g. HssMatricesBackend specializes op_mul for its
# balanced-chain evaluator). This is NOT a dense fallback: nothing ever
# converts between families.
# ---------------------------------------------------------------------------

# Core manipulations (backends must implement these for their family):
#   compress, op_add, op_scale, op_mul (binary), op_adjoint, op_norm,
#   apply, solve, blockdiag_blocks, to_cpu, same_time_blocks,
#   recompress_inplace!, working_eltype (see CORE_PRIMITIVES).
# Derived manipulations (default implementations below, expressed via the
# core; backends with a cheaper structured version override them):
#   op_sub, multi-arg op_mul chains.
#
# Eltype homogeneity: operands of the binary primitives must share an
# eltype unless the backend explicitly widens (the HSS backend does, via
# `_widen_eltype`); the generic family relies on Base promotion and
# MethodErrors on families without native promotion.
op_add(A::AbstractMatrix, B::AbstractMatrix) = A + B
op_scale(alpha::Number, A::AbstractMatrix) = alpha * A
op_mul(A::AbstractMatrix, B::AbstractMatrix) = A * B
op_adjoint(A::AbstractMatrix) = adjoint(A)
apply(A::AbstractMatrix, x) = A * x
structure(A::AbstractMatrix) = nothing

# Derived: difference via add/scale
op_sub(A::AbstractMatrix, B::AbstractMatrix) = op_add(A, op_scale(-1, B))

# Derived: Frobenius norm (optional, test-only - `norm(::AbstractOperator)`
# deliberately does not route through it; structured backends override with
# their cheap structured version, e.g. HSS via tr(m'm))
op_norm(A::AbstractMatrix) = norm(A)

# Construction on the generic family: the legacy callables' semantics.
function compress(b::MatrixFamilyBackend, axis, f; stationary=false)
    f00 = f(axis[1], axis[1])
    @assert size(f00, 1) == size(f00, 2) "Function f must return square matrices"
    bs = size(f00, 1)
    r = Array{eltype(f00),2}(undef, bs * length(axis), bs * length(axis))
    for it in eachindex(axis)
        for itp in eachindex(axis)
            r[blockrange(it, bs), blockrange(itp, bs)] .= f(axis[it], axis[itp])
        end
    end
    return r
end
compress(b::MatrixFamilyBackend, m::AbstractMatrix) = Matrix(m)
compress(b::MatrixFamilyBackend, axis, tab::BlockCirculantMatrix) = tab[:, :]
function compress(b::MatrixFamilyBackend, blocks::AbstractArray{T,3}) where {T}
    bs = size(blocks, 1)
    @assert size(blocks, 1) == size(blocks, 2) "Blocks must be square matrices"
    N = size(blocks, 3)
    I = [(t - 1) * bs + i for i in 1:bs, j in 1:bs, t in 1:N]
    J = [(t - 1) * bs + j for i in 1:bs, j in 1:bs, t in 1:N]
    sp = SparseArrays.sparse(I[:], J[:], blocks[:], N * bs, N * bs)
    return compress(b, sp)
end

to_cpu(b::AbstractBackend, m::AbstractMatrix) = Matrix(m)
function same_time_blocks(b::AbstractBackend, m::AbstractMatrix, bs)
    N = div(size(m, 1), bs)
    return [copy(@view m[blockrange(i, bs), blockrange(i, bs)]) for i in 1:N]
end
recompress_inplace!(::AbstractBackend, ::AbstractRepresentation, m) = m
function blockdiag_blocks(b::AbstractBackend, m::AbstractMatrix, bs)
    blocks = same_time_blocks(b, m, bs)
    B = Array{eltype(m),3}(undef, bs, bs, length(blocks))
    for k in eachindex(blocks)
        B[:, :, k] .= blocks[k]
    end
    return B
end

function solve(b::AbstractBackend, left::AbstractMatrix, right::AbstractMatrix;
    strategy::AbstractSolveStrategy=DirectSolve())
    supports_strategy(b, strategy) || throw(ArgumentError(
        "strategy $(typeof(strategy)) not supported by backend $(typeof(b))"))
    return ldiv(left, right)
end

"""
    CORE_PRIMITIVES

The documented core primitive list of the backend contract, as
`(function, arity)` pairs. This tuple is the **source of truth** for the
contract-audit meta-test (`src/backend_interface_tests.jl`), which fails
if a shipped backend lacks a method for any listed primitive; the contract
tables in `docs/src/backend_contract.md` reference the same names.
"""
const CORE_PRIMITIVES = (
    (compress, 2),
    (op_add, 2),
    (op_scale, 2),
    (op_mul, 2),
    (op_adjoint, 1),
    (op_norm, 1),
    (apply, 2),
    (solve, 3),
    (blockdiag_blocks, 2),
    (to_cpu, 1),
    (same_time_blocks, 2),
    (recompress_inplace!, 2),
    (working_eltype, 1),
)

# ---------------------------------------------------------------------------
# Expression DAG (numeric tier). The semantic tier (quadrature, dressing,
# causality support) lives one level up in the lazy kernel-level DAG
# (src/Kernels/kernel_dag.jl); its lowering pass emits the nodes below and
# stays package-owned.
# ---------------------------------------------------------------------------

"""
    AbstractOpNode

Numeric DAG node. Flat and acyclic over compressed leaves.
"""
abstract type AbstractOpNode end

"""Leaf: an already-compressed operator of the backend's family."""
struct CompressedLeaf{O} <: AbstractOpNode
    op::O
end

struct AddNode <: AbstractOpNode
    args::Tuple{Vararg{AbstractOpNode}}
end

struct ScaleNode <: AbstractOpNode
    alpha::Any
    arg::AbstractOpNode
end

"""Variadic chain-product node (see `op_mul`)."""
struct MulNode <: AbstractOpNode
    args::Tuple{Vararg{AbstractOpNode}}
end

struct AdjointNode <: AbstractOpNode
    arg::AbstractOpNode
end

"""
    SolveNode <: AbstractOpNode

Solve node of the numeric DAG: `solve(backend, left, right; strategy)`
as a graph node, evaluated by the backend's interpreter. Construction
fails fast with an `ArgumentError` when the backend does not support the
strategy.
"""
struct SolveNode{B,L,R,S} <: AbstractOpNode
    backend::B
    left::L
    right::R
    strategy::S
end

function SolveNode(backend::AbstractBackend, left::AbstractOpNode, right::AbstractOpNode,
    strategy::AbstractSolveStrategy=DirectSolve())
    supports_strategy(backend, strategy) || throw(ArgumentError(
        "strategy $(typeof(strategy)) not supported by backend $(typeof(backend))"))
    return SolveNode{typeof(backend),typeof(left),typeof(right),typeof(strategy)}(
        backend, left, right, strategy)
end

# lazy constructors - no work happens here
op_add(args::AbstractOpNode...) = AddNode(args)
op_sub(a::AbstractOpNode, b::AbstractOpNode) = AddNode((a, ScaleNode(-1, b)))
op_scale(alpha, arg::AbstractOpNode) = ScaleNode(alpha, arg)
op_mul(args::AbstractOpNode...) = MulNode(args)
op_adjoint(arg::AbstractOpNode) = AdjointNode(arg)

"""
    evaluate(b, ex::AbstractOpNode)

Evaluate a numeric DAG with backend `b`. The default implementation is the
in-family interpreter: it reduces the graph with the mandatory binary
primitives (left fold for `op_mul`), which makes it correct for every
backend that implements the primitive set. Backends may override for
fusion, reassociation and joint coefficient optimization.
"""
# (default evaluate moved to the CSE section at the end of this file)
evaluate(b::AbstractBackend, m::AbstractCompressedOperator) = m

# ---------------------------------------------------------------------------
# Bridge: every AbstractRepresentation evaluates through a backend
# (MatrixFamilyBackend by default). This is the single evaluation entry
# point of the package: kernel algebra, solver and observables all build
# DAGs and call evaluate.
# ---------------------------------------------------------------------------

"""
    backend(r::AbstractRepresentation, T::Type=ComplexF64)

The backend a representation evaluates through. Defaults to
`MatrixFamilyBackend{T}` (the generic AbstractMatrix family reference
backend, working in `T`); structured representations override this to
return their dedicated backend. `T` is the working eltype of the kernels
built with `r` (task: `working_eltype(backend(r, T)) == T`).
"""
backend(::AbstractRepresentation, ::Type{T}=ComplexF64) where {T<:Number} = MatrixFamilyBackend{T}()

"""
    evaluate(r::AbstractRepresentation, ex::AbstractOpNode; eltype=nothing)

Evaluate a DAG with the backend of a representation. This is how
representation tags (dense, HSS, JLArrays, ...) reach the DAG world
without changes at call sites. `eltype` optionally pins the working
eltype of the bridged backend.
"""
function evaluate(r::AbstractRepresentation, ex::AbstractOpNode; eltype=nothing)
    b = eltype === nothing ? backend(r) : backend(r, eltype)
    return evaluate(b, ex)
end

"""
    evaluate(r::AbstractRepresentation, m::AbstractMatrix)

Single-leaf convenience: evaluate an operator in the representation's
family.
"""
evaluate(r::AbstractRepresentation, m::AbstractMatrix) = m

# ---------------------------------------------------------------------------
# Common-subexpression elimination (CSE) on lowered concrete DAGs
#
# The kernel-level DAG already caches materialization per node (a DAG is
# never evaluated twice), but a single lowered graph can still contain
# the same concrete subgraph at several positions - e.g. `lead_current`
# uses `Σ_R * G_R` both directly and inside an adjoint, and a chain with
# a repeated factor repeats the shared prefix. `cse` extracts every
# subgraph that occurs more than once into named bindings, evaluated
# exactly once by `evaluate_cse`.
# ---------------------------------------------------------------------------

"""DAG leaf referring to a named binding produced by `cse`."""
struct BindingNode <: AbstractOpNode
    name::Symbol
end

"""
    cse(ex::AbstractOpNode) -> (bindings, ex′)

Common-subexpression elimination on a lowered concrete DAG. Returns the
ordered `bindings` (`Vector{Pair{Symbol,AbstractOpNode}}`, topologically
sorted: a binding's body only refers to leaves and earlier bindings) and
the rewritten root `ex′` in which every repeated subgraph is replaced by
a `BindingNode` leaf. Each binding is evaluated exactly once per
`evaluate_cse` call - the repeated subgraph is computed once, not once
per occurrence.

Two subgraphs are common iff they are structurally equal over identical
leaf objects (leaf identity by `===` on the stored operators). Leaves
are never extracted: they are cheap references.

Hash collisions cannot merge distinct subgraphs: nodes are bucketed by
`_cse_key` only as a pre-filter, and bucket members are compared by
`===` (the exact structural equality `_cse_key` approximates) before
being counted as the same subgraph.
"""
function cse(ex::AbstractOpNode)
    buckets = Dict{UInt,Vector{AbstractOpNode}}()
    counts = IdDict{AbstractOpNode,Int}()
    _cse_count(ex, buckets, counts)
    names = IdDict{AbstractOpNode,Symbol}()
    bindings = Pair{Symbol,AbstractOpNode}[]
    root = _cse_build(ex, buckets, counts, names, bindings)
    return bindings, root
end

# structural key: identical operations over identical leaves. Scalars
# are keyed by value (hash), matrices by object identity.
_cse_key(ex::CompressedLeaf) = objectid(ex.op)
_cse_key(ex::BindingNode) = hash(:binding, ex.name)
function _cse_key(ex::AddNode)
    h = hash(:add)
    for a in ex.args
        h = hash(h, _cse_key(a))
    end
    return h
end
function _cse_key(ex::ScaleNode)
    return hash(:scale, hash(ex.alpha, _cse_key(ex.arg)))
end
function _cse_key(ex::MulNode)
    h = hash(:mul)
    for a in ex.args
        h = hash(h, _cse_key(a))
    end
    return h
end
_cse_key(ex::AdjointNode) = hash(:adj, _cse_key(ex.arg))
_cse_key(ex::SolveNode) = hash(:solve, hash(_cse_key(ex.left), _cse_key(ex.right)))

# representative of `ex`'s equivalence class among the already-seen
# nodes, or `nothing` if none is registered yet. `_cse_key` buckets;
# `===` confirms, so hash collisions can never merge distinct subgraphs.
function _cse_rep(ex::AbstractOpNode, buckets)
    v = get(buckets, _cse_key(ex), nothing)
    v === nothing && return nothing
    for r in v
        r === ex && return r
    end
    return nothing
end

function _cse_count(ex::AbstractOpNode, buckets, counts)
    rep = _cse_rep(ex, buckets)
    if rep === nothing
        push!(get!(buckets, _cse_key(ex), AbstractOpNode[]), ex)
        counts[ex] = 1
    else
        counts[rep] += 1
    end
    for c in _cse_children(ex)
        _cse_count(c, buckets, counts)
    end
    return nothing
end

_cse_children(::CompressedLeaf) = ()
_cse_children(::BindingNode) = ()
_cse_children(ex::AddNode) = ex.args
_cse_children(ex::ScaleNode) = (ex.arg,)
_cse_children(ex::MulNode) = ex.args
_cse_children(ex::AdjointNode) = (ex.arg,)
_cse_children(ex::SolveNode) = (ex.left, ex.right)

# bottom-up rebuild: children are bound before the parent, so a binding
# body only refers to leaves and earlier bindings (topological order)
function _cse_build(ex::CompressedLeaf, buckets, counts, names, bindings)
    return ex
end
function _cse_build(ex::AbstractOpNode, buckets, counts, names, bindings)
    rep = _cse_rep(ex, buckets)
    if rep !== nothing && counts[rep] >= 2
        # repeated subgraph: bind once, reference everywhere
        if !haskey(names, rep)
            new_children = map(c -> _cse_build(c, buckets, counts, names, bindings), _cse_children(ex))
            body = _rebuild_like(ex, new_children)
            name = Symbol(:cse_, length(bindings) + 1)
            push!(bindings, name => body)
            names[rep] = name
        end
        return BindingNode(names[rep])
    end
    new_children = map(c -> _cse_build(c, buckets, counts, names, bindings), _cse_children(ex))
    return _rebuild_like(ex, new_children)
end

function _rebuild_like(ex::AddNode, children)
    return AddNode(Tuple(children))
end
function _rebuild_like(ex::ScaleNode, children)
    return ScaleNode(ex.alpha, children[1])
end
function _rebuild_like(ex::MulNode, children)
    return MulNode(Tuple(children))
end
function _rebuild_like(ex::AdjointNode, children)
    return AdjointNode(children[1])
end
function _rebuild_like(ex::SolveNode, children)
    return SolveNode(ex.backend, children[1], children[2], ex.strategy)
end

"""
    evaluate_cse(b, bindings, ex)

Evaluate a CSE-lowered graph: each binding is evaluated exactly once, in
topological order, then the root `ex` is evaluated with the binding
values available as leaves. Backends with a fused `evaluate` may instead
schedule the bindings themselves (e.g. batching or device-placing them);
this function is the reference semantics.
"""
function evaluate_cse(b::AbstractBackend, bindings, ex)
    env = Dict{Symbol,Any}()
    for (name, node) in bindings
        env[name] = _eval_with(b, node, env)
    end
    return _eval_with(b, ex, env)
end

function _eval_with(b::AbstractBackend, ex::AbstractOpNode, env)
    ex isa CompressedLeaf && return ex.op
    ex isa BindingNode && return env[ex.name]
    ex isa AddNode && return foldl(op_add, map(a -> _eval_with(b, a, env), ex.args))
    ex isa ScaleNode && return op_scale(ex.alpha, _eval_with(b, ex.arg, env))
    ex isa AdjointNode && return op_adjoint(_eval_with(b, ex.arg, env))
    ex isa MulNode && return foldl(op_mul, map(a -> _eval_with(b, a, env), ex.args))
    ex isa SolveNode && return solve(b, _eval_with(b, ex.left, env), _eval_with(b, ex.right, env); strategy=ex.strategy)
    throw(ArgumentError("unsupported node type: $(typeof(ex))"))
end

"""
    evaluate(b::AbstractBackend, ex::AbstractOpNode)

Evaluate a numeric DAG with backend `b`. The default implementation is
the in-family interpreter; it first applies CSE so repeated subgraphs
are computed once. Backends may override for fusion/reassociation (and
typically want to run `cse` first as well).
"""
function evaluate(b::AbstractBackend, ex::AbstractOpNode)
    bindings, root = cse(ex)
    return evaluate_cse(b, bindings, root)
end
