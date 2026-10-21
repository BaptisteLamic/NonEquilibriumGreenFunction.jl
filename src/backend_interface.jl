# Backend interface (see docs/src/backend_interface_design.md and
# docs/src/backend_contract.md)
#
# Implemented and consumed: the expression DAG below is the only numeric
# evaluation path of the package. Kernel algebra, solve_dyson, observables,
# scalar/local-kernel operations and discretization adjoint/negation build
# DAGs and evaluate them via the compression's backend; legacy
# AbstractCompression objects are bridged through backend(c) (see the
# legacy bridge section).

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
    AbstractCompressionBackend

Backend contract (numeric tier). Subtypes own: device, threads, precision
(`sample_type`, `coeff_type`, `working_type`), and the compressed-operator
family they produce. The mandatory primitive set and the optional
`evaluate` override are specified in
`docs/src/backend_interface_design.md`. There is deliberately no dense
fallback: every primitive must be implemented in-family.
"""
abstract type AbstractCompressionBackend end

"""
    working_eltype(b::AbstractCompressionBackend)

The scalar type the backend works in: drives `scalartype` and the
`solve_dyson` assembly. Not derived from any stored matrix.

**Advisory in stage 1**: the generic `AbstractMatrix` primitives never
consult it (eltype promotion happens naturally through Base), and the
legacy bridge's `backend(c)` defaults to `ComplexF64` regardless of the
stored matrices' eltype. Backends that actually work in a different
precision (mixed-precision backends) must consume it; until then it is
carried but inert.
"""
function working_eltype end

"""
    MatrixFamilyBackend{T} <: AbstractCompressionBackend

Reference backend for any matrix family built on `AbstractMatrix` with
the standard Base operations (dense `Matrix`, HSS, GPU-style arrays
whose family implements Base ops). It is the bridge for the legacy
compressions: `NONCompression` and any `AbstractCompression` that has
not been ported to a dedicated backend evaluate through it. The
primitives are the generic `AbstractMatrix` methods defined below, so
the family stays in-family (dense+dense stays dense, HSS+HSS stays HSS);
structured backends (`HssMatricesBackend`) specialize the same
generic functions for in-family wins (e.g. balanced chains).
"""
struct MatrixFamilyBackend{T<:Number} <: AbstractCompressionBackend end

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

"""In-family norm of a compressed operator (mandatory)."""
function op_norm end

"""
    apply(m, x)

Operator action of `m` on `x` (used for residual closures and lazy
products). In-family; no dense fallback.
"""
function apply end

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
not yet implemented by any backend).
"""
function solve end

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
#   op_add, op_scale, op_mul (binary), op_adjoint, solve, blockdiag_blocks,
#   compress.
# Derived manipulations (default implementations below, expressed via the
# core; backends with a cheaper structured version override them):
#   op_sub, op_norm, multi-arg op_mul chains.
op_add(A::AbstractMatrix, B::AbstractMatrix) = A + B
op_scale(alpha::Number, A::AbstractMatrix) = alpha * A
op_mul(A::AbstractMatrix, B::AbstractMatrix) = A * B
op_adjoint(A::AbstractMatrix) = adjoint(A)
apply(A::AbstractMatrix, x) = A * x
structure(A::AbstractMatrix) = nothing

# Derived: difference via add/scale
op_sub(A::AbstractMatrix, B::AbstractMatrix) = op_add(A, op_scale(-1, B))

# Derived: Frobenius norm via randomized probing (exact for the generic
# matrix family where `norm` exists; structured backends override with
# their cheap structured norm, e.g. HSS via tr(m'm))
op_norm(A::AbstractMatrix) = norm(A)

function solve(b::AbstractCompressionBackend, left::AbstractMatrix, right::AbstractMatrix;
    strategy::AbstractSolveStrategy=DirectSolve())
    strategy isa DirectSolve || throw(ArgumentError(
        "strategy $(typeof(strategy)) not supported by backend $(typeof(b))"))
    return ldiv(left, right)
end

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
as a graph node, evaluated by the backend's interpreter.
"""
struct SolveNode <: AbstractOpNode
    backend::AbstractCompressionBackend
    left::AbstractOpNode
    right::AbstractOpNode
    strategy::AbstractSolveStrategy
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
evaluate(b::AbstractCompressionBackend, m::AbstractCompressedOperator) = m

# ---------------------------------------------------------------------------
# Legacy compression bridge: every AbstractCompression evaluates through a
# backend (MatrixFamilyBackend by default). This is the single evaluation
# entry point of the package: kernel algebra, solver and observables all
# build DAGs and call evaluate.
# ---------------------------------------------------------------------------

"""
    backend(c::AbstractCompression)

The backend a legacy compression evaluates through. Defaults to
`MatrixFamilyBackend` (the generic AbstractMatrix family reference
backend, working in `ComplexF64`); structured compressions override this
to return their dedicated backend. Precision is not yet derived from the
compression's stored matrices (the eltype is inert for the generic
family; see `working_eltype`).
"""
backend(::AbstractCompression) = MatrixFamilyBackend{ComplexF64}()

"""
    evaluate(c::AbstractCompression, ex::AbstractOpNode)

Evaluate a DAG with the backend of a legacy compression. This is how
existing `AbstractCompression` objects (dense, HSS, JLArrays, ...) reach
the DAG world without changes at call sites.
"""
evaluate(c::AbstractCompression, ex::AbstractOpNode) = evaluate(backend(c), ex)

"""
    evaluate(c::AbstractCompression, m::AbstractMatrix)

Single-leaf convenience: evaluate a compressed operator in the
compression's family.
"""
evaluate(c::AbstractCompression, m::AbstractMatrix) = m

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
function evaluate_cse(b::AbstractCompressionBackend, bindings, ex)
    env = Dict{Symbol,Any}()
    for (name, node) in bindings
        env[name] = _eval_with(b, node, env)
    end
    return _eval_with(b, ex, env)
end

function _eval_with(b::AbstractCompressionBackend, ex::AbstractOpNode, env)
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
    evaluate(b::AbstractCompressionBackend, ex::AbstractOpNode)

Evaluate a numeric DAG with backend `b`. The default implementation is
the in-family interpreter; it first applies CSE so repeated subgraphs
are computed once. Backends may override for fusion/reassociation (and
typically want to run `cse` first as well).
"""
function evaluate(b::AbstractCompressionBackend, ex::AbstractOpNode)
    bindings, root = cse(ex)
    return evaluate_cse(b, bindings, root)
end
