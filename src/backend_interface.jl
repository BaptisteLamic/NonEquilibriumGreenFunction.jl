# Backend interface (DRAFT - design phase, see docs/src/backend_interface_design.md)
#
# Non-breaking skeleton: these types and contracts are defined but not yet
# consumed by the existing compression/matrix paths. Migration proceeds per
# the plan in the design document; existing code keeps working unchanged.

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
"""
function working_eltype end

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
"""
struct StructureOptimizingSolve <: AbstractSolveStrategy
    maxiter::Int
    tol::Float64
end

"""
    solve(b, left, right; strategy=DirectSolve())

Solve `left * X = right` in the family of backend `b`. `DirectSolve` is
the factorization path; `StructureOptimizingSolve` keeps `structure(left)`
fixed and optimizes coefficients (backend-owned iteration). Replaces
`ldiv` in the migration.
"""
function solve end

# ---------------------------------------------------------------------------
# Expression DAG (numeric tier; semantic tier nodes are added in migration
# step 5 - quadrature/dressing/integration stay package-owned)
# ---------------------------------------------------------------------------

"""Numeric DAG node. Flat and acyclic over compressed leaves."""
abstract type AbstractOpNode end

"""Leaf: an already-compressed operator."""
struct CompressedLeaf <: AbstractOpNode
    op::AbstractCompressedOperator
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
function evaluate(b::AbstractCompressionBackend, ex::AbstractOpNode)
    ex isa CompressedLeaf && return ex.op
    ex isa AddNode && return foldl(op_add, map(a -> evaluate(b, a), ex.args))
    ex isa ScaleNode && return op_scale(ex.alpha, evaluate(b, ex.arg))
    ex isa AdjointNode && return op_adjoint(evaluate(b, ex.arg))
    ex isa MulNode && begin
        vals = map(a -> evaluate(b, a), ex.args)
        return foldl(op_mul, vals)
    end
    ex isa SolveNode && return solve(b, evaluate(b, ex.left), evaluate(b, ex.right); strategy=ex.strategy)
    throw(ArgumentError("unsupported node type: $(typeof(ex))"))
end
evaluate(b::AbstractCompressionBackend, m::AbstractCompressedOperator) = m
