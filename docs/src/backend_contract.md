# Backend contract and the expression DAG

This page documents the **single backend contract** implemented today: the
**numeric-tier contract** that backends implement, the **expression DAG**
the package builds over them, and the reference backends with their
optimized DAG evaluation. The design rationale is in
[Backend Interface Design](backend_interface_design.md); the
representation tags that select a backend for the kernel constructors are
documented in [Custom representations](custom_compression.md).

## Overview

Evaluation has **three layers**, two of which are DAGs:

```
Kernel-level DAG   (physics constructs, lazy - package-owned)
    KernelProd nodes: kernel `*`, adjoint; traits (causality, locality,
    blocksize, quadrature) computed at construction
        |
        |  lower(op)              the decomposition pass: emits concrete
        v                        DAG nodes for exactly the operations the
Concrete DAG       (matrix manipulation - the backend contract)
    op_add / op_sub / op_scale / op_mul / op_adjoint / SolveNode
        |
        |  evaluate(b, ex)        default in-family interpreter, or a
        v                        backend override (HSS: balanced chains)
family of the backend (e.g. Matrix, HssMatrix, JLArray)
```

The split is deliberate: **the backend DAG contains only concrete matrix
manipulation**; physics (quadrature weighting, dressing, edge
corrections, causality) lives in the kernel-level DAG and the lowering
pass, which are package-owned. Backends never see physics.

### The whole computation lives in the kernel-level DAG

Every kernel-level operation is a lazy construct - products, sums,
differences, scalar scaling, adjoints:

- `A * B` returns a lazy `KernelProd` node carrying its traits
  (causality, locality, blocksize, quadrature);
- `A + B` / `A - B` return `SumOperator` nodes - the DAG's sum
  construct - with `I + K` supported via `UniformScaling` children;
- `λ * op`, `adjoint(op)` distribute onto the tree without evaluating.

Nothing is computed at construction. The entire expression - e.g. all of
`lead_current`'s `(1/2)((Σ_R G_R)' - Σ_R G_R) + G_R Σ_K + ...` - is a
lazy tree. It materializes only at a **forcing point**: `matrix(op)`,
`solve_dyson`, the observables (`same_time`, `keldysh_trace`), `norm`,
`==`, or `compress!`. At that point the whole tree lowers to ONE
concrete DAG (all dressings, weights, edge corrections and sums
included) and the backend evaluates it as a single graph - so a backend
with a fused `evaluate` sees the entire computation, not per-operation
fragments.

Equality of lazy operators is **by value** (the materialized matrices),
consistent with `Kernel` equality: different trees representing the same
operator compare equal.

### A DAG is evaluated at most once (materialization cache)

Every lazy node (`KernelProd`, `SumOperator`) caches its materialized
result on the node itself. Consequences:

- repeated forcing (`matrix` twice, then `same_time`, `keldysh_trace`,
  `norm`, `==`) reuses the cached result - identical objects, zero
  re-evaluation;
- a child shared by several trees is evaluated once and reused as a leaf
  by every parent (the fusion boundary is the node boundary: each
  node's own lowering is still one fused concrete DAG);
- a new tree built from already-evaluated children only evaluates its
  new nodes.

The cache lives on the nodes (not in global state): it is garbage
collected with the tree, and building a fresh expression from the same
kernels evaluates fresh.

### Equality of lazy operators

Comparisons use a three-tier scheme that never evaluates unnecessarily:

1. **node identity** (`A === B`): true, zero cost, no evaluation;
2. **cheap traits** (causality, axis, blocksize): mismatched trees fail
   fast *without* materializing;
3. **value comparison** (materialized matrices) as the last resort -
   through the node caches, so each tree is evaluated at most once ever;
   repeated comparisons are free.

### Common-subexpression elimination on the lowered graph

The kernel-level cache guarantees a *tree* is never evaluated twice, but
a single lowered concrete graph can still contain the same subgraph at
several positions (e.g. a shared product used directly and inside an
adjoint). `cse` extracts every subgraph occurring more than once:

```julia
AB  = op_mul(CompressedLeaf(A), CompressedLeaf(B))
ex  = op_add(op_mul(AB, CompressedLeaf(C)), op_scale(2.0, AB))

bindings, root = cse(ex)      # 1 binding: cse_1 = A*B
result = evaluate_cse(b, bindings, root)   # A*B computed once
```

- `cse(ex) -> (bindings, root)`: returns the topologically ordered
  bindings (a binding body only refers to leaves and earlier bindings)
  and the rewritten root referencing them through `BindingNode` leaves.
- `evaluate_cse(b, bindings, root)`: the reference evaluation - each
  binding is computed exactly once, then the root consumes them.
- The **default `evaluate` applies CSE automatically**, and the HSS
  backend's fused `evaluate` runs `cse` first as well, so every backend
  gets it.

Two subgraphs are common iff structurally equal over identical leaf
objects; leaves themselves are never extracted (cheap references).

Products with `LocalKernel` operands stay concrete (exact block-diagonal
fast paths); these are their own lowering and never reach the lazy tree.

Two rules govern the concrete DAG:

1. **Everything is in-family.** Operations between operators of a
   backend's family produce operators of the same family. There is no
   dense fallback; an unsupported operation is a `MethodError`.
2. **The backend owns configuration.** Working eltype, device and
   threading live on the backend (`AbstractBackend`), never on the
   stored matrices.

### Core and derived manipulations

Backends implement a small **core** (listed once in `CORE_PRIMITIVES` in
`src/backend_interface.jl`, which is the source of truth - the
contract-audit meta-test fails if a shipped backend lacks a method for
any listed primitive); other operations have default implementations
expressed through the core (staying in-family by construction) and exist
only as override points for backends with cheaper structured versions:

| Core (implement) | Derived (default implementation) |
|---|---|
| `compress`, `op_add`, `op_scale`, `op_mul` (binary), `op_adjoint`, `solve`, `blockdiag_blocks`, `to_cpu`, `same_time_blocks`, `recompress_inplace!`, `working_eltype` | `op_sub` (= `op_add(a, op_scale(-1, b))`), multi-arg `op_mul` chains (left fold) |

### Eltype homogeneity

Operands of the binary primitives must share an eltype unless the backend
explicitly widens (the HSS backend does, via `_widen_eltype`); the
generic `AbstractMatrix` family relies on Base promotion and
`MethodError`s on families without native promotion.

### The `evaluate` override is optional

Backends may expose their own DAG evaluation (`evaluate(b, ex)`) for
fusion/reassociation/joint optimization; otherwise the default
in-family interpreter is used. `HssMatricesBackend` overrides it with
balanced-chain `op_mul` evaluation; `MatrixFamilyBackend` uses the
default.

### Representation tags select the backend

The kernel constructors take a `representation=` tag
(`DenseRepresentation`, `HssRepresentation`, `JLArrayRepresentation`,
custom ones): a pure parameter holder. `backend(r)` returns the
`AbstractBackend` it evaluates through (`MatrixFamilyBackend` by default,
`HssMatricesBackend` for `HssRepresentation`, `JLArrayBackend` for
`JLArrayRepresentation`). `evaluate(r, ex)` evaluates a DAG with it.
Construction of kernel matrices goes through the `compress` primitive of
that backend - there is no other construction entry point.

## The DAG

A graph is built lazily - construction does no numerical work - and
evaluated with `evaluate`:

```julia
using NonEquilibriumGreenFunction

b = HssMatricesBackend(atol=1e-10, rtol=1e-10, leafsize=32)

# leaves wrap already-compressed operators of the backend's family
A, B, C, D = ...   # e.g. HssMatrix objects

ex = op_mul(CompressedLeaf(A), CompressedLeaf(B),
            op_adjoint(CompressedLeaf(C)))          # one variadic chain node
result = evaluate(b, ex)                            # HssMatrix
```

Node constructors (all lazy, all pure):

| Constructor | Node |
|---|---|
| `CompressedLeaf(op)` | leaf: an already-compressed operator |
| `op_add(ex...)` | sum node |
| `op_sub(a, b)` | difference node (`a + (-1)*b`) |
| `op_scale(alpha, ex)` | scalar scaling node |
| `op_mul(ex...)` | **variadic chain-product node** |
| `op_adjoint(ex)` | adjoint node |
| `SolveNode(b, left, right, strategy)` | solve node (fails fast with `ArgumentError` if `b` does not support `strategy`) |

The variadic `op_mul` is the multi-kernel entry point: chains like
`G_R * Σ_K * G_R'` reach the backend as **one node**, so the backend can
choose the contraction order and fuse the pass instead of folding
pairwise.

### Evaluation semantics

`evaluate(b, ex)` must be equivalent to the default interpreter within
tolerance. The default interpreter:

- reduces `op_mul` by **left fold** over the binary primitive,
- reduces `op_add`/`op_sub`/`op_scale`/`op_adjoint` with the binary
  primitives,
- resolves `SolveNode` via `solve(b, left, right; strategy)`.

Backends may override `evaluate` to reassociate or fuse (any order is
valid if the result matches the left fold within tolerance).

## The primitive set

A backend is an `AbstractBackend` subtype implementing the **core**
manipulations (see `CORE_PRIMITIVES`); derived manipulations have
in-family default implementations and are override points:

| Primitive | Core/derived | Meaning |
|---|---|---|
| `compress(b, axis, f; stationary=false)`, `compress(b, m)`, `compress(b, blocks)` | core | the only construction entry points: kernel matrix from a sampler, (re)compress a matrix, block-diagonal matrix |
| `working_eltype(b)` | core | the scalar type assembly and solve work in |
| `op_add(a, b)` | core | in-family sum |
| `op_scale(alpha, a)` | core | in-family scalar scaling |
| `op_mul(a, b)` | core | in-family binary product (variadic chains left-fold to it unless the backend overrides `evaluate`) |
| `op_adjoint(a)` | core | in-family adjoint |
| `solve(b, left, right; strategy)` | core | in-family solve (see strategies) |
| `blockdiag_blocks(b, a, bs)` | core | `bs×bs×N` equal-time blocks (the observables/solver boundary) |
| `to_cpu(b, a)` | core | materialize as a CPU `Matrix` |
| `same_time_blocks(b, a, bs)` | core | one `bs×bs` CPU block per time step |
| `recompress_inplace!(b, rep, m)` | core | in-place recompression with the representation's parameters |
| `op_sub(a, b)` | derived | `op_add(a, op_scale(-1, b))` |
| `op_norm(a)` | **optional, test-only** | in-family norm; `norm(::AbstractOperator)` deliberately does not route through it (it uses the exact Frobenius norm) |
| `apply(a, x)` | derived | operator action |
| `structure(a)` | optional | the structure of `a` (`nothing` default) |
| `empty_like(a)` | optional | zero-coefficient operator with the same structure |

Optional: `evaluate(b, ex)` (whole-graph optimization).

### Solve strategies

```julia
DirectSolve()                    # in-family factorization solve
StructureOptimizingSolve(maxiter, tol)
                                 # structure of `left` fixed, coefficients
                                 # optimized by the backend to `tol`
```

`StructureOptimizingSolve` is the contract for "HSS structure known,
coefficients optimized": the package supplies `structure(left)` and
`empty_like`; the iteration is entirely backend-owned. Backends that do
not support a strategy reject it **at construction** of the `SolveNode`
(`ArgumentError`) and in `solve` itself - building the node never
succeeds silently.

## Threading

Backends and callers share one budget facility:

```julia
with_thread_budget(4) do         # set an ambient budget of 4 threads
    fetch(Threads.@spawn op_mul(A, B, C))   # backend caps itself at min(config, 4)
end
```

`current_thread_budget()` reads the ambient budget (defaults to
`Threads.nthreads()`); the value is task-local and inherited by spawned
children, so outer task fan-outs (e.g. Negf.jl evaluating several kernel
products concurrently) compose with backend-internal parallelism without
oversubscription.

### Thread safety of lazy nodes

The materialization caches embedded in the kernel-level DAG nodes
(`KernelProd`, `SumOperator`) are **not synchronized**. Concurrent
forcing is supported only over *disjoint* trees: each task must build
and force its own nodes (e.g. Negf.jl-style outer fan-outs of
independent chains, with the forcing point inside each task).
Forcing the *same* lazy node from two tasks races on the check-then-set
cache and can interleave writes on a non-thread-safe backend
(HssMatrices); materialize shared subtrees before handing them to
several tasks. Note that kernel products are lazy: a spawned task that
only *constructs* a product does no numeric work - the forcing point
(`matrix` and friends) must run inside the task. Synchronization is
planned for the threading stage of the backend rework.

## The matrix-family reference backend

`MatrixFamilyBackend{T}` is the reference backend for any family built on
`AbstractMatrix` with standard Base operations: dense `Matrix`, HSS
matrices, GPU-style arrays whose family implements the Base ops. It is
the default bridge for representation tags without a dedicated backend.
Its primitives are the generic `AbstractMatrix` methods of the contract
(`+`, `-`, scalar `*`, `*`, `adjoint`, `norm`, `\`), which stay
in-family for every matrix family - this is not a dense fallback, since
nothing ever converts between families.

## The reference HSS backend

`HssMatricesBackend(; atol, rtol, kest, leafsize, working_type=ComplexF64)`
adapts HssMatrices.jl to the contract:

- primitives map to native HSS operations (`A + B`, `A * B`, `adjoint`,
  `alpha * A`, the structured Frobenius norm `computeMatrixNorm`, the ULV
  solve `ldiv!` for `DirectSolve`),
- `compress` builds HSS matrices natively: `randcompress_adaptive` from a
  sampled axis function (circulant map when `stationary=true`), `hss`
  from a dense matrix, `hss_blkdiag` (exact) from a block array,
- `structure(m) === m` - the `HssMatrix` (its cluster tree and rank
  profile) is the structure,
- `empty_like(m)` zeroes `D`, `B12`, `B21` and keeps the generator bases
  `U`, `V`, `R1`, `R2`, `W1`, `W2`, so the tree shape and rank profile
  are preserved verbatim,
- `StructureOptimizingSolve` is not yet implemented (rejected at
  `SolveNode` construction) - it is the entry point for the upcoming
  optimized-coefficient solver.

### Optimized DAG evaluation for HSS

The HSS backend overrides `evaluate`:

- it runs `cse` **first**, so repeated subgraphs (also inside a root
  `MulNode` chain) are extracted into bindings and evaluated exactly
  once,
- variadic `op_mul` chains are evaluated in **balanced (tree) order**:
  a chain `A*B*C*D` is evaluated as `(A*B)*(C*D)` rather than the
  default left fold. For HSS this halves the depth of sequential
  products (`log2(k)` vs `k-1` steps), which bounds intermediate rank
  growth and accumulates less compression error per step,
- all other node types use the default in-family interpreter, which is
  exact for the HSS family since every primitive is in-family.

The acceptance property (checked in the test suite): the balanced
evaluation matches both the dense reference and the left-fold result
within tolerance, for chains of length 3 and 4, and a repeated subchain
inside a pure-`MulNode` root is computed once.

## Testing a backend

`test_representation_contract` (the contract audit: every representation
checked against the dense reference - construction, algebra, solver,
recompression, block-diagonal construction) plus the DAG contract tests
in `src/backend_interface_tests.jl` are the templates: a backend should
check every primitive in `CORE_PRIMITIVES` against a dense reference (the
meta-test iterates `CORE_PRIMITIVES` and fails if the backend lacks a
method), then check its `evaluate` (if overridden) against the default
interpreter on graphs covering every node type - including chain
comparisons at combined tolerance, since reassociation changes
floating-point rounding.
