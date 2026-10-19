# Backend contract and the expression DAG

This page documents the backend interface implemented today: the
**numeric-tier contract** that compression backends implement, the
**expression DAG** the package builds over them, and the reference HSS
backend with its optimized DAG evaluation. The full design rationale and
migration plan are in [Backend Interface Design](backend_interface_design.md);
the legacy `AbstractCompression` interface still in production use is
documented in [Custom compression](custom_compression.md).

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
        |  evaluate(cpr, ex)      default in-family interpreter, or a
        v                        backend override (HSS: balanced chains)
AbstractCompressedOperator
```

The split is deliberate: **the backend DAG contains only concrete matrix
manipulation**; physics (quadrature weighting, dressing, edge
corrections, causality) lives in the kernel-level DAG and the lowering
pass, which are package-owned. Backends never see physics.

### Kernel products are lazy constructs

`A * B` between kernels returns a lazy `KernelProd` node - nothing is
computed. The whole chain `G_R * Σ_K * G_R'` is a tree of `KernelProd`
nodes carrying its traits. It materializes only at a **forcing point**:
`matrix(op)`, `solve_dyson`, the observables (`same_time`,
`keldysh_trace`), `norm`, or `compress!`. At that point the tree
lowers to ONE concrete DAG (all dressings, weights and corrections
included) and the backend evaluates it as a single graph - so a backend
with a fused `evaluate` sees the whole weighted chain, not per-product
fragments.

Products with `LocalKernel` operands stay concrete (exact block-diagonal
fast paths), and `SumOperator`s containing lazy parts materialize their
parts through the same lowering.

Two rules govern the concrete DAG:

1. **Everything is in-family.** Operations between operators of a
   backend's family produce operators of the same family. There is no
   dense fallback; an unsupported operation is a `MethodError`.
2. **The backend owns configuration.** Working eltype, device and
   threading live on the backend (`AbstractCompressionBackend`), never
   on the stored matrices.

### Core and derived manipulations

Backends implement a small **core**; other operations have default
implementations expressed through the core (staying in-family by
construction) and exist only as override points for backends with
cheaper structured versions:

| Core (implement) | Derived (default implementation) |
|---|---|
| `op_add`, `op_scale`, `op_mul` (binary), `op_adjoint`, `solve`, `blockdiag_blocks`, `compress` | `op_sub` (= `op_add(a, op_scale(-1, b))`), multi-arg `op_mul` chains (left fold), `op_norm`, `apply` |

### The `evaluate` override is optional

Backends may expose their own DAG evaluation (`evaluate(b, ex)`) for
fusion/reassociation/joint optimization; otherwise the default
in-family interpreter is used. `HssMatricesBackend` overrides it with
balanced-chain `op_mul` evaluation; `MatrixFamilyBackend` uses the
default.

### Legacy compressions evaluate through backends

`AbstractCompression` objects (dense `NONCompression`, `HssCompression`,
`JLArrayCompression`, custom ones) are bridged: `backend(c)` returns
their evaluation backend (`MatrixFamilyBackend` by default,
`HssMatricesBackend` for `HssCompression`). `evaluate(c, ex)` evaluates
a DAG with it, so all existing code and compressions reach the DAG
world unchanged.

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
| `SolveNode(b, left, right, strategy)` | solve node |

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

A backend is an `AbstractCompressionBackend` subtype implementing the
**core** manipulations; derived manipulations have in-family default
implementations and are override points:

| Primitive | Core/derived | Meaning |
|---|---|---|
| `working_eltype(b)` | core | the scalar type assembly and solve work in |
| `op_add(a, b)` | core | in-family sum |
| `op_scale(alpha, a)` | core | in-family scalar scaling |
| `op_mul(a, b)` | core | in-family binary product (variadic chains left-fold to it unless the backend overrides `evaluate`) |
| `op_adjoint(a)` | core | in-family adjoint |
| `solve(b, left, right; strategy)` | core | in-family solve (see strategies) |
| `blockdiag_blocks(a, bs)` | core | `bs×bs×N` equal-time blocks (the observables/solver boundary) |
| `op_sub(a, b)` | derived | `op_add(a, op_scale(-1, b))` |
| `op_norm(a)` | derived | norm (generic families use `norm`; structured backends override with the cheap structured version, e.g. HSS `tr(m'm)`) |
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
`empty_like`; the iteration itself is entirely backend-owned.

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

## The matrix-family reference backend

`MatrixFamilyBackend{T}` is the reference backend for any family built on
`AbstractMatrix` with standard Base operations: dense `Matrix`, HSS
matrices, GPU-style arrays whose family implements the Base ops. It is
the default bridge for legacy `AbstractCompression` objects without a
dedicated backend. Its primitives are the generic `AbstractMatrix`
methods of the contract (`+`, `-`, scalar `*`, `*`, `adjoint`, `norm`,
`\`), which stay in-family for every matrix family - this is not a dense
fallback, since nothing ever converts between families.

## The reference HSS backend

`HssMatricesBackend(; atol, rtol, kest, leafsize, working_type=ComplexF64)`
adapts HssMatrices.jl to the contract:

- primitives map to native HSS operations (`A + B`, `A * B`, `adjoint`,
  `alpha * A`, the structured Frobenius norm `computeMatrixNorm`, the ULV
  solve `ldiv!` for `DirectSolve`),
- `structure(m) === m` - the `HssMatrix` (its cluster tree and rank
  profile) is the structure,
- `empty_like(m)` zeroes `D`, `B12`, `B21` and keeps the generator bases
  `U`, `V`, `R1`, `R2`, `W1`, `W2`, so the tree shape and rank profile
  are preserved verbatim,
- `StructureOptimizingSolve` is not yet implemented (errors) - it is the
  entry point for the upcoming optimized-coefficient solver.

### Optimized DAG evaluation for HSS

The HSS backend overrides `evaluate` with a **balanced-chain**
evaluation of variadic `op_mul`:

- a chain of length `k` is split at the middle and evaluated
  recursively: `A*B*C*D` becomes `(A*B)*(C*D)`, giving `log2(k)`
  sequential product steps instead of `k-1` for the left fold,
- shallower product chains mean intermediate HSS representations go
  through fewer recompressions, which bounds intermediate rank growth
  and accumulates less compression error,
- all other node types (add, scale, adjoint, solve, and mixed graphs)
  fall through to the default interpreter, which is exact for the HSS
  family since every primitive is in-family.

The acceptance property (checked in the test suite): the balanced
evaluation matches both the dense reference and the left-fold result
within tolerance, for chains of length 3 and 4.

## Testing a backend

`test_compression_interface` (legacy path) plus the DAG contract tests
in `src/backend_interface_tests.jl` are the templates: a backend should
check every primitive against a dense reference, then check its
`evaluate` (if overridden) against the default interpreter on graphs
covering every node type - including chain comparisons at combined
tolerance, since reassociation changes floating-point rounding.
