# Backend contract and the expression DAG

This page documents the backend interface implemented today: the
**numeric-tier contract** that compression backends implement, the
**expression DAG** the package builds over them, and the reference HSS
backend with its optimized DAG evaluation. The full design rationale and
migration plan are in [Backend Interface Design](backend_interface_design.md);
the legacy `AbstractCompression` interface still in production use is
documented in [Custom compression](custom_compression.md).

## Overview

The DAG is the **only evaluation path of the package**. Every numeric
operation - kernel algebra (`+`, `-`, `*`, scalar scaling, `adjoint`),
`solve_dyson`, the local-kernel fast paths - builds an expression DAG
and evaluates it with the compression's backend:

```
Kernel `*`, `+`, `-`, adjoint, solve_dyson        (user-facing, semantic tier:
        |                                         quadrature weighting, dressing,
        |                                         causality - package-owned)
        v
op_add / op_sub / op_scale / op_mul / op_adjoint / SolveNode   (numeric tier)
        |
        |  evaluate(cpr, ex)      default in-family interpreter, or a backend
        v                          override (HSS: balanced chains)
AbstractCompressedOperator
```

Two rules govern the numeric tier:

1. **Everything is in-family.** Operations on compressed operators of a
   backend's family produce operators of the same family. There is no
   dense fallback; an unsupported operation is a `MethodError`.
2. **The backend owns configuration.** Working eltype, device and
   threading live on the backend (`AbstractCompressionBackend`), never
   on the stored matrices.

### Semantic vs numeric tier in practice

Kernel products are **not** pure matrix products: each pairwise product
applies the quadrature rule's weighting (semantic tier, package-owned).
A chain `G_R * Σ_K * G_R'` at the kernel level therefore applies the
trapezoid dressing at *each* product - writing it as a raw variadic
`op_mul` over the matrices would compute something different (no
dressing). Use kernel `*` for physics; use the explicit `op_mul` DAG
nodes only for raw matrix chains inside one family. Since the kernel
operators evaluate through the DAG internally, the DAG remains the only
evaluation path in both cases.

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

## The mandatory primitive set

A backend is an `AbstractCompressionBackend` subtype implementing:

| Primitive | Meaning |
|---|---|
| `working_eltype(b)` | the scalar type assembly and solve work in |
| `op_add(a, b)` | in-family sum |
| `op_sub(a, b)` | in-family difference |
| `op_scale(alpha, a)` | in-family scalar scaling |
| `op_mul(a, b)` | in-family binary product (the variadic chain reduces to it unless the backend overrides `evaluate`) |
| `op_adjoint(a)` | in-family adjoint |
| `op_norm(a)` | in-family norm (mandatory) |
| `apply(a, x)` | operator action on a dense block of vectors |
| `structure(a)` | the structure of `a` (`nothing` if not exposed) |
| `empty_like(a)` | zero-coefficient operator with the same structure |
| `solve(b, left, right; strategy)` | in-family solve (see strategies) |
| `blockdiag_blocks(a, bs)` | `bs×bs×N` equal-time blocks (the observables/solver boundary) |

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
