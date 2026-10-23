# Backend Interface Design

Status: implemented. The compression system has been reworked into a
**backend interface** in preparation for a new HSS backend that performs
compression on GPU and CPU, runs in mixed precision, is multithreaded,
and supports structure-known / coefficient-optimized solving strategies.

The full interface is in place: the numeric-tier skeleton
(`src/backend_interface.jl`) defines the types, the mandatory primitive
set, the thread-budget facility and the default DAG interpreter; the
kernel-level DAG (`src/Kernels/kernel_dag.jl`) provides the lazy physics
tier with its package-owned lowering pass; and the reference HSS backend
(`src/hss_backend.jl`) implements the contract with an optimized
whole-graph `evaluate`. See the *Backend contract* page for the
user-facing reference of what is implemented today.

## Motivation

The current contract (`AbstractCompression` callable + duck-typed matrix
family, see `docs/src/custom_compression.md`) carries several assumptions
that the new backend cannot satisfy:

1. The sampling API is a raw Julia closure with no negotiation: backends
   cannot ask for batches, a different element type, a device, or their own
   parallel schedule.
2. Compression and solve strategy are conflated: the only solve hook is
   `ldiv`, with no way to express "same HSS tree, optimize the coefficients".
3. Precision and device are not first-class: `scalartype` is derived from
   whatever `f` returned at sample time; `to_cpu` is a one-way escape hatch.
4. Threading is implicit (`Threads.@threads` inside package code), which
   deadlocks/oversubscribes when composed with outer task parallelism
   (e.g. Negf.jl spawning several kernel products concurrently).

## Principles

- **Backend-owned representation.** The compressed object is an opaque
  `AbstractCompressedOperator`, NOT an `AbstractMatrix`. No `getindex`,
  no Base operator overload obligations, no materialization expectations.
- **Eltype, precision, device, threads are configured on the backend.**
  The backend declares `sample_type`, `coeff_type`, `working_type`,
  `device` and thread configuration. The package never sees an eltype;
  `scalartype` reads `working_eltype(backend)`.
- **No dense fallback.** Every primitive is mandatory. If a backend cannot
  implement an op in-family, it fails with a `MethodError` at first use.
  Dense is itself a backend (`NONCompression` becomes one), used as the
  reference in tests - never as a fallback for other backends.
- **Quadrature/masking/dressing are package-owned semantics.** Backends
  never implement discretization schemes; there is exactly one quadrature
  implementation, so scheme divergence between backends is impossible.
- **Causality is leaf metadata (support), not an operation.** The support
  pattern (lower/upper/full) is a trait carried from the kernel map; the
  package samples only the support; the backend receives the support as a
  tag it may exploit structurally (block-triangular HSS) but never executes.
- **Threading is task-based and budgeted.** Backends use `Threads.@spawn`
  (never `Threads.@threads`), inherit an ambient thread budget
  (`with_thread_budget` / `current_thread_budget`), and stay within
  `min(config.max_threads, current_thread_budget())`. This composes with
  Negf.jl-style outer `@spawn` fan-outs of multiple kernel products.

## Architecture: two-tier expression DAG

```
Semantic tier  (package-interpreted, ALWAYS; backends never see this as work)
    SamplerLeaf(map, axis; causality)        lazy leaf, support from causality
    op_quadrature(rule, ex)                 rule weights + edge corrections
    op_dress(ex, blockdiag_ex)               block-diagonal weighting
    op_integrate(ex)                         dt scale
        |
        |  lower(b, graph)                    package-owned lowering pass
        v
Numeric tier   (the backend contract)
    CompressedLeaf(m)                        already-compressed operator
    op_add / op_sub / op_scale / op_mul(variadic) / op_adjoint
    solve_node(b, left, right; strategy)
        |
        |  evaluate(b, numeric_ex)            default in-family interpreter;
        v                                     backends may override for fusion
    AbstractCompressedOperator + traits
```

### The lowering pass

`lower(b, graph)` is package-owned and generic over backends. It walks the
semantic tier, reads the leaf's causality to derive the support pattern,
samples only the support, applies quadrature weights on the support, and
emits a flat numeric DAG over `AbstractCompressedOperator` leaves plus a
trait record per node: `(causality, locality, blocksize, working_eltype)`.

The `t = tp` support boundary (trapezoid half-weights, edge corrections in
`kernel_algebra.jl`) is exactly where support and quadrature meet; that
logic has exactly one home: the quadrature nodes in the semantic tier.

### Graph evaluation

`evaluate(b, ex)` has a **default in-family interpreter** that reduces the
numeric DAG with the mandatory binary primitives (left-fold for `op_mul`).
This makes the binary primitives the correctness floor for every backend;
fusion/whole-graph optimization is a pure performance override:

- backends may reassociate `op_mul` chains (variadic node) for better
  HSS contraction order and rank behavior,
- backends may jointly optimize coefficients of a whole subgraph with
  known structure (generalizing `StructureOptimizingSolve` to any
  subgraph solved in structure space),
- mixed precision becomes a graph-scheduling decision (per-node
  compute/convert types) owned by the backend.

Semantics: any evaluation order is valid within a combined tolerance of
the left-fold reference. The graph is flat and acyclic over compressed
leaves; kernel-map leaves are restricted to the existing `AbstractKernelMap`
types (no user node types, no symbolic CAS).

The typical hot pattern - multiplying several kernels at the same time
(`G_K = G_R * Σ_K * G_R'`, `lead_current`'s chains, Negf.jl spawning
several products concurrently) - is served twice: many chains run under
outer `@spawn` with the shared budget, and each chain is one variadic
`op_mul` node the backend may fuse.

## The backend contract (numeric tier)

Mandatory primitives - `AbstractCompressionBackend` subtypes must
implement all of them in-family:

| Primitive | Meaning |
|---|---|
| `compress(b, sampler; support, params)` | build the family from support-tagged samples |
| `compress(b, m; params)` | (re)compress an in-family operator (replaces `c(m)` + `recompress_inplace!`) |
| `structure(m)` | the structure (tree/rank profile) of `m`; `nothing` = unavailable |
| `empty_like(st)` | same structure, zeroed coefficients |
| `op_add(m1, m2)` | in-family sum |
| `op_sub(m1, m2)` | in-family difference |
| `op_scale(alpha, m)` | in-family scalar scaling |
| `op_mul(A, B, C...)` | variadic in-family chain product |
| `op_adjoint(m)` | in-family adjoint |
| `op_norm(m)` | in-family norm (mandatory) |
| `apply(m, x)` | operator action (residuals, lazy products) |
| `blockdiag_blocks(m, bs)` | one `bs x bs` CPU block per time step (observables + solver assembly) |
| `solve(b, left, right; strategy)` | `DirectSolve` (factorization) or `StructureOptimizingSolve` (structure of `left` fixed, backend iterates coefficients of `right`-side solution to tolerance) |
| `working_eltype(b)` | drives `scalartype` and `solve_dyson` assembly |
| `size(m)` | problem size |

Optional override:

| Override | Meaning |
|---|---|
| `evaluate(b, ex)` | whole-graph evaluation (fusion, reassociation, joint coefficient optimization, per-node precision) |

### The reference HSS implementation

`HssMatricesBackend` (`src/hss_backend.jl`) is the reference
implementation of the contract and the first optimized DAG consumer:

- all primitives map to native in-family HssMatrices operations (`A + B`,
  `A * B`, `adjoint`, `alpha * A`, the structured Frobenius norm, the
  ULV solve for `DirectSolve`),
- `structure(m)` returns the `HssMatrix` itself (its cluster tree and
  rank profile are the structure),
- `empty_like(m)` zeroes the coefficients (`D`, `B12`, `B21`) while
  keeping the full generator structure (`U`, `V`, `R`, `W` bases) - the
  seed for coefficient optimization with fixed structure,
- `evaluate` overrides the default interpreter with **balanced-chain**
  `op_mul` evaluation: a chain `A*B*C*D` is evaluated as `(A*B)*(C*D)`
  instead of the left fold `((A*B)*C)*D`. This halves the depth of
  sequential HSS products (`log2(k)` vs `k-1` steps), which bounds
  intermediate rank growth and accumulates less compression error per
  step. All other nodes fall through to the default interpreter, which is
  exact for the family.

Precision configuration lives on the backend:

```julia
struct MyHssBackend <: AbstractCompressionBackend
    sample_type::Type        # what f(t,tp) samples are converted to
    coeff_type::Type         # what the compressed representation stores
    working_type::Type       # what assembly/solve promotes to
    device::DeviceConfig     # :cpu / :gpu + array type
    threads::ThreadConfig     # max_threads, blas_threads
end
```

`StructureOptimizingSolve`: the HSS structure is known (from `left`), the
coefficients are optimized; the iteration is entirely backend-owned. The
package only supplies `structure(left)`, `empty_like`, and the residual
closure; it never iterates on coefficients itself.

## Threading contract

1. Backends use task-based parallelism only (`Threads.@spawn`, never
   `Threads.@threads`).
2. Thread budget is inherited, not global: `with_thread_budget(f, n)` sets
   a task-local budget inherited by spawned children;
   `current_thread_budget()` defaults to the backend's `max_threads` when
   no ambient budget exists.
3. External thread pools are part of the budget: BLAS threads and GPU
   stream/width are bounded by the same budget so that outer fan-outs
   (`K` concurrent kernel products) times inner parallelism stays within
   the total.
4. Package code never spawns inside a backend call: the backend owns its
   call's parallelism completely; composition happens at the algebra /
   caller level.

Acceptance: `lead_current`'s full expression (worst chain case) under
`@spawn` fan-out at budgets {1, nthreads, nthreads/2} produces identical
results with no oversubscription.

## Migration history (each step kept the suite green; all steps are
implemented)

0. Trait system + node types (define the graph before any backend work).
1. `AbstractCompressedOperator` + package-level op names + default
   in-family interpreter; port package-internal `Threads.@threads` loops
   to budgeted `@spawn` equivalents.
2. `AbstractCompressionBackend`; precision/device/threads fields;
   `scalartype` -> `working_eltype`.
3. `solve(b, left, right; strategy)` + strategies; `ldiv` becomes a
   deprecated shim.
4. Port the two reference backends (dense `NONCompression`, HSS
   `HssMatricesBackend`) onto the contract; prove the floor.
5. Lowering pass: move `kernel_algebra.jl`/`quadrature.jl` semantics into
   semantic nodes; `Kernel` constructors, `solve_keldysh` and
   `lead_current` become graph builders.
6. `test_compression_interface` becomes a contract audit: (a) semantic
   tier checked once against the dense backend; (b) per-backend numeric
   `evaluate` checked against the default interpreter on graphs covering
   every node type; chain comparisons use combined tolerances.
7. Docs rewrite (`custom_compression.md` -> backend contract).

The migration is complete: the new backend (GPU / mixed-precision /
multithreaded HSS) lands as one self-contained addition: a single
`AbstractCompressionBackend` subtype implementing the primitives above -
nothing else in the package changes.

## Open points

- `StructureOptimizingSolve` structure caching across `solve_dyson`
  calls (changes compression equality/hash conventions) - deferred.
- Mixed-precision direction default (low-storage/high-solve vs
  high-sample/low-coefficients) - backend configuration, both legal.
