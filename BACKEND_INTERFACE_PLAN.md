# Backend interface evolution plan: one contract (delegated implementation)

Companion to `CLEANUP_PLAN.md` (PR #57). This plan covers the **stage-2 backend
interface work** that the review of `src/backend_interface.jl`, `src/hss_backend.jl`,
`src/matrixinterface.jl`, `ext/JLArraysExt.jl` and the contract docs identified.

**Goal: collapse the two parallel contracts into a single one.** Today a backend
author must implement both the legacy callable (`c(axis, f; stationary=)`, `c(m)`,
`c(blocks)`, `ldiv`, `to_cpu`, `same_time_blocks`, `recompress_inplace!` on an
`AbstractCompression` subtype plus a duck-typed matrix family) and the new numeric
primitive set (`op_*`, `solve`, `evaluate` on an `AbstractCompressionBackend`),
bridged one-way by `backend(c)`. The plan below makes the backend contract the
only contract: the legacy types become thin adapters over it, then disappear.

Each item is independently implementable and verifiable, sized for a flash model.
Order matters: 1-4 define the single contract, 7-9 port the package and docs onto
it, 5-6 and 10-12 are consistency/edge fixes that can land any time after 1-4.

Context decisions (from review discussion):
- `JLArrayCompression` is a **demonstration** of the matrix-family interface,
  not a production GPU backend. Items touching it are low priority / optional.
- **Breaking changes are accepted** (pre-1.0). The old `AbstractCompression` name
  goes away outright — it now means two things (callable matrix-builder and
  parameter tag), which is exactly what the single-contract rework removes.

## 1. Define the single contract: make `compress` a real backend primitive

- Files: `src/backend_interface.jl`, `docs/src/backend_contract.md:139`,
  `docs/src/backend_interface_design.md` (mandatory-primitives table).
- The contract tables list `compress(b, sampler; ...)` / `compress(b, m)` as core
  primitives, but no `compress` function exists; construction goes through the
  legacy callable `c(axis, f)` / `c(m)` / `c(blocks)`.
- Add `compress(b::AbstractCompressionBackend, axis, f; support, params...)` and
  `compress(b, m; params...)` as the **only** construction entry points of the
  numeric tier. `MatrixFamilyBackend` implements them via the legacy callable;
  `HssMatricesBackend` via native HSS construction.
- Verify: grep finds a definition matching every contract-table row.

## 2. Rename the legacy type and make backends the primary concept

- Rationale: "compression" currently covers both the callable matrix-builder and
  the parameter tag; the single contract removes the duality.
- Introduce e.g. `AbstractMatrixRepresentation` (or `AbstractBackendParams`) as
  the name for what is today the `AbstractCompression` callable subtype, and
  **remove** `AbstractCompression` outright (breaking change accepted).
  `HssCompression`/`NONCompression`/`JLArrayCompression` become subtypes of the
  new abstract type but are now pure parameter holders; their callable methods
  move to `compress(b, ...)` on the corresponding backend.
- `backend(c)` becomes the identity-ish accessor `backend(r)` returning the
  `AbstractCompressionBackend` carrying those parameters.
- Touch points: `src/matrixinterface.jl` (definition), `src/compression.jl`
  (subtypes + callables), `ext/JLArraysExt.jl`, `src/backend_interface.jl`
  (bridge), `src/Kernels/*` (call sites of `c(...)`), exports in
  `src/NonEquilibriumGreenFunction.jl`, `docs/src/custom_compression.md` ->
  rewrite as the backend-contract page, `test/test_compression_interface.jl`.
- Verify: no remaining references to `AbstractCompression` (grep across src/,
  ext/, test/, docs/); full `Pkg.test()` green; docs use the new name.

## 3. Make the matrix-family methods backend primitives, not duck-typed Base ops

- Files: `src/backend_interface.jl:255-262,375-380`, `docs/src/backend_contract.md`.
- Today `ldiv`, `to_cpu`, `same_time_blocks`, `recompress_inplace!` are free
  functions on `AbstractMatrix` with fallbacks; the design doc says primitives
  must be mandatory in-family with no fallback.
- Move them into the contract as backend-dispatched primitives
  (`op_solve`/`solve` already exists; add `blockdiag_blocks` as a mandatory core
  — it already has an HSS specialization — plus `to_cpu`/`same_time_blocks` as
  core rows), with `MatrixFamilyBackend` providing the generic `AbstractMatrix`
  implementations. Delete the fallback free functions once call sites are ported.
- Verify: contract table has no "fallback" column; every row has a method per
  shipped backend.

## 4. State the eltype-homogeneity requirement in the contract

- File: `docs/src/backend_contract.md` (primitive section).
- `op_mul`/`op_add` are mandatory in-family; mixed-eltype operands are only
  handled for HSS via `_widen_eltype` (`src/hss_backend.jl:40-59,65-72`). The
  generic `op_mul(A::AbstractMatrix, B::AbstractMatrix) = A*B` MethodErrors on
  families without native promotion.
- Add one sentence: operands of binary primitives must share an eltype unless the
  backend explicitly widens (HSS does).
- Verify: sentence present; HSS widening tests still pass.

## 5. Run `cse` in the HSS `evaluate` override

- File: `src/hss_backend.jl:135-140`.
- The top-level `MulNode` fast path returns before `cse` runs, and
  `_eval_with` (`src/backend_interface.jl:445-453`) evaluates bindings with
  `foldl(op_mul, ...)`, losing balanced ordering inside bindings. The contract
  doc already tells fused backends to "run `cse` first".
- Change: in `evaluate(b::HssMatricesBackend, ex::AbstractOpNode)`, apply `cse`
  first, evaluate bindings (recursively through the balanced path), then the
  root; keep the balanced-chain fast path for the root `MulNode`.
- Verify: existing balanced-chain tests pass; add a test where a repeated
  subchain inside a pure-`MulNode` root is computed once (counting backend
  pattern from `backend_interface_tests.jl`).

## 6. Wire `op_norm` into `norm(::AbstractOperator)` or mark it optional

- File: `src/operators.jl:197-200` still does `norm(matrix(op)) / sqrt(prod(size))`,
  bypassing the mandatory `op_norm` primitive (only tests use it).
- Either route operator norm through `evaluate(cpr, op_norm-node)` /
  `op_norm(backend, matrix)` so structured families use `computeMatrixNorm`,
  or mark `op_norm` "optional, test-only" in `backend_contract.md`.
- Verify: `norm(::Kernel)` result unchanged within tolerance on HSS and dense.

## 7. Rewrite the docs around the single contract

- Files: `docs/src/backend_contract.md`, `docs/src/custom_compression.md`
  (rename/rewrite as the backend-authoring page), `docs/src/backend_interface_design.md`.
- Remove the "legacy compressions are bridged" framing and the "implement both
  contracts" reality; document one contract: backend subtype + core primitives +
  optional `evaluate` override, plus the acceptance test (`test_compression_interface`
  becomes a contract audit per migration step 6).
- State the eltype-homogeneity requirement (item 3 below) and the opacity status
  (item 7 below).
- Verify: a reader can implement a new backend from the docs alone, listing
  exactly one method set.

## 8. Port the package's call sites onto the single contract

- Files: `src/Kernels/kernels.jl` (constructors call `compression(axis, f)`),
  `src/Kernels/kernel_dag.jl` (`cpr(blocks)`, `recompress_inplace!`),
  `src/Kernels/kernel_solver.jl`, `src/operators.jl` (`compress!`),
  `src/compression.jl`, `src/triangularLowRankMatrix.jl`, `src/circulant_matrix.jl`.
- Replace every `c(...)` / `cpr(blocks)` call with `compress(backend(cpr), ...)`;
  replace `recompress_inplace!(cpr, m)` with a backend primitive; delete the
  legacy callable methods once no call site remains.
- Verify: `grep -rn "cpr(\|compression(\w*)(axis\|)(blocks" src/` returns nothing;
  full `Pkg.test()` green.

## 9. State the opacity status honestly

- Files: `docs/src/backend_interface_design.md` (principles), `src/backend_interface.jl`.
- Generic primitives are typed on `AbstractMatrix`; every shipped family is an
  `AbstractMatrix`. Either wrap families in an opaque `AbstractCompressedOperator`
  subtype now (breaking, allowed) or reword the design doc to "families are
  currently `AbstractMatrix`-based; opacity is the target".
- Verify: docs no longer promise opacity that the code doesn't deliver (or the
  wrapper exists and primitives dispatch on it).

## 10. One DAG-path test for the JLArrays demonstration extension

- File: `test/test_compression_interface.jl` (next to the existing
  JLArrayCompression testitem at line 27).
- The ext is only exercised through the legacy interface; after the port it must
  go through the single contract like any backend: add a testitem building a small
  DAG (`op_mul`/`op_add` over `CompressedLeaf`s of `JLArray`s) and evaluating via
  `evaluate(backend(cpr), ex)`.
- Verify: new testitem passes (requires `using JLArrays`).
- Note: demonstration-only per maintainer; keep this item last/optional.

## 11. `working_eltype` consistency check

- Files: `src/backend_interface.jl:102-108,363` — `backend(::AbstractCompression)`
  hardcodes `MatrixFamilyBackend{ComplexF64}()` regardless of stored eltype, so
  `working_eltype` can disagree with `scalartype` for non-ComplexF64 kernels.
- After the port, derive the working eltype from the representation/backend, and
  add a test asserting `working_eltype(backend(cpr))` matches the family for
  `NONCompression()`/`HssCompression()` on a Float32 kernel.
- Verify: test passes.

## 12. Clarify `SolveNode` failure mode

- Files: `src/backend_interface.jl:185-197` (docstring already honest),
  `docs/src/backend_contract.md` (solve strategies).
- `StructureOptimizingSolve` can be *built* into a `SolveNode` and only fails at
  evaluation. Fail fast in the `SolveNode` constructor for strategies the backend
  rejects.
- Verify: constructing `SolveNode(b, l, r, StructureOptimizingSolve(10, 1e-8))`
  with `HssMatricesBackend` throws immediately; test covers it.

## Deliberately out of scope

- Production GPU backend (Metal/CUDA) — JLArrays ext stays a demo.
- Mixed-precision backends actually consuming `working_eltype`.
- The `Threads.@threads` -> budgeted `@spawn` port (tracked in CLEANUP_PLAN.md).
- Merging PRs #55/#56; the cleanup plan in PR #57.
