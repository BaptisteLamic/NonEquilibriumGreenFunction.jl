# Backend interface evolution plan (delegated implementation)

Companion to `CLEANUP_PLAN.md` (PR #57). This plan covers the **stage-2 backend
interface work** that the review of `src/backend_interface.jl`, `src/hss_backend.jl`,
`src/matrixinterface.jl`, `ext/JLArraysExt.jl` and the contract docs identified.
Each item is independently implementable and verifiable, sized for a flash model.

Context decisions (from review discussion):
- `JLArrayCompression` is a **demonstration** of the legacy matrix-family interface,
  not a production GPU backend. Items touching it are low priority / optional.
- **`Compression` may not be the right name anymore** — the type now means two things
  (a callable that builds matrices, and a tag carrying backend parameters). The
  rename below resolves this; until it lands, docs must describe both roles.

Scope guard: no silent breaking changes. Renames ship with deprecated aliases
(pattern already established by PR #49 `TrapzDiscretisation` -> `UniformDiscretisation`).

## 1. Add the `compress` primitive (or mark it stage-2 in the docs)

- Files: `src/backend_interface.jl`, `docs/src/backend_contract.md:139`,
  `docs/src/backend_interface_design.md` (mandatory-primitives table).
- The contract tables list `compress(b, sampler; ...)` / `compress(b, m)` as core
  primitives, but no `compress` function exists; compression goes through the
  legacy callable `c(axis, f)` / `c(m)` / `c(blocks)`.
- Preferred: add `compress(b::AbstractCompressionBackend, m; params...)` with a
  default that routes to the bridged legacy callable, so the numeric tier has the
  documented entry point. Alternative (smaller): mark the rows "stage 2, not yet
  in the contract" in both docs.
- Verify: grep finds a definition matching every contract-table row, or the
  "stage 2" label is present.

## 2. Rename `AbstractCompression` -> a backend-parameter/representation name

- Rationale: "compression" now covers both the callable matrix-builder and the
  parameter tag; the design doc's direction is backend-owned representation.
- Proposed: keep `AbstractCompression` as the **deprecated alias**, introduce
  e.g. `AbstractMatrixRepresentation` (or `AbstractBackendParams`) as the primary
  name for the callable subtype; `backend(c)` becomes `backend(r)`.
- Touch points: `src/matrixinterface.jl` (definition), `src/compression.jl`
  (subtypes `HssCompression`, `NONCompression`), `ext/JLArraysExt.jl`
  (`JLArrayCompression`), `src/backend_interface.jl` (bridge), exports in
  `src/NonEquilibriumGreenFunction.jl`, `docs/src/custom_compression.md`,
  `test/test_compression_interface.jl`, `test_compression_interface` (function
  name may stay or gain an alias).
- Verify: `@deprecate`-style alias keeps old code working; full `Pkg.test()` green;
  docs use the new name with a "formerly `AbstractCompression`" note.

## 3. State the eltype-homogeneity requirement in the contract

- File: `docs/src/backend_contract.md` (primitive section).
- `op_mul`/`op_add` are mandatory in-family; mixed-eltype operands are only
  handled for HSS via `_widen_eltype` (`src/hss_backend.jl:40-59,65-72`). The
  generic `op_mul(A::AbstractMatrix, B::AbstractMatrix) = A*B` MethodErrors on
  families without native promotion.
- Add one sentence: operands of binary primitives must share an eltype unless the
  backend explicitly widens (HSS does).
- Verify: sentence present; HSS widening tests still pass.

## 4. Run `cse` in the HSS `evaluate` override

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

## 5. Wire `op_norm` into `norm(::AbstractOperator)` or mark it optional

- File: `src/operators.jl:197-200` still does `norm(matrix(op)) / sqrt(prod(size))`,
  bypassing the mandatory `op_norm` primitive (only tests use it).
- Either route operator norm through `evaluate(cpr, op_norm-node)` /
  `op_norm(backend, matrix)` so structured families use `computeMatrixNorm`,
  or mark `op_norm` "optional, test-only" in `backend_contract.md`.
- Verify: `norm(::Kernel)` result unchanged within tolerance on HSS and dense.

## 6. Document the two parallel contracts explicitly

- File: `docs/src/backend_contract.md` ("Legacy compressions evaluate through
  backends" section) and `docs/src/custom_compression.md` NOTE.
- Today a backend author must implement BOTH the legacy callable
  (`c(axis, f; stationary=)`, `c(m)`, `c(blocks)`, `ldiv`, `to_cpu`,
  `same_time_blocks`, `recompress_inplace!`) AND the new primitive set; the bridge
  only goes one direction (`backend(c)`).
- Add a short "implementing a new backend today" subsection listing both sets and
  stating the migration plan steps 4-7 collapse them.
- Verify: a reader can list both required method sets from the docs alone.

## 7. Relax or implement the "opaque, not an AbstractMatrix" claim

- Files: `docs/src/backend_interface_design.md` (principles), `src/backend_interface.jl:255-262,380`.
- Generic primitives are typed on `AbstractMatrix` and
  `evaluate(c, m::AbstractMatrix) = m`; every real family is an `AbstractMatrix`.
- Either (small) reword the design doc to "families are currently
  `AbstractMatrix`-based; `AbstractCompressedOperator` is the stage-2 target", or
  (large, stage 2) wrap families in an opaque type. Do the reword now.
- Verify: docs no longer promise opacity that the code doesn't deliver.

## 8. One DAG-path test for the JLArrays demonstration extension

- File: `test/test_compression_interface.jl` (next to the existing
  JLArrayCompression testitem at line 27).
- The ext is only exercised through the legacy interface; add a testitem building a
  small DAG (`op_mul`/`op_add` over `CompressedLeaf`s of `JLArray`s) and evaluating
  via `evaluate(cpr, ex)` through the `MatrixFamilyBackend` bridge.
- Verify: new testitem passes (requires `using JLArrays`).
- Note: demonstration-only per maintainer; keep this item last/optional.

## 9. `working_eltype` consistency check

- Files: `src/backend_interface.jl:102-108,363` — `backend(::AbstractCompression)`
  hardcodes `MatrixFamilyBackend{ComplexF64}()` regardless of stored eltype, so
  `working_eltype` can disagree with `scalartype` for non-ComplexF64 kernels.
- Add a test asserting `working_eltype(backend(cpr))` matches the family for
  `NONCompression()`/`HssCompression()` on a Float32 kernel, or document the
  inertness more loudly in `scalartype`.
- Verify: test passes or docstring updated.

## 10. Clarify `SolveNode` failure mode

- Files: `src/backend_interface.jl:185-197` (docstring already honest),
  `docs/src/backend_contract.md` (solve strategies).
- `StructureOptimizingSolve` can be *built* into a `SolveNode` and only fails at
  evaluation. Add one line to the contract: "building a node with an unsupported
  strategy is allowed; evaluation throws `ArgumentError`" — or fail fast in the
  `SolveNode` constructor. Pick fail-fast if cheap.
- Verify: doc matches behavior; test covers it.

## Deliberately out of scope

- Production GPU backend (Metal/CUDA) — JLArrays ext stays a demo.
- Mixed-precision backends actually consuming `working_eltype`.
- The `Threads.@threads` -> budgeted `@spawn` port (tracked in CLEANUP_PLAN.md).
- Merging PRs #55/#56; the cleanup plan in PR #57.
