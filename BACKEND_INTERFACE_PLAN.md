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
it, and 13 (the contract-audit suite) should exist before task 8 starts so the port
is regression-checked as it goes. Task 2 is **rename-only**: it renames types,
kwargs, files and exports but does **not** port call sites (that is task 8) — do not
combine the two, or the diff becomes unreviewable. Task 11 depends on the port and
must land **after task 8**, not earlier. Tasks 5-6 and 10-12 (minus 11) are
consistency/edge fixes that can land any time after 1-4.

**Testing policy (applies to every item below):** each task must land with its
tests in the same commit. Concretely:

- Every new/changed public function gets at least one `@testitem` in `test/`
  (or in `src/backend_interface_tests.jl` for backend-contract internals), run by
  `@run_package_tests`.
- Every backend primitive change must keep `test_compression_interface` green for
  `DenseRepresentation` (formerly `NONCompression`), `HssRepresentation`, and
  `JLArrayRepresentation` (the contract audit), plus the DAG tests in
  `src/backend_interface_tests.jl`.
- Every port (item 8) is verified by the **full suite** (`Pkg.test()`), not just
  the touched file — call-site ports are where silent breakage hides.
- Docs-only items still need a build check: `julia --project=docs docs/make.jl`
  with no new Documenter warnings.
- No test may be weakened (tolerance loosened, assertion deleted) to make a task
  pass; if a test legitimately changes meaning, say so in the commit message.
- `Verify:` lines below list the minimum test commands in addition to greps.

Context decisions (from review discussion):
- `JLArrayCompression` is a **demonstration** of the matrix-family interface,
  not a production GPU backend. Items touching it are low priority / optional.
- **Breaking changes are accepted** (pre-1.0). The old `AbstractCompression` name
  goes away outright — it now means two things (callable matrix-builder and
  parameter tag), which is exactly what the single-contract rework removes.
- **A backend is not necessarily a compression.** A dense GPU-array family (e.g.
  JLArray/CUDA dense) is a valid backend that compresses nothing. Naming must
  therefore describe *representation + evaluation strategy*, not compression:
  `HssCompression`/`NONCompression`/`JLArrayCompression` become representation
  tags (e.g. `HssRepresentation`, `DenseRepresentation`), and the word
  "compression" survives only where it is literally true (the `compress`
  primitive, HSS parameters `atol`/`rtol`/`kest`/`leafsize`).

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
- Tests: new `@testitem` asserting `compress(b, axis, f)` / `compress(b, m)` match
  the legacy callable output for `MatrixFamilyBackend` and `HssMatricesBackend`
  (dense reference, `atol` from the backend params); run `Pkg.test()`.

## 2. Rename the legacy types to representation/backend names (not "compression")

- Rationale: "compression" covers both the callable matrix-builder and the
  parameter tag today, and a backend need not compress at all — a dense GPU
  array family is a first-class backend. The single contract removes the duality
  and the name must stop implying compression.
- Introduce e.g. `AbstractMatrixRepresentation` as the name for what is today
  the `AbstractCompression` subtype, and **remove** `AbstractCompression`
  outright (breaking change accepted). Rename the concrete tags accordingly:
  `HssCompression` -> e.g. `HssRepresentation`, `NONCompression` ->
  `DenseRepresentation` (CPU dense), `JLArrayCompression` ->
  `JLArrayRepresentation`. They become pure parameter holders; their callable
  methods move to `compress(b, ...)` on the corresponding backend. The kwarg
  `compression=` on kernel constructors becomes e.g. `representation=` (or
  `backend=`).
- `backend(c)` becomes the accessor `backend(r)` returning the
  `AbstractCompressionBackend` (itself renamed if it still says "compression",
  e.g. `AbstractBackend`) carrying those parameters.
- Touch points: `src/matrixinterface.jl` (definition), `src/compression.jl`
  (subtypes + callables; **renamed to `src/representations.jl`** in this task),
  `ext/JLArraysExt.jl`, `src/backend_interface.jl` (bridge + abstract backend
  name), `src/Kernels/*` (kwarg only — call sites of `c(...)` are task 8),
  exports in
  `src/NonEquilibriumGreenFunction.jl`, `docs/src/custom_compression.md` ->
  rewrite as the backend-contract page, `test/test_compression_interface.jl`,
  `test/test_kernels.jl`, examples in `docs/lit/`.
- Verify: no remaining references to `AbstractCompression`/`*Compression`
  (grep across src/, ext/, test/, docs/, benchmark/); full `Pkg.test()` green;
  docs use the new names.
- Tests: rename-only churn is covered by the full suite; add one `@testitem` that
  a user-facing constructor works end to end with the new names
  (`RetardedKernel(...; representation=HssRepresentation())` -> `solve_dyson`
  -> `matrix`), and one test that a dense GPU-style representation (JLArray) is
  accepted wherever a representation is (proving "not necessarily a
  compression").

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
- Tests: extend `test_compression_interface` so each moved primitive
  (`blockdiag_blocks`, `to_cpu`, `same_time_blocks`, recompress) is asserted
  against the dense reference for every shipped compression; full `Pkg.test()`.

## 4. State the eltype-homogeneity requirement in the contract

- File: `docs/src/backend_contract.md` (primitive section).
- `op_mul`/`op_add` are mandatory in-family; mixed-eltype operands are only
  handled for HSS via `_widen_eltype` (`src/hss_backend.jl:40-59,65-72`). The
  generic `op_mul(A::AbstractMatrix, B::AbstractMatrix) = A*B` MethodErrors on
  families without native promotion.
- Add one sentence: operands of binary primitives must share an eltype unless the
  backend explicitly widens (HSS does).
- Verify: sentence present; HSS widening tests still pass.
- Tests: add a mixed-eltype `op_mul`/`op_add` case (Float64 x ComplexF64) to the
  HSS backend testitem, and a `@test_throws MethodError` case for the generic
  family without promotion.

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
- Tests: the new counting-`op_mul` testitem goes in `src/backend_interface_tests.jl`;
  assert `mulcount == 1` for the repeated subchain and value equality with the
  dense reference; keep the existing chain-length 3/4 tests untouched.

## 6. Mark `op_norm` optional, test-only (do not rewire `norm`)

- File: `src/operators.jl:197-200` still does `norm(matrix(op)) / sqrt(prod(size))`,
  bypassing the mandatory `op_norm` primitive (only tests use it).
- Decision: **demote, don't wire.** Rewiring `norm(::AbstractOperator)` would
  replace an exact Frobenius computation with randomized probing on dense
  (a user-facing value change) and, on HSS, compare against compressed
  intermediates where `rtol=1e-10` is unattainably tight. Mark `op_norm`
  "optional, test-only" in `backend_contract.md` and leave
  `norm(::AbstractOperator)` as is.
- Verify: `backend_contract.md` marks `op_norm` optional/test-only;
  `norm(::Kernel)` results unchanged on HSS and dense.
- Tests: none beyond the docs build check; the existing `op_norm` usage in
  tests must keep passing unchanged.

## 7. Rewrite the docs around the single contract

- Files: `docs/src/backend_contract.md`, `docs/src/custom_compression.md`
  (rename/rewrite as the backend-authoring page), `docs/src/backend_interface_design.md`.
- Remove the "legacy compressions are bridged" framing and the "implement both
  contracts" reality; document one contract: backend subtype + core primitives +
  optional `evaluate` override, plus the acceptance test (`test_compression_interface`
  becomes a contract audit per migration step 6).
- State the eltype-homogeneity requirement (task 4) and the opacity status
  (task 9).
- Verify: a reader can implement a new backend from the docs alone, listing
  exactly one method set.
- Tests: docs build check (`julia --project=docs docs/make.jl`, no new warnings);
  the mock-backend testitem in `src/backend_interface_tests.jl` must still
  implement exactly the documented method set — if it needs extra methods, the
  docs are wrong, fix the docs.

## 8. Port the package's call sites onto the single contract

- Files: `src/Kernels/kernels.jl` (constructors call `compression(axis, f)`),
  `src/Kernels/kernel_dag.jl` (`cpr(blocks)`, `recompress_inplace!`),
  `src/Kernels/kernel_solver.jl`, `src/operators.jl` (`compress!`),
  `src/compression.jl`, `src/triangularLowRankMatrix.jl`, `src/circulant_matrix.jl`.
- Replace every `c(...)` / `cpr(blocks)` call with `compress(backend(cpr), ...)`;
  replace `recompress_inplace!(cpr, m)` with a backend primitive; delete the
  legacy callable methods once no call site remains.
- Verify: `grep -rnE '\b(compression|cpr)\s*\(' src/` returns only the intended
  survivors (reviewed manually — it also matches `compression(op::SumOperator)`
  at `src/operators.jl:452`, which is the operator-level concept, not the
  callable); full `Pkg.test()` green.
- Tests: this is the big port — the **entire** suite (`Pkg.test()`, plus the
  circulant tests) is the acceptance gate; additionally add a
  `@testitem` per call-site file touched (`kernels.jl`, `kernel_dag.jl`,
  `kernel_solver.jl`, `operators.jl`) exercising the ported path against a dense
  reference.

## 9. State the opacity status honestly

- Files: `docs/src/backend_interface_design.md` (principles), `src/backend_interface.jl`.
- Generic primitives are typed on `AbstractMatrix`; every shipped family is an
  `AbstractMatrix`. Either wrap families in an opaque `AbstractCompressedOperator`
  subtype now (breaking, allowed) or reword the design doc to "families are
  currently `AbstractMatrix`-based; opacity is the target".
- Verify: docs no longer promise opacity that the code doesn't deliver (or the
  wrapper exists and primitives dispatch on it).
- Tests: if the opaque wrapper is introduced, every primitive needs a dispatch
  test per family (dense/HSS) in `src/backend_interface_tests.jl`; if only the
  doc is reworded, the docs build check suffices.

## 10. One DAG-path test for the JLArrays demonstration extension

- File: `test/test_compression_interface.jl` (next to the existing
  JLArrayCompression testitem at line 27).
- The ext is only exercised through the legacy interface; after the port it must
  go through the single contract like any backend: add a testitem building a small
  DAG (`op_mul`/`op_add` over `CompressedLeaf`s of `JLArray`s) and evaluating via
  `evaluate(backend(cpr), ex)`.
- Verify: new testitem passes (requires `using JLArrays`).
- Tests: the testitem itself is the deliverable: build
  `op_add(op_mul(CompressedLeaf(A), CompressedLeaf(B)), CompressedLeaf(C))` over
  `JLArray`s, evaluate via `evaluate(backend(cpr), ex)`, compare to the dense
  result with `rtol=1e-12`.
- Note: demonstration-only per maintainer; keep this item last/optional.

## 11. `working_eltype` consistency check

- Files: `src/backend_interface.jl:102-108,363` — `backend(::AbstractCompression)`
  hardcodes `MatrixFamilyBackend{ComplexF64}()` regardless of stored eltype, so
  `working_eltype` can disagree with `scalartype` for non-ComplexF64 kernels.
- After the port, derive the working eltype from the representation/backend, and
  add a test asserting `working_eltype(backend(cpr))` matches the family for
  `DenseRepresentation()`/`HssRepresentation()` on a Float32 kernel.
- Verify: test passes.
- Tests: `@testitem` asserting `working_eltype(backend(cpr)) == scalartype(k)`
  for a Float32 kernel under both `DenseRepresentation()` and
  `HssRepresentation()`.

## 12. Clarify `SolveNode` failure mode

- Files: `src/backend_interface.jl:185-197` (docstring already honest),
  `docs/src/backend_contract.md` (solve strategies).
- `StructureOptimizingSolve` can be *built* into a `SolveNode` and only fails at
  evaluation. Fail fast in the `SolveNode` constructor for strategies the backend
  rejects.
- Verify: constructing `SolveNode(b, l, r, StructureOptimizingSolve(10, 1e-8))`
  with `HssMatricesBackend` throws immediately; test covers it.
- Tests: `@test_throws ArgumentError SolveNode(...)` for the unsupported
  strategy, and a positive case that `DirectSolve()` still constructs and
  evaluates.

## 13. Contract-audit test suite for the single backend contract

- Files: `test/test_compression_interface.jl`, `src/test_compression_interface.jl`,
  `src/backend_interface_tests.jl`.
- Turn `test_compression_interface` into the contract audit described by
  migration step 6 of `docs/src/backend_interface_design.md`: (a) the semantic
  tier checked once against the dense backend, (b) per-backend numeric `evaluate`
  checked against the default interpreter on graphs covering every node type
  (add/sub/scale/mul-chain/adjoint/solve), with chain comparisons at combined
  tolerance.
- Add a meta-test that fails if a shipped backend lacks a method for any
  documented core primitive (dispatch check via `hasmethod`). To keep the
  meta-test from drifting from the docs, define the primitive list **once** as a
  `const` tuple (e.g. `CORE_PRIMITIVES = ((op_add, 2), (op_mul, 2), ...)`) in
  `src/backend_interface.jl` next to the abstract-backend definition; the
  meta-test iterates it, and the contract tables in `backend_contract.md`
  reference the same list by name. If the docs ever disagree with the tuple,
  the meta-test — not the prose — is the source of truth.
- Verify: full `Pkg.test()` green on the CI matrix (Julia 1.11 and latest 1.x,
  per `.github/workflows/ci.yml`; the package floor in `Project.toml` is 1.10),
  including the JLArray testitem.

## Deliberately out of scope

- Production GPU backend (Metal/CUDA) — JLArrays ext stays a demo.
- Mixed-precision backends actually consuming `working_eltype`.
- The `Threads.@threads` -> budgeted `@spawn` port (tracked in CLEANUP_PLAN.md).
- Merging PRs #55/#56; the cleanup plan in PR #57.
