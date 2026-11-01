# Cleanup plan (delegated implementation)

Tracking document for the code/docs review of `main` (HEAD `8c5cf22`, PR #54) plus
open PR #56. Each item is independently implementable, verifiable, and sized for a
small ("flash") model. Do them in order; each is a separate commit on this branch.

Scope guard: docs/CI/hygiene only. No behavior changes, no API changes, no new deps.

## 1. Docs: add the two backend pages to the site nav

- File: `docs/make.jl`, `pages=` (lines 25-35).
- Add a `"Backend" => ["backend_contract.md", "backend_interface_design.md"]` entry.
- Why: both pages exist and are cross-linked from `index.md`, `api.md`,
  `custom_compression.md`, but are missing from `pages=`, so the built site warns
  and the links dangle.
- Verify: `julia --project=docs docs/make.jl` builds without "not included in
  pages" warnings.

## 2. Docs: fix stale `stationary=true` advice

- File: `docs/src/userguide.md:133` ("Prefer `stationary=true` ...").
- Replace with advice to use a `Stationary` map: "Prefer a `Stationary(f)` map
  whenever the physics allows — the circulant path is the fastest construction."
- Why: since PR #44 the public constructors take typed maps; there is no
  `stationary` keyword on `RetardedKernel`/`AcausalKernel`.
- Verify: grep shows no remaining `stationary=true` in user-facing docs
  (`custom_compression.md:31` documents the internal callable and may keep it).

## 3. Docs: update `custom_compression.md` NOTE to past tense

- File: `docs/src/custom_compression.md:3-8`.
- Reword "is proposed" / "will be rewritten as part of that migration" to state the
  first stage is merged (PR #54) and this page documents the legacy interface still
  in production use.
- Verify: consistent with `backend_interface_design.md`'s header.

## 4. Docs: reconcile the backend contract with the code

- Files: `docs/src/backend_contract.md` (line ~139 primitive table),
  `docs/src/backend_interface_design.md` (mandatory-primitives table).
- The tables list `compress` as a core/mandatory primitive, but no `compress`
  function exists in `src/backend_interface.jl`; compression goes through the
  legacy `c(...)` callables. Mark `compress` as "stage 2, not yet in the contract"
  in both tables.
- Also mark `MatrixFamilyBackend` and `backend` as not-yet-exported in the contract
  page (they are used in tests as `NEGF.MatrixFamilyBackend`; exports live in
  `src/NonEquilibriumGreenFunction.jl:71-72`). Either export them (code change —
  out of scope for this pass, flag it) or document them as internal.
- Verify: every name in the contract tables has a matching definition or an
  explicit "not yet implemented" label.

## 5. Docs: document the exported-but-undocumented `theq_lesser_time_kernel`

- File: `docs/src/api.md` — add `theq_lesser_time_kernel` under "Utilities".
- It is exported (`src/NonEquilibriumGreenFunction.jl:85`) but appears in no
  `@docs` block (`@autodocs Public=false` in `internals.md` excludes exports).
- Verify: `api.md` entry resolves to the docstring in `src/Physics/physics.jl`.

## 6. Docs: add short docstrings for the thin `@docs` entries

- Files: `src/backend_interface.jl` — `CompressedLeaf`, `BindingNode`,
  `op_add`/`op_sub`/`op_scale` (one-liners are fine, but `CompressedLeaf` and
  `BindingNode` render empty in `api.md`).
- Fix the stale "randomized probing" comment on `op_norm`
  (`src/backend_interface.jl:232-235`; the implementation is exact `norm(A)`).
- Verify: `api.md` entries all render non-empty.

## 7. CI: bump deprecated actions

- File: `.github/workflows/ci.yml` — `actions/checkout@v3` -> `@v4`,
  `actions/cache@v3` -> `@v4` (test job), `codecov/codecov-action@v3` -> `@v5`.
- Why: v3 actions run on Node 16, deprecated; PR #56's body already flags codecov.
- Verify: `actionlint` or a workflow dry-run if available; otherwise YAML lint.

## 8. CI: align matrix with the declared compat bound

- Same file, line ~19: matrix tests `'1.11'` labeled "Minimum Julia version" while
  `Project.toml` declares `julia = "1.10"`. Change to `'1.10'`.
- This is exactly what draft PR #56 does — merge #56 instead of duplicating it.
- Note: keep this out of this PR if #56 lands first.

## 9. Tests: remove the dead loop variable; drop the flaky absolute tolerance

- Files: `test/runtests.jl:26` (delete `T = Float32` inside the eltype loop),
  `test/test_kernels.jl:173` (delete the absolute `norm(...) < 1E-10`, keep the
  relative check on the next line).
- Again, these are exactly PR #56 — merge #56 rather than duplicating.

## 10. Repo hygiene: untrack `.DS_Store` and `.vscode`

- Run: `git rm --cached .DS_Store .vscode/settings.json`
- File: `.gitignore` — add `/.DS_Store` (the existing `/.vscode` entry is
  ineffective for already-tracked files).
- Verify: `git status` clean after; `git ls-files | grep -i ds_store` empty.

## 11. Docs: figure-output location consistency

- Files: `docs/lit/mqdm.jl:265`, `docs/lit/sqds.jl:227` save SVGs into
  `docs/lit/` via `joinpath(@__DIR__, ...)`, while `examples/README.md` says
  committed figures live in `examples/*.svg`.
- Either save to `examples/` or add `docs/lit/*.svg` to `.gitignore`.
- Verify: docs build leaves no untracked files in `docs/lit/`.

## 12. Contributor docs: refresh `AGENTS.md`

- Add `src/backend_interface.jl`, `src/hss_backend.jl`,
  `src/Kernels/kernel_dag.jl` to the layout; mention the forcing points
  (`matrix`, `solve_dyson`, observables, `compress!`) and the thread-safety
  contract (disjoint trees only; shared nodes must be materialized first).
- Verify: matches `docs/src/backend_contract.md`.

## 13. Readability: add a minimal runnable example

- Files: `docs/src/index.md` and/or `docs/src/userguide.md`.
- Add a ~15-line snippet that runs end to end: axis, `RetardedKernel` with a
  `Stationary` map, `LocalKernel` self-energy, `solve_dyson`, `matrix`.
- Replace the placeholder workflow sketch (`g_func`, `Σ_R_func`, `I_avr = ...`)
  with either real runnable code or an explicit pointer to the examples.
- Verify: paste the snippet into a REPL with the package loaded; it runs.

## 14. Readability: short "how to read the DAG code" section

- File: `docs/src/internals.md` (currently 19 lines).
- Add a short orientation: kernel-level DAG (`KernelProd`/`SumOperator` in
  `src/Kernels/kernel_dag.jl`) -> `lower` -> numeric DAG nodes
  (`src/backend_interface.jl`) -> `evaluate` (`src/hss_backend.jl` for HSS).
- Verify: every file name referenced exists.

## Deliberately out of scope (larger, track separately)

- Port remaining `Threads.@threads` (`src/compression.jl:27,113`,
  `src/triangularLowRankMatrix.jl:159,177,195`) to budgeted `@spawn` per the
  threading contract — migration step 1 of `backend_interface_design.md`.
- Move `@testitem` blocks out of `src/` so `Test`/`TestItems` drop from `[deps]`
  to `[extras]` (flagged in PR #56).
- Global RNG seeding in tests.
- HSS `evaluate` running `cse` at the top-level `MulNode` and binding evaluation
  using the balanced path (`src/hss_backend.jl:135-140`).
- Exporting `MatrixFamilyBackend`/`backend`, adding a `compress` shim, ASCII
  identifiers in `src/`/`test/`, renaming `NONCompression`/`build_CirculantlinearMap`.
- `same_time(::SumOperator)` fusing through the DAG (`src/operators.jl:546`).
- Merging open PR #55 (trapezoid boundary docs fix) and PR #56 (test fixes).
