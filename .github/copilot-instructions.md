# Copilot instructions for NonEquilibriumGreenFunction.jl

This file contains concise, actionable guidance intended for AI coding agents working on this repository.

Focus: Julia package that solves the non-equilibrium Dyson equation with quasi-linear complexity.

Key files and layout
- `Project.toml` — package metadata and dependencies (Julia 1.10 compatibility).
- `README.md` — high-level purpose and example notebooks under `examples/`.
- `src/NonEquilibriumGreenFunction.jl` — main module; it loads most implementation files via `include(...)` and exports the public API.
- `src/` — core source. Notable subfolders:
  - `src/Kernels/` — kernel implementations and symbolic helpers (e.g. `symbolics_extension.jl`, `kernels.jl`).
  - several single-file modules included by the main module (e.g. `compression.jl`, `circulant_matrix.jl`, `operators.jl`).
- `test/` — tests; see `test/runtests.jl` and per-feature tests.
- `examples/` — Jupyter notebooks demonstrating usage (useful for reproducing examples).

What an AI agent should know (actionable rules)
- Project uses Julia package conventions. Always respect `Project.toml` for dependencies and compat bounds.
- Run and validate with the package environment: use `julia --project=@.` or `using Pkg; Pkg.activate(".")` before running REPL scripts or tests.
- Run tests with: `julia --project=@. -e 'using Pkg; Pkg.test()'` (this runs `test/runtests.jl`).
- The main module loads implementation via `include("...")`. To change public API, update `src/NonEquilibriumGreenFunction.jl` exports accordingly — do not add breaking exports silently.

Code and style patterns to follow
- Files are plain Julia scripts included into the module (not nested modules). Keep edits local to the file you touch and ensure `include` order remains valid.
- Naming: core module is `NonEquilibriumGreenFunction`. Many utilities are top-level functions exported by the main module (look at the `export` block in `src/NonEquilibriumGreenFunction.jl`).
- Kernels live in `src/Kernels/` — changes to symbolic/kernel logic often require updating both `symbolics_extension.jl` and `kernels.jl` together.
- The project uses `Symbolics` / `SymbolicUtils` in parts; preserve symbolic simplification patterns and types when editing symbolics-related code.

Testing and validation
- Add focused tests under `test/` when changing behavior. Tests are discovered via `test/runtests.jl`.
- Quick local verification: run `julia --project=@. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`.

Common tasks & examples
- To add a new exported function:
  - Implement in an existing file under `src/` (or create a new file and add an `include` into `src/NonEquilibriumGreenFunction.jl`).
  - Add the symbol to the `export` list in `src/NonEquilibriumGreenFunction.jl`.
  - Add a unit test in `test/` and run `Pkg.test()`.

- To change kernel symbolics behavior: edit `src/Kernels/symbolics_extension.jl` and update usages in `src/Kernels/kernels.jl`. Keep Symbolics transformation steps minimal and well-covered by tests.

Do not do
- Do not change the package UUID or version in `Project.toml` unless releasing a new package version.
- Do not remove `include(...)` lines from `src/NonEquilibriumGreenFunction.jl` without ensuring the code still loads and tests pass.

If unsure, run these quick checks
- `julia --project=@. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` — full test run.
- Open example notebooks in `examples/` to see typical usage and expected outputs.

If you need more context
- Read `src/NonEquilibriumGreenFunction.jl` (exports and includes) and `README.md` examples first.
- For symbolic/kernel work, inspect `src/Kernels/symbolics_extension.jl` and `src/Kernels/kernels.jl` together.

Please ask the maintainer (via PR comment) before making large API changes.
