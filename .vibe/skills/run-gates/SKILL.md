---
name: run-gates
description: Load this skill before pushing or merging code to run all quality gates for NonEquilibriumGreenFunction.jl. Covers smoke load, Pkg.test (Aqua + JET + TestItems + quick-benchmark smoke), docs build, and the benchmark regression check against this machine's baseline. Also load when the user asks to "run gates", "run checks before push", "verify before push", or "run all quality checks".
---

# Run All Quality Gates for NonEquilibriumGreenFunction.jl

This skill runs every quality gate in sequence before a push. If any gate
fails, stop and report the failure — do not continue to the next gate or
attempt the push. All commands assume the repo root as the working
directory.

## Gates in order

### 1. Smoke load

Verify the package compiles and exports are accessible:

```bash
julia --project=@. -e 'using NonEquilibriumGreenFunction; println("smoke load OK")'
```

### 2. Full test suite

Aqua.jl quality checks, JET.jl top-level static analysis, all `@testitem`
suites, the legacy BlockCirculantMatrix tests, and the quick-benchmark
smoke test:

```bash
julia --project=@. -e 'using Pkg; Pkg.test()'
```

### 3. Docs build (when the change touches docs or example-facing API)

The Literate examples in `docs/lit/` are executed at build time, so this
gate catches API changes that would break the published examples:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path=dirname(pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Skip this gate only if the diff cannot affect the examples or docs (no API,
export, or `docs/` changes). When in doubt, run it.

### 4. Benchmark regression check (when the change may affect performance)

Only for changes to kernel algebra, compression, the solver, quadrature, or
anything on the hot path:

```bash
julia --project=test benchmark/run.jl
```

This compares the full suite against this machine's baseline
(`benchmark/baselines/<hardware-id>.json`) with the median estimator and
writes a report to `benchmark/results/judge.md`. Baselines are advisory
references, not hard gates — investigate judge results for the groups you
touched (discretization, operations, compression, solver) before deciding.

If no baseline exists for this machine, the runner will say so; create one
with `--update-baseline` (and commit the JSON) only if you accept the
current numbers as the reference.

## Report format

Report the outcome of each gate in order, for example:

```
1. smoke load: PASS
2. Pkg.test: PASS (Aqua, JET, TestItems, quick-bench smoke)
3. docs build: PASS (mqdm, sqds, noise executed)
4. benchmarks: SKIPPED (docs-only change)
```

## What not to do

- Do not push after a failed gate.
- Do not run the full benchmark suite for docs-only changes — it takes
  several minutes and measures nothing relevant.
- Do not update a baseline to silence a regression — only when intentionally
  accepting new performance numbers.
- Never hand-edit any Manifest.toml while setting up environments; use
  Pkg commands (`Pkg.develop`, `Pkg.instantiate`).
