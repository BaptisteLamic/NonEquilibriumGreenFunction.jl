---
name: julia-bench
description: Load this skill when running, writing, or debugging benchmarks for NonEquilibriumGreenFunction.jl. Covers PkgBenchmark.jl, the benchmark/ suite and its six groups, the quick smoke suite, per-hardware baselines, regression judging, and updating baselines. Also load when the user asks to measure performance, compare against a baseline, or check for a complexity regression.
---

# Julia Benchmarking for NonEquilibriumGreenFunction.jl

This skill covers the benchmark infrastructure in `benchmark/`, which uses
[PkgBenchmark.jl](https://juliaci.github.io/PkgBenchmark.jl/stable/). All
commands assume the repo root as the working directory.

## Structure

```
benchmark/
  benchmarks.jl        # full suite: top-level `const SUITE = BenchmarkGroup()`
  quick_benchmarks.jl  # lightweight smoke suite (samples=1, N=32)
  run.jl               # one-liner runner (run + judge vs baseline)
  baselines/           # committed: one JSON per hardware kind
    <hardware-id>.json
  results/             # gitignored: per-run outputs (new.json, judge.md)
  tune.json            # gitignored: PkgBenchmark tuning cache
```

Benchmark dependencies (BenchmarkTools, PkgBenchmark) live in the **test
environment** (`test/Project.toml`) — there is no separate
`benchmark/Project.toml`. Always run with `--project=test`.

## Running benchmarks

```bash
# Run the full suite and compare against this machine's baseline
julia --project=test benchmark/run.jl

# (Re)create this machine's baseline
julia --project=test benchmark/run.jl --update-baseline

# Quick smoke suite (samples=1) — validates the infrastructure runs,
# does not measure performance
julia --project=test -e 'include("benchmark/quick_benchmarks.jl"); run_quick_benchmarks(verbose=true)'
```

## What run.jl does

1. Detects the machine's CPU model and slugifies it into a hardware id
   (e.g. `apple-m1-pro`).
2. Runs the full `SUITE` via `benchmarkpkg("NonEquilibriumGreenFunction")`
   and saves the result to `benchmark/results/new.json`.
3. With `--update-baseline`: copies the result to
   `benchmark/baselines/<hardware-id>.json` and exits.
4. Without it: compares against this machine's baseline
   (`judge` with the median estimator), prints the regression report and
   writes it to `benchmark/results/judge.md`.

If no baseline exists for the detected hardware, the runner warns — it does
**not** fall back to another hardware's baseline. Run with
`--update-baseline` first.

## Benchmark groups

`benchmark/benchmarks.jl` defines six groups:

- **discretization** — building `RetardedKernel`/`AdvancedKernel`/`AcausalKernel`
  (plus Dirac) over `N ∈ {64, 256}`, varying blocksize and compression
  scheme (`NONCompression`, `HssCompression`).
- **operations** — kernel algebra: addition, subtraction, scalar
  multiplication, kernel×kernel products (retarded×retarded,
  retarded×acausal, kernel×discretization and back), adjoint.
- **compression** — dense (`NONCompression`) vs HSS at several `atol`
  values.
- **solver** — `solve_dyson` end-to-end.
- **indexing** — block/time indexing utilities.
- **utils** — miscellaneous helpers.

## Baselines

Baselines are **per-hardware** and **committed** under `benchmark/baselines/`.
They are advisory references, not hard gates: machines with the same CPU
model can still differ by core count, RAM, and thermals.

To add or refresh a baseline for your machine:

```bash
julia --project=test benchmark/run.jl --update-baseline
git add benchmark/baselines/<hardware-id>.json
```

## When to run what

- After an algorithm change in kernel algebra, compression, or the
  solver: full suite (`benchmark/run.jl`) and check the judge report for
  regressions — the package claims quasi-linear complexity, and the
  benchmark suite is where an accidental dense fallback shows up.
- As a cheap smoke check (e.g. before pushing): the quick suite is already
  run as part of `Pkg.test()`.
- Never commit `benchmark/results/` or `tune.json` — they are gitignored.

## Gotchas

- The full suite takes several minutes (kernels at `N=64` and `N=256`
  across blocksizes and compression schemes); the first run also pays a
  one-time PkgBenchmark tuning cost cached in `tune.json`.
- Always `--project=test` — in the main environment PkgBenchmark is not
  installed.
- `--update-baseline` accepts this machine's current numbers as the new
  reference; use it deliberately, not to silence a regression.
