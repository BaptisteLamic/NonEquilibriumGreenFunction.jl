# Benchmarks

Performance benchmarks for `NonEquilibriumGreenFunction.jl`, using
[PkgBenchmark.jl](https://juliaci.github.io/PkgBenchmark.jl/stable/).

## Quick start

```bash
# Run the full suite and compare against this machine's baseline
julia --project=test benchmark/run.jl

# (Re)create this machine's baseline
julia --project=test benchmark/run.jl --update-baseline
```

The test environment (`test/Project.toml`) holds the benchmark dependencies
(`BenchmarkTools`, `PkgBenchmark`, `NonEquilibriumGreenFunction`, ...), so
always run with `--project=test`. No separate `benchmark/Project.toml` is
needed — PkgBenchmark runs the suite in the caller's active environment.

## What it does

`benchmark/run.jl`:

1. Detects the machine's CPU model (`Sys.cpu_info()`) and slugifies it into a
   hardware id (e.g. `apple-m1-pro`, `amd-ryzen-9-7950x`).
2. Runs the full `SUITE` defined in `benchmark/benchmarks.jl` via
   `benchmarkpkg`, and saves the result to `benchmark/results/new.json`.
3. **With `--update-baseline`**: copies the result to
   `benchmark/baselines/<hardware-id>.json` and exits. Run this the first time
   on a new machine, or when you intentionally accept new performance numbers.
4. **Without `--update-baseline`**: compares the run against this machine's
   baseline (`judge` with the median estimator), prints the regression report
   to stdout, and writes it to `benchmark/results/judge.md`.

If no baseline exists for the detected hardware, the runner warns and tells
you to run with `--update-baseline`. It does **not** fall back to another
hardware's baseline — comparing against a different machine would be
misleading.

## Layout

```
benchmark/
  benchmarks.jl       # Full suite: top-level `const SUITE = BenchmarkGroup()`.
                      # Consumed by PkgBenchmark. Defines six groups:
                      # discretization, operations, compression, solver,
                      # indexing, utils.
  quick_benchmarks.jl # Lightweight smoke suite (samples=1, N=32) used by the
                      # test suite to validate the benchmark code runs.
  run.jl              # One-liner runner (see Quick start).
  baselines/          # Committed: one JSON per hardware kind.
    <hardware-id>.json
  results/            # Gitignored: per-run outputs (new.json, judge.md).
  tune.json           # Gitignored: PkgBenchmark tuning cache.
```

## Baselines

Baselines are **per-hardware** and **committed** under `benchmark/baselines/`.
They are advisory references, not hard gates: machines with the same CPU model
can still differ by core count, RAM, frequency scaling, and thermals.

To add or refresh a baseline for your machine:

```bash
julia --project=test benchmark/run.jl --update-baseline
git add benchmark/baselines/<hardware-id>.json
git commit
```

## Smoke test

A fast smoke test (`run_quick_benchmarks` in `quick_benchmarks.jl`) runs as
part of the standard test suite via `test/runtests.jl`. It validates that the
benchmark infrastructure executes without measuring performance:

```bash
julia --project=@. -e 'using Pkg; Pkg.test()'
```

To run the quick suite standalone:

```bash
julia --project=test -e 'include("benchmark/quick_benchmarks.jl"); run_quick_benchmarks(verbose=true)'
```

## Notes

- The full suite takes several minutes (it builds kernels at `N=64` and `N=256`
  across blocksizes and compression schemes). The first run also pays a
  one-time PkgBenchmark tuning cost, cached in `benchmark/tune.json`.
- `results/` and `tune.json` are gitignored; `baselines/` is tracked.
