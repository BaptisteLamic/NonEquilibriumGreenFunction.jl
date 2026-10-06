---
name: julia-test
description: Load this skill when running, writing, or debugging Julia tests for NonEquilibriumGreenFunction.jl. Covers Pkg.test, @testitem/TestItems, Aqua.jl quality checks, JET.jl static analysis, the quick-benchmark smoke test, and the test environment. Also load when a test is failing or the user asks to verify changes.
---

# Julia Testing for NonEquilibriumGreenFunction.jl

This skill covers running and writing tests for the package using Julia's
standard testing infrastructure. All commands assume the repo root as the
working directory.

## Running tests

```bash
# Full test suite (resolves test/Project.toml automatically)
julia --project=@. -e 'using Pkg; Pkg.test()'

# Smoke load (verify the package compiles and exports work)
julia --project=@. -e 'using NonEquilibriumGreenFunction; println("OK")'
```

## Test architecture

```
test/
  runtests.jl                    # entry point
  Project.toml                  # test-only deps
  test_kernels.jl               # @testitem blocks (kernel algebra, discretization)
  test_layers.jl                # @testitem blocks (layers/operators)
  test_adaptive_richardson.jl   # @testitem blocks (AdaptiveRichardson)
  test_compression_interface.jl # @testitem blocks (compression contract)
  test_aqua.jl                  # Aqua.jl quality checks
  test_jet.jl                    # JET.jl static analysis
  test_BlockCirculantMatrix.jl  # legacy plain tests (manual include)
```

`test/runtests.jl` does three things:

1. Runs every `@testitem` block via `@run_package_tests`
   (TestItemRunner/TestItems). Files containing `@testitem` blocks are
   auto-discovered — no manual `include` needed.
2. Manually includes the legacy plain-test file
   `test_BlockCirculantMatrix.jl` (no `@testitem`, so not auto-discovered).
3. Runs a benchmark smoke test: `@testitem "benchmarks_quick.jl"` includes
   `benchmark/quick_benchmarks.jl` and asserts `run_quick_benchmarks(verbose=false)`
   completes without warnings.

## Writing tests with @testitem

Tests use the TestItems.jl macro `@testitem`, which is auto-discovered by
`@run_package_tests`:

```julia
@testitem "Discretisation creation and accessor" begin
    using LinearAlgebra
    bs, N, Dt = 2, 128, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    A = randn(ComplexF64, bs * N, bs * N)
    dA = TrapzDiscretisation(ax, A, bs, NONCompression())
    @test matrix(dA) == A
end
```

- Each `@testitem` runs in its own scope: `using` inside the block is local
  to it.
- New test files containing only `@testitem` blocks need no registration;
  just create the file under `test/`.
- Only legacy plain-test files (with bare `@testset`/`@test` outside
  `@testitem`) need a manual `include` in `test/runtests.jl`.

## Aqua.jl checks

`test/test_aqua.jl` runs the **full** `Aqua.test_all(NonEquilibriumGreenFunction)`
with no disabled checks. A new dependency or export that violates Aqua
(ambiguities, stale deps, undefined exports, compat mismatches) will fail
the suite.

## JET.jl analysis

`test/test_jet.jl` runs `report_package(NonEquilibriumGreenFunction)` and
asserts that `result.res.toplevel_error_reports` is empty. Inference error
reports are not asserted — they may originate from external packages.
Per AGENTS.md, test only for top-level errors. On incompatible Julia
versions the test skips itself (`JET.JET_AVAILABLE` guard).

## Debugging a failing test

1. Run the full suite to see which `@testitem` fails:
   `julia --project=@. -e 'using Pkg; Pkg.test()'`
2. If Aqua fails, identify the sub-check (ambiguities, undefined_exports,
   deps_compat, piracy, stale_deps, ...); these are usually real issues to
   fix in the package, not the test.
3. If JET fails, read the top-level error reports — usually missing methods
   or type instability introduced by the change.
4. If a `@testitem` fails, shrink the reproduction: a smaller axis
   (`LinRange` with small `N`), scalar blocksize `bs=1`, and
   `NONCompression()` keeps the case minimal.
5. The quick-benchmark smoke test failing usually means the benchmark
   infrastructure itself broke (see the `julia-bench` skill).

## Gotchas

- `Pkg.test()` builds a temporary environment from `test/Project.toml`;
  test-only deps go there, not in the top-level `Project.toml`.
- `@testitem` blocks are isolated: `using NonEquilibriumGreenFunction`
  inside the block is needed to use the package in new test files.
- Plain-test files without `@testitem` will silently not run unless
  included manually from `test/runtests.jl` — check that a new legacy-style
  test file is actually being executed.
- The suite includes the quick-benchmark smoke test: a failure there is a
  benchmark-infrastructure failure, not necessarily a package failure.
