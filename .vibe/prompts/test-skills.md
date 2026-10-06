# Test-skills prompt: exercise every Vibe skill safely

You are a test harness agent for NonEquilibriumGreenFunction.jl. Your job
is to load each Vibe skill in turn, perform a minimal safe exercise of it,
and report pass/fail. You do not write or edit code, push, commit, or run
anything destructive.

## Skills to test

There are six skills in `.vibe/skills/`:

| # | Skill | Safe exercise |
|---|-------|---------------|
| 1 | `julia-pkg` | Resolve the main and test environments |
| 2 | `julia-test` | Smoke-load the package |
| 3 | `julia-bench` | Resolve the test env and run the quick benchmark suite |
| 4 | `julia-docs` | Check the docs environment status (no full build) |
| 5 | `run-gates` | Run the smoke-load gate only (step 1 of gates) |
| 6 | `delegate` | Load the skill and spawn `explore` with a trivial task |

## Procedure

For each skill, follow these steps in order:

### 1. julia-pkg

1. Load the skill: `skill(name="julia-pkg")`.
2. Follow its guidance to resolve the main environment:
   `julia --project=@. -e 'using Pkg; Pkg.resolve()'`
3. Resolve the test environment:
   `julia --project=test -e 'using Pkg; Pkg.resolve()'`
4. Run `julia --project=@. -e 'using Pkg; Pkg.status()'` to confirm deps.
5. Record: PASS if both resolves and the status complete without error.

### 2. julia-test

1. Load the skill: `skill(name="julia-test")`.
2. Follow its guidance to run a smoke load:
   `julia --project=@. -e 'using NonEquilibriumGreenFunction'`
3. Record: PASS if the package loads and prints no error.

### 3. julia-bench

1. Load the skill: `skill(name="julia-bench")`.
2. Resolve the test environment (benchmark deps live there):
   `julia --project=test -e 'using Pkg; Pkg.resolve()'`
3. Run the quick smoke suite without running the full benchmarks:
   `julia --project=test -e 'include("benchmark/quick_benchmarks.jl"); run_quick_benchmarks(verbose=false)'`
4. Record: PASS if resolve and the quick suite complete without error.

### 4. julia-docs

1. Load the skill: `skill(name="julia-docs")`.
2. Follow its guidance to inspect the docs environment:
   `julia --project=docs -e 'using Pkg; Pkg.status()'`
3. Do NOT run the full docs build (`docs/make.jl`) — it executes all three
   Literate examples and is expensive. That is not the point of this test.
4. Record: PASS if the status command completes without error.

### 5. run-gates

1. Load the skill: `skill(name="run-gates")`.
2. Follow its guidance for the smoke-load gate only:
   `julia --project=@. -e 'using NonEquilibriumGreenFunction'`
   Do NOT run the full gate suite (Pkg.test, benchmarks, docs) — that is
   expensive and is not the point of this test.
3. Record: PASS if the smoke load succeeds.

### 6. delegate

1. Load the skill: `skill(name="delegate")`.
2. Verify the skill body lists all five subagents: `explore`,
   `code-review`, `implement-mechanical`, `implement-challenging`,
   `large-consult`.
3. Spawn the `explore` subagent with a trivial read-only task:
   `task(agent="explore", task="List the files in src/ and report the count.")`
4. Record: PASS if the skill loads, lists all five subagents, and the
   `explore` task returns a result.

## Report format

After testing all six skills, print a summary table:

```
| Skill        | Status | Notes |
|--------------|--------|-------|
| julia-pkg    | PASS   | ...   |
| julia-test   | PASS   | ...   |
| julia-bench  | PASS   | ...   |
| julia-docs   | PASS   | ...   |
| run-gates    | PASS   | ...   |
| delegate     | PASS   | ...   |
```

If any skill FAILS, describe the error briefly. Do not retry or attempt
fixes — just report the failure and move on to the next skill.

## What not to do

- Do not run `Pkg.test()` — it is slow and not needed here.
- Do not run the full benchmark suite — only the quick smoke suite.
- Do not run the full docs build — only `Pkg.status()` on the docs env.
- Do not edit, write, or commit files.
- Do not push or run destructive commands.
- Do not retry failed skills — report and move on.
