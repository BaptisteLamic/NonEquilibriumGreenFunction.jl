---
name: julia-docs
description: Load this skill when building, updating, or debugging the documentation for NonEquilibriumGreenFunction.jl. Covers the Documenter + Literate pipeline in docs/, the executed examples (mqdm, sqds, noise), the docs environment setup, and local builds. Also load when changing docs/lit/ example scripts, docs/src/ pages, or when an API change may break the examples.
---

# Documentation for NonEquilibriumGreenFunction.jl

The docs are built with [Documenter.jl](https://documenter.juliadocs.org/stable/)
and [Literate.jl](https://github.com/fredrikekre/Literate.jl). They are built
on every push, and the worked examples are **executed at documentation build
time**, so the published docs always show runnable code. An API change that
breaks an example breaks the docs build.

## Setup and build

From the repo root (see README.md):

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path=dirname(pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

- The first command registers the local package in the docs environment and
  installs its dependencies. Run it once, and again whenever
  `docs/Project.toml` changes.
- `Pkg.develop(path=dirname(pwd()))` is what makes the docs build against
  the local working copy rather than the registered release.

## Pipeline

`docs/make.jl`:

1. Executes the Literate examples in `docs/lit/` and generates markdown into
   `docs/src/generated/`:
   - `mqdm.jl` — Metal–QD–Metal junction (Green function + current)
   - `sqds.jl` — Superconductor–QD–Superconductor junction (Nambu)
   - `noise.jl` — current noise of a QD junction (two-time correlators)
2. Runs `makedocs` over the pages:
   - `src/index.md` (Home), `src/userguide.md`, the generated examples,
     `src/api.md`, `src/custom_compression.md`, `src/internals.md`.
3. Deploys via `deploydocs` (previews enabled; deploys happen on CI).

## Working on examples

- `docs/lit/*.jl` are plain Literate scripts: `#` lines become markdown,
  code chunks are executed in order at build time. Chunking matters —
  keep code blocks small so output interleaves correctly.
- Examples must run with the docs environment (`docs/Project.toml` deps:
  CairoMakie, DSP, LaTeXStrings, QuadGK, SpecialFunctions, StaticArrays,
  ...). A package used only inside an example must be added to
  `docs/Project.toml` — via `Pkg.add` in the docs env, never by hand-editing
  a Manifest.
- When changing a public API used by an example, update the example in the
  same change, and build the docs locally to confirm it still executes.
- Plots are produced by CairoMakie; in headless environments CairoMakie
  needs `Makie.set_theme!(...)` / a display-less backend — it is already
  configured non-interactive by default (no window) via CairoMakie.

## Checks after docs changes

1. Local build: `julia --project=docs docs/make.jl` — all three examples
   execute; watch for errors in the generated output.
2. Inspect `docs/src/generated/*.md` for mangled Literate chunking (code
   or output landing in the wrong block).
3. If pages or examples were added, update the `pages` list and `EXAMPLES`
   list in `docs/make.jl`.

## Gotchas

- The docs build executes the full examples — it is slow (minutes). Do not
  run it as part of a quick feedback loop; use the test suite instead.
- Never hand-edit files under `docs/src/generated/` — they are regenerated
  by Literate; edit the `docs/lit/*.jl` sources instead.
- `docs/Manifest.toml` is machine-generated — never edit it by hand; use
  `Pkg.develop` / `Pkg.add` / `Pkg.instantiate` in the docs env.
- On CI the docs deploy from the `main` branch (`devbranch="main"`); this
  working branch builds locally but will not deploy.
