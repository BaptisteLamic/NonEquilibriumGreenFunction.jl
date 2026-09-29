# NonEquilibriumGreenFunction

[![Build status (Github Actions)](https://github.com/BaptisteLamic/NonEquilibriumGreenFunction.jl/workflows/CI/badge.svg)](https://github.com/BaptisteLamic/NonEquilibriumGreenFunction.jl/actions)
[![codecov](https://codecov.io/gh/BaptisteLamic/NonEquilibriumGreenFunction.jl/branch/main/graph/badge.svg?token=BHAETIA0KL)](https://codecov.io/gh/BaptisteLamic/NonEquilibriumGreenFunction.jl)
[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://BaptisteLamic.github.io/NonEquilibriumGreenFunction.jl/)
[![DOI](https://zenodo.org/badge/623330633.svg)](https://zenodo.org/badge/latestdoi/623330633)

## Overview

This package solves the non-equilibrium Dyson equation in the time domain with quasi-linear time complexity. It is the research code accompanying:

1. The thesis: [Quantum transport in voltage-biased Josephson junctions](https://www.theses.fr/s210157#)
2. The paper: [Solving the Transient Dyson Equation with Quasilinear Complexity via Matrix Compression](https://arxiv.org/html/2410.11057v1)

## Features

- Solves the transient Dyson equation $G = g + g \Sigma G$ in the time domain
- Quasi-linear $\mathcal{O}(N \log N)$ time complexity via kernel-matrix compression (HSS compression, or FFT-accelerated circulant compression for stationary kernels)
- Typed kernel maps for local, retarded, acausal and singular self-energy terms, with kernel algebra (`+`, `-`, `*`, composition)
- Works with scalar and Nambu (block) kernels, e.g. for superconducting leads
- Explicit compression/matrix interface with support for [JLArrays.jl](https://github.com/JuliaGPU/JLArrays.jl)

## Documentation

The full documentation, including the API reference and the two worked examples, is
hosted at <https://BaptisteLamic.github.io/NonEquilibriumGreenFunction.jl/>.

The documentation is built with [Documenter.jl](https://documenter.juliadocs.org/stable/) and
[Literate.jl](https://github.com/fredrikekre/Literate.jl) on every push. The two worked
examples (Metal–QD–Metal and Superconductor–QD–Superconductor junctions) are executed at
documentation build time, so the published docs always show runnable code.

Build locally with:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path=dirname(pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Examples

The worked examples live as Literate scripts in `docs/lit/` and are rendered (and executed)
in the published documentation:

- **Metal – Quantum Dot – Metal junction** ([`docs/lit/mqdm.jl`](docs/lit/mqdm.jl)): Green
  function of a non-interacting quantum dot connected to two metal leads, a complexity
  benchmark, and the average current under a voltage bias.
- **Superconductor – Quantum Dot – Superconductor junction** ([`docs/lit/sqds.jl`](docs/lit/sqds.jl)):
  Green function of a quantum dot connected to two superconducting leads (Nambu kernels,
  stationary circulant compression) and the transient current response to a voltage ramp.

![Benchmark_QD_equilibrium](examples/QD_benchmark.svg)
![QD_Iavr](examples/average_current_QD.svg)
![QD_Iavr](examples/transient_current_SQDS.svg)

## Benchmarks

Performance benchmarks live in `benchmark/` and are managed with
[PkgBenchmark.jl](https://juliaci.github.io/PkgBenchmark.jl/stable/). See
[`benchmark/README.md`](benchmark/README.md) for per-hardware baselines and how to run them.

## Installation

```julia
using Pkg
Pkg.add("https://github.com/BaptisteLamic/NonEquilibriumGreenFunction.jl")
```
