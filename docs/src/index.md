# NonEquilibriumGreenFunction

Solving the non-equilibrium Dyson equation in the time domain with quasi-linear complexity.

This package is the research code accompanying:

1. The thesis: [Quantum transport in voltage-biased Josephson junctions](https://www.theses.fr/s210157#)
2. The paper: [Solving the Transient Dyson Equation with Quasilinear Complexity via Matrix Compression](https://arxiv.org/html/2410.11057v1)

## Overview

`NonEquilibriumGreenFunction.jl` solves the transient Dyson equation

```math
G = g + g \Sigma G
```

on a discretized time axis. Time-domain discretization turns the integral equation into a
linear system whose naive cost scales as ``\mathcal O(N^3)`` in the number of time steps.
By compressing the kernel matrices (HSS compression, or FFT-accelerated circulant
structure for stationary kernels), the cost drops to ``\mathcal O(N \log N)``.

## Typical use case

A simulation always follows the same pipeline, regardless of the physics:

1. **Discretize the kernels**: the bare retarded Green function `g`, the lead self-energies
   `Σ` (built from `InstantaneousKernel` for instantaneous terms and
   `RetardedKernel` / `AcausalKernel` for continuous ones), and the
   thermal occupation `ρ` (an acausal kernel).
2. **Solve the retarded Dyson equation**: `solve_dyson(g, g * Σ_R)` returns the full
   retarded Green function `G_R`.
3. **Dress the kinetic branch**: build the Keldysh self-energy `Σ_K` from the lead
   couplings and `ρ`, then `G_K = G_R * Σ_K * G_R'`.
4. **Extract observables**: combine `G_R`, `G_K` and the lead self-energies into
   currents or densities.

The two worked examples cover the two common flavours (see the *Examples* section of
the manual):

- [Metal - Quantum Dot - Metal junction](generated/mqdm.md): scalar (bs=1) kernels in
  equilibrium and under a voltage bias, including a complexity benchmark.
- [Superconductor - Quantum Dot - Superconductor junction](generated/sqds.md): Nambu
  (bs=2) kernels with superconducting leads, `Stationary` circulant compression,
  and the transient current response to a voltage ramp.

## Installation

```julia
using Pkg
Pkg.add("https://github.com/BaptisteLamic/NonEquilibriumGreenFunction.jl")
```

## Manual Outline

```@contents
Pages = ["userguide.md", "generated/mqdm.md", "generated/sqds.md", "api.md"]
Depth = 2
```

## Benchmarks

Performance benchmarks live in `benchmark/` and are managed with
[PkgBenchmark.jl](https://juliaci.github.io/PkgBenchmark.jl/stable/). See
`benchmark/README.md` for the per-hardware baselines and how to run them.
