# NonEquilibriumGreenFunction

[![Build status (Github Actions)](https://github.com/BaptisteLamic/NonEquilibriumGreenFunction.jl/workflows/CI/badge.svg)](https://github.com/BaptisteLamic/NonEquilibriumGreenFunction.jl/actions)
[![codecov](https://codecov.io/gh/BaptisteLamic/NonEquilibriumGreenFunction.jl/branch/main/graph/badge.svg?token=BHAETIA0KL)](https://codecov.io/gh/BaptisteLamic/NonEquilibriumGreenFunction.jl)
[![DOI](https://zenodo.org/badge/623330633.svg)](https://zenodo.org/badge/latestdoi/623330633)

## Overview

This package solves the non-equilibrium Dyson equation in the time domain with quasi-linear time complexity. It is the research code accompanying:

1. The thesis: [Quantum transport in voltage-biased Josephson junctions](https://www.theses.fr/s210157#)
2. The paper: [Efficient computation of non-equilibrium Green's functions in the time domain](https://arxiv.org/html/2410.11057v1)

## Features

- Solves non-equilibrium Dyson equation in time domain
- Quasi-linear time complexity 

## Examples

### Metal - Quantum Dot - Metal Junction

The notebook `examples/MQDM_junction.ipynb` demonstrates how to compute the Green function of a non-interacting quantum dot connected to two leads and evaluate its current.

![Benchmark_QD_equilibrium](examples/QD_benchmark.svg)
![QD_Iavr](examples/average_current_QD.svg)

### Superconductor - Quantum Dot - Superconductor Junction

The notebook `examples/SQDS_junction.ipynb` shows how to compute the Green function of a non-interacting quantum dot connected to two superconducting leads and evaluate its current.

![QD_Iavr](examples/transient_current_SQDS.svg)

## Installation

```julia
using Pkg
Pkg.add("https://github.com/BaptisteLamic/NonEquilibriumGreenFunction.jl")
```