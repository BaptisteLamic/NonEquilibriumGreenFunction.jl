# Examples

This directory contains Jupyter notebooks demonstrating the usage of NonEquilibriumGreenFunction.jl.

## Available Examples

- **MQDM_junction.ipynb**: Metal - Quantum Dot - Metal Junction
  - Demonstrates how to compute the Green function of a non-interacting quantum dot connected to two leads
  - Evaluates and visualizes the current through the system

- **SQDS_junction.ipynb**: Superconductor - Quantum Dot - Superconductor Junction
  - Shows how to compute the Green function of a non-interacting quantum dot connected to two superconducting leads
  - Evaluates and visualizes the transient current

## Setup

The examples require additional dependencies beyond the main package. To set up the examples environment:

```bash
cd examples
julia --project=@. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
```

This will:
1. Link to the local NonEquilibriumGreenFunction package in the parent directory
2. Install all additional dependencies required by the notebooks
3. Create a `Manifest.toml` file with pinned dependency versions

## Running the Examples

### Option 1: From the command line
```bash
# Navigate to the examples directory
cd examples

# Start Julia with the examples environment
julia --project=@.

# In the Julia REPL, you can now open notebooks with:
using IJulia
notebook()
```

### Option 2: Using Jupyter directly
Make sure to set the kernel to use the examples environment:
```bash
# First, add the examples environment as a Jupyter kernel
julia --project=@. -e 'using IJulia; installkernel("NonEquilibriumGreenFunction Examples")'
```

Then select the "NonEquilibriumGreenFunction Examples" kernel when opening the notebooks.

## Dependencies

The examples use the following packages in addition to NonEquilibriumGreenFunction:

- **BenchmarkTools**: For performance benchmarking
- **CairoMakie**: For plotting and visualization
- **DSP**: For digital signal processing utilities
- **LaTeXStrings**: For LaTeX-formatted labels in plots
- **QuadGK**: For numerical integration
- **Revise**: For live code reloading during development
- **StaticArrays**: For static array operations

All dependencies from the main package (FFTW, HssMatrices, LinearAlgebra, NNlib, SparseArrays, SpecialFunctions, Statistics, StatsBase, TestItems) are automatically available through the parent package.
