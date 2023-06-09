
# NonEquilibriumGreenFunction
[![Build status (Github Actions)](https://github.com/BaptisteLamic/NonEquilibriumGreenFunction.jl/workflows/CI/badge.svg)](https://github.com/BaptisteLamic/NonEquilibriumGreenFunction.jl/actions)
[![DOI](https://zenodo.org/badge/623330633.svg)](https://zenodo.org/badge/latestdoi/623330633)



> :warning: **Package under heavy development/refactoring**

Research code accompanying the thesis: [Quantum transport in voltage-biased Josephson junctions](https://www.theses.fr/s210157#)
It solves the non-equilibrium Dyson equation in the time domain with quasi-linear time complexity.


## Example current accros a Metal - Quantum Dot - Metal junction.
The notebook ![QD](examples/MQDM_junction.ipynb) shows how to compute the Green function of a non-interacting quantum-dot connected to two leads and 
evaluate its current. 
![Benchmark_QD_equilibrium](examples/QD_benchmark.svg)
![QD_Iavr](examples/average_current_QD.svg)

## Example current accros a Superconductor - Quantum Dot - Superconductor junction.
The notebook ![QD](examples/MQDM_junction.ipynb) shows how to compute the Green function of a non-interacting quantum-dot connected to two superconducting leads and 
evaluate its current. 
![QD_Iavr](examples/transient_current_SQDS.svg)