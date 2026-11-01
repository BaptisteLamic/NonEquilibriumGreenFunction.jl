# Internals [internal]

## How to read the DAG code

The package evaluates kernel expressions through two stacked DAGs:

1. **Kernel-level DAG** (`src/Kernels/kernel_dag.jl`): physics constructs
   (`KernelProd`, `SumOperator`) are lazy nodes that carry causality,
   quadrature and caching but do no numeric work. A forcing point
   (`matrix`, `solve_dyson`, the observables, `compress!`) calls
   `materialize`/`lower`, which turns the lazy tree into one concrete DAG.
2. **Numeric DAG** (`src/backend_interface.jl`): concrete nodes
   (`CompressedLeaf`, `AddNode`, `ScaleNode`, `MulNode`, `AdjointNode`,
   `SolveNode`, `BindingNode`) over already-compressed operators of a
   backend's family. `evaluate(b, ex)` runs it; the reference HSS backend
   (`src/hss_backend.jl`) overrides `evaluate` with balanced-chain fusion,
   `MatrixFamilyBackend` uses the default left-fold interpreter. `cse`
   extracts repeated subgraphs into `BindingNode`s so each is computed once.

So: edit kernel semantics in `src/Kernels/`, backend primitives in
`src/backend_interface.jl` / `src/hss_backend.jl`.

Docstrings for the internal machinery, kept out of the public API page.

```@autodocs
Modules = [NonEquilibriumGreenFunction]
Public = false
```

```@docs
NonEquilibriumGreenFunction.Kernels._discretize_uniformScaling
NonEquilibriumGreenFunction.Kernels.blocksize_and_eltype
NonEquilibriumGreenFunction.Kernels._quadrature_prod
NonEquilibriumGreenFunction.Kernels._boundary_blockdiag
```

```@autodocs
Modules = [NonEquilibriumGreenFunction.AdaptiveRichardson]
```
