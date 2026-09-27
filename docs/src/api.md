# API Reference

## Discretization

```@docs
AbstractDiscretisation
TrapzDiscretisation
discretize_retardedkernel
discretize_advancedkernel
discretize_acausalkernel
discretize_dirac
discretize_lowrank_kernel
```

## Solving

```@docs
solve_dyson
```

## Operators

```@docs
Kernel
DiracOperator
SumOperator
SimpleOperator
CompositeOperator
make_similar
compress!
adjoint
norm
```

## Accessors

```@docs
axis
blocksize
matrix
discretization
causality
compression
scalartype
```

## Indexing

```@docs
getindex
```

## Causality

```@docs
Retarded
Advanced
Acausal
Instantaneous
isretarded
isadvanced
isacausal
```

## Compression

```@docs
BlockCirculantMatrix
HssCompression
NONCompression
build_linearMap
build_CirculantlinearMap
```

## Utilities

```@docs
thermal_kernel
pauli
blockrange
blockindex
```
