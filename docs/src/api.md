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
solve_keldysh
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

## Observables

```@docs
lead_current
current_signal
same_time
keldysh_trace
```

## Operator algebra

```@docs
NonEquilibriumGreenFunction.Kernels.AbstractOperator
+(::NonEquilibriumGreenFunction.Kernels.AbstractOperator, ::NonEquilibriumGreenFunction.Kernels.AbstractOperator)
+(::NonEquilibriumGreenFunction.Kernels.AbstractOperator, ::UniformScaling)
+(::UniformScaling, ::NonEquilibriumGreenFunction.Kernels.AbstractOperator)
-(::NonEquilibriumGreenFunction.Kernels.AbstractOperator, ::NonEquilibriumGreenFunction.Kernels.AbstractOperator)
-(::NonEquilibriumGreenFunction.Kernels.AbstractOperator, ::UniformScaling)
-(::UniformScaling, ::NonEquilibriumGreenFunction.Kernels.AbstractOperator)
*(::DiracOperator, ::DiracOperator)
*(::DiracOperator, ::SimpleOperator)
*(::SimpleOperator, ::DiracOperator)
*(::SumOperator, ::SumOperator)
*(::SumOperator, ::Union{Number, UniformScaling, NonEquilibriumGreenFunction.Kernels.AbstractOperator})
*(::Union{Number, UniformScaling, NonEquilibriumGreenFunction.Kernels.AbstractOperator}, ::SumOperator)
==(::SimpleOperator, ::SimpleOperator)
==(::SumOperator, ::SumOperator)
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
