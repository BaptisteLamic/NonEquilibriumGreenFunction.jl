# API Reference

## Discretization

```@docs
AbstractDiscretisation
UniformDiscretisation
RetardedKernel
AdvancedKernel
AcausalKernel
AbstractKernelMap
Stationary
TwoTime
Separable
Singular
```

## Quadrature

```@docs
AbstractQuadrature
RectangleQuadrature
TrapezoidQuadrature
quadrature
singular_weights
```

## Solving

```@docs
solve_dyson
solve_keldysh
```

## Operators

```@docs
Kernel
LocalKernel
SumOperator
SimpleOperator
CompositeOperator
make_similar
compress!
adjoint
norm
```

## Backend interface and expression DAG

The numeric-tier backend contract and the lazy expression DAG. See the
[Backend contract](backend_contract.md) page for the full reference.

```@docs
AbstractCompressionBackend
AbstractCompressedOperator
HssMatricesBackend
working_eltype
structure
empty_like
op_add
op_sub
op_scale
op_mul
op_adjoint
op_norm
apply
solve
DirectSolve
StructureOptimizingSolve
CompressedLeaf
AbstractOpNode
SolveNode
evaluate
cse
evaluate_cse
BindingNode
with_thread_budget
current_thread_budget
ThreadConfig
DeviceConfig
```

## Kernel-level DAG

Lazy physics constructs materialized at the forcing points (`matrix`,
`solve_dyson`, observables, `compress!`); see the
[Backend contract](backend_contract.md) page.

```@docs
NonEquilibriumGreenFunction.Kernels.KernelProd
NonEquilibriumGreenFunction.Kernels.AbstractLazyOperator
NonEquilibriumGreenFunction.Kernels.materialize
NonEquilibriumGreenFunction.Kernels.lower
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
*(::LocalKernel, ::LocalKernel)
*(::LocalKernel, ::SimpleOperator)
*(::SimpleOperator, ::LocalKernel)
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
isretarded
isadvanced
isacausal
```

## Locality

```@docs
AbstractLocality
Local
Smooth
locality
islocal
locality_of_prod
locality_of_sum
```

## Kernel products

```@docs
prod(::Acausal, ::Acausal, ::AbstractDiscretisation, ::AbstractDiscretisation)
```

## Compression

```@docs
BlockCirculantMatrix
HssCompression
NONCompression
AbstractCompression
ldiv
to_cpu
same_time_blocks
recompress_inplace!
test_compression_interface
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
