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
