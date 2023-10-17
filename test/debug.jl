
using LinearAlgebra
using Revise
using NonEquilibriumGreenFunction
using Symbolics
@variables η
D = Differential(η)
@variables a::Kernel
expr = :(0*a + a)
simplify_kernel(expr)