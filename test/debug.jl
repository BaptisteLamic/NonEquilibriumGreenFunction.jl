using Revise
using NonEquilibriumGreenFunction
using Symbolics
using LinearAlgebra
@variables G_R::Kernel G_K::Kernel
@variables Σ_R::Kernel Σ_K::Kernel
τz = [1 2; 2 3] // 2
G = [0 G_R'; G_R G_K]
Σl = [Σ_K Σ_R; Σ_R' 0]
expr = simplify_kernel(-tr(τz * (G * Σl)))
expr isa SymbolicOperator
f = build_function(expr, G_R, G_K, Σ_R, Σ_K, expression=Val{false})