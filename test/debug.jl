using Revise
using Test
using NonEquilibriumGreenFunction
using Symbolics
@variables x,y::Kernel
Dx = Differential(x)
Dy = Differential(y)
@variables Gx(x)::Kernel
@variables Gy(y)::Kernel
@variables G::Kernel
@test Dy(Gx) |> expand_derivatives == 0
r = Dx(Gx*Gy)
expand_derivatives( r ) 