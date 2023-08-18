using Revise
using Test
using NonEquilibriumGreenFunction
using Symbolics
@variables x,y
Dx = Differential(x)
Dy = Differential(y)
@variables Gx(x)::Kernel
@variables Gy(y)::Kernel
Dy(Gy*Gx) |> expand_derivatives == Gx*Gy

t1 = Dy(Gy*Gx) |> expand_derivatives
t2 = Dy(Gy*Gx) |> expand_derivatives
==(t1,t2)