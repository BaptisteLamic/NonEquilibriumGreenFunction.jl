using NonEquilibriumGreenFunction
using Symbolics
using LinearAlgebra
using Test
@variables x,y
Dx = Differential(x)
Dy = Differential(y)
@variables Gx(x)::Kernel
@variables Gy(y)::Kernel
@test Dy(Gx) |> expand_derivatives == 0
@test isequal(Dy(Gx*Gy) |> expand_derivatives, Gx*Dy(Gy))
@test isequal(Dy(Gy*Gx) |> expand_derivatives, Dy(Gy)*Gx)
@test !isequal(Dy(Gy*Gx) |> expand_derivatives, Gx*Dy(Gy))
@test isequal(Dy(Gy + Gx) |> expand_derivatives, Dy(Gy))
@test isequal(Dy(-Gy)|> expand_derivatives, - Dy(Gy)|> expand_derivatives)
# We do not want the default derivation rules to spill our calculation
@test isequal( Dx(tr(log(inv(Gx)))) |> expand_derivatives, Dx(tr(log(inv(Gx)))) ) 

A = [Gx Gx; Gx Gx] 
*(A,A)