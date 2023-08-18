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

using Test
using Symbolics
using SymbolicUtils
@variables x,y
Dx = Differential(x)
Dy = Differential(y)
@variables Gx(x)::Kernel
@variables Gy(y)::Kernel
expand_derivatives( Dy(Gx*Gy) )