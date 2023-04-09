
using Revise 
using Symbolics
using TestItems
using NonEquilibriumGreenFunction
@variables y::Kernel
expr = 0 - y*y
is_number(x) = x isa Number
#=
rules = SymbolicUtils.Prewalk( SymbolicUtils.Chain([
  #=  @rule 0 * ~x => 0
    @rule 0 + ~x => ~x
    @rule ~x + 0 => ~x=#
    @rule 0 - ~x => - ~x
  #=  @rule ~x - 0 => ~x
    @rule 1 * ~x => ~x
    @rule ~x * ~n::is_number => ~n * ~x
    @rule ~a::is_number * ~b::is_number * ~z => (~a * ~b) * ~z =#
]))
=#
rules = SymbolicUtils.Prewalk( @rule 0 - ~x => - ~x )

rules(expr) |> typeof