struct SymbolicOperator{C<:AbstractCausality} <: AbstractOperator
    expression
    causality::C
    axis
end

axis(g::SymbolicOperator) = g.axis
blocksize(g::SymbolicOperator) =  error("Not implemented error")
step(k::SymbolicOperator) = k |> axis |> step