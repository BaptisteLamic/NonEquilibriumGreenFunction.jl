using SymbolicUtils: Symbolic, BasicSymbolic
import SymbolicUtils: similarterm
import Base: zero, one


function similarterm(x::Symbolic{K}, head, args; metadata = nothing)  where K <: AbstractOperator
    similarterm(x, head, args, Kernel ; metadata = metadata)
end
zero(::Type{SymbolicUtils.BasicSymbolic{K}}) where K <: AbstractOperator = 0 
one(::Type{SymbolicUtils.BasicSymbolic{K}}) where K <: AbstractOperator = 1

function *(term::Symbolic{K}) where K <: AbstractOperator
    similarterm(term, *, [term],)
end
function *(left::Symbolic{K}, right::Symbolic{K}) where K <: AbstractOperator
    similarterm(left, *, [left, right],)
end
function *(left::Number, right::Symbolic{K}) where K <: AbstractOperator
    similarterm(right, *, [left, right], )
end
function *(left::Symbolic{K}, right::Number) where K <: AbstractOperator
    similarterm(left, *, [left, right], )
end
function +(left::Symbolic{K}, right::Symbolic{K}) where K <: AbstractOperator
    similarterm(left, +, [left, right], )
end
function +(left::Symbolic{K}, right::Number) where K <: AbstractOperator
    similarterm(left, +, [left, right], )
end
function +(left::Number, right::Symbolic{K}) where K <: AbstractOperator
    similarterm(right, +, [left, right], )
end
function -(term::Symbolic{K}) where K <: AbstractOperator
    similarterm(term, -, [term], )
end
function -(left::Symbolic{K}, right::Symbolic{K}) where K <: AbstractOperator
    similarterm(left, -, [left, right], )
end
function -(left::Symbolic{K}, right::Number) where K <: AbstractOperator
    similarterm(left, -, [left, right], )
end
function -(left::Number, right::Symbolic{K}) where K <: AbstractOperator
    similarterm(right, -, [left, right], )
end
adjoint(kernel::Symbolic{K}) where K <: AbstractOperator = similarterm(kernel, adjoint, [kernel], K) 

function simplify_kernel(expr)
    is_number(x) = x isa Number
    rules = RuleSet([
        @rule 0 * ~x => 0
        @rule 0 + ~x => ~x
        @rule ~x + 0 => ~x
        @rule 0 - ~x => - ~x
        @rule ~x - 0 => ~x
        @rule 1 * ~x => ~x
        @rule ~x * ~n::is_number => ~n * ~x
        @rule ~a::is_number * ~b::is_number * ~z => (~a * ~b) * ~z
    ])
    simplify(expr, rules)
end


@testitem "Symbolic differentiation" begin
    using Symbolics
    @variables x,y
    Dx = Differential(x)
    Dy = Differential(y)
    @variables Gx(x)::Kernel
    @variables Gy(y)::Kernel
    @test Dy(Gx) |> expand_derivatives == 0
    Dy(Gx*Gy) |> expand_derivatives
end
