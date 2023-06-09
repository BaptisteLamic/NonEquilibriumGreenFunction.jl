using SymbolicUtils: Symbolic, BasicSymbolic
import SymbolicUtils: similarterm
import Base: zero


function similarterm(x::Symbolic{Kernel}, head, args; metadata = nothing)
    similarterm(x, head, args, Kernel ; metadata = metadata)
end
zero(::Type{SymbolicUtils.BasicSymbolic{Kernel}}) = 0

function *(left::Symbolic{Kernel}, right::Symbolic{Kernel})
    similarterm(left, *, [left, right],)
end
function *(left::Number, right::Symbolic{Kernel})
    similarterm(right, *, [left, right], )
end
function *(left::Symbolic{Kernel}, right::Number)
    similarterm(left, *, [left, right], )
end
function +(left::Symbolic{Kernel}, right::Symbolic{Kernel})
    similarterm(left, +, [left, right], )
end
function +(left::Symbolic{Kernel}, right::Number)
    similarterm(left, +, [left, right], )
end
function +(left::Number, right::Symbolic{Kernel})
    similarterm(right, +, [left, right], )
end
function -(term::Symbolic{Kernel})
    similarterm(term, -, [term], )
end
function -(left::Symbolic{Kernel}, right::Symbolic{Kernel})
    similarterm(left, -, [left, right], )
end
function -(left::Symbolic{Kernel}, right::Number)
    similarterm(left, -, [left, right], )
end
function -(left::Number, right::Symbolic{Kernel})
    similarterm(right, -, [left, right], )
end
adjoint(kernel::Symbolic{Kernel}) = similarterm(kernel, adjoint, [kernel], Kernel)

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
