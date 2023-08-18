using SymbolicUtils: Symbolic, BasicSymbolic
import SymbolicUtils: similarterm
import Symbolics: expand_derivatives, unwrap
using Symbolics: wrap
import Base: zero, one, isequal, log, inv
import LinearAlgebra: tr

function expand_derivatives(O::BasicSymbolic{K}, simplify=false; occurrences=nothing) where K <: AbstractOperator 
    return _expand_derivatives(O,simplify; occurrences = occurrences) |> simplify_kernel
end
function _expand_derivatives(O::BasicSymbolic{K}, simplify=false; occurrences=nothing) where K <: AbstractOperator 
    if istree(O) && isa(operation(O), Differential)
        arg = only(arguments(O))
        arg = expand_derivatives(arg, false)

        if occurrences == nothing
            occurrences = Symbolics.occursin_info(operation(O).x, arg)
        end

        Symbolics._isfalse(occurrences) && return 0
        occurrences isa Bool && return 1 # means it's a `true`

        D = operation(O)

        if !istree(arg)
            return D(arg) # Cannot expand
        elseif (op = operation(arg); Symbolics.issym(op))
            inner_args = arguments(arg)
            if any(isequal(D.x), inner_args)
                return D(arg) # base case if any argument is directly equal to the i.v.
            else
                return sum(inner_args, init=0) do a
                    return expand_derivatives(Differential(a)(arg)) *
                           expand_derivatives(D(a))
                end
            end
        elseif op === (Symbolics.IfElse.ifelse)
            args = arguments(arg)
            O = op(args[1], D(args[2]), D(args[3]))
            return expand_derivatives(O, simplify; occurrences)
        elseif isa(op, Differential)
            # The recursive expand_derivatives was not able to remove
            # a nested Differential. We can attempt to differentiate the
            # inner expression wrt to the outer iv. And leave the
            # unexpandable Differential outside.
            if isequal(op.x, D.x)
                return D(arg)
            else
                inner = expand_derivatives(D(arguments(arg)[1]), false)
                # if the inner expression is not expandable either, return
                if istree(inner) && operation(inner) isa Differential
                    return D(arg)
                else
                    return expand_derivatives(op(inner), simplify)
                end
            end
       elseif isa(op, typeof(*))
            #custom section
            inner_args = arguments(arg)
            l = length(inner_args)
            @assert l == 2
           return expand_derivatives(D(inner_args[1]) * inner_args[2] +   inner_args[1] * D(inner_args[2]))
        elseif isa(op, typeof(+))
            #custom section
            inner_args = arguments(arg)
            l = length(inner_args)
            @assert l == 2
           return expand_derivatives(D(inner_args[1])) + expand_derivatives(D(inner_args[2]))
        elseif isa(op, typeof(-)) && length(arguments(arg)) == 2
            #custom section
            inner_args = arguments(arg)
           return expand_derivatives(D(inner_args[1])) - expand_derivatives(D(inner_args[2]))
        elseif isa(op, typeof(-)) && length(arguments(arg)) == 1
            #custom section
            inner_args = arguments(arg)
           return - expand_derivatives(D(inner_args[1]))
        else 
            return O
        end
    elseif !Symbolics.hasderiv(O)
        return O
    else
        args = map(a->expand_derivatives(a, false), arguments(O))
        O1 = operation(O)(args...)
        return simplify ? SymbolicUtils.simplify(O1) : O1
    end
end

function similarterm(x::Symbolic{K}, head, args; metadata = nothing)  where K <: AbstractOperator
    similarterm(x, head, args, Kernel ; metadata = metadata)
end
#zero(::SymbolicUtils.Symbolic{K}) where K <: AbstractOperator = 0 
#one(::SymbolicUtils.Symbolic{K}) where K <: AbstractOperator = 1


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

function *(left::Symbolic{K}, right::Symbolic{N}) where {K <: AbstractOperator, N <: Number}
    similarterm(left, *, [left, right], )
end

function *(left::Symbolic{N}, right::Symbolic{K}) where {K <: AbstractOperator, N <: Number}
    similarterm(right, *, [left, right], )
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

(inv)(G::Symbolic{K}) where K <: AbstractOperator = similarterm(G, inv, [ G], )
log(G::Symbolic{K}) where K <: AbstractOperator= similarterm(G, log, [G], )
tr(G::Symbolic{K}) where K <: AbstractOperator = similarterm(G, tr, [G], )

function (/)(left::Symbolic{K}, right::Symbolic{K}) where K <: AbstractOperator
    return   similarterm(left, /, [left, right], )
end

function simplify_kernel(expr)
    is_number(x) = x isa Number
    rules = [
        @rule 0 * ~x => 0
        @rule 0 + ~x => ~x
        @rule ~x + 0 => ~x
        @rule 0 - ~x => - ~x
        @rule ~x - 0 => ~x
        @rule 1 * ~x => ~x
        @rule -1 * ~x => - ~x
        @rule ~x * ~n::is_number => ~n * ~x
        @rule ~a::is_number * ~b::is_number * ~z => (~a * ~b) * ~z
    ] |> SymbolicUtils.Chain |> SymbolicUtils.Postwalk
    simplify(expr, rewriter = rules)
end

@symbolic_wrap struct SymbolicOperator <: AbstractOperator
    val
end
export SymbolicOperator
unwrap(x::SymbolicOperator) = x.val


function *(term::SymbolicOperator) 
   wrap(*(unwrap(term)))
end

function *(left::SymbolicOperator, right::SymbolicOperator) 
   wrap(unwrap(left) * unwrap(right))
end
function *(left::Number, right::SymbolicOperator) 
    wrap(left * unwrap(right))
end
function *(left::SymbolicOperator, right::Number) 
    wrap(unwrap(left)*unwrap(right))
end

function +(left::SymbolicOperator, right::SymbolicOperator)
    wrap(unwrap(left) + unwrap(right))
end
function +(left::Number, right::SymbolicOperator)
    wrap(unwrap(left) + unwrap(right))
end
function +(left::SymbolicOperator, right::Number)
    wrap(unwrap(left) + unwrap(right))
end

function -(term::SymbolicOperator)
    wrap(- unwrap(term))
end
function -(left::SymbolicOperator, right::SymbolicOperator)
    wrap(unwrap(left) - unwrap(right))
end
function -(left::SymbolicOperator, right::Number)
    wrap(unwrap(left) - unwrap(right))
end
function -(left::Number, right::SymbolicOperator)
    wrap(unwrap(left) - unwrap(right))
end
(inv)(G::SymbolicOperator)  = G |> unwrap |> inv |> wrap
function (/)(x::SymbolicOperator,G::SymbolicOperator) 
    return  wrap(unwrap(x)/unwrap(G))
end
log(G::SymbolicOperator) = G |> unwrap |> log |> wrap
tr(G::SymbolicOperator) = G |> unwrap |> tr |> wrap



zero(::Type{SymbolicOperator}) = SymbolicOperator(0)
one(::Type{SymbolicOperator}) = SymbolicOperator(1)
zero(::SymbolicOperator) = zero(SymbolicOperator)
one(::SymbolicOperator) = one(SymbolicOperator)

adjoint(term::SymbolicOperator) = wrap(adjoint(unwrap(term)))

(D::Differential)(expr::SymbolicOperator) = expr |> unwrap |> D |> wrap
simplify_kernel(expr::SymbolicOperator) = expr |> unwrap |> simplify_kernel |> wrap
Base.isequal(a::SymbolicOperator,b::SymbolicOperator) = isequal(unwrap(a), unwrap(b))
function expand_derivatives(O::SymbolicOperator, simplify=false; occurrences=nothing)
    return wrap(expand_derivatives(unwrap(O),simplify; occurrences = occurrences))
end
convert(::Type{SymbolicOperator}, x::Number) = SymbolicOperator(x)
function Base.promote_rule(::Type{SymbolicOperator}, ::Type{K}) where K<:Number 
     return SymbolicOperator
end

Base.display(A::SymbolicOperator) = display(unwrap(A))
#=function Base.show(io::IO, A::SymbolicOperator) 
    display(A)
end=#

@testitem "Wrapper and promotion rule" begin
    using Symbolics
    using LinearAlgebra
    @variables Gx::Kernel
    @test one(Gx) * Gx |> simplify_kernel == Gx
    @test zero(Gx) * Gx |> simplify_kernel == 0
end

@testitem "Symbolic differentiation" begin
    using Symbolics
    using LinearAlgebra
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
    @test isequal( Dx(tr(log(inv(Gx)))) |> expand_derivatives,  Dx(tr(log(inv(Gx)))) |> simplify_kernel)
end
