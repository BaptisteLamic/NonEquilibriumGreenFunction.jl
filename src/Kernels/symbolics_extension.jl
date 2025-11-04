export SymbolicOperator

using SymbolicUtils
import TermInterface: maketerm, head, children, operation, arguments, isexpr, iscall

export maketerm, make_leaf

import Base: zero, one, isequal, log, inv
import LinearAlgebra: tr
import Base: +, -, *, /, adjoint

using Moshi.Match: @match
using Moshi.Data: @data
using Moshi.Derive: @derive
@data SymbolicExpression begin
    struct Node
        head
        operation #TODO: make that type stable
        args::Vector{SymbolicExpression.Type}
    end
    Leaf(Any)
end
@derive SymbolicExpression[Hash, Eq, Show]

function head(x::SymbolicExpression.Type)
    @match x begin
        SymbolicExpression.Node(head, _, _) => head
        SymbolicExpression.Leaf(_) => error("Leaf has no head")
    end
end
function children(x::SymbolicExpression.Type)
    @match x begin
        SymbolicExpression.Node(_, operation, args) => return [operation; args]
        SymbolicExpression.Leaf(_) => error("Leaf has no children")
    end
end
function operation(x::SymbolicExpression.Type)
    @match x begin
        SymbolicExpression.Node(_, operation, _) => operation
        SymbolicExpression.Leaf(_) => error("Leaf has no operation")
    end
end
function arguments(x::SymbolicExpression.Type)
    @match x begin
        SymbolicExpression.Node(_, _, arguments) => arguments
        SymbolicExpression.Leaf(_) => error("Leaf has no arguments")
    end
end
function isexpr(x::SymbolicExpression.Type)
    @match x begin
        SymbolicExpression.Node(_, _, _) => true
        SymbolicExpression.Leaf(_) => false
    end
end
function iscall(x::SymbolicExpression.Type)
    @match x begin
        SymbolicExpression.Node(head, _, _) => head == :call
        SymbolicExpression.Leaf(_) => false
    end
end
function maketerm(::Type{SymbolicExpression.Type}, head, args, metadata = nothing)
    operation = args[1]
    arguments::Vector{SymbolicExpression.Type} = args[2:end]
    return SymbolicExpression.Node(head, operation, arguments)
end

@testitem "SymbolicExpression TermInterface" begin
    using Symbolics
    using TermInterface
    using NonEquilibriumGreenFunction.SymbolicExpression
    leaf_a = SymbolicExpression.Leaf(:a)
    leaf_b = SymbolicExpression.Leaf(:b)
    kernel = SymbolicExpression.Node(:call, *, [leaf_a, leaf_b])
    @test kernel == SymbolicExpression.Node(:call, *, [leaf_a, leaf_b])
    @test head(kernel) == :call
    @test children(kernel) == [*, leaf_a, leaf_b]
    @test operation(kernel) == *
    @test arguments(kernel) == [leaf_a, leaf_b]
    @test isexpr(kernel)
    @test !isexpr(leaf_a)
    @test iscall(kernel)
    kernel_bis = maketerm(typeof(kernel), head(kernel), children(kernel), nothing)
    @test head(kernel_bis) == head(kernel)
    @show children(kernel_bis)
    @test children(kernel_bis) == children(kernel)
    @test operation(kernel_bis) == operation(kernel)
    @test arguments(kernel_bis) == arguments(kernel)
    @test kernel == kernel_bis
end

struct SymbolicOperator
    expr::SymbolicExpression.Type
end

function SymbolicOperator(operation, args::Vector{SymbolicOperator})
    expression = SymbolicExpression.Node(:call, operation, unwrap.(args))
    return SymbolicOperator(expression)
end

function (==)(a::SymbolicOperator, b::SymbolicOperator)
    return a.expr == b.expr
end

function make_leaf(value)
    return SymbolicOperator(SymbolicExpression.Leaf(value))
end

function wrap(x::SymbolicExpression.Type)
    return SymbolicOperator(x)
end
function unwrap(x::SymbolicOperator)
    return x.expr
end
function unwrap(::Type{SymbolicOperator})
    return SymbolicExpression.Type
end
wrap(x) = x
unwrap(x) = x

head(x::SymbolicOperator) = x |> unwrap |> head  |> wrap
children(x::SymbolicOperator) = x |> unwrap |> children .|> wrap
operation(x::SymbolicOperator) = x |> unwrap |> operation |> wrap
arguments(x::SymbolicOperator) = x |> unwrap |> arguments .|> wrap
isexpr(x::SymbolicOperator) = x |> unwrap |> isexpr
iscall(x::SymbolicOperator) = x |> unwrap |> iscall
function maketerm(::Type{T}, head, args, metadata = nothing) where T <: SymbolicOperator
      wrap(maketerm(unwrap(T), head, unwrap.(args), metadata))
end

@testitem "SymbolicOperator TermInterface" begin
    using Symbolics
    using TermInterface
    leaf_a = make_leaf(:a)
    @test leaf_a == make_leaf(:a)
    leaf_b = make_leaf(:b)
    kernel = SymbolicOperator(*, [leaf_a, leaf_b])
    @test head(kernel) == :call
    @test children(kernel) == [*, leaf_a, leaf_b]
    @test operation(kernel) == *
    @test arguments(kernel) == [leaf_a, leaf_b]
    @test isexpr(kernel)
    @test iscall(kernel)
    @test kernel == maketerm(typeof(kernel), head(kernel), children(kernel), nothing)
end


function +(x::SymbolicOperator)
    SymbolicOperator(+, [x])
end
 function +(x::SymbolicOperator, y::SymbolicOperator)
    if x  isa Number && y isa Number
        x+y
    else
        if x isa Number
            SymbolicOperator(+, [x,y])
        else 
            SymbolicOperator(+, [x,y])
        end
    end
end
 function +(x::Number, y::SymbolicOperator)
    SymbolicOperator(+, [x,y])
end
 function +(x::SymbolicOperator, y::Number)
    SymbolicOperator(+, [x,y])
end

 function -(x::SymbolicOperator)
    SymbolicOperator(-, [x])
end
 function -(x::SymbolicOperator, y::SymbolicOperator)
    if x  isa Number && y isa Number
        x-y
    else
        if x isa Number
            SymbolicOperator(-, [x,y])
        else 
            SymbolicOperator(-, [x,y])
        end
    end
end
 function -(x::Number, y::SymbolicOperator)
   SymbolicOperator(-, [x,y])
end
 function -(x::SymbolicOperator, y::Number)
   SymbolicOperator(-, [x,y])
end

function *(x::SymbolicOperator, y::SymbolicOperator)
    if x  isa Number && y isa Number
        x*y
    else
        if x isa Number
            SymbolicOperator(*, [x,y])
        else 
            SymbolicOperator(*, [x,y])
        end
    end
end
 function *(x::Number, y::SymbolicOperator)
   SymbolicOperator(*, [x,y])
end
 function *(x::SymbolicOperator, y::Number)
   SymbolicOperator(*, [x,y])
end

 function inv(G::SymbolicOperator)
    SymbolicOperator(inv, [G])
end
 function adjoint(G::SymbolicOperator)
    SymbolicOperator(adjoint, [G])
end
 function log(G::SymbolicOperator)
    SymbolicOperator(log, [G])
end
 function tr(G::SymbolicOperator)
    SymbolicOperator(tr, [G])
end
 function (/)(left::SymbolicOperator, right::SymbolicOperator)
    return SymbolicOperator(/, [left, right])
end

Broadcast.broadcastable(x::SymbolicOperator) = x

Base.zero(::SymbolicOperator) = zero(typeof(SymbolicOperator))
function Base.zero(::typeof(SymbolicOperator))
    return 0
end

simplify_kernel(expr) = _simplify_kernel(expr)
function _simplify_kernel(expr)
    is_number(x) = x isa Number
    is_operator(::K) where K <: SymbolicOperator = true  
    is_operator(x) = false  
    rules = [
        @acrule 0 * ~x => 0
        @acrule 0 + ~x => ~x
        @acrule ~x + 0 => ~x
        @rule 0 - ~x => -(~x)
        @rule ~x - 0 => ~x
        @rule ~x - ~x => 0
        @rule 1 * ~x => ~x
        @rule -1 * ~x => - ~x
        @rule -( ~a + ~b) => - (~a) - (~b) 
        @rule ~c - ( ~a + ~b) => ~c - (~a) - (~b) 
        @rule -( ~a - ~b) => - (~a) + (~b) 

        @rule ~x * ~n::is_number => ~n * ~x
        @acrule ~n::is_number * ~x + ~m::is_number * ~x  => (~n+m) * ~x
        @rule ~x + ~x  => 2 * ~x
        @rule ~n::is_number * ~x + ~x  => (~n+1) * ~x
        @rule ~n::is_number * ~x - ~x  => (~n-1) * ~x
        @rule ~a::is_number * ~b::is_number * ~z => (~a * ~b) * ~z
        @rule ~n::is_number * (~x + ~y)  =>  ~n * ~x + ~n* ~y

        @rule inv(inv(~a::is_operator)) => ~a 
    ] |> SymbolicUtils.Chain |> SymbolicUtils.Postwalk |>  SymbolicUtils.Fixpoint
    simplify(expr, rewriter = rules)
end

convert(::Type{SymbolicOperator}, x::Number) = make_leaf(x)
function Base.promote_rule(::Type{SymbolicOperator}, ::Type{K}) where K<:Number 
     return SymbolicOperator
end

Base.display(A::SymbolicOperator) = error("Display not implemented yet for SymbolicOperator")

@testitem "Basic matching" begin
    using Symbolics
    using LinearAlgebra
    @variables x
    G = make_leaf(:G)
    expr = 1 * G 
    @test expr |> simplify_kernel == G
    expr = 0 * G 
    @test expr |> simplify_kernel == 0
    @test isequal(0 - G |> simplify_kernel, -G)
    @test zero(G) == 0
    A = [G G; G G]
    @test A isa Matrix{SymbolicOperator}
    @test isequal(tr(A) |> simplify_kernel, 2G )
    @test (A * A) isa Matrix{SymbolicOperator}
end

@testitem "simplification" begin
    #TODO: add detection of singular kernel ?
    using Symbolics
    using LinearAlgebra
    G = make_leaf(:G)
    Σ = make_leaf(:Σ)
    @test isequal(inv(inv(G)) |> simplify_kernel, G)
    @test isequal(simplify_kernel( G - G), 0)
    @test isequal(simplify_kernel( G + G + G - G),2G)
end

@testitem "Construction of the current observable" begin
    using Symbolics
    using LinearAlgebra
    G_R = make_leaf(:G_R)
    G_K = make_leaf(:G_K)
    Σ_R = make_leaf(:Σ_R)
    Σ_K = make_leaf(:Σ_K)
    τz = [1 2; 2 3] // 2
    G = [0 G_R'; G_R G_K]
    Σl = [Σ_K Σ_R; Σ_R' 0]
    expr = simplify_kernel(-tr(τz * (G * Σl - Σl * G)))
    @test expr isa SymbolicOperator
    bs, N, Dt = 2, 128, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    A = randn(ComplexF32, bs * N, bs * N)
    GR = RetardedKernel(ax, A, bs, NONCompression())
    GK = AcausalKernel(ax, A, bs, NONCompression())
    f = build_function(expr, G_R, G_K, Σ_R, Σ_K, expression=Val{false})
    @test diag(matrix(f(GR,GK,GR,GK))) isa Vector{ComplexF32}
    simplify_kernel(0 - (Σ_R*G_K + Σ_K*adjoint(G_R)))
end
