export SymbolicOperator

import TermInterface: maketerm, head, children, operation, arguments, isexpr, iscall

export maketerm, head, children, operation, arguments, isexpr, iscall, make_leaf

import Base: zero, one, isequal, log, inv
import LinearAlgebra: tr
import Base: +, -, *, /, adjoint

using Moshi.Match: @match
using Moshi.Data: @data
using Moshi.Derive: @derive

using Metatheory
using Metatheory.EGraphs
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
function maketerm(::Type{SymbolicExpression.Type}, head, args, metadata=nothing)
    operation = args[1]
    arguments = Vector{SymbolicExpression.Type}(args[2:end])
    return SymbolicExpression.Node(head, operation, arguments)
end

@testitem "SymbolicExpression TermInterface" begin
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
function wrap(::Type{SymbolicExpression.Type})
    return SymbolicOperator
end
function unwrap(::Type{SymbolicOperator})
    return SymbolicExpression.Type
end
wrap(x) = x
unwrap(x) = x

head(x::SymbolicOperator) = x |> unwrap |> head |> wrap
children(x::SymbolicOperator) = x |> unwrap |> children .|> wrap
operation(x::SymbolicOperator) = x |> unwrap |> operation |> wrap
arguments(x::SymbolicOperator) = x |> unwrap |> arguments .|> wrap
isexpr(x::SymbolicOperator) = x |> unwrap |> isexpr
iscall(x::SymbolicOperator) = x |> unwrap |> iscall
function maketerm(::Type{T}, head, args, metadata=nothing) where T<:SymbolicOperator
    wrap(maketerm(unwrap(T), head, unwrap.(args), metadata))
end

@testitem "SymbolicOperator TermInterface" begin
    using TermInterface
    using NonEquilibriumGreenFunction: wrap, unwrap
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
    @test iscall(kernel)
    @test !iscall(leaf_b)
    @test wrap(unwrap(kernel)) == kernel
end


function +(x::SymbolicOperator)
    return SymbolicOperator(:+, [x])
end
function +(x::SymbolicOperator, y::SymbolicOperator)
    if x isa Number && y isa Number
        return x + y
    else
        return SymbolicOperator(:+, [x, y])
    end
end
function +(x::Number, y::SymbolicOperator)
    return SymbolicOperator(:+, [x, y])
end
function +(x::SymbolicOperator, y::Number)
    return SymbolicOperator(:+, [x, y])
end

function -(x::SymbolicOperator)
    return SymbolicOperator(:-, [x])
end
function -(x::SymbolicOperator, y::SymbolicOperator)
    if x isa Number && y isa Number
        return make_leaf(x.expr - y.expr)
    else
        return SymbolicOperator(:-, [x, y])
    end
end
function -(x::Number, y::SymbolicOperator)
    return SymbolicOperator(:-, [x, y])
end
function -(x::SymbolicOperator, y::Number)
    return SymbolicOperator(:-, [x, y])
end

function *(x::SymbolicOperator, y::SymbolicOperator)
    if x isa Number && y isa Number
        return x * y
    else
        return SymbolicOperator(:*, [x, y])
    end
end
function *(x::Number, y::SymbolicOperator)
    return SymbolicOperator(:*, [x, y])
end
function *(x::SymbolicOperator, y::Number)
    return SymbolicOperator(:*, [x, y])
end

function inv(G::SymbolicOperator)
    return SymbolicOperator(:inv, [G])
end
function adjoint(G::SymbolicOperator)
    return SymbolicOperator(:adjoint, [G])
end
function log(G::SymbolicOperator)
    return SymbolicOperator(:log, [G])
end
function tr(G::SymbolicOperator)
    return SymbolicOperator(:tr, [G])
end
function (/)(left::SymbolicOperator, right::SymbolicOperator)
    return SymbolicOperator(:/, [left, right])
end

Broadcast.broadcastable(x::SymbolicOperator) = x

Base.zero(::SymbolicOperator) = zero(typeof(SymbolicOperator))
function Base.zero(::typeof(SymbolicOperator))
    return 0
end

@testitem "Test operations" begin
    using LinearAlgebra
    G = make_leaf(:G)
    Σ = make_leaf(:Σ)
    @test isequal(G + Σ, SymbolicOperator(:+, [G, Σ]))
    @test isequal(G * Σ, SymbolicOperator(:*, [G, Σ]))
    @test isequal(G - Σ, SymbolicOperator(:-, [G, Σ]))
    @test isequal(2 * G, SymbolicOperator(:*, [make_leaf(2), G]))
    @test isequal(make_leaf(2) * G, 2G)
    @test head(2G) == :call
end

simplify_kernel(expr) = _simplify_kernel(expr)
const symbolic_zero = make_leaf(0)
const symbolic_unity = make_leaf(0)
using Metatheory.Rewriters
function _simplify_kernel(expr)
    isNumber(x) = x isa Number
    th = @theory x y begin
        y + y => y * y
        #y - y => symbolic_zero
        #symbolic_zero - y => -y
        #symbolic_zero * y => symbolic_zero
        #inv(inv(x)) == x
    end
    graph = EGraph(expr)
    saturate!(graph, th)
    extracted = extract!(graph, astsize)
    return extracted
end

convert(::Type{SymbolicOperator}, x::Number) = make_leaf(x)
function Base.promote_rule(::Type{SymbolicOperator}, ::Type{K}) where K<:Number
    return SymbolicOperator
end

Base.display(A::SymbolicOperator) = error("Display not implemented yet for SymbolicOperator")

@testitem "Basic matching" begin
    using LinearAlgebra
    G = make_leaf(:G)
    @test G + G |> simplify_kernel == 2 * G
    @test 0 * G |> simplify_kernel == NonEquilibriumGreenFunction.symbolic_zero
    @test isequal(0 - G |> simplify_kernel, -G)
    @test zero(G) == 0
    A = [G G; G G]
    @test A isa Matrix{SymbolicOperator}
    @test isequal(tr(A) |> simplify_kernel, 2G)
    @test (A * A) isa Matrix{SymbolicOperator}
end

@testitem "simplification" begin
    #TODO: add detection of singular kernel ?
    using LinearAlgebra
    G = make_leaf(:G)
    Σ = make_leaf(:Σ)
    @test isequal(inv(inv(G)) |> simplify_kernel, G)
    @test isequal(simplify_kernel(G - G), NonEquilibriumGreenFunction.symbolic_zero)
    @test isequal(simplify_kernel(G + G + G - G), 2G)
end

@testitem "Construction of the current observable" begin
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
    @test diag(matrix(f(GR, GK, GR, GK))) isa Vector{ComplexF32}
    simplify_kernel(0 - (Σ_R * G_K + Σ_K * adjoint(G_R)))
end
