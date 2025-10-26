export SymbolicOperator

using SymbolicUtils
import TermInterface: maketerm, head, children, operation, arguments, isexpr, iscall

export maketerm

import Base: zero, one, isequal, log, inv
import LinearAlgebra: tr

#=
ex = :(f(a, b))
@test head(ex) == :call
@test children(ex) == [:f, :a, :b]
@test operation(ex) == :f
@test arguments(ex) == [:a, :b]
@test isexpr(ex)
@test iscall(ex)
@test ex == maketerm(Expr, :call, [:f, :a, :b], nothing)


ex = :(arr[i, j])
@test head(ex) == :ref
@test_throws ErrorException operation(ex)
@test_throws ErrorException arguments(ex)
@test isexpr(ex)
@test !iscall(ex)
@test ex == maketerm(Expr, :ref, [:arr, :i, :j], nothing)
=#

struct SymbolicOperator <: AbstractOperator
    head::Symbol
    operation::Symbol
    args::Vector{SymbolicOperator}
end

function Base.:(==)(a::SymbolicOperator, b::SymbolicOperator)
    return a.head == b.head && a.operation == b.operation && a.args == b.args
end
function Base.hash(a::SymbolicOperator, h::UInt)
    return hash(a.head, hash(a.operation, hash(a.args, h)))
end

head(x::SymbolicOperator) = x.head
children(x::SymbolicOperator)::Vector{Union{SymbolicOperator, Symbol}} = [x.operation; x.args]
operation(x::SymbolicOperator) = x.operation
arguments(x::SymbolicOperator) = x.args
isexpr(::SymbolicOperator) = true
iscall(x::SymbolicOperator) = x.head == :call
function maketerm(::SymbolicOperator, head, args, metadata = nothing)
    return SymbolicOperator(head, args[1], args[2:end])
end
@testitem "SymbolicOperator maketerm" begin
    using Symbolics
    using TermInterface
    leaf_a = SymbolicOperator(:call, :a, [])
    @test leaf_a == SymbolicOperator(:call, :a, [])
    leaf_b = SymbolicOperator(:call, :b, [])
    kernel = SymbolicOperator(:call, :f, [leaf_a, leaf_b])
    @test head(kernel) == :call
    @test children(kernel) == [:f, SymbolicOperator(:call, :a, []), SymbolicOperator(:call, :b, [])]
    @test operation(kernel) == :f
    @test arguments(kernel) == [leaf_a, leaf_b]
    @test isexpr(kernel)
    @test iscall(kernel)
    @test kernel == maketerm(kernel, head(kernel), children(kernel), nothing)
end



function +(x::AbstractOperator)
    maketerm(x, +, [x],)
end
 function +(x::AbstractOperator, y::AbstractOperator)
    if x  isa Number && y isa Number
        x+y
    else
        if x isa Number
            maketerm(y, +, [x,y],)
        else 
            maketerm(x, +, [x,y],)
        end
    end
end
 function +(x::Number, y::AbstractOperator)
    maketerm(y, +, [x,y],)
end
 function +(x::AbstractOperator, y::Number)
    maketerm(x, +, [x,y],)
end

 function -(x::AbstractOperator)
    maketerm(x, -, [x],)
end
 function -(x::AbstractOperator, y::AbstractOperator)
    if x  isa Number && y isa Number
        x-y
    else
        if x isa Number
            maketerm(y, -, [x,y],)
        else 
            maketerm(x, -, [x,y],)
        end
    end
end
 function -(x::Number, y::AbstractOperator)
   maketerm(y, -, [x,y],)
end
 function -(x::AbstractOperator, y::Number)
   maketerm(x, -, [x,y],)
end



function *(x::AbstractOperator, y::AbstractOperator)
    if x  isa Number && y isa Number
        x*y
    else
        if x isa Number
            maketerm(y, *, [x,y],)
        else 
            maketerm(x, *, [x,y],)
        end
    end
end
 function *(x::Number, y::AbstractOperator)
   maketerm(y, *, [x,y],)
end
 function *(x::AbstractOperator, y::Number)
   maketerm(x, *, [x,y],)
end

 function inv(G::AbstractOperator)
    maketerm(G, inv, [G])
end
 function adjoint(G::AbstractOperator)
    maketerm(G, adjoint, [G])
end
 function log(G::AbstractOperator)
    maketerm(G, log, [G])
end
 function tr(G::AbstractOperator)
    maketerm(G, tr, [G])
end
 function (/)(left::AbstractOperator, right::AbstractOperator)
    return   maketerm(left, /, [left, right], )
end

Broadcast.broadcastable(x::SymbolicOperator) = x

Base.zero(::AbstractOperator) = zero(typeof(AbstractOperator))
function Base.zero(::typeof(AbstractOperator))
    return 0
end

simplify_kernel(expr) = _simplify_kernel(expr)
function _simplify_kernel(expr)
    is_number(x) = x isa Number
    is_operator(::SymbolicOperator) = true  
    is_operator(::K) where K <: AbstractOperator = true  
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

Base.isequal(a::SymbolicOperator,b::SymbolicOperator) = isequal(a.val, b.val)
convert(::Type{SymbolicOperator}, x::Number) = SymbolicOperator(x)
function Base.promote_rule(::Type{SymbolicOperator}, ::Type{K}) where K<:Number 
     return SymbolicOperator
end


Base.display(A::SymbolicOperator) = error("Display not implemented yet for SymbolicOperator")

@testitem "Wrapper and promotion rule" begin
    using Symbolics
    using LinearAlgebra
    @variables G::Kernel x
    @test 1 * G |> simplify_kernel == G
    @test 0 * G |> simplify_kernel == 0
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
    @variables G::Kernel Σ::Kernel
    @test isequal(inv(inv(G)) |> simplify_kernel, G)
    @test isequal(simplify_kernel( G - G), 0)
    @test isequal(simplify_kernel( G + G + G - G),2G)
end

@testitem "Construction of the current observable" begin
    using Symbolics
    using LinearAlgebra
    @variables G_R::Kernel G_K::Kernel
    @variables Σ_R::Kernel Σ_K::Kernel
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
