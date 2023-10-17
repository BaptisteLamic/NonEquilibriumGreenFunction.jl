export SymbolicOperator

using SymbolicUtils: Symbolic, BasicSymbolic
using Symbolics: wrap

import Symbolics: unwrap


import SymbolicUtils: similarterm

import Base: zero, one, isequal, log, inv
import LinearAlgebra: tr


function similarterm(x::Symbolic{K}, head, args; metadata = nothing)  where K <: AbstractOperator
    similarterm(x, head, args, Kernel ; metadata = metadata)
end

@symbolic_wrap struct SymbolicOperator <: AbstractOperator
    val
end
unwrap(x::SymbolicOperator) = x.val
#=
wrapper_type(::Type{AbstractOperator}) = SymbolicOperator
symtype(a::SymbolicOperator) = AbstractOperator
=#

@wrapped function +(x::AbstractOperator)
     similarterm(x, +, [x],)
end 
@wrapped function +(x::AbstractOperator, y::AbstractOperator)
     similarterm(x, +, [x,y],)
end
@wrapped function +(x::Number, y::AbstractOperator)
    similarterm(y, +, [x,y],)
end
@wrapped function +(x::AbstractOperator, y::Number)
    similarterm(x, +, [x,y],)
end

@wrapped function -(x::AbstractOperator)
    similarterm(x, -, [x],)
end 
@wrapped function -(x::AbstractOperator, y::AbstractOperator)
    similarterm(x, -, [x,y],)
end
@wrapped function -(x::Number, y::AbstractOperator)
   similarterm(y, -, [x,y],)
end
@wrapped function -(x::AbstractOperator, y::Number)
   similarterm(x, -, [x,y],)
end

@wrapped function *(x::AbstractOperator)
    similarterm(x, *, [x],)
end 
@wrapped function *(x::AbstractOperator, y::AbstractOperator)
    similarterm(x, *, [x,y],)
end
@wrapped function *(x::Number, y::AbstractOperator)
   similarterm(y, *, [x,y],)
end
@wrapped function *(x::AbstractOperator, y::Number)
   similarterm(x, *, [x,y],)
end

@wrapped function inv(G::AbstractOperator)
    similarterm(G, inv, [G])
end
@wrapped function adjoint(G::AbstractOperator)
    similarterm(G, adjoint, [G])
end
@wrapped function log(G::AbstractOperator)
    similarterm(G, log, [G])
end
@wrapped function tr(G::AbstractOperator)
    similarterm(G, tr, [G])
end
@wrapped function (/)(left::AbstractOperator, right::AbstractOperator)
    return   similarterm(left, /, [left, right], )
end

@wrapped function simplify_kernel(expr::AbstractOperator)
    is_number(x) = x isa Number
    is_operator(::BasicSymbolic{K}) where K <: AbstractOperator = true  
    is_operator(::K) where K <: AbstractOperator = true  
    is_operator(x) = false  
    rules = [
        @acrule 0 * ~x => 0
        @acrule 0 + ~x => ~x
        @acrule ~x + 0 => ~x
        @rule 0 - ~x => - ~x
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

Base.isequal(a::SymbolicOperator,b::SymbolicOperator) = isequal(unwrap(a), unwrap(b))
convert(::Type{SymbolicOperator}, x::Number) = SymbolicOperator(x)
function Base.promote_rule(::Type{SymbolicOperator}, ::Type{K}) where K<:Number 
     return SymbolicOperator
end

Base.display(A::SymbolicOperator) = display(unwrap(A))

@testitem "Wrapper and promotion rule" begin
    using Symbolics
    using LinearAlgebra
    @variables Gx::Kernel
    @test 1 * Gx |> simplify_kernel == Gx
    @test 0 * Gx |> simplify_kernel == 0
    A = [Gx Gx; Gx Gx]
    @test_broken (A * A) isa Matrix{SymbolicOperator}
end

@testitem "simplification" begin
    #TODO: add detection of singular kernel ?
    using Symbolics
    using LinearAlgebra
    @variables η
    @variables G(η)::Kernel Σ(η)::Kernel
    @test isequal(inv(inv(G)) |> simplify_kernel, G)
    @test isequal(simplify_kernel( G - G), 0)
    @test isequal(simplify_kernel( G + G + G - G),2G)
end

@testitem "Construction of the current observable" begin
    using Symbolics
    using LinearAlgebra
    @variables x,y
    @variables G_R(x)::Kernel G_K(y)::Kernel
    @variables Σ_R::Kernel Σ_K::Kernel
    τz = [1 2; 2 3] // 2
    G = [0 G_R'; G_R G_K]
    Σl = [Σ_K Σ_R; Σ_R' 0]
    @test_broken expr = -tr(τz * (G * Σl - Σl * G)) isa SymbolicOperator
end
