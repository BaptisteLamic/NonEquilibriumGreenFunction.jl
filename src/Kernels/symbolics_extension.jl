export SymbolicOperator

using SymbolicUtils: Symbolic, BasicSymbolic
using Symbolics: wrap

import Symbolics: unwrap

import SymbolicUtils: similarterm
import Base: zero, one, isequal, log, inv
import LinearAlgebra: tr


function similarterm(x::Symbolic{K}, head, args; metadata = nothing)  where K <: AbstractOperator
    similarterm(x, head, args, AbstractOperator ; metadata = metadata)
end

@symbolic_wrap struct SymbolicOperator <: AbstractOperator
    val
end
Base.:(==)(a::SymbolicOperator, b::SymbolicOperator) = unwrap(a) == unwrap(b)
Base.:hash(a::SymbolicOperator) = hash(a.val)
unwrap(x::SymbolicOperator) = x.val
#=wrapper_type(::Type{AbstractOperator}) = SymbolicOperator
symtype(::SymbolicOperator) = Kernel=#


@wrapped function +(x::AbstractOperator)
    similarterm(x, +, [x],)
end
@wrapped function +(x::AbstractOperator, y::AbstractOperator)
    if x  isa Number && y isa Number
        x+y
    else
        if x isa Number
            similarterm(y, +, [x,y],)
        else 
            similarterm(x, +, [x,y],)
        end
    end
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
    if x  isa Number && y isa Number
        x-y
    else
        if x isa Number
            similarterm(y, -, [x,y],)
        else 
            similarterm(x, -, [x,y],)
        end
    end
end
@wrapped function -(x::Number, y::AbstractOperator)
   similarterm(y, -, [x,y],)
end
@wrapped function -(x::AbstractOperator, y::Number)
   similarterm(x, -, [x,y],)
end



@wrapped function *(x::AbstractOperator, y::AbstractOperator)
    if x  isa Number && y isa Number
        x*y
    else
        if x isa Number
            similarterm(y, *, [x,y],)
        else 
            similarterm(x, *, [x,y],)
        end
    end
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

SymbolicUtils.simplify(n::SymbolicOperator; kw...) = wrap(SymbolicUtils.simplify(unwrap(n); kw...))
SymbolicUtils.simplify_fractions(n::SymbolicOperator; kw...) = wrap(SymbolicUtils.simplify_fractions(unwrap(n); kw...))
SymbolicUtils.expand(n::SymbolicOperator) = wrap(SymbolicUtils.expand(unwrap(n)))
substitute(expr::SymbolicOperator, s::Pair; kw...) = wrap(substituter(s)(unwrap(expr); kw...)) # backward compat
substitute(expr::SymbolicOperator, s::Vector; kw...) = wrap(substituter(s)(unwrap(expr); kw...))
substitute(expr::SymbolicOperator, s::Dict; kw...) = wrap(substituter(s)(unwrap(expr); kw...))
SymbolicUtils.Code.toexpr(x::SymbolicOperator) = SymbolicUtils.Code.toexpr(unwrap(x))
SymbolicUtils.Code.toexpr(x::SymbolicOperator,st) = SymbolicUtils.Code.toexpr(unwrap(x),st)
SymbolicUtils.setmetadata(x::SymbolicOperator, t, v) = wrap(SymbolicUtils.setmetadata(unwrap(x), t, v))
SymbolicUtils.getmetadata(x::SymbolicOperator, t) = SymbolicUtils.getmetadata(unwrap(x), t)
SymbolicUtils.hasmetadata(x::SymbolicOperator, t) = SymbolicUtils.hasmetadata(unwrap(x), t)

Broadcast.broadcastable(x::SymbolicOperator) = x

Base.zero(::AbstractOperator) = zero(typeof(AbstractOperator))
function Base.zero(::typeof(AbstractOperator))
    return 0
end

simplify_kernel(expr) = expr |> unwrap |> _simplify_kernel |> wrap
function _simplify_kernel(expr)
    is_number(x) = x isa Number
    is_operator(::BasicSymbolic{K}) where K <: AbstractOperator = true  
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

Base.isequal(a::SymbolicOperator,b::SymbolicOperator) = isequal(unwrap(a), unwrap(b))
convert(::Type{SymbolicOperator}, x::Number) = SymbolicOperator(x)
function Base.promote_rule(::Type{SymbolicOperator}, ::Type{K}) where K<:Number 
     return SymbolicOperator
end
#Dirty ?
function *(A::Matrix{SymbolicOperator},B::Matrix{SymbolicOperator})
    return wrap.(unwrap.(A)*unwrap.(B))
end
function *(A::Matrix,B::Matrix{SymbolicOperator})
    return wrap.(A*unwrap.(B))
end
function *(A::Matrix{SymbolicOperator},B::Matrix)
    return wrap.(unwrap.(A)*B)
end


Base.display(A::SymbolicOperator) = display(unwrap(A))

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
