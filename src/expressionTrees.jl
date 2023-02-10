abstract type KernelExpression end
abstract type KernelExpressionTree <: KernelExpression end 
abstract type KernelExpressionLeaf <: KernelExpression end 
struct KernelAdd <: KernelExpressionTree
    left::KernelExpression
    right::KernelExpression
end
struct KernelMul <: KernelExpressionTree 
    left::KernelExpression
    right::KernelExpression
end
struct KernelLDiv <: KernelExpressionTree
    left::KernelExpression
    right::KernelExpression
end
struct NullLeaf <: KernelExpressionLeaf end
struct KernelLeaf{T<:AbstractKernel} <: KernelExpressionLeaf
    kernel::T
end 
struct ScalarLeaf{T<:Number} <: KernelExpressionLeaf
    scalar::T
end
convert(::Type{T}, kernel::K) where {T <: KernelExpression, K <: AbstractKernel} = KernelLeaf(kernel)
convert(::Type{T}, scalar::K) where {T <: KernelExpression, K <: Number} = ScalarLeaf(scalar)
convert(::Type{T}, λI::UniformScaling) where {T <: KernelExpression} = ScalarLeaf(λI.λ)

istree(::KernelExpression) = false
istree(::KernelExpressionTree) = true

arguments(expr::KernelExpressionTree) = SA[expr.left, expr.right]

*(lambda::Number,kernel::KernelExpression) = KernelMul(lambda,kernel)
-(kernel::KernelExpression) = -1*kernel

function *(left::KernelExpression,right::KernelExpression)
    simplify_NullLeaf(::NullLeaf,::NullLeaf) = NullLeaf()
    simplify_NullLeaf(::KernelExpression,::NullLeaf) = NullLeaf()
    simplify_NullLeaf(::NullLeaf,::KernelExpression) = NullLeaf()
    simplify_NullLeaf(left::KernelExpression,right::KernelExpression) = KernelMul(left,right)
    return simplify_NullLeaf(left,right)
end
function \(left::KernelExpression,right::KernelExpression)
    simplify_NullLeaf(::NullLeaf,::KernelExpression) = error("Singular expression")
    simplify_NullLeaf(::NullLeaf,::KernelExpression) = error("Singular expression")
    simplify_NullLeaf(::KernelExpression,::NullLeaf) = NullLeaf()  
    simplify_NullLeaf(left::KernelExpression,right::KernelExpression) = KernelLDiv(left,right)
    return simplify_NullLeaf(left,right)
end
function +(left::KernelExpression,right::KernelExpression)
    simplify_NullLeaf(::NullLeaf,::NullLeaf) = NullLeaf()
    simplify_NullLeaf(left::KernelExpression,::NullLeaf) = left
    simplify_NullLeaf(::NullLeaf,right::KernelExpression) = right
    simplify_NullLeaf(left::KernelExpression,right::KernelExpression) = KernelAdd(left,right)
    return simplify_NullLeaf(left,right)
end
-(left::KernelExpression,right::KernelExpression) = left + (-1*right)
function operation(::KernelAdd)
    return function(tab)
        @assert length(tab) == 2
        return add(tab[1],tab[2])
    end 
end
function operation(::KernelMul)
    return function(tab)
        @assert length(tab) == 2
        return mul(tab[1],tab[2])
    end 
end
function operation(::KernelLDiv)
    return function(tab)
        @assert length(tab) == 2
        return ldiv(tab[1],tab[2])
    end 
end
function evaluate_expression(expr::KernelExpression) 
    kernels = evaluate_expression.( arguments(expr) )
    operation(expr)(kernels)
end
evaluate_expression(expr::KernelLeaf) = expr.kernel
evaluate_expression(expr::ScalarLeaf) = expr.scalar

@testitem "Test KernelExpression construction" begin
    using StaticArrays
    T = ComplexF32
    Ker = RetardedKernel
    bs = 2
    N = 128
    Dt = 2.
    ax = LinRange(-Dt/2,Dt,N)
    foo(x,y) = T <: Complex ? T.(1im .* [x x+y; x-y y]) : T.([x x+y; x-y y])
    foo(x) = T <: Complex ? T.(1im * [x 2x; 0 x]) : T.([x 2x; 0 x])
    A = randn(T,bs*N,bs*N)
    GA = Ker(ax,A,bs,NONCompression())
    GB = Ker(ax,foo, compression = NONCompression());
    LA = KernelLeaf(GA)
    LB = KernelLeaf(GB)
    @test arguments(KernelAdd(GA,GB)) == SA[LA, LB]
    KernelAdd(1,GB)
    @test LA + NullLeaf() == LA
    @test LA + LB == KernelAdd(LA,LB)
    @test LA * NullLeaf() == NullLeaf()
    @test LA * LB == KernelMul(LA,LB)
    @test LA \ NullLeaf() == NullLeaf()
    @test LA \ LB == KernelLDiv(LA,LB)
    @test -LA == KernelMul(ScalarLeaf(-1),LA)
    @test LB-LA == KernelAdd(LB,KernelMul(ScalarLeaf(-1),LA))
end

@testitem "Test KernelExpression evaluation" begin
    using StaticArrays
    T = ComplexF32
    Ker = RetardedKernel
    bs = 2
    N = 128
    Dt = 2.
    ax = LinRange(-Dt/2,Dt,N)
    foo(x,y) = T <: Complex ? T.(1im .* [x x+y; x-y y]) : T.([x x+y; x-y y])
    foo(x) = T <: Complex ? T.(1im * [x 2x; 0 x]) : T.([x 2x; 0 x])
    A = randn(T,bs*N,bs*N)
    GA = Ker(ax,A,bs,NONCompression())
    GB = Ker(ax,foo, compression = NONCompression());
    LA = KernelLeaf(GA)
    LB = KernelLeaf(GB)
    @test evaluate_expression(LA * LB) == GA*GB
    @test evaluate_expression(LA + LB) == GA+GB
    #@test evaluate_expression(LA \ LB) == GA\GB
end