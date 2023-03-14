abstract type KernelExpression end
abstract type KernelExpressionTree <: KernelExpression end
abstract type KernelExpressionLeaf <: KernelExpression end
abstract type BinaryOperation <: KernelExpressionTree end
abstract type UnaryOperation <: KernelExpressionTree end

struct KernelAdd <: BinaryOperation
    left::KernelExpression
    right::KernelExpression
end
struct KernelMul <: BinaryOperation
    left::KernelExpression
    right::KernelExpression
end
struct KernelLDiv <: BinaryOperation
    left::KernelExpression
    right::KernelExpression
end
struct LocalPart <: UnaryOperation
    expr::KernelExpression
end
struct NonLocalPart <: UnaryOperation
    expr::KernelExpression
end

struct NullLeaf <: KernelExpressionLeaf end
struct KernelLeaf{T<:AbstractKernel} <: KernelExpressionLeaf
    kernel::T
end
struct ScalarLeaf{T<:Number} <: KernelExpressionLeaf
    scalar::T
end

const ExpressionLike = Union{AbstractKernel,KernelExpression,UniformScaling}

convert(::Type{T}, kernel::K) where {T<:KernelExpression,K<:AbstractKernel} = KernelLeaf(kernel)
convert(::Type{T}, scalar::K) where {T<:KernelExpression,K<:Number} = ScalarLeaf(scalar)
convert(::Type{T}, λI::UniformScaling) where {T<:KernelExpression} = ScalarLeaf(λI.λ)

istree(::KernelExpression) = false
istree(::KernelExpressionTree) = true

left(expr::BinaryOperation) = expr.left
right(expr::BinaryOperation) = expr.right

arguments(expr::BinaryOperation) = SA[expr|>left, expr|>right]
arguments(expr::UnaryOperation) = SA[expr.expr]


islocal(expr::KernelLeaf) = expr.kernel |> islocal
islocal(::ScalarLeaf) = true
islocal(::NullLeaf) = true

*(lambda::Number, kernel::AbstractKernel) = KernelMul(lambda, kernel)
*(lambda::Number, kernel::KernelExpression) = KernelMul(lambda, kernel)
-(kernel::KernelExpression) = -1 * kernel
-(kernel::AbstractKernel) = -KernelLeaf(kernel)

*(left::ExpressionLike, right::ExpressionLike) = expression_mul(convert(KernelExpression, left), convert(KernelExpression, right))
function expression_mul(left::KernelExpression, right::KernelExpression)
    simplify_NullLeaf(::NullLeaf, ::NullLeaf) = NullLeaf()
    simplify_NullLeaf(::KernelExpression, ::NullLeaf) = NullLeaf()
    simplify_NullLeaf(::NullLeaf, ::KernelExpression) = NullLeaf()
    simplify_NullLeaf(left::KernelExpression, right::KernelExpression) = KernelMul(left, right)
    return simplify_NullLeaf(left, right)
end
\(left::ExpressionLike, right::ExpressionLike) = expression_ldiv(convert(KernelExpression, left), convert(KernelExpression, right))
function expression_ldiv(left::KernelExpression, right::KernelExpression)
    simplify_NullLeaf(::NullLeaf, ::KernelExpression) = error("Singular expression")
    simplify_NullLeaf(::NullLeaf, ::KernelExpression) = error("Singular expression")
    simplify_NullLeaf(::KernelExpression, ::NullLeaf) = NullLeaf()
    simplify_NullLeaf(left::KernelExpression, right::KernelExpression) = KernelLDiv(left, right)
    return simplify_NullLeaf(left, right)
end

+(left::ExpressionLike, right::ExpressionLike) = expression_add(convert(KernelExpression, left), convert(KernelExpression, right))
function expression_add(left::KernelExpression, right::KernelExpression)
    simplify_NullLeaf(::NullLeaf, ::NullLeaf) = NullLeaf()
    simplify_NullLeaf(left::KernelExpression, ::NullLeaf) = left
    simplify_NullLeaf(::NullLeaf, right::KernelExpression) = right
    simplify_NullLeaf(left::KernelExpression, right::KernelExpression) = KernelAdd(left, right)
    return simplify_NullLeaf(left, right)
end
-(left::ExpressionLike, right::ExpressionLike) = left + (-1 * right)

local_part(expr::ExpressionLike) = expression_local_part(convert(KernelExpression, expr))
function expression_local_part(expr::KernelExpression)
    _simplify(::NullLeaf) = NullLeaf()
    _simplify(leaf::ScalarLeaf) = leaf
    _simplify(leaf::KernelLeaf) = islocal(leaf) ? leaf : NullLeaf()
    _simplify(expr::KernelMul) = local_part(left(expr)) * local_part( right(expr) )
    _simplify(expr::KernelAdd) = local_part(left(expr)) + local_part( right(expr) )
    _simplify(expr::KernelLDiv) = KernelLDiv(
        expr |> left |> local_part,
        expr |> right |> local_part
    )
    _simplify(expr::KernelExpression) = LocalPart(expr)
    return _simplify(expr)
end

nonlocal_part(expr) = expression_nonlocal_part(convert(KernelExpression, expr))
function expression_nonlocal_part(expr::KernelExpression)
    _simplify(::NullLeaf) = NullLeaf()
    _simplify(::ScalarLeaf) = NullLeaf()
    _simplify(leaf::KernelLeaf) = islocal(leaf) ? NullLeaf() : leaf
    function _simplify(expr::KernelMul)
        nonlocal_part(expr |>  left)*local_part(expr |> right) + 
        local_part(expr |> left)*nonlocal_part(expr |> right) + 
        local_part(expr |> left)*local_part(expr |> right)
    end
    _simplify(expr::KernelAdd) = nonlocal_part(left(expr)) + nonlocal_part(right(expr)) 
    function _simplify(expr::KernelLDiv)
        sol_local = local_part(left(expr))  \  local_part(right(expr))
        left_nonlocal = expr |> left |> local_part
        sol_nonlocal_A = - left(expr) \ (left_nonlocal * sol_local)
        sol_nonlocal_B = left(expr) \ right(expr)
        return sol_nonlocal_A + sol_nonlocal_B
    end
    _simplify(expr::KernelExpression) = NonLocalPart(expr)
    return _simplify(expr)
end

function ldiv(A::KernelAdd,B::KernelLeaf{RetardedKernel})
    Aδ = local_part(A)
    Ar = nonlocal_part(A)
    return kernel_ldiv(Aδ, Ar, B)
end
function ldiv(A::AbstractKernel,B::AbstractKernel)
    return kernel_ldiv(A,B)
end

function operation(::KernelAdd)
    return function (tab)
        @assert length(tab) == 2
        return add(tab[1], tab[2])
    end
end
function operation(::KernelMul)
    return function (tab)
        @assert length(tab) == 2
        return mul(tab[1], tab[2])
    end
end
function operation(::KernelLDiv)
    return function (tab)
        @assert length(tab) == 2
        return ldiv(tab[1], tab[2])
    end
end

function evaluate_expression(expr::KernelExpression)
    kernels = evaluate_expression.(arguments(expr))
    operation(expr)(kernels)
end
evaluate_expression(expr::KernelLeaf) = expr.kernel
evaluate_expression(expr::ScalarLeaf) = expr.scalar

function matrix(expr)
    @show expr
    return matrix( evaluate_expression( expr ) ) 
end