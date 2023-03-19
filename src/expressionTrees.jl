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
struct ScalarLeaf{T<:UniformScaling} <: KernelExpressionLeaf
    scaling::T
end
ScalarLeaf(a::Number) = ScalarLeaf(a * I)

const ExpressionLike = Union{AbstractKernel,KernelExpression,UniformScaling}

convert(::Type{T}, kernel::K) where {T<:KernelExpression,K<:AbstractKernel} = KernelLeaf(kernel)
convert(::Type{T}, scalar::K) where {T<:KernelExpression,K<:Number} = ScalarLeaf(scalar * I)
convert(::Type{T}, λI::UniformScaling) where {T<:KernelExpression} = ScalarLeaf(λI)

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

*(left::ExpressionLike, right::ExpressionLike) = mul(convert(KernelExpression, left), convert(KernelExpression, right))
function mul(left::KernelExpression, right::KernelExpression)
    simplify_NullLeaf(::NullLeaf, ::NullLeaf) = NullLeaf()
    simplify_NullLeaf(::KernelExpression, ::NullLeaf) = NullLeaf()
    simplify_NullLeaf(::NullLeaf, ::KernelExpression) = NullLeaf()
    simplify_NullLeaf(left::KernelExpression, right::KernelExpression) = KernelMul(left, right)
    return simplify_NullLeaf(left, right)
end
\(left::ExpressionLike, right::ExpressionLike) = ldiv(convert(KernelExpression, left), convert(KernelExpression, right))
function ldiv(left::KernelExpression, right::KernelExpression)
    simplify_NullLeaf(::NullLeaf, ::KernelExpression) = error("Singular expression")
    simplify_NullLeaf(::NullLeaf, ::KernelExpression) = error("Singular expression")
    simplify_NullLeaf(::KernelExpression, ::NullLeaf) = NullLeaf()
    simplify_NullLeaf(left::KernelExpression, right::KernelExpression) = KernelLDiv(left, right)
    return simplify_NullLeaf(left, right)
end

+(left::ExpressionLike, right::ExpressionLike) = add(convert(KernelExpression, left), convert(KernelExpression, right))
function add(left::KernelExpression, right::KernelExpression)
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
    _simplify(expr::KernelMul) = local_part(left(expr)) * local_part(right(expr))
    _simplify(expr::KernelAdd) = local_part(left(expr)) + local_part(right(expr))
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
        nonlocal_part(expr |> left) * local_part(expr |> right) +
        local_part(expr |> left) * nonlocal_part(expr |> right) +
        local_part(expr |> left) * local_part(expr |> right)
    end
    _simplify(expr::KernelAdd) = nonlocal_part(left(expr)) + nonlocal_part(right(expr))
    function _simplify(expr::KernelLDiv)
        sol_local = local_part(left(expr)) \ local_part(right(expr))
        left_nonlocal = expr |> left |> local_part
        sol_nonlocal_A = -left(expr) \ (left_nonlocal * sol_local)
        sol_nonlocal_B = left(expr) \ right(expr)
        return sol_nonlocal_A + sol_nonlocal_B
    end
    _simplify(expr::KernelExpression) = NonLocalPart(expr)
    return _simplify(expr)
end

#The default implementation return an non-evaluated expression. 
function add(left::ExpressionLike, right::ExpressionLike)
    add(convert(KernelExpression, left), convert(KernelExpression, right))
end
function mul(left::ExpressionLike, right::ExpressionLike)
    mul(convert(KernelExpression, left), convert(KernelExpression, right))
end
function ldiv(left::ExpressionLike, right::ExpressionLike)
    ldiv(convert(KernelExpression, left), convert(KernelExpression, right))
end
function ldiv(A::KernelAdd, B::KernelLeaf{G}) where {G<:RetardedKernel}
    Aδ = evaluate_expression(local_part(A))
    Ar = evaluate_expression(nonlocal_part(A))
    B = evaluate_expression(B)
    return ldiv(TimeLocalKernel(
            axis(Ar), scalartype(Ar)(Aδ.λ)*I, blocksize(Ar), compression(Ar)
        ), Ar, B)
end
# TimeLocalKernel(axis, u::UniformScaling, blocksize::Int, compression)
#function ldiv(Aδ::ExpressionLike, Ar::ExpressionLike, B::ExpressionLike)
#    (convert(KernelExpression,Aδ) + convert(KernelExpression, Ar)) \ convert(KernelExpression,B)
#end


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

#evaluate_expression(expr::KernelExpression) = expr
evaluate_expression(expr::KernelLeaf) = expr.kernel
evaluate_expression(expr::ScalarLeaf) = expr.scaling
function evaluate_expression(expr::KernelExpressionTree)
    kernels = evaluate_expression.(arguments(expr))
    operation(expr)(kernels)
end

function matrix(expr)
    evaluated = evaluate_expression(expr)
    if evaluated isa AbstractKernel
        return matrix(evaluated)
    else
        return error("The expression cannot be converted to matrix")
    end
end