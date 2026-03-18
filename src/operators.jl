
abstract type AbstractOperator end
abstract type CompositeOperator <: AbstractOperator end
abstract type SimpleOperator <: AbstractOperator end

step(k::SimpleOperator) = k |> discretization |> step
blocksize(g::SimpleOperator) = g |> discretization |> blocksize
axis(g::SimpleOperator) = g |> discretization |> axis
compression(g::SimpleOperator) = g |> discretization |> compression 
matrix(g::SimpleOperator) = g |> discretization |> matrix
scalartype(g::SimpleOperator) = g |> discretization |> scalartype
size(g::SimpleOperator) = g |> discretization |> size


import Base: ==
==(A::SimpleOperator,B::SimpleOperator) = axis(A) == axis(B) && compression(A) == compression(B) && matrix(A) == matrix(B)


function compress!(discretization::AbstractDiscretisation)
    cpr = discretization |> compression
    cpr( discretization |> matrix )
    return discretization
end
function compress!(op::SimpleOperator)
    op |> discretization |> compress!
    return op  
end
function similar(op::AbstractOperator, new_matrix_discretization::Union{AbstractMatrix,UniformScaling})
    similar(op,similar(discretization(op),new_matrix_discretization))
end
function similar(op::AbstractOperator, compression::AbstractCompression)
    similar(op,similar(discretization(op),matrix(op),compression = compression))
end

-(op::AbstractOperator) = -1 * op

*(λ::Number, op::SimpleOperator) = similar(op, λ * matrix(op))
*(op::SimpleOperator, λ::Number) = λ * op
*(scaling::UniformScaling, op::SimpleOperator) = scaling.λ * op
*(op::SimpleOperator, scaling::UniformScaling) = scaling.λ * op

function norm(operator::AbstractOperator)
    mat = matrix(operator)
    return norm(mat) / sqrt(prod(size(mat)))
end

struct DiracOperator{D<:AbstractDiscretisation} <: SimpleOperator
    discretization::D
end
function discretize_dirac(axis, f; compression::AbstractCompression=HssCompression())
    f00 = f(axis[1])
    T = eltype(f00)
    bs = size(f00, 1)
    δ = zeros(T, bs, bs, length(axis))
    for i = 1:length(axis)
        δ[:, :, i] .= f(axis[i])
    end
    matrix = build_blockdiag(δ, compression=compression)
    return DiracOperator(
        TrapzDiscretisation(
            axis,
            matrix,
            bs,
            compression
        )
    )
end

discretization(op::DiracOperator) = op.discretization
matrix(op::DiracOperator) = matrix(discretization(op))
causality(::DiracOperator) = Instantaneous()
function similar(::DiracOperator, new_discretization::AbstractDiscretisation)
    DiracOperator(new_discretization)
end
*(left::DiracOperator, right::DiracOperator) = similar(left, matrix(left) * matrix(right))
*(left::DiracOperator, right::SimpleOperator) = similar(right, matrix(left) * matrix(right))
*(left::SimpleOperator, right::DiracOperator) = similar(left, matrix(left) * matrix(right))
adjoint(op::DiracOperator) = DiracOperator( discretization(op)' )

struct SumOperator{L<:AbstractOperator,R<:AbstractOperator} <: CompositeOperator
    left::L
    right::R
end

==(A::SumOperator,B::SumOperator) = A.left == B.left && A.right == B.right

function _discretize_uniformScaling(discretization::TrapzDiscretisation, I)
    bs = blocksize(discretization)
    ax = axis(discretization)
    T =  scalartype(discretization)
    δ = zeros(T, bs, bs, length(ax))
    block_mat = diagm([T(I.λ) for k in 1:bs])
    for i = 1:length(ax)
        δ[:, :, i] = block_mat
    end
    matrix = build_blockdiag(δ, compression=compression(discretization))
    TrapzDiscretisation(
        ax,
        matrix,
        bs,
        compression(discretization)
    )
end
function SumOperator(left::AbstractOperator,right::UniformScaling)   
    uniformScaling_discretization = _discretize_uniformScaling(discretization(left), right)
    SumOperator( left, DiracOperator(uniformScaling_discretization))
end
function SumOperator(left::UniformScaling,right::AbstractOperator)   
    uniformScaling_discretization = _discretize_uniformScaling(discretization(right), left)
    SumOperator(DiracOperator(uniformScaling_discretization), right)
end
function compress!(op::SumOperator)
    compress!(op.left)
    compress!(op.right) 
end
causality(op::SumOperator) = causality_of_sum(causality(op.left), causality(op.right))
function compression(op::SumOperator)
    cl = compression(op.left)
    cr = compression(op.right)
    @assert cr == cl
    return cr
end

+(left::AbstractOperator, right::AbstractOperator) = SumOperator(left, right)
+(left::AbstractOperator, right::UniformScaling) = SumOperator(left, right)
+(left::UniformScaling, right::AbstractOperator) = SumOperator(left, right)
-(left::AbstractOperator, right::AbstractOperator) = SumOperator(left, -right)
-(left::AbstractOperator, right::UniformScaling) = SumOperator(left, -right)
-(left::UniformScaling, right::AbstractOperator) = SumOperator(left, -right)

function *(left::SumOperator, right::SumOperator)
    left.left * right.left + left.left * right.right + 
    left.right * right.left + left.right * right.right
end
*(left::SumOperator, right::Union{Number,UniformScaling,AbstractOperator}) = left.left * right + left.right * right
*(left::Union{Number,UniformScaling,AbstractOperator}, right::SumOperator) = left * right.left + left * right.right

adjoint(op::SumOperator) = SumOperator(op.left', op.right')


