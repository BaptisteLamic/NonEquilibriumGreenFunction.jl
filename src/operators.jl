
"""
    AbstractOperator

Abstract base type for all operators in the NonEquilibriumGreenFunction package.
"""
abstract type AbstractOperator end

"""
    CompositeOperator <: AbstractOperator

Abstract type for operators composed of other operators (e.g., SumOperator).
"""
abstract type CompositeOperator <: AbstractOperator end

"""
    SimpleOperator <: AbstractOperator

Abstract type for simple operators that wrap a single discretization (e.g., Kernel, DiracOperator).
"""
abstract type SimpleOperator <: AbstractOperator end

"""
    step(k::SimpleOperator)

Returns the step size of the operator's discretization axis.
"""
step(k::SimpleOperator) = k |> discretization |> step

"""
    blocksize(g::SimpleOperator)

Returns the block size of the operator.
"""
blocksize(g::SimpleOperator) = g |> discretization |> blocksize

"""
    axis(g::SimpleOperator)

Returns the discretization axis of the operator.
"""
axis(g::SimpleOperator) = g |> discretization |> axis

"""
    compression(g::SimpleOperator)

Returns the compression method used by the operator.
"""
compression(g::SimpleOperator) = g |> discretization |> compression

"""
    matrix(g::SimpleOperator)

Returns the underlying matrix of the operator.
"""
matrix(g::SimpleOperator) = g |> discretization |> matrix

"""
    scalartype(g::SimpleOperator)

Returns the scalar type of the operator's matrix elements.
"""
scalartype(g::SimpleOperator) = g |> discretization |> scalartype

"""
    size(g::SimpleOperator)

Returns the size of the operator's matrix.
"""
size(g::SimpleOperator) = g |> discretization |> size

import Base: ==
"""
    ==(A::SimpleOperator, B::SimpleOperator)

Compare two SimpleOperators for equality based on their axis, compression, and matrix.
"""
==(A::SimpleOperator,B::SimpleOperator) = axis(A) == axis(B) && compression(A) == compression(B) && matrix(A) == matrix(B)

"""
    make_similar(op::SimpleOperator, new_matrix::Union{AbstractMatrix,UniformScaling})

Create a new SimpleOperator with a modified matrix.

# Arguments
- `op`: Original SimpleOperator
- `new_matrix`: New matrix or UniformScaling to use

# Returns
A new SimpleOperator of the same type as `op` but with the new matrix.
"""
function make_similar(op::SimpleOperator, new_matrix::Union{AbstractMatrix,UniformScaling})
    new_dis = make_similar(discretization(op), new_matrix)
    # Reconstruct the operator with the new discretization
    # This works for SimpleOperator subtypes that have a similar constructor
    return typeof(op)(new_dis)
end

"""
    make_similar(op::SimpleOperator, new_compression::AbstractCompression)

Create a new SimpleOperator with a modified compression method.

# Arguments
- `op`: Original SimpleOperator
- `new_compression`: New compression method to use

# Returns
A new SimpleOperator of the same type as `op` but with the new compression.
"""
function make_similar(op::SimpleOperator, new_compression::AbstractCompression)
    new_dis = make_similar(discretization(op), matrix(op), compression = new_compression)
    return typeof(op)(new_dis)
end


"""
    compress!(discretization::AbstractDiscretisation)

In-place compression of a discretization's matrix.

# Arguments
- `discretization`: Discretization to compress

# Returns
The same discretization with its matrix compressed.

# Note
This modifies the discretization in-place by replacing its matrix with a compressed version.
"""
function compress!(discretization::AbstractDiscretisation)
    cpr = discretization |> compression
    cpr(discretization |> matrix)
    return discretization
end

"""
    compress!(op::SimpleOperator)

In-place compression of a simple operator's matrix.

# Arguments
- `op`: SimpleOperator to compress

# Returns
The same operator with its matrix compressed.
"""
function compress!(op::SimpleOperator)
    op |> discretization |> compress!
    return op  
end

# make_similar for AbstractOperator - dispatch to type-specific implementations
# These will be implemented for each concrete operator type (DiracOperator, Kernel, etc.)
# No generic implementation here to avoid infinite recursion

-(op::AbstractOperator) = -1 * op

*(λ::Number, op::SimpleOperator) = make_similar(op, λ * matrix(op))
*(op::SimpleOperator, λ::Number) = λ * op
*(scaling::UniformScaling, op::SimpleOperator) = scaling.λ * op
*(op::SimpleOperator, scaling::UniformScaling) = scaling.λ * op

"""
    norm(operator::AbstractOperator)

Compute the normalized Frobenius norm of an operator.

# Arguments
- `operator`: Operator to compute norm of

# Returns
The Frobenius norm divided by sqrt(prod(size(matrix))), which normalizes for the matrix size.

# Note
This normalization makes the norm independent of the discretization size,
which is useful for comparing operators defined on different axes.
"""
function norm(operator::AbstractOperator)
    mat = matrix(operator)
    return norm(mat) / sqrt(prod(size(mat)))
end

"""
    DiracOperator{D<:AbstractDiscretisation} <: SimpleOperator

An operator representing a diagonal matrix (Dirac delta in time space).

# Fields
- `discretization::D`: The underlying discretization

# Note
The matrix of a DiracOperator is block-diagonal with each block being the
identity (or a specified function) at each time point.
"""
struct DiracOperator{D<:AbstractDiscretisation} <: SimpleOperator
    discretization::D
end

"""
    discretize_dirac(axis, f; compression=HssCompression())

Create a DiracOperator from a function f defined on an axis.

# Arguments
- `axis`: Time axis for discretization
- `f`: Function that returns a block matrix at each time point
- `compression`: Compression method to use (default: HssCompression)

# Returns
A DiracOperator representing the diagonal matrix with f(axis[i]) at each block.
"""
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

"""
    discretization(op::DiracOperator)

Returns the discretization of a DiracOperator.
"""
discretization(op::DiracOperator) = op.discretization

"""
    matrix(op::DiracOperator)

Returns the underlying matrix of a DiracOperator.
"""
matrix(op::DiracOperator) = matrix(discretization(op))

"""
    causality(::DiracOperator)

Returns the causality of a DiracOperator (always Instantaneous).
"""
causality(::DiracOperator) = Instantaneous()

"""
    make_similar(::DiracOperator, new_discretization::AbstractDiscretisation)

Create a new DiracOperator with a different discretization.

# Arguments
- `new_discretization`: New discretization to use

# Returns
A new DiracOperator with the specified discretization.
"""
function make_similar(::DiracOperator, new_discretization::AbstractDiscretisation)
    return DiracOperator(new_discretization)
end

"""
    *(left::DiracOperator, right::DiracOperator)

Multiply two DiracOperators.
"""
*(left::DiracOperator, right::DiracOperator) = make_similar(left, matrix(left) * matrix(right))

"""
    *(left::DiracOperator, right::SimpleOperator)

Multiply a DiracOperator with a SimpleOperator.
"""
*(left::DiracOperator, right::SimpleOperator) = make_similar(right, matrix(left) * matrix(right))

"""
    *(left::SimpleOperator, right::DiracOperator)

Multiply a SimpleOperator with a DiracOperator.
"""
*(left::SimpleOperator, right::DiracOperator) = make_similar(left, matrix(left) * matrix(right))

"""
    adjoint(op::DiracOperator)

Return the adjoint of a DiracOperator.
"""
adjoint(op::DiracOperator) = DiracOperator(discretization(op)')

"""
    SumOperator{L<:AbstractOperator,R<:AbstractOperator} <: CompositeOperator

An operator representing the sum of two operators.

# Fields
- `left::L`: Left operator in the sum
- `right::R`: Right operator in the sum
"""
struct SumOperator{L<:AbstractOperator,R<:AbstractOperator} <: CompositeOperator
    left::L
    right::R
end

"""
    ==(A::SumOperator, B::SumOperator)

Compare two SumOperators for equality based on their left and right components.
"""
==(A::SumOperator,B::SumOperator) = A.left == B.left && A.right == B.right

"""
    _discretize_uniformScaling(discretization::TrapzDiscretisation, I::UniformScaling)

Create a block-diagonal discretization from a UniformScaling operator.

# Arguments
- `discretization`: Base discretization to use axis and blocksize from
- `I`: UniformScaling operator

# Returns
A TrapzDiscretisation representing the identity scaled by I.λ.
"""
function _discretize_uniformScaling(discretization::TrapzDiscretisation, I)
    bs = blocksize(discretization)
    ax = axis(discretization)
    T = scalartype(discretization)
    δ = zeros(T, bs, bs, length(ax))
    block_mat = diagm([T(I.λ) for k in 1:bs])
    for i = 1:length(ax)
        δ[:, :, i] = block_mat
    end
    matrix = build_blockdiag(δ, compression=compression(discretization))
    return TrapzDiscretisation(
        ax,
        matrix,
        bs,
        compression(discretization)
    )
end

"""
    SumOperator(left::AbstractOperator, right::UniformScaling)

Create a SumOperator from an AbstractOperator and a UniformScaling.
"""
function SumOperator(left::AbstractOperator, right::UniformScaling)   
    uniformScaling_discretization = _discretize_uniformScaling(discretization(left), right)
    return SumOperator(left, DiracOperator(uniformScaling_discretization))
end

"""
    SumOperator(left::UniformScaling, right::AbstractOperator)

Create a SumOperator from a UniformScaling and an AbstractOperator.
"""
function SumOperator(left::UniformScaling, right::AbstractOperator)   
    uniformScaling_discretization = _discretize_uniformScaling(discretization(right), left)
    return SumOperator(DiracOperator(uniformScaling_discretization), right)
end

"""
    compress!(op::SumOperator)

In-place compression of a SumOperator's components.
"""
function compress!(op::SumOperator)
    compress!(op.left)
    compress!(op.right) 
end

"""
    causality(op::SumOperator)

Returns the causality of a SumOperator based on its components.
"""
causality(op::SumOperator) = causality_of_sum(causality(op.left), causality(op.right))

"""
    compression(op::SumOperator)

Returns the compression method of a SumOperator (both components must have the same compression).

# Throws
AssertionError if left and right operators have different compression methods.
"""
function compression(op::SumOperator)
    cl = compression(op.left)
    cr = compression(op.right)
    @assert cr == cl "SumOperator components must have the same compression"
    return cr
end

"""
    +(left::AbstractOperator, right::AbstractOperator)

Add two AbstractOperators by creating a SumOperator.
"""
+(left::AbstractOperator, right::AbstractOperator) = SumOperator(left, right)

"""
    +(left::AbstractOperator, right::UniformScaling)

Add an AbstractOperator and a UniformScaling.
"""
+(left::AbstractOperator, right::UniformScaling) = SumOperator(left, right)

"""
    +(left::UniformScaling, right::AbstractOperator)

Add a UniformScaling and an AbstractOperator.
"""
+(left::UniformScaling, right::AbstractOperator) = SumOperator(left, right)

"""
    -(left::AbstractOperator, right::AbstractOperator)

Subtract two AbstractOperators by creating a SumOperator with the negated right operand.
"""
-(left::AbstractOperator, right::AbstractOperator) = SumOperator(left, -right)

"""
    -(left::AbstractOperator, right::UniformScaling)

Subtract a UniformScaling from an AbstractOperator.
"""
-(left::AbstractOperator, right::UniformScaling) = SumOperator(left, -right)

"""
    -(left::UniformScaling, right::AbstractOperator)

Subtract an AbstractOperator from a UniformScaling.
"""
-(left::UniformScaling, right::AbstractOperator) = SumOperator(left, -right)

"""
    *(left::SumOperator, right::SumOperator)

Multiply two SumOperators using the distributive property:
(left.left + left.right) * (right.left + right.right) = 
  left.left*right.left + left.left*right.right + 
  left.right*right.left + left.right*right.right
"""
function *(left::SumOperator, right::SumOperator)
    return left.left * right.left + left.left * right.right + 
           left.right * right.left + left.right * right.right
end

"""
    *(left::SumOperator, right::Union{Number,UniformScaling,AbstractOperator})

Multiply a SumOperator with another operator/number using distributive property.
"""
*(left::SumOperator, right::Union{Number,UniformScaling,AbstractOperator}) = left.left * right + left.right * right

"""
    *(left::Union{Number,UniformScaling,AbstractOperator}, right::SumOperator)

Multiply another operator/number with a SumOperator using distributive property.
"""
*(left::Union{Number,UniformScaling,AbstractOperator}, right::SumOperator) = left * right.left + left * right.right

"""
    adjoint(op::SumOperator)

Return the adjoint of a SumOperator: (left + right)' = left' + right'
"""

adjoint(op::SumOperator) = SumOperator(op.left', op.right')

@testitem "Equality test" begin
    using LinearAlgebra
    N, Dt = 256, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    c = 100
    kernelA = discretize_retardedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
    kernelB = discretize_advancedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
    kernelC = discretize_retardedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
    @assert kernelA != kernelB
    @assert kernelA == deepcopy(kernelA)
    @assert kernelC == kernelA
    @assert causality(kernelA + kernelB) == Acausal()
    @assert kernelA + kernelB == kernelB + kernelC
end
@testitem "Mechanical actions" begin
    using LinearAlgebra
    N, Dt = 256, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    c = 100
    cprA = HssCompression(atol = 1E-4,rtol = 1E-4)
    cprB = HssCompression(atol = 1E-6,rtol = 1E-6)
    kernelA = discretize_retardedkernel(ax, (x, y) -> cos(x - y), compression=cprA)
    kernelB = make_similar(kernelA, cprB)
    @test compression(kernelA) == cprA
    @test compression(kernelB) == cprB
    compress!(kernelB)
    @test norm(matrix(kernelA) - matrix(kernelB)) < cprB.atol
    @test norm(matrix(kernelA) - matrix(kernelB)) / norm(matrix(kernelB)) < cprB.rtol
end
@testitem "Action of UniformScaling on kernel" begin
    using LinearAlgebra
    N, Dt = 256, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    for T in (Float64,)
        c = 100
        kernelA = discretize_retardedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
        kernelB = discretize_retardedkernel(ax, (x, y) -> sin(x - y), compression=NONCompression())
        @test typeof(I*kernelA) == typeof(kernelA) 
        @test matrix(2*I*kernelA) == 2*matrix(kernelA)
        @test typeof(kernelA*I) == typeof(kernelA) 
        @test matrix(kernelA*2I) == 2*matrix(kernelA)
        @test matrix((I + kernelA)*kernelB) == matrix(kernelB + kernelA*kernelB)
        @test matrix((kernelA + I)*kernelB) == matrix(kernelB + kernelA*kernelB)
    end
end

@testitem "Dirac operator action" begin 
    using LinearAlgebra
    N, Dt = 64, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    for T in (Float64, ComplexF64)
        c = 100
        tol = c * max(1E-6, eps(real(T)))
        dirac = discretize_dirac(ax, x->1., compression=NONCompression())
        kernel = discretize_retardedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
        @test norm(matrix(kernel * dirac - kernel)) / norm(matrix(kernel)) < tol
    end
end

@testitem "Product of Dirac operator" begin
    using LinearAlgebra
    N, Dt = 64, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    for T in (Float64, ComplexF64)
        c = 100
        tol = c * max(1E-6, eps(real(T)))
        dirac = discretize_dirac(ax, sin, compression=NONCompression())
        kernel = discretize_retardedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
        target_left = discretize_retardedkernel(ax, (x, y) -> sin(x) * cos(x - y), compression=NONCompression())
        target_right = discretize_retardedkernel(ax, (x, y) -> cos(x - y) * sin(y), compression=NONCompression())
        @test norm(matrix(dirac * kernel - target_left)) / norm(matrix(target_left)) < tol
        @test norm(matrix(kernel * dirac - target_right)) / norm(matrix(target_right)) < tol
    end
end

@testitem "adjoint of SimpleOperator" begin
    using LinearAlgebra
    N, Dt = 256, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    c = 100
    kernelA = discretize_retardedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
    kernelB = discretize_advancedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
    @test matrix(adjoint(kernelA)) == adjoint(matrix(kernelA)) 
    @test matrix(adjoint(kernelB)) == adjoint(matrix(kernelB)) 
    @test causality(adjoint(kernelA)) == Advanced()
    @test causality(adjoint(kernelB)) == Retarded()
end

@testitem "Dirac operator scalar operation" begin
    using LinearAlgebra
    N, Dt = 64, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    for T in (Float64, ComplexF64)
        c = 100
        tol = c * max(1E-6, eps(real(T)))
        dirac = discretize_dirac(ax, sin, compression=NONCompression())
        @test matrix(2I * dirac) == 2 * matrix(dirac)
    end
end
@testitem "Dirac operator adjoint" begin
    using LinearAlgebra
    N, Dt = 64, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    for T in (Float64, ComplexF64)
        c = 100
        tol = c * max(1E-6, eps(real(T)))
        foo(x) = sin(x) + (T<:Complex ? 1im : 0)
        dirac = discretize_dirac(ax, foo, compression=NONCompression())
        @test matrix(dirac') == matrix(dirac)'
    end
end


@testitem "Product of sum operator 1" begin
    using LinearAlgebra
    N, Dt = 256, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    for T in (Float64,)
        c = 100
        tol = c * max(1E-6, eps(real(T)))
        kernelA = discretize_retardedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
        kernelB = discretize_retardedkernel(ax, (x, y) -> sin(x - y), compression=NONCompression())
        sumOp = SumOperator(kernelA, kernelB)
        target_right = kernelA * kernelB + kernelB * kernelB
        @test typeof(sumOp * kernelB) == typeof(target_right)
        @test norm(matrix(sumOp * kernelB - target_right)) / norm(matrix(target_right)) < tol
        target_left = kernelB * kernelA + kernelB * kernelB
        @test typeof(kernelB * sumOp) == typeof(target_left)
        @test norm(matrix(kernelB * sumOp - target_left)) / norm(matrix(target_left)) < tol
    end
end

@testitem "Product of sum operator 2" begin
    using LinearAlgebra
    N, Dt = 256, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    for T in (Float64,)
        c = 100
        tol = c * max(1E-6, eps(real(T)))
        dirac = discretize_dirac(ax, sin, compression=NONCompression())
        kernel = discretize_retardedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
        sumOp = dirac + kernel
        target_right = dirac * kernel + kernel * kernel
        target_left = kernel * dirac + kernel * kernel
        @test typeof(sumOp * kernel) == typeof(target_right)
        @test typeof(kernel * sumOp) == typeof(target_left)
        @test typeof(kernel * sumOp) == typeof(target_left)
        @test norm(matrix(2 * sumOp * kernel - 2 * target_right)) / norm(matrix(2 * target_right)) < tol
        @test norm(matrix(2 * kernel * sumOp - 2 * target_left)) / norm(matrix(2 * target_left)) < tol
    end
end

@testitem "Scalar operation on sum product" begin
    using LinearAlgebra
    N, Dt = 256, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    for T in (Float64,)
        c = 100
        tol = c * max(1E-6, eps(real(T)))
        dirac = discretize_dirac(ax, sin, compression=NONCompression())
        kernel = discretize_retardedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
        sumOp = dirac + kernel
        @test matrix((-sumOp).left) == -matrix(dirac)
        @test matrix((-sumOp).right) == -matrix(kernel)
        @test matrix((3*sumOp).left) == 3*matrix(dirac)
        @test matrix((3*sumOp).right) == 3*matrix(kernel)
    end
end

@testitem "Product of sum operator 3" begin
    using LinearAlgebra
    N, Dt = 256, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    for T in (Float64,)
        c = 100
        tol = c * max(1E-6, eps(real(T)))
        dirac = discretize_dirac(ax, sin, compression=NONCompression())
        kernel = discretize_retardedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
        sumOp = dirac + kernel
        target = dirac*dirac + dirac*kernel + kernel*dirac + kernel*kernel
        @test typeof(sumOp * sumOp) == typeof(target)
        @test norm(matrix((sumOp * sumOp - target)*kernel)) / norm(matrix(target*kernel)) < tol
    end
end

@testitem "Adjoint of sum operator" begin
    using LinearAlgebra
    N, Dt = 256, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    for T in (Float64,)
        c = 100
        tol = c * max(1E-6, eps(real(T)))
        dirac = discretize_dirac(ax, sin, compression=NONCompression())
        kernel = discretize_retardedkernel(ax, (x, y) -> cos(x - y), compression=NONCompression())
        sumOp = dirac + kernel
        adjoint_sumOp = sumOp'
        @test matrix(sumOp.left') == matrix(dirac')
        @test matrix(sumOp.right') == matrix(kernel')
    end
end

 
