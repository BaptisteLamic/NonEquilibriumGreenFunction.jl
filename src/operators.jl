
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
    kernelB = similar(kernelA, cprB)
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

