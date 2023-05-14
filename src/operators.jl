abstract type AbstractCausality end
struct Retarded <: AbstractCausality end
struct Acausal <: AbstractCausality end
struct Advanced <: AbstractCausality end
struct Instantaneous <: AbstractCausality end

causality_of_sum(::C, ::C) where {C<:AbstractCausality} = C()
causality_of_sum(::AbstractCausality, ::AbstractCausality) = Acausal()
causality_of_sum(left::AbstractCausality, ::Instantaneous) = left
causality_of_sum(::Instantaneous, right::AbstractCausality) = right
causality_of_sum(::Instantaneous, ::Instantaneous) = Instantaneous

causality_of_prod(::Retarded, ::Retarded) = Retarded()
causality_of_prod(::Advanced, ::Advanced) = Advanced()
causality_of_prod(::Retarded, ::Acausal) = Acausal()
causality_of_prod(::Acausal, ::Advanced) = Acausal()
causality_of_prod(::Acausal, ::Acausal) = Acausal()
causality_of_prod(::Instantaneous, ::Instantaneous) = Instantaneous()
causality_of_prod(::Instantaneous, ::T) where {T<:AbstractCausality} = T()
causality_of_prod(::T, ::Instantaneous) where {T<:AbstractCausality} = T()

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

function compress!(discretization::AbstractDiscretisation)
    cpr = discretization |> compression
    cpr( discretization |> matrix)
    return discretization
end
function compress!(op::SimpleOperator)
    op |> discretization |> compress!
    return op  
end
function similar(op::AbstractOperator, new_matrix_discretization::AbstractMatrix)
    similar(op,similar(discretization(op),new_matrix_discretization))
end

-(op::AbstractOperator) = -1 * op

*(λ::Number, op::SimpleOperator) = similar(op, λ * matrix(op))
*(op::SimpleOperator, λ::Number) = λ * op
*(scaling::UniformScaling, op::SimpleOperator) = scaling.λ * op
*(op::SimpleOperator, scaling::UniformScaling) = scaling.λ * op

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
    matrix = blockdiag(δ, compression=compression)
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
causality(op::SumOperator) = causality_of_sum(causality(op.left), causality(op.right))
function compression(op::SumOperator)
    cl = compression(op.left)
    cr = compression(op.right)
    @assert cr == cr
    return cr
end

+(left::AbstractOperator, right::AbstractOperator) = SumOperator(left, right)
-(left::AbstractOperator, right::AbstractOperator) = SumOperator(left, -right)

function *(left::SumOperator, right::SumOperator)
    left.left * right.left + left.left * right.right + 
    left.right * right.left + left.right * right.right
end
*(left::SumOperator, right::Union{Number,UniformScaling,AbstractOperator}) = left.left * right + left.right * right
*(left::Union{Number,UniformScaling,AbstractOperator}, right::SumOperator) = left * right.left + left * right.right

adjoint(op::SumOperator) = SumOperator(op.left', op.right')

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

