
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