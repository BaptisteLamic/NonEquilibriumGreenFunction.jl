using Revise
using Test
using NonEquilibriumGreenFunction
using LinearAlgebra
N, Dt = 64, 2.
ax = LinRange(-Dt / 2, Dt, N)
for T in (Float64, ComplexF64)
    c = 100
    tol = c * max(1E-6, eps(real(T)))
    dirac = discretize_dirac(ax, sin, compression=NONCompression())
    kernel = discretize_retardedkernel(ax,(x,y)->cos(x-y), compression = NONCompression())
    sumOp = dirac + kernel
    target_right = dirac*kernel + kernel * kernel
    target_left = kernel*dirac + kernel * kernel
    @test typeof(sumOp * kernel) == typeof(target_right)
    @test typeof(kernel * sumOp) == typeof(target_left)
    @test norm(matrix(sumOp*kernel + target_right)) / norm(matrix(target_right)) < tol
    @test norm(matrix(kernel * sumOp - target_left)) / norm(matrix(target_left)) < tol
end