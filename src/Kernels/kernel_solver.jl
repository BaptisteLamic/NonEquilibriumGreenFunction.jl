


export kernelSolve

"""
    Solve the equation  G = g + K⋅G  for G
"""
function solve_dyson(g::Kernel,K::Kernel)
    @assert isretarded(g) & isretarded(K)
    cp = compression(g)
    bs = blocksize(g)
    diag_K = extract_blockdiag(K |> matrix, bs)
    diag_g = extract_blockdiag(g |> matrix, bs)
    eye = cp(sparse(scalartype(K)(1)*I,size(diag_K)...)) #bypass limitation of HssMatrices.jl
    left = cp( eye - scalartype(K)(step(K)) * (matrix(K) - cp(1//2 * diag_K)) )
    right = cp( matrix(g) - cp(1 // 2 * diag_g))
    sol_biased = _ldiv!(left,right)
    correction = cp(diag_g - extract_blockdiag( sol_biased,bs))
    return similar(g, sol_biased + correction )
end

function _ldiv!(left::T,right::T) where {T<:AbstractMatrix}
    return left \ right
end
function _ldiv!(left::HssMatrix,right::HssMatrix)
    return ldiv!(left, right)
end

function kernelSolve(A::Kernel,B::Kernel)
    @assert isretarded(A) & isretarded(B)
    cp = compression(A)
    bs = blocksize(A)
    diag_A = extract_blockdiag(A |> matrix, bs)
    diag_B = extract_blockdiag(B |> matrix, bs)

    left = cp(scalartype(A)(step(A)) * (matrix(A) - cp(1//2 * diag_A)) )
    right = cp( matrix(B) - cp(1 // 2 * diag_B))

    sol_biased = _ldiv!(left,right)
    correction = cp(diag_B - extract_blockdiag( sol_biased,bs))
    return similar(B, sol_biased + correction )
end

@testitem "simplification" begin
    #TODO: refine the test
    using LinearAlgebra
    using HssMatrices
    N, Dt = 256, 0.2
    ax = LinRange(-Dt / 2, Dt, N)
    c = 100
    kernelA = discretize_retardedkernel(ax, (x, y) -> cos(x - y)*exp(-abs(x - y)))
    kernelB = discretize_retardedkernel(ax, (x, y) -> 1 + sin(x - y))
    kernelC = kernelSolve(kernelA, kernelB)
    kernelD = kernelA * kernelC
    @test norm(full(matrix(kernelD-kernelB))) / prod(size(kernelB)) < 1e-10
end