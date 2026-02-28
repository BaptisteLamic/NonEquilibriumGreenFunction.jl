<<<<<<< HEAD
@testitem "_build_triangular_low_rank_matrix_map" begin
=======
@testitem "TriangularLowRankCompression_map" begin
>>>>>>> 46f23ea (Refactor compression functions to use eachindex for improved performance; add tests for triangular low-rank compression and norm estimation)
    using LinearAlgebra
    using NonEquilibriumGreenFunction: computeMatrixNorm
    N, Dt = 128, 2.0
    axis = LinRange(-Dt / 2, Dt, N)
    for causality in (Acausal(), Retarded(), Advanced())
        for T = [Float64, ComplexF64, ComplexF32]
            tol = 20 * max(1E-14, eps(real(T)))
            f(x) = T <: Complex ? T(exp(-1im * x)) : T(cos(x))
            g(x) = T <: Complex ? T(exp(-4im * x)) : T(sin(4x))
            f00 = f(axis[1])
            @assert size(f00, 1) == size(f00, 2)
            blocksize = size(f00, 1)
            u = zeros(eltype(f00), blocksize, blocksize, length(axis))
            v = zeros(eltype(f00), blocksize, blocksize, length(axis))
            for k in eachindex(axis)
                u[:, :, k] .= f(axis[k])
                v[:, :, k] .= g(axis[k])
            end
            matrix = NonEquilibriumGreenFunction.BlockTriangularLowRankMatrix(u, v, causality)
            lm = NonEquilibriumGreenFunction.build_triangularLowRankMap(matrix)
            x = randn(T, size(matrix, 2))
            @test computeMatrixNorm(lm * x - matrix * x) < tol * length(x)
            @test computeMatrixNorm(lm' * x - matrix' * x) < tol * length(x)
        end
    end
end

@testitem "Norm estimate" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction: computeMatrixNorm
    using HssMatrices
    N = 1024
    for T = [Float64, ComplexF64, ComplexF32]
        tol = 100 * max(1E-12, eps(real(T)))
        A = [sin(i + j)^4 for i in 1:N, j in 1:N]
        if T <: Complex
            A = A * (1 + 1im)
        end
        hssA = hss(A, atol=tol / 100, rtol=tol / 100)
<<<<<<< HEAD
        @test computeMatrixNorm(A) - computeMatrixNorm(hssA) < 2*tol
=======
        @test computeMatrixNorm(A) - computeMatrixNorm(hssA) < tol
>>>>>>> 46f23ea (Refactor compression functions to use eachindex for improved performance; add tests for triangular low-rank compression and norm estimation)
        @test abs(computeMatrixNorm(A) - computeMatrixNorm(hssA)) / computeMatrixNorm(A) < tol
    end
end