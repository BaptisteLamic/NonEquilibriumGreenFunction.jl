@testitem "TriangularLowRankCompression_map" begin
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
        @test computeMatrixNorm(A) - computeMatrixNorm(hssA) < 2*tol
        @test abs(computeMatrixNorm(A) - computeMatrixNorm(hssA)) / computeMatrixNorm(A) < tol
    end
end

@testitem "Low-rank kernel HSS compression correctness" begin
    using LinearAlgebra
    N, Dt, bs = 8, 1.0, 2
    ax = LinRange(0, Dt, N)
    for T in [Float64, ComplexF64, ComplexF32]
        for causality in (Retarded, Advanced, Acausal)
            tol = 100 * max(1E-12, eps(real(T)))
            f(x) = T.([x 0; 0 x])
            g(x) = T.([x^2 0; 0 x^2])
            K = discretize_lowrank_kernel(
                TrapzDiscretisation,
                causality,
                ax,
                f,
                g;
                compression=HssCompression(atol=1e-12, rtol=1e-12)
            )
            _getMask(::Type{Acausal}) = (i, j) -> T(true)
            _getMask(::Type{Retarded}) = (i, j) -> T(i >= j)
            _getMask(::Type{Advanced}) = (i, j) -> T(i <= j)
            mask = _getMask(causality)
            K_ref = zeros(eltype(matrix(K)), N * bs, N * bs)
            for i in 1:N, j in 1:N
                K_ref[blockrange(i, bs), blockrange(j, bs)] .= f(ax[i]) * g(ax[j]) * mask(i, j)
            end
            @test norm(matrix(K) - K_ref) < tol
        end
    end
end