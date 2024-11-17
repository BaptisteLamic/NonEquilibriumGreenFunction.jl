struct BlockTriangularLowRankMatrix{T} <: AbstractMatrix{T}
    #The two first dims are the blocks size
    u::Array{T,3}
    v::Array{T,3}
    causality::AbstractCausality
end

function BlockTriangularLowRankMatrix(u::AbstractArray{T,3}, v::AbstractArray{T,3}, causality) where {T}
    @assert size(u, 1) == size(u, 2)
    @assert size(u) == size(v)
    BlockTriangularLowRankMatrix{T}(u, v, causality)
end

mask(::Retarded, x, y) = x >= y ? 1 : 0
mask(::Advanced, x, y) = x <= y ? 1 : 0
mask(::Acausal, x, y) = 1

function getNumberOfBlocks(A::BlockTriangularLowRankMatrix)
    return size(A.u, 3)
end
function blocksize(A::BlockTriangularLowRankMatrix)
    return size(A.u, 1)
end

function size(A::BlockTriangularLowRankMatrix)
    n = getNumberOfBlocks(A) * blocksize(A)
    return (n, n)
end
function getindex(A::BlockTriangularLowRankMatrix, I::Vararg{Int,2})
    #TODO: optimize for multi entries requests
    blck_i, s_i = blockindex(I[1], blocksize(A))
    blck_j, s_j = blockindex(I[2], blocksize(A))
    @views block = A.u[:, :, blck_i] * A.v[:, :, blck_j]
    return mask(A.causality, blck_i, blck_j) .* block[s_i, s_j]
end

function _mul!(r, A, x, ::Acausal)
    b = zeros(eltype(x), blocksize(A), size(x, 2))
    for k = 1:getNumberOfBlocks(A)
        @views b += A.v[:, :, k] * x[blockrange(k, blocksize(A)), :]
    end
    Threads.@threads for k = 1:getNumberOfBlocks(A)
        @views r[blockrange(k, blocksize(A)), :] = A.u[:, :, k] * b
    end
end
function _mul!(r, A, x, ::Retarded)
    b = zeros(eltype(x), blocksize(A), size(x, 2), getNumberOfBlocks(A))
    b[:, :, 1] += A.v[:, :, 1] * x[blockrange(1, blocksize(A)), :]
    nblocks = getNumberOfBlocks(A)
    for k = 2:nblocks
        @views b[:, :, k] += b[:, :, k-1] + A.v[:, :, k] * x[blockrange(k, blocksize(A)), :]
    end
    Threads.@threads for k = 1:getNumberOfBlocks(A)
        @views r[blockrange(k, blocksize(A)), :] = A.u[:, :, k] * b[:, :, k]
    end
end

function _mul!(r, A, x, ::Advanced)
    b = zeros(eltype(x), blocksize(A), size(x, 2), getNumberOfBlocks(A))
    nblocks = getNumberOfBlocks(A)
    b[:, :, end] += A.v[:, :, end] * x[blockrange(nblocks, blocksize(A)), :]
    for k = nblocks-1:-1:1
        @views b[:, :, k] += b[:, :, k+1] + A.v[:, :, k] * x[blockrange(k, blocksize(A)), :]
    end
    Threads.@threads for k = 1:getNumberOfBlocks(A)
        @views r[blockrange(k, blocksize(A)), :] = A.u[:, :, k] * b[:, :, k]
    end
end

function _cmul!(r, A, x, ::Acausal)
    b = zeros(eltype(x), blocksize(A), size(x, 2))
    for k = 1:getNumberOfBlocks(A)
        @views b += A.u[:, :, k]' * x[blockrange(k, blocksize(A)), :]
    end
    Threads.@threads for k = 1:getNumberOfBlocks(A)
        @views r[blockrange(k, blocksize(A)), :] = A.v[:, :, k]' * b
    end
end
function _cmul!(r, A, x, ::Retarded)
    b = zeros(eltype(x), blocksize(A), size(x, 2), getNumberOfBlocks(A))
    nblocks = getNumberOfBlocks(A)
    b[:, :, end] += A.u[:, :, end]' * x[blockrange(nblocks, blocksize(A)), :]
    for k = nblocks-1:-1:1
        @views b[:, :, k] += b[:, :, k+1] + A.u[:, :, k]' * x[blockrange(k, blocksize(A)), :]
    end
    Threads.@threads for k = 1:getNumberOfBlocks(A)
        @views r[blockrange(k, blocksize(A)), :] = A.v[:, :, k]' * b[:, :, k]
    end
end
function _cmul!(r, A, x, ::Advanced)
    b = zeros(eltype(x), blocksize(A), size(x, 2), getNumberOfBlocks(A))
    b[:, :, 1] += A.u[:, :, 1]' * x[blockrange(1, blocksize(A)), :]
    nblocks = getNumberOfBlocks(A)
    for k = 2:nblocks
        @views b[:, :, k] += b[:, :, k-1] + A.u[:, :, k]' * x[blockrange(k, blocksize(A)), :]
    end
    Threads.@threads for k = 1:getNumberOfBlocks(A)
        @views r[blockrange(k, blocksize(A)), :] = A.v[:, :, k]' * b[:, :, k]
    end
end

function _mul(A::BlockTriangularLowRankMatrix{T}, x::AbstractArray{T,2}) where {T}
    r = similar(x)
    _mul!(r, A, x, A.causality)
    return r
end

function _cmul(A::BlockTriangularLowRankMatrix{T}, x::AbstractArray{T,2}) where {T}
    r = similar(x)
    _cmul!(r, A, x, A.causality)
    return r
end

@testitem "Basic accessor" begin
    import NonEquilibriumGreenFunction.BlockTriangularLowRankMatrix
    T = ComplexF64
    _blocksize = 3
    nbBlocks = 10
    u = [sin(i * j + k) for i in 1:_blocksize, j in 1:_blocksize, k in 1:nbBlocks]
    v = [cos(i * j + k) for i in 1:_blocksize, j in 1:_blocksize, k in 1:nbBlocks]
    causality = Acausal()
    triangle = BlockTriangularLowRankMatrix(u, v, causality)
    @test blocksize(triangle) == _blocksize
    @test size(triangle) == (nbBlocks * _blocksize, nbBlocks * _blocksize)
    @test NonEquilibriumGreenFunction.getNumberOfBlocks(triangle) == nbBlocks
end

@testitem "mul" begin
    import NonEquilibriumGreenFunction.BlockTriangularLowRankMatrix
    using LinearAlgebra
    for T in [Float64, ComplexF32, ComplexF64]
        tol = 100 * max(1E-6, eps(real(T)))
        _blocksize = 3
        nbBlocks = 10
        u = [T(sin(i * j + k)) for i in 1:_blocksize, j in 1:_blocksize, k in 1:nbBlocks]
        v = [T(cos(i * j + k)) for i in 1:_blocksize, j in 1:_blocksize, k in 1:nbBlocks]
        for causality in [Acausal(), Retarded(), Advanced()]
            triangle = BlockTriangularLowRankMatrix(u, v, causality)
            referenceMatrix = zeros(T, size(triangle))
            for p = 1:nbBlocks
                for q = 1:nbBlocks
                    referenceMatrix[blockrange(p, _blocksize), blockrange(q, _blocksize)] =
                        NonEquilibriumGreenFunction.mask(causality, p, q) * u[:, :, p] * v[:, :, q]
                end
            end
            x = randn(T, nbBlocks * _blocksize, 2)
            result_mul = NonEquilibriumGreenFunction._mul(triangle, x)
            reference_mul = referenceMatrix * x
            @test size(x) == size(result_mul)
            @test norm(result_mul - reference_mul) < tol
        end
    end
end

@testitem "Adjoint mul" begin
    import NonEquilibriumGreenFunction.BlockTriangularLowRankMatrix
    using LinearAlgebra
    for T in [Float64, ComplexF32, ComplexF64]
        tol = 100 * max(1E-6, eps(real(T)))
        _blocksize = 3
        nbBlocks = 10
        u = [T(sin(i * j + k)) for i in 1:_blocksize, j in 1:_blocksize, k in 1:nbBlocks]
        v = [T(cos(i * j + k)) for i in 1:_blocksize, j in 1:_blocksize, k in 1:nbBlocks]
        for causality in [Acausal(), Retarded(), Advanced()]
            triangle = BlockTriangularLowRankMatrix(u, v, causality)
            referenceMatrix = zeros(T, size(triangle))
            for p = 1:nbBlocks
                for q = 1:nbBlocks
                    referenceMatrix[blockrange(p, _blocksize), blockrange(q, _blocksize)] =
                        NonEquilibriumGreenFunction.mask(causality, p, q) * u[:, :, p] * v[:, :, q]
                end
            end
            x = randn(T, nbBlocks * _blocksize, 2)
            result_mul = NonEquilibriumGreenFunction._cmul(triangle, x)
            reference_mul = referenceMatrix' * x
            @test size(x) == size(result_mul)
            @test norm(result_mul - reference_mul) < tol
        end
    end
end

@testitem "Elements access" begin
    import NonEquilibriumGreenFunction.BlockTriangularLowRankMatrix
    using LinearAlgebra
    for T in [Float64, ComplexF32, ComplexF64]
        tol = 100 * max(1E-6, eps(real(T)))
        _blocksize = 3
        nbBlocks = 10
        u = [T(sin(i * j + k)) for i in 1:_blocksize, j in 1:_blocksize, k in 1:nbBlocks]
        v = [T(cos(i * j + k)) for i in 1:_blocksize, j in 1:_blocksize, k in 1:nbBlocks]
        for causality in [Acausal(), Retarded(), Advanced()]
            triangle = BlockTriangularLowRankMatrix(u, v, causality)
            referenceMatrix = zeros(T, size(triangle))
            for p = 1:nbBlocks
                for q = 1:nbBlocks
                    referenceMatrix[blockrange(p, _blocksize), blockrange(q, _blocksize)] =
                        NonEquilibriumGreenFunction.mask(causality, p, q) * u[:, :, p] * v[:, :, q]
                end
            end
            I = 1:2:size(triangle)[1]
            J = 2:3:size(triangle)[2]
            @test size(triangle[I,J])  ==  size(referenceMatrix[I,J])
            @test norm(triangle[I,J] - referenceMatrix[I,J]) < tol
        end
    end
end

