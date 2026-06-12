
"""
    build_CirculantlinearMap(ax0, f)

Build a circulant linear map from a function f evaluated on a shifted axis.

# Arguments
- `ax0`: Original time axis
- `f`: Function f(t, tp) that returns a block matrix

# Returns
A LinearMap representing the circulant matrix structure.

# Throws
AssertionError if f does not return square matrices.
"""
function build_CirculantlinearMap(ax0, f)
    f00 = f(ax0[1], ax0[1])
    T = eltype(f00)
    bs = size(f00, 1)
    N = length(ax0)
    @assert size(f00, 1) == size(f00, 2) "Function f must return square matrices"
    ax = (1-N:N-1) * step(ax0)
    #(ax0[1]-(N-1)*step(ax0)):step(ax0):ax0[end]

    m = zeros(eltype(f00), bs, bs, length(ax))
    Threads.@threads for i in eachindex(ax)
        m[:, :, i] .= f(ax[i], 0)
    end
    return build_linearMap(BlockCirculantMatrix(m))
end

"""
    build_linearMap(A::AbstractMatrix)

Build a LinearMap from an AbstractMatrix for use with HSS compression.

# Arguments
- `A`: Matrix to wrap in a LinearMap

# Returns
A LinearMap that wraps the matrix operations of A.

# Note
This creates a LinearMap with custom mul!, cmul!, and getindex operations
that delegate to the corresponding methods of A.
"""
function build_linearMap(A::AbstractMatrix)
    T = eltype(A)
    my_mul!(y, _, x) = _mul!(y, A, x)
    my_cmul!(y, _, x) = _cmul!(y, A, x)
    my_getindex(I, J) = getindex(A, I, J)
    return LinearMap{T}(size(A, 1), size(A, 2), my_mul!, my_cmul!, my_getindex)
end

"""
    build_CirculantlinearMap(A::BlockCirculantMatrix)

Build a LinearMap from a BlockCirculantMatrix.
"""
build_CirculantlinearMap(A::BlockCirculantMatrix) = build_linearMap(A)

"""
    build_triangularLowRankMap(A::BlockTriangularLowRankMatrix)

Build a LinearMap from a BlockTriangularLowRankMatrix.
"""
build_triangularLowRankMap(A::BlockTriangularLowRankMatrix) = build_linearMap(A)

"""
    build_linearMap(axis, f; blk=512)

Build a LinearMap from a function f defined on an axis, suitable for HSS compression.

# Arguments
- `axis`: Time axis for discretization
- `f`: Function f(t, tp) that returns a block matrix
- `blk`: Block size for matrix-vector multiplication (default: 512)

# Returns
A LinearMap representing the matrix defined by f on the axis.

# Throws
AssertionError if f does not return square matrices.

# Note
This function creates a lazy LinearMap that computes matrix elements on-demand
using the function f. This is useful for large matrices that cannot be stored explicitly.
The `blk` parameter controls the blocking for matrix-vector multiplication.
"""
function build_linearMap(axis, f; blk=512)
    f00 = f(axis[1], axis[1])
    @assert size(f00, 1) == size(f00, 2) "Function f must return square matrices"
    T = eltype(f00)
    bs = size(f00, 1)
    N = length(axis) * bs

    # Type-stable getindex implementations (borrowed from HssMatrices)
    @inline getindex(i::Int, j::Int) = getindex([i], [j])[1]
    @inline getindex(i::Int, j::AbstractRange) = reshape(getindex([i], j), :)
    @inline getindex(i::AbstractRange, j::Int) = reshape(getindex(i, [j]), :)
    @inline getindex(i::AbstractRange, j::AbstractRange) = getindex(collect(i), collect(j))
    @inline getindex(::Colon, ::Colon) = getindex(1:N, 1:N)

    @inline function _getindex!(r, I, J)
        #Assume that indices are sorted
        _bI, sI = blockindex(I, bs)
        _bJ, sJ = blockindex(J, bs)
        bI, bI_count = rle(_bI)
        bJ, bJ_count = rle(_bJ)
        offset_I = cumsum(bI_count)
        offset_J = cumsum(bJ_count)
        Threads.@threads for idx_bj in eachindex(bJ)
            for idx_bi in eachindex(bI)
                blck = f(axis[bI[idx_bi]], axis[bJ[idx_bj]])
                for j = offset_J[idx_bj]+1-bJ_count[idx_bj]:offset_J[idx_bj]
                    for i = offset_I[idx_bi]+1-bI_count[idx_bi]:offset_I[idx_bi]
                        r[i, j] = blck[sI[i], sJ[j]]
                    end
                end
            end
        end
        return r
    end

    @inline function getindex(I::Vector{Int}, J::Vector{Int})
        r = Matrix{T}(undef, length(I), length(J))
        Ip = sortperm(I)
        Jp = sortperm(J)
        if (length(I) == 0 || length(J) == 0)
            return Matrix{T}(undef, length(I), length(J))
        end
        return _getindex!(r, I[Ip], J[Jp])[invperm(Ip), invperm(Jp)]
    end

    @inline function _mul!(y, _, x)
        op = Matrix{T}(undef, min(blk, N), N)
        for p = 1:div(N, blk)
            _getindex!(op, (p-1)*blk+1:p*blk, 1:N)
            y[(p-1)*blk+1:p*blk, :] = op * x
        end
        if N % blk != 0
            i0 = div(N, blk) * blk
            op2 = @view op[1:(N-i0), :]
            op2 = _getindex!(op2, i0+1:N, 1:N)
            y[i0+1:N, :] = op2 * x
        end
        return y
    end

    @inline function _cmul!(y, _, x)
        op = Matrix{T}(undef, N, min(blk, N))
        for p = 1:div(N, blk)
            _getindex!(op, 1:N, (p-1)*blk+1:p*blk)
            y[(p-1)*blk+1:p*blk, :] = op' * x
        end
        if N % blk != 0
            i0 = div(N, blk) * blk
            op2 = @view op[:, 1:(N-i0)]
            op2 = _getindex!(op2, 1:N, i0+1:N)
            y[i0+1:N, :] = op2' * x
        end
        return y
    end
    lm = LinearMap{T}(N, N, _mul!, _cmul!, getindex)
    return lm
end

"""
    AbstractCompression

Abstract type for compression methods.
"""
abstract type AbstractCompression end

"""
    HssCompression <: AbstractCompression

HSS (Hierarchically Semi-Separable) matrix compression parameters.

# Fields
- `atol::Float64`: Absolute tolerance for compression
- `rtol::Float64`: Relative tolerance for compression
- `kest::Int`: Estimated rank for compression
- `leafsize::Int`: Leaf size for the HSS tree
"""
struct HssCompression <: AbstractCompression
    atol::Float64
    rtol::Float64
    kest::Int
    leafsize::Int
end

"""
    HssCompression(; atol=1E-4, rtol=1E-4, kest=20, leafsize=32)

Construct an HssCompression object with specified parameters.

# Arguments
- `atol`: Absolute tolerance (default: 1e-4)
- `rtol`: Relative tolerance (default: 1e-4)
- `kest`: Estimated rank (default: 20)
- `leafsize`: Leaf size for HSS tree (default: 32)
"""
function HssCompression(; atol=1E-4, rtol=1E-4, kest=20, leafsize=32)
    return HssCompression(atol, rtol, kest, leafsize)
end

"""
    NONCompression <: AbstractCompression

No compression - returns full dense matrices.
"""
struct NONCompression <: AbstractCompression end


"""
    (Compression::HssCompression)(axis, f; stationary=false)

Compress a function-defined matrix using HSS compression.

# Arguments
- `axis`: Time axis for discretization
- `f`: Function f(t, tp) that returns a block matrix
- `stationary`: If true, use circulant structure (default: false)

# Returns
An HssMatrix containing the compressed representation.

# Throws
AssertionError if f does not return square matrices.
"""
function (Compression::HssCompression)(axis, f; stationary=false)
    f00 = f(axis[1], axis[1])
    @assert size(f00, 1) == size(f00, 2) "Function f must return square matrices"
    lm = stationary ? build_CirculantlinearMap(axis, f) : build_linearMap(axis, f)
    bs = size(f00, 1)
    cc = bisection_cluster(length(axis) * bs, leafsize=Compression.leafsize)
    r = randcompress_adaptive(lm, cc, cc, atol=Compression.atol, rtol=Compression.rtol, kest=Compression.kest, leafsize=Compression.leafsize)
    return r
end

"""
    (Compression::HssCompression)(tab::HssMatrix)

Recompress an existing HssMatrix with the compression parameters.

# Arguments
- `tab`: HssMatrix to recompress

# Returns
A recompressed HssMatrix.
"""
function (Compression::HssCompression)(tab::HssMatrix)
    return recompress!(tab, atol=Compression.atol, rtol=Compression.rtol, leafsize=Compression.leafsize)
end

"""
    (Compression::HssCompression)(tab::AbstractMatrix{T}) where {T<:Number}

Compress a dense matrix using HSS compression.

# Arguments
- `tab`: Dense matrix to compress

# Returns
An HssMatrix containing the compressed representation.
"""
function (Compression::HssCompression)(tab::AbstractMatrix{T}) where {T<:Number}
    return hss(tab, atol=Compression.atol, rtol=Compression.rtol, leafsize=Compression.leafsize)
end

"""
    (Compression::HssCompression)(axis, tab::BlockCirculantMatrix)

Compress a BlockCirculantMatrix using HSS compression.

# Arguments
- `axis`: Time axis
- `tab`: BlockCirculantMatrix to compress

# Returns
An HssMatrix containing the compressed representation.
"""
function (Compression::HssCompression)(axis, tab::BlockCirculantMatrix)
    lm = build_CirculantlinearMap(tab)
    cc = bisection_cluster(size(tab, 1), leafsize=Compression.leafsize)
    r = randcompress_adaptive(lm, cc, cc, atol=Compression.atol, rtol=Compression.rtol, kest=Compression.kest)
    return r
end

"""
    (Compression::NONCompression)(axis, f; stationary=false)

Build a full dense matrix without compression.

# Arguments
- `axis`: Time axis for discretization
- `f`: Function f(t, tp) that returns a block matrix
- `stationary`: Ignored for NONCompression (default: false)

# Returns
A dense matrix of size (bs*length(axis), bs*length(axis)).

# Throws
AssertionError if f does not return square matrices.
"""
function (Compression::NONCompression)(axis, f; stationary=false)
    f00 = f(axis[1], axis[1])
    @assert size(f00, 1) == size(f00, 2) "Function f must return square matrices"
    bs = size(f00, 1)
    r = Array{eltype(f00),2}(undef, bs * length(axis), bs * length(axis))
    for it in eachindex(axis)
        for itp in eachindex(axis)
            r[blockrange(it, bs), blockrange(itp, bs)] .= f(axis[it], axis[itp])
        end
    end
    return r
end

"""
    (Compression::NONCompression)(axis, tab::BlockCirculantMatrix)

Convert BlockCirculantMatrix to a dense matrix.

# Arguments
- `axis`: Time axis (unused but kept for API consistency)
- `tab`: BlockCirculantMatrix to convert

# Returns
A dense matrix representation.
"""
function (Compression::NONCompression)(axis, tab::BlockCirculantMatrix)
    return tab[:, :]
end

"""
    (Compression::NONCompression)(tab::AbstractMatrix{T}) where {T<:Number}

Convert any matrix to a dense Matrix.

# Arguments
- `tab`: Matrix to convert

# Returns
A dense Matrix.
"""
function (Compression::NONCompression)(tab::AbstractMatrix{T}) where {T<:Number}
    return Matrix(tab)
end

"""
    (Compression::NONCompression)(axis, f, g)

Build a dense matrix from two functions f and g, computing f(t) * g(tp).

# Arguments
- `axis`: Time axis for discretization
- `f`: Function of time t
- `g`: Function of time tp

# Returns
A dense matrix where element (i,j) = f(axis[i]) * g(axis[j]).
"""
function (Compression::NONCompression)(axis, f, g)
    return Compression(axis, (t, tp) -> f(t) * g(tp))
end

"""
    triangularLowRankCompression(compression::AbstractCompression, causality, axis, f, g)

Compress a low-rank kernel f(t) * g(tp) with triangular masking using the specified compression.

# Arguments
- `compression`: Compression method to use
- `causality`: Causality type (Retarded, Advanced, or Acausal)
- `axis`: Time axis for discretization
- `f`: Function of time t
- `g`: Function of time tp

# Returns
A compressed matrix representation.

# Throws
AssertionError if f or g do not return square matrices.
"""
function triangularLowRankCompression(compression::AbstractCompression, causality, axis, f, g)
    f00 = f(axis[1])
    @assert size(f00, 1) == size(f00, 2) "Functions f and g must return square matrices"
    fg(t, tp) = f(t) * g(tp)
    _mask(::Retarded) = (x, y) -> x >= y ? fg(x, y) : zero(f00)
    _mask(::Advanced) = (x, y) -> x <= y ? fg(x, y) : zero(f00)
    _mask(::Acausal) = (x, y) -> fg(x, y)
    fg_masked = _mask(causality)
    return compression(axis, fg_masked)
end

"""
    triangularLowRankCompression(compression::HssCompression, causality, axis, f, g)

Compress a low-rank kernel using HSS compression with optimized BlockTriangularLowRankMatrix.

# Arguments
- `compression`: HssCompression parameters
- `causality`: Causality type (Retarded, Advanced, or Acausal)
- `axis`: Time axis for discretization
- `f`: Function of time t
- `g`: Function of time tp

# Returns
An HssMatrix containing the compressed representation.

# Throws
AssertionError if f or g do not return square matrices.
"""
function triangularLowRankCompression(compression::HssCompression, causality, axis, f, g)
    f00 = f(axis[1])
    @assert size(f00, 1) == size(f00, 2) "Functions f and g must return square matrices"
    blocksize = size(f00, 1)
    u = zeros(eltype(f00), blocksize, blocksize, length(axis))
    v = zeros(eltype(f00), blocksize, blocksize, length(axis))
    for k in 1:length(axis)
        u[:, :, k] .= f(axis[k])
        v[:, :, k] .= g(axis[k])
    end
    matrix = BlockTriangularLowRankMatrix(u, v, causality)
    lm = build_triangularLowRankMap(matrix)
    cc = bisection_cluster(size(matrix, 1), leafsize=compression.leafsize)
    r = randcompress_adaptive(lm, cc, cc, atol=compression.atol, rtol=compression.rtol, kest=compression.kest, leafsize=compression.leafsize)
    return r
end

"""
    computeMatrixNorm(matrix)

Compute the Frobenius norm of a matrix.

# Arguments
- `matrix`: Matrix to compute norm of

# Returns
The Frobenius norm of the matrix.
"""
function computeMatrixNorm(matrix)
    return norm(matrix)
end

"""
    computeMatrixNorm(matrix::HssMatrix)

Compute the Frobenius norm of an HssMatrix efficiently.

# Arguments
- `matrix`: HssMatrix to compute norm of

# Returns
The Frobenius norm of the matrix.

# Note
This implementation avoids materializing the full matrix for HSS matrices.
"""
function computeMatrixNorm(matrix::HssMatrix)
    norm2 = real(tr(matrix' * matrix))
    if norm2 <= 0
        return zero(norm2)
    else
        return sqrt(norm2)
    end
end


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
            for k in 1:length(axis)
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
        @test computeMatrixNorm(A) - computeMatrixNorm(hssA) < tol
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