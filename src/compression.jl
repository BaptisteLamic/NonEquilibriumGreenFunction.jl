
function build_CirculantlinearMap(ax0, f)
    f00 = f(ax0[1], ax0[1])
    T = eltype(f00)
    bs = size(f00, 1)
    N = length(ax0)
    @assert size(f00, 1) == size(f00, 2)
    ax = (1-N:N-1) * step(ax0)
    #(ax0[1]-(N-1)*step(ax0)):step(ax0):ax0[end]

    m = zeros(eltype(f00), bs, bs, length(ax))
    Threads.@threads for i = 1:length(ax)
        m[:, :, i] .= f(ax[i], 0)
    end
    A = BlockCirculantMatrix(m)
    return build_CirculantlinearMap(A::BlockCirculantMatrix)
end

function build_CirculantlinearMap(A::BlockCirculantMatrix)
    my_mul!(y, _, x) = _mul!(y, A, x)
    my_cmul!(y, _, x) = _cmul!(y, A, x)
    my_getindex(I, J) = getindex(A, I, J)
    lm = LinearMap{eltype(A)}(size(A, 1), size(A, 2), my_mul!, my_cmul!, my_getindex)
    return lm
end
function build_triangularLowRankMap(A::BlockTriangularLowRankMatrix)
    my_mul!(y, _, x) = _mul!(y, A, x)
    my_cmul!(y, _, x) = _cmul!(y, A, x)
    my_getindex(I, J) = getindex(A, I, J)
    lm = LinearMap{eltype(A)}(size(A, 1), size(A, 2), my_mul!, my_cmul!, my_getindex)
    return lm
end

function build_linearMap(axis, f; blk=512)
    f00 = f(axis[1], axis[1])
    @assert size(f00, 1) == size(f00, 2)
    T = eltype(f00)
    bs = size(f00, 1)
    N = length(axis) * bs

    #borrowed from HssMatrices
    getindex(i::Int, j::Int) = getindex([i], [j])[1]
    getindex(i::Int, j::AbstractRange) = reshape(getindex([i], j), :)
    getindex(i::AbstractRange, j::Int) = reshape(getindex(i, [j]), :)
    getindex(i::AbstractRange, j::AbstractRange) = getindex(collect(i), collect(j))
    getindex(::Colon, ::Colon) = getindex(1:N, 1:N)
    getindex(i, ::Colon) = getindex(i, 1:size(hssA, 2))
    getindex(::Colon, j) = getindex(1:size(hssA, 1), j)

    function _getindex!(r, I, J)
        #Assume that indices are sorted
        _bI, sI = blockindex(I, bs)
        _bJ, sJ = blockindex(J, bs)
        bI, bI_count = rle(_bI)
        bJ, bJ_count = rle(_bJ)
        offset_I = cumsum(bI_count)
        offset_J = cumsum(bJ_count)
        Threads.@threads for idx_bj in 1:length(bJ)
            for idx_bi in 1:length(bI)
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

    function getindex(I::Vector{Int}, J::Vector{Int})
        r = Matrix{T}(undef, length(I), length(J))
        Ip = sortperm(I)
        Jp = sortperm(J)
        if (length(I) == 0 || length(J) == 0)
            return Matrix{T}(undef, length(I), length(J))
        end
        return _getindex!(r, I[Ip], J[Jp])[invperm(Ip), invperm(Jp)]
    end

    function _mul!(y, _, x)
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

    function _cmul!(y, _, x)
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

abstract type AbstractCompression end

struct HssCompression <: AbstractCompression
    atol::Float64
    rtol::Float64
    kest::Int
    leafsize::Int
end
HssCompression(; atol=1E-4, rtol=1E-4, kest=20, leafsize=32) = HssCompression(atol, rtol, kest, leafsize)
struct NONCompression <: AbstractCompression end


function (Compression::HssCompression)(axis, f; stationary=false)
    lm = stationary ? build_CirculantlinearMap(axis, f) : build_linearMap(axis, f)
    bs = size(f(axis[1], axis[1]), 1)
    cc = bisection_cluster(length(axis) * bs, leafsize=Compression.leafsize)
    r = randcompress_adaptive(lm, cc, cc, atol=Compression.atol, rtol=Compression.rtol, kest=Compression.kest, leafsize=Compression.leafsize)
    recompress!(r, atol=Compression.atol, rtol=Compression.rtol, leafsize=Compression.leafsize)
    return r
end
function (Compression::HssCompression)(axis, f, g)
    f00 = f(axis[1])
    T, bs = eltype(f00), size(f00, 1)
    u = Array{T}(undef, (bs, bs, length(axis)))
    v = Array{T}(undef, (bs, bs, length(axis)))
    for k in 1:length(axis)
        u[:, :, k] .= f(axis[k])
        v[:, :, k] .= g(axis[k])
    end
    n = length(axis) * bs
    u = reshape(u, (n, bs))
    v = reshape(u, (n, bs))
    cc = bisection_cluster(length(axis) * bs, leafsize=Compression.leafsize)
    return lowrank2hss(u, v, cc, cc)
end
function (Compression::HssCompression)(tab::HssMatrix)
    return recompress!(tab, atol=Compression.atol, rtol=Compression.rtol, leafsize=Compression.leafsize)
end
function (Compression::HssCompression)(tab::AbstractMatrix{T}) where {T<:Number}
    return hss(tab, atol=Compression.atol, rtol=Compression.rtol, leafsize=Compression.leafsize)
end
function (Compression::HssCompression)(axis, tab::BlockCirculantMatrix)
    lm = build_CirculantlinearMap(tab)
    cc = bisection_cluster(size(tab, 1), leafsize=Compression.leafsize)
    r = randcompress_adaptive(lm, cc, cc, atol=Compression.atol, rtol=Compression.rtol, kest=Compression.kest)
    recompress!(r, atol=Compression.atol, rtol=Compression.rtol, leafsize=Compression.leafsize)
end

function (Compression::NONCompression)(axis, f; stationary=false)
    f00 = f(axis[1], axis[1])
    bs = size(f00, 1)
    r = Array{eltype(f00),2}(undef, bs * length(axis), bs * length(axis))
    for it in 1:length(axis)
        for itp in 1:length(axis)
            r[blockrange(it, bs), blockrange(itp, bs)] .= f(axis[it], axis[itp])
        end
    end
    return r
end
function (Compression::NONCompression)(axis, tab::BlockCirculantMatrix)
    return tab[:, :]
end
function (Compression::NONCompression)(tab::AbstractMatrix{T}) where {T<:Number}
    return Matrix(tab)
end
function (Compression::NONCompression)(axis, f, g)
    Compression(axis, (t, tp) -> f(t) * g(tp))
end

function triangularLowRankCompression(compression::AbstractCompression, causality, axis, f, g)
    f00 = f(axis[1])
    @assert size(f00, 1) == size(f00, 2)
    fg(t, tp) = f(t) * g(tp)
    _mask(::Retarded) = (x, y) -> x >= y ? fg(x, y) : zero(f00)
    _mask(::Advanced) = (x, y) -> x <= y ? fg(x, y) : zero(f00)
    _mask(::Acausal) = (x, y) -> fg(x, y)
    fg_masked = _mask(causality)
    return compression(axis, fg_masked)
end

function triangularLowRankCompression(compression::HssCompression, causality, axis, f, g)
    f00 = f(axis[1])
    @assert size(f00, 1) == size(f00, 2)
    blocksize = size(f00, 1)
    u = zeros(eltype(f00), blocksize, blocksize, length(axis))
    v = zeros(eltype(f00), blocksize, blocksize, length(axis))
    for k in 1:length(axis)
        u[:, :, k] .= f(axis[k])
        v[:, :, k] .= g(axis[k])
    end
    matrix = BlockTriangularLowRankMatrix(u, v, causality)
    lm = build_triangularLowRankMap(matrix)
    @show size(matrix, 1)
    cc = bisection_cluster(size(matrix, 1), leafsize=compression.leafsize)
    r = randcompress_adaptive(lm, cc, cc, atol=compression.atol, rtol=compression.rtol, kest=compression.kest, leafsize=compression.leafsize)
    recompress!(r, atol=compression.atol, rtol=compression.rtol, leafsize=compression.leafsize)
    return r
end

@testitem "TriangularLowRankCompression_map" begin
    using LinearAlgebra
    using NonEquilibriumGreenFunction
    N, Dt = 128, 2.0
    axis = LinRange(-Dt / 2, Dt, N)
    for causality in (Acausal(), Retarded(), Advanced())
        for T = [Float64, ComplexF64, ComplexF32]
            tol = 5 * max(1E-14, eps(real(T)))
            f(x) = T <: Complex ? T(exp(-1im * x)) : T(cos(x))
            g(x) = T <: Complex ? T(exp(-4im * x)) : T(sin(4x))
            f00 = f( axis[1])
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
            x = randn(T,size(matrix,2))
            @test norm(lm*x - matrix*x) < tol * length(x)
            @test norm(lm'*x - matrix'*x) < tol * length(x)
        end
    end
end