

function build_linearMap(axis,f;blk = 512)
    f00 = f(axis[1],axis[1])
    @assert size(f00,1) == size(f00,2)
    T = eltype(f00)
    bs = size(f00,1)
    N = length(axis)*bs

    #borrowed from HssMatrices
    getindex(i::Int, j::Int) = getindex([i], [j])[1]
    getindex(i::Int, j::AbstractRange) = getindex([i], j)[:]
    getindex(i::AbstractRange, j::Int) = getindex(i, [j])[:]
    getindex(i::AbstractRange, j::AbstractRange) = getindex(collect(i), collect(j))
    getindex(::Colon, ::Colon) = getindex(1:N,1:N)
    getindex(i, ::Colon) = getindex(i, 1:size(hssA,2))
    getindex(::Colon, j) = getindex(1:size(hssA,1), j)

    function _getindex!(r,I,J)
        #Assume that indices are sorted
        _bI, sI = blockindex(I, bs)
        _bJ, sJ = blockindex(J, bs)
        bI , bI_count = rle(_bI)
        bJ , bJ_count = rle(_bJ)
        offset_I = cumsum(bI_count)
        offset_J = cumsum(bJ_count)
        Threads.@threads for idx_bj in 1:length(bJ)
            #blck = f(axis[ bI[1] ], axis[ bJ[1] ])
            for idx_bi in 1:length(bI)
                blck = f(axis[ bI[idx_bi] ], axis[ bJ[ idx_bj ] ])
                for j  = offset_J[idx_bj]+1-bJ_count[idx_bj]:offset_J[idx_bj]
                    for i  = offset_I[idx_bi]+1-bI_count[idx_bi]:offset_I[idx_bi]
                        r[i,j] = blck[ sI[i], sJ[j] ]
                    end
                end
            end
        end
        return r
    end

    function getindex(I::Vector{Int},J::Vector{Int})
        r = Matrix{T}(undef,length(I),length(J))
        Ip = sortperm(I); Jp = sortperm(J)
        if (length(I) == 0 || length(J) == 0) return Matrix{T}(undef, length(I), length(J)) end
        return _getindex!(r,I[Ip], J[Jp])[invperm(Ip), invperm(Jp)]
    end

    function _mul!(y,_,x)
        op = Matrix{T}(undef, min(blk,N),N)
        for p = 1:div(N,blk)
            _getindex!(op,(p-1)*blk+1:p*blk,1:N)
            y[(p-1)*blk+1:p*blk ,:] = op*x
        end
        if N % blk != 0
            i0 = div(N,blk)*blk
            op2 = @view op[1:(N-i0),:] 
            op2 = _getindex!(op2,i0+1:N,1:N)
            y[i0+1:N,:] = op2*x
        end
        return y
    end
    
    function _cmul!(y,_,x)
        op = Matrix{T}(undef, N, min(blk,N))
        for p = 1:div(N,blk)
            _getindex!(op,1:N,(p-1)*blk+1:p*blk)
            y[(p-1)*blk+1:p*blk ,:] = op'*x
        end
        if N % blk != 0
            i0 = div(N,blk)*blk
            op2 = @view op[:,1:(N-i0)] 
            op2 = _getindex!(op2,1:N,i0+1:N)
            y[i0+1:N,:] = op2'*x
        end
        return y
    end 
    lm = LinearMap{T}(N,N,_mul!,_cmul!, getindex)
    return lm
end

abstract type AbstractCompressor end
struct HssCompressor{T,G} <: AbstractCompressor
    atol::T
    rtol::T
    kest::G
end
HssCompressor(; atol = 1E-6, rtol = 1E-6, kest = 20) = HssCompressor(atol,rtol,kest)
struct NONCompressor <: AbstractCompressor end


function (compressor :: HssCompressor)(axis,f)
    lm = build_linearMap(axis, f)
    bs  = size(f(axis[1],axis[1]),1)
    cc = bisection_cluster(length(axis)*bs)
    r = randcompress_adaptive(lm,cc,cc,atol = compressor.atol, rtol = compressor.rtol, kest = compressor.kest)
    recompress!(r,atol = compressor.atol, rtol = compressor.rtol)
    return r
end 
function (compressor :: HssCompressor)(tab::HssMatrix)
    return recompress!(tab,atol = compressor.atol, rtol = compressor.rtol)
end
function (compressor :: HssCompressor)(tab::AbstractArray)
    return hss(tab,atol = compressor.atol, rtol = compressor.rtol)
end

function (compressor :: NONCompressor)(axis,f)
    f00 = f(axis[1],axis[1]) 
    bs = size(f00,1)
    r = Array{eltype(f00),2}(undef, bs*length(axis),bs*length(axis))
    for it in 1:length(axis)
        for itp in 1:length(axis)
            r[blockrange(it,bs),blockrange(itp,bs)] .= f(axis[it],axis[itp])
        end
    end
    return r
end 
function (compressor :: NONCompressor)(tab::AbstractArray)
    return tab
end

function(compressor::AbstractCompressor)(g::G) where G<:AbstractGreenFunction
    G(axis(g),dirac(g),compressor(regular(g)),blocksize(g))
end