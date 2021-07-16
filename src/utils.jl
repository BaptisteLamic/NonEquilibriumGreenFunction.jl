@inline blockrange(i,bs) = (i-1)*bs+1:i*bs
@inline function blockindex(p::Number,bs)
    a,i = divrem(p-1,bs)
    return (a+1,i+1)
end
function blockindex(tp,bs)
    r_block = zeros(eltype(tp),length(tp))
    r_inblock = zeros(eltype(tp),length(tp))
    for i = 1:length(tp)
        b, ib = blockindex(tp[i],bs)
        (r_block[i],r_inblock[i]) = blockindex(tp[i],bs)
    end
    return (r_block,r_inblock)
end


function _extract_blockdiag(m::AbstractMatrix{T},bs,d::Int = 0) where T
    @assert size(m,1) == size(m,2)
    N = div(size(m,1), bs)
    if abs(d) >= N
        return spzeros(T,size(m)...)
    else
        shift_I = d < 0 ? abs(d) : 0
        shift_J = d > 0 ? d : 0
        J = [(blk+shift_J-1)*bs+j for j = 1:bs,p = 1:bs, blk = 1:N-abs(d) ]
        J = reshape(J,:)
        I = [(blk+shift_I-1)*bs+j for p = 1:bs, j = 1:bs, blk = 1:N-abs(d) ]
        I = reshape(I,:)
        V = [m[I[k],J[k]] for k = 1:length(I)]
        return _adapt(m,sparse(I,J,V,size(m)...))
    end
end
function extract_blockdiag(m::AbstractMatrix{T},bs,d = 0; compressor = HssCompressor()) where T
    return sum(_extract_blockdiag(m,bs,_d) for _d in d) |> compressor
end

#=
function blockdiag(A::AbstractArray{T,3};compressor = HssCompressor()) where T
    @assert size(A,1) == size(A,2)
    bs = size(A,1)
    N = size(A,3)
    I = [(blk-1)*bs+j for p = 1:bs, j = 1:bs, blk = 1:N ]
    I = reshape(I,:)
    J = [(blk-1)*bs+j for j = 1:bs,p = 1:bs, blk = 1:N ]
    J = reshape(J,:)
    V = Array{T,1}(undef,length(I))
    Threads.@threads for k = 1:length(I)
        _,i = blockindex(I[k],bs)
        blk,j = blockindex(J[k],bs)
        V[k] = A[i,j,blk]
    end
    return compressor(sparse(I,J,V))
end
=#
function blockdiag(A::AbstractArray{T,3}, d::Integer = 0;compressor = HssCompressor()) where T
    shift_I = d < 0 ? abs(d) : 0
    shift_J = d > 0 ? d : 0
    @assert size(A,1) == size(A,2)
    bs = size(A,1)
    N = size(A,3)
    I = [(t+shift_I-1)*bs+ib for ib in 1:bs, jb in 1:bs, t = 1:N]
    J = [(t+shift_J-1)*bs+jb for ib in 1:bs, jb in 1:bs, t = 1:N]
    return sparse(I[:],J[:],A[:],(N+abs(d))*bs,(N+abs(d))*bs) |> hss
end
function blockdiag(A::AbstractArray{<:AbstractMatrix{T},1}, d::Integer = 0;compressor = HssCompressor()) where T
    _A = zeros(T,size(A[1])..., length(A))
    for p = 1:length(A)
        for j = 1:size(A[1],2)
            for i = 1:size(A[1],1)
                _A[i,j,p] = A[p][i,j]
            end
        end
    end
    blockdiag(_A,d,compressor = compressor)
end
function row(A::AbstractMatrix,r,bs = 1)
    J = reshape([j for i in (r-1)*bs+1:r*bs, j in 1:size(A,2)], :)
    I = reshape([i for i in (r-1)*bs+1:r*bs, j in 1:size(A,2)], :)
    V = reshape(A[(r-1)*bs+1:r*bs,:],:)
    return _adapt(A,sparse(I, J, V,size(A)... ))
end

function col(A::AbstractMatrix,c,bs = 1)
    J = reshape([j for i in 1:size(A,1), j in (c-1)*bs+1:c*bs], :)
    I = reshape([i for i in 1:size(A,1), j in (c-1)*bs+1:c*bs], :)
    V = reshape(A[:,(c-1)*bs+1:c*bs],:)
    return _adapt(A,sparse(I, J, V,size(A) ... ))
end


function _adapt(::HssMatrix,a)
    return hss(a)
end
function _adapt(::G,a) where {G<:AbstractArray}
    return G(a)
end