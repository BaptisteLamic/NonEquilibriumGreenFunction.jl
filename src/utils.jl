
@inline blockrange(i,bs) = (i-1)*bs+1:i*bs
@inline function blockindex(p::Number,bs)
    a,i = divrem(p-1,bs)
    return (a+1,i+1)
end
function blockindex(tp,bs)
    r_block = zeros(eltype(tp),length(tp))
    r_inblock = zeros(eltype(tp),length(tp))
    for i = 1:length(tp)
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
        return sparse(I,J,V,size(m)...)
    end
end
function extract_blockdiag(m::AbstractMatrix{T},bs,d = 0; compression = HssCompression()) where T
    return sum(_extract_blockdiag(m,bs,_d) for _d in d) |> compression
end
function blockdiag(A::AbstractArray{T,3}, d::Integer = 0;compression = HssCompression()) where T
    shift_I = d < 0 ? abs(d) : 0
    shift_J = d > 0 ? d : 0
    @assert size(A,1) == size(A,2)
    bs = size(A,1)
    N = size(A,3)
    I = [(t+shift_I-1)*bs+ib for ib in 1:bs, jb in 1:bs, t = 1:N]
    J = [(t+shift_J-1)*bs+jb for ib in 1:bs, jb in 1:bs, t = 1:N]
    return sparse(I[:],J[:],A[:],(N+abs(d))*bs,(N+abs(d))*bs) |> compression
end
function blockdiag(A::AbstractArray{<:AbstractMatrix{T},1}, d::Integer = 0;compression = HssCompression()) where T
    _A = zeros(T,size(A[1])..., length(A))
    for p = 1:length(A)
        for j = 1:size(A[1],2)
            for i = 1:size(A[1],1)
                _A[i,j,p] = A[p][i,j]
            end
        end
    end
    blockdiag(_A,d,compression = compression)
end

function _adapt(::HssMatrix,a)
    return hss(a)
end
function _adapt(::G,a) where {G<:AbstractArray}
    return G(a)
end
function _adapt(a::AbstractArray)
    return Array(a)
end
function _adapt(a::HssMatrix)
    return a
end
