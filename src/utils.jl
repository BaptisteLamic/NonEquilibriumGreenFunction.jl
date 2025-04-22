
@inline blockrange(i, bs) = (i-1)*bs+1:i*bs
@inline function blockindex(p::Number, bs)
    a, i = divrem(p - 1, bs)
    return (a + 1, i + 1)
end
function blockindex(tp, bs)
    r_block = zeros(eltype(tp), length(tp))
    r_inblock = zeros(eltype(tp), length(tp))
    for i = 1:length(tp)
        (r_block[i], r_inblock[i]) = blockindex(tp[i], bs)
    end
    return (r_block, r_inblock)
end
function sparse_extract_blockdiag(m::AbstractMatrix{T}, bs, diagonalIndices=0) where {T}
    @assert size(m, 1) == size(m, 2) "Matrix must be square to extract block diagonal"
    N = div(size(m, 1), bs)
    I = Vector{Int}()
    J = Vector{Int}()
    for d in diagonalIndices
        if abs(d) <= N
            shift_I = d < 0 ? abs(d) : 0
            shift_J = d > 0 ? d : 0
            newJ = [(blk + shift_J - 1) * bs + j for j = 1:bs, p = 1:bs, blk = 1:N-abs(d)]
            J = [J; reshape(newJ, :)]
            newI = [(blk + shift_I - 1) * bs + j for p = 1:bs, j = 1:bs, blk = 1:N-abs(d)]
            I = [I; reshape(newI, :)]
        end
    end
    IJ = sort([ij for ij in zip(I, J)])
    V = Vector{T}(undef, length(IJ))
    for i in eachindex(IJ)
        @inbounds V[i] = m[IJ[i][1], IJ[i][2]]
    end
    V = [m[ij[1], ij[2]] for ij in IJ]
    #=while i<=length(I)
        next = findnext(Ip-> Ip != IJ[i], IJ, i+1)
        next_i = isnothing(next) ? length(IJ)+1 : Int(next)
        V[i:next_i-1] = @views m[I[i:next_i-1], J[i:next_i-1]]
        i = next_i
    end=#
    return sparse(I, J, V, size(m)...)
end
function extract_blockdiag(m::AbstractMatrix{T}, bs, d=0) where {T}
    return sparse_extract_blockdiag(m, bs, d)
end
function blockdiag(A::AbstractArray{T,3}, d::Integer=0; compression=HssCompression()) where {T}
    shift_I = d < 0 ? abs(d) : 0
    shift_J = d > 0 ? d : 0
    @assert size(A, 1) == size(A, 2)
    bs = size(A, 1)
    N = size(A, 3)
    I = [(t + shift_I - 1) * bs + ib for ib in 1:bs, jb in 1:bs, t = 1:N]
    J = [(t + shift_J - 1) * bs + jb for ib in 1:bs, jb in 1:bs, t = 1:N]
    return sparse(I[:], J[:], A[:], (N + abs(d)) * bs, (N + abs(d)) * bs) |> compression
end
function blockdiag(A::AbstractArray{<:AbstractMatrix{T},1}, d::Integer=0; compression=HssCompression()) where {T}
    _A = zeros(T, size(A[1])..., length(A))
    for p = 1:length(A)
        @assert size(A[p], 1) == size(A[p], 2)
        @assert size(A[p]) == size(A[1])
        @simd for j = 1:size(A[1], 2)
            @simd for i = 1:size(A[1], 1)
                _A[i, j, p] = A[p][i, j]
            end
        end
    end
    blockdiag(_A, d, compression=compression)
end

function _adapt(::HssMatrix, a)
    return hss(a)
end
function _adapt(::G, a) where {G<:AbstractArray}
    return G(a)
end
function _adapt(a::AbstractArray)
    return Array(a)
end
function _adapt(a::HssMatrix)
    return a
end
