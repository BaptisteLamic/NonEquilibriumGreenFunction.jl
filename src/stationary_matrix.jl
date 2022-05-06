struct StationaryBlockMatrix{T,BS} <: AbstractMatrix{T}
    data::Array{T}
end
function StationaryBlockMatrix(x::AbstractArray)
    #the length of the block vector defining the matrix must be odd 
    @assert ( size(x,1) ÷ size(x,2) ) % 2 == 1
    StationaryBlockMatrix{eltype(x),size(x,2)}(x)
end
function size(A::StationaryBlockMatrix)
    n = blocksize(A)*(size(A.data,1) ÷ size(A.data,2) ÷ 2)
    return (n,n)
end
blocksize(A::StationaryBlockMatrix{T,BS}) where {T,BS} = BS  
function getindex(A::StationaryBlockMatrix, I::Vararg{Int,2})
    n = size(A,1) ÷ blocksize(A)
    blck_i, s_i = blockindex(I[1],blocksize(A))
    blck_j, s_j = blockindex(I[2],blocksize(A))
    blck = blck_i-blck_j + n
    A.data[blck*blocksize(A)+s_i,s_j]
end
function _pad_for_convolution(x,bs)
    nz = length(x)*bs
    r = similar(x,size(x,1) + nz, size(x,2))
    r[1:length(x),:] .= x
    r[length(x)+1:end,:] .= 0
    return r
end
function _conv(A::StationaryBlockMatrix,x)
    padded_x = _pad_for_convolution(x,blocksize(A))
    #TODO 
end