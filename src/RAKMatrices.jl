#abstract type AbstractKernel{T,A, C <: AbstractCompression} <: AbstractArray{AbstractArray{T,2},2} end
struct RAKMatrix{T}  <: AbstractArray{AbstractArray{T,2},2}
    data::Array{AbstractKernel,2}
end
function RAKMatrix(data)
    @assert iscompatible(data[1], data[2]) ||  iscompatible(data[2], data[3]) || iscompatible (data[3], data[4])
    RAKMatrix{eltype(data[1])}(data)
end
function RAKMatrix(A,B,C,D)
    @assert iscompatible(A, B) ||  iscompatible(B, C) || iscompatible (B,D)
    data = Array{AbstractKernel}(undef,2,2)
    data[1,1] = A
    data[1,2] = B
    data[2,1] = C
    data[2,2] = D
    RAKMatrix(data)
end
function RAKMatrix(R,A,K; order)
    @assert iscompatible(R, A) ||  iscompatible(R, K)
    if order == :correlation
        return RAKMatrix(NullKernel(R), A, R, K)
    elseif order == :action
        return RAKMatrix(K, A, R, NullKernel(R))
    else
        throw(ArgumentError("$order is not a valid value for the named argument order."))
    end
end

RAKMatrix(R,K; order) = RAKMatrix(R,R',K, order = order)

function _getindex!(out,A::RAKMatrix, I,J) where T
    #assume that the index are sorted
    sbk = blocksize(A)
    values = [A.data[p,q][I,J] for p = 1:2, q = 1:2]
    for i = 1:length(I)
        for j = 1:length(J) 
            @show [values[1,1][i,j] values[1,2][i,j]; values[2,1][i,j] values[2,2][i,j] ]
            @show out
            out[i,j] = [values[1,1][i,j] values[1,2][i,j]; values[2,1][i,j] values[2,2][i,j] ]
        end
    end
    return out
end

function getindex!(out,A::RAKMatrix,i::Int, j::Int)
    return out .= [ A.data[1,1][i,j] A.data[1,2][i,j] ; A.data[2,1][i,j] A.data[2,2][i,j] ]
end
getindex!(out,A::RAKMatrix,i::Int, j) = getindex!(out,A, [i], j)[:]
getindex!(out,A::RAKMatrix, i, j::Int) = getindex!(out,A, i, [j])[:]
getindex!(out,A::RAKMatrix,::Colon, ::Colon) = getindex!(out,A,1:size(A,1),1:size(A,2))
getindex!(out,A::RAKMatrix, i, ::Colon) = getindex!(out,A,i, 1:size(A,2))
getindex!(out,A::RAKMatrix, ::Colon, j) = getindex!(out,A,1:size(A,1), j)
getindex!(out,A::RAKMatrix, i::Int, ::Colon) = getindex!(out,A,i::Int, 1:size(A,2))[1]
getindex!(out,A::RAKMatrix, ::Colon, j::Int) = getindex!(out,A,1:size(A,1), j::Int)[1]
function getindex!(out,A::RAKMatrix, I,J)
    Ip = sortperm(I); Jp = sortperm(J)
    if (length(I) == 0 || length(J) == 0) 
        return Matrix{eltype(A)}(undef, length(I), length(J))
    else
        return _getindex!(out,A,I[Ip], J[Jp])[invperm(Ip), invperm(Jp)]
    end
end
function getindex(A::RAKMatrix, I,J)
    out = Matrix{eltype(A)}(undef, length(I), length(J) )
    return getindex!(out,A,I,J)
end
function getindex(A::RAKMatrix,i::Int, j::Int)
    return [ A.data[1,1][i,j] A.data[1,2][i,j] ; A.data[2,1][i,j] A.data[2,2][i,j] ]
end
#=
function getindex!(out,A::RAKMatrix,I::Vararg{Int,2})
    return [A.data[i,j][I...] for i = 1:2, j = 1:2]
end
=#
size(A::RAKMatrix) = size(A.data[1,1])
axis(A::RAKMatrix) = axis(A.data[1,1])
blocksize(A::RAKMatrix) = blocksize(A.data[1])

## Define printing functions
function Base.show(io::IO, K::RAKMatrix)
    print(io, "RAKMatrix\n")
end

function Base.show(io::IO, ::MIME"text/plain", k::RAKMatrix)
    show(io, k)
    println(io,"axis = $(axis(k))")
    println(io,"blocksize = $(blocksize(k))")
end

function (*)(A::RAKMatrix,B::RAKMatrix)
    return A.data * B.data |> RAKMatrix
end
function (*)(A::RAKMatrix, B::AbstractMatrix)
    size(B,2) == size(B,1) == 2 ||  throw(DimensionMismatch("B should be a 2 by 2 matrix, or a RAKMatrix"))
    return A.data * B |> RAKMatrix
end
function (*)(A::AbstractMatrix, B::RAKMatrix)
    size(A,2) == size(A,1) == 2 ||  throw(DimensionMismatch("A should be a 2 by 2 matrix, or a RAKMatrix"))
    return A * B.data |> RAKMatrix
end

function compress(A::RAKMatrix)
    A.data .|> compress |> RAKMatrix
end
