#abstract type AbstractKernel{T,A, C <: AbstractCompression} <: AbstractArray{AbstractArray{T,2},2} end
struct RAKMatrix  <: AbstractArray{AbstractArray{AbstractKernel,2},2}
    data::Array{AbstractKernel,2}
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

function getindex(A::RAKMatrix,I::Vararg{Int,2})
    return [A.data[i,j][I...] for i = 1:2, j = 1:2]
end
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