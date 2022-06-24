import Base.getindex
struct KernelMatrix{T} <: AbstractMatrix{T}
    foo
    X
    Y
end
function KernelMatrix(foo, X,Y)
    T = foo(X[1],Y[1]) |> typeof
    return KernelMatrix{T}(foo, X, Y)
end
function Base.size(A::KernelMatrix)
    (length(A.X), length(A.Y))
end
function _getindex(A::KernelMatrix, I,J) where T
    #assume that the index are sorted
    return A.foo( A.X[I], A.Y[J] )
end
#=
function getindex(A::KernelMatrix,I::Matrix{CartesianIndex{2}})
    r = Matrix{eltype(A)}(size(A)...)
    Threads.@thread for i = 1:size(r,1)
        for j = 1:size(r,2)
            r[i,j] = getindex(A,i,j)
        end
    end
end

function getindex(A::KernelMatrix,CartesianIndex{2})
    i,j = Tuple(CartesianIndex)
    return getindex(A,i,j)
end
=#
getindex(A::KernelMatrix,i::Int, j::Int) = _getindex(A,i, j)[1]
getindex(A::KernelMatrix,i::Int, j) = reshape(_getindex(A, i, j),:)
getindex(A::KernelMatrix, i, j::Int) = reshape(_getindex(A, i, j),:)
getindex(A::KernelMatrix,::Colon, ::Colon) = _getindex(A,1:size(A,1),1:size(A,2))
getindex(A::KernelMatrix, i, ::Colon) = _getindex(A,i, 1:size(A,2))
getindex(A::KernelMatrix, ::Colon, j) = _getindex(A,1:size(A,1), j)
getindex(A::KernelMatrix, i::Int, ::Colon) = reshape(_getindex(A,i, 1:size(A,2)),:)
getindex(A::KernelMatrix, ::Colon, j::Int) = reshape(_getindex(A,1:size(A,1), j),:)
function getindex(A::KernelMatrix, I,J)
    if (length(I) == 0 || length(J) == 0) 
        return Matrix{eltype(A)}(undef, length(I), length(J))
    else
        return _getindex(A, I, J)
    end
end 


function Base.view(K::KernelMatrix,I,J)
    return KernelMatrix(K.foo,K.X[I],K.Y[J])
end