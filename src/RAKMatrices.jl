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
        return RAKMatrix(K, R, A, NullKernel(R))
    else
        throw(ArgumentError("$order is not a valid value for the named argument order."))
    end
end

RAKMatrix(R,K; order) = RAKMatrix(R,R',K; order)
scalartype(A::RAKMatrix) = scalartype(A.data[1])
#=

=#
function getindex!(out,A::RAKMatrix, I,J) where T
    bs = blocksize(A)
    values = [A.data[p,q][I,J] for p = 1:2, q = 1:2]
    Threads.@threads for i = 1:length(I)
        for j = 1:length(J)
            for p = 1:2
                for q = 1:2
                    out[i,j][blockrange(p,bs),blockrange(q,bs)] .= values[p,q][i,j]
                end
            end 
        end
    end
    return out
end
function getindex!(out,A::RAKMatrix,i::Int, j::Int)
    return out .= [ A.data[1,1][i,j] A.data[1,2][i,j] ; A.data[2,1][i,j] A.data[2,2][i,j] ]
end
getindex!(out,A::RAKMatrix,i::Int, j) = reshape(getindex!(out,A, [i], j),:)
getindex!(out,A::RAKMatrix, i, j::Int) = reshape(getindex!(out,A, i, [j]),:)
getindex!(out,A::RAKMatrix,::Colon, ::Colon) = getindex!(out,A,1:size(A,1),1:size(A,2))
getindex!(out,A::RAKMatrix, i, ::Colon) = getindex!(out,A,i, 1:size(A,2))
getindex!(out,A::RAKMatrix, ::Colon, j) = getindex!(out,A,1:size(A,1), j)
getindex!(out,A::RAKMatrix, i::Int, ::Colon) = reshape( getindex!(out,A,i::Int, 1:size(A,2) ), : )
getindex!(out,A::RAKMatrix, ::Colon, j::Int) = reshape( getindex!(out,A,1:size(A,1), j),:)

getindex(A::RAKMatrix,i::Int, j) = getindex(A, [i], j)[:]
getindex(A::RAKMatrix, i, j::Int) = getindex(A, i, [j])[:]
getindex(A::RAKMatrix,::Colon, ::Colon) = getindex(A,1:size(A,1),1:size(A,2))
getindex(A::RAKMatrix, i, ::Colon) = getindex(A,i, 1:size(A,2))
getindex(A::RAKMatrix, ::Colon, j) = getindex(A,1:size(A,1), j)
getindex(A::RAKMatrix, i::Int, ::Colon) = reshape(getindex(A,i::Int, 1:size(A,2)),:)
getindex(A::RAKMatrix, ::Colon, j::Int) = reshape(getindex(A,1:size(A,1), j::Int),:)
function getindex(A::RAKMatrix, I,J)
    bs = blocksize(A)
    out = [Matrix{scalartype(A)}(undef,2bs,2bs) for i in I, j in J]
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

(*)(A::RAKMatrix,B::RAKMatrix) = _mul_RAK(A.data, B.data)
(*)(A::RAKMatrix, B::AbstractMatrix) = _mul_RAK(A.data, B)
(*)(A::AbstractMatrix, B::RAKMatrix) = _mul_RAK(A, B.data)

function _mul_RAK(A::AbstractMatrix, B::AbstractMatrix)
    r = Matrix{AbstractKernel}(undef,2,2)
    term = Array{AbstractKernel,3}(undef,2,2,2)
    @Threads.threads for p in 1:2
        for q in 1:2
            for k in 1:2
                term[p,q,k] = A[p,k]*B[k,q]
            end
        end
    end 
    @Threads.threads for p in 1:2
        for q in 1:2
            r[p,q] = term[p,q,1] + term[p,q,2]
        end
    end 
    return r |> RAKMatrix
end

for op in (:*,:\)
    @eval begin
        function $op(λ::Number,A::RAKMatrix)
            r = Matrix{AbstractKernel}(undef,2,2)
            @Threads.threads for p in 1:2
                for q in 1:2
                    r[p,q] = $op(λ, A.data[p,q])
                end
            end 
            return RAKMatrix(r)
        end
    end
end
for op in (:+,:-)
    @eval begin
        function $op(A::RAKMatrix)
            r = Matrix{AbstractKernel}(undef,2,2)
            @Threads.threads for p in 1:2
                for q in 1:2
                    r[p,q] = $op(A.data[p,q])
                end
            end 
            return RAKMatrix(r)
        end
    end
end
for op in (:+,:-)
    @eval begin
        function $op(A::RAKMatrix,B::RAKMatrix)
            r = Matrix{AbstractKernel}(undef,2,2)
            @Threads.threads for p in 1:2
                for q in 1:2
                    r[p,q] = $op(A.data[p,q],B.data[p,q])
                end
            end 
            return RAKMatrix(r)
        end
    end
end

function compress(A::RAKMatrix)
    A.data .|> compress |> RAKMatrix
end

function tr_K(A::RAKMatrix)
    return A.data[1,1] + A.data[2,2]
end