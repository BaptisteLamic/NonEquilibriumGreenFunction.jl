abstract type AbstractKernel end
abstract type RegularKernel <: AbstractKernel end
struct KernelData{A,M,C}
    axis::A
    matrix::M
    blocksize::Int
    compression::C
end
axis(k::KernelData) = k.axis
matrix(k::KernelData) = k.matrix
blocksize(k::KernelData) = k.blocksize
compression(k::KernelData) = k.compression
scalartype(k::KernelData) = eltype(matrix(k))

struct RetardedKernel{A,M,C} <: RegularKernel
    data::KernelData{A,M,C}
end
struct AdvancedKernel{A,M,C} <: RegularKernel
    data::KernelData{A,M,C}
end
struct Kernel{A,M,C} <: RegularKernel
    data::KernelData{A,M,C}
end
struct TimeLocalKernel{A,M,C} <: AbstractKernel
    data::KernelData{A,M,C}
end
function TimeLocalKernel(axis, u::UniformScaling, blocksize::Int, compression)
    N = length(axis) * blocksize
    matrix = sparse(u, N, N) .|> eltype(u) |> compression
    TimeLocalKernel(axis, matrix, blocksize, compression)
end
function (::Type{K})(axis, matrix, blocksize, compression) where {K<:AbstractKernel}
    return K(KernelData(axis, matrix, blocksize, compression))
end
scalartype(k::AbstractKernel) = scalartype(k.data)

function (==)(a::AbstractKernel, b::AbstractKernel)
    return typeof(a) == typeof(b) && iscompatible(a, b) && matrix(a) == matrix(b)
end

function (::Type{K})(axis, f; compression=HssCompression(), stationary=false) where {K<:RegularKernel}
    f00 = f(axis[1], axis[1])
    bs = size(f00, 1)
    _mask(::Type{RetardedKernel}) = (x, y) -> x >= y ? f(x, y) : zero(f00)
    _mask(::Type{AdvancedKernel}) = (x, y) -> x <= y ? f(x, y) : zero(f00)
    _mask(::Type{Kernel}) = (x, y) -> f(x, y)
    f_masked = _mask(K)
    matrix = compression(axis, f_masked, stationary=stationary)
    K(axis, matrix, bs, compression)
end

function TimeLocalKernel(axis, f; compression=HssCompression(), stationary=false)
    f00 = f(axis[1])
    bs = size(f00, 1)
    δ = zeros(eltype(f00), bs, bs, length(axis))
    for i = 1:length(axis)
        δ[:, :, i] .= f(axis[i])
    end
    matrix = blockdiag(δ, compression=compression)
    TimeLocalKernel(axis, matrix, bs, compression)
end

function matrix(g::AbstractKernel)
    matrix(g.data)
end

function similar(A::K, matrix::AbstractArray) where K <: AbstractKernel
    cpr = compression(A)
    return K(axis(A), cpr(matrix), blocksize(A), cpr)
end

## Define getter 
blocksize(g::AbstractKernel) = blocksize(g.data)
axis(g::AbstractKernel) = axis(g.data)
compression(g::AbstractKernel) = compression(g.data)

## Define printing functions
function Base.show(io::IO, k::K) where {K<:AbstractKernel}
    print(io, "$K\n")
end

function Base.show(io::IO, ::MIME"text/plain", k::K) where {K<:AbstractKernel}
    show(io, k)
    println(io, "axis = $(axis(k))")
    println(io, "blocksize = $(blocksize(k))")
    println(io, "compression = $(compression(k))")
end

## Define utility functions

function iscompatible(g::AbstractKernel, k::AbstractKernel)
    axis(g) == axis(k) && blocksize(g) == blocksize(k) && compression(g) == compression(k)
end

isretarded(::K) where {K<:AbstractKernel} = false
isretarded(::K) where {K<:Union{RetardedKernel,TimeLocalKernel}} = true

isadvanced(::K) where {K<:AbstractKernel} = false
isadvanced(::K) where {K<:Union{AdvancedKernel,TimeLocalKernel}} = true
step(k::AbstractKernel) = scalartype(k)(step(axis(k)))

##Algebra
mul(λ::Number, kernel::AbstractKernel) = similar(kernel,λ*matrix(kernel))
ldiv(λ::Number, kernel::AbstractKernel) = similar(kernel, λ\matrix(kernel))
function add(left::K,right::K) where K <: AbstractKernel
    @assert iscompatible(left,right)
    similar(left, matrix(left)+matrix(right))
end

adjoint(g::Kernel) = similar(g, _adapt(matrix(g)'))
adjoint(g::RetardedKernel) = AdvancedKernel(axis(g), _adapt(matrix(g)'), blocksize(g), compression(g))
adjoint(g::AdvancedKernel) = RetardedKernel(axis(g), _adapt(matrix(g)'), blocksize(g), compression(g))
adjoint(g::TimeLocalKernel) = K(axis(g), _adapt(matrix(g)'), blocksize(g), compression(g))

function compress!(kernel::AbstractKernel)
    compression(kernel)(matrix(kernel))
    return kernel  
end
tr(g::AbstractKernel) = step(axis(K)) * tr(matrix(g))
diag(g::AbstractKernel) = diag(matrix(g))
norm(g::AbstractKernel) = norm(matrix(g))

## AbstractArray Interface
size(g::AbstractKernel) = (length(axis(g)), length(axis(g)))
size(g::AbstractKernel, k) = size(g)[k]

function getindex(A::AbstractKernel, ::Colon, I, ::Colon, J)
    sbk = blocksize(A)
    bk_I = vcat(blockrange.(I, sbk)...)
    bk_J = vcat(blockrange.(J, sbk)...)
    values = matrix(A)[bk_I, bk_J]
    return reshape(values, sbk, length(I), sbk, length(J))
end
function _getindex(A::AbstractKernel, I, J)
    #assume that the index are sorted
    sbk = blocksize(A)
    values = reshape(getindex(A, :, I, :, J), length(I) * sbk, length(J) * sbk)
    r = [view(values, sbk*(i-1)+1:sbk*i, sbk*(j-1)+1:sbk*j) for i = 1:length(I), j = 1:length(J)]
    return r
end
function getindex(A::AbstractKernel, i::Int, j::Int)
    bs = blocksize(A)
    return matrix(A)[blockrange(i, bs), blockrange(j, bs)]
end

#getindex(A::AbstractKernel,i::Int, j::Int) = getindex(A,[i], [j])[1]
getindex(A::AbstractKernel, i::Int, j) = reshape(getindex(A, [i], j), :)
getindex(A::AbstractKernel, i, j::Int) = reshape(getindex(A, i, [j]), :)
getindex(A::AbstractKernel, ::Colon, ::Colon) = getindex(A, 1:size(A, 1), 1:size(A, 2))
getindex(A::AbstractKernel, i, ::Colon) = getindex(A, i, 1:size(A, 2))
getindex(A::AbstractKernel, i::Int, ::Colon) = getindex(A, [i], 1:size(A, 2))
getindex(A::AbstractKernel, ::Colon, j) = reshape(getindex(A, 1:size(A, 1), j), :)

function getindex(A::AbstractKernel, I, J)
    Ip = sortperm(I)
    Jp = sortperm(J)
    if (length(I) == 0 || length(J) == 0)
        return Matrix{eltype(A)}(undef, length(I), length(J))
    else
        return @views _getindex(A, I[Ip], J[Jp])[invperm(Ip), invperm(Jp)]
    end
end
