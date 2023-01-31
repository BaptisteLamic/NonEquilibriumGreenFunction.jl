#TODO factorise the type systems into ExpressionTree * Kernel
abstract type AbstractKernel end
abstract type LeafKernel <: AbstractKernel end
abstract type DataFullKernel <: LeafKernel end
abstract type RegularKernel <: DataFullKernel end
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
struct TimeLocalKernel{A,M,C} <: DataFullKernel
    data::KernelData{A,M,C}
end
function TimeLocalKernel(axis, u::UniformScaling, blocksize::Int, compression)
    N = length(axis) * blocksize
    matrix = sparse(u, N, N) .|> eltype(u) |> compression
    TimeLocalKernel(axis, matrix, blocksize, compression)
end
function (::Type{K})(axis, matrix, blocksize, compression) where {K<:DataFullKernel}
    return K(KernelData(axis, matrix, blocksize, compression))
end
scalartype(k::DataFullKernel) = scalartype(k.data)

struct SumKernel{L<:AbstractKernel,R<:AbstractKernel} <: AbstractKernel
    kernelL::L
    kernelR::R
end
scalartype(g::SumKernel) = scalartype(g.kernelL)
blocksize(g::SumKernel) = blocksize(g.kernelL)
compression(g::SumKernel) = compression(g.kernelR)
axis(g::SumKernel) = axis(g.kernelL)

struct NullKernel{T,A,C} <: LeafKernel
    axis::A
    blocksize::Int
    compression::C
end
scalartype(k::NullKernel{T,A,C}) where {T,A,C} = T

(==)(a::NullKernel, b::NullKernel) = iscompatible(a, b)
function (==)(a::SumKernel, b::SumKernel)
    if a.kernelL == b.kernelL && a.kernelR == b.kernelR
        return true
    elseif a.kernelR == b.kernelL && a.kernelL == b.kernelR
        return true
    else
        return false
    end
end
function (==)(a::AbstractKernel, b::AbstractKernel)
    return typeof(a) == typeof(b) && iscompatible(a, b) && matrix(a) == matrix(b)
end
function NullKernel(axis, matrix, blocksize::Int, compression)
    NullKernel{eltype(matrix),typeof(axis),typeof(compression)}(axis, blocksize, compression)
end
function NullKernel(k::AbstractKernel)
    NullKernel{scalartype(k),typeof(axis(k)),typeof(compression(k))}(axis(k), blocksize(k), compression(k))
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

function SumKernel(kernelL::AbstractKernel, kernelR::AbstractKernel)
    @assert iscompatible(kernelL, kernelR)
    cp = compression(kernelL)
    bs = blocksize(kernelL)
    ax = axis(kernelL)
    SumKernel{typeof(kernelL),typeof(kernelR)}(ax, kernelL, kernelR, bs, cp)
end

function matrix(A::NullKernel)
    N = length(axis(A)) * blocksize(A)
    return spzeros(scalartype(A), N, N) |> compression(A)
end
function matrix(g::DataFullKernel)
    matrix(g.data)
end
function matrix(g::SumKernel)
    return matrix(g.kernelL) + matrix(g.kernelR)
end


#implement type conversion

function similar(g::LeafKernel, cpr::AbstractCompression)
    typeof(g)(axis(g), cpr(matrix(g)), blocksize(g), cpr)
end
function similar(g::SumKernel, cpr::AbstractCompression)
    SumKernel(similar(g.kernelL, cpr), similar(g.kernelR, cpr))
end
function similar(A::K, matrix::AbstractArray) where K <: DataFullKernel
    cpr = compression(A)
    return K(axis(A), cpr(matrix), blocksize(A), cpr)
end

## Define getter 
blocksize(g::AbstractKernel) = blocksize(g.data)
blocksize(g::Union{NullKernel,SumKernel}) = g.blocksize

compression(g::AbstractKernel) = compression(g.data)
compression(g::Union{NullKernel,SumKernel}) = g.compression
### AbstractArray Interface
axis(g::AbstractKernel) = axis(g.data)
axis(g::Union{NullKernel,SumKernel}) = g.axis
size(g::AbstractKernel) = (length(axis(g)), length(axis(g)))
size(g::AbstractKernel, k) = size(g)[k]

function getindex(A::AbstractKernel, ::Colon, I, ::Colon, J)
    sbk = blocksize(A)
    bk_I = vcat(blockrange.(I, sbk)...)
    bk_J = vcat(blockrange.(J, sbk)...)
    values = matrix(A)[bk_I, bk_J]
    return reshape(values, sbk, length(I), sbk, length(J))
end
function getindex(A::NullKernel, ::Colon, I, ::Colon, J)
    sbk = blocksize(A)
    return zeros(scalartype(A), sbk, length(I), sbk, length(J))
end


function _getindex(A::AbstractKernel, I, J)
    #assume that the index are sorted
    sbk = blocksize(A)
    ax = axis(A)
    values = reshape(getindex(A, :, I, :, J), length(I) * sbk, length(J) * sbk)
    r = [view(values, sbk*(i-1)+1:sbk*i, sbk*(j-1)+1:sbk*j) for i = 1:length(I), j = 1:length(J)]
    return r
end


function getindex(A::AbstractKernel, i::Int, j::Int)
    bs = blocksize(A)
    return matrix(A)[blockrange(i, bs), blockrange(j, bs)]
end
function getindex(A::SumKernel, i::Int, j::Int)
    r = A.kernelL[i, j]
    r .+= A.kernelR[i, j]
    return r

end
function getindex(A::NullKernel, i::Int, j::Int)
    return zeros(scalartype(A), blocksize(A), blocksize(A))
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
isretarded(::K) where {K<:Union{RetardedKernel,TimeLocalKernel,NullKernel}} = true
function isretarded(k::SumKernel)
    isretarded(k.kernelL) && isretarded(k.kernelR)
end

isadvanced(::K) where {K<:AbstractKernel} = false
isadvanced(::K) where {K<:Union{AdvancedKernel,TimeLocalKernel,NullKernel}} = true
function isadvanced(k::SumKernel)
    isadvanced(k.kernelL) && isadvanced(k.kernelR)
end
#=
We should adapt the names here
=#
nonlocal_part(k::AbstractKernel) = k
nonlocal_part(k::TimeLocalKernel) = NullKernel(k)
nonlocal_part(k::SumKernel) = nonlocal_part(k.kernelR) + nonlocal_part(k.kernelL)

timelocal_part(k::TimeLocalKernel) = k
timelocal_part(k::AbstractKernel) = NullKernel(k)
timelocal_part(k::SumKernel) = timelocal_part(k.kernelL) + timelocal_part(k.kernelR)

function step(k::AbstractKernel)
    scalartype(k)(step(axis(k)))
end

#compression
function (cpr::AbstractCompression)(k::AbstractKernel)
    similar(k, cpr)
end
function (cpr::AbstractCompression)(k::NullKernel{T,A,C}) where {T,A,C}
    NullKernel{T,A,typeof(cpr)}(axis(k), blocksize(k), cpr)
end

##Algebra
### Scalar operations defined by metaprogramming
####General form
for op in (:*, :\)
    @eval begin
        function $op(λ::T, g::K) where {T<:Number,K<:AbstractKernel}
            return K(axis(g), compression(g)($op(λ, matrix(g))), blocksize(g), compression(g))
        end
    end
end
for op in (:*, :\)
    @eval begin
        function $op(λI::UniformScaling, g::K) where {K<:AbstractKernel}
            return λI.λ * g
        end
    end
end
for op in (:*, :\)
    @eval begin
        function $op(g::K, λI::UniformScaling) where {K<:AbstractKernel}
            return λI.λ * g
        end
    end
end
#### Specialized version for SumKernel
for op in (:*, :\)
    @eval begin
        function $op(λ::Number, g::SumKernel)
            kl = $op(λ, g.kernelL)
            kr = $op(λ, g.kernelR)
            return SumKernel(kl, kr)
        end
    end
end
#### Specialized version for NullKernel
for op in (:*, :\)
    @eval begin
        $op(λ::Number, g::NullKernel) = g
    end
end
### Sign changes 
####General form
for op in (:+, :-)
    @eval begin
        function $op(g::K) where {K<:AbstractKernel}
            K(axis(g), $op(matrix(g)), blocksize(g), compression(g))
        end
    end
end
#### Specialized version for SumKernel
for op in (:+, :-)
    @eval begin
        function $op(g::SumKernel)
            kl = $op(g.kernelL)
            kr = $op(g.kernelR)
            return SumKernel(kl, kr)
        end
    end
end
#### Specialized version for NullKernel
for op in (:+, :-)
    @eval begin
        $op(g::NullKernel) = g
    end
end

### +Group operation on Kernel-Kernel operations
####General variant
for op in (:+, :-)
    @eval begin
        function $op(gl::AbstractKernel, gr::AbstractKernel)
            iscompatible(gl, gr)
            SumKernel(
                gl,
                $op(gr)
            )
        end
    end
end
#### operations on sames objects
for op in (:+, :-)
    @eval begin
        function $op(gl::K, gr::K) where {K<:AbstractKernel}
            iscompatible(gl, gr)
            K(
                axis(gl),
                $op(matrix(gl), matrix(gr)),
                blocksize(gl),
                compression(gl)
            )
        end
    end
end
####Specialized for SumKernel
for op in (:+, :-)
    @eval begin
        function $op(gl::SumKernel, gr::SumKernel)
            iscompatible(gl, gr)
            SumKernel(
                gl,
                $op(gr)
            )
        end
    end
end
####Specialized for NullKernel
for op in (:+, :-)
    @eval begin
        function $op(gl::NullKernel, gr::AbstractKernel)
            iscompatible(gl, gr)
            $op(gr)
        end
    end
end
for op in (:+, :-)
    @eval begin
        function $op(gl::AbstractKernel, gr::NullKernel)
            iscompatible(gl, gr)
            gl
        end
    end
end
for op in (:+, :-)
    @eval begin
        function $op(gl::NullKernel, gr::NullKernel)
            iscompatible(gl, gr)
            gl
        end
    end
end
####Specialized for UniformScaling
for op in (:+, :-)
    @eval begin
        function $op(gl::UniformScaling, gr::AbstractKernel)
            _I = scalartype(gr)(gl.λ) * I
            TimeLocalKernel(axis(gr), _I, blocksize(gr), compression(gr)) + $op(gr)
        end
    end
end
for op in (:+, :-)
    @eval begin
        function $op(gl::AbstractKernel, gr::UniformScaling)
            _I = $op(T(gr.λ)) * I
            gl + TimeLocalKernel(axis(gl), _I, blocksize(gl), compression(gl))
        end
    end
end


function adjoint(g::Kernel)
    return similar(g, _adapt(matrix(g)'))
end
function adjoint(g::RetardedKernel)
    return AdvancedKernel(axis(g), _adapt(matrix(g)'), blocksize(g), compression(g))
end
function adjoint(g::AdvancedKernel)
    return RetardedKernel(axis(g), _adapt(matrix(g)'), blocksize(g), compression(g))
end
function adjoint(g::K) where {K<:Union{TimeLocalKernel,Kernel}}
    K(axis(g), _adapt(matrix(g)'), blocksize(g), compression(g))
end
adjoint(g::NullKernel) = g
function adjoint(g::SumKernel)
    return g.kernelL' + g.kernelR'
end
function compress!(A::AbstractKernel)
    A
end
function compress!(g::K) where {K<:Union{TimeLocalKernel,Kernel,RetardedKernel,AdvancedKernel}}
    compression(g)(matrix(g))
    return g
end

function tr(A::NullKernel)
    A[1, 1]
end
function tr(g::K) where {K<:Union{TimeLocalKernel,Kernel,RetardedKernel,AdvancedKernel}}
    step(axis(K)) * tr(matrix(g))
end
function diag(A::NullKernel)
    N = length(axis) * blocksize(A)
    return diag(spzeros(A[1, 1], N, N) |> compression(A))
end
function diag(g::K) where {K<:Union{Kernel,RetardedKernel,AdvancedKernel}}
    diag(matrix(g))
end
function diag(g::SumKernel)
    return diag(g.kernelL) + diag(g.kernelR)
end
norm(g::AbstractKernel) = norm(matrix(g))