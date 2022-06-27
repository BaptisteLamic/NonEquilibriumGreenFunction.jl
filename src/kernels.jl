abstract type AbstractKernel{T,A, C <: AbstractCompression} <: AbstractArray{AbstractArray{T,2},2} end

struct RetardedKernel{T,A,M,C} <: AbstractKernel{T,A,C}
    axis::A
    matrix::M
    blocksize::Int
    compression::C
end
struct AdvancedKernel{T,A,M,C} <: AbstractKernel{T,A,C}
    axis::A
    matrix::M
    blocksize::Int
    compression::C
end
struct Kernel{T,A,M,C} <: AbstractKernel{T,A,C}
    axis::A
    matrix::M
    blocksize::Int
    compression::C
end
struct TimeLocalKernel{T,A,M,C} <: AbstractKernel{T,A,C}  
    axis::A
    matrix::M
    blocksize::Int
    compression::C
end
struct SumKernel{T,A,C,L<:AbstractKernel{T,A,C},R<:AbstractKernel{T,A,C}} <: AbstractKernel{T,A,C}
    axis::A
    kernelL::L
    kernelR::R
    blocksize::Int
    compression::C
end
struct NullKernel{T,A,C} <: AbstractKernel{T,A,C}
    axis::A
    blocksize::Int
    compression::C
end


function NullKernel(axis, matrix, blocksize::Int, compression)
    NullKernel{eltype(matrix),typeof(axis),typeof(compression)}(axis, blocksize, compression)
end
function NullKernel(k::AbstractKernel{T,A,C}) where {T,A,C}
    NullKernel{T,A,C}(axis(k), blocksize(k), compression(k))
end

scalartype(K::AbstractKernel{T,A, C}) where {T, A, C} = T
### Definitions of the constructors
function RetardedKernel(axis, matrix, blocksize::Int, compression)
    RetardedKernel{eltype(matrix),typeof(axis),typeof(matrix),typeof(compression)}(axis, matrix, blocksize, compression)
end
function RetardedKernel(axis,f; compression = HssCompression(), stationary = false)
    f00 = f(axis[1],axis[1])
    bs = size(f00,1)
    f_masked = (x,y) -> x>=y ? f(x,y) : zero(f00)
    matrix = compression(axis,f_masked,stationary = stationary)
    RetardedKernel(axis,matrix,bs,compression)
end

function AdvancedKernel(axis, matrix, blocksize::Int, compression)
    AdvancedKernel{eltype(matrix),typeof(axis),typeof(matrix),typeof(compression)}(axis, matrix, blocksize, compression)
end
function AdvancedKernel(axis,f; compression = HssCompression(), stationary = false)
    f00 = f(axis[1],axis[1])
    bs = size(f00,1)
    f_masked = (x,y) -> x<=y ? f(x,y) : zero(f00)
    matrix = compression(axis,f_masked, stationary = stationary)
    AdvancedKernel(axis,matrix,bs,compression)
end

function Kernel(axis, matrix, blocksize::Int, compression)
    Kernel{eltype(matrix),typeof(axis),typeof(matrix),typeof(compression)}(axis, matrix, blocksize, compression)
end
function Kernel(axis,f; compression = HssCompression(), stationary = false)
    f00 = f(axis[1],axis[1])
    bs = size(f00,1)
    matrix = compression(axis,f, stationary = stationary)
    Kernel(axis,matrix,bs,compression)
end

function TimeLocalKernel(axis, matrix, blocksize::Int, compression)
    TimeLocalKernel{eltype(matrix),typeof(axis),typeof(matrix),typeof(compression)}(axis, matrix, blocksize, compression)
end
function TimeLocalKernel(axis, u::UniformScaling, blocksize::Int, compression )
    N = length(axis)*blocksize
    matrix = sparse(u,N,N) .|> eltype(u) |> compression
    TimeLocalKernel(axis,matrix,blocksize,compression)
end
function TimeLocalKernel(axis,f; compression = HssCompression(), stationary = false)
    f00 = f(axis[1])
    bs = size(f00,1)
    δ = zeros(eltype(f00),bs,bs,length(axis))
    Threads.@threads for i = 1:length(axis)
        δ[:,:,i] = f(axis[i]) 
    end
    matrix = blockdiag(δ,compression = compression)
    TimeLocalKernel(axis,matrix,bs,compression)
end

function SumKernel(kernelL::AbstractKernel{T, A, C}, kernelR::AbstractKernel{T, A ,C}) where {T, A, C}
    @assert iscompatible(kernelL, kernelR)
    cp = compression(kernelL)
    bs = blocksize(kernelL)
    ax = axis(kernelL)
    SumKernel{T,A,C,typeof(kernelL),typeof(kernelR)}(ax, kernelL, kernelR, bs, cp)
end


#implement type conversion
for K in (:RetardedKernel,:AdvancedKernel,:NullKernel,:TimeLocalKernel,:Kernel)
    @eval begin
        function similar(g::$K,cpr::AbstractCompression)
            $K(axis(g), cpr(matrix(g)), blocksize(g), cpr)
        end
    end
end
function similar(g::SumKernel,cpr::AbstractCompression)
    SumKernel(axis(g), similar(g.kernelL,cpr),similar(g.kernelR,cpr), blocksize(g), cpr)
end
function similar(A::AbstractKernel,matrix::AbstractArray)
    typeof(A)(axis(A), matrix, blocksize(A), compression(A))
end

#



## Define getter 
blocksize(g::AbstractKernel) = g.blocksize
compression(g::AbstractKernel) = g.compression
### AbstractArray Interface
axis(g::AbstractKernel) = g.axis
size(g::AbstractKernel) = ( length(axis(g)), length(axis(g)) )

function getindex(A::AbstractKernel,::Colon,I,::Colon,J)
    sbk = blocksize(A)
    ax = axis(A)
    bk_I = vcat(blockrange.( I, sbk )...)
    bk_J = vcat(blockrange.( J, sbk )...)
    values = matrix(A)[bk_I,bk_J]
    return reshape(values,sbk,length(I), sbk, length(J))
end
function getindex(A::NullKernel,::Colon,I,::Colon,J)
    sbk = blocksize(A)
    return zeros(scalartype(A),sbk,length(I), sbk, length(J))
end


function _getindex(A::AbstractKernel, I, J)
    #assume that the index are sorted
    sbk = blocksize(A)
    ax = axis(A)
    values = reshape(getindex(A,:,I,:,J), length(I)*sbk, length(J)*sbk)
    r = [ view( values, sbk*(i-1)+1:sbk*i, sbk*(j-1)+1:sbk*j )  for i = 1:length(I), j = 1:length(J) ]
    return r
end


function getindex(A::AbstractKernel,i::Int,j::Int)
    bs = blocksize(A)
    return matrix(A)[blockrange(i,bs),blockrange(j,bs)]
end
function getindex(A::SumKernel,i::Int,j::Int)
    r = A.kernelL[i,j]
    r .+= A.kernelR[i,j]
    return r
    
end
function getindex(A::NullKernel,i::Int,j::Int)
    return zeros(scalartype(A), blocksize(A), blocksize(A))
end

#getindex(A::AbstractKernel,i::Int, j::Int) = getindex(A,[i], [j])[1]
getindex(A::AbstractKernel,i::Int, j) = reshape(getindex(A, [i], j),:)
getindex(A::AbstractKernel, i, j::Int) = reshape(getindex(A, i, [j]),:)
getindex(A::AbstractKernel,::Colon, ::Colon) = getindex(A,1:size(A,1),1:size(A,2))
getindex(A::AbstractKernel, i, ::Colon) = getindex(A,i, 1:size(A,2))
getindex(A::AbstractKernel, i::Int, ::Colon) = getindex(A,[i], 1:size(A,2))
getindex(A::AbstractKernel, ::Colon, j) = reshape(getindex(A,1:size(A,1), j),:)

function getindex(A::AbstractKernel, I,J)
    Ip = sortperm(I); Jp = sortperm(J)
    if (length(I) == 0 || length(J) == 0) 
        return Matrix{eltype(A)}(undef, length(I), length(J))
    else
        return @views _getindex(A,I[Ip], J[Jp])[invperm(Ip), invperm(Jp)]
    end
end
#=
function getindex(A::SumKernel,I::Vararg{Int,2})
    A.kernelL[I...] + A.kernelR[I...]
end
function getindex(K::NullKernel{T,A,C},I::Vararg{Int,2}) where {T,A,C}
    return zeros(T,blocksize(K), blocksize(K))
end
=#
## Define printing functions
function Base.show(io::IO, k::K) where {T, A, C, K<:AbstractKernel{T, A, C}}
    print(io, "$K\n")
end

function Base.show(io::IO, ::MIME"text/plain", k::K) where {T, A, M, C, K<:AbstractKernel{T, A, C}}
    show(io, k)
    println(io,"axis = $(axis(k))")
    println(io,"blocksize = $(blocksize(k))")
    println(io,"compression = $(compression(k))")
end

## Define utility functions

function iscompatible(g::AbstractKernel,k::AbstractKernel)
     axis(g) == axis(k) && blocksize(g) == blocksize(k) && compression(g) == compression(k)
end

isretarded(::K) where K<: AbstractKernel = false
isretarded(::K) where {K <: Union{RetardedKernel,TimeLocalKernel,NullKernel}} = true
function isretarded(k::SumKernel)
    isretarded(k.kernelL) && isretarded(k.kernelR)
end

isadvanced(::K) where K<: AbstractKernel = false
isadvanced(::K) where {K <: Union{AdvancedKernel,TimeLocalKernel,NullKernel}} = true
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

function step(k::AbstractKernel{T,A,C}) where {T,A,C}
    T( step( axis( k ) ) )
end

#compression
function(cpr::AbstractCompression)(k::AbstractKernel)
    similar(k, cpr)
end
function(cpr::AbstractCompression)(k::NullKernel{T,A,C}) where {T,A,C}
    NullKernel{T,A,typeof(cpr)}(axis(k), blocksize(k), cpr)
end

##Algebra
### Scalar operations defined by metaprogramming
####General form
for op in (:*,:\)
    @eval begin
        function $op(λ::T,g::K) where {T<: Number,K<:AbstractKernel}
            return K(axis(g), compression(g)($op(λ,g.matrix)), blocksize(g),compression(g))
        end
    end
end
#### Specialized version for SumKernel
for op in (:*,:\)
    @eval begin
        function $op(λ::Number,g::SumKernel)
            kl =  $op(λ,g.kernelL)
            kr =  $op(λ,g.kernelR)
            return SumKernel(axis(g),kl,kr, blocksize(g),compression(g))
        end
    end
end
#### Specialized version for NullKernel
for op in (:*,:\)
    @eval begin
        $op(λ::Number,g::NullKernel) = g
    end
end
### Sign changes 
####General form
for op in (:+,:-)
    @eval begin
        function $op(g::K) where K <: AbstractKernel
            K(axis(g), $op(g.matrix), blocksize(g), compression(g))
        end
    end
end
#### Specialized version for SumKernel
for op in (:+,:-)
    @eval begin
        function $op(g::SumKernel)           
            kl =  $op(g.kernelL)
            kr =  $op(g.kernelR)
            return SumKernel(kl, kr)
        end
    end
end
#### Specialized version for NullKernel
for op in (:+,:-)
    @eval begin
        $op(g::NullKernel) = g
    end
end

### +Group operation on Kernel-Kernel operations
####General variant
for op in (:+,:-)
    @eval begin
        function $op(gl::AbstractKernel,gr::AbstractKernel)
            iscompatible(gl, gr)
            SumKernel(  
                gl,
                $op(gr)
                )
        end
    end
end
#### operations on sames objects
for op in (:+,:-)
    @eval begin
        function $op(gl::K,gr::K) where K <: AbstractKernel
            iscompatible(gl, gr)
            K(
                axis(gl),
                $op(gl.matrix, gr.matrix),
                blocksize(gl),
                compression(gl)
                )
        end
    end
end
####Specialized for SumKernel
for op in (:+,:-)
    @eval begin
        function $op(gl::SumKernel,gr::SumKernel)
            iscompatible(gl, gr)
            SumKernel(  
                gl,
                $op(gr)
                )
        end
    end
end
####Specialized for NullKernel
for op in (:+,:-)
    @eval begin
        function $op(gl::NullKernel,gr::AbstractKernel)
            iscompatible(gl, gr)
            $op(gr)
        end
    end
end
for op in (:+,:-)
    @eval begin
        function $op(gl::AbstractKernel,gr::NullKernel)
            iscompatible(gl, gr)
            gl
        end
    end
end
for op in (:+,:-)
    @eval begin
        function $op(gl::NullKernel,gr::NullKernel)
            iscompatible(gl, gr)
            gl
        end
    end
end

####Specialized for UniformScaling
for op in (:+,:-)
    @eval begin
        function $op(gl::UniformScaling,gr::AbstractKernel{T,A,C}) where {T,A,C}
            _I = T(gl.λ)*I
            TimeLocalKernel(axis(gr),_I,blocksize(gr),compression(gr)) + $op(gr)
        end
    end
end
for op in (:+,:-)
    @eval begin
        function $op(gl::AbstractKernel{T,A,C},gr::UniformScaling) where {T,A,C}
            _I = $op(T(gr.λ))*I
            gl + TimeLocalKernel(axis(gl),_I,blocksize(gl),compression(gl))
        end
    end
end



### Products
#### TimeLocalKernels
function _prod(gl::TimeLocalKernel,gr::AbstractKernel) 
    @assert iscompatible(gl,gr)
    typeof(gr)(axis(gr),
        gl.matrix*gr.matrix,
        blocksize(gr),
        compression(gr)
    )
end
function _prod(gl::AbstractKernel,gr::TimeLocalKernel) 
    @assert iscompatible(gl,gr)
    typeof(gl)(axis(gl),
    
        gl.matrix*gr.matrix,
        blocksize(gl),
        compression(gl)
    )
end
#=
We must define the following function to fix ambiguity
=#
function _prod(gl::TimeLocalKernel,gr::TimeLocalKernel) 
    @assert iscompatible(gl,gr)
    typeof(gr)(axis(gl),
        gl.matrix*gr.matrix,
        blocksize(gl),
        compression(gl)

    )
end
#### RetardedKernel
function _prod(gl::Ker, gr::Ker) where Ker<:Union{RetardedKernel,AdvancedKernel}
    #TODO check this code
    @assert iscompatible(gl,gr)
    bs = blocksize(gl)
    dl = extract_blockdiag(gl.matrix,bs,compression = compression(gl))
    dr = extract_blockdiag(gr.matrix,bs,compression = compression(gl))
    T = eltype(dl)
    weighted_L = gl.matrix - T(0.5)*dl
    weighted_R = gr.matrix - T(0.5)*dr
    biased_product = weighted_L * weighted_R
    result = biased_product - extract_blockdiag(biased_product,bs, compression = compression(gl)) # Diagonal is zero
    result *= T(step(axis(gl)))
    return Ker(axis(gl),
                result,
                blocksize(gl),
                compression(gl)
            )
end
#### Retarded * Advanced and Advanced * Retarded
#= 
For the trapz rule and neglecting the side effect that should be absorbed
in propoer definition of the problem the same rule apply
=#
function _prod(gl::L,gr::R) where { L <: Union{RetardedKernel,AdvancedKernel}, R<: Union{RetardedKernel,AdvancedKernel} }
    @assert iscompatible(gl,gr)
    bs = blocksize(gl)
    dl = extract_blockdiag(gl.matrix,bs,compression = compression(gl))
    dr = extract_blockdiag(gr.matrix,bs,compression = compression(gl))
    T = eltype(dl)
    weighted_L = gl.matrix - T(0.5)*dl
    weighted_R = gr.matrix - T(0.5)*dr
    biased_result = weighted_L * weighted_R
    correction = T(1/4)*dl*dr
    result = biased_result + correction
    result *= T(step(axis(gl)))
    return Kernel(axis(gl),
                result,
                blocksize(gl),
                compression(gl)
            )
end
#### Kernel * Advanced and Kernel * Retarded
#= 
For the trapz rule and neglecting the side effect that should be absorbed
in propoer definition of the problem the same rule apply
=#
function _prod(gl::Kernel,gr::R) where R<: Union{RetardedKernel,AdvancedKernel}
    @assert iscompatible(gl,gr)
    bs = blocksize(gl)
    dr = extract_blockdiag(gr.matrix,bs,compression = compression(gl))
    T = eltype(dr)
    weighted_L = gl.matrix
    weighted_R = gr.matrix - T(0.5)*dr
    result = weighted_L * weighted_R
    result *= T(step(axis(gl)))
    return Kernel(axis(gl),
                result,
                blocksize(gl),
                compression(gl)
            )
end

#### Retarded * Kernel and Advanced * Kernel
#= 
For the trapz rule and neglecting the side effect that should be absorbed
in propoer definition of the problem the same rule apply
=#
function _prod(gl::L,gr::Kernel) where L<: Union{RetardedKernel,AdvancedKernel}
    @assert iscompatible(gl,gr)
    bs = blocksize(gl)
    dl = extract_blockdiag(gl.matrix,bs,compression = compression(gl))
    T = eltype(dl)
    weighted_L = gl.matrix - T(0.5)*dl
    weighted_R = gr.matrix 
    result = weighted_L * weighted_R
    result *= T(step(axis(gl)))
    return Kernel(axis(gl),
                result,
                blocksize(gl),
                compression(gl)
            )
end

#### Retarded * Kernel and Advanced * Kernel
#= 
For the trapz rule and neglecting the side effect that should be absorbed
in propoer definition of the problem the same rule apply
=#
function _prod(gl::Kernel,gr::Kernel) where L<: Union{RetardedKernel,AdvancedKernel}
    @assert iscompatible(gl,gr)
    bs = blocksize(gl)
    T = eltype(gl.matrix)
    weighted_L = gl.matrix
    weighted_R = gr.matrix 
    result = weighted_L * weighted_R
    result *= T(step(axis(gl)))
    return Kernel(axis(gl),
                result,
                blocksize(gl),
                compression(gl)
            )
end
#### NullKernel rules
function _prod(gl::NullKernel,gr::K) where {K<:AbstractKernel}
    #TODO check this code}
    @assert iscompatible(gl,gr)
    return gl
end
function _prod(gl::K,gr::NullKernel) where {K<:AbstractKernel}
    #TODO check this code}
    @assert iscompatible(gl,gr)
    return gr
end
function _prod(gl::TimeLocalKernel,gr::NullKernel) where {K<:AbstractKernel}
    #TODO check this code}
    @assert iscompatible(gl,gr)
    return gr
end
function _prod(gl::NullKernel,gr::TimeLocalKernel) where {K<:AbstractKernel}
    #TODO check this code}
    @assert iscompatible(gl,gr)
    return gl
end
function _prod(gl::NullKernel,gr::NullKernel)
    #TODO check this code}
    @assert iscompatible(gl,gr)
    return gl
end
####Specialized for UniformScaling
function _prod(gl::UniformScaling,gr::AbstractKernel{T,A,C}) where {T,A,C}
    _I = T(gl.λ)*I
    TimeLocalKernel(axis(gr),_I,blocksize(gr),compression(gr))*gr
end
function _prod(gl::AbstractKernel{T,A,C},gr::UniformScaling) where {T,A,C}
    _I = T(gr.λ)*I
    TimeLocalKernel(axis(gl),_I,blocksize(gl),compression(gl))*gr
end

#### Intercept the generic rule to multiply AbstractArray
function _prod(gl::AbstractKernel, gr::AbstractKernel)
    error("$(typeof(gl)) * $(typeof(gr)): not implemented")
end

#### SumKernel * AbstractKernel
function *(gl::SumKernel,gr::AbstractKernel)
    @assert iscompatible(gl,gr)
    return gl.kernelL * gr + gl.kernelR * gr 
end
#### AbstractKernel * SumKernel
function *(gl::AbstractKernel,gr::SumKernel)
    @assert iscompatible(gl,gr)
    return gl*gr.kernelL  + gl * gr.kernelR
end
#### SumKernel * SumKernel
#=
Required to avoid type ambiguity 
=#
function *(gl::SumKernel,gr::SumKernel)
    @assert iscompatible(gl,gr)
    return gl*gr.kernelL  + gl*gr.kernelR 
end
#### AbstractKernel * AbstractKernel
function *(gl::AbstractKernel,gr::AbstractKernel)
    @assert iscompatible(gl,gr)
    return _prod(gl,gr) 
end

function adjoint(g::Kernel) 
    return similar(g, _adapt( g.matrix' ) )
end
function adjoint(g::RetardedKernel) 
    return AdvancedKernel(axis(g),_adapt( g.matrix' ),blocksize(g),compression(g))
end
function adjoint(g::AdvancedKernel) 
    return RetardedKernel(axis(g),_adapt( g.matrix' ),blocksize(g),compression(g))
end
function adjoint(g::K) where K <: Union{TimeLocalKernel,Kernel}
    K(axis(g),_adapt( g.matrix' ),blocksize(g),compression(g))
end
adjoint(g::NullKernel) = g
function adjoint(g::SumKernel) 
    return g.kernelL'+g.kernelR' 
end
function compress(A::AbstractKernel)
    A
end
function compress(g::K) where K <: Union{TimeLocalKernel,Kernel,RetardedKernel,AdvancedKernel}
    g.compression(g.matrix)
    return g
end

function tr(A::NullKernel)
    A[1,1]
end
function tr(g::K) where K <: Union{TimeLocalKernel,Kernel,RetardedKernel,AdvancedKernel}
    step(axis(K))*tr(g.matrix)
end

function diag(A::NullKernel)
    N = length(axis)*blocksize
    return diag(spzeros(A[1,1],N,N) |> compression(A))
end
function diag(g::K) where K <: Union{Kernel,RetardedKernel,AdvancedKernel}
    diag(g.matrix)
end
function diag(g::SumKernel)
    return diag(g.kernelL) + diag(g.kernelR)
end

function matrix(A::NullKernel)
    N = length(axis(A))*blocksize(A)
    return spzeros(scalartype(A),N,N) |> compression(A)
end
function matrix(g::K) where K <: Union{Kernel,RetardedKernel,AdvancedKernel,TimeLocalKernel}
    g.matrix
end
function matrix(g::SumKernel)
    return matrix(g.kernelL) + matrix(g.kernelR)
end