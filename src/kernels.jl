abstract type AbstractKernel{T,A,M <: AbstractArray, C <: AbstractCompression} <: AbstractArray{AbstractArray{T,2},2} end

struct RetardedKernel{T,A,M,C} <: AbstractKernel{T,A,M,C}
    axis::A
    matrix::M
    blocksize::Int
    compression::C
end
struct AdvancedKernel{T,A,M,C} <: AbstractKernel{T,A,M,C}
    axis::A
    matrix::M
    blocksize::Int
    compression::C
end
struct TimeLocalKernel{T,A,M,C} <: AbstractKernel{T,A,M,C}  
    axis::A
    matrix::M
    blocksize::Int
    compression::C
end
struct SumKernel{T,A,M,C,L<:AbstractKernel{T,A,M,C},R<:AbstractKernel{T,A,M,C}} <: AbstractKernel{T,A,M,C}
    axis::A
    kernelL::L
    kernelR::R
    blocksize::Int
    compression::C
end

### Definitions of the constructors
function RetardedKernel(axis, matrix, blocksize::Int, compression)
    RetardedKernel{eltype(matrix),typeof(axis),typeof(matrix),typeof(compression)}(axis, matrix, blocksize, compression)
end
function RetardedKernel(axis,f; compression = HssCompression())
    f00 = f(axis[1],axis[1])
    bs = size(f00,1)
    f_masked = (x,y) -> x>=y ? f(x,y) : zero(f00)
    matrix = compression(axis,f_masked)
    RetardedKernel(axis,matrix,bs,compression)
end

function AdvancedKernel(axis, matrix, blocksize::Int, compression)
    AdvancedKernel{eltype(matrix),typeof(axis),typeof(matrix),typeof(compression)}(axis, matrix, blocksize, compression)
end
function AdvancedKernel(axis,f; compression = HssCompression())
    f00 = f(axis[1],axis[1])
    bs = size(f00,1)
    f_masked = (x,y) -> x<=y ? f(x,y) : zero(f00)
    matrix = compression(axis,f_masked)
    AdvancedKernel(axis,matrix,bs,compression)
end

function TimeLocalKernel(axis, matrix, blocksize::Int, compression)
    TimeLocalKernel{eltype(matrix),typeof(axis),typeof(matrix),typeof(compression)}(axis, matrix, blocksize, compression)
end
function TimeLocalKernel(axis,f; compression = HssCompression())
    f00 = f(axis[1],axis[1])
    bs = size(f00,1)
    δ = zeros(eltype(f00),bs,bs,length(axis))
    Threads.@threads for i = 1:length(axis)
        δ[:,:,i] .= f(axis[i]) 
    end
    matrix = blockdiag(δ,compression = compression)
    TimeLocalKernel(axis,matrix,bs,compression)
end

function SumKernel(kernelL::AbstractKernel{T, A, M, C}, kernelR::AbstractKernel{T, A, M ,C}) where {T, A, M, C}
    @assert iscompatible(kernelL, kernelR)
    cp = compression(kernelL)
    bs = blocksize(kernelL)
    ax = axis(kernelL)
    SumKernel{T,A,M,C,typeof(kernelL),typeof(kernelR)}(ax, kernelL, kernelR, bs, cp)
end

## Define getter 
blocksize(g::AbstractKernel) = g.blocksize
compression(g::AbstractKernel) = g.compression
### AbstractArray Interface
axis(g::AbstractKernel) = g.axis
size(g::AbstractKernel) = ( length(axis(g)), length(axis(g)) )

function getindex(A::AbstractKernel,I::Vararg{Int,2}) 
    A.matrix[blockrange(I[1], blocksize(A) ), blockrange(I[2], blocksize(A) )]
end
function getindex(A::SumKernel,I::Vararg{Int,2})
    A.kernelL[I...] + A.kernelR[I...]
end

function iscompatible(g::AbstractKernel,k::AbstractKernel)
     axis(g) == axis(k) && blocksize(g) == blocksize(k) && compression(g) == compression(k)
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
            kl =  compression(g)($op(λ,g.kernelL))
            kr =  compression(g)($op(λ,g.kernelR))
            return SumKernel(axis(g),kl,kr, blocksize(g),compression(g))
        end
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
            kl =  compression(g)($op(g.kernelL))
            kr =  compression(g)($op(g.kernelR))
            return SumKernel(kl, kr)
        end
    end
end

### +Group operation on Kernel-Kernel operations
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

### Products
#### TimeLocalKernels
function *(gl::TimeLocalKernel,gr::Union{RetardedKernel,AdvancedKernel,TimeLocalKernel}) 
    @assert iscompatible(gl,gr)
    typeof(gr)(axis(gl),
        gl.matrix*gr.matrix,
        blocksize(gl),
        compression(gl)
    )
end
function *(gl::Union{RetardedKernel,AdvancedKernel,TimeLocalKernel},gr::TimeLocalKernel) 
    @assert iscompatible(gl,gr)
    typeof(gr)(axis(gl),
        gl.matrix*gr.matrix,
        blocksize(gl),
        compression(gl)
    )
end
#### RetardedKernel
function *(gl::Ker, gr::Ker) where Ker<:Union{RetardedKernel,AdvancedKernel}
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
function *(gl::AbstractKernel, gr::AbstractKernel)
    error("Not implemented")
end