
### Products
#### TimeLocalKernels
function _prod(gl::TimeLocalKernel,gr::AbstractKernel) 
    @assert iscompatible(gl,gr)
    typeof(gr)(axis(gr),
        matrix(gl)*matrix(gr),
        blocksize(gr),
        compression(gr)
    )
end
function _prod(gl::AbstractKernel,gr::TimeLocalKernel) 
    @assert iscompatible(gl,gr)
    typeof(gl)(axis(gl),
        matrix(gl)*matrix(gr),
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
        matrix(gl)*matrix(gr),
        blocksize(gl),
        compression(gl)

    )
end
#### RetardedKernel
function _prod(gl::Ker, gr::Ker) where Ker<:Union{RetardedKernel,AdvancedKernel}
    #TODO check this code
    @assert iscompatible(gl,gr)
    bs = blocksize(gl)
    dl = extract_blockdiag(matrix(gl),bs,compression = compression(gl))
    dr = extract_blockdiag(matrix(gr),bs,compression = compression(gl))
    T = eltype(dl)
    weighted_L = matrix(gl) - T(0.5)*dl
    weighted_R = matrix(gr) - T(0.5)*dr
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
    dl = extract_blockdiag(matrix(gl),bs,compression = compression(gl))
    dr = extract_blockdiag(matrix(gr),bs,compression = compression(gl))
    T = eltype(dl)
    weighted_L = matrix(gl) - T(0.5)*dl
    weighted_R = matrix(gr) - T(0.5)*dr
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
    dr = extract_blockdiag(matrix(gr),bs,compression = compression(gl))
    T = eltype(dr)
    weighted_L = matrix(gl)
    weighted_R = matrix(gr) - T(0.5)*dr
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
    dl = extract_blockdiag(matrix(gl),bs,compression = compression(gl))
    T = eltype(dl)
    weighted_L = matrix(gl) - T(0.5)*dl
    weighted_R = matrix(gr) 
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
function _prod(gl::Kernel,gr::Kernel)
    @assert iscompatible(gl,gr)
    bs = blocksize(gl)
    T = eltype(matrix(gl))
    weighted_L = matrix(gl)
    weighted_R = matrix(gr) 
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
    @assert iscompatible(gl,gr)
    return gl
end
function _prod(gl::K,gr::NullKernel) where {K<:AbstractKernel}
    @assert iscompatible(gl,gr)
    return gr
end
function _prod(gl::TimeLocalKernel,gr::NullKernel)
    @assert iscompatible(gl,gr)
    return gr
end
function _prod(gl::NullKernel,gr::TimeLocalKernel)
    @assert iscompatible(gl,gr)
    return gl
end
function _prod(gl::NullKernel,gr::NullKernel)
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
