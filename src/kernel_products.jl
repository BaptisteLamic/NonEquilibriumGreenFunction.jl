### Products
function prod(gl::AbstractKernel,gr::AbstractKernel)
    @assert iscompatible(gl,gr)
    return _prod(gl,gr) 
end
_prod(gl::TimeLocalKernel,gr::AbstractKernel) = similar(gr,matrix(gl)*matrix(gr))
_prod(gl::AbstractKernel,gr::TimeLocalKernel) = similar(gl,matrix(gl)*matrix(gr))
_prod(gl::TimeLocalKernel,gr::TimeLocalKernel) = similar(gr,matrix(gl)*matrix(gr))

_prod(gl::NullKernel,gr::K) where {K<:AbstractKernel} = gl
_prod(gl::K,gr::NullKernel) where {K<:AbstractKernel} = gr
_prod(gl::TimeLocalKernel,gr::NullKernel) = gr
_prod(gl::NullKernel,gr::TimeLocalKernel) = gl
_prod(gl::NullKernel,gr::NullKernel) = gl

_trapz_dressing(g,d) = return matrix(g) - eltype(d)(0.5)*d
function _biased_product(gl::L,gr::R) where { L <: Union{RetardedKernel,AdvancedKernel}, R<: Union{RetardedKernel,AdvancedKernel} }
    bs = blocksize(gl)
    dl = extract_blockdiag(matrix(gl),bs,compression = compression(gl))
    dr = extract_blockdiag(matrix(gr),bs,compression = compression(gl))
    weighted_L = _trapz_dressing(gl,dl)
    weighted_R = _trapz_dressing(gr,dr)
    return weighted_L * weighted_R,dl,dr
end
function _prod(gl::Ker, gr::Ker) where Ker<:Union{RetardedKernel,AdvancedKernel}
    biased_result,dl,_ = _biased_product(gl,gr) 
    result = biased_result - extract_blockdiag(biased_result,blocksize(gl), compression = compression(gl)) # Diagonal is zero
    result *= step(gl)
    return similar(gl, result)
end

#For the trapz rule and neglecting the side effect that should be absorbed
#in propoer definition of the problem the same rule apply
function _prod(gl::L,gr::R) where { L <: Union{RetardedKernel,AdvancedKernel}, R<: Union{RetardedKernel,AdvancedKernel} }
    biased_result,dl,dr = _biased_product(gl,gr) 
    correction = eltype(dl)(1/4)*dl*dr
    result = biased_result + correction
    result *= step(gl)
    return similar(gl,result)
end
function _prod(gl::Kernel,gr::R) where R<: Union{RetardedKernel,AdvancedKernel}
    dr = extract_blockdiag(matrix(gr),blocksize(gl),compression = compression(gl))
    T = eltype(dr)
    weighted_L = matrix(gl)
    weighted_R = _trapz_dressing(gr,dr)
    result = weighted_L * weighted_R
    result *= step(gl)
    return similar(gl,result)
end
function _prod(gl::L,gr::Kernel) where L<: Union{RetardedKernel,AdvancedKernel}
    dl = extract_blockdiag(matrix(gl),blocksize(gl),compression = compression(gl))
    T = eltype(dl)
    weighted_L = _trapz_dressing(gl,dl)
    weighted_R = matrix(gr) 
    result = weighted_L * weighted_R
    result *= step(gl)
    return similar(gl,result)
end
function _prod(gl::Kernel,gr::Kernel)
    weighted_L = matrix(gl)
    weighted_R = matrix(gr) 
    result = weighted_L * weighted_R
    result *= step(gl)
    return similar(gl,result)
end

function *(gl::SumKernel,gr::AbstractKernel)
    @assert iscompatible(gl,gr)
    return gl.kernelL * gr + gl.kernelR * gr 
end
function *(gl::AbstractKernel,gr::SumKernel)
    @assert iscompatible(gl,gr)
    return gl*gr.kernelL  + gl * gr.kernelR
end
#Required to avoid type ambiguity 
function *(gl::SumKernel,gr::SumKernel)
    @assert iscompatible(gl,gr)
    return gl*gr.kernelL  + gl*gr.kernelR 
end
function *(gl::AbstractKernel,gr::AbstractKernel)
    return prod(gl,gr) 
end
