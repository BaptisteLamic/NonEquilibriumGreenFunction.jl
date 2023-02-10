### Products
function mul(gl::AbstractKernel,gr::AbstractKernel)
    @assert iscompatible(gl,gr)
    return _mul(gl,gr) 
end
_mul(gl::TimeLocalKernel,gr::AbstractKernel) = similar(gr,matrix(gl)*matrix(gr))
_mul(gl::AbstractKernel,gr::TimeLocalKernel) = similar(gl,matrix(gl)*matrix(gr))
_mul(gl::TimeLocalKernel,gr::TimeLocalKernel) = similar(gr,matrix(gl)*matrix(gr))


_trapz_dressing(g,d) = return matrix(g) - eltype(d)(0.5)*d
function _biased_mul(gl::L,gr::R) where { L <: Union{RetardedKernel,AdvancedKernel}, R<: Union{RetardedKernel,AdvancedKernel} }
    bs = blocksize(gl)
    dl = extract_blockdiag(matrix(gl),bs,compression = compression(gl))
    dr = extract_blockdiag(matrix(gr),bs,compression = compression(gl))
    weighted_L = _trapz_dressing(gl,dl)
    weighted_R = _trapz_dressing(gr,dr)
    return weighted_L * weighted_R,dl,dr
end
function _mul(gl::Ker, gr::Ker) where Ker<:Union{RetardedKernel,AdvancedKernel}
    biased_result,_,_ = _biased_mul(gl,gr) 
    result = biased_result - extract_blockdiag(biased_result,blocksize(gl), compression = compression(gl)) # exact diagonal is zero
    result *= step(gl)
    return similar(gl, result)
end

#For the trapz rule and neglecting the side effect that should be absorbed
#in propoer definition of the problem the same rule apply
function _mul(gl::L,gr::R) where { L <: Union{RetardedKernel,AdvancedKernel}, R<: Union{RetardedKernel,AdvancedKernel} }
    biased_result,dl,dr = _biased_mul(gl,gr) 
    correction = eltype(dl)(1/4)*dl*dr
    result = biased_result + correction
    result *= step(gl)
    return similar(gl,result)
end
function _mul(gl::Kernel,gr::R) where R<: Union{RetardedKernel,AdvancedKernel}
    dr = extract_blockdiag(matrix(gr),blocksize(gl),compression = compression(gl))
    T = eltype(dr)
    weighted_L = matrix(gl)
    weighted_R = _trapz_dressing(gr,dr)
    result = weighted_L * weighted_R
    result *= step(gl)
    return similar(gl,result)
end
function _mul(gl::L,gr::Kernel) where L<: Union{RetardedKernel,AdvancedKernel}
    dl = extract_blockdiag(matrix(gl),blocksize(gl),compression = compression(gl))
    weighted_L = _trapz_dressing(gl,dl)
    weighted_R = matrix(gr) 
    result = weighted_L * weighted_R
    result *= step(gl)
    return similar(gl,result)
end
function _mul(gl::Kernel,gr::Kernel)
    weighted_L = matrix(gl)
    weighted_R = matrix(gr) 
    result = weighted_L * weighted_R
    result *= step(gl)
    return similar(gl,result)
end

function *(gl::AbstractKernel,gr::AbstractKernel)
    return mul(gl,gr) 
end
