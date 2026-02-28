export KernelFunction
export KernelDomain
export xaxis
export yaxis
export is_included
export blocksize

import Base: IndexStyle, axes, eltype, getindex, size



struct KernelDomain{T}
    x_min::T
    x_max::T
    y_min::T
    y_max::T
    x_steps::Int
    y_steps::Int
end
xaxis(domain::KernelDomain) = range(domain.x_min, domain.x_max, length=domain.x_steps)
yaxis(domain::KernelDomain) = range(domain.y_min, domain.y_max, length=domain.y_steps)

function KernelDomain(x_lims, y_lims=x_lims; n_steps=nothing, x_steps=n_steps, y_steps=n_steps)
    return KernelDomain(first(x_lims), last(x_lims), first(y_lims), last(y_lims), x_steps, y_steps)
end

function Base.size(domain::KernelDomain, i)
    if i in (1, 2)
        return domain.x_steps
    elseif i == 2
        return domain.y_steps
    end
    return 1
end
function Base.size(domain::KernelDomain)
    return (size(domain, 1), size(domain, 2))
end
function is_included(domain1::KernelDomain, domain2::KernelDomain)
    return domain1.x_min >= domain2.x_min && domain1.x_max <= domain2.x_max &&
           domain1.y_min >= domain2.y_min && domain1.y_max <= domain2.y_max
end


struct KernelFunction{D,T,BlockSize}
    block_getter
    element_getter
    domain::KernelDomain{D}
end


function KernelFunction(block_getter, domain::KernelDomain{D}) where D
    f0 = block_getter(domain.x_min, domain.y_min)
    @assert size(f0, 1) == size(f0, 2) "Kernel function must return square matrices."
    blocksize = size(f0, 1)
    T = eltype(f0)
    element_getter = _build_default_element_getter(block_getter, blocksize)
    return KernelFunction{D,T,blocksize}(block_getter, element_getter, domain)
end
"""Build a kernel function using an element getter. The element getter must take as input the coordinates (x, y) and the in-block indices (i, j) and return the value of the kernel at that point."""
function KernelFunction(element_getter, domain::KernelDomain{D}, blocksize::Int) where D
    block_getter = _build_default_block_getter(element_getter, blocksize)
    T = eltype(element_getter(domain.x_min, domain.y_min, 1, 1))
    return KernelFunction{D,T,blocksize}(block_getter, element_getter, domain)
end
function Base.size(kf::KernelFunction, dim)
    if dim in (1, 2)
        return size(kf.domain, dim) * blocksize(kf)
    end
    return 1
end
function Base.size(kf::KernelFunction)
    return (size(kf, 1), size(kf, 2))
end
function Base.eltype(::KernelFunction{D,T,BlockSize}) where {D,T,BlockSize}
    return T
end

function getindex(kf::KernelFunction, i::Int, j::Int)
    bs = blocksize(kf)
    block_i, inblock_i = blockindex(i, bs)
    block_j, inblock_j = blockindex(j, bs)
    x = xaxis(kf.domain)[block_i]
    y = yaxis(kf.domain)[block_j]
    return kf.element_getter(x, y, inblock_i, inblock_j)
end

function getindex(kf::KernelFunction, ::Colon, j::Int)
    return [kf[i, j] for i in 1:size(kf, 1)]
end
function getindex(kf::KernelFunction, i::Int, ::Colon)
    return [kf[i, j] for j in 1:size(kf, 2)]
end
function getindex(kf::KernelFunction, ::Colon, ::Colon)
    results = Matrix{eltype(kf)}(undef, size(kf))
    for j in axes(kf, 2)
        for i in axes(kf, 1)
            results[i, j] = kf[i, j]
        end
    end
    return results
end


IndexStyle(::KernelFunction) = IndexCartesian()
axes(kf::KernelFunction) = (Base.OneTo(size(kf, 1)), Base.OneTo(size(kf, 2)))
function axes(kf::KernelFunction, i)
    if i < 1 || i > 2
        throw(ArgumentError("Dimension must be 1 or 2."))
    else
        return axes(kf)[i]
    end
    return axes(kf)[i]
end

function col!(buf, kf::KernelFunction, j)
    buf[:] .= kf[:, j]
    return buf
end

function row!(buf, kf::KernelFunction, i)
    for j in axes(kf, 2)
        buf[j] = kf[i, j]
    end
    return buf
end

function blocksize(::KernelFunction{D,T,BlockSize}) where {D,T,BlockSize}
    return BlockSize
end


function _build_default_element_getter(block_getter, blocksize)
    function _getter(x, y, i, j)
        if i < 1 || i > blocksize || j < 1 || j > blocksize
            throw(BoundsError("In-kernel indices must be between 1 and the block size."))
        end
        return block_getter(x, y)[i, j]
    end
end
function _build_default_block_getter(element_getter, blocksize)
    function _getter(x, y, i, j)
        if i < 1 || i > blocksize || j < 1 || j > blocksize
            throw(BoundsError("In-kernel indices must be between 1 and the block size."))
        end
        return [element_getter(x, y, i, j) for i in 1:blocksize, j in 1:blocksize]
    end
end
function (kf::KernelFunction)(x, y)
    return kf.block_getter(x, y)
end


function restrict_domain(kf, new_domain::KernelDomain)
    #We first test that the new domain is included in the old one.
    if !is_included(new_domain, kf.domain)
        throw(ArgumentError("The new domain must be included in the old one."))
    end
    return typeof(kf)(kf.block_getter, kf.element_getter, new_domain)
end

function fill_with_kernel!(array, kf)
    bs = blocksize(kf)
    x_dom = xaxis(kf.domain)
    y_dom = yaxis(kf.domain)
    for j in eachindex(y_dom)
        for i in eachindex(x_dom)
            array[(i-1)*bs+1:i*bs, (j-1)*bs+1:j*bs] .= kf.block_getter(x_dom[i], y_dom[j])
        end
    end
    return array
end

function eval_kernel(kf, ix, iy, i, j)
    x = xaxis(kf.domain)[ix]
    y = yaxis(kf.domain)[iy]
    return kf.block_getter(x, y)[i, j]
end

export eval_kernel