export KernelFunction
export KernelDomain
export xaxis
export yaxis
export is_included

using ACAFact

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


struct KernelFunction{F,D,T}
    f::F
    domain::KernelDomain{D}
    eltype::Type{T}
    blocksize::Int
end
function KernelFunction(f, domain::KernelDomain)
    f0 = f(domain.x_min, domain.y_min)
    @assert size(f0, 1) == size(f0, 2) "Kernel function must return square matrices."
    blocksize = size(f0, 1)
    T = eltype(f0)
    return KernelFunction(f, domain, T, blocksize)
end

function Base.size(kf::KernelFunction, dim)
    if dim in (1, 2)
        return size(kf.domain, dim) * kf.blocksize
    end
    return 1
end
function Base.size(kf::KernelFunction)
    return (size(kf, 1), size(kf, 2))
end
function Base.eltype(kf::KernelFunction)
    return kf.eltype
end


function ACAFact.col!(buf, kf::KernelFunction, j)
    step_size = kf.blocksize
    block_number, inblock_index = blockindex(j, kf.blocksize)
    row_axis = xaxis(kf.domain)
    col_axis = yaxis(kf.domain)
    for i in eachindex(row_axis)
        buf[(i-1)*step_size+1:i*step_size] = kf.f(row_axis[i], col_axis[block_number])[:, inblock_index]
    end
    return buf
end

function ACAFact.row!(buf, kf::KernelFunction, j)
    step_size = kf.blocksize
    block_number, inblock_index = blockindex(j, kf.blocksize)
    row_axis = xaxis(kf.domain)
    col_axis = yaxis(kf.domain)
    for i in eachindex(col_axis)
        buf[(i-1)*step_size+1:i*step_size] = kf.f(row_axis[block_number], col_axis[i])[inblock_index, :]
    end
    return buf
end

function restrict_domain(kf, new_domain::KernelDomain)
    #We first test that the new domain is included in the old one.
    if !is_included(new_domain, kf.domain)
        throw(ArgumentError("The new domain must be included in the old one."))
    end
    return KernelFunction(kf.f, new_domain, kf.eltype, kf.blocksize)
end


function fill_with_kernel!(array, kf)
    bs = kf.blocksize
    x_dom = xaxis(kf.domain)
    y_dom = yaxis(kf.domain)
    for j in eachindex(y_dom)
        for i in eachindex(x_dom)
            array[(i-1)*bs+1:i*bs, (j-1)*bs+1:j*bs] .= kf.f(x_dom[i], y_dom[j])
        end
    end
    return array
end

function eval_kernel(kf, ix, iy, i, j)
    x = xaxis(kf.domain)[ix]
    y = yaxis(kf.domain)[iy]
    return kf.f(x, y)[i, j]
end

function (kf::KernelFunction)(x, y)
    return kf.f(x, y)
end
function (kf::KernelFunction)(ix::Int, iy::Int, i::Int, j::Int)
    return eval_kernel(kf, ix, iy, i, j)
end

export eval_kernel