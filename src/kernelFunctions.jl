export KernelFunction

using ACAFact

struct KernelFunction{F,A,T}
    f::F
    domain :: Tuple{A,A}
    eltype::Type{T}
    blocksize::Int
end
function KernelFunction(f, dom1, dom2)
    f0 = f(dom1[1], dom2[1])
    @assert size(f0, 1) == size(f0, 2) "Kernel function must return square matrices."
    blocksize = size(f0, 1)
    T = eltype(f0)
    return KernelFunction(f, (dom1, dom2), T, blocksize)
end
"""
Instantiate a KernelFunction from a function `f(x,y)` taking two arguments, and a domain tuple `(a,b)`.
"""
function KernelFunction(f, kernelDomain)
    return KernelFunction(f, kernelDomain, kernelDomain)
end

function Base.size(kf::KernelFunction, dim)
    n = length(kf.domain[dim]) * kf.blocksize
    return n
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
    row_axis = kf.domain[1]
    col_axis = kf.domain[2]
    for i in 1:length(col_axis)
        buf[(i-1)*step_size+1:i*step_size] = kf.f(col_axis[i], row_axis[block_number])[:, inblock_index]
    end
    return buf
end

function ACAFact.row!(buf, kf::KernelFunction, j)
    step_size = kf.blocksize
    block_number, inblock_index = blockindex(j, kf.blocksize)
    row_axis = kf.domain[1]
    col_axis = kf.domain[2]
    for i in 1:length(row_axis)
        buf[(i-1)*step_size+1:i*step_size] = kf.f(col_axis[block_number], row_axis[i])[inblock_index, :]
    end
    return buf
end

@testitem "Test symetric KernelFunction" begin
    using ACAFact
    modulation(x) = sin(x)
    axis = (0.0:0.1:1.0)
    kf = KernelFunction((x, y) -> [modulation(x - 2y) 0.0; 0.0 modulation(2x - y)], axis)
    @test kf.blocksize == 2
    @test kf.eltype == Float64
    @test size(kf, 1) == length(axis) * kf.blocksize
    buf = zeros(kf.eltype, size(kf, 1))
    ACAFact.col!(buf, kf, 1)
    @test buf[2:kf.blocksize:end] == zeros(length(axis))
    @test norm(buf[1:kf.blocksize:end] - modulation.(axis)) < 1e-14
    @test norm(buf[1:kf.blocksize:end]) > 1
    ACAFact.row!(buf, kf, 1)
    @test buf[2:kf.blocksize:end] == zeros(length(axis))
    @test norm(buf[1:kf.blocksize:end] - modulation.(-2 .* axis)) < 1e-14
    @test norm(buf[1:kf.blocksize:end]) > 1
end

function update_domain(kf,dom1, dom2)
    return KernelFunction(kf.f, (dom1, dom2), kf.eltype, kf.blocksize)
end

@testitem "Test domain reduction" begin
     using ACAFact
    modulation(x) = sin(x)
    axis = (0.0:0.1:1.0)
    kf = KernelFunction((x, y) -> [modulation(x - 2y) 0.0; 0.0 modulation(2x - y)], axis)
    buf_full_kernel = zeros(kf.eltype, size(kf, 1))
    reduced_domain = 0.0:0.1:0.5
    reduced_kf = NonEquilibriumGreenFunction.update_domain(kf,reduced_domain,axis)
    buf_reduced_kernel = zeros(kf.eltype, size(reduced_kf, 1))
    ACAFact.row!(buf_full_kernel, kf, 1)
    ACAFact.row!(buf_reduced_kernel, reduced_kf, 1)
    @test buf_reduced_kernel == buf_full_kernel[1:size(reduced_kf,1)]
    @test size(kf,1) > size(reduced_kf,1)
end