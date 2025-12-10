export KernelFunction

using ACAFact

struct KernelFunction{F,A,T}
    f::F
    axis::A
    eltype::Type{T}
    blocksize::Int
end
"""
Instantiate a KernelFunction from a function `f(x,y)` taking two arguments, and a domain tuple `(a,b)`.
"""
function KernelFunction(f, kernelDomain)
    f0 = f(kernelDomain[1], kernelDomain[1])
    @assert size(f0, 1) == size(f0, 2) "Kernel function must return square matrices."
    blocksize = size(f0, 1)
    T = eltype(f0)
    return KernelFunction(f, kernelDomain, T, blocksize)
end

function Base.size(kf::KernelFunction, dim)
    n = length(kf.axis) * kf.blocksize
    return n
end
function Base.size(kf::KernelFunction)
    n = size(kf, 1)
    return (n, n)
end
function Base.eltype(kf::KernelFunction)
    return kf.eltype
end


function ACAFact.col!(buf, kf::KernelFunction, j)
    step_size = kf.blocksize
    block_number, inblock_index = blockindex(j, kf.blocksize)
    for i in 1:length(kf.axis)
        buf[(i-1)*step_size+1:i*step_size] = kf.f(kf.axis[i], kf.axis[block_number])[:, inblock_index]
    end
    return buf
end

function ACAFact.row!(buf, kf::KernelFunction, j)
    step_size = kf.blocksize
    block_number, inblock_index = blockindex(j, kf.blocksize)
    for i in 1:length(kf.axis)
        buf[(i-1)*step_size+1:i*step_size] = kf.f(kf.axis[block_number], kf.axis[i])[inblock_index, :]
    end
    return buf
end


@testitem "Test KernelFunction" begin
    using ACAFact
    using StaticArrays
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