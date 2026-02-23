export KernelFunction

using ACAFact

struct KernelFunction{F,A,T}
    f::F
    domain::Tuple{A,A}
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
    for i in 1:length(row_axis)
        buf[(i-1)*step_size+1:i*step_size] = kf.f(row_axis[i], col_axis[block_number])[:, inblock_index]
    end
    return buf
end

function ACAFact.row!(buf, kf::KernelFunction, j)
    step_size = kf.blocksize
    block_number, inblock_index = blockindex(j, kf.blocksize)
    row_axis = kf.domain[1]
    col_axis = kf.domain[2]
    for i in 1:length(col_axis)
        buf[(i-1)*step_size+1:i*step_size] = kf.f(row_axis[block_number], col_axis[i])[inblock_index, :]
    end
    return buf
end

@testitem "Test symetric KernelFunction" begin
    domain = (0.0:0.1:1.0)
    modulation_11(x,y) = exp(-abs(x))*cos(y)
    modulation_22(x,y) = (atan(x) * sin(y) + 1)
    kf = KernelFunction((x, y) -> [modulation_11(x,y) 1.0; 0.0 modulation_22(x, y)], domain)
    @test kf.blocksize == 2
    @test kf.eltype == Float64
    @test size(kf, 1) == length(domain) * kf.blocksize
    buf = zeros(kf.eltype, size(kf, 1))
    NonEquilibriumGreenFunction.ACAFact.col!(buf, kf, 1)
    @test buf[2:kf.blocksize:end] == zeros(length(domain))
    @test norm(buf[1:kf.blocksize:end] - modulation_11.(domain, domain[1])) < 1e-14
    @test norm(buf[1:kf.blocksize:end]) > 1
    NonEquilibriumGreenFunction.ACAFact.row!(buf, kf, 1)
    @test buf[2:kf.blocksize:end] == ones(length(domain))
    @test norm(buf[1:kf.blocksize:end] - modulation_11.(domain[1], domain)) < 1e-14
    @test norm(buf[1:kf.blocksize:end]) > 1
end

function restrict_domain(kf, dom1, dom2)
    #We first test that the new domain is included in the old one.
    for (new_dom, old_dom) in zip((dom1, dom2), kf.domain)
        if !all(in(old_dom), new_dom)
            throw(DomainError(new_dom, "new domain must be a subset of $(old_dom)"))
        end
    end
    return KernelFunction(kf.f, (dom1, dom2), kf.eltype, kf.blocksize)
end

@testitem "Test domain reduction" begin
    domain = (0.0:0.1:1.0)
    modulation_11(x,y) = exp(-abs(x))*cos(y)
    modulation_22(x,y) = (atan(x) * sin(y) + 1)
    kf = KernelFunction((x, y) -> [modulation_11(x,y) 0.0; 0.0 modulation_22(x, y)], domain)
    buf_full_kernel = zeros(kf.eltype, size(kf, 2))
    reduced_domain = 0.0:0.1:0.5
    reduced_kf = NonEquilibriumGreenFunction.restrict_domain(kf, domain, reduced_domain)
    buf_reduced_kernel = zeros(kf.eltype, size(reduced_kf, 2))
    NonEquilibriumGreenFunction.ACAFact.row!(buf_full_kernel, kf, 1)
    NonEquilibriumGreenFunction.ACAFact.row!(buf_reduced_kernel, reduced_kf, 1)
    @test length(buf_reduced_kernel) == length(reduced_domain)*kf.blocksize
    @test buf_reduced_kernel[1:reduced_kf.blocksize:end] == modulation_11.(reduced_domain[1], reduced_domain)
    @test norm(modulation_11.(domain[1], domain)) > 1
    @test size(kf, 2) > size(reduced_kf, 2)
end

function fill_with_kernel!(array, kf)
    bs = kf.blocksize
    for j in 1:length(kf.domain[2])
        for i in 1:length(kf.domain[1])
            array[(i-1)*bs+1:i*bs,(j-1)*bs+1:j*bs] .= kf.f(kf.domain[1][i], kf.domain[2][j])
        end
    end
    return array
end

@testitem "fill_with_kernel!" begin
    modulation(x) = sin(x)
    kf = KernelFunction(
        (x, y) -> [modulation(x - pi*y) 2.0; -1.0 modulation(2x - y)],
        (0.0:0.1:1.0),
        (0.0:0.2:10.0)
    )
    buf_row = zeros(kf.eltype, size(kf, 2))
    matrix = zeros(kf.eltype, size(kf))
    NonEquilibriumGreenFunction.fill_with_kernel!(matrix, kf)
    for i in 1:size(matrix,1)
        NonEquilibriumGreenFunction.ACAFact.row!(buf_row, kf, i)
        @test matrix[i,:] == buf_row
    end

    buf_col = zeros(kf.eltype, size(kf, 1))
    matrix = zeros(kf.eltype, size(kf))
    NonEquilibriumGreenFunction.fill_with_kernel!(matrix, kf)
    for i in 1:size(matrix,1)
        NonEquilibriumGreenFunction.ACAFact.col!(buf_col, kf, i)
        @test matrix[:,i] == buf_col
    end
end