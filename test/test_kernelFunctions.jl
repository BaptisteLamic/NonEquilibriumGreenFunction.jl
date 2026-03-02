using NonequilibriumGreenFunction
@testitem "Test accessor KernelFunction" begin
    domain = KernelDomain((0.0, 1.0), n_steps=512)
    modulation_11(x, y) = exp(-abs(x)) * cos(y)
    modulation_22(x, y) = (atan(x) * sin(y) + 1)
    kf = KernelFunction((x, y) -> [modulation_11(x, y) 1.0; 0.0 modulation_22(x, y)], domain)
    @test blocksize(kf) == 2
    @test eltype(kf) == Float64
    @test size(kf, 1) == size(domain, 1) * blocksize(kf)
    @test size(kf) == (size(domain, 2) * blocksize(kf), size(domain, 2) * blocksize(kf))
end


@testitem "Test symetric KernelFunction" begin
    domain = KernelDomain((0.0, 1.0), n_steps=512)
    modulation_11(x, y) = exp(-abs(x)) * cos(y)
    modulation_22(x, y) = (atan(x) * sin(y) + 1)
    kf = KernelFunction((x, y) -> [modulation_11(x, y) 1.0; 0.0 modulation_22(x, y)], domain)
    buf = zeros(eltype(kf), size(kf, 1))
    NonEquilibriumGreenFunction.col!(buf, kf, 1)
    @test norm(buf[1:blocksize(kf):end] - modulation_11.(xaxis(domain), yaxis(domain)[1])) < 1e-14
    @test norm(buf[1:blocksize(kf):end]) > 1
    NonEquilibriumGreenFunction.row!(buf, kf, 1)
    @test norm(buf[1:blocksize(kf):end] - modulation_11.(xaxis(domain)[1], yaxis(domain))) < 1e-14
    @test norm(buf[1:blocksize(kf):end]) > 1
end

@testitem "Test domain reduction" begin
    domain = KernelDomain((0.0, 1.0), n_steps=512)
    modulation_11(x, y) = exp(-abs(x)) * cos(y)
    modulation_22(x, y) = (atan(x) * sin(y) + 1)
    kf = KernelFunction((x, y) -> [modulation_11(x, y) 0.0; 0.0 modulation_22(x, y)], domain)
    buf_full_kernel = zeros(eltype(kf), size(kf, 2))
    reduced_domain = KernelDomain((0.2, 0.8), n_steps=256)
    reduced_kf = NonEquilibriumGreenFunction.restrict_domain(kf, reduced_domain)
    buf_reduced_kernel = zeros(eltype(kf), size(reduced_kf, 2))
    NonEquilibriumGreenFunction.row!(buf_full_kernel, kf, 1)
    NonEquilibriumGreenFunction.row!(buf_reduced_kernel, reduced_kf, 1)
    @test reduced_kf.domain == reduced_domain
    @test buf_reduced_kernel[1:blocksize(reduced_kf):end] == modulation_11.(xaxis(reduced_domain)[1], yaxis(reduced_domain))
    @test norm(modulation_11.(xaxis(domain)[1], yaxis(domain))) > 1
    @test size(kf, 2) > size(reduced_kf, 2)
end

@testitem "fill_with_kernel!" begin
    domain = KernelDomain((0.0, 1.0), n_steps=512)
    modulation(x) = sin(x)
    kf = KernelFunction(
        (x, y) -> [modulation(x - pi * y) 2.0; -1.0 modulation(2x - y)],
        domain
    )
    buf_row = zeros(eltype(kf), size(kf, 2))
    matrix = zeros(eltype(kf), size(kf))
    NonEquilibriumGreenFunction.fill_with_kernel!(matrix, kf)
    for i in axes(matrix, 1)
        NonEquilibriumGreenFunction.row!(buf_row, kf, i)
        @test matrix[i, :] == buf_row
    end

    buf_col = zeros(eltype(kf), size(kf, 1))
    matrix = zeros(eltype(kf), size(kf))
    NonEquilibriumGreenFunction.fill_with_kernel!(matrix, kf)
    for i in axes(matrix, 2)
        NonEquilibriumGreenFunction.col!(buf_col, kf, i)
        @test matrix[:, i] == buf_col
    end
end

@testitem "Test LinearOperator interface" begin
    using LinearAlgebra
    using LowRankApprox
    domain = KernelDomain((0.0, 1.0), n_steps=512)
    modulation(x) = sin(x)
    kf = KernelFunction(
        (x, y) -> [modulation(x - pi * y) 2.0; -1.0 modulation(2x - y)],
        domain
    )
    operator = LinearOperator(kf)
    @test size(kf) == size(operator)
    v = randn(eltype(kf), size(kf, 2), 4)
    matrix = kf[:, :]
    y1 = operator * v
    y2 = matrix * v
    @test norm(y1 - y2) / norm(y2) < 1E-14
    u = randn(eltype(kf), size(kf, 1), 4)
    yt1 = operator' * v
    yt2 = matrix' * v
    @test norm(yt1 - yt2) / norm(yt2) < 1E-14
end