
@testitem "Test accessor KernelFunction" begin
    domain = KernelDomain((0.0, 1.0), n_steps=512)
    modulation_11(x, y) = exp(-abs(x)) * cos(y)
    modulation_22(x, y) = (atan(x) * sin(y) + 1)
    kf = KernelFunction((x, y) -> [modulation_11(x, y) 1.0; 0.0 modulation_22(x, y)], domain)
    @test kf.blocksize == 2
    @test kf.eltype == Float64
    @test size(kf, 1) == size(domain, 1) * kf.blocksize
    @test size(kf) == (size(domain, 2) * kf.blocksize, size(domain, 2) * kf.blocksize)
end


@testitem "Test symetric KernelFunction" begin
    domain = KernelDomain((0.0, 1.0), n_steps=512)
    modulation_11(x, y) = exp(-abs(x)) * cos(y)
    modulation_22(x, y) = (atan(x) * sin(y) + 1)
    kf = KernelFunction((x, y) -> [modulation_11(x, y) 1.0; 0.0 modulation_22(x, y)], domain)
    buf = zeros(kf.eltype, size(kf, 1))
    NonEquilibriumGreenFunction.col!(buf, kf, 1)
    @test norm(buf[1:kf.blocksize:end] - modulation_11.(xaxis(domain), yaxis(domain)[1])) < 1e-14
    @test norm(buf[1:kf.blocksize:end]) > 1
    NonEquilibriumGreenFunction.row!(buf, kf, 1)
    @test norm(buf[1:kf.blocksize:end] - modulation_11.(xaxis(domain)[1], yaxis(domain))) < 1e-14
    @test norm(buf[1:kf.blocksize:end]) > 1
end

@testitem "Test domain reduction" begin
    domain = KernelDomain((0.0, 1.0), n_steps=512)
    modulation_11(x, y) = exp(-abs(x)) * cos(y)
    modulation_22(x, y) = (atan(x) * sin(y) + 1)
    kf = KernelFunction((x, y) -> [modulation_11(x, y) 0.0; 0.0 modulation_22(x, y)], domain)
    buf_full_kernel = zeros(kf.eltype, size(kf, 2))
    reduced_domain = KernelDomain((0.2, 0.8), n_steps=256)
    reduced_kf = NonEquilibriumGreenFunction.restrict_domain(kf, reduced_domain)
    buf_reduced_kernel = zeros(kf.eltype, size(reduced_kf, 2))
    NonEquilibriumGreenFunction.row!(buf_full_kernel, kf, 1)
    NonEquilibriumGreenFunction.row!(buf_reduced_kernel, reduced_kf, 1)
    @test reduced_kf.domain == reduced_domain
    @test buf_reduced_kernel[1:reduced_kf.blocksize:end] == modulation_11.(xaxis(reduced_domain)[1], yaxis(reduced_domain))
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
    buf_row = zeros(kf.eltype, size(kf, 2))
    matrix = zeros(kf.eltype, size(kf))
    NonEquilibriumGreenFunction.fill_with_kernel!(matrix, kf)
    for i in axes(matrix, 1)
        NonEquilibriumGreenFunction.row!(buf_row, kf, i)
        @test matrix[i, :] == buf_row
    end

    buf_col = zeros(kf.eltype, size(kf, 1))
    matrix = zeros(kf.eltype, size(kf))
    NonEquilibriumGreenFunction.fill_with_kernel!(matrix, kf)
    for i in axes(matrix, 2)
        NonEquilibriumGreenFunction.col!(buf_col, kf, i)
        @test matrix[:, i] == buf_col
    end
end

@testitem "Test KernelFunction call" begin
    domain = KernelDomain((0.0, 1.0), n_steps=512)
    modulation(x) = sin(x)
    kf = KernelFunction(
        (x, y) -> [modulation(x - pi * y) 2.0; -1.0 modulation(2x - y)],
        domain
    )
    @test kf(0.5, 0.5) == [modulation(0.5 - pi * 0.5) 2.0; -1.0 modulation(2 * 0.5 - 0.5)]
    ix, iy = 128, 128
    x = xaxis(domain)[ix]
    y = yaxis(domain)[iy]
    @test kf(ix, iy, 1, 1) == modulation(x - pi * y)
    @test kf(ix, iy, 2, 2) == modulation(2x - y)
end