@testitem "Discretisation creation and accessor" begin
    using LinearAlgebra
    for T = [Float64, ComplexF64]
        bs, N, Dt = 2, 128, 2.0
        ax = LinRange(-Dt / 2, Dt, N)
        A = randn(T, bs * N, bs * N)
        B = randn(T, bs * N, bs * N)
        dA = TrapzDiscretisation(ax, A, bs, NONCompression())
        dB = similar(dA, B)
        @test matrix(dA) == A
        @test matrix(dB) == B
        @test axis(dA) == ax
        @test scalartype(dA) == T
        I = (2:5, 2:25)
        @test dA[I...] == [A[blockrange(i, bs), blockrange(j, bs)] for i in I[1], j in I[2]]
        p, q = 2, 67
        @test dA[p, q] == A[blockrange(p, bs), blockrange(q, bs)]
    end
end

@testitem "Kernel creation and accessor" begin
    using LinearAlgebra
    for compressionMethod in (NONCompression(), HssCompression())
        for T = [Float32, Float64, ComplexF64]
            bs, N, Dt = 2, 128, 2.0
            ax = LinRange(-Dt / 2, Dt, N)
            for (Kernel, Causality) in zip(
                (RetardedKernel, AdvancedKernel, AcausalKernel),
                (Retarded, Advanced, Acausal)
            )
                A = randn(T, bs * N, bs * N)
                GA = Kernel(ax, A, bs, compressionMethod)
                @test causality(GA) isa Causality
                @test matrix(GA) == A
                @test eltype(matrix(GA)) == eltype(A)
                @test eltype(matrix(GA)) == T
                @test axis(GA) == ax
                @test compression(GA) == compressionMethod
                @test blocksize(GA) == bs
                @test scalartype(GA) == T
                @test axis(GA) == ax
            end
        end
    end
end

@testitem "Kernel discretization" begin
    using LinearAlgebra
    for T = [Float64, ComplexF64]
        bs, N, Dt = 2, 128, 2.0
        tol = 100 * max(1E-6, eps(real(T)))
        ax = LinRange(-Dt / 2, Dt, N)
        foo(x, y) = T <: Complex ? T.(1im .* [x x+y; x-y y]) : T.([x x+y; x-y y])
        foo_st(x) = T <: Complex ? T.(1im * [x 2x; 0 x]) : T.([x 2x; 0 x])
        foo_st(x, y) = foo_st(x - y)
        for cpr in (NONCompression(), HssCompression())
            for Kernel in (discretize_retardedkernel, discretize_advancedkernel, discretize_acausalkernel)
                GA = Kernel(ax, foo_st, compression=cpr, stationary=true)
                compress!(GA)
                GB = Kernel(ax, foo_st, compression=cpr, stationary=false)
                compress!(GB)
                @test matrix(GA) - matrix(GB) |> norm < tol
            end
        end
    end
end
@testitem "scalar low rank kernel discretization" begin
    using LinearAlgebra
    N, Dt = 128, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    for _causality in (Acausal, Retarded, Advanced)
        for T = [Float64, ComplexF64, ComplexF32]
            tol = 5 * max(1E-14, eps(real(T)))
            f(x) = T <: Complex ? T(exp(-1im * x)) : T(cos(x))
            g(x) = T <: Complex ? T(exp(-4im * x)) : T(sin(4x))
            _getMask(::Type{Acausal}) = (i, j) -> T(true)
            _getMask(::Type{Retarded}) = (i, j) -> T(i >= j)
            _getMask(::Type{Advanced}) = (i, j) -> T(i <= j)
            mask = _getMask(_causality)
            refMatrix = [f(x) * g(y) * mask(x, y) for x in ax, y in ax]
            for cpr in (NONCompression(), HssCompression())
                GA = discretize_lowrank_kernel(TrapzDiscretisation, _causality, ax, f, g, compression=cpr)
                compress!(GA)
                @test matrix(GA) - refMatrix |> norm < tol * N^2
                @test _causality() == causality(GA)
            end
        end
    end
end

@testitem "Kernel discretization scalar kernel" begin
    using LinearAlgebra
    for T = [Float64, ComplexF64]
        bs, N, Dt = 2, 256, 2.0
        tol = 100 * max(1E-6, eps(real(T)))
        ax = LinRange(-Dt / 2, Dt, N)
        foo(x, y) = T <: Complex ? T.(1im .* x + y) : T.(x + y)
        foo_st(x) = T <: Complex ? T.(1im * x) : T.(x)
        foo_st(x, y) = foo_st(x - y)
        for cpr in (NONCompression(), HssCompression())
            for Kernel in (discretize_retardedkernel, discretize_advancedkernel, discretize_acausalkernel)
                GA = Kernel(ax, foo_st, compression=cpr, stationary=true)
                compress!(GA)
                GB = Kernel(ax, foo_st, compression=cpr, stationary=false)
                compress!(GB)
                @test matrix(GA) - matrix(GB) |> norm < tol
            end
        end
    end
end


@testitem "Kernel discretization causality" begin
    using LinearAlgebra
    for T = [Float64, ComplexF64]
        N, Dt = 128, 2.0
        tol = 100 * max(1E-6, eps(real(T)))
        ax = LinRange(-Dt / 2, Dt, N)
        foo(x, y) = T <: Complex ? T.(1im .* [x x+y; x-y y]) : T.([x x+y; x-y y])
        for Kernel in (discretize_retardedkernel, discretize_advancedkernel, discretize_acausalkernel)
            GA = Kernel(ax, foo, compression=NONCompression(), stationary=false)
            _causality = causality(GA)
            bs = blocksize(GA)
            B = zeros(T, N * bs, N * bs)
            for it in 1:length(ax)
                t0 = _causality == Retarded() || _causality == Acausal() ? 1 : it
                t1 = _causality == Advanced() || _causality == Acausal() ? length(ax) : it
                for itp in t0:t1
                    B[blockrange(it, bs), blockrange(itp, bs)] .= foo(ax[it], ax[itp])
                end
            end
            @test B == matrix(GA)
        end
    end
end


@testitem "kernel sum and diff" begin
    using LinearAlgebra
    for T = [Float64, ComplexF64]
        bs, N, Dt = 2, 128, 2.0
        ax = LinRange(-Dt / 2, Dt, N)
        for left_kernel in (RetardedKernel, AdvancedKernel, AcausalKernel),
            right_kernel in (RetardedKernel, AdvancedKernel, AcausalKernel)

            GA = left_kernel(ax, randn(T, bs * N, bs * N), bs, NONCompression())
            GB = right_kernel(ax, randn(T, bs * N, bs * N), bs, NONCompression())
            gsum = GA + GB
            gdiff = GA - GB
            @test matrix(gsum) == matrix(GA) + matrix(GB)
            @test matrix(gdiff) == matrix(GA) - matrix(GB)
            if left_kernel == right_kernel
                @test causality(gsum) == causality(GA)
                @test causality(gdiff) == causality(GA)
            else
                @test causality(gsum) == Acausal()
                @test causality(gdiff) == Acausal()
            end
        end
    end
end


@testitem "kernel adjoint" begin
    using LinearAlgebra
    for T = [Float64, ComplexF64]
        for cpr in (HssCompression(), NONCompression())
            for kernel in (RetardedKernel, AdvancedKernel, AcausalKernel)
                bs, N, Dt = 2, 128, 2.0
                ax = LinRange(-Dt / 2, Dt, N)
                A = randn(T, bs * N, bs * N)
                GA = kernel(ax, A, bs, cpr)
                @test norm(matrix(GA') - A') < 1E-10
                @test norm(matrix(GA') - A') / norm(A) < 1E-10
            end
        end
    end
end


@testitem "kernel scaling" begin
    using LinearAlgebra
    for T = [Float64, ComplexF64]
        for cpr in (HssCompression(), NONCompression())
            for kernel_creator in (RetardedKernel, AdvancedKernel, AcausalKernel)
                bs, N, Dt = 2, 128, 2.0
                ax = LinRange(-Dt / 2, Dt, N)
                GA = kernel_creator(ax, randn(T, bs * N, bs * N), bs, cpr)
                @test matrix(GA * 2) == 2 * matrix(GA)
                @test GA * 2I isa Kernel
                @test matrix(GA * 2I) == 2I * matrix(GA)
                @test causality(GA * 2I) == causality(GA)
                @test matrix(-GA) == -matrix(GA)
                @test -GA isa Kernel
            end
        end
    end
end


@testitem "kernel products" begin
    using LinearAlgebra
    N, Dt = 128, 2.0
    ax = LinRange(-Dt / 2, Dt, N)
    x0 = Dt / 4.5
    sigma0 = Dt / 5

    function _trapz(p, k, q)
        if q >= p
            return 0.0
        elseif k == p || k == q
            return 1 / 2
        elseif k > p || k < q
            return 0.0
        else
            return 1.0
        end
    end
    function _integrate(f, g, p, q, k_sup, k_inf, step)
        if k_sup <= k_inf
            return zero(f[1, 1])
        else
            return step * sum([_trapz(k_sup, k, k_inf) * f[p, k] * g[k, q] for k = k_inf:k_sup])
        end
    end
    for cpr in (HssCompression(), NONCompression())
        for T = [Float64, ComplexF64]
            c = 100
            tol = c * max(1E-6, eps(real(T)))
            foo(x, y) = T <: Complex ? T.(1im .* [x x+y; x-y y]) : T.([x x+y; x-y y])
            foo(x) = T <: Complex ? T.(1im * [x 2x; 0 x]) : T.([x 2x; 0 x])
            gooL(x, y) = exp(-((x - x0)^2 + y^2) / sigma0^2) .* foo(x, y)
            gooR(x, y) = exp(-((x + x0)^2 + y^2) / sigma0^2) .* foo(x, y)
            for (Left, Right) in zip(
                (discretize_retardedkernel, discretize_advancedkernel, discretize_acausalkernel, discretize_retardedkernel, discretize_acausalkernel),
                (discretize_retardedkernel, discretize_advancedkernel, discretize_acausalkernel, discretize_acausalkernel, discretize_advancedkernel)
            )
                GL = Left(ax, gooL, compression=NONCompression())
                GR = Right(ax, gooR, compression=NONCompression())
                discretization_GL = discretization(GL)
                discretization_GR = discretization(GR)
                causality_left = causality(GL)
                causality_right = causality(GR)
                prod = GL * GR
                prod_discretization = discretization(prod)
                bs = blocksize(GL)
                integral = zeros(T, N * bs, N * bs)
                for p in 1:length(ax)
                    for q in 1:length(ax)
                        k_inf = 1
                        k_sup = length(ax)
                        #compute the integral boundary
                        isretarded(GR) && (k_inf = max(k_inf, q))
                        isadvanced(GR) && (k_sup = min(k_sup, q))
                        isretarded(GL) && (k_sup = min(k_sup, p))
                        isadvanced(GL) && (k_inf = max(k_inf, p))
                        integral[blockrange(p, bs), blockrange(q, bs)] = _integrate(
                            discretization_GL,
                            discretization_GR,
                            p, q, k_sup, k_inf, step(axis(GR))
                        )
                    end
                end
                if causality(GL) == causality(GR) == Retarded()
                    @test causality(prod) == Retarded()
                elseif causality(GL) == causality(GR) == Advanced()
                    @test causality(prod) == Advanced()
                else
                    @test causality(prod) == Acausal()
                end
                @test norm(integral - matrix(prod)) / norm(integral) < tol
            end
        end
    end
end

@testitem "Scaling of norm estimates with N" begin
    using LinearAlgebra
    atol = 1E-5
    rtol = 1E-5
    kest = 20
    compression = HssCompression(atol=atol, rtol=rtol, kest=kest)
    t0, t1 = 0, 1
    for T in [Float64, ComplexF64]
        g(x) = T(sin(9 * x))
        g(x, y) = g(x - y)
        n1 = 128
        n2 = 2 * n1
        g1 = discretize_retardedkernel(LinRange(t0, t1, n1), g, compression=compression)
        g2 = discretize_retardedkernel(LinRange(t0, t1, n2), g, compression=compression)
        @test abs(norm(g1) / norm(g2) - 1) < 0.1
    end
end

@testitem "Scaling of norm estimates with tf" begin
    using LinearAlgebra
    atol = 1E-5
    rtol = 1E-5
    kest = 20
    n = 128
    compression = HssCompression(atol=atol, rtol=rtol, kest=kest)
    for T in [Float64, ComplexF64]
        g(x) = T(cos(10 * x))
        g(x, y) = g(x - y)
        tf1 = 1
        tf2 = 2 * tf1
        g1 = discretize_retardedkernel(LinRange(0, tf1, n), g, compression=compression)
        g2 = discretize_retardedkernel(LinRange(0, tf2, n), g, compression=compression)
        @test abs(norm(g1) / norm(g2) - 1) < 0.1
    end
end

@testitem "solving dyson equation" begin
    using LinearAlgebra
    atol = 1E-5
    rtol = 1E-5
    kest = 20
    for compressionMethod in (NONCompression(), HssCompression(atol=atol, rtol=rtol, kest=kest))
        for T in [Float32, Float64, ComplexF32, ComplexF64]
            g(x) = T(sin(9 * x))
            g(x, y) = g(x - y)
            k(x) = T(-cos(9 * x))
            k(x, y) = k(x - y)
            sol_ana(x) = T((18 * exp(-x / 2) * sin(sqrt(323) * x / 2)) / sqrt(323) * (x >= 0 ? 1.0 : 0.0))
            sol_ana(x, y) = sol_ana(x - y)
            t0, t1 = 0, 10
            ax = LinRange(t0, t1, 2^8)
            G0 = discretize_retardedkernel(ax, g, compression=compressionMethod)
            @test scalartype(G0) == T
            K = discretize_retardedkernel(ax, k, compression=compressionMethod)
            @test scalartype(K) == T
            G = solve_dyson(G0, K)
            G_ana = discretize_retardedkernel(ax, sol_ana, compression=compressionMethod)
            @test norm(matrix(G - G_ana)) / norm(G_ana |> matrix) < 1E-3
        end
    end
end