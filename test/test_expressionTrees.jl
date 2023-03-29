
@testitem "Test KernelExpression construction" begin
    using StaticArrays
    using LinearAlgebra
    for T in (ComplexF32,Float64)
        bs = 2
        N = 128
        ax = LinRange(-1. / 2, 1. ,N)
        foo(x,y) = T <: Complex ? T.(1im .* [x x+y; x-y y]) : T.([x x+y; x-y y])
        foo(x) = T <: Complex ? T.(1im * [x 2x; 0 x]) : T.([x 2x; 0 x])
        A = randn(T,bs*N,bs*N)
        GA = RetardedKernel(ax,A,bs,NONCompression())
        GB = AdvancedKernel(ax,foo, compression = NONCompression());
        LA = KernelLeaf(GA)
        LB = KernelLeaf(GB)
        LC = RetardedKernel(ax,randn(T,size(A)...),bs,NONCompression())
        @test arguments(KernelAdd(GA,GB)) == SA[LA, LB]
        KernelAdd(1,GB)
        @test LA + NullLeaf() == LA
        @test GA + LB == KernelAdd(LA,LB)
        @test LA * NullLeaf() == NullLeaf()
        @test LA * GB == KernelMul(LA,LB)
        @test evaluate_expression(LA * GB) isa AbstractKernel
        @test LA \ NullLeaf() == NullLeaf()
        @test LA \ GB == KernelLDiv(LA,LB)
        @test -GA isa KernelLeaf
        @test matrix(-GA) == -matrix(GA)
        @test_skip LB-LA == KernelAdd(LB,-LA)
        @test matrix(-LC) ==  - matrix(LC)
        matrix(LA-LC)
        @test matrix(LA-LC) == matrix(LA) - matrix(LC)
        @test ScalarLeaf(1).scaling == 1*I
    end
end

@testitem "Test KernelExpression evaluation" begin
    using StaticArrays
    for T in (ComplexF32,Float64)
        bs = 2
        N = 128
        ax = LinRange(-1/2,1/2,N)
        foo(x,y) = T <: Complex ? T.(1im .* [x x+y; x-y y]) : T.([x x+y; x-y y])
        foo(x) = T <: Complex ? T.(1im * [x 2x; 0 x]) : T.([x 2x; 0 x])
        A = randn(T,bs*N,bs*N)
        GA = RetardedKernel(ax,A,bs,NONCompression())
        GB = RetardedKernel(ax,foo, compression = NONCompression());
        LA = KernelLeaf(GA)
        LB = KernelLeaf(GB)
        @test evaluate_expression(LA * LB) == mul(GA,GB)
        @test evaluate_expression(LA * LB) isa RetardedKernel
        @test evaluate_expression(LA + LB) == add(GA,GB)
        @test evaluate_expression(LA + LB) isa RetardedKernel
        @test evaluate_expression(LA \ LB) == ldiv(GA,GB)
        @test evaluate_expression(LA \ LB) isa RetardedKernel
        @test evaluate_expression(-LB) isa RetardedKernel

    end
end

@testitem "test KernelExpression manipulation" begin
    using StaticArrays
    using LinearAlgebra
        for T in (ComplexF32,Float64)
            bs = 2
            N = 128
            ax = LinRange(-1. / 2, 1. ,N)
            foo(x,y) = T <: Complex ? T.(1im .* [x x+y; x-y y]) : T.([x x+y; x-y y])
            foo(x) = T <: Complex ? T.(1im * [x 2x; 0 x]) : T.([x 2x; 0 x])
            A = randn(T,bs*N,bs*N)
            G = RetardedKernel(ax,A,bs,NONCompression())
            Gδ = TimeLocalKernel(ax,foo, compression = NONCompression());
            L = KernelLeaf(G)
            Lδ = KernelLeaf(Gδ)
            @test local_part(L) == NullLeaf()
            @test nonlocal_part(Lδ) == NullLeaf()
            @test local_part(G + Gδ) == Lδ
            @test nonlocal_part(Lδ + L) == L
            @test matrix(adjoint(L)) == adjoint(matrix(L))
            @test matrix(compress!(L)) == matrix(compress!(L.kernel))
            @test compress!(adjoint(Lδ)*L*Lδ) isa KernelExpression
            @test evaluate_expression(compress!(adjoint(Lδ)*L*Lδ)) isa RetardedKernel
        end
end

@testitem"solve_dyson" begin
    using LinearAlgebra
    for T in [Float32,Float64,ComplexF32,ComplexF64]
        g(x) = T(sin(9*x))
        g(x,y) = g(x-y)
        k(x) = T(-cos(9*x))
        k(x,y) = k(x-y)
        sol_ana(x) = T((18*exp(-x/2)*sin(sqrt(323)*x/2))/sqrt(323)*(x>=0 ? 1. : 0.))
        sol_ana(x,y) = sol_ana(x-y)
        t0,t1 = 0,10
        ax = LinRange(t0,t1,2^8)
        atol = 1E-5
        rtol = 1E-5
        kest = 20
        compression = HssCompression(atol = atol, rtol = rtol, kest = kest)
        G0 = RetardedKernel(ax,g, compression = compression)
        K = RetardedKernel(ax,k, compression = compression)
        G = solve_dyson(G0,K) |> evaluate_expression
        G_ana = RetardedKernel(ax,sol_ana,compression = compression)
        @test norm(matrix(G-G_ana))/norm(G_ana |> matrix) < 1E-3
    end
end