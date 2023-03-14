
@testitem "Test KernelExpression construction" begin
    using StaticArrays
    for T in (ComplexF32,Float64)
        bs = 2
        N = 128
        ax = LinRange(-1. / 2, 1. ,N)
        foo(x,y) = T <: Complex ? T.(1im .* [x x+y; x-y y]) : T.([x x+y; x-y y])
        foo(x) = T <: Complex ? T.(1im * [x 2x; 0 x]) : T.([x 2x; 0 x])
        A = randn(T,bs*N,bs*N)
        GA = RetardedKernel(ax,A,bs,NONCompression())
        GB = RetardedKernel(ax,foo, compression = NONCompression());
        LA = KernelLeaf(GA)
        LB = KernelLeaf(GB)
        @test arguments(KernelAdd(GA,GB)) == SA[LA, LB]
        KernelAdd(1,GB)
        @test LA + NullLeaf() == LA
        @test GA + LB == KernelAdd(LA,LB)
        @test LA * NullLeaf() == NullLeaf()
        @test LA * GB == KernelMul(LA,LB)
        @test LA \ NullLeaf() == NullLeaf()
        @test LA \ GB == KernelLDiv(LA,LB)
        @test -GA == KernelMul(ScalarLeaf(-1),LA)
        @test LB-LA == KernelAdd(LB,KernelMul(ScalarLeaf(-1),LA))
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
        @test evaluate_expression(LA + LB) == add(GA,GB)
        @test evaluate_expression(LA \ LB) == ldiv(GA,GB)
    end
end

@testitem "test KernelExpression manipulation" begin
    using StaticArrays
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
        end
    end