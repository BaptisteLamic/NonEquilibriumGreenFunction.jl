using NonEquilibriumGreenFunction
using Test
using LinearAlgebra
using SparseArrays
using HssMatrices
using StaticArrays
using LsqFit


include("test_kernels.jl")
include("test_kernel_solve.jl")

@testset "NonEquilibriumGreenFunction.jl" for T = [Float32,Float64,ComplexF32,ComplexF64], Gr = (RetardedGreenFunction,AdvancedGreenFunction,GreenFunction)
    bs = 2
    N = 128
    Dt = 0.5
    ax = LinRange(-Dt,Dt,N)
    foo(x,y) = T.([x x+y; x-y y])
    foo(x) = T.([x 2x; 0 x])

    @testset "Compressed Matrix creation" begin
        c = 25. #security factor
        rtol = 1E-4
        atol = 1E-4
        kest = 20
        lm = build_linearMap(ax,foo)
        A = zeros(T,N*bs,N*bs)
        for it in 1:length(ax)
            for itp in 1:length(ax)
                A[blockrange(it,bs),blockrange(itp,bs)] .= foo(ax[it],ax[itp])
            end
        end
        @test A == lm*Matrix{T}(I,size(lm))
        x = randn(T,size(lm,1));
        @test x'*lm ≈ x'*A
        @test A*x ≈ lm*x
        compression = HssCompression(atol = atol, rtol = rtol, kest = kest)
        hssA = compression(ax,foo)
        @test norm(A-hssA)/norm(A) ≤ c*rtol || norm(A-hssA) ≤ c*atol
    end

    @testset "Green function creation" begin
        A = randn(T,bs*N,bs*N)
        GA = Gr(ax,A,bs)
        @test axis(GA) == ax
        @test regular(GA) == A
        @test blocksize(GA) == bs
        @test dirac(GA) == zeros(T,bs,bs,N)

        B = randn(T,bs,bs,N)
        δB = randn(T,bs,bs,N)
        GB = Gr(ax,δB,B,bs)
        @test axis(GB) == ax
        @test regular(GB) == B
        @test blocksize(GB) == bs
        @test dirac(GB) == δB

        A = zeros(T,N*bs,N*bs)
        for it in 1:length(ax)
            t0 = Gr == AdvancedGreenFunction ? it : 1
            t1 = Gr == RetardedGreenFunction ? it : length(ax)
            for itp in t0:t1
                A[blockrange(it,bs),blockrange(itp,bs)] .= foo(ax[it],ax[itp])
            end
        end
        GR = Gr(ax,foo,foo, compression = NONCompression());
        @test A == regular(GR)
    end
    
    @testset "Green Function arithmetic" begin
        A = randn(T,bs*N,bs*N)
        δ = randn(T,bs,bs,N)
        gA = Gr(ax,δ,A,bs)
        α = T(2/3)
        #=
        for op in (*,/)  
            @test all(dirac(op(gA,α)) .≈ op(dirac(gA),α))
        end
        =#
        for op in (*,\)  
            @test all(dirac(op(α,gA)) .≈ op(α,dirac(gA)))
        end
        for op in (-,+) 
            @test all(dirac(op(gA)) .≈ op(dirac(gA)))
            @test all(regular(op(gA)) .≈ op(regular(gA)))

            @test all(dirac(op(α*gA,gA)) .≈ op(α*dirac(gA), dirac(gA)))
            @test all(regular(op(α*gA,gA)) .≈ op(α*regular(gA), regular(gA)))
        end
    end
end



@testset "kernel product" for T = [Float32,Float64,ComplexF32,ComplexF64]
    atol = 1E-5
    rtol = 1E-5
    kest = 20
    compression = HssCompression(atol = atol, rtol = rtol, kest = kest)
    f_c(t,tp) = exp(-(t-tp))
    g_c(t,tp) = exp(-2*(t-tp))
    gf(t,tp) = exp(tp-t) - exp(2*tp-2*t)
    t0,t1 = 0,1
    xs = Float64[]
    ys = Float64[]
    for p = 3:7
        _N = 2^p
        ax = LinRange(t0,t1,2^p)
        G = RetardedGreenFunction(ax,x->T(0),g_c, compression = compression)
        F = RetardedGreenFunction(ax,x->T(0),f_c, compression = compression)
        GF = RetardedGreenFunction(ax,x->T(0),gf, compression = compression)
        n_gf = cc_prod(retarded(G), retarded(F),ax)
        err = norm((retarded(GF).-n_gf))./norm(retarded(GF))
        push!(xs,_N)
        push!(ys,err)
    end
    model(t,p) = p[1] .- p[2] .* t
    p0 = [0.,1.]
    fit = curve_fit(model,log.(xs),log.(ys),p0)
    @test abs(fit.param[2]-2)<0.2
end

@testset "utils.jl" for T = [Float32,Float64,ComplexF32,ComplexF64]
    T = Float32
    m = randn(T,12,12)
    bs = 2
    dm = NonEquilibriumGreenFunction.extract_blockdiag(m,bs)
    N = minimum(div.(size(m),bs))
    @test dm - blockdiag((sparse(m[blockrange(i,bs),blockrange(i,bs)]) for i = 1:N)...) |> norm  == 0
    A = [(i+10*j)+100*blk |> T for i = 1:2,j = 1:2,blk = 1:3];
    B = blockdiag(sparse([111 121; 112 122 ] .|> T),sparse([211 221; 212 222 ].|> T),sparse([311 321; 312 322 ].|> T))
    C = blockdiag(A)
    @test_skip C-B |> norm ≈ 0

end

@testset "dyson.jl" for T = [Float32,Float64,ComplexF64]
    g(x) = T(sin(9*x))
    g(x,y) = g(x-y)
    k(x) = T(-cos(9*x))
    k(x,y) = k(x-y)
    sol_ana(x) = (18*exp(-x/2)*sin(sqrt(323)*x/2))/sqrt(323)*(x>=0 ? 1. : 0.)
    sol_ana(x,y) = sol_ana(x-y)
    t0,t1 = 0,10
    ax = LinRange(t0,t1,2^10)
    G = RetardedGreenFunction(ax,x->T(0),g)
    K = RetardedGreenFunction(ax,x->T(0),k)
    sol = NonEquilibriumGreenFunction.cc_solve_dyson(retarded(G),retarded(K),axis(G),blocksize(G) )
    G_ana = [sol_ana(x,y) for x in ax, y in ax]
    @test norm(G_ana-sol) ≤ 1E-4*norm(G_ana)

    Sol = solve_dyson(G,K);
    @test norm(G_ana-regular(Sol)) ≤ 1E-4*norm(G_ana)

    atol = 1E-5
    rtol = 1E-5
    kest = 20
    compression = HssCompression(atol = atol, rtol = rtol, kest = kest)

    G = RetardedGreenFunction(ax,g,g, compression = compression);
    K = RetardedGreenFunction(ax,x->@SMatrix([T(0)]),k, compression = compression);
    Gf = solve_dyson(G,K);
    @test (dirac(Gf) - dirac(G))/norm(dirac(G)) |> norm ≈ 0
end

