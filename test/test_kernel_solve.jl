@testset "$T kernel_solve.jl" for T = [Float32,Float64,ComplexF64]
    @testset "$Ker solve" for Ker in (TimeLocalKernel,RetardedKernel,AdvancedKernel)
        # "safety" factor
        c = 50.
        # increase tolerance for Float32 and ComplexF32
        tol = max(1E-6,50*eps(real(T)))

        bs = 2
        N = 128
        Dt = 1//2
        ax = LinRange(-Dt,Dt,N)

        x0 = Dt/4.5
        sigma0 = Dt/5
        foo(x,y) = T.([x x+y; x-y y])
        gooL(x, y) = ( (x-x0)^2+y^2 ) .* foo(x, y)
        gooL(x) = gooL(x,x)
        gooR(x, y) = ( (x+x0)^2+y^2 ) .* foo(x, y)
        gooR(x) = gooR(x,x)
        
        A = TimeLocalKernel(ax, gooL, compression = NONCompression())
        B = Ker(ax, gooR, compression = NONCompression())
        @test norm(matrix( mul(A,ldiv(A,B)) )  - matrix(B)) < tol
    end

    @testset "$T solve_dyson" begin
        g(x) = T(sin(9*x))
        g(x,y) = g(x-y)
        k(x) = T(-cos(9*x))
        k(x,y) = k(x-y)
        sol_ana(x) = T((18*exp(-x/2)*sin(sqrt(323)*x/2))/sqrt(323)*(x>=0 ? 1. : 0.))
        sol_ana(x,y) = sol_ana(x-y)
        t0,t1 = 0,10
        ax = LinRange(t0,t1,2^10)
        atol = 1E-5
        rtol = 1E-5
        kest = 20
        compression = HssCompression(atol = atol, rtol = rtol, kest = kest)
        G0 = RetardedKernel(ax,g, compression = compression)
        K = RetardedKernel(ax,k, compression = compression)
        G = solve_dyson(G0,K)
        G_ana = RetardedKernel(ax,sol_ana,compression = compression)
        @test norm((nonlocal_part(G)-G_ana) |> matrix)/norm(G_ana |> matrix) < 1E-4
    end
end