@testset "$T kernel_solve.jl" for T = [Float32,Float64,ComplexF64]
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
    
    @testset "$Ker solve" for Ker in (TimeLocalKernel,RetardedKernel,AdvancedKernel)
        A = TimeLocalKernel(ax, gooL, compression = NONCompression())
        B = Ker(ax, gooR, compression = NONCompression())
        @test norm(A*(A\B) - B) < tol
    end
end