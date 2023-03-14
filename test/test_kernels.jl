@testset "$T kernels.jl" for T = [Float64,ComplexF64]
    # "safety" factor
    c = 100
    # increase tolerance for Float32 and ComplexF32
    tol = c*max(1E-6,eps(real(T)))

    bs = 2
    N = 128
    Dt = 2.
    ax = LinRange(-Dt/2,Dt,N)

    foo(x,y) = T <: Complex ? T.(1im .* [x x+y; x-y y]) : T.([x x+y; x-y y])
    foo(x) = T <: Complex ? T.(1im * [x 2x; 0 x]) : T.([x 2x; 0 x])
    
    foo_st(x) =  T <: Complex ? T.( 1im * [x 2x; 0 x]) : T.([x 2x; 0 x])
    foo_st(x,y) = foo_st(x-y)

    @testset "$Ker construction" for Ker = (RetardedKernel, AdvancedKernel, Kernel, TimeLocalKernel)
        A = randn(T,bs*N,bs*N)
        GA = Ker(ax,A,bs,NONCompression())
        @test matrix(GA)== A

        @testset "getter" begin
            @test axis(GA) == ax
            @test blocksize(GA) == bs
            I = (2:5,2:25)
            @test GA[I...] == [A[blockrange(i, bs ), blockrange(j, bs )] for i in I[1], j in I[2]] 
        end

        B = zeros(T,N*bs,N*bs)
        for it in 1:length(ax)
            t0 = Ker == RetardedKernel || Ker == Kernel ? 1 : it
            t1 = Ker == AdvancedKernel || Ker == Kernel ? length(ax) : it
            for itp in t0:t1
                B[blockrange(it,bs),blockrange(itp,bs)] .= foo(ax[it],ax[itp])
            end
        end
        GB = Ker(ax,foo, compression = NONCompression());
        @test B == matrix(GB)

        GA = Ker(ax,foo_st, compression = NONCompression(),stationary = true)
        GB = Ker(ax,foo_st, compression = NONCompression(),stationary = false)
        @test matrix(GA)- matrix(GB)|> norm < tol

        GA = Ker(ax,foo, compression = HssCompression(),stationary = false)
        GB = Ker(ax,foo, compression = NONCompression(),stationary = false)
        @test full(matrix(GA)) - matrix(GB)|> norm < tol

        
        GA = Ker(ax,foo_st, compression = HssCompression(),stationary = true)
        GB = Ker(ax,foo_st, compression = NONCompression(),stationary = true)
        @test full(matrix(GA)) - matrix(GB)|> norm < tol
    end
    @testset "convolution $KerL $KerR" for KerL in (RetardedKernel, AdvancedKernel, Kernel),
        KerR in (RetardedKernel, AdvancedKernel, Kernel)
        #utility functions
        function _trapz(p,k,q) 
            if q >= p
                return 0.
            elseif k==p || k== q
                return 1/2
            elseif k > p || k < q
                return 0.
            else 
                return 1.
            end
        end
        function _integrate(f, g, p, q, k_sup, k_inf,step)
            if k_sup <= k_inf
                return  zero(f[1,1])
            else
                return step * sum( [ _trapz(k_sup, k, k_inf) * f[p, k]*g[k, q] for k = k_inf:k_sup ] )
            end
        end
        ############################

        x0 = Dt/4.5
        sigma0 = Dt/5
        gooL(x, y) = sigma0^4*exp( -( (x-x0)^2+y^2 )/sigma0^2 ) .* foo(x, y)
        gooR(x, y) = sigma0^4*exp( -( (x+x0)^2+y^2 )/sigma0^2 ) .* foo(x, y)

        GL = KerL(ax,gooL, compression = NONCompression())
        GR = KerR(ax,gooR, compression = NONCompression())
        PROD = GL*GR
        _diff = zeros(T,N,N)
        integral = zeros(T,N,N)
        for p in 1:length(ax), q in 1:length(ax)
            k_inf = 1
            k_sup = length(ax)
            #compute the integral boundary
            KerR == RetardedKernel && ( k_inf = max(k_inf, q) )
            KerR == AdvancedKernel && ( k_sup = min(k_sup, q) )
            KerL == RetardedKernel && ( k_sup = min(k_sup, p) )
            KerL == AdvancedKernel && ( k_inf = max(k_inf, p) )
            _int = _integrate(GL, GR, p, q, k_sup, k_inf,step(axis(GR)))
            _diff[p,q] = norm( PROD[p,q] - _int ) 
            integral[p,q] = norm( _int ) 
        end
        @test norm(_diff)/norm(integral) < tol
    end
    @testset "convolution TimeLocalKernel $KerR" for KerR in (RetardedKernel, AdvancedKernel, Kernel)
        x0 = Dt/4.5
        sigma0 = Dt/5
        gooL(x, y) = sigma0^4*exp( -( (x-x0)^2+y^2 )/sigma0^2 ) .* foo(x, y)
        gooL(x) = gooL(x,x)
        gooR(x, y) = sigma0^4*exp( -( (x+x0)^2+y^2 )/sigma0^2 ) .* foo(x, y)

        GL = TimeLocalKernel(ax,gooL, compression = NONCompression())
        GR = KerR(ax,gooR, compression = NONCompression())
        PR = GL*GR
        @test typeof(PR) == typeof(GR)
        @test norm( matrix(GL)*matrix(GR) - matrix(PR) ) < tol 
    end
    @testset "convolution $KerL TimeLocalKernel" for KerL in (RetardedKernel, AdvancedKernel, Kernel)
        x0 = Dt/4.5
        sigma0 = Dt/5
        gooL(x, y) = sigma0^4*exp( -( (x-x0)^2+y^2 )/sigma0^2 ) .* foo(x, y)
        gooL(x) = gooL(x,x)
        gooR(x, y) = sigma0^4*exp( -( (x+x0)^2+y^2 )/sigma0^2 ) .* foo(x, y)
        gooR(x) = gooR(x,x)

        GL = KerL(ax,gooL, compression = NONCompression())
        GR = TimeLocalKernel(ax,gooR, compression = NONCompression())
        PR = GL*GR
        @test typeof(PR) == typeof(GL)
        @test norm( matrix(GL)*matrix(GR) - matrix(PR) ) < tol 
    end
end