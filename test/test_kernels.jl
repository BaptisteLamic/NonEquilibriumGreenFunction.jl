@testset "kernels.jl" for T = [Float32,Float64,ComplexF64]
    
    bs = 2
    N = 128
    Dt = 0.5
    ax = LinRange(-Dt,Dt,N)
    foo(x,y) = T.([x x+y; x-y y])
    foo(x) = T.([x 2x; 0 x])

    @testset "$Ker construction" for Ker = (RetardedKernel, AdvancedKernel, TimeLocalKernel)
        A = randn(T,bs*N,bs*N)
        GA = Ker(ax,A,bs,NONCompression())
        @test GA.matrix == A

        @testset "getter" begin
            @test axis(GA) == ax
            @test blocksize(GA) == bs
            I = (2:5,2:25)
            @test GA[I...] == [A[blockrange(i, bs ), blockrange(j, bs )] for i in I[1], j in I[2]] 
        end

        B = zeros(T,N*bs,N*bs)
        for it in 1:length(ax)
            t0 = Ker == RetardedKernel ? 1 : it
            t1 = Ker == AdvancedKernel ? length(ax) : it
            for itp in t0:t1
                B[blockrange(it,bs),blockrange(itp,bs)] .= foo(ax[it],ax[itp])
            end
        end
        GB = Ker(ax,foo, compression = NONCompression());
        @test B == GB.matrix 
    end

    @testset "SumKernel construction" begin
        AL = randn(T,bs*N,bs*N)
        AR = randn(T,bs*N,bs*N)
        GL = RetardedKernel(ax,AL,bs,NONCompression())
        GR = AdvancedKernel(ax,AR,bs,NONCompression())
        GS = SumKernel(GL,GR)

        @test GS.kernelL == GL
        @test GS.kernelR == GR

        @testset "getter" begin
            @test axis(GS) == ax
            @test blocksize(GS) == bs
            I = (2:5,2:25)
            @test GS[I...] == [(GL.matrix+GR.matrix)[blockrange(i, bs ), blockrange(j, bs )] for i in I[1], j in I[2]]   
        end
    end

    @testset "Kernel arithmetic $Ker $Ker2" for Ker in (RetardedKernel, AdvancedKernel, TimeLocalKernel ,SumKernel),
         Ker2 in (RetardedKernel, AdvancedKernel, TimeLocalKernel ,SumKernel)
        G = if Ker == SumKernel 
                AL = randn(T,bs*N,bs*N)
                AR = randn(T,bs*N,bs*N)
                GL = RetardedKernel(ax,AL,bs,NONCompression())
                GR = AdvancedKernel(ax,AR,bs,NONCompression())
                SumKernel(GL,GR)
            else
                A = randn(T,bs*N,bs*N)
                Ker(ax,A,bs,NONCompression())
            end
        G2 = if Ker2 == SumKernel 
                AL = randn(T,bs*N,bs*N)
                AR = randn(T,bs*N,bs*N)
                GL = RetardedKernel(ax,AL,bs,NONCompression())
                GR = AdvancedKernel(ax,AR,bs,NONCompression())
                SumKernel(GL,GR)
            else
                A = randn(T,bs*N,bs*N)
                Ker2(ax,A,bs,NONCompression())
            end
        α = T(2/3)
        for op in (*,\)  
            @test all( op(α, G)[:, :] .≈   op.(α, G[:, :]) )
        end
        for op in (-,+) 
            @test all( op(G)[:, :] .≈   op.(G[:, :]) )
            @test all( op(G,G2)[:, :] .≈   op.(G[:, :],G2[:, :]) )
        end
    end

end