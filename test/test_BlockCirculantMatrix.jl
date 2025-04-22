@testset "BlockCirculantMatrix" for T = [Float64,ComplexF64]
    # "safety" factor
    c = 500

    tol = max(1E-6,c*eps(real(T)))
    foo(x) = if T <: Complex
        T.([-x 5x; x 3im*x])
    else
        T.([-x 5x; x 3x])
    end
    foo(x,y) = foo(x-y)
    ax = 0:256
    bs = size(foo(0),1)
    mA = hcat([vcat([foo(x,y) for x in ax]...) for y in ax]...)
    lm = NonEquilibriumGreenFunction.build_CirculantlinearMap(ax,foo)
    @test (lm[:,:]-mA |> norm) < tol
    
    x = randn(T, length(ax)*bs,bs)
    @test norm(lm'*x - mA'*x) <tol
    @test norm(lm'*x - mA'*x) / norm(mA'*x) <tol
    @test norm(lm*x - mA*x) <tol
    @test norm(lm*x - mA*x) / norm(mA*x) <tol
    
    bs = 2
    cc = bisection_cluster(length(ax)*bs)
    r = randcompress_adaptive(lm,cc,cc)
    recompress!(r)
    @test r - lm[:,:] |> norm < 1e-9
end