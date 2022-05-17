@testset "$T kernels.jl" for T = [Float32,Float64,ComplexF64]
    bs = 2
    N = 128
    Dt = 0.5
    ax = LinRange(-Dt/2,Dt,N)
    function dnorm(x,sigma = 1)
        exp(-0.5*x^2/sigma^2)/sqrt(2π*sigma^2)
    end
    function ifft_dnorm(x,sigma = 1)
        exp(-x^2*sigma^2/2)/2pi
    end
    σ  = 12
    t_ax = (1-N:N-1)*Dt
    A = energy2time(x-> dnorm(x,σ),N,Dt, 1)
    σ  = 12
    dt = 0.005
    N = 2048
    t_ax = (1-N:N-1)*dt
    A = energy2time(x-> dnorm(x,σ),N,dt, 0)
    ntf =  A.data[:] 
    atf = ifft_dnorm.(t_ax, σ)
    @test norm(ntf-atf)/norm(atf) < 1E-3
end