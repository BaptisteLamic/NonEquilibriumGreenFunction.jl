using Moshi.Match: @match
using Moshi.Data: @data
using Moshi.Derive: @derive
using StaticArrays


struct LowRankBlock{M}
    u :: M
    v :: M
end

#=
@data StaticHodlr{N,K,T} begin
    struct StaticLeaf{N,T}
        matrix::SMatrix{N,N,T}
    end
    struct StaticNode{N,K,T}
        A :: StaticHodlr{N,K,T}
        B :: StaticHodlr{N,K,T}
        upper_offdiag :: StaticLowRankBlock{N,K,T}
        lower_offdiag :: StaticLowRankBlock{N,K,T}
    end
end 
@derive StaticHodlr[Hash, Eq, Show]
=#

@testitem "Test static HODLR" begin
    using LinearAlgebra
    using ACAFact
    using StaticArrays
    dom = range(0.0, 1.0, length=100)
    m = @SMatrix [1 2; 3 4]
    K = KernelFunction((x,y)->m .* exp(-abs2(x-y)), dom)
    @show typeof(K)
    (U, V) = aca(K, 1e-12, maxrank=100)
    NonEquilibriumGreenFunction.LowRankBlock(U, V)
    atol = 1E-5
    rtol = 1E-5
end