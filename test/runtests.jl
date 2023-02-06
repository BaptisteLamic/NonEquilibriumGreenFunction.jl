using NonEquilibriumGreenFunction
using Test
using LinearAlgebra
using SparseArrays
using HssMatrices
using StaticArrays
using LsqFit
using TestItemRunner

#run in files test
@run_package_tests

include("test_physics.jl")
include("test_BlockCirculantMatrix.jl")
include("test_kernels.jl")
include("test_kernel_solve.jl")



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
    @test C-B |> norm ≈ 0

end
