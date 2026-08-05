#!/usr/bin/env julia
# Quick test to verify benchmark module loads and can be used

using Pkg
Pkg.activate("..")

println("Quick benchmark infrastructure test...")

# Test that we can load the main package
using NonEquilibriumGreenFunction
println("✓ Main package loads")

# Test BenchmarkTools
using BenchmarkTools
using LinearAlgebra
println("✓ BenchmarkTools loads")

# Test that we can create a simple benchmark
try
    ax = LinRange(-1.0, 1.0, 64)
    f(t, tp) = cos(t - tp) * (t >= tp ? Matrix{Float64}(I, 2, 2) : zero(Matrix{Float64}))
    
    # Test discretization
    k = @benchmarkable discretize_retardedkernel($ax, $f, compression=NONCompression()) samples=1 evals=1
    println("✓ Can create discretization benchmark")
    
    # Test that it runs
    result = run(k, verbose=false)
    println("✓ Discretization benchmark runs")
    
catch e
    println("✗ Benchmark creation failed: ", e)
    exit(1)
end

# Test utility functions
try
    indices = 1:100
    b = @benchmarkable blockindex.($indices, 4) samples=10 evals=1
    result = run(b, verbose=false)
    println("✓ Utility benchmark runs")
catch e
    println("✗ Utility benchmark failed: ", e)
    exit(1)
end

println("\nQuick test passed! ✓")
println("Full benchmark suite is ready to use.")