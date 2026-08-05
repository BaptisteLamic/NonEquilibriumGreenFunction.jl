#!/usr/bin/env julia
# Test script to verify benchmarks work

# First, activate the project
using Pkg
Pkg.activate("..")

println("Testing benchmark infrastructure...")

# Test that we can load the main package
try
    using NonEquilibriumGreenFunction
    println("✓ Main package loads successfully")
catch e
    println("✗ Main package failed to load: ", e)
    exit(1)
end

# Test that we can load BenchmarkTools
try
    using BenchmarkTools
    println("✓ BenchmarkTools loads successfully")
catch e
    println("✗ BenchmarkTools failed to load: ", e)
    exit(1)
end

# Test that we can load JSON3
try
    using JSON3
    println("✓ JSON3 loads successfully")
catch e
    println("✗ JSON3 failed to load: ", e)
    exit(1)
end

# Test that we can load the benchmark module
try
    include("benchmarks.jl")
    println("✓ Benchmark module loads successfully")
catch e
    println("✗ Benchmark module failed to load: ", e)
    @error "Benchmark test failed" exception=(e, catch_backtrace())
    exit(1)
end

# Import the functions we need
using .NonEquilibriumGreenFunctionBenchmarks: run_benchmarks, compare_benchmarks, save_benchmarks, load_benchmarks

# Test a single benchmark group - just utils for speed
try
    println("\nTesting utils benchmarks...")
    results = run_benchmarks(groups=["utils"], verbose=false)
    println("✓ Utils benchmarks completed")
    println("  Groups: ", keys(results))
    println("  Number of benchmarks: ", sum(length, values(results)))
catch e
    println("✗ Utils benchmarks failed: ", e)
    @error "Benchmark test failed" exception=(e, catch_backtrace())
    exit(1)
end

# Test comparison function
try
    println("\nTesting comparison function...")
    baseline = run_benchmarks(groups=["utils"], verbose=false)
    new = run_benchmarks(groups=["utils"], verbose=false)
    comparison = compare_benchmarks(baseline, new, verbose=true)
    println("✓ Comparison function works")
catch e
    println("✗ Comparison function failed: ", e)
    @error "Comparison test failed" exception=(e, catch_backtrace())
    exit(1)
end

# Test save/load
try
    println("\nTesting save/load functions...")
    test_results = run_benchmarks(groups=["utils"], verbose=false)
    save_benchmarks(test_results, "test_save.json")
    loaded = load_benchmarks("test_save.json")
    println("✓ Save/load functions work")
    rm("test_save.json")
catch e
    println("✗ Save/load functions failed: ", e)
    @error "Save/load test failed" exception=(e, catch_backtrace())
    exit(1)
end

println("\n" * 2)
println("All benchmark infrastructure tests passed! ✓")
println("\nTo run full benchmarks: julia run_benchmarks.jl")
println("To run specific groups: julia run_benchmarks.jl discretization operations")