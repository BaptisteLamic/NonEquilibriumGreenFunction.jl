#!/usr/bin/env julia
# Run benchmarks for NonEquilibriumGreenFunction

using Dates

# Add the package to the path
import Pkg
Pkg.activate(joinpath(@__DIR__))

# Load the benchmark module
using NonEquilibriumGreenFunctionBenchmarks

println("="^80)
println("NonEquilibriumGreenFunction - Benchmark Suite")
println("="^80)
println("Starting at: ", now())
println()

# Parse command line arguments
const groups = if length(ARGS) > 0
    ARGS
else
    nothing
end

if !isnothing(groups)
    println("Running specific groups: ", join(groups, ", "))
else
    println("Running all benchmark groups")
end
println()

try
    # Run benchmarks
    results = run_benchmarks(groups=groups, verbose=true)

    # Save results
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "benchmark_results_$timestamp.json"
    save_benchmarks(results, filename)

    println()
    println("="^80)
    println("Benchmarks completed successfully!")
    println("Results saved to: $filename")
    println("Completed at: ", now())
    println("="^80)

catch e
    println()
    println("="^80)
    println("ERROR: Benchmarks failed!")
    println("Exception: ", e)
    println("="^80)
    rethrow(e)
end