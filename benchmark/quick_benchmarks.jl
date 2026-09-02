# Fast, self-contained benchmark smoke test.
#
# This file is intentionally cheap to include: it builds a tiny benchmark
# suite (samples=1, evals=1) covering one entry each of the core operations
# (discretization, kernel arithmetic, utility indexing) and runs it. It is
# used by the test suite to validate that the benchmark infrastructure
# executes without actually measuring performance.
#
# The full benchmark suite lives in benchmarks.jl and is run via PkgBenchmark
# (see run.jl). This file does NOT build the heavy SUITE.

using BenchmarkTools
using NonEquilibriumGreenFunction
using LinearAlgebra
using Random

const QUICK_N = 32
const QUICK_BS = 2
const QUICK_SEED = 12345
const QUICK_DT = 2.0

function quick_setup(; bs=QUICK_BS, seed=QUICK_SEED)
    Random.seed!(seed)
    ax = LinRange(-QUICK_DT/2, QUICK_DT, QUICK_N)
    f_retarded(t, tp) = (t >= tp ? cos(t - tp) : zero(Float64)) * Matrix{Float64}(I, bs, bs)
    return ax, f_retarded
end

"""
    run_quick_benchmarks(; verbose=false)

Run a minimal benchmark suite to validate the benchmark infrastructure.
Returns `true` if all quick benchmarks complete without error.
"""
function run_quick_benchmarks(; verbose=false)
    suite = BenchmarkGroup()

    # Discretization
    ax, f_ret = quick_setup()
    suite["discretization_quick"] = @benchmarkable discretize_retardedkernel($ax, $f_ret, compression=NONCompression()) samples=1 evals=1

    # Operations
    k1 = discretize_retardedkernel(ax, f_ret, compression=NONCompression())
    k2 = discretize_retardedkernel(ax, f_ret, compression=NONCompression())
    suite["addition_quick"] = @benchmarkable $k1 + $k2 samples=1 evals=1
    suite["multiplication_quick"] = @benchmarkable $k1 * $k2 samples=1 evals=1

    # Utility
    indices = 1:100
    suite["blockindex_quick"] = @benchmarkable blockindex.($indices, $QUICK_BS) samples=1 evals=1

    if verbose
        println("Running quick benchmark validation...")
    end

    run(suite, verbose=verbose)

    if verbose
        println("Quick benchmarks completed successfully.")
    end

    return true
end
