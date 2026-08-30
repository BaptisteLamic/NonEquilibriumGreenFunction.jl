using BenchmarkTools
using NonEquilibriumGreenFunction
using LinearAlgebra
using Random

const SUITE = BenchmarkGroup()

# Configuration parameters
const SMALL_N = 64
const MEDIUM_N = 256
const LARGE_N = 512
const BLOCKSIZES = [1, 4, 8]
const COMPRESSIONS = [NONCompression(), HssCompression(atol=1e-6, rtol=1e-6)]
const DT = 2.0
const SEED = 12345

"""
    setup_benchmark(N; bs=1, seed=SEED)

Create test axis and kernel functions for benchmarks.
"""
function setup_benchmark(N; bs=1, seed=SEED)
    Random.seed!(seed)
    ax = LinRange(-DT/2, DT, N)
    f_retarded(t, tp) = (t >= tp ? cos(t - tp) : zero(Float64)) * Matrix{Float64}(I, bs, bs)
    f_acausal(t, tp) = cos(t - tp) * Matrix{Float64}(I, bs, bs)
    f_dirac(t) = Matrix{Float64}(I, bs, bs)
    return ax, f_retarded, f_acausal, f_dirac
end

# =============================================================================
# 1. DISCRETIZATION BENCHMARKS
# =============================================================================
SUITE["discretization"] = BenchmarkGroup()

for N in [SMALL_N, MEDIUM_N], bs in BLOCKSIZES, cpr in COMPRESSIONS
    ax, f_ret, f_acausal, f_dirac = setup_benchmark(N; bs=bs)

    SUITE["discretization"]["retarded", "N=$N", "bs=$bs", "$(typeof(cpr).name.name)"] =
        @benchmarkable discretize_retardedkernel($ax, $f_ret, compression=$cpr) samples=10 evals=1 gcsample=true

    SUITE["discretization"]["advanced", "N=$N", "bs=$bs", "$(typeof(cpr).name.name)"] =
        @benchmarkable discretize_advancedkernel($ax, $f_ret, compression=$cpr) samples=10 evals=1 gcsample=true

    SUITE["discretization"]["acausal", "N=$N", "bs=$bs", "$(typeof(cpr).name.name)"] =
        @benchmarkable discretize_acausalkernel($ax, $f_acausal, compression=$cpr) samples=10 evals=1 gcsample=true

    SUITE["discretization"]["dirac", "N=$N", "bs=$bs", "$(typeof(cpr).name.name)"] =
        @benchmarkable discretize_dirac($ax, $f_dirac, compression=$cpr) samples=10 evals=1 gcsample=true
end

# =============================================================================
# 2. MATRIX OPERATIONS BENCHMARKS
# =============================================================================
SUITE["operations"] = BenchmarkGroup()

for N in [SMALL_N, MEDIUM_N], bs in BLOCKSIZES, cpr in COMPRESSIONS
    ax, f_ret, f_acausal, _ = setup_benchmark(N; bs=bs)
    k1 = discretize_retardedkernel(ax, f_ret, compression=cpr)
    k2 = discretize_retardedkernel(ax, f_ret, compression=cpr)
    k_ac = discretize_acausalkernel(ax, f_acausal, compression=cpr)
    d = discretize_dirac(ax, x -> Matrix{Float64}(I, bs, bs), compression=cpr)

    SUITE["operations"]["addition", "N=$N", "bs=$bs"] =
        @benchmarkable $k1 + $k2 samples=100 evals=10 gcsample=true

    SUITE["operations"]["subtraction", "N=$N", "bs=$bs"] =
        @benchmarkable $k1 - $k2 samples=100 evals=10 gcsample=true

    SUITE["operations"]["scalar_mul", "N=$N", "bs=$bs"] =
        @benchmarkable 2.0 * $k1 samples=100 evals=10 gcsample=true

    SUITE["operations"]["multiplication_RR", "N=$N", "bs=$bs"] =
        @benchmarkable $k1 * $k2 samples=5 evals=1 gcsample=true

    SUITE["operations"]["multiplication_RAc", "N=$N", "bs=$bs"] =
        @benchmarkable $k1 * $k_ac samples=5 evals=1 gcsample=true

    SUITE["operations"]["multiplication_DK", "N=$N", "bs=$bs"] =
        @benchmarkable $d * $k1 samples=5 evals=1 gcsample=true

    SUITE["operations"]["multiplication_KD", "N=$N", "bs=$bs"] =
        @benchmarkable $k1 * $d samples=5 evals=1 gcsample=true

    SUITE["operations"]["adjoint", "N=$N", "bs=$bs"] =
        @benchmarkable adjoint($k1) samples=100 evals=10 gcsample=true
end

# =============================================================================
# 3. COMPRESSION BENCHMARKS
# =============================================================================
SUITE["compression"] = BenchmarkGroup()

for N in [SMALL_N, MEDIUM_N], bs in BLOCKSIZES
    ax, f_ret, _, _ = setup_benchmark(N; bs=bs)

    SUITE["compression"]["dense", "N=$N", "bs=$bs"] =
        @benchmarkable NONCompression()($ax, $f_ret) samples=5 evals=1 gcsample=true

    for (atol, rtol) in [(1e-4, 1e-4), (1e-6, 1e-6)]
        cpr = HssCompression(atol=atol, rtol=rtol)
        SUITE["compression"]["hss_atol=$(atol)", "N=$N", "bs=$bs"] =
            @benchmarkable $cpr($ax, $f_ret) samples=5 evals=1 gcsample=true
    end
end

# =============================================================================
# 4. SOLVER BENCHMARKS
# =============================================================================
SUITE["solver"] = BenchmarkGroup()

for N in [SMALL_N], bs in [1, 4]
    ax, f_ret, _, _ = setup_benchmark(N; bs=bs)
    g = discretize_retardedkernel(ax, (t, tp) -> exp(-abs(t-tp)) * (t >= tp ? Matrix{Float64}(I, bs, bs) : zero(Matrix{Float64})),
                                   compression=HssCompression(atol=1e-8, rtol=1e-8))
    K = discretize_retardedkernel(ax, (t, tp) -> 0.5 * exp(-abs(t-tp)) * (t >= tp ? Matrix{Float64}(I, bs, bs) : zero(Matrix{Float64})),
                                   compression=HssCompression(atol=1e-8, rtol=1e-8))

    SUITE["solver"]["dyson", "N=$N", "bs=$bs"] =
        @benchmarkable solve_dyson($g, $K) samples=3 evals=1 gcsample=true
end

# =============================================================================
# 5. INDEXING BENCHMARKS
# =============================================================================
SUITE["indexing"] = BenchmarkGroup()

for N in [SMALL_N, MEDIUM_N], bs in BLOCKSIZES, cpr in COMPRESSIONS
    ax, f_ret, _, _ = setup_benchmark(N; bs=bs)
    k = discretize_retardedkernel(ax, f_ret, compression=cpr)
    dis = discretization(k)

    SUITE["indexing"]["single_block", "N=$N", "bs=$bs"] =
        @benchmarkable $dis[1, 1] samples=1000 evals=100 gcsample=true

    SUITE["indexing"]["multi_block", "N=$N", "bs=$bs"] =
        @benchmarkable $dis[1:5, 1:5] samples=100 evals=10 gcsample=true

    SUITE["indexing"]["full_matrix", "N=$N", "bs=$bs"] =
        @benchmarkable $dis[:, :] samples=5 evals=1 gcsample=true
end

# =============================================================================
# 6. UTILITY BENCHMARKS
# =============================================================================
SUITE["utils"] = BenchmarkGroup()

for bs in [1, 4, 16, 64]
    indices = 1:10000
    SUITE["utils"]["blockindex_bs=$bs"] =
        @benchmarkable blockindex.($indices, $bs) samples=100 evals=10 gcsample=true

    SUITE["utils"]["blockrange_bs=$bs"] =
        @benchmarkable [blockrange(i, $bs) for i in $indices] samples=100 evals=10 gcsample=true
end
