# Benchmark runner using PkgBenchmark.
#
# Usage (run from the repo root, with the test environment active):
#
#   julia --project=test benchmark/run.jl
#       Run the full benchmark suite and compare against this machine's
#       baseline (benchmark/baselines/<cpu>.json).
#
#   julia --project=test benchmark/run.jl --update-baseline
#       Run the suite and (over)write this machine's baseline with the
#       results. Use after the first run on a new machine, or when you
#       intentionally accept new performance numbers.
#
# Committed baselines are per-hardware *advisory references*, not hard gates:
# machines with the same CPU model can still differ by core count, RAM,
# frequency scaling, and thermals.

using PkgBenchmark
using BenchmarkTools
using Printf

const BENCH_DIR = @__DIR__
const BASELINES_DIR = joinpath(BENCH_DIR, "baselines")
const RESULTS_DIR = joinpath(BENCH_DIR, "results")

"""
    cpu_slug()

Return a filesystem-safe slug identifying this machine's CPU model.
Falls back to "unknown-cpu" when the model cannot be determined.
"""
function cpu_slug()
    info = Sys.cpu_info()
    model = isempty(info) ? "" : String(info[1].model)
    model = lowercase(model)
    # Keep [a-z0-9], replace everything else with '-'
    slug = map(c -> ('a' <= c <= 'z' || '0' <= c <= '9') ? c : '-', model)
    # Collapse runs of '-'
    slug = replace(slug, r"-+" => "-")
    # Strip leading/trailing '-'
    slug = strip(slug, '-')
    # Reject empty / traversal
    if isempty(slug) || slug == "." || slug == ".." || occursin("..", slug)
        return "unknown-cpu"
    end
    return slug
end

function parse_args(args)
    update_baseline = "--update-baseline" in args
    return update_baseline
end

function main()
    update_baseline = parse_args(ARGS)

    hw = cpu_slug()
    baseline_file = joinpath(BASELINES_DIR, "$hw.json")

    @printf("Hardware id: %s\n", hw)
    @printf("Baseline file: %s\n", baseline_file)
    println()

    mkpath(RESULTS_DIR)
    new_file = joinpath(RESULTS_DIR, "new.json")

    println("Running benchmark suite...")
    results = benchmarkpkg("NonEquilibriumGreenFunction"; verbose=true)
    writeresults(new_file, results)
    println()
    println("Results written to: $new_file")

    if update_baseline
        mkpath(BASELINES_DIR)
        cp(new_file, baseline_file; force=true)
        println("Baseline updated: $baseline_file")
        println("Commit this file to share the baseline for this hardware.")
        return
    end

    if !isfile(baseline_file)
        println()
        @warn "No baseline found for hardware '$hw' at $baseline_file."
        println("Run with --update-baseline to create one:")
        println("  julia --project=test benchmark/run.jl --update-baseline")
        return
    end

    println()
    println("Comparing against baseline...")
    baseline = readresults(baseline_file)
    judgement = judge(results, baseline, median)

    show(stdout, MIME("text/plain"), judgement)
    println()

    judge_md = joinpath(RESULTS_DIR, "judge.md")
    export_markdown(judge_md, judgement; export_invariants=true)
    println("Comparison report written to: $judge_md")
end

main()
