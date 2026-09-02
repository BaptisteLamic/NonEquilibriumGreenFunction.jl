@testitem "JET Static Analysis" begin
    using JET
    using NonEquilibriumGreenFunction  # Ensure module is loaded
    # `JET_AVAILABLE` only exists in JET 0.12+; on older versions JET is
    # always usable once it has been imported successfully.
    # Yet, it may just crash on incompatible julia version 
    jet_available = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : true
    if jet_available

        # Use report_package for whole-package analysis
        # This analyzes all method definitions in the package
        result = report_package(NonEquilibriumGreenFunction)

        # We only test for toplevel errors as inference errors may come from external packages
        # (the `result.res.toplevel_error_reports`/`inference_error_reports` API is stable
        # across JET 0.9–0.12)
        @test isempty(result.res.toplevel_error_reports)

        # Optional: Verbose output on CI for debugging
        if get(ENV, "GITHUB_ACTIONS", "false") == "true"
            total_errors = length(result.res.toplevel_error_reports) + length(result.res.inference_error_reports)
            if total_errors > 0
                @warn "JET Analysis found $total_errors potential issues"
                toplevel_errors = length(result.res.toplevel_error_reports)
                inference_errors = length(result.res.inference_error_reports)
                @warn "Top-level errors: $toplevel_errors, Inference errors: $inference_errors"
            else
                @info "JET Analysis: No issues detected"
            end
        end
    else
        @warn "JET is not available; skipping static analysis tests"
        return
    end
end