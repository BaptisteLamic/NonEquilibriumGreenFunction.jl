@testitem "JET Static Analysis" begin
    using JET
    using NonEquilibriumGreenFunction  # Ensure module is loaded
    
    # Use report_package for whole-package analysis
    # This analyzes all method definitions in the package
    result = report_package(NonEquilibriumGreenFunction)
    
    # Check if any issues were found (JET 0.12.0 API)
    # We only test for toplevel errors as inference errors may come from external packages
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
end