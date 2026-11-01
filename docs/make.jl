using NonEquilibriumGreenFunction
using Documenter
using Literate
using LinearAlgebra

const EXAMPLES = ["mqdm", "sqds"]

function generate_examples()
    lit_dir = joinpath(@__DIR__, "lit")
    out_dir = joinpath(@__DIR__, "src", "generated")
    mkpath(out_dir)
    for example in EXAMPLES
        Literate.markdown(joinpath(lit_dir, "$example.jl"), out_dir;
            flavor=Literate.DocumenterFlavor(), execute=true,
            documenter=false, credit=false)
    end
end

generate_examples()

makedocs(;
    modules=[NonEquilibriumGreenFunction],
    sitename="NonEquilibriumGreenFunction",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true"),
    pages=[
        "Home" => "index.md",
        "User guide" => "userguide.md",
        "Examples" => [
            "Metal - QD - Metal junction" => "generated/mqdm.md",
            "Superconductor - QD - Superconductor junction" => "generated/sqds.md",
        ],
        "API" => "api.md",
        "Backend" => [
            "Backend contract" => "backend_contract.md",
            "Backend interface design" => "backend_interface_design.md",
        ],
        "Custom representations" => "custom_compression.md",
        "Internals" => "internals.md",
    ],
)

deploydocs(;
    repo="github.com/BaptisteLamic/NonEquilibriumGreenFunction.jl",
    devbranch="main",
    push_preview=true,
)
