@testitem "Aqua Validation" begin
    using Aqua
    VERSION >= v"1.11" || return  # Aqua 0.8+ works best on Julia 1.11+
    Aqua.test_all(NonEquilibriumGreenFunction)
end
