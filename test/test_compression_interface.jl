@testitem "compression interface: NONCompression" begin
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: test_compression_interface
    test_compression_interface(NONCompression(); N=32, bs=2, atol=1e-10)
end

@testitem "compression interface: HssCompression" begin
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: test_compression_interface
    test_compression_interface(HssCompression(atol=1e-10, rtol=1e-10, leafsize=8); N=32, bs=2, atol=1e-6)
end

@testitem "compression interface: custom dense family" begin
    using NonEquilibriumGreenFunction
    # A user-defined compression wrapping the dense family. Verifies that
    # only the documented interface is required to define a new
    # representation.
    using NonEquilibriumGreenFunction: AbstractCompression, test_compression_interface
    struct DiagWrapCompression <: AbstractCompression end
    function (c::DiagWrapCompression)(axis, f; stationary=false)
        return NONCompression()(axis, f; stationary=stationary)
    end
    (c::DiagWrapCompression)(m::AbstractMatrix) = Matrix(m)
    test_compression_interface(DiagWrapCompression(); N=16, bs=1, atol=1e-10)
end

@testitem "compression interface: JLArrayCompression" begin
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: test_compression_interface
    using JLArrays
    cpr = Base.get_extension(NonEquilibriumGreenFunction, :JLArraysExt).JLArrayCompression()
    test_compression_interface(cpr; N=16, bs=2, atol=1e-8)
end
