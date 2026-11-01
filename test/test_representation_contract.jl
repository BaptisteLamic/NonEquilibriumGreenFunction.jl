@testitem "representation contract audit: DenseRepresentation" begin
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: test_representation_contract
    test_representation_contract(DenseRepresentation(); N=32, bs=2, atol=1e-10)
end

@testitem "representation contract audit: HssRepresentation" begin
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: test_representation_contract
    test_representation_contract(HssRepresentation(atol=1e-10, rtol=1e-10, leafsize=8); N=32, bs=2, atol=1e-6)
end

@testitem "representation contract audit: custom dense family" begin
    using NonEquilibriumGreenFunction
    # A user-defined representation wrapping the dense family. Verifies that
    # only the documented interface is required to define a new
    # representation.
    using NonEquilibriumGreenFunction: AbstractRepresentation, test_representation_contract
    struct DiagWrapRepresentation <: AbstractRepresentation end
    NonEquilibriumGreenFunction.backend(::DiagWrapRepresentation, ::Type{T}=ComplexF64) where {T<:Number} =
        NonEquilibriumGreenFunction.MatrixFamilyBackend{T}()
    test_representation_contract(DiagWrapRepresentation(); N=16, bs=1, atol=1e-10)
end

@testitem "representation contract audit: JLArrayRepresentation (dense GPU-style family)" begin
    using NonEquilibriumGreenFunction
    using NonEquilibriumGreenFunction: test_representation_contract
    using JLArrays
    rep = Base.get_extension(NonEquilibriumGreenFunction, :JLArraysExt).JLArrayRepresentation()
    test_representation_contract(rep; N=16, bs=2, atol=1e-8)
end

@testitem "JLArray dense GPU-style family works through the single contract" begin
    # A dense GPU-style representation (compresses nothing) is accepted
    # anywhere a representation is: build a small DAG over JLArrays and
    # evaluate it through evaluate(backend(rep), ex).
    import NonEquilibriumGreenFunction as NEGF
    using JLArrays
    using JLArrays: JLArray
    using LinearAlgebra
    ext = Base.get_extension(NEGF, :JLArraysExt)
    rep = ext.JLArrayRepresentation()
    b = NEGF.backend(rep, ComplexF64)
    n = 8
    A = JLArray(randn(n, n) .+ im .* randn(n, n))
    B = JLArray(randn(n, n) .+ im .* randn(n, n))
    C = JLArray(randn(n, n) .+ im .* randn(n, n))
    ex = NEGF.op_add(NEGF.op_mul(NEGF.CompressedLeaf(A), NEGF.CompressedLeaf(B)), NEGF.CompressedLeaf(C))
    r = NEGF.evaluate(b, ex)
    @test r isa JLArray
    @test Array(r) ≈ Array(A) * Array(B) + Array(C) rtol=1e-12
end
