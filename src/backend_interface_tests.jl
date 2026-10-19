@testitem "Thread budget" begin
    import NonEquilibriumGreenFunction as NEGF
    @test NEGF.current_thread_budget() == Threads.nthreads()
    @test NEGF.with_thread_budget(3) do
        NEGF.current_thread_budget()
    end == 3
    t = Threads.@spawn NEGF.with_thread_budget(5) do
        NEGF.current_thread_budget()
    end
    @test fetch(t) == 5
    @test NEGF.current_thread_budget() == Threads.nthreads()
    # nested budgets restore correctly
    @test NEGF.with_thread_budget(2) do
        NEGF.with_thread_budget(4) do
            NEGF.current_thread_budget()
        end == 4 && NEGF.current_thread_budget() == 2
    end
end

@testitem "DAG interpreter with a minimal backend" begin
    import NonEquilibriumGreenFunction as NEGF
    import NonEquilibriumGreenFunction: AbstractCompressionBackend, AbstractCompressedOperator,
        CompressedLeaf, DirectSolve
    using LinearAlgebra

    struct MockOp <: AbstractCompressedOperator
        m::Matrix{ComplexF64}
    end
    struct MockBackend <: AbstractCompressionBackend end

    NEGF.working_eltype(::MockBackend) = ComplexF64
    NEGF.op_add(a::MockOp, b::MockOp) = MockOp(a.m + b.m)
    NEGF.op_sub(a::MockOp, b::MockOp) = MockOp(a.m - b.m)
    NEGF.op_scale(alpha, a::MockOp) = MockOp(alpha * a.m)
    NEGF.op_mul(a::MockOp, b::MockOp) = MockOp(a.m * b.m)
    NEGF.op_adjoint(a::MockOp) = MockOp(adjoint(a.m))
    NEGF.op_norm(a::MockOp) = norm(a.m)
    NEGF.apply(a::MockOp, x) = a.m * x
    NEGF.structure(::MockOp) = nothing
    NEGF.solve(::MockBackend, left::MockOp, right::MockOp; strategy=DirectSolve()) = MockOp(left.m \ right.m)

    b = MockBackend()
    A = MockOp(randn(4, 4) + im * randn(4, 4))
    B = MockOp(randn(4, 4) + im * randn(4, 4))
    C = MockOp(randn(4, 4) + im * randn(4, 4))

    ex = NEGF.op_mul(CompressedLeaf(A), CompressedLeaf(B), NEGF.op_adjoint(CompressedLeaf(C)))
    @test ex isa NEGF.MulNode

    r = NEGF.evaluate(b, ex)
    @test r.m ≈ A.m * B.m * adjoint(C.m)

    ex2 = NEGF.op_sub(NEGF.op_add(CompressedLeaf(A), NEGF.op_scale(2.0, CompressedLeaf(B))), CompressedLeaf(A))
    @test NEGF.evaluate(b, ex2).m ≈ 2 .* B.m
    ex3 = NEGF.SolveNode(b, CompressedLeaf(A), CompressedLeaf(B), DirectSolve())
    @test NEGF.evaluate(b, ex3).m ≈ A.m \ B.m

    @test NEGF.op_norm(NEGF.evaluate(b, ex2)) ≈ norm(2 .* B.m)
    @test NEGF.apply(A, ones(4)) ≈ A.m * ones(4)
    @test NEGF.structure(A) === nothing
end
