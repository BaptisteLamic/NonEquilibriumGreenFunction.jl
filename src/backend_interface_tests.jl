@testitem "Thread budget" begin
    using NonEquilibriumGreenFunction: current_thread_budget, with_thread_budget
    @test current_thread_budget() == Threads.nthreads()
    @test with_thread_budget(3) do
        current_thread_budget()
    end == 3
    # budget propagates to spawned children
    t = Threads.@spawn with_thread_budget(5) do
        current_thread_budget()
    end
    @test fetch(t) == 5
    # budget is restored after the block
    @test current_thread_budget() == Threads.nthreads()
end

@testitem "DAG interpreter with a minimal backend" begin
    using NonEquilibriumGreenFunction: AbstractCompressionBackend, AbstractCompressedOperator,
        working_eltype, op_add, op_sub, op_scale, op_mul, op_adjoint, op_norm, apply,
        structure, empty_like, solve, DirectSolve, CompressedLeaf, evaluate
    using LinearAlgebra

    # minimal dense-backed mock: proves the *default interpreter* runs any
    # backend implementing the mandatory primitive set
    struct MockOp <: AbstractCompressedOperator
        m::Matrix{ComplexF64}
    end
    struct MockBackend <: AbstractCompressionBackend end

    working_eltype(::MockBackend) = ComplexF64
    op_add(a::MockOp, b::MockOp) = MockOp(a.m + b.m)
    op_sub(a::MockOp, b::MockOp) = MockOp(a.m - b.m)
    op_scale(alpha, a::MockOp) = MockOp(alpha * a.m)
    op_mul(a::MockOp, b::MockOp) = MockOp(a.m * b.m)
    op_adjoint(a::MockOp) = MockOp(adjoint(a.m))
    op_norm(a::MockOp) = norm(a.m)
    apply(a::MockOp, x) = a.m * x
    structure(::MockOp) = nothing
    solve(::MockBackend, left::MockOp, right::MockOp; strategy=DirectSolve()) = MockOp(left.m \ right.m)

    b = MockBackend()
    A = MockOp(randn(4, 4) + im * randn(4, 4))
    B = MockOp(randn(4, 4) + im * randn(4, 4))
    C = MockOp(randn(4, 4) + im * randn(4, 4))

    # lazy construction does no work
    ex = op_mul(CompressedLeaf(A), CompressedLeaf(B), op_adjoint(CompressedLeaf(C)))
    @test ex isa NonEquilibriumGreenFunction.MulNode

    r = evaluate(b, ex)
    @test r.m ≈ A.m * B.m * adjoint(C.m)

    # add/sub/scale/solve nodes
    ex2 = op_sub(op_add(CompressedLeaf(A), op_scale(2.0, CompressedLeaf(B))), CompressedLeaf(A))
    @test evaluate(b, ex2).m ≈ 2 .* B.m
    ex3 = NonEquilibriumGreenFunction.SolveNode(b, CompressedLeaf(A), CompressedLeaf(B), DirectSolve())
    @test evaluate(b, ex3).m ≈ A.m \ B.m

    # norm/apply are part of the mandatory set
    @test op_norm(evaluate(b, ex2)) ≈ norm(2 .* B.m)
    @test apply(A, ones(4)) ≈ A.m * ones(4)
    @test structure(A) === nothing
end
