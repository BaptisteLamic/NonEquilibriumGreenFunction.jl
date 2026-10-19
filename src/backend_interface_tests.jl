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

@testitem "HSS backend: primitives and balanced-chain evaluation" begin
    import NonEquilibriumGreenFunction as NEGF
    import NonEquilibriumGreenFunction: HssMatricesBackend, CompressedLeaf, DirectSolve
    using NonEquilibriumGreenFunction: HssMatricesBackend
    using HssMatrices
    using LinearAlgebra

    b = HssMatricesBackend(atol=1e-12, rtol=1e-12, kest=20, leafsize=16)
    @test NEGF.working_eltype(b) == ComplexF64

    n = 64
    A = hss(Matrix(randn(n, n) + im * randn(n, n)), atol=1e-12, rtol=1e-12, leafsize=16)
    B = hss(Matrix(randn(n, n) + im * randn(n, n)), atol=1e-12, rtol=1e-12, leafsize=16)
    C = hss(Matrix(randn(n, n) + im * randn(n, n)), atol=1e-12, rtol=1e-12, leafsize=16)
    D = hss(Matrix(randn(n, n) + im * randn(n, n)), atol=1e-12, rtol=1e-12, leafsize=16)
    ref(A) = Matrix(A)

    # primitives stay in-family
    S = NEGF.op_add(A, B)
    @test S isa HssMatrix
    @test ref(S) ≈ ref(A) + ref(B)
    Df = NEGF.op_sub(A, B)
    @test ref(Df) ≈ ref(A) - ref(B)
    Sc = NEGF.op_scale(2.0 + 0im, A)
    @test ref(Sc) ≈ 2 .* ref(A)
    Ad = NEGF.op_adjoint(A)
    @test ref(Ad) ≈ ref(A)'
    @test NEGF.op_norm(A) ≈ norm(ref(A)) rtol=1e-6
    x = randn(n, 3) .+ im .* randn(n, 3)
    @test Matrix(NEGF.apply(A, x)) ≈ ref(A) * x

    # structure / empty_like
    @test NEGF.structure(A) === A
    Z = NEGF.empty_like(A)
    @test Z isa HssMatrix
    @test norm(Matrix(Z)) == 0

    # balanced-chain evaluation of a variadic op_mul
    ex = NEGF.op_mul(CompressedLeaf(A), CompressedLeaf(B), CompressedLeaf(C), CompressedLeaf(D))
    r = NEGF.evaluate(b, ex)
    @test r isa HssMatrix
    @test ref(r) ≈ ref(A) * ref(B) * ref(C) * ref(D) atol=1e-8

    # balanced order must match the left fold (default interpreter) within tolerance
    r_left = foldl(NEGF.op_mul, [A, B, C, D])
    @test ref(r) ≈ ref(r_left) atol=1e-8

    # chain of length 3: (A*B)*C balanced as (A*B)*C or A*(B*C) - result equal either way
    ex3 = NEGF.op_mul(CompressedLeaf(A), CompressedLeaf(B), CompressedLeaf(C))
    @test ref(NEGF.evaluate(b, ex3)) ≈ ref(A) * ref(B) * ref(C) atol=1e-8

    # mixed graph: add/scale/adjoint + solve
    ex2 = NEGF.op_sub(NEGF.op_add(CompressedLeaf(A), NEGF.op_scale(2.0 + 0im, CompressedLeaf(B))), CompressedLeaf(A))
    @test ref(NEGF.evaluate(b, ex2)) ≈ 2 .* ref(B) atol=1e-8
    # solve: use a compressible operator (smooth kernel) - random dense
    # matrices do not compress and HSS solves of them are meaningless
    M = [exp(-abs(i - j) / 8.0) * (1.0 + 1im) for i in 1:n, j in 1:n]
    As = hss(Matrix(M), atol=1e-12, rtol=1e-12, leafsize=16)
    R = Matrix(randn(n, n) .+ im .* randn(n, n))
    Bs = hss(copy(R), atol=1e-12, rtol=1e-12, leafsize=16)
    exs = NEGF.SolveNode(b, CompressedLeaf(As), CompressedLeaf(Bs), DirectSolve())
    X = ref(NEGF.evaluate(b, exs))
    Xref = ref(As) \ R
    @test norm(X - Xref) / norm(Xref) < 1e-6
end

@testitem "DAG evaluated at most once (materialization cache)" begin
    import NonEquilibriumGreenFunction as NEGF
    using NonEquilibriumGreenFunction
    using LinearAlgebra

    # counting evaluate: wraps a real compression, counts backend calls
    struct CountingCompression <: AbstractCompression
        inner::NONCompression
        count::Base.RefValue{Int}
    end
    # bypass: evaluation goes through evaluate(c, ex)
    function NEGF.evaluate(c::CountingCompression, ex::NEGF.AbstractOpNode)
        c.count[] += 1
        return NEGF.evaluate(c.inner, ex)
    end
    # construction interface: delegate everything else
    (c::CountingCompression)(args...; kw...) = c.inner(args...; kw...)
    (c::CountingCompression)(blocks::AbstractArray{T,3}) where {T} = c.inner(blocks)

    cnt = Base.RefValue(0)
    cpr = CountingCompression(NONCompression(), cnt)
    ax = 0:0.1:1.0
    A = RetardedKernel(ax, Stationary(τ -> ComplexF64(exp(-τ))), compression=cpr)
    B = RetardedKernel(ax, Stationary(τ -> ComplexF64(cos(τ))), compression=cpr)
    P = A * B
    T = P + A
    # force the same tree through several forcing points
    m1 = matrix(T)
    m2 = matrix(T)
    st = same_time(T)
    kt = keldysh_trace(T)
    n1 = norm(T)
    @test m1 === m2
    # every node evaluated exactly once (P's product node + T's sum node)
    @test cnt[] == 2
    # forcing the child separately must not re-evaluate it
    mP = matrix(P)
    @test cnt[] == 2
    # a NEW tree sharing the child: only the new sum node evaluates
    T2 = P + B
    mT2 = matrix(T2)
    @test cnt[] == 3
    # values still correct
    ref = matrix(A * B) + matrix(A)
    @test norm(m1 - ref) / norm(ref) < 1e-12
end
