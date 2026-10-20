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
    # comparisons: repeated == reuses the caches (no re-evaluation)
    @test T == T                # node identity: no evaluation
    @test cnt[] == 3
    # trait/value path through caches: both sides already cached, no
    # re-evaluation; values differ (P+A vs P+B)
    @test (T == T2) == false
    @test cnt[] == 3
    # a structurally different tree materializes exactly once on first
    # comparison (T is cached), and compares unequal by value
    T3 = A + B
    @test (T == T3) == false
    @test cnt[] == 4
    # values still correct; the reference product is a NEW node, so it
    # evaluates exactly once more
    ref = matrix(A * B) + matrix(A)
    @test cnt[] == 5
    @test norm(m1 - ref) / norm(ref) < 1e-12
end

@testitem "CSE extracts repeated subgraphs from the lowered graph" begin
    import NonEquilibriumGreenFunction as NEGF
    using LinearAlgebra

    n = 16
    A = randn(n, n) .+ im .* randn(n, n)
    B = randn(n, n) .+ im .* randn(n, n)
    C = randn(n, n) .+ im .* randn(n, n)

    # shared subgraph node used at two positions
    AB = NEGF.op_mul(NEGF.CompressedLeaf(A), NEGF.CompressedLeaf(B))
    ex = NEGF.op_add(NEGF.op_mul(AB, NEGF.CompressedLeaf(C)),
                     NEGF.op_scale(2.0, AB))

    bindings, root = NEGF.cse(ex)
    @test length(bindings) == 1               # A*B extracted once
    @test root isa NEGF.AddNode

    # counting backend: the shared A*B must be computed once. Counted
    # matrices wrap the values so no global op_mul method is touched.
    struct Counted{T}
        m::T
    end
    struct CountBackend <: NEGF.AbstractCompressionBackend end
    mulcount = Ref(0)
    NEGF.op_mul(X::Counted, Y::Counted) = (mulcount[] += 1; Counted(X.m * Y.m))
    NEGF.op_add(X::Counted, Y::Counted) = Counted(X.m + Y.m)
    NEGF.op_scale(alpha, X::Counted) = Counted(alpha * X.m)
    cex = NEGF.op_add(NEGF.op_mul(NEGF.CompressedLeaf(Counted(A)), NEGF.CompressedLeaf(Counted(B))),
                      NEGF.op_mul(NEGF.CompressedLeaf(Counted(A)), NEGF.CompressedLeaf(Counted(B))))
    mulcount[] = 0
    rc = NEGF.evaluate(CountBackend(), cex)
    @test mulcount[] == 1                      # identical subgraph computed once (not 2)
    @test norm(rc.m - (A * B + A * B)) / norm(A * B + A * B) < 1e-12

    # no false extraction: distinct subgraphs stay inline
    ex2 = NEGF.op_add(NEGF.op_mul(NEGF.CompressedLeaf(A), NEGF.CompressedLeaf(B)),
                      NEGF.op_mul(NEGF.CompressedLeaf(A), NEGF.CompressedLeaf(C)))
    b2, r2 = NEGF.cse(ex2)
    @test length(b2) == 0                      # no node repeats
end

@testitem "CSE is safe under hash collisions" begin
    import NonEquilibriumGreenFunction as NEGF
    using LinearAlgebra
    # A forced hash collision between two *distinct* MulNodes: the
    # pre-filter key must never merge subgraphs that are not `===`.
    n = 8
    A = randn(n, n) .+ im .* randn(n, n)
    B = randn(n, n) .+ im .* randn(n, n)
    C = randn(n, n) .+ im .* randn(n, n)
    X = NEGF.op_mul(NEGF.CompressedLeaf(A), NEGF.CompressedLeaf(B))
    Y = NEGF.op_mul(NEGF.CompressedLeaf(A), NEGF.CompressedLeaf(C))
    ex = NEGF.op_add(X, Y)
    # sabotage only the key used for bucketing: both nodes hash equal
    NEGF._cse_key(::NEGF.MulNode) = 0x0000000000000001
    b, root = NEGF.cse(ex)
    @test length(b) == 0          # distinct subgraphs: no extraction despite equal keys
    m = NEGF.evaluate(NEGF.MatrixFamilyBackend{ComplexF64}(), ex)
    @test norm(m - (A * B + A * C)) / norm(A * B + A * C) < 1e-12
    # and the identical-subgraph win still works with a colliding key
    Z = NEGF.op_mul(NEGF.CompressedLeaf(A), NEGF.CompressedLeaf(B))
    ex2 = NEGF.op_add(X, Z)
    b2, _ = NEGF.cse(ex2)
    @test length(b2) == 1         # X === Z structurally: extracted once
end

@testitem "solve_dyson materializes a lazy non-local operand" begin
    import NonEquilibriumGreenFunction as NEGF
    using NonEquilibriumGreenFunction
    using LinearAlgebra
    ax = 0:0.5:5
    cpr = HssCompression()
    g = RetardedKernel(ax, Stationary(τ -> ComplexF64(-1im)); compression=cpr)
    # non-local retarded self-energy: build the KernelProd directly so it
    # stays a lazy node (adjoint of a LocalKernel product would go through
    # the concrete fast path), then solve through the documented pattern
    cl = LocalKernel(ax, t -> ComplexF64(0.5); compression=cpr)
    mid = NEGF.Kernels.KernelProd(cl', g, quadrature(g), Advanced()) |> adjoint
    Σ = NEGF.Kernels.KernelProd(mid, cl, quadrature(g), Retarded())
    @test Σ isa NEGF.Kernels.AbstractLazyOperator
    # documented forcing-point pattern: solve_dyson accepts the lazy tree
    G = solve_dyson(g, g * Σ)
    @test G isa Kernel
    # materialize is exported for the manual escape hatch
    k = NEGF.materialize(g * Σ)
    @test k isa Kernel
    # value matches the dense reference exactly
    gd = RetardedKernel(ax, Stationary(τ -> ComplexF64(-1im)); compression=NONCompression())
    cld = LocalKernel(ax, t -> ComplexF64(0.5); compression=NONCompression())
    midd = NEGF.Kernels.KernelProd(cld', gd, quadrature(gd), Advanced()) |> adjoint
    Σd = NEGF.Kernels.KernelProd(midd, cld, quadrature(gd), Retarded())
    Gd = solve_dyson(gd, gd * Σd)
    @test norm(matrix(G) - matrix(Gd)) / norm(matrix(Gd)) < 1e-10
end
