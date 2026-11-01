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
    import NonEquilibriumGreenFunction: AbstractBackend, AbstractCompressedOperator,
        CompressedLeaf, DirectSolve
    using LinearAlgebra

    struct MockOp <: AbstractCompressedOperator
        m::Matrix{ComplexF64}
    end
    struct MockBackend <: AbstractBackend end

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

    # counting evaluate: wraps a real representation, counts backend calls
    struct CountingRepresentation <: NEGF.AbstractRepresentation
        inner::DenseRepresentation
        count::Base.RefValue{Int}
    end
    # the representation bridges to a counting backend over the dense family
    NEGF.backend(r::CountingRepresentation, ::Type{T}=ComplexF64) where {T<:Number} =
        CountingBackend{T}(r.count)
    struct CountingBackend{T<:Number} <: NEGF.AbstractBackend
        count::Base.RefValue{Int}
    end
    NEGF.working_eltype(::CountingBackend{T}) where {T} = T
    # construction delegates to the dense backend
    NEGF.compress(b::CountingBackend, args...; kw...) = NEGF.compress(NEGF.MatrixFamilyBackend{ComplexF64}(), args...; kw...)
    # evaluation counts, then delegates to the default interpreter
    function NEGF.evaluate(b::CountingBackend, ex::NEGF.AbstractOpNode)
        b.count[] += 1
        return invoke(NEGF.evaluate, Tuple{NEGF.AbstractBackend,NEGF.AbstractOpNode},
            NEGF.MatrixFamilyBackend{ComplexF64}(), ex)
    end

    cnt = Base.RefValue(0)
    rep = CountingRepresentation(DenseRepresentation(), cnt)
    ax = 0:0.1:1.0
    A = RetardedKernel(ax, Stationary(τ -> ComplexF64(exp(-τ))), representation=rep)
    B = RetardedKernel(ax, Stationary(τ -> ComplexF64(cos(τ))), representation=rep)
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
    struct CountBackend <: NEGF.AbstractBackend end
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
    rep = HssRepresentation()
    g = RetardedKernel(ax, Stationary(τ -> ComplexF64(-1im)); representation=rep)
    # non-local retarded self-energy: build the KernelProd directly so it
    # stays a lazy node (adjoint of a LocalKernel product would go through
    # the concrete fast path), then solve through the documented pattern
    cl = LocalKernel(ax, t -> ComplexF64(0.5); representation=rep)
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
    gd = RetardedKernel(ax, Stationary(τ -> ComplexF64(-1im)); representation=DenseRepresentation())
    cld = LocalKernel(ax, t -> ComplexF64(0.5); representation=DenseRepresentation())
    midd = NEGF.Kernels.KernelProd(cld', gd, quadrature(gd), Advanced()) |> adjoint
    Σd = NEGF.Kernels.KernelProd(midd, cld, quadrature(gd), Retarded())
    Gd = solve_dyson(gd, gd * Σd)
    @test norm(matrix(G) - matrix(Gd)) / norm(matrix(Gd)) < 1e-10
end

@testitem "compress is the construction primitive of the numeric tier" begin
    import NonEquilibriumGreenFunction as NEGF
    using LinearAlgebra
    ax = 0:0.1:0.5
    f = (t, tp) -> ComplexF64(exp(-(t - tp)^2))
    for rep in (NEGF.DenseRepresentation(), NEGF.HssRepresentation(atol=1e-12, rtol=1e-12, leafsize=8))
        b = NEGF.backend(rep, ComplexF64)
        m = NEGF.compress(b, ax, f)
        ref = zeros(ComplexF64, 6, 6)
        for i in 1:6, j in 1:6
            ref[i, j] = f(ax[i], ax[j])
        end
        @test Matrix(m) ≈ ref atol=1e-8
        m2 = NEGF.compress(b, Matrix(m))
        @test Matrix(m2) ≈ Matrix(m) atol=1e-8
        blocks = [ComplexF64(i + j + 10 * k) for i in 1:2, j in 1:2, k in 1:3]
        bd = NEGF.compress(b, blocks)
        bref = zeros(ComplexF64, 6, 6)
        for k in 1:3
            bref[NEGF.blockrange(k, 2), NEGF.blockrange(k, 2)] .= blocks[:, :, k]
        end
        @test Matrix(bd) ≈ bref atol=1e-8
    end
end

@testitem "contract audit meta-test: shipped backends implement every CORE_PRIMITIVE" begin
    import NonEquilibriumGreenFunction as NEGF
    backends = (
        NEGF.MatrixFamilyBackend{ComplexF64}(),
        NEGF.HssMatricesBackend(atol=1e-10, rtol=1e-10, leafsize=8),
    )
    # does `f` have a method with at least `arity` parameters in which some
    # argument position is annotated with (a subtype of) backend type `bt`?
    function covers(f, bt, arity)
        for m in methods(f)
            sig = m.sig
            while sig isa UnionAll
                sig = sig.body
            end
            sig isa DataType || continue
            ps = sig.parameters
            length(ps) >= arity || continue
            for p in ps[2:end]
                p isa UnionAll && (p = p.body)
                if p isa DataType && (bt <: p.name.wrapper || p <: bt)
                    return true
                end
                # family-dispatched primitives (op_add on HssMatrix, ...):
                # the backend is implied by the family, not an argument
                p isa DataType && p.name.wrapper <: AbstractMatrix && return true
            end
        end
        return false
    end
    for b in backends
        for (f, arity) in NEGF.CORE_PRIMITIVES
            @test covers(f, typeof(b), arity)
        end
    end
end

@testitem "SolveNode fails fast on unsupported strategies" begin
    import NonEquilibriumGreenFunction as NEGF
    using HssMatrices
    using LinearAlgebra
    b = NEGF.HssMatricesBackend(atol=1e-12, rtol=1e-12, leafsize=16)
    A = hss(Matrix(randn(8, 8) .+ im .* randn(8, 8)), atol=1e-12, rtol=1e-12, leafsize=16)
    B = hss(Matrix(randn(8, 8) .+ im .* randn(8, 8)), atol=1e-12, rtol=1e-12, leafsize=16)
    @test_throws ArgumentError NEGF.SolveNode(b, NEGF.CompressedLeaf(A), NEGF.CompressedLeaf(B),
        NEGF.StructureOptimizingSolve(10, 1e-8))
    node = NEGF.SolveNode(b, NEGF.CompressedLeaf(A), NEGF.CompressedLeaf(B), NEGF.DirectSolve())
    @test node isa NEGF.SolveNode
    @test Matrix(NEGF.evaluate(b, node)) ≈ Matrix(A) \ Matrix(B) atol=1e-6
end

@testitem "HSS evaluate runs CSE: repeated subchain inside a MulNode computed once" begin
    import NonEquilibriumGreenFunction as NEGF
    using HssMatrices
    using LinearAlgebra
    n = 32
    A = hss(Matrix([exp(-abs(i - j) / 4.0) * (1.0 + 0.5im) for i in 1:n, j in 1:n]), atol=1e-12, rtol=1e-12, leafsize=16)
    B = hss(Matrix([exp(-abs(i - j) / 5.0) * (0.5 - 0.25im) for i in 1:n, j in 1:n]), atol=1e-12, rtol=1e-12, leafsize=16)
    C = hss(Matrix([exp(-abs(i - j) / 6.0) * (0.25 + 1.0im) for i in 1:n, j in 1:n]), atol=1e-12, rtol=1e-12, leafsize=16)
    b = NEGF.HssMatricesBackend(atol=1e-12, rtol=1e-12, kest=20, leafsize=16)
    AB = NEGF.op_mul(NEGF.CompressedLeaf(A), NEGF.CompressedLeaf(B))
    ex = NEGF.op_mul(AB, NEGF.CompressedLeaf(C), AB)
    bindings, root = NEGF.cse(ex)
    @test length(bindings) == 1
    r = NEGF.evaluate(b, ex)
    @test r isa HssMatrix
    @test Matrix(r) ≈ Matrix(A) * Matrix(B) * Matrix(C) * Matrix(A) * Matrix(B) atol=1e-8
end

@testitem "eltype homogeneity: HSS widens, generic family MethodErrors" begin
    import NonEquilibriumGreenFunction as NEGF
    using HssMatrices
    using LinearAlgebra
    n = 16
    Af = hss(Matrix([exp(-abs(i - j) / 4.0) for i in 1:n, j in 1:n]), atol=1e-12, rtol=1e-12, leafsize=8)
    Bc = hss(Matrix([exp(-abs(i - j) / 5.0) * (1.0 + 1.0im) for i in 1:n, j in 1:n]), atol=1e-12, rtol=1e-12, leafsize=8)
    S = NEGF.op_mul(Af, Bc)
    @test eltype(S) == ComplexF64
    @test Matrix(S) ≈ Matrix(Af) * Matrix(Bc) atol=1e-8
    # generic family without native promotion: a family whose Base `*`
    # itself rejects mixed-eltype operands MethodErrors through the
    # generic op_mul = A * B
    struct NoPromoteMatrix <: AbstractMatrix{Float64}
        m::Matrix{Float64}
    end
    Base.size(A::NoPromoteMatrix) = size(A.m)
    Base.getindex(A::NoPromoteMatrix, i::Int, j::Int) = A.m[i, j]
    Base.:*(A::NoPromoteMatrix, B::AbstractMatrix) = throw(MethodError(*, (A, B)))
    X = NoPromoteMatrix(randn(4, 4))
    Y = randn(4, 4) .+ im .* randn(4, 4)
    @test_throws MethodError NEGF.op_mul(X, Y)
end

@testitem "working_eltype(backend(rep, T)) matches the kernel scalartype" begin
    import NonEquilibriumGreenFunction as NEGF
    using NonEquilibriumGreenFunction
    ax = 0:0.1:0.5
    for rep in (DenseRepresentation(), HssRepresentation(leafsize=8))
        k = RetardedKernel(ax, Stationary(τ -> Float32(-1f0)); representation=rep)
        @test scalartype(k) == Float32
        @test NEGF.working_eltype(NEGF.backend(rep, Float32)) == Float32
        @test NEGF.working_eltype(NEGF.backend(rep, scalartype(k))) == scalartype(k)
    end
end

@testitem "end-to-end construction with the new names" begin
    using NonEquilibriumGreenFunction
    using LinearAlgebra
    ax = 0:0.05:1.0
    g = RetardedKernel(ax, Stationary(τ -> ComplexF64(-1im * exp(-τ))); representation=HssRepresentation(atol=1e-10, rtol=1e-10, leafsize=8))
    K = RetardedKernel(ax, Stationary(τ -> ComplexF64(-0.05im * exp(-τ))); representation=HssRepresentation(atol=1e-10, rtol=1e-10, leafsize=8))
    G = solve_dyson(g, K)
    gd = RetardedKernel(ax, Stationary(τ -> ComplexF64(-1im * exp(-τ))); representation=DenseRepresentation())
    Kd = RetardedKernel(ax, Stationary(τ -> ComplexF64(-0.05im * exp(-τ))); representation=DenseRepresentation())
    Gd = solve_dyson(gd, Kd)
    @test norm(matrix(G) - matrix(Gd)) / norm(matrix(Gd)) < 1e-6
end
