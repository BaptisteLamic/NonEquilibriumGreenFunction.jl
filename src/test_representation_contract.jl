"""
    test_representation_contract(rep::AbstractRepresentation; kwargs...)

Contract audit: test that a representation tag implements the single
backend contract used throughout the package (construction through
`compress(backend(rep), ...)`, DAG evaluation, and the matrix-family
primitives). This is the acceptance test for new representations and their
matrix families.

Checks, each against a dense `DenseRepresentation()` reference on the same
problem:

1. **Construction**: `RetardedKernel`, `AdvancedKernel`,
   `AcausalKernel` (with `Stationary` and `TwoTime` maps),
   `LocalKernel` and `Separable`; sizes, `eltype`,
   `blocksize`, causality.
2. **Algebra**: `+`, `-`, `*`, scalar `*`, `adjoint`, `==`, `norm`,
   `same_time`, `keldysh_trace`, compared to the dense reference.
3. **Solver**: `solve_dyson(g, K)` vs the dense reference (exercises `ldiv`
   and the block-diagonal path end to end).
4. **Recompression**: `compress(backend(rep), matrix(kernel))`,
   `make_similar(op, rep)` and `compress!`.

Keyword arguments:

- `N = 32`: number of time steps.
- `bs = 2`: block size.
- `atol = 1e-8`: comparison tolerance.
- `types = (ComplexF64,)`: scalar types to test.

Returns `true` if all checks pass; throws a `Test` failure otherwise.
"""
function test_representation_contract(rep::AbstractRepresentation;
    N::Int=32, bs::Int=2, atol::Real=1e-8, types=(ComplexF64,))
    @testset "representation contract: $(nameof(typeof(rep)))" begin
        ax = LinRange(0.0, 1.0, N)
        for T in types
            @testset "$(T)" begin
                f = bs == 1 ?
                    ((t, tp) -> T(exp(-(t - tp)^2))) :
                    ((t, tp) -> T.(Matrix{Float64}(I, bs, bs) .* exp(-(t - tp)^2)))
                ref = DenseRepresentation()
                for caus in (Retarded(), Advanced(), Acausal())
                    ctor = caus isa Retarded ? RetardedKernel :
                           caus isa Advanced ? AdvancedKernel :
                           AcausalKernel
                    k_ref = ctor(ax, TwoTime(f); representation=ref)
                    k = ctor(ax, TwoTime(f); representation=rep)
                    @test size(matrix(k)) == size(matrix(k_ref))
                    @test scalartype(k) == T
                    @test blocksize(k) == bs
                    @test causality(k) == caus
                    _assert_matrices_approx_equal(matrix(k), matrix(k_ref), atol)
                end
                @testset "stationary" begin
                    f_s = bs == 1 ?
                        (τ -> T(exp(-τ^2))) :
                        (τ -> T.(Matrix{Float64}(I, bs, bs) .* exp(-τ^2)))
                    k_ref = RetardedKernel(ax, Stationary(f_s); representation=ref)
                    k = RetardedKernel(ax, Stationary(f_s); representation=rep)
                    _assert_matrices_approx_equal(matrix(k), matrix(k_ref), atol)
                end
                @testset "local" begin
                    d_ref = LocalKernel(ax, t -> T(2); representation=ref)
                    d = LocalKernel(ax, t -> T(2); representation=rep)
                    @test locality(d) == Local()
                    @test causality(d) == Acausal()
                    _assert_matrices_approx_equal(matrix(d), matrix(d_ref), atol)
                end
                @testset "lowrank" begin
                    lf = t -> T.([t 0; 0 t])
                    lg = t -> T.([t^2 0; 0 t^2])
                    l_ref = RetardedKernel(ax, Separable(lf, lg); representation=ref)
                    l = RetardedKernel(ax, Separable(lf, lg); representation=rep)
                    _assert_matrices_approx_equal(matrix(l), matrix(l_ref), atol)
                end
                _fK = bs == 1 ?
                    ((t, tp) -> T(0.1 * exp(-(t - tp)))) :
                    ((t, tp) -> T.(Matrix{Float64}(I, bs, bs) .* (0.1 * exp(-(t - tp)))))
                @testset "algebra" begin
                    g_ref = RetardedKernel(ax, TwoTime(f); representation=ref)
                    K_ref = RetardedKernel(ax, TwoTime(_fK); representation=ref)
                    g = RetardedKernel(ax, TwoTime(f); representation=rep)
                    K = RetardedKernel(ax, TwoTime(_fK); representation=rep)
                    _assert_matrices_approx_equal(matrix(g + K), matrix(g_ref + K_ref), atol)
                    _assert_matrices_approx_equal(matrix(g - K), matrix(g_ref - K_ref), atol)
                    _assert_matrices_approx_equal(matrix(g * K), matrix(g_ref * K_ref), atol)
                    _assert_matrices_approx_equal(matrix(2 * g), matrix(2 * g_ref), atol)
                    _assert_matrices_approx_equal(matrix(g'), matrix(g_ref'), atol)
                    @test norm(g) ≈ norm(g_ref) rtol=atol
                    @test same_time(g) ≈ same_time(g_ref) atol=atol
                    @test keldysh_trace(g) ≈ keldysh_trace(g_ref) atol=atol
                end
                @testset "solver" begin
                    _fg = bs == 1 ? (τ -> T(-1im)) :
                        (τ -> T.(-1im .* Matrix{Float64}(I, bs, bs)))
                    _fKs = bs == 1 ? (τ -> T(-0.05im)) :
                        (τ -> T.(-0.05im .* Matrix{Float64}(I, bs, bs)))
                    g_ref = RetardedKernel(ax, Stationary(_fg); representation=ref)
                    K_ref = RetardedKernel(ax, Stationary(_fKs); representation=ref)
                    G_ref = solve_dyson(g_ref, K_ref)
                    g = RetardedKernel(ax, Stationary(_fg); representation=rep)
                    K = RetardedKernel(ax, Stationary(_fKs); representation=rep)
                    G = solve_dyson(g, K)
                    _assert_matrices_approx_equal(matrix(G), matrix(G_ref), atol)
                    @test same_time(G) ≈ same_time(G_ref) atol=atol
                end
                @testset "recompression and make_similar" begin
                    k = RetardedKernel(ax, TwoTime(f); representation=rep)
                    m2 = compress(backend(rep, T), matrix(k))
                    @test size(m2) == size(matrix(k))
                    _assert_matrices_approx_equal(m2, matrix(k), atol)
                    k2 = make_similar(k, rep)
                    @test representation(k2) == rep
                    k3 = deepcopy(k)
                    compress!(k3)
                    _assert_matrices_approx_equal(matrix(k3), matrix(k), atol)
                end
                @testset "block-diagonal constructor" begin
                    blocks = [T(i + 10 * j + 100 * blk) for i in 1:bs, j in 1:bs, blk in 1:N]
                    m = compress(backend(rep, T), blocks)
                    @test size(m) == (N * bs, N * bs)
                    ref = zeros(T, N * bs, N * bs)
                    for blk in 1:N
                        ref[blockrange(blk, bs), blockrange(blk, bs)] .= blocks[:, :, blk]
                    end
                    _assert_matrices_approx_equal(m, ref, atol)
                end
            end
        end
    end
    return true
end

function _assert_matrices_approx_equal(m, m_ref, atol)
    b = MatrixFamilyBackend{eltype(m isa AbstractMatrix ? m : Matrix(m))}()
    a = m isa AbstractMatrix ? to_cpu(b, m) : Matrix(m)
    @test isapprox(a, to_cpu(b, m_ref); atol=atol, norm=x -> norm(x))
    return nothing
end
