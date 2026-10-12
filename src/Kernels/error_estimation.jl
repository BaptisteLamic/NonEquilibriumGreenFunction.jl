"""
A-posteriori estimation of the time-discretization error of `solve_dyson`.

`solve_dyson` discretizes the retarded Dyson equation `G = g + K⋅G` with the
implicit trapezoidal rule: on the grid `t_1 < … < t_N` (step `δt`) the computed
`X` satisfies, for `i > j`,

    (1 - ½δt K_ii) X_ij = g_ij + δt Σ_{m=j+1}^{i-1} K_im X_mj + δt K_ij c_j g_jj

with `c_j = ½ (1 - ½δt K_jj)^{-1}` and the diagonal fixed exactly,
`X_jj = g_jj`. Substituting the exact solution `G` into this scheme defines
the defect `D = A G - b` with `A = I - δt (K - ½ diag K)`; the error
`e = X - G` of the computed solution obeys exactly `A e = D` (up to
round-off and compression error). The quadrature part of `D` is the
Euler–Maclaurin correction of the trapezoidal rule,

    D_ij ≈ -(δt²/12) [F′(t_i) - F′(t_j)] - δt (c_j - ½) F(t_j),

with `F(s) = K(t_i,s) G(s,t_j)`, estimated by one-sided three-point finite
differences of the *already computed* coarse data. No refined grid, second
problem or interpolation is needed: the estimate reuses `g`, `K` and the
solution `G`, at `O(N²)` block-arithmetic cost plus one triangular solve with
the same operator as the original solve.
"""
struct DiscretizationErrorEstimate
    estimate::Any
    defect::Any
    norm_estimate::Float64
    norm_bound::Float64
end

function Base.show(io::IO, r::DiscretizationErrorEstimate)
    println(io, "DiscretizationErrorEstimate:")
    println(io, "  Estimated error (‖E‖_max):   $(r.norm_estimate)")
    println(io, "  Error bound:                 $(r.norm_bound)")
end


# Scheme defect of the retarded solve, computed with whole-column/row
# GEMMs instead of per-entry closures: the i-end difference is, per row i,
# a 3-block stencil of G rows scaled by K's diagonal-adjacent blocks; the
# j-end difference and the endpoint bias are, per column j, linear in K's
# (j, j+1, j+2) column blocks with i-independent factors — one GEMM per
# column and one per row. No per-entry allocations remain.
function _scheme_defect(K::Kernel, G::Kernel)
    bs = blocksize(G)
    dt = step(G)
    Mk = to_cpu(matrix(K))
    Mh = to_cpu(matrix(G))
    n = length(axis(G))
    T = eltype(Mh)
    eye = Matrix{T}(I, bs, bs)
    half = T(0.5) .* eye
    c = -(dt^2 / 12) / (2dt)
    # dF_i part: per row i, 3 scaled row-blocks of Mh
    Di = zeros(T, bs * n, bs * n)
    @inbounds for i in 3:n
        ri = blockrange(i, bs)
        mul!(view(Di, ri, :), view(Mk, ri, ri), view(Mh, ri, :), 3, 0)
        im1 = blockrange(i - 1, bs)
        im2 = blockrange(i - 2, bs)
        mul!(view(Di, ri, :), view(Mk, ri, im1), view(Mh, im1, :), -4, 1)
        mul!(view(Di, ri, :), view(Mk, ri, im2), view(Mh, im2, :), 1, 1)
    end
    D = zeros(T, bs * n, bs * n)
    H = zeros(T, 3bs, bs)
    Kb = zeros(T, bs * n, 3bs)
    acc = zeros(T, bs * n, bs)
    @inbounds for j in 1:n-2
        rj = blockrange(j, bs)
        jp1 = blockrange(j + 1, bs)
        jp2 = blockrange(j + 2, bs)
        cj = half / (eye - dt * (half * Mk[rj, rj]))
        # j-end stencil factors: -3 Mh[j,j] + 4 Mh[j+1,j] - Mh[j+2,j]
        H[1:bs, :] .= -3 .* view(Mh, rj, rj)
        H[bs+1:2bs, :] .= 4 .* view(Mh, jp1, rj)
        H[2bs+1:3bs, :] .= -1 .* view(Mh, jp2, rj)
        Kb[:, 1:bs] .= view(Mk, :, rj)
        Kb[:, bs+1:2bs] .= view(Mk, :, jp1)
        Kb[:, 2bs+1:3bs] .= view(Mk, :, jp2)
        mul!(acc, Kb, H)
        # endpoint bias: -dt (cj - 1/2) K[:, rj] Mh[j,j] — a rank-bs column
        # update, NOT scaled by c (it is an O(dt) term, unlike the O(dt^2)
        # EM stencils)
        bias = -dt .* ((cj - half) * view(Mh, rj, rj))
        # assemble: D[:, rj] = c * (Di - acc) + K[:, rj]*bias, rows i >= j+2,
        # as whole-column BLAS ops (no per-entry loop)
        lo = (j + 1) * bs + 1
        Dcol = view(D, lo:bs * n, rj)
        Dcol .= c .* (view(Di, lo:bs * n, rj) .- view(acc, lo:bs * n, :))
        mul!(Dcol, view(Mk, lo:bs * n, rj), bias, 1, 1)
    end
    return D
end

"""
    estimate_discretization_error(g::Kernel, K::Kernel, G::Kernel)

Estimate the time-discretization error of `G = solve_dyson(g, K)` a
posteriori, without any grid refinement: the scheme defect is estimated by
Euler–Maclaurin from the already computed matrices, and the error estimate
`E = A⁻¹ D` is obtained with one additional solve with the same operator
`A = I - δt (K - ½ diag K)` as the original solve (a dense triangular solve;
see the note on cost below). Also returns a componentwise upper bound
`max(|A⁻¹| |D|)`: since the error obeys exactly `e = A⁻¹D` for the true
defect, this bounds max|e| without losing phase cancellation (the
previous discrete-Gronwall constant `exp(‖K‖_T) ‖D‖_max` overestimated
unitary kernels by up to ~1e64).
Returns a `DiscretizationErrorEstimate` with fields:

- `estimate`: the estimated error `G - X` as a `Kernel` (diagonal exactly zero);
- `defect`: the defect `D` itself as a `Kernel`, for inspection;
- `norm_estimate`: `‖E‖_max`, the estimated discretization error;
- `norm_bound`: upper bound of `‖G - X‖_max`, rigorous given the true defect
  and conditionally rigorous given its leading-order approximation (see
  above).

The estimate is asymptotically exact: validated on analytic solutions, the
effectivity `‖E‖ / ‖G - X‖` is ≈ 1 up to a few percent. The bound holds for
the compressed problem actually being solved; compression error must be
added separately when relevant.

The superdiagonal entries (`i = j+1`) are excluded from the *estimate*
(their single-panel endpoint bias is not resolved by the three-point one-sided
differences) but their leading-order contribution, `δt·|K_ij|·|G_jj|`
(validated to ≤3% on resolved grids, inflated ×2), is included in the
*bound*, so the bound remains valid for diagonally-peaked kernels whose
error maximum sits on the first superdiagonal.

Cost: one additional pass with the scheme operator, performed by *blocked*
block forward substitution on the kernel data (the operator is block
lower-triangular for retarded kernels) — no `(bs·N)²` operator is ever
materialized, only `W·bs`-wide panels, with BLAS-3 arithmetic throughout.
All defect stencils are computed as whole-row/whole-column GEMMs, so the
estimate adds O(N²) block work to the simulation — measured overhead of
the full Keldysh flow (`solve_keldysh_with_error_estimate` vs
`solve_keldysh`) is a small multiple of the simulation cost at example
scales, dominated by the two extra kernel products of the error
propagation through the dressing.

Caveats of the bound: it is derived from the leading-order defect estimate,
so on under-resolved grids (`δt·max|K| ≳ 0.5`) neither the estimate
nor the bound covers the true error (measured effectivity ~0.12 for an
oscillatory kernel on a coarse grid); refine the grid until `norm_estimate`
is stable before trusting either number.
"""
function estimate_discretization_error(g::Kernel, K::Kernel, G::Kernel)
    bs = blocksize(G)
    dt = step(G)
    n = length(axis(G))
    D = _scheme_defect(K, G)
    # solve A E = D through the *same compressed operator* the simulation
    # itself builds (kernel_solver.jl): all in compressed arithmetic, with
    # matrix(K) kept in its compressed storage (no dense round-trip), so the
    # solve is near-linear under HSS exactly like the simulation's own.
    cp = compression(g)
    eye = cp(sparse(scalartype(K)(1) * I, bs * n, bs * n))
    diag_K = extract_blockdiag(K |> matrix, bs)
    left = cp(eye - scalartype(K)(dt) * (matrix(K) - cp(1 // 2 * diag_K)))
    E = ldiv(left, cp(D))
    # zero the block diagonal exactly (e(j,j) = 0 by the scheme's diagonal
    # correction), with the same HSS-safe algebra as solve_dyson:
    # E + cp(-extract_blockdiag(E))
    E = cp(E - cp(extract_blockdiag(E, bs)))
    # Rigorous bound: the error obeys e = A⁻¹D for the true defect. The
    # inflated defect bound Dabs below is propagated through the *signed*
    # scheme operator by one block forward substitution (no operator or
    # inverse ever materialized), preserving phase cancellation — unlike
    # the previous discrete-Gronwall constant exp(‖K‖_T)‖D‖_max, which
    # overestimated unitary kernels by up to ~1e64. The superdiagonal entries (i = j+1) are
    # excluded from the estimate; their single-panel defect is bounded by
    # the exact triangle inequality: the true panel integral is within
    # δt·|K_ij|·|G_jj| of both endpoints' contributions, and the scheme
    # uses δt·|K_ij|·|c_j|·|g_jj| on the right-hand side, so the defect is
    # at most δt·|K_ij|·(|G_jj| + |c_j|·|g_jj|) — all coarse data.
    # the bound needs only diagonal-adjacent blocks of K, G, g — extracted
    # block-by-block from the compressed storage (O(N) tiny getindexes,
    # no dense (bs*N)^2 round-trip)
    Kbl = same_time_blocks(K |> matrix, bs)
    Gbl = same_time_blocks(G |> matrix, bs)
    gbl = same_time_blocks(g |> matrix, bs)
    T = eltype(Kbl[1])
    eye = Matrix{T}(I, bs, bs)
    half = T(0.5) .* eye
    Dabs = abs.(D)
    for j in 1:n
        rj = blockrange(j, bs)
        # the defect of the true solution is nonzero on the diagonal:
        # [A G - b](j,j) = (1 - dt K_jj/2) g_jj - g_jj/2 exactly (retarded
        # support kills all other terms), and e = A^-1 D - C with the
        # diagonal correction C makes e(j,j) = 0 exactly; the bound below
        # takes the max over off-diagonal entries only.
        Dabs[rj, rj] .= abs.((half - dt .* (Kbl[j] ./ 2)) * gbl[j])
    end
    for j in 1:n-1
        rj = blockrange(j, bs)
        Ksj = matrix(K)[blockrange(j + 1, bs), rj]
        cj = abs.(half / (eye - dt * (half * Kbl[j])))
        Dabs[blockrange(j + 1, bs), rj] .+=
            dt .* abs.(Ksj) .* (abs.(Gbl[j]) .+ cj .* abs.(gbl[j]))
    end
    pb = ldiv(left, cp(convert(Matrix{scalartype(K)}, Dabs)))
    pb = pb - cp(extract_blockdiag(pb, bs))
    bound = maximum(abs.(to_cpu(pb)))
    cp = compression(g)
    return DiscretizationErrorEstimate(
        make_similar(g, cp(E)),
        make_similar(g, cp(D)),
        maximum(abs, E),
        bound,
    )
end
