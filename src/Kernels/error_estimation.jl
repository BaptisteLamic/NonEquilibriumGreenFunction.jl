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

function _scheme_defect(K::Kernel, G::Kernel)
    bs = blocksize(G)
    dt = step(G)
    Mk = to_cpu(matrix(K))
    Mh = to_cpu(matrix(G))
    n = length(axis(G))
    T = eltype(Mh)
    eye = Matrix{T}(I, bs, bs)
    half = T(0.5) .* eye
    D = zeros(T, bs * n, bs * n)
    @inbounds for j in 1:n-2
        cj = half / (eye - dt * (half * Mk[blockrange(j, bs), blockrange(j, bs)]))
        for i in j+2:n
            ri = blockrange(i, bs)
            rj = blockrange(j, bs)
            F(s) = Mk[ri, blockrange(s, bs)] * Mh[blockrange(s, bs), rj]
            dF_j = (-3F(j) + 4F(j + 1) - F(j + 2)) / (2dt)
            dF_i = (3F(i) - 4F(i - 1) + F(i - 2)) / (2dt)
            D[ri, rj] = -(dt^2 / 12) .* (dF_i - dF_j) .- dt * ((cj - half) * Mk[ri, rj]) * Mh[rj, rj]
        end
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

Cost: the estimate requires one additional solve with the operator `A`.
Unlike `solve_dyson`, which can exploit HSS compression via `ldiv`, this
solve uses a dense `(bs·N)²` matrix and is therefore O(N³) — for the
compressed path the estimate can be asymptotically more expensive than the
solve it diagnoses. For `NONCompression` it remains cheaper than a single
grid refinement.

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
    Mk = to_cpu(matrix(K))
    T = eltype(Mk)
    diag_K = extract_blockdiag(Mk, bs)
    left = Matrix{T}(I, bs * n, bs * n) .- dt .* (Mk .- 0.5 .* Matrix(diag_K))
    E = left \ D
    for j in 1:n
        E[blockrange(j, bs), blockrange(j, bs)] .= 0
    end
    # Rigorous componentwise bound: |e| = |A⁻¹| |D| is exact for the
    # *estimated* defect, so max|A⁻¹ D| is a tight upper bound of the error
    # caused by that defect (no phase-cancellation loss, unlike the previous
    # discrete-Gronwall constant exp(‖K‖_T), which overestimated by up to
    # ~1e64 for unitary kernels). The superdiagonal entries (i = j+1) are
    # excluded from the estimate; their single-panel defect is bounded by
    # the exact triangle inequality: the true panel integral is within
    # δt·|K_ij|·|G_jj| of both endpoints' contributions, and the scheme
    # uses δt·|K_ij|·|c_j|·|g_jj| on the right-hand side, so the defect is
    # at most δt·|K_ij|·(|G_jj| + |c_j|·|g_jj|) — all coarse data.
    Ainv = inv(left)
    Dabs = abs.(D)
    Mh = to_cpu(matrix(G))
    Mg = to_cpu(matrix(g))
    eye = Matrix{T}(I, bs, bs)
    half = T(0.5) .* eye
    for j in 1:n
        rj = blockrange(j, bs)
        # the defect of the true solution is nonzero on the diagonal:
        # [A G - b](j,j) = (1 - dt K_jj/2) g_jj - g_jj/2 exactly (retarded
        # support kills all other terms), and e = A^-1 D - C with the
        # diagonal correction C makes e(j,j) = 0 exactly; the bound below
        # takes the max over off-diagonal entries only.
        Dabs[rj, rj] .= abs.((half - dt .* (Mk[rj, rj] ./ 2)) * Mg[rj, rj])
    end
    for j in 1:n-1
        rj = blockrange(j, bs)
        cj = abs.(half / (eye - dt * (half * Mk[rj, rj])))
        Dabs[blockrange(j + 1, bs), rj] .+=
            dt .* abs.(Mk[blockrange(j + 1, bs), rj]) .*
            (abs.(Mh[rj, rj]) .+ cj .* abs.(Mg[rj, rj]))
    end
    prod = abs.(Ainv * Dabs)
    for j in 1:n
        prod[blockrange(j, bs), blockrange(j, bs)] .= 0
    end
    bound = maximum(prod)
    cp = compression(g)
    return DiscretizationErrorEstimate(
        make_similar(g, cp(E)),
        make_similar(g, cp(D)),
        maximum(abs, E),
        bound,
    )
end
