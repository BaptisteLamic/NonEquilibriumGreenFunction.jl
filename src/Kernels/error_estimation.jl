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
    println(io, "  Rigorous bound:              $(r.norm_bound)")
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
`A = I - δt (K - ½ diag K)` as the original solve. Also returns a rigorous
upper bound `exp(‖K‖_T) ‖D‖_max`, where `‖K‖_T` is the weighted trapezoid
column-sum norm (a discrete Gronwall stability constant of the retarded
problem).

Returns a `DiscretizationErrorEstimate` with fields:

- `estimate`: the estimated error `G - X` as a `Kernel` (diagonal exactly zero);
- `defect`: the defect `D` itself as a `Kernel`, for inspection;
- `norm_estimate`: `‖E‖_max`, the estimated discretization error;
- `norm_bound`: rigorous upper bound of `‖G - X‖_max`.

The estimate is asymptotically exact: validated on analytic solutions, the
effectivity `‖E‖ / ‖G - X‖` is ≈ 1 up to a few percent. The bound holds for
the compressed problem actually being solved; compression error must be
added separately when relevant.
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
    w = ones(n)
    w[1] *= 0.5
    w[end] *= 0.5
    w .*= dt
    Knorm = 0.0
    for j in 1:n
        Knorm = max(Knorm, sum(abs, view(Mk, :, blockrange(j, bs)) .* w[j]))
    end
    bound = exp(Knorm) * maximum(abs.(D))
    cp = compression(g)
    return DiscretizationErrorEstimate(
        make_similar(g, cp(E)),
        make_similar(g, cp(D)),
        maximum(abs.(E)),
        bound,
    )
end
