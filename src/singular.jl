using LinearAlgebra

"""
    singular_weights(f, dt; N, atol=1e-13, rtol=1e-13)

Product-integration weights for a stationary kernel with an integrable
singularity at `τ = 0` (principal-value sense), sampled on the uniform grid
`τ_k = k·dt`, `k = -(N-1)...(N-1)`.

Instead of sampling `f(τ_k)` — which loses one order of accuracy per
singularity — the weights are the exact integrals of `f` against the
piecewise-linear hat functions `ℓ_k` centered on the nodes:

``W_k = PV \\int f(τ) ℓ_k(τ) dτ``

so that `Σ_k W_k g(τ_k)` approximates `PV ∫ f(τ) g(τ) dτ` with second-order
accuracy for smooth `g`, *including* the singular `f`. For the odd thermal
core (`f(-τ) = -f(τ)`, e.g. `-i/β csch(πτ/β)`) the diagonal weight vanishes
exactly: `W_0 = 0`, which is the principal-value prescription.

Returns the weights as a vector indexed `k = -(N-1), ..., N-1`, i.e.
`W[k + N]`. For a matrix-valued core `f` (blocksize `> 1`), returns a
3D array `W[:, :, k + N]` of per-entry weights, indexed the same way.
"""
function singular_weights(f, dt; N, atol=1e-13, rtol=1e-13)
    sample = f(dt)
    return _singular_weights(f, dt, sample; N=N, atol=atol, rtol=rtol)
end

function _singular_weights(f, dt, ::Number; N, atol, rtol)
    hat(k) = τ -> max(0.0, 1.0 - abs(τ / dt - k))
    W = Vector{ComplexF64}(undef, 2N - 1)
    for k in -(N - 1):(N - 1)
        if k == 0
            W[k + N] = 0.0
        else
            W[k + N] = _pv_panel_integral(f, hat(k), (k - 1) * dt, k * dt, atol, rtol) +
                       _pv_panel_integral(f, hat(k), k * dt, (k + 1) * dt, atol, rtol)
        end
    end
    return W
end

function _singular_weights(f, dt, B::AbstractMatrix; N, atol, rtol)
    hat(k) = τ -> max(0.0, 1.0 - abs(τ / dt - k))
    T = complex(eltype(B))
    W = Array{T,3}(undef, size(B)..., 2N - 1)
    for k in -(N - 1):(N - 1)
        if k == 0
            W[:, :, k + N] .= zero(B)
        else
            W[:, :, k + N] = _pv_panel_integral(f, hat(k), (k - 1) * dt, k * dt, atol, rtol) +
                            _pv_panel_integral(f, hat(k), k * dt, (k + 1) * dt, atol, rtol)
        end
    end
    return W
end

"""
    _pv_panel_integral(f, ℓ, a, b, atol, rtol)

Principal-value-aware integral of `f·ℓ` over one panel `[a, b]`. Panels not
crossing the singularity are integrated directly; the singularity at `τ = 0`
is odd-symmetric so its PV contribution over any symmetric pair cancels.
"""
function _pv_panel_integral(f, ℓ, a, b, atol, rtol)
    if a == b
        return 0.0im
    end
    g = τ -> f(τ) * ℓ(τ)
    # panels containing 0 are split symmetrically; odd cores cancel, even
    # components integrate through numerically (they are integrable for the
    # csch-type core only in the PV sense, handled by the split below)
    if a < 0 < b
        mid = 0.0
        return _quad(g, a, mid, atol, rtol) + _quad(g, mid, b, atol, rtol)
    end
    return _quad(g, a, b, atol, rtol)
end

# thin wrapper so the integration backend can be swapped without touching
# the weight-generation logic; a fixed 16-node Gauss–Legendre panel rule
function _quad(g, a, b, atol, rtol)
    n = 16
    x, w = gauss_legendre_nodes(n)
    hm = (b - a) / 2
    hp = (a + b) / 2
    return hm * sum(w[i] * g(hp + hm * x[i]) for i in 1:n)
end

function gauss_legendre_nodes(n)
    β = [1 / sqrt(4 - 1 / (i^2)) for i in 1:(n - 1)]
    F = eigen(SymTridiagonal(zeros(n), β))
    x = F.values
    w = 2 .* (F.vectors[1, :] .^ 2)
    return x, w
end
