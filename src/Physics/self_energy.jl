export second_born_self_energy, hartree_fock_self_energy, lesser_greater

import ..NonEquilibriumGreenFunction.Kernels: Kernel, LocalKernel, adjoint, TwoTime, Stationary,
    RetardedKernel, AcausalKernel, isretarded, isacausal, islocal
import ..NonEquilibriumGreenFunction: discretization, matrix, axis, blocksize, blockrange,
    compression

# Reverse time lookup on a uniform axis: the compression machinery samples
# block functions at axis points only, so the round-trip is exact.
@inline function _time_index(ax, t)
    return round(Int, (t - first(ax)) / step(ax)) + 1
end

# On-demand bs×bs block of a (possibly compressed) kernel matrix at time
# blocks (I, J), without materializing the full storage.
@inline function _block(m, I, J, bs)
    r = Vector{Int}(blockrange(I, bs))
    c = Vector{Int}(blockrange(J, bs))
    return m[r, c]
end

# Anchor pair of time-block indices realizing the separation Δ = I - J
# (|Δ| ≤ N-1); any pair with the same separation carries the same block for
# stationary (circulant) storage.
@inline function _separation_anchor(Δ, N)
    I = 1 + max(Δ, 0)
    return I, I - Δ
end

"""
    lesser_greater(G_R, G_K; stationary=false)

Contour components of a Keldysh-rotated Green function:
`G⁼ = (G^K - G^R + G^A)/2` and `G⁽ = (G^K + G^R - G^A)/2`, with
`G^A = (G^R)†`. Returns `(G_lesser, G_greater)` as kernel maps that read the
blocks of `(G_R, G_K)` on demand, without materializing any dense `N×N`
matrix. With `stationary = true` the maps are `Stationary` cores reading
blocks by time separation, matching the circulant storage of stationary
kernels.
"""
function lesser_greater(G_R, G_K; stationary=false)
    ax = axis(G_R)
    bs = blocksize(G_R)
    N = length(ax)
    MR, MK = matrix(G_R), matrix(G_K)
    MA = adjoint(G_R) |> matrix
    if stationary
        G⁼core = τ -> begin
            I, J = _separation_anchor(round(Int, τ / step(ax)), N)
            return (_block(MK, I, J, bs) - _block(MR, I, J, bs) + _block(MA, I, J, bs)) / 2
        end
        G⁽core = τ -> begin
            I, J = _separation_anchor(round(Int, τ / step(ax)), N)
            return (_block(MK, I, J, bs) + _block(MR, I, J, bs) - _block(MA, I, J, bs)) / 2
        end
        return Stationary(G⁼core), Stationary(G⁽core)
    end
    G⁼map = TwoTime((t, tp) -> begin
        I = _time_index(ax, t)
        J = _time_index(ax, tp)
        return (_block(MK, I, J, bs) - _block(MR, I, J, bs) + _block(MA, I, J, bs)) / 2
    end)
    G⁽map = TwoTime((t, tp) -> begin
        I = _time_index(ax, t)
        J = _time_index(ax, tp)
        return (_block(MK, I, J, bs) + _block(MR, I, J, bs) - _block(MA, I, J, bs)) / 2
    end)
    return G⁼map, G⁽map
end

# Per-block second-Born contraction (bubble and exchange are entrywise in
# the in-block indices; no factor mixes time blocks):
# B_ij = Σ_kl V_ik V_jl [ξ G_ij Ĝ_lk G_kl - G_il Ĝ_lk G_kj]
function _born_block(g, gt, V; xi=2)
    bs = size(g, 1)
    T = eltype(g)
    S = similar(g)
    @inbounds for i in 1:bs, j in 1:bs
        acc = zero(T)
        for k in 1:bs, l in 1:bs
            vv = V[i, k] * V[j, l]
            acc += vv * (xi * g[i, j] * gt[l, k] * g[k, l] - g[i, l] * gt[l, k] * g[k, j])
        end
        S[i, j] = acc
    end
    return S
end

"""
    second_born_self_energy(G_R, G_K, V; spin_degeneracy=2, stationary=false)

Second-Born correlation self-energy for a density-density interaction with
coupling matrix `V` (`V = U` for a single level). Returns `(; Σ_R, Σ_K)` in
the Keldysh-symmetric representation used by `solve_keldysh`:

- `Σ_R = θ(t-t')(Σ⁽ - Σ⁼)`, a `Retarded` kernel;
- `Σ_K = Σ⁼ + Σ⁽`, an `Acausal` kernel;

with `Σ⁼ = i B(G⁼, G⁽(·,·))` and `Σ⁽ = -i B(G⁽, G⁼(·,·))` built from the
contour components of `(G_R, G_K)` following the density-density vertex of
Tuovinen, Covito & Sentef (arXiv:1905.01180, Eq. (16)). Every
Green-function factor sits in the same `(t, t')` block: the 2B diagrams are
pointwise in `(t, t')`.

The self-energies are built as typed kernel maps sampled on demand through
the compression of `G_R` (default HSS): no dense `N×N` matrix is ever
materialized, and the construction cost follows the adaptive compression
sampling. `G_R` must be retarded and `G_K` acausal. ħ = 1.

`spin_degeneracy` (ξ = 2 by default) multiplies the bubble (direct)
diagram, which carries a closed loop; ξ = 2 reproduces the spin-compensated
second Born. Pass 1 for spin-resolved levels.

If both Green functions are time-translation invariant, pass
`stationary = true`: the self-energy cores are then read by time separation
and discretized through the circulant path (O(N) core evaluations).
"""
function second_born_self_energy(G_R, G_K, V; spin_degeneracy=2, stationary=false)
    isretarded(G_R) || throw(ArgumentError("G_R must be a retarded kernel"))
    isacausal(G_K) || throw(ArgumentError("G_K must be an acausal (Keldysh) kernel"))
    bs = blocksize(G_R)
    Vm = V isa Number ? (V .* Matrix{eltype(matrix(G_R))}(I, bs, bs)) : Matrix(V)
    size(Vm) == (bs, bs) || throw(ArgumentError(
        "interaction matrix of size $(size(Vm)) incompatible with blocksize $bs"))
    ax = axis(G_R)
    G⁼, G⁽ = lesser_greater(G_R, G_K; stationary=stationary)
    cpr = compression(discretization(G_R))

    B_lesser = (t, tp) -> _born_block(G⁼(t, tp), G⁽(tp, t), Vm; xi=spin_degeneracy)
    B_greater = (t, tp) -> _born_block(G⁽(t, tp), G⁼(tp, t), Vm; xi=spin_degeneracy)

    Σ_K_core = (t, tp) -> im .* B_lesser(t, tp) .- im .* B_greater(t, tp)
    Σ_R_core = (t, tp) -> -im .* B_greater(t, tp) .- im .* B_lesser(t, tp)

    wrap(f) = stationary ? Stationary(τ -> f(τ, 0)) : TwoTime(f)
    Σ_K = AcausalKernel(ax, wrap(Σ_K_core); compression=cpr)
    Σ_R = RetardedKernel(ax, wrap(Σ_R_core); compression=cpr)
    return (; Σ_R, Σ_K)
end

"""
    hartree_fock_self_energy(G_R, G_K, V; spin_degeneracy=2, stationary=false)

Hartree-Fock self-energy of the density-density interaction `V` as a
time-local operator:
`Σ_HF = ξ Diagonal(V diag(ρ)) - V∘ρ'` with the one-body density matrix
`ρ(t) = -i G⁼(t,t)` and the same `spin_degeneracy` convention as
`second_born_self_energy`. The equal-time blocks are read on demand from
the (possibly compressed) kernel matrices; no dense `N×N` matrix is ever
materialized. Returns a `LocalKernel` with the axis and compression of
`G_R`. ħ = 1. `stationary` is accepted for interface symmetry and only
affects how the density blocks are located.
"""
function hartree_fock_self_energy(G_R, G_K, V; spin_degeneracy=2, stationary=false)
    bs = blocksize(G_R)
    Vm = V isa Number ? (V .* Matrix{eltype(matrix(G_R))}(I, bs, bs)) : Matrix(V)
    size(Vm) == (bs, bs) || throw(ArgumentError(
        "interaction matrix of size $(size(Vm)) incompatible with blocksize $bs"))
    G⁼, _ = lesser_greater(G_R, G_K; stationary=stationary)
    ax = axis(G_R)
    return LocalKernel(ax, t -> begin
        r = -im .* G⁼(t, t)
        return spin_degeneracy .* Diagonal(Vm * diag(r)) .- Vm .* transpose(r)
    end; compression=compression(discretization(G_R)))
end
