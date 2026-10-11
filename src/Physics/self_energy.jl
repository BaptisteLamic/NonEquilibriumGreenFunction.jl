export second_born_self_energy, hartree_fock_self_energy, lesser_greater

import ..NonEquilibriumGreenFunction.Kernels: Kernel, LocalKernel, adjoint
import ..NonEquilibriumGreenFunction: discretization, matrix, axis, blocksize, blockrange,
    make_similar, compression, build_blockdiag, TrapzDiscretisation, Retarded, Acausal
import ..NonEquilibriumGreenFunction: isretarded, isacausal

"""
    lesser_greater(G_R, G_K)

Contour components of a Keldysh-rotated Green function:
`G⁼ = (G^K - G^R + G^A)/2` and `G⁽ = (G^K + G^R - G^A)/2`, with
`G^A = (G^R)†`. Returns the pair `(G_lesser, G_greater)` as dense matrices in
the block-time representation of `G_R`.
"""
function lesser_greater(G_R, G_K)
    MR = Matrix(matrix(G_R))
    MK = Matrix(matrix(G_K))
    MA = Matrix(matrix(adjoint(G_R)))
    G_lesser = (MK - MR + MA) / 2
    G_greater = (MK + MR - MA) / 2
    return G_lesser, G_greater
end

# Blocks of the second-Born contraction `xi*bubble - exchange` for the
# density-density vertex v_ijkl = V_ij δ_il δ_jk (Tuovinen, Covito & Sentef,
# arXiv:1905.01180, Eq. (16)): with `G = X(t,t')` and `Gt = X'(t',t)`,
#   B_ij = Σ_kl V_ik V_jl [xi G_ij Ĝ_lk G_kl - G_il Ĝ_lk G_kj]
# where Ĝ_lk = X'(t',t) evaluated in the same block. Every factor sits in the
# same block, so B is a per-block entrywise contraction.
function _born_blocks(G, Gt, V, bs; xi=2)
    N = div(size(G, 1), bs)
    T = eltype(G)
    S = similar(G)
    @inbounds for I in 1:N, J in 1:N
        rI = blockrange(I, bs)
        rJ = blockrange(J, bs)
        g = G[rI, rJ]
        gt = Gt[rI, rJ]
        for i in 1:bs, j in 1:bs
            acc = zero(T)
            for k in 1:bs, l in 1:bs
                vv = V[i, k] * V[j, l]
                acc += vv * (xi * g[i, j] * gt[l, k] * g[k, l] - g[i, l] * gt[l, k] * g[k, j])
            end
            S[i + (I - 1) * bs, j + (J - 1) * bs] = acc
        end
    end
    return S
end

# X(t',t) in block-time representation: the block-transpose of the stored
# X(t,t') matrix (in-block indices stay attached to their time argument).
function _block_transpose(X, bs)
    N = div(size(X, 1), bs)
    Y = similar(X)
    @inbounds for I in 1:N, J in 1:N
        Y[blockrange(I, bs), blockrange(J, bs)] .= X[blockrange(J, bs), blockrange(I, bs)]
    end
    return Y
end

# Retarded masking: zero the strict upper block triangle (t < t').
function _mask_retarded(X, bs)
    Y = copy(X)
    N = div(size(X, 1), bs)
    @inbounds for J in 1:N, I in 1:(J - 1)
        Y[blockrange(I, bs), blockrange(J, bs)] .= 0
    end
    return Y
end

"""
    second_born_self_energy(G_R, G_K, V; spin_degeneracy=2)

Second-Born correlation self-energy for a density-density interaction with
coupling matrix `V` (`V = U` for a single level). Returns `(; Σ_R, Σ_K)` in
the Keldysh-symmetric representation used by `solve_keldysh`:

- `Σ_R = θ(t-t')(Σ⁽ - Σ⁼)`, a `Retarded` kernel;
- `Σ_K = Σ⁼ + Σ⁽`, an `Acausal` kernel;

with `Σ⁼ = i B(G⁼, G⁽(·,·))` and `Σ⁽ = -i B(G⁽, G⁼(·,·))` built from the
contour components of `(G_R, G_K)`. Both outputs carry the axis, blocksize,
compression and quadrature of `G_R`. `G_R` must be retarded and `G_K`
acausal. ħ = 1.

`spin_degeneracy` (ξ = 2 by default) multiplies the bubble (direct) diagram,
which carries a closed loop; ξ = 2 reproduces the spin-compensated second
Born. Pass 1 for spin-resolved levels.
"""
function second_born_self_energy(G_R, G_K, V; spin_degeneracy=2)
    isretarded(G_R) || throw(ArgumentError("G_R must be a retarded kernel"))
    isacausal(G_K) || throw(ArgumentError("G_K must be an acausal (Keldysh) kernel"))
    bs = blocksize(G_R)
    Vm = V isa Number ? (V .* Matrix{eltype(matrix(G_R))}(I, 1, 1)) : Matrix(V)
    size(Vm) == (bs, bs) || throw(ArgumentError(
        "interaction matrix of size $(size(Vm)) incompatible with blocksize $bs"))
    G_lesser, G_greater = lesser_greater(G_R, G_K)
    S_lesser = im .* _born_blocks(G_lesser, _block_transpose(G_greater, bs), Vm, bs; xi=spin_degeneracy)
    S_greater = -im .* _born_blocks(G_greater, _block_transpose(G_lesser, bs), Vm, bs; xi=spin_degeneracy)
    dis = discretization(G_R)
    Σ_R = Kernel(make_similar(dis, _mask_retarded(S_greater - S_lesser, bs)), Retarded())
    Σ_K = Kernel(make_similar(dis, S_lesser + S_greater), Acausal())
    return (; Σ_R, Σ_K)
end

"""
    hartree_fock_self_energy(G_R, G_K, V; spin_degeneracy=2)

Hartree-Fock self-energy of the density-density interaction `V` as a
time-local operator:
`Σ_HF = ξ Diagonal(V diag(ρ)) - V∘ρ'` with the one-body density matrix
`ρ(t) = -i G⁼(t,t)` and the same `spin_degeneracy` convention as
`second_born_self_energy`. Returns a `LocalKernel` with the axis and
compression of `G_R`. ħ = 1.
"""
function hartree_fock_self_energy(G_R, G_K, V; spin_degeneracy=2)
    bs = blocksize(G_R)
    Vm = V isa Number ? (V .* Matrix{eltype(matrix(G_R))}(I, 1, 1)) : Matrix(V)
    size(Vm) == (bs, bs) || throw(ArgumentError(
        "interaction matrix of size $(size(Vm)) incompatible with blocksize $bs"))
    G_lesser, _ = lesser_greater(G_R, G_K)
    T = eltype(G_lesser)
    ax = axis(G_R)
    N = length(ax)
    blocks = Array{T,3}(undef, bs, bs, N)
    for I in 1:N
        rI = blockrange(I, bs)
        rho = -im .* G_lesser[rI, rI]
        blocks[:, :, I] .= spin_degeneracy .* Diagonal(Vm * diag(rho)) .- Vm .* transpose(rho)
    end
    cpr = compression(discretization(G_R))
    return LocalKernel(TrapzDiscretisation(ax, build_blockdiag(blocks; compression=cpr), bs, cpr))
end
