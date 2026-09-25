module Physics

using ..NonEquilibriumGreenFunction: polygamma
using ..NonEquilibriumGreenFunction: I
import ..NonEquilibriumGreenFunction.Kernels: Kernel, solve_dyson, causality, isretarded, isadvanced, isacausal,
    discretization, matrix, compression, blocksize, axis, step, scalartype, make_similar,
    Retarded, Acausal, Instantaneous, AbstractCausality, adjoint
import Base: *

include("physics.jl")

"""
    solve_keldysh(g, Σ_R, Σ_K; check=true)

Solve the Dyson equation for the retarded branch and dress the kinetic branch.

`g` is the bare retarded Green function, `Σ_R` the retarded self-energy and
`Σ_K` the Keldysh self-energy. Returns `(; G_R, G_K)` where
`G_R = solve_dyson(g, g * Σ_R)` and `G_K = G_R * Σ_K * G_R'`.

When `check` is true the causalities are validated: `g` and `Σ_R` must be
retarded or instantaneous and `Σ_K` acausal or instantaneous.
"""
function solve_keldysh(g::Kernel, Σ_R, Σ_K; check=true)
    if check
        retarded_ok(x) = isretarded(x) || causality(x) == Instantaneous()
        acausal_ok(x) = isacausal(x) || causality(x) == Instantaneous()
        @assert retarded_ok(g) "g must be retarded"
        @assert retarded_ok(Σ_R) "Σ_R must be retarded or instantaneous"
        @assert acausal_ok(Σ_K) "Σ_K must be acausal or instantaneous"
    end
    G_R = solve_dyson(g, g * Σ_R)
    G_K = G_R * Σ_K * G_R'
    return (; G_R, G_K)
end

"""
    lead_current(G_R, G_K, Σ_R, Σ_K)

Average current through a lead in terms of the full Green functions and the
lead self-energies (Keldysh-symmetric representation):

`(1/2) tr[ -Σ_R G_R + (Σ_R G_R)' + G_R Σ_K + G_K Σ_R' - Σ_K G_R' - Σ_R G_K + (3/2)(G_R Σ_R - (G_R Σ_R)') ]`

Returns the operator whose block-diagonal is the time-resolved current; take
`diag(matrix(op))` (or the Keldysh trace for `bs > 1`) for the signal.
"""
function lead_current(G_R, G_K, Σ_R, Σ_K)
    ΣRG_R = Σ_R * G_R
    G_RΣ_R = G_R * Σ_R
    return -(
        (1 // 2) * (adjoint(ΣRG_R) - ΣRG_R) +
        G_R * Σ_K +
        G_K * adjoint(Σ_R) -
        (Σ_K * adjoint(G_R) + Σ_R * G_K) +
        (3 // 2) * (G_RΣ_R - adjoint(G_RΣ_R))
    )
end


export solve_keldysh, lead_current
end
