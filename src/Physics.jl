module Physics

using LinearAlgebra: diagm, Matrix, I, Diagonal, diag
using ..NonEquilibriumGreenFunction: polygamma
import ..NonEquilibriumGreenFunction.Kernels: Kernel, solve_dyson, causality, isretarded, isacausal,
    adjoint, keldysh_trace
import ..NonEquilibriumGreenFunction: islocal
import Base: *

include("Physics/physics.jl")
include("Physics/self_energy.jl")

"""
    solve_keldysh(g, Σ_R, Σ_K; check=true)

Solve the Dyson equation for the retarded branch and dress the kinetic branch.

`g` is the bare retarded Green function, `Σ_R` the retarded self-energy and
`Σ_K` the Keldysh self-energy. Returns `(; G_R, G_K)` where
`G_R = solve_dyson(g, g * Σ_R)` and `G_K = G_R * Σ_K * G_R'`.

`Σ_R` and `Σ_K` are the *total* self-energies of the central region (sum over
all leads); pass the lead-only self-energies to `lead_current`, not here.

When `check` is true the causalities are validated: `g` and `Σ_R` must be
retarded or instantaneous and `Σ_K` acausal or instantaneous.
"""
function solve_keldysh(g::Kernel, Σ_R, Σ_K; check=true)
    if check
        retarded_ok(x) = isretarded(x) || islocal(x)
        acausal_ok(x) = isacausal(x) || islocal(x)
        @assert retarded_ok(g) "g must be retarded or instantaneous"
        @assert retarded_ok(Σ_R) "Σ_R must be retarded or instantaneous"
        @assert acausal_ok(Σ_K) "Σ_K must be acausal or instantaneous"
    end
    G_R = solve_dyson(g, g * Σ_R)
    G_K = G_R * Σ_K * G_R'
    return (; G_R, G_K)
end

"""
    lead_current(G_R, G_K, Σ_R, Σ_K; check=true)

Average current through a lead in terms of the full Green functions and the
*lead-only* self-energies, in the Keldysh-symmetric representation (see Jauho,
Haug & Wilkins, and the package examples for the derivation):

`-[ (1/2)((Σ_R G_R)' - Σ_R G_R) + G_R Σ_K + G_K Σ_R' - (Σ_K G_R' + Σ_R G_K) + (3/2)(G_R Σ_R - (G_R Σ_R)') ]`

Returns the operator whose block-diagonal encodes the time-resolved current;
extract the physical signal with `current_signal` (do not read `diag` of the raw
matrix for `bs > 1`).

Convention: positive current flows *into* the lead. `G_R`/`G_K` are the full
(retarded/kinetic) Green functions of the central region dressed with the
total self-energy (e.g. from `solve_keldysh`), while `Σ_R`/`Σ_K` here are the
self-energies of the considered lead only.

When `check` is true the causalities are validated: `G_R` must be retarded,
`G_K` and `Σ_K` acausal-or-instantaneous, and `Σ_R` retarded-or-instantaneous.
"""
function lead_current(G_R, G_K, Σ_R, Σ_K; check=true)
    if check
        retarded_ok(x) = isretarded(x) || islocal(x)
        acausal_ok(x) = isacausal(x) || islocal(x)
        @assert isretarded(G_R) "G_R must be retarded"
        @assert retarded_ok(Σ_R) "Σ_R must be retarded or instantaneous"
        @assert acausal_ok(G_K) "G_K must be acausal or instantaneous"
        @assert acausal_ok(Σ_K) "Σ_K must be acausal or instantaneous"
    end
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

"""
    current_signal(op)

Time-resolved signal of a current/observable operator: the trace over the
Keldysh indices of each same-time block, one value per time step. For
`bs == 1` this is `diag(matrix(op))`.

`lead_current` returns an operator; use this function to extract the
physical signal from it.
"""
current_signal(op) = keldysh_trace(op)


export solve_keldysh, lead_current, current_signal

export second_born_self_energy, hartree_fock_self_energy, lesser_greater
end
