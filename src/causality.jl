"""
    AbstractCausality

Abstract type for causality classifications.

Causality determines the time-ordering support of Green's functions:

- `Retarded`: G(t, t') = 0 for t < t' (causal, future doesn't affect past)
- `Advanced`: G(t, t') = 0 for t > t' (anti-causal, past doesn't affect future)
- `Acausal`: G(t, t') ≠ 0 for all t, t' (no time ordering constraint)

Operators supported on the diagonal `t = t'` (contact terms) are *not* a
causality: they satisfy all three constraints simultaneously and are
classified on the locality axis instead (see [`AbstractLocality`](@ref)).
Their canonical causality is `Acausal`.
"""
abstract type AbstractCausality end

"""
    Retarded <: AbstractCausality

Retarded causality: the Green's function is zero for t < t'.

This represents causal propagation where the future cannot affect the past.
"""
struct Retarded <: AbstractCausality end

"""
    Acausal <: AbstractCausality

Acausal causality: the Green's function has no time-ordering constraints.

It can be non-zero for all time combinations.
"""
struct Acausal <: AbstractCausality end

"""
    Advanced <: AbstractCausality

Advanced causality: the Green's function is zero for t > t'.

This represents anti-causal propagation where the past cannot affect the future.
"""
struct Advanced <: AbstractCausality end

"""
    causality_of_sum(::C, ::C) where {C<:AbstractCausality}

The sum of two operators with the same causality retains that causality.
"""
causality_of_sum(::C, ::C) where {C<:AbstractCausality} = C()

"""
    causality_of_sum(::AbstractCausality, ::AbstractCausality)

The sum of two operators with different causalities is Acausal.
"""
causality_of_sum(::AbstractCausality, ::AbstractCausality) = Acausal()

"""
    causality_of_prod(::Retarded, ::Retarded)

The product of two Retarded operators is Retarded.
"""
causality_of_prod(::Retarded, ::Retarded) = Retarded()

"""
    causality_of_prod(::Advanced, ::Advanced)

The product of two Advanced operators is Advanced.
"""
causality_of_prod(::Advanced, ::Advanced) = Advanced()

"""
    causality_of_prod(::Retarded, ::Acausal)

The product of Retarded and Acausal is Acausal.
"""
causality_of_prod(::Retarded, ::Acausal) = Acausal()

"""
    causality_of_prod(::Acausal, ::Advanced)

The product of Acausal and Advanced is Acausal.
"""
causality_of_prod(::Acausal, ::Advanced) = Acausal()

"""
    causality_of_prod(::Acausal, ::Acausal)

The product of two Acausal operators is Acausal.
"""
causality_of_prod(::Acausal, ::Acausal) = Acausal()
