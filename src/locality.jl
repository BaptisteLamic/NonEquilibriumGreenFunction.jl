"""
    AbstractLocality

Abstract type for the locality axis, which classifies *how* an operator
acts on the time grid, independently of its support (causality):

- `Local`: distributions supported on the diagonal `t = t'` (Dirac contact
  terms, and by extension differential stencils). Local operators are
  applied exactly: they never enter the quadrature.
- `Smooth`: regular kernels, integrated by the quadrature rule.

The locality axis composes through the algebra via `locality_of_prod`
and `locality_of_sum`.
"""
abstract type AbstractLocality end

"""
    locality(op)

Returns the locality of an operator: `Local` for contact terms applied
exactly (never through the quadrature), `Smooth` for regular kernels.
Implemented for `LocalKernel`, `Kernel` and `SumOperator` in the `Kernels`
module.
"""
function locality end

"""
    Local <: AbstractLocality

Locality of operators supported on the time diagonal: contact terms
(proportional to `δ(t - t')`) and their differential generalizations.
`Local` is the unit of the kernel algebra: composing a `Local` operator
with any other operator leaves the result's causality and locality
unchanged. A `Local` operator satisfies every support constraint, so it
is simultaneously retarded, advanced and acausal; the canonical
representative used by the algebra is `Acausal`.
"""
struct Local <: AbstractLocality end

"""
    Smooth <: AbstractLocality

Locality of regular kernels: operators acting between distinct time
steps, integrated by the discretization's quadrature rule.
"""
struct Smooth <: AbstractLocality end

"""
    locality_of_prod(left::AbstractLocality, right::AbstractLocality)

Locality of a product of operators. `Local` is the composition unit;
`Smooth` absorbs it.
"""
locality_of_prod(::Local, ::Local) = Local()
locality_of_prod(::Local, ::Smooth) = Smooth()
locality_of_prod(::Smooth, ::Local) = Smooth()
locality_of_prod(::Smooth, ::Smooth) = Smooth()

"""
    locality_of_sum(left::AbstractLocality, right::AbstractLocality)

Locality of a sum of operators: the sum is `Local` only if both terms are.
"""
locality_of_sum(::Local, ::Local) = Local()
locality_of_sum(::AbstractLocality, ::AbstractLocality) = Smooth()

"""
    islocal(op)

Returns `true` if the operator is `Local` (a contact term applied exactly,
never through the quadrature).
"""
islocal(op) = locality(op) == Local()
