export AbstractQuadrature, RectangleQuadrature, TrapezoidQuadrature, quadrature

"""
    AbstractQuadrature

First-class quadrature rule for the time integrals of the kernel algebra.

A product ``(K L)(t, t') = \\int K(t, t'') L(t'', t')\\, dt''`` is
discretized by a *1D* rule along the contraction variable `t''` for each
`(t, t')`. Making the rule a structural part of the discretization (rather
than hard-coded weights) keeps the product compositional while letting the
rule own three concerns:

1. **Interior weights** — the bulk quadrature weights along `t''`.
2. **Diagonal (collapse) weight** — when a row's integration interval
   degenerates (`t'' = t` or `t'' = t'`, e.g. the first row of a retarded
   kernel), the weight of the collapse point is a decision of the rule,
   not a matrix entry.
3. **Boundary policy** — how the rule treats the domain edges, where the
   integration line collapses to a single point; kernels that do not
   vanish at the boundary must not leak through the circulant wraparound.

Concrete implementations: [`RectangleQuadrature`](@ref) (left rule; the
historical behaviour) and [`TrapezoidQuadrature`](@ref) (second order for
smooth kernels).
"""
abstract type AbstractQuadrature end

"""
    RectangleQuadrature() <: AbstractQuadrature

Left-point rectangle rule: every node carries weight `\\delta t`,
including the endpoints. This is the rule the package historically used
(`\\delta t \\cdot M_L M_R`), so results are bit-for-bit identical to the
pre-quadrature-refactor behaviour. First-order accurate on smooth
kernels; the degenerate boundary row is handled by the causal masking.
"""
struct RectangleQuadrature <: AbstractQuadrature end

"""
    TrapezoidQuadrature() <: AbstractQuadrature

Trapezoidal rule: interior nodes carry weight `\\delta t`, the two
endpoints of each integration interval carry `\\delta t/2`. Second-order
accurate for smooth kernels, for any blocksize, including kernels that
do not vanish at the domain boundary (the half-weights encode the
boundary line collapse instead of relying on wraparound cancellation).

Second order for acausal × acausal (edge weights on the full axis),
for retarded × acausal and acausal × advanced (half weight on the
domain-edge node of the moving interval, exact zero on the degenerate
boundary line), and — via the diagonal dressing, which is the trapezoid
treatment of the moving interval — for retarded × retarded and
advanced × advanced under both rules.
"""
struct TrapezoidQuadrature <: AbstractQuadrature end

"""
    quadrature(dis::AbstractDiscretisation)

Return the quadrature rule of a discretization. Every discretization
carries its rule structurally; fallback for foreign types is
[`RectangleQuadrature`](@ref) (the historical behaviour).
"""
function quadrature end

quadrature(::Any) = RectangleQuadrature()

"""
    edge_weights(q::AbstractQuadrature)

Per-node weights for the full-domain acausal quadrature, indexed like the
axis. Interior nodes carry `1`; the two domain-edge nodes carry the rule's
boundary weight (the line-collapse correction). The product weights are
`δt .* edge_weights`.
"""
function edge_weights end

edge_weights(::RectangleQuadrature, N) = ones(N)

function edge_weights(::TrapezoidQuadrature, N)
    w = ones(N)
    w[1] = 0.5
    w[end] = 0.5
    return w
end
