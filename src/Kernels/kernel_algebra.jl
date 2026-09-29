function +(left::Kernel, right::Kernel)
    Kernel(discretization(left) + discretization(right), causality_of_sum(left |> causality, right |> causality)) |> compress!
end

function +(left::D, right::D) where {D<:AbstractDiscretisation}
    make_similar(left, matrix(left) + matrix(right))
end
function -(left::Kernel, right::Kernel)
    Kernel(discretization(left) - discretization(right), causality_of_sum(left |> causality, right |> causality)) |> compress!
end
function -(left::D, right::D) where {D<:AbstractDiscretisation}
    make_similar(left, matrix(left) - matrix(right))
end

function *(left::Kernel, right::Kernel)
    prod_causality = causality_of_prod(left |> causality, right |> causality)
    result_dis = prod(
        left |> causality,
        right |> causality,
        left |> discretization,
        right |> discretization
    )
    return Kernel(
        result_dis,
        prod_causality
    )
end

function _dressing(g::TrapzDiscretisation, d)
     return matrix(g) - compression(g)(eltype(d)(0.5) * d)
end
function _biased_mul(::C, ::C, gl::TrapzDiscretisation, gr::TrapzDiscretisation) where {C<:Union{Retarded,Advanced}}
    bs = blocksize(gl)
    dl = extract_blockdiag(matrix(gl), bs)
    dr = extract_blockdiag(matrix(gr), bs)
    weighted_L = _dressing(gl, dl)
    weighted_R = _dressing(gr, dr)
    return weighted_L * weighted_R, dl, dr
end
function prod(c_left::C, c_right::C, left::AbstractDiscretisation, right::AbstractDiscretisation) where {C<:Union{Retarded,Advanced}}
    biased_result, dl, dr = _biased_mul(c_left, c_right, left, right)
    result = biased_result - compression(left)((1/4)*dl*dr)
    result = step(left)*result
    return make_similar(left, result)
end
function prod(::Acausal, ::Advanced, left::AbstractDiscretisation, right::AbstractDiscretisation)
    dr = extract_blockdiag(matrix(right), blocksize(right))
    weighted_R = _dressing(right, dr)
    weighted_R = _acausal_advanced_edges(quadrature(left), right, weighted_R)
    result = step(left)*matrix(left) * weighted_R
    return make_similar(left, result)
end

"""
    _boundary_blockdiag(dis, firstblock)

Block-diagonal matrix with the given `bs×bs` first block and identity
elsewhere, in the representation of the discretization's compression.
Used by the quadrature rules to weight the `t'' = t₀` domain edge of a
moving integration interval and to zero degenerate boundary lines.
"""
function _boundary_blockdiag(dis, firstblock)
    bs = blocksize(dis)
    N = length(axis(dis))
    T = scalartype(dis)
    blocks = Array{T,3}(undef, bs, bs, N)
    for i in 2:N
        blocks[:, :, i] .= Matrix{T}(I, bs, bs)
    end
    blocks[:, :, 1] .= firstblock
    return build_blockdiag(blocks; compression=compression(dis))
end

# Rectangle rule: historical behaviour, no edge correction.
_acausal_advanced_edges(::RectangleQuadrature, right, weighted_R) = weighted_R

# Trapezoid rule: the integration interval t'' ∈ [t₀, t'] has its domain
# edge at t₀ (half weight on the first row-block of the dressed right
# kernel) and degenerates to a point at t' = t₀ (the first column-block
# of the product vanishes exactly).
function _acausal_advanced_edges(::TrapezoidQuadrature, right, weighted_R)
    bs = blocksize(right)
    half = _boundary_blockdiag(right, 0.5 * Matrix{scalartype(right)}(I, bs, bs))
    zer = _boundary_blockdiag(right, zeros(scalartype(right), bs, bs))
    return half * weighted_R * zer
end
function prod(::Retarded, ::Acausal, left::AbstractDiscretisation, right::AbstractDiscretisation)
    dl = extract_blockdiag(matrix(left), blocksize(left))
    weighted_L = _dressing(left, dl)
    weighted_L = _retarded_acausal_edges(quadrature(left), left, weighted_L)
    result = step(left)*weighted_L * matrix(right)
    return make_similar(left, result)
end

# Rectangle rule: historical behaviour, no edge correction.
_retarded_acausal_edges(::RectangleQuadrature, left, weighted_L) = weighted_L

# Trapezoid rule: the integration interval t'' ∈ [t₀, t] has its domain
# edge at t₀ (half weight on the first column-block of the dressed left
# kernel) and degenerates to a point at t = t₀ (the first row-block of
# the product vanishes exactly).
function _retarded_acausal_edges(::TrapezoidQuadrature, left, weighted_L)
    bs = blocksize(left)
    half = _boundary_blockdiag(left, 0.5 * Matrix{scalartype(left)}(I, bs, bs))
    zer = _boundary_blockdiag(left, zeros(scalartype(left), bs, bs))
    return zer * weighted_L * half
end
"""
    prod(::Acausal, ::Acausal, left, right)

Discretized product of two acausal kernels: a quadrature-weighted matrix
product whose weights are owned by the discretization's quadrature rule
([`AbstractQuadrature`](@ref)). With `RectangleQuadrature` (the default)
this is the historical `δt * M_L * M_R`; with `TrapezoidQuadrature` the
two domain-edge nodes carry half weights, which restores second-order
convergence for smooth kernels at any blocksize. Products involving
`Local` operators never reach this path; they are applied exactly by the
operator-level `*` methods.
"""
function prod(::Acausal, ::Acausal, left::AbstractDiscretisation, right::AbstractDiscretisation)
    result = _quadrature_prod(quadrature(left), left, right)
    return make_similar(left, result)
end

"""
    _quadrature_prod(q::AbstractQuadrature, left, right)

Quadrature-weighted product `M_L · W · M_R` with
`W = diag(δt .* edge_weights(q, N))` applied along the contraction
variable `t''` (columns of the left matrix, rows of the right one). For
the rectangle rule all weights are 1, recovering the historical
`δt * M_L * M_R`.
"""
function _quadrature_prod(q::AbstractQuadrature, left::AbstractDiscretisation, right::AbstractDiscretisation)
    ML, MR = matrix(left), matrix(right)
    bs = blocksize(left)
    n = length(axis(left))
    w = step(left) .* edge_weights(q, n)
    # W on the contraction variable: scale the row-blocks of MR
    WR = _scale_rowblocks(MR, w, bs, n)
    return ML * WR
end

_quadrature_prod(::RectangleQuadrature, left::AbstractDiscretisation, right::AbstractDiscretisation) =
    step(left) * matrix(left) * matrix(right)

function _scale_rowblocks(M, w, bs, n)
    Mr = copy(M)
    for i in 1:n
        Mr[blockrange(i, bs), :] .*= w[i]
    end
    return Mr
end

function adjoint(kernel::Kernel)
    _new_causality(::Retarded) = Advanced()
    _new_causality(::Advanced) = Retarded()
    _new_causality(::Acausal) = Acausal()
    return Kernel(discretization(kernel)', kernel |> causality |> _new_causality )
end