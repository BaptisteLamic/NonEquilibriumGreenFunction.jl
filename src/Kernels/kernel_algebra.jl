function +(left::Kernel, right::Kernel)
    Kernel(discretization(left) + discretization(right), causality_of_sum(left |> causality, right |> causality)) |> compress!
end

function +(left::D, right::D) where {D<:AbstractDiscretisation}
    similar(left, matrix(left) + matrix(right))
end
function -(left::Kernel, right::Kernel)
    Kernel(discretization(left) - discretization(right), causality_of_sum(left |> causality, right |> causality)) |> compress!
end
function -(left::D, right::D) where {D<:AbstractDiscretisation}
    similar(left, matrix(left) - matrix(right))
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
    return similar(left, result)
end
function prod(::Acausal, ::Advanced, left::AbstractDiscretisation, right::AbstractDiscretisation)
    dr = extract_blockdiag(matrix(right), blocksize(right))
    weighted_R = _dressing(right, dr)
    result = step(left)*matrix(left) * weighted_R
    return similar(left, result)
end
function prod(::Retarded, ::Acausal, left::AbstractDiscretisation, right::AbstractDiscretisation)
    dl = extract_blockdiag(matrix(left), blocksize(left))
    weighted_L = _dressing(left, dl) 
    result = step(left)*weighted_L * matrix(right)
    return similar(left, result)
end
function prod(::T, ::T, left::AbstractDiscretisation, right::AbstractDiscretisation) where T <: Union{Instantaneous,Acausal}
    result = step(left)*matrix(left) * matrix(right)
    return similar(left, result)
end

function adjoint(kernel::Kernel)
    _new_causality(::Retarded) = Advanced()
    _new_causality(::Advanced) = Retarded()
    _new_causality(::Acausal) = Acausal()
    return Kernel(discretization(kernel)', kernel |> causality |> _new_causality )
end