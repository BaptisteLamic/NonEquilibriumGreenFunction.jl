function +(left::Kernel, right::Kernel)
    Kernel(discretization(left) + discretization(right), causality_of_sum(left |> causality, right |> causality))
end
function causality_of_sum(left::C, right::C) where {C<:AbstractCausality}
    @assert left == right
    return left
end
function +(left::D, right::D) where {D<:AbstractDiscretisation}
    similar(left, matrix(left) + matrix(right))
end
function -(left::Kernel, right::Kernel)
    Kernel(discretization(left) - discretization(right), causality_of_sum(left |> causality, right |> causality))
end
function -(left::D, right::D) where {D<:AbstractDiscretisation}
    similar(left, matrix(left) - matrix(right))
end

function +(left::UniformScaling, right::Kernel)
    left*dirac(right) + right
end
function -(left::UniformScaling, right::Kernel)
    left*dirac(right) - right
end
function +(left::Kernel, right::UniformScaling)
    left + right*dirac(left) 
end
function -(left::Kernel, right::UniformScaling)
    left - right*dirac(left) 
end

-(discretization::AbstractDiscretisation) = similar(discretization, -matrix(discretization))
-(kernel::Kernel) = Kernel(-discretization(kernel), causality(kernel))

function *(λ::Number, discretization::AbstractDiscretisation)
    return similar(discretization, λ * matrix(discretization))
end
*(λ::Number, kernel::Kernel) = Kernel(λ * discretization(kernel), causality(kernel))
*(kernel::Kernel, λ::Number) = λ * kernel
*(scaling::UniformScaling, kernel::Kernel) = scaling.λ * kernel
*(kernel::Kernel, scaling::UniformScaling) = scaling.λ * kernel


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
causality_of_prod(::Retarded, ::Retarded) = Retarded()
causality_of_prod(::Advanced, ::Advanced) = Advanced()
causality_of_prod(::Retarded, ::Acausal) = Acausal()
causality_of_prod(::Acausal, ::Advanced) = Acausal()
causality_of_prod(::Acausal, ::Acausal) = Acausal()

function _dressing(g::TrapzDiscretisation, d)
     return matrix(g) - eltype(d)(0.5) * d
end
function _biased_mul(::C, ::C, gl::TrapzDiscretisation, gr::TrapzDiscretisation) where {C<:Union{Retarded,Advanced}}
    bs = blocksize(gl)
    dl = extract_blockdiag(matrix(gl), bs, compression=compression(gl))
    dr = extract_blockdiag(matrix(gr), bs, compression=compression(gl))
    weighted_L = _dressing(gl, dl)
    weighted_R = _dressing(gr, dr)
    return weighted_L * weighted_R, dl, dr
end
function prod(c_left::C, c_right::C, left::AbstractDiscretisation, right::AbstractDiscretisation) where {C<:Union{Retarded,Advanced}}
    biased_result, dl, dr = _biased_mul(c_left, c_right, left, right)
    result = biased_result - (1/4)*dl*dr
    result = step(left)*result
    return similar(left, result)
end
function prod(::Acausal, ::Advanced, left::AbstractDiscretisation, right::AbstractDiscretisation)
    dr = extract_blockdiag(matrix(right), blocksize(right), compression=compression(right))
    weighted_R = _dressing(right, dr)
    result = step(left)*matrix(left) * weighted_R
    return similar(left, result)
end
function prod(::Retarded, ::Acausal, left::AbstractDiscretisation, right::AbstractDiscretisation)
    dl = extract_blockdiag(matrix(left), blocksize(left), compression=compression(left))
    weighted_L = _dressing(left, dl)
    result = step(left)*weighted_L * matrix(right)
    return similar(left, result)
end
function prod(::Acausal, ::Acausal, left::AbstractDiscretisation, right::AbstractDiscretisation)
    result = step(left)*matrix(left) * matrix(right)
    return similar(left, result)
end
