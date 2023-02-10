function ldiv(A::AbstractKernel,B::AbstractKernel)
    return A\B
end

#Interception of standart AbstractArray method 
function \(A::AbstractKernel,B::AbstractKernel)
    error("$(typeof(A)) \\ $(typeof(B)): not implemented")
end

function \(A::TimeLocalKernel,B::AbstractKernel)
    @assert iscompatible(A,B)
    similar(B, matrix(A) \ matrix(B))
end
#=
function \(A::TimeLocalKernel,B::SumKernel)
    @assert iscompatible(A,B)
    A\B.kernelL + A\B.kernelR
end
=#
#=
function \(A::AbstractKernel,B::NullKernel)
    #We should check that A is non zero
    return B
end
=#
#=
function \(A::SumKernel,B::RetardedKernel)
    @assert iscompatible(A,B)
    @assert isretarded(A)
    cp=compression(A)
    bs=blocksize(A)
    Aδ = timelocal_part(A)
    Ar = nonlocal_part(A)
    diag_Ar = extract_blockdiag(Ar |> matrix,bs, compression = cp)
    diag_B = extract_blockdiag(matrix(B),bs, compression = cp)
    A_op = cp(step(Ar)*( matrix(Ar) - 1//2 * diag_Ar) + matrix(Aδ) )
    sol_biased = similar(B,A_op\cp( matrix(B) - 1//2 * diag_B))
    correction = similar(B, matrix(Aδ) \ diag_B - extract_blockdiag(matrix(sol_biased),bs, compression = cp))
    return sol_biased + correction
end
=#
#=
function \(A::SumKernel,B::TimeLocalKernel)
    @assert iscompatible(A,B)
    @assert isretarded(A)
    Aδ = timelocal_part(A)
    Ar = nonlocal_part(A)
    Xδ = Aδ \ B
    Xr = -A\(Ar*Xδ)
    return Xδ + Xr
end
=#
function \(A::RetardedKernel, B::RetardedKernel)
    @assert iscompatible(A,B)
    cp=compression(A)
    bs=blocksize(A)
    diag_A = extract_blockdiag(matrix(A),bs, compression = cp)
    diag_B = extract_blockdiag(matrix(B),bs, compression = cp)
    A_op = step(A)*(matrix(A) - 1//2 * diag_A)
    sol_biased = similar(B,A_op\(matrix(B) - 1//2 * diag_B))
    correction = - similar(B,extract_blockdiag(matrix(sol_biased),bs, compression = cp))
    return sol_biased + correction
end
#=
function \(A::AbstractKernel,B::SumKernel)
    return A\B.kernelL + A\B.kernelR
end
=#
function solve_dyson(g::AbstractKernel,k::AbstractKernel) 
    (I-k)\g
end