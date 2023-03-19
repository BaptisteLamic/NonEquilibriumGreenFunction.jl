

#Interception of standart AbstractArray method 
#=function ldiv(A::AbstractKernel,B::AbstractKernel)
    error("$(typeof(A)) \\ $(typeof(B)): not implemented")
end=#
function ldiv(A::TimeLocalKernel,B::AbstractKernel)
    @assert iscompatible(A,B)
    similar(B, matrix(A) \ matrix(B))
end
function ldiv(Aδ::TimeLocalKernel,Ar::RetardedKernel, B::RetardedKernel)
    @assert iscompatible(Ar,B)
    @assert iscompatible(Aδ,B)
    cp=compression(Ar)
    bs=blocksize(Ar)
    diag_Ar = extract_blockdiag(Ar |> matrix,bs, compression = cp)
    diag_B = extract_blockdiag(matrix(B),bs, compression = cp)
    A_op = cp(step(Ar)*( matrix(Ar) - 1//2 * diag_Ar) + matrix(Aδ) )
    sol_biased = similar(B,A_op\cp( matrix(B) - 1//2 * diag_B))
    correction = similar(B, matrix(Aδ) \ diag_B - extract_blockdiag(matrix(sol_biased),bs, compression = cp))
    return sol_biased + correction
end
function ldiv(A::RetardedKernel, B::RetardedKernel)
    @assert iscompatible(A,B)
    cp=compression(A)
    bs=blocksize(A)
    diag_A = extract_blockdiag(matrix(A),bs, compression = cp)
    diag_B = extract_blockdiag(matrix(B),bs, compression = cp)
    A_op = step(A)*(matrix(A) - 1//2 * diag_A)
    sol_biased = A_op\(matrix(B) - 1//2 * diag_B)
    correction = - extract_blockdiag(sol_biased,bs, compression = cp)
    return similar(A, sol_biased + correction)
end
function solve_dyson(g::AbstractKernel,k::AbstractKernel) 
    (I-k)\g
end