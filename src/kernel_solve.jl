#Interception of standart AbstractArray method 
function \(A::AbstractKernel,B::AbstractKernel)
    error("$(typeof(A)) \\ $(typeof(B)): not implemented")
end

function \(A::TimeLocalKernel,B::AbstractKernel)
    @assert iscompatible(A,B)
    similar(B, A.matrix \ B.matrix)
end
function \(A::TimeLocalKernel,B::NullKernel)
    @assert iscompatible(A,B)
    NullKernel(A)
end
function \(A::TimeLocalKernel,B::SumKernel)
    @assert iscompatible(A,B)
    A\B.kernelL + A\B.kernelR
end


function \(A::AbstractKernel,B::NullKernel)
    #We should check that A is non zero
    return B
end

function \(A::SumKernel,B::RetardedKernel)
    @assert iscompatible(A,B)
    @assert isretarded(A)
    cp=compression(A)
    bs=blocksize(A)
    Aδ = timelocal_part(A)
    Ar = nonlocal_part(A)
    diag_Ar = extract_blockdiag(Ar.matrix,bs, compression = cp)
    diag_B = extract_blockdiag(B.matrix,bs, compression = cp)
    A_op = step(Ar)*(Ar.matrix - 1//2 * diag_Ar) + Aδ.matrix
    sol_biased = similar(B,A_op\(B.matrix - 1//2 * diag_B))
    correction = similar(B,Aδ.matrix \ diag_B - extract_blockdiag(sol_biased.matrix,bs, compression = cp))
    return sol_biased + correction
end
function \(A::SumKernel,B::TimeLocalKernel)
    @assert iscompatible(A,B)
    @assert isretarded(A)
    cp=compression(A)
    bs=blocksize(A)
    Aδ = timelocal_part(A)
    Ar = nonlocal_part(A)
    Xδ = Aδ \ B
    Xr = -A\(Ar*Xδ)
    return Xδ + Xr
end

function \(A::RetardedKernel,B::TimeLocalKernel)
    @assert iscompatible(A,B)
    return NullKernel(A)
end
function \(A::RetardedKernel, B::RetardedKernel)
    @assert iscompatible(A,B)
    cp=compression(A)
    bs=blocksize(A)
    diag_A = extract_blockdiag(A.matrix,bs, compression = cp)
    diag_B = extract_blockdiag(B.matrix,bs, compression = cp)
    A_op = step(A)*(A.matrix - 1//2 * diag_A)
    sol_biased = similar(B,A_op\(B.matrix - 1//2 * diag_B))
    correction = - similar(B,extract_blockdiag(sol_biased.matrix,bs, compression = cp))
    return sol_biased + correction
end

function \(A::AbstractKernel,B::SumKernel)
    return A\B.kernelL + A\B.kernelR
end

function solve_dyson(g::AbstractKernel,k::AbstractKernel) 
    (I-k)\g
end
#=
function solve_dyson(g::AbstractKernel,k::AbstractKernel)
    @assert iscompatible(g,k)
    @assert isretarded(g) && isretarded(k)
    cp = compression(g)
    bs = blocksize(g)

    gd = timelocal_part(g)
    gr = nonlocal_part(g)
    kd = timelocal_part(k)
    kr = nonlocal_part(k)
    Gd = (I-kd)\gd
    s = gr+kr*Gd

    N = length(axis(g))*bs
    _Id = sparse(LinearAlgebra.I,N,N) .|> eltype(g[1,1]) |> cp

    diag_s = extract_blockdiag(s.matrix,bs, compression = cp)
    #Due to a bogue in HssMatrix we must introduce a identity matrix

    k_op = if isa(kr, NullKernel)
        kd.matrix
    else
        diag_k = extract_blockdiag(kr.matrix,bs, compression = cp)
        step(k)*(kr.matrix - 1//2 * diag_k) + kd.matrix
    end
    Gr = (_Id - k_op)\(s.matrix - 1//2 * diag_s)

    diag_Gr = extract_blockdiag(Gr, bs, compression = cp)
    correction_Gr = (_Id - kd.matrix)\diag_s
    Gr = similar(gr,Gr-diag_Gr + correction_Gr)
    return Gr + Gd
end
=#