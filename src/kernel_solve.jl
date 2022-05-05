#Interception of standart AbstractArray method 
function \(A::AbstractKernel,B::AbstractKernel)
    error("$(typeof(A)) \\ $(typeof(B)): not implemented")
end

function \(A::TimeLocalKernel,B::AbstractKernel)
    @assert iscompatible(A,B)
    similar(B, A.matrix \ B.matrix)
end
function \(A::TimeLocalKernel,B::SumKernel)
    @assert iscompatible(A,B)
    A\B.kernelL + A\B.kernelR
end


function \(A::SumKernel,B::TimeLocalKernel)
    @assert iscompatible(A,B)
    @assert isretarded(A) && isretarded(B)
    Ar = nonlocal_part(A)
    Ad = timelocal_part(A)
    Xd = Ad\B
    Xr = -(Xd + (Ad\Ar)*Xd)
    error("code broken")
    return similar(A,Xr,Xd)
end

function solve_dyson(g,k::RetardedKernel)
    solve_dyson(g ,k + 0*I)
end
function solve_dyson(g::RetardedKernel,k::SumKernel)
    solve_dyson(g + 0*I, k)
end
function solve_dyson(g::SumKernel,k::SumKernel)
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

    diag_g = extract_blockdiag(gr.matrix,bs, compression = cp)
    diag_k = extract_blockdiag(kr.matrix,bs, compression = cp)

    #Due to a bogue in HssMatrix we must introduce a identity matrix
    N = length(axis(g))*bs
    _Id = sparse(LinearAlgebra.I,N,N) .|> eltype(diag_g) |> cp

    k_op = step(k)*(kr.matrix - 1//2 * diag_k) + kd.matrix
    Gr = (_Id - k_op)\(gr.matrix - 1//2 * diag_g)

    diag_Gr = extract_blockdiag(Gr, bs, compression = cp)
    correction_Gr = (_Id - kd.matrix)\diag_g
    Gr = similar(gr,Gr-diag_Gr + correction_Gr)
    return Gr + Gd
end