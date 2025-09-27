


export kernelSolve

"""
    Solve the equation  G = g + K⋅G  for G
"""
function solve_dyson(g::Kernel,K::Kernel)
    @assert isretarded(g) & isretarded(K)
    cp = compression(g)
    bs = blocksize(g)
    diag_K = extract_blockdiag(K |> matrix, bs)
    diag_g = extract_blockdiag(g |> matrix, bs)
    eye = cp(sparse(scalartype(K)(1)*I,size(diag_K)...)) #bypass limitation of HssMatrices.jl
    left = cp( eye - scalartype(K)(step(K)) * (matrix(K) - cp(1//2 * diag_K)) )
    right = cp( matrix(g) - cp(1 // 2 * diag_g))
    sol_biased = _ldiv!(left,right)
    correction = cp(diag_g - extract_blockdiag( sol_biased,bs))
    return similar(g, cp(sol_biased + correction) )
end

function _ldiv!(left::T,right::T) where {T<:AbstractMatrix}
    return left \ right
end
function _ldiv!(left::HssMatrix,right::HssMatrix)
    return ldiv!(left, right)
end