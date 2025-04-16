
function inv(k::Kernel)
    @assert isretarded(k)
    cp = compression(k)
    bs = blocksize(k)
    diag_k = extract_blockdiag(k |> matrix, bs, compression = cp)
    eye = cp(sparse(scalartype(k)(1)*I,size(diag_k)...))
    diag_eye = extract_blockdiag(eye, bs, compression = cp)
    right = cp( eye - 1 // 2 * diag_eye)
    #left = cp( scalartype(k)(step(k)) * (matrix(k) - 1//2 * diag_k) ) 
    right = eye
    left = matrix(k) * step(k)
    sol_biased = left\right
    #correction = cp( diag_eye - extract_blockdiag( sol_biased,bs, compression = cp) )
    #return similar(k, cp(sol_biased + correction) ) 
    return similar(k, sol_biased) 
end

"""
    Solve the equation  G = g + K⋅G  for G
"""
function solve_dyson(g::Kernel,K::Kernel)
    @assert isretarded(g) & isretarded(K)
    cp = compression(g)
    bs = blocksize(g)
    diag_K = extract_blockdiag(K |> matrix, bs, compression = cp)
    diag_g = extract_blockdiag(g |> matrix, bs, compression = cp)
    eye = cp(sparse(scalartype(K)(1)*I,size(diag_K)...)) #bypass limitation of HssMatrices.jl
    left = cp( eye - scalartype(K)(step(K)) * (matrix(K) - 1//2 * diag_K) )
    right = cp( matrix(g) -1 // 2 * diag_g)
    sol_biased = ldiv!(left,right)
    correction = diag_g - extract_blockdiag( sol_biased,bs, compression = cp)
    return similar(g, cp(sol_biased + correction) )
end