"""
    Solve the equation  G = g + K⋅G  for G
"""
function solve_dyson(g::Kernel, K::Kernel)
    @assert isretarded(g) & isretarded(K)
    cp = compression(g)
    bs = blocksize(g)
    N = length(axis(g))
    T = scalartype(K)
    blocks_K = blockdiag_blocks(matrix(K), bs)
    blocks_g = blockdiag_blocks(matrix(g), bs)
    eye = cp(repeat(Matrix{T}(I, bs, bs), 1, 1, N))
    left = eye - T(step(K)) * (matrix(K) - cp(blocks_K .* (1 // 2)))
    right = cp(matrix(g) - cp(blocks_g .* (1 // 2)))
    sol_biased = ldiv(left, right)
    blocks_corr = blocks_g - blockdiag_blocks(sol_biased, bs)
    return make_similar(g, cp(sol_biased + cp(blocks_corr)))
end
