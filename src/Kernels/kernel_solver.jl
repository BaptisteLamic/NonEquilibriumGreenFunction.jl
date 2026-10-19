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
    b = backend(cp)
    eye = cp(repeat(Matrix{T}(I, bs, bs), 1, 1, N))
    K_dressed = evaluate(b, op_sub(CompressedLeaf(matrix(K)), CompressedLeaf(cp(blocks_K .* (1 // 2)))))
    left = evaluate(b, op_sub(CompressedLeaf(eye), op_scale(T(step(K)), CompressedLeaf(K_dressed))))
    right = evaluate(b, op_sub(CompressedLeaf(matrix(g)), CompressedLeaf(cp(blocks_g .* (1 // 2)))))
    sol_biased = evaluate(b, SolveNode(b, CompressedLeaf(left), CompressedLeaf(right), DirectSolve()))
    blocks_corr = blocks_g - blockdiag_blocks(sol_biased, bs)
    return make_similar(g, evaluate(b, op_add(CompressedLeaf(sol_biased), CompressedLeaf(cp(blocks_corr)))))
end
