
function solve_dyson(g::RetardedGreenFunction{A,D,R},k::RetardedGreenFunction{A,D,R}; Compression = HssCompression()) where {A,D,R}
    @assert iscompatible(g,k)
    T = eltype(g)
    Gδ =  _solve_dyson_singular(dirac(g),dirac(k))
    s = regular(g) + _prod(regular(k),Gδ)
    Gr = cc_solve_dyson(s,regular(k),dirac(k),axis(g),blocksize(g), Compression = Compression)
    G = RetardedGreenFunction(axis(g),Gδ, Gr, blocksize(g))
    return G
end

function cc_solve_dyson(g::AbstractArray{T,2},k::AbstractArray{T,2}, axis, bs ; Compression = HssCompression()) where T
    cc_solve_dyson(g,k,zeros(T,bs,bs,length(axis) ) , axis, bs ; Compression = HssCompression())
end

function cc_solve_dyson(g::AbstractArray{T,2},k::AbstractArray{T,2}, k_δ::AbstractArray{T,3}, axis, bs ; Compression = HssCompression()) where T
    N = length(axis)*bs
    id = sparse(LinearAlgebra.I,N,N) .|> T |> Compression
    G, bd_g = _integral_operator(g, bs, Compression = Compression)
    K, bd_k = _integral_operator(k, bs, Compression = Compression)
    m_k_δ = blockdiag(k_δ, Compression = Compression)
    K = T(step( axis ))*K+m_k_δ
    S =  (id-K)\G
    #Let's correct the diagonal:
    ds = extract_blockdiag(S, bs, Compression = Compression)
    stt = (id-m_k_δ)\bd_g
    s = S - ds + stt
    Compression(s)
    return s
end

function _solve_dyson_singular(g::AbstractArray{T,3},k::AbstractArray{T,3}) where T
    sol = similar(g)
    Threads.@threads for i = 1:size(g,3)
        sol[:,:,i] = (I- k[:,:,i]) \  g[:,:,i]
    end
    return sol
end
