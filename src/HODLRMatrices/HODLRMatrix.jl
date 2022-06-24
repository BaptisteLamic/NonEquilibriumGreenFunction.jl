
struct LowRankMatrix{T<:Number,ST<:Number} <: AbstractMatrix{T}
    U::Matrix{T}
    S::Matrix{ST}
    V::Matrix{T}
end
Base.size(A::LowRankMatrix) = (size(A.U,1),size(A.V,1))
Base.getindex(A::LowRankMatrix, i, j::Int) = getindex(A,i,[j])
Base.getindex(A::LowRankMatrix, i::Int, j) = getindex(A,[i],j)
function Base.getindex(A::LowRankMatrix,i,j)
    U, S, V = A.U, A.S, A.V
    return @views U[i,:]*S*V[j,:]'
end
function Base.getindex(A::LowRankMatrix,i::Int,j::Int)
    U, S, V = A.U, A.S, A.V
    return @views transpose(U[i,:])*S*conj.(V[j,:])
end
function rank(A::LowRankMatrix) 
    rank(A.S)
end
Base.Array(A::LowRankMatrix) = A[:,:]

struct HODLRMatrix{T<:Number,ST<:Number} <: AbstractMatrix{T}
    #LowRank Blocks
    B12::LowRankMatrix{T,ST}
    B21::LowRankMatrix{T,ST}
    #DiagonalBlocks
    A11::AbstractMatrix{T}
    A22::AbstractMatrix{T}
end
Base.size(A::HODLRMatrix) = (size(A.A11,1)+size(A.B21,1),size(A.A11,2)+size(A.B12,2))
function Base.getindex(A::HODLRMatrix,i::Int,j::Int)
    if i <= size(A.A11,1) && j <= size(A.A11,1)
        return A.A11[i,j]
    elseif i > size(A.A11,1) && j <= size(A.A11,2)
        return A.B21[i-size(A.A11,1),j]
    elseif i <= size(A.A11,1) && j > size(A.A11,2)
        return A.B12[i,j-size(A.A11,2)]
    else 
        return A.A22[i-size(A.A11,1),j-size(A.A11,2)]
    end 
end

Base.getindex(A::HODLRMatrix, i, j::Int) = reshape(getindex(A,i,[j]),:)
Base.getindex(A::HODLRMatrix, i::Int, j) = reshape(getindex(A,[i],j),:)
Base.getindex(A::HODLRMatrix, i::Colon, j::Int) = reshape(_getidx(A,i,[j]),:)
Base.getindex(A::HODLRMatrix, i::Int, j::Colon) = reshape(_getidx(A,[i],j),:)
Base.getindex(A::HODLRMatrix,i,::Colon) = getindex(A, i, 1:size(A,2))
Base.getindex(A::HODLRMatrix,::Colon, j) = getindex(A, 1:size(A,1), j)
Base.getindex(A::HODLRMatrix,::Colon, ::Colon) = _getidx(A, 1:size(A,1), 1:size(A,2))
function Base.getindex(A::HODLRMatrix, i,j)
  m, n  = size(A)
  ip = sortperm(i); jp = sortperm(j)
  if (length(i) == 0 || length(j) == 0) return Matrix{T}(undef, length(i), length(j)) end
  return @views _getidx(A, i[ip], j[jp])[invperm(ip), invperm(jp)]
end
function _getidx(A::HODLRMatrix, i, j)
    m1, n1 = size(A.A11)
    i1 = @views i[i .<= m1]; j1 = @views j[j .<= n1]
    i2 = @views i[i .> m1] .- m1; j2 = @views j[j .> n1] .- n1
    B12 = A.B12[i1,j2]
    B21 = A.B21[i2,j1]
    A11 = _getidx(A.A11, i1, j1)
    A22 = _getidx(A.A22, i2, j2)
    return [A11 B12; B21 A22]
end
function _getidx(A::AbstractMatrix, i, j)
    return A[i,j]
end

Base.Array(A::HODLRMatrix) = A[:,:]

#Construct HODLR representation by randomized algorithm
function HODLRMatrix(A::AbstractMatrix{T},opts::HODLROptions=HODLROptions(T); args...) where T
    opts = isempty(args) ? opts : copy(opts; args...)
    return _compress(A,opts)
end
function _compress(A::AbstractMatrix,opts::HODLROptions)
    if any(size(A) .<= opts.leafsize)
        return A[:,:]
    else
        m1, n1 = size(A) .÷ 2
        #U12, vS12, V12 = psvd(@view( A[1:m1, n1+1:end] ), LRAOptions(opts) )
        U12, S12, V12 = aca(@view( A[1:m1, n1+1:end] ) )
        #U21, vS21, V21 = psvd(@view( A[m1+1:end, 1:n1] ), LRAOptions(opts) )
        U21, S21, V21 = aca(@view( A[m1+1:end, 1:n1] ))
        A11 = _compress( @view( A[1:m1, 1:n1] ), opts)
        A22 = _compress( @view( A[m1+1:end, n1+1:end] ), opts)
        #B12 = LowRankMatrix(U12,diagm(vS12),Matrix(V12))
        #B21 = LowRankMatrix(U21,diagm(vS21),Matrix(V21))
        B12 = LowRankMatrix(U12,S12,Matrix(V12'))
        B21 = LowRankMatrix(U21,S21,Matrix(V21'))
        return HODLRMatrix(B12,B21,A11,A22)
    end
end

function hodlr_rank(A::LowRankMatrix) 
    rank(A)
end
hodlr_rank(A::AbstractArray) = 0
function hodlr_rank(A::HODLRMatrix)
    r = maximum(hodlr_rank.((A.A11, A.A22, A.B12, A.B21)))
end


_collect(s) = s |> collect |> sort
function aca(A::AbstractMatrix; n_samples = 4, rank_estimate = 4, 
        maxiter = min(4*rank_estimate,minimum(size(A)) - rank_estimate), max_refine = 10, max_retry = 2, r_IJ = false)
    m,n = size(A)
    retry = 0
    m_iter = 0
    #select an initial guess
    I = sample(1:m,rank_estimate,replace = false) |> Set
    J = sample(1:n,rank_estimate,replace = false) |> Set
    
    #Initial interpolation
    column = zeros(eltype(A),size(A,1),maxiter+rank_estimate)
    buffer_column = similar(column)
    column[:,1:rank_estimate] = A[:,_collect(J)]
    row = zeros(eltype(A),maxiter+rank_estimate,size(A,2))
    buffer_row = similar(row)
    row[1:rank_estimate,:] = A[_collect(I),:]
    _A = A[_collect(I),_collect(J)]
    S = pinv( _A )
    
    while m_iter != maxiter && retry < max_retry
        #pick random set of sample
        LI = sample(setdiff(1:m,I), n_samples, replace = false)
        LJ = sample(setdiff(1:n,J), n_samples, replace = false)
        #Evaluate the interpolation on these samples
        Aitp = @views column[_collect(LI),1:rank_estimate+m_iter]*S*row[1:rank_estimate+m_iter,_collect(LJ)]
        #Select the most significant sample
        Δ = abs.( Aitp - A[_collect(LI),_collect(LJ)] )
        i_s, j_s = argmax(Δ) |> Tuple
        n_iter = 1
        #Refine the sample selection
        while n_iter != max_refine
            old_i_s,old_j_s = i_s,j_s
            #improve the choice of i_s
            Aitp = @views column[:,1:rank_estimate+m_iter]*S*row[1:rank_estimate+m_iter,[j_s]]
            i_s = argmax( abs.(Aitp[:] - A[:,j_s]) )
            #improve the choise of j_s
            Aitp = @views column[[i_s],1:rank_estimate+m_iter]*S*row[1:rank_estimate+m_iter,:]
            j_s = argmax( abs.(Aitp[:] - A[i_s,:]) )
            n_iter += 1
            if i_s == old_i_s && j_s == old_j_s
                #println("done")
                break
            elseif n_iter == maxiter
                break
            end
        end
        if i_s in I || j_s in J
            #println("retry")
            retry += 1
            continue
        else
            retry = 0
            #Update the interpolation
            m_iter += 1
            row[rank_estimate+m_iter,:] = A[[i_s],:]
            column[:,rank_estimate+m_iter] = A[:,[j_s]]
            #reorganise the array
            buffer_row[1:rank_estimate+m_iter,:] = @view row[sortperm([_collect(I); i_s]),:]
            row[1:rank_estimate+m_iter,:] = @view buffer_row[1:rank_estimate+m_iter,:]
            buffer_column[:,1:rank_estimate+m_iter] = @view column[:,sortperm([_collect(J); j_s])]
            column[:,1:rank_estimate+m_iter] = @view buffer_column[:,1:rank_estimate+m_iter]
            push!(I,i_s)
            push!(J,j_s)
            _A = A[_collect(I),_collect(J)]
            S = pinv( _A )
        end
    end
    if !r_IJ
         column[:,1:rank_estimate+m_iter],S,row[1:rank_estimate+m_iter,:]
    else
        column[:,1:rank_estimate+m_iter],S,row[1:rank_estimate+m_iter,:], _collect(I), _collect(J)
    end
end