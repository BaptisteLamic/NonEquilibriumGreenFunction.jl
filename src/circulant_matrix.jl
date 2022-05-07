struct BlockCirculantMatrix{T} <: AbstractMatrix{T}
    data::Array{T,3}
end
function BlockCirculantMatrix(data)
    @assert size(data,1) == size(data,2)
    BlockCirculantMatrix{eltype(data)}(data)
end
blocksize(A::BlockCirculantMatrix) = size(A.data,1) 
function size(A::BlockCirculantMatrix)
    n = blocksize(A)*(size(A.data,3) ÷ 2 +  1)
    return (n,n)
end
function getindex(A::BlockCirculantMatrix, I::Vararg{Int,2})
    n = size(A,1) ÷ blocksize(A)
    blck_i, s_i = blockindex(I[1],blocksize(A))
    blck_j, s_j = blockindex(I[2],blocksize(A))
    blck = blck_i-blck_j + n
    A.data[s_i,s_j,blck]
end
function _pad_for_convolution(x)
    nz = size(x,3)-1
    r = similar(x,size(x,1),size(x,2),size(x,3) + nz)
    r[:,:,1:size(x,3)] .= x
    r[:,:,size(x,3)+1:end] .= 0
    return r
end
function _g_conv!(r,A::BlockCirculantMatrix,x,f)
    reshaped_x =  permutedims(reshape(x,blocksize(A),:,size(x,2)),(1,3,2))
    padded_x = _pad_for_convolution(reshaped_x)
    fft_x = fft(ifftshift(padded_x,3),3)
    fft_m = fft(ifftshift(A.data,3),3)
    p_r = permutedims(
        fftshift(ifft( 
            batched_mul(f(fft_m), fft_x)
                    ,3)
            , 3
            )[:,:,1:size(reshaped_x,3)]
        ,(1,3,2))
    _r = (reshape(p_r,:,size(x,2)))
    if eltype(A) <: Real
        if real(eltype(A)) <: Integer
            return r .= _r .|> real .|> round .|>eltype(A)
        else
            return r .= _r .|> real .|>eltype(A)
        end
    else
        return r .= _r  .|>eltype(A)
    end
end
function _mul!(r,A::BlockCirculantMatrix,x)
    _g_conv!(r,A,x,identity)
end
function _mul(A::BlockCirculantMatrix,x)
    r = similar(x)
    _mul!(r,A,x)
    return r
end
function _cmul!(r,A::BlockCirculantMatrix,x)
    _g_conv!(r,A,x,batched_adjoint)
end
function _cmul(A::BlockCirculantMatrix,x)
    r = similar(x)
    _cmul!(r,A,x)
    return r
end
#