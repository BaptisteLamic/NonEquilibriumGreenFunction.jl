"""
    BlockCirculantMatrix{T} <: AbstractMatrix{T}

A block circulant matrix representation for efficient convolution operations.

# Fields
- `data::Array{T,3}`: 3D array storing the circulant blocks

The matrix is structured such that each block A.data[:,:,k] represents a circulant
shift, and the full matrix has a circulant structure across blocks.
"""
struct BlockCirculantMatrix{T} <: AbstractMatrix{T}
    data::Array{T,3}
end

"""
    BlockCirculantMatrix(data)

Construct a BlockCirculantMatrix from a 3D array.

# Arguments
- `data`: 3D array where data[i,j,k] is the (i,j) element of the k-th circulant block

# Returns
A BlockCirculantMatrix

# Throws
AssertionError if the first two dimensions of data are not equal (blocks must be square).
"""
function BlockCirculantMatrix(data)
    @assert size(data,1) == size(data,2) "Blocks must be square matrices"
    return BlockCirculantMatrix{eltype(data)}(data)
end

"""
    blocksize(A::BlockCirculantMatrix)

Returns the size of each block in the circulant matrix.
"""
blocksize(A::BlockCirculantMatrix) = size(A.data,1)

"""
    size(A::BlockCirculantMatrix)

Returns the size of the full matrix as a tuple (n, n).
The size is computed as blocksize * (number_of_blocks ÷ 2 + 1).
"""
function size(A::BlockCirculantMatrix)
    n = blocksize(A) * (size(A.data,3) ÷ 2 + 1)
    return (n, n)
end

"""
    getindex(A::BlockCirculantMatrix, i, j)

Returns the element at position (i, j) in the circulant matrix.
Uses the circulant structure to map indices to the appropriate block.
"""
@inline function getindex(A::BlockCirculantMatrix, I::Vararg{Int,2})
    n = size(A,1) ÷ blocksize(A)
    blck_i, s_i = blockindex(I[1], blocksize(A))
    blck_j, s_j = blockindex(I[2], blocksize(A))
    blck = blck_i - blck_j + n
    return A.data[s_i, s_j, blck]
end
"""
    _pad_for_convolution(x)

Pads the input array for convolution to avoid circular artifacts.
"""
function _pad_for_convolution(x)
    nz = size(x,3)-1
    padSize = size(x,3) + nz
    r = similar(x, size(x,1), size(x,2), padSize)
    r[:, :, 1:size(x,3)] .= x
    r[:, :, size(x,3)+1:end] .= 0
    return r
end

"""
    _g_conv!(r, A::BlockCirculantMatrix, x, f)

Performs a generalized convolution using FFT-based acceleration.
The function `f` is applied to the frequency-domain representation of A.data.
"""
function _g_conv!(r, A::BlockCirculantMatrix, x, f)
    reshaped_x = permutedims(reshape(x, blocksize(A), :, size(x,2)), (1,3,2))
    padded_x = _pad_for_convolution(reshaped_x)
    fft_x = fft(ifftshift(padded_x, 3), 3)
    fft_m = fft(ifftshift(A.data, 3), 3)
    p_r = permutedims(
        fftshift(ifft(
            batched_mul(f(fft_m), fft_x)
                    , 3)
            , 3
            )[:, :, 1:size(reshaped_x,3)]
        , (1,3,2))
    _r = reshape(p_r, :, size(x,2))
    if eltype(A) <: Real
        if real(eltype(A)) <: Integer
            return r .= _r .|> real .|> round .|> eltype(A)
        else
            return r .= _r .|> real .|> eltype(A)
        end
    else
        return r .= _r .|> eltype(A)
    end
end

"""
    _mul!(r, A::BlockCirculantMatrix, x)

In-place matrix-vector multiplication for BlockCirculantMatrix.
"""
@inline function _mul!(r, A::BlockCirculantMatrix, x)
    _g_conv!(r, A, x, identity)
end

"""
    _mul(A::BlockCirculantMatrix, x)

Matrix-vector multiplication for BlockCirculantMatrix.
"""
function _mul(A::BlockCirculantMatrix, x)
    r = similar(x)
    _mul!(r, A, x)
    return r
end

"""
    _cmul!(r, A::BlockCirculantMatrix, x)

In-place adjoint matrix-vector multiplication for BlockCirculantMatrix.
"""
@inline function _cmul!(r, A::BlockCirculantMatrix, x)
    _g_conv!(r, A, x, batched_adjoint)
end

"""
    _cmul(A::BlockCirculantMatrix, x)

Adjoint matrix-vector multiplication for BlockCirculantMatrix.
"""
function _cmul(A::BlockCirculantMatrix, x)
    r = similar(x)
    _cmul!(r, A, x)
    return r
end
