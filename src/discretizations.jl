"""
    AbstractDiscretisation{A,M,C}

Abstract type for discretization methods.

Type parameters:
- `A`: Type of the axis
- `M`: Type of the matrix
- `C`: Type of the compression
"""
abstract type AbstractDiscretisation{A,M,C} end

"""
    TrapzDiscretisation{A,M,C} <: AbstractDiscretisation{A,M,C}

Trapezoidal rule discretization for time-axis integrals.

# Fields
- `axis::A`: Time axis
- `matrix::M`: Discretized matrix
- `blocksize::Int`: Size of each block
- `compression::C`: Compression method used
"""
struct TrapzDiscretisation{A,M,C} <: AbstractDiscretisation{A,M,C}
    axis::A
    matrix::M
    blocksize::Int
    compression::C
end

"""
    step(k::AbstractDiscretisation)

Returns the step size of the discretization's axis.
"""
step(k::AbstractDiscretisation) = k |> axis |> step

"""
    scalartype(k::AbstractDiscretisation)

Returns the scalar type of the discretization's matrix elements.
"""
scalartype(k::AbstractDiscretisation) = k |> matrix |> eltype

"""
    size(dis::AbstractDiscretisation)

Returns the size of the discretization as a tuple.
"""
size(dis::AbstractDiscretisation) = (length(axis(dis)), length(axis(dis)))

"""
    size(dis::AbstractDiscretisation, k)

Returns the k-th dimension of the discretization's size.
"""
size(dis::AbstractDiscretisation, k) = size(dis)[k]

"""
    blocksize(k::TrapzDiscretisation)

Returns the block size of the discretization.
"""
blocksize(k::TrapzDiscretisation) = k.blocksize

"""
    compression(k::TrapzDiscretisation)

Returns the compression method of the discretization.
"""
compression(k::TrapzDiscretisation) = k.compression

"""
    matrix(k::TrapzDiscretisation)

Returns the matrix of the discretization.
"""
matrix(k::TrapzDiscretisation) = k.matrix

"""
    axis(k::TrapzDiscretisation)

Returns the axis of the discretization.
"""
axis(k::TrapzDiscretisation) = k.axis

"""
    getindex(A::AbstractDiscretisation, ::Colon, I, ::Colon, J)

Get a submatrix indexed by block indices I and J.
Returns a 4D array of shape (blocksize, length(I), blocksize, length(J)).
"""
function getindex(A::AbstractDiscretisation, ::Colon, I, ::Colon, J)
    sbk = blocksize(A)
    bk_I = vcat(blockrange.(I, sbk)...)
    bk_J = vcat(blockrange.(J, sbk)...)
    values = matrix(A)[bk_I, bk_J]
    return reshape(values, sbk, length(I), sbk, length(J))
end

"""
    _getindex(A::AbstractDiscretisation, I, J)

Internal function to get a submatrix assuming sorted indices.
"""
function _getindex(A::AbstractDiscretisation, I, J)
    #assume that the index are sorted
    sbk = blocksize(A)
    values = reshape(getindex(A, :, I, :, J), length(I) * sbk, length(J) * sbk)
    r = [view(values, sbk*(i-1)+1:sbk*i, sbk*(j-1)+1:sbk*j) for i = 1:length(I), j = 1:length(J)]
    return r
end

"""
    getindex(A::AbstractDiscretisation, i::Int, j::Int)

Get a block matrix at block indices (i, j).
"""
function getindex(A::AbstractDiscretisation, i::Int, j::Int)
    bs = blocksize(A)
    return matrix(A)[blockrange(i, bs), blockrange(j, bs)]
end

"""
    getindex(A::AbstractDiscretisation, i::Int, j)

Get a submatrix at block row i and block columns j.
"""
getindex(A::AbstractDiscretisation, i::Int, j) = reshape(getindex(A, [i], j), :)

"""
    getindex(A::AbstractDiscretisation, i, j::Int)

Get a submatrix at block rows i and block column j.
"""
getindex(A::AbstractDiscretisation, i, j::Int) = reshape(getindex(A, i, [j]), :)

"""
    getindex(A::AbstractDiscretisation, ::Colon, ::Colon)

Get the full matrix.
"""
getindex(A::AbstractDiscretisation, ::Colon, ::Colon) = getindex(A, 1:size(A, 1), 1:size(A, 2))

"""
    getindex(A::AbstractDiscretisation, i, ::Colon)

Get a submatrix at block rows i and all block columns.
"""
getindex(A::AbstractDiscretisation, i, ::Colon) = getindex(A, i, 1:size(A, 2))

"""
    getindex(A::AbstractDiscretisation, i::Int, ::Colon)

Get a submatrix at block row i and all block columns.
"""
getindex(A::AbstractDiscretisation, i::Int, ::Colon) = getindex(A, [i], 1:size(A, 2))

"""
    getindex(A::AbstractDiscretisation, ::Colon, j)

Get a submatrix at all block rows and block columns j.
"""
getindex(A::AbstractDiscretisation, ::Colon, j) = reshape(getindex(A, 1:size(A, 1), j), :)

"""
    getindex(A::AbstractDiscretisation, ::Colon, j::Int)

Get a submatrix at all block rows and block column j.
"""
getindex(A::AbstractDiscretisation, ::Colon, j::Int) = reshape(getindex(A, 1:size(A, 1), j), :)

"""
    getindex(A::AbstractDiscretisation, I, J)

Get a submatrix indexed by block indices I and J, with unsorted indices handled.
"""
function getindex(A::AbstractDiscretisation, I, J)
    Ip = sortperm(I)
    Jp = sortperm(J)
    if (length(I) == 0 || length(J) == 0)
        return Matrix{eltype(A)}(undef, length(I), length(J))
    else
        return @views _getindex(A, I[Ip], J[Jp])[invperm(Ip), invperm(Jp)]
    end
end

"""
    make_similar(discretization::D, new_matrix::AbstractMatrix; axis, blocksize, compression) where {D<:AbstractDiscretisation}

Create a new discretization with a modified matrix.

# Arguments
- `discretization`: Original discretization
- `new_matrix`: New matrix to use
- `axis`: Optional new axis (defaults to original axis)
- `blocksize`: Optional new blocksize (defaults to original blocksize)
- `compression`: Optional new compression (defaults to original compression)

# Returns
A new discretization of the same type with the specified modifications.
"""
function make_similar(discretization::D, new_matrix::AbstractMatrix; axis=axis(discretization),
    blocksize=blocksize(discretization), compression=compression(discretization)
) where {D<:AbstractDiscretisation}
    return D(
        axis,
        new_matrix,
        blocksize,
        compression
    )
end

"""
    adjoint(dis::TrapzDiscretisation)

Return the adjoint of a TrapzDiscretisation.
"""
function adjoint(dis::TrapzDiscretisation)
    return make_similar(dis, dis |> matrix |> adjoint)
end

"""
    -(discretization::AbstractDiscretisation)

Negate a discretization's matrix.
"""
-(discretization::AbstractDiscretisation) = make_similar(discretization, -matrix(discretization))