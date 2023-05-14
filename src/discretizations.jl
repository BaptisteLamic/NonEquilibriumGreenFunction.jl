abstract type AbstractDiscretisation{A,M,C} end
struct TrapzDiscretisation{A,M,C} <: AbstractDiscretisation{A,M,C}
    axis::A
    matrix::M
    blocksize::Int
    compression::C
end


axis(k::AbstractDiscretisation) = k.axis
step(k::AbstractDiscretisation) = k |> axis |> step
matrix(k::AbstractDiscretisation) = k.matrix
blocksize(k::AbstractDiscretisation) = k.blocksize
compression(k::AbstractDiscretisation) = k.compression
scalartype(k::AbstractDiscretisation) = k |> matrix |> eltype
size(dis::AbstractDiscretisation) = (length(axis(dis)), length(axis(dis)))
size(dis::AbstractDiscretisation, k) = size(dis)[k]


function getindex(A::AbstractDiscretisation, ::Colon, I, ::Colon, J)
    sbk = blocksize(A)
    bk_I = vcat(blockrange.(I, sbk)...)
    bk_J = vcat(blockrange.(J, sbk)...)
    values = matrix(A)[bk_I, bk_J]
    return reshape(values, sbk, length(I), sbk, length(J))
end
function _getindex(A::AbstractDiscretisation, I, J)
    #assume that the index are sorted
    sbk = blocksize(A)
    values = reshape(getindex(A, :, I, :, J), length(I) * sbk, length(J) * sbk)
    r = [view(values, sbk*(i-1)+1:sbk*i, sbk*(j-1)+1:sbk*j) for i = 1:length(I), j = 1:length(J)]
    return r
end
function getindex(A::AbstractDiscretisation, i::Int, j::Int)
    bs = blocksize(A)
    return matrix(A)[blockrange(i, bs), blockrange(j, bs)]
end
getindex(A::AbstractDiscretisation, i::Int, j) = reshape(getindex(A, [i], j), :)
getindex(A::AbstractDiscretisation, i, j::Int) = reshape(getindex(A, i, [j]), :)
getindex(A::AbstractDiscretisation, ::Colon, ::Colon) = getindex(A, 1:size(A, 1), 1:size(A, 2))
getindex(A::AbstractDiscretisation, i, ::Colon) = getindex(A, i, 1:size(A, 2))
getindex(A::AbstractDiscretisation, i::Int, ::Colon) = getindex(A, [i], 1:size(A, 2))
getindex(A::AbstractDiscretisation, ::Colon, j) = reshape(getindex(A, 1:size(A, 1), j), :)
getindex(A::AbstractDiscretisation, ::Colon, j::Int) = reshape(getindex(A, 1:size(A, 1), j), :)
function getindex(A::AbstractDiscretisation, I, J)
    Ip = sortperm(I)
    Jp = sortperm(J)
    if (length(I) == 0 || length(J) == 0)
        return Matrix{eltype(A)}(undef, length(I), length(J))
    else
        return @views _getindex(A, I[Ip], J[Jp])[invperm(Ip), invperm(Jp)]
    end
end

function similar(discretization::D,new_matrix)  where D <: AbstractDiscretisation 
    D(
        axis(discretization),
        new_matrix,
        blocksize(discretization),
        compression(discretization)
    )
end
function adjoint(dis :: TrapzDiscretisation)
    similar(dis, dis |> matrix |> adjoint |> _adapt)
end

-(discretization::AbstractDiscretisation) = similar(discretization, -matrix(discretization))