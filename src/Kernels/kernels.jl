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

abstract type AbstractCausality end
struct Retarded <: AbstractCausality end
struct Acausal <: AbstractCausality end
struct Advanced <: AbstractCausality end

struct Kernel{D<:AbstractDiscretisation, C<:AbstractCausality}
    discretization::D
    causality::C
end

discretization(k::Kernel) = k.discretization
causality(k::Kernel) = k.causality
step(k::Kernel) = k |> discretization |> step
blocksize(g::Kernel) = g |> discretization |> blocksize
axis(g::Kernel) = g |> discretization |> axis
compression(g::Kernel) = g |> discretization |> compression 
matrix(g::Kernel) = g |> discretization |> matrix
scalartype(g::Kernel) = g |> discretization |> scalartype
size(g::Kernel) = g |> discretization |> size
isretarded(g::Kernel) = causality(g) == Retarded()
isadvanced(g::Kernel) = causality(g) == Advanced()
isacausal(g::Kernel) = causality(g) == Acausal()


function dirac_kernel(::C,axis, f;
     compression::AbstractCompression=HssCompression()) where {C<:AbstractCausality}
    f00 = f(axis[1])
    T = eltype(f00)
    bs = size(f00,1)
    δ = zeros(T, bs, bs, length(axis))
    for i = 1:length(axis)
        δ[:, :, i] .= f(axis[i])
    end
    if C <: Union{Retarded,Advanced}
        δ .*= 2/step(axis)
    else
        @assert C == Acausal
        δ .*= 1/step(axis)
    end 
    matrix = blockdiag(δ,compression = compression)
    return Kernel(
        TrapzDiscretisation(
            axis,
            matrix,
            bs,
            compression
        ), C()
    )
end

function dirac(kernel::Kernel{D,C}) where {D,C}
    bs = blocksize(kernel)
    T = scalartype(kernel)
    return dirac_kernel(C(),axis(kernel), (t) -> Matrix(T(1)*I,bs,bs), compression = compression(kernel))
end
function similar(g::Kernel, new_matrix )
    return Kernel(similar( g |> discretization, new_matrix), g |> causality)
end

function discretize_kernel(::Type{D},::Type{C},axis, f; compression=HssCompression(), stationary=false) where {D<:AbstractDiscretisation, C<:AbstractCausality}
    causality = C()
    f00 = f(axis[1],axis[1])
    @assert size(f00,1) == size(f00,2)
    bs = size(f00,1)
    _mask(::Retarded) = (x, y) -> x >= y ? f(x, y) : zero(f00)
    _mask(::Advanced) = (x, y) -> x <= y ? f(x, y) : zero(f00)
    _mask(::Acausal) = (x, y) -> f(x, y) 
    f_masked = _mask(causality)
    matrix = compression(axis, f_masked, stationary=stationary)
    discretization = D(axis, matrix, bs, compression)
    return Kernel(discretization, causality)
end

function Kernel{D,C}(axis, matrix, blocksize, compression) where {D<:AbstractDiscretisation, C<:AbstractCausality}
    causality = C()
    discretization = D(axis, matrix, blocksize, compression)
    Kernel(discretization, causality)
end

function discretize_retardedkernel(axis, f; compression=HssCompression(), stationary=false)
    discretize_kernel(TrapzDiscretisation,Retarded,
        axis, f;
        compression=compression, stationary=stationary
        )
end
function RetardedKernel(axis, matrix, blocksize, compression)
    Kernel{TrapzDiscretisation,Retarded}(axis, matrix, blocksize, compression)
end
function discretize_advancedkernel(axis, f; compression=HssCompression(), stationary=false)
    discretize_kernel(TrapzDiscretisation,Advanced,
        axis, f;
        compression=compression, stationary=stationary
        )
end
function AdvancedKernel(axis, matrix, blocksize, compression)
    Kernel{TrapzDiscretisation,Advanced}(axis, matrix, blocksize, compression)
end
function discretize_acausalkernel(axis, f; compression=HssCompression(), stationary=false)
    discretize_kernel(TrapzDiscretisation,Acausal,
        axis, f;
        compression=compression, stationary=stationary
        )
end
function discretize_lowrank_kernel(axis, f,g; compression=HssCompression())
    f00 = f(axis[1])
    g00 = g(axis[1])
    @assert size(f00,1) == size(f00,2)
    @assert size(f00) == size(g00)
    bs = size(f00,1)
    matrix = compression(axis, f, g)
    discretization = TrapzDiscretisation(axis, matrix, bs, compression)
    return Kernel(discretization, Acausal())
end
function AcausalKernel(axis, matrix, blocksize, compression)
    Kernel{TrapzDiscretisation,Acausal}(axis, matrix, blocksize, compression)
end

function compress!(discretization::AbstractDiscretisation)
    cpr = discretization |> compression
    cpr( discretization |> matrix)
    return discretization
end
function compress!(kernel::Kernel)
    kernel |> discretization |> compress!
    return kernel  
end


include("kernel_algebra.jl")
include("kernel_solver.jl")

