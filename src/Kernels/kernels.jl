struct Kernel{D<:AbstractDiscretisation, C<:AbstractCausality} <: SimpleOperator
    discretization::D
    causality::C
end

discretization(k::Kernel) = k.discretization
causality(k::Kernel) = k.causality
isretarded(g::Kernel) = causality(g) == Retarded()
isadvanced(g::Kernel) = causality(g) == Advanced()
isacausal(g::Kernel) = causality(g) == Acausal()

function similar(g::Kernel, new_discretization::AbstractDiscretisation )
    return Kernel(new_discretization, g |> causality)
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

function discretize_lowrank_kernel(::Type{D},::Type{C}, axis, f,g ;compression=HssCompression())  where {D<:AbstractDiscretisation, C<:AbstractCausality}
    f00 = f(axis[1])
    g00 = g(axis[1])
    @assert size(f00,1) == size(f00,2)
    @assert size(f00) == size(g00)
    bs = size(f00,1)
    matrix = triangularLowRankCompression(compression,C(), axis, f, g)
    discretization = TrapzDiscretisation(axis, matrix, bs, compression)
    return Kernel(discretization, C())
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

function AcausalKernel(axis, matrix, blocksize, compression)
    Kernel{TrapzDiscretisation,Acausal}(axis, matrix, blocksize, compression)
end


include("kernel_algebra.jl")
include("kernel_solver.jl")
