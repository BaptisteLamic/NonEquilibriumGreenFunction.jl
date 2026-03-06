
using StaticArrays
using LowRankApprox

export HodlrContext

@kwdef struct HodlrContext
    tol::Real = 1e-6
    leafsize::Int = 64
    sampling_threshold::Int = 100^2
end
abstract type HodlrTree{T} end

include("PartitionTrees.jl")
include("LowRankBlocks.jl")
include("holdr_implementation.jl")