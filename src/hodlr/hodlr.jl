
using StaticArrays
using LowRankApprox

export HodlrSettings

@kwdef struct HodlrSettings
    tol::Real = 1e-6
    leafsize::Int = 64
    sampling_threshold::Int = 100^2
end
function combine_settings(setting_left::HodlrSettings,setting_right::HodlrSettings)::HodlrSettings
    if setting_left != setting_right
        throw(ArgumentError("Does not support setting mixing yet, we got $(setting_left) and $(setting_right)"))
    end
    return setting_left
end
abstract type HodlrTree{T} end
 
include("PartitionTrees.jl")
include("LowRankBlocks.jl")
include("holdr_implementation.jl")

struct Hodlr{T} <: AbstractMatrix{T}
    tree::HodlrTree
end
# AbstractArray interface
size(A::Hodlr) = size(A.tree)
size(A::Hodlr, i) = size(A.tree, i)
getindex(A::Hodlr, I::Vararg{Int, 2}) = getindex(A.tree, I...
# Interface for the solver
full(A::Hodlr) = full(A.tree)