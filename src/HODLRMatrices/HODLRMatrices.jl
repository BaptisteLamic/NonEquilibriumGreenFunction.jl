module HODLRMatrices

using LinearAlgebra
import LinearAlgebra.rank
using LowRankApprox
using StatsBase
using Base.Iterators: product

#HODLROptions adapted from LowRankApprox.jl
mutable struct HODLROptions 
    atol::Float64
    rtol::Float64
    rank::Int
    sketch::Symbol
    leafsize::Int
end
function HODLROptions(::Type{T}; args...) where T
    opts = HODLROptions(
    0,                  # atol
    5*eps(real(T)),     # rtol
    -1,                 # rank
    :sub,               # sketch
    64                  # leafsize
    )
    for (key, value) in args
      setfield!(opts, key, value)
    end
    opts
  end
HODLROptions(; args...) = HODLROptions(Float64; args...)
function copy(opts::HODLROptions; args...)
    opts_ = HODLROptions()
    for field in fieldnames(typeof(opts))
        setfield!(opts_, field, getfield(opts, field))
    end
    for (key, value) in args
        setfield!(opts_, key, value)
    end
    opts_
end
function LowRankApprox.LRAOptions(opts::HODLROptions)
    return LRAOptions(atol = opts.atol, rtol = opts.rtol, rank = opts.rank,sketch = opts.sketch) 
end


include("HODLRMatrix.jl")
include("KernelMatrix.jl")

export LowRankMatrix, HODLRMatrix
export HODLROptions
export KernelMatrix
export hodlr_rank
export aca

end
