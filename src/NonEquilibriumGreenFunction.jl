module NonEquilibriumGreenFunction

using SpecialFunctions: Threads
using HssMatrices: Threads
using Base: offset_if_vec
using StatsBase: Threads
using NNlib: similar
using LinearAlgebra: similar, Threads, length, Core
using HssMatrices
using SparseArrays
using LinearAlgebra
using StatsBase
using NNlib
using SpecialFunctions

import Base: +, -, *,/, \, adjoint, transpose, eltype, size, adjoint, one
import Base: getindex
import HssMatrices: full
full(A::AbstractArray) = Array(A)
import SparseArrays: blockdiag

export AbstractGreenFunction, GreenFunction, RetardedGreenFunction, AdvancedGreenFunction
export axis, retarded, advanced, regular, dirac, blocksize
export getindex
export build_greenfunction, build_linearMap,blockrange,blockindex, col, row

export AbstractKernel, RetardedKernel, AdvancedKernel, Kernel, SumKernel, TimeLocalKernel
export NONCompression, HssCompression
export solve_dyson
export cc_prod, extract_blockdiag, blockdiag
include("greenfunction.jl")
include("compression.jl")
include("dyson.jl")
include("utils.jl")
include("physics.jl")
include("kernels.jl")

end
