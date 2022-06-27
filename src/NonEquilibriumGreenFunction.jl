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
using FFTW
using DSP
using StaticArrays
using BlockArrays
using LowRankApprox

import Base: +, -, *,/, \, adjoint, transpose, eltype, size, adjoint, one
import Base: getindex, step
import Base: similar, zero
import HssMatrices: full, compress
import SparseArrays: blockdiag
import LinearAlgebra.I
import LinearAlgebra.diag

include("HODLRMatrices/HODLRMatrices.jl")

include("greenfunction.jl")
include("circulant_matrix.jl")
include("compression.jl")
include("dyson.jl")
include("utils.jl")
include("kernels.jl")
include("kernel_solve.jl")
include("physics.jl")
include("RAKMatrices.jl")

export HODLRMatrices

export AbstractGreenFunction, GreenFunction, RetardedGreenFunction, AdvancedGreenFunction
export axis, retarded, advanced, regular, dirac, blocksize
export getindex
export build_greenfunction, build_linearMap,blockrange,blockindex, col, row, build_CirculantlinearMap

export AbstractKernel, RetardedKernel, AdvancedKernel, Kernel, SumKernel, TimeLocalKernel
export NullKernel
export isretarded, isadvanced, timelocal_part, nonlocal_part
export BlockCirculantMatrix
export NONCompression, HssCompression
export solve_dyson
export energy2time
export cc_prod, extract_blockdiag, blockdiag
export pauli, pauliRAK
export RAKMatrix
export matrix
export energy2RetardedKernel
export compression
export compress
export scalartype
export getindex!
export tr_K
export HODLRCompression


end
