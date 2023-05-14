module NonEquilibriumGreenFunction

using SpecialFunctions: Threads
using HssMatrices: Threads
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
using StaticArrays
using DSP
using TestItems
using Symbolics
using SymbolicUtils

import Base: +, -, *,/, \, adjoint, transpose, eltype, size, adjoint, one
import Base: sum
import Base: ==
import Base: getindex, step
import Base: similar, zero
import SparseArrays: blockdiag
import LinearAlgebra.I
import LinearAlgebra.diag
import LinearAlgebra.norm
import Base: convert

include("circulant_matrix.jl")
include("compression.jl")
include("utils.jl")
include("discretizations.jl")
include("operators.jl")
include("Kernels/kernels.jl")
include("physics.jl")

export axis, blocksize
export getindex
export build_linearMap,blockrange,blockindex, build_CirculantlinearMap

#new export 
export TrapzDiscretisation, AbstractDiscretisation
export Retarded, Advanced, Acausal, Instantaneous
export isretarded, isadvanced, isacausal
export discretization
export SimpleOperator, CompositeOperator
export DiracOperator, discretize_dirac
export SumOperator
export Kernel
export RetardedKernel, AdvancedKernel, AcausalKernel
export discretize_retardedkernel, discretize_advancedkernel, discretize_acausalkernel
export discretize_lowrank_kernel
export causality
export solve_dyson
export adjoint
export simplify_kernel

export BlockCirculantMatrix
export NONCompression, HssCompression
export energy2time
export extract_blockdiag, blockdiag
export pauli
export matrix
export energy2RetardedKernel
export compression
export compress!
export scalartype
export getindex!


end
