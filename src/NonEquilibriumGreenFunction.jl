module NonEquilibriumGreenFunction

using NNlib: batched_mul, batched_adjoint
using HssMatrices
using SparseArrays
using LinearAlgebra
using StatsBase
using SpecialFunctions: polygamma
using FFTW
using TestItems
using Test

import Base: +, -, *, /, \, adjoint, transpose, eltype, size, one
import Base: sum
import Base: ==
import Base: getindex, step
import Base: zero
import Base: convert, prod
import LinearAlgebra.I
import LinearAlgebra.diag
import LinearAlgebra.norm

include("causality.jl")
include("matrixinterface.jl")
include("locality.jl")
include("singular.jl")
include("circulant_matrix.jl")
include("triangularLowRankMatrix.jl")
include("compression.jl")
include("utils.jl")
include("quadrature.jl")
include("discretizations.jl")

include("Kernels.jl")
using .Kernels

include("Physics.jl")
using .Physics
include("test_compression_interface.jl")

include("AdaptiveRichardson.jl")

export Kernels, Physics
export singular_weights

export axis, blocksize
export getindex
export build_linearMap, blockrange, blockindex, build_CirculantlinearMap
#new export
export TrapzDiscretisation, AbstractDiscretisation
export AbstractLocality, Local, Smooth, locality, islocal, locality_of_prod, locality_of_sum
export Retarded, Advanced, Acausal
export isretarded, isadvanced, isacausal
export discretization
export SimpleOperator, CompositeOperator
export LocalKernel
export SumOperator
export Kernel
export RetardedKernel, AdvancedKernel, AcausalKernel
export AbstractKernelMap, Stationary, TwoTime, Separable, Singular
export causality
export solve_dyson
export adjoint
export norm
export BlockCirculantMatrix
export NONCompression, HssCompression
export AbstractCompression, ldiv, to_cpu, same_time_blocks, recompress_inplace!
export test_compression_interface
export pauli
export matrix
export compression
export compress!
export scalartype
export make_similar
export thermal_kernel, theq_lesser_time_kernel
export solve_keldysh, lead_current, current_signal
export same_time, keldysh_trace

end
