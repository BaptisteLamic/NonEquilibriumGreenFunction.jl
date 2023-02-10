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

import Base: +, -, *,/, \, adjoint, transpose, eltype, size, adjoint, one
import Base: sum
import Base: ==
import Base: getindex, step
import Base: similar, zero
import HssMatrices: full, compress!
import SparseArrays: blockdiag
import LinearAlgebra.I
import LinearAlgebra.diag
import LinearAlgebra.norm
import HssMatrices: compress!
import Base: convert

include("circulant_matrix.jl")
include("compression.jl")
include("utils.jl")
include("kernels.jl")
include("expressionTrees.jl")
include("kernel_products.jl")
include("kernel_solve.jl")
include("physics.jl")
include("RAKMatrices.jl")

export axis, retarded, advanced, regular, dirac, blocksize
export getindex
export build_greenfunction, build_linearMap,blockrange,blockindex, col, row, build_CirculantlinearMap

export AbstractKernel, RetardedKernel, AdvancedKernel, Kernel, TimeLocalKernel
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
export compress!
export scalartype
export getindex!
export tr_K

export KernelExpression, KernelExpressionLeaf, KernelExpressionTree, KernelLeaf, NullLeaf, ScalarLeaf
export KernelMul, KernelAdd,  KernelLDiv, KernelRDiv
export arguments, istree
export evaluate_expression

end
