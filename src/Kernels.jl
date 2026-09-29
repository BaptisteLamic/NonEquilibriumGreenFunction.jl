module Kernels

using ..NonEquilibriumGreenFunction: AbstractDiscretisation, AbstractCompression, TrapzDiscretisation,
    HssCompression, causality_of_sum, causality_of_prod,
    Retarded, Advanced, Acausal, AbstractCausality,
    AbstractLocality, Local, Smooth, locality_of_sum, locality_of_prod,
    extract_blockdiag, build_blockdiag, triangularLowRankCompression,
    blockrange, blockindex
import ..NonEquilibriumGreenFunction: locality, islocal

using HssMatrices
using SparseArrays
using LinearAlgebra
using ..NonEquilibriumGreenFunction: I
using TestItems

import ..NonEquilibriumGreenFunction: matrix, axis, blocksize, scalartype, step, compression, make_similar
import ..NonEquilibriumGreenFunction: ldiv, to_cpu, same_time_blocks, recompress_inplace!

import Base: +, -, *, ==, adjoint, step, getindex, size, sum, prod
import LinearAlgebra: norm, adjoint

include("operators.jl")
include("Kernels/kernel_maps.jl")
include("Kernels/kernels.jl")

export AbstractOperator, SimpleOperator, CompositeOperator, Kernel, LocalKernel, SumOperator
export locality, islocal, AbstractLocality, Local, Smooth
export RetardedKernel, AdvancedKernel, AcausalKernel
export AbstractKernelMap, Stationary, TwoTime, Separable
export causality, isretarded, isadvanced, isacausal, discretization
export same_time, keldysh_trace
export solve_dyson
export compress!, make_similar

end
