# Custom matrix representations and backends

The package programs against a single backend contract (see
[Backend contract](backend_contract.md)), so you can plug in your own
matrix representation (dense CPU, HSS, GPU-style arrays such as
`JLArray`, ...) by defining a **representation tag** - a pure parameter
holder subtype of `AbstractRepresentation` - plus, if needed, a dedicated
`AbstractBackend` subtype and a few family-specific primitive methods.

> A backend is **not necessarily a compression**: a dense GPU-array family
> (e.g. JLArray/CUDA dense) is a valid backend that compresses nothing.
> The word "compression" survives only where it is literally true (the
> `compress` primitive, HSS tolerances `atol`/`rtol`/`kest`/`leafsize`).

## The representation tag

A representation tag describes the family kernel matrices are stored in
and carries the construction parameters. The package never calls it
directly: construction goes through `compress(backend(r), ...)` on the
backend the tag bridges to.

```julia
using NonEquilibriumGreenFunction

struct MyRepresentation <: AbstractRepresentation
    # any parameters you need (e.g. HSS tolerances)
end

# the backend the tag evaluates through (default: MatrixFamilyBackend,
# the generic AbstractMatrix family backend):
NonEquilibriumGreenFunction.backend(::MyRepresentation, ::Type{T}=ComplexF64) where {T<:Number} =
    NonEquilibriumGreenFunction.MatrixFamilyBackend{T}()
```

With just the tag and the default backend, kernels are stored in the
generic `AbstractMatrix` family (dense `Matrix`): the package calls

| Method | Required | Meaning |
|---|---|---|
| `backend(r, T)` | no | the `AbstractBackend` the tag bridges to; defaults to `MatrixFamilyBackend{T}` |
| `compress(b, axis, f; stationary=false)` | if `b` is custom | build the kernel matrix of `f(t, tp)` (already causality-masked by the caller). `f` returns a square block of fixed size. `stationary=true` means `f` only depends on `t - tp`. |
| `compress(b, m::AbstractMatrix)` | if `b` is custom | (re)compress a matrix into the family. `m` may be a dense CPU `Matrix` or a `SparseMatrixCSC` produced internally (block diagonals). Must be pure. |
| `compress(b, blocks::AbstractArray{T,3})` | if `b` is custom | block-diagonal matrix with block `blocks[:, :, k]` on diagonal block `k`. Used by `solve_dyson`, kernel dressings and quadrature weights. |
| `Base.:(==)` on the tag | yes | between tag instances (free for plain immutable structs); needed for operator equality. |
| `recompress_inplace!(b, r, m)` | no | in-place recompression used by `compress!`; default is a no-op. |

## The backend contract

If the family needs more than the generic `AbstractMatrix` ops (structured
norms, native solves, GPU placement, ...), define a dedicated backend
subtype and implement the core primitive set - the list in
`CORE_PRIMITIVES` (`src/backend_interface.jl`), mirrored in the
[contract table](backend_contract.md#the-primitive-set):

`compress`, `op_add`, `op_scale`, `op_mul`, `op_adjoint`, `solve`,
`blockdiag_blocks`, `to_cpu`, `same_time_blocks`, `recompress_inplace!`,
`working_eltype` (plus `op_norm`/`apply` if you want them; `op_sub` and
variadic `op_mul` chains have defaults). The optional `evaluate(b, ex)`
override enables whole-graph fusion. Every primitive must be in-family:
there is no dense fallback, and an unsupported operation is a
`MethodError`. Operands of binary primitives must share an eltype unless
the backend explicitly widens.

Only these entry points are used by the kernel constructors
(`RetardedKernel`, `AdvancedKernel`, `AcausalKernel`, `LocalKernel`),
kernel algebra (`+`, `-`, `*`, `adjoint`), `solve_dyson`,
`same_time`/`keldysh_trace` and `compress!`.

## Testing your implementation

`test_representation_contract(rep; N=32, bs=2, atol=1e-8, types=(ComplexF64,))`
is the contract audit: it checks the whole contract against a dense
reference - construction of all kernel kinds (including `Stationary`
maps, `LocalKernel`, `Separable`), the algebra, `solve_dyson`,
`same_time`, `keldysh_trace`, recompression, `make_similar` and
`compress!`.

```julia
using Test
using NonEquilibriumGreenFunction

@testset "MyRepresentation" begin
    test_representation_contract(MyRepresentation())
end
```

The meta-test in `src/backend_interface_tests.jl` additionally fails if
a shipped backend lacks a method for any primitive in `CORE_PRIMITIVES`.

## Example: GPU-style dense arrays with JLArrays.jl

`JLArrayRepresentation` (in the `JLArraysExt` package extension, available
when JLArrays.jl is loaded) stores kernels as dense `JLArray`s - a dense
GPU-style family that compresses nothing, accepted anywhere a
representation is:

```julia
using NonEquilibriumGreenFunction
using JLArrays

rep = Base.get_extension(NonEquilibriumGreenFunction, :JLArraysExt).JLArrayRepresentation()
g = RetardedKernel(axis, TwoTime(f); representation=rep)
G = solve_dyson(g, K)
```

It defines a `JLArrayBackend <: AbstractBackend` whose `compress` builds
a CPU matrix blockwise and transfers it with `JLArray(r)`, whose
`compress(b, m)` wraps `to_cpu`, whose `ldiv` round-trips through the CPU
(JLArrays.jl has no GPU `\`), and whose `same_time_blocks` uses ranged
indexing to avoid GPU scalar indexing. Use it as a template for other GPU
backends (Metal.jl, CUDA.jl, ...).
