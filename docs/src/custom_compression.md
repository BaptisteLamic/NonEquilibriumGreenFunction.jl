# Custom compression methods and matrix representations

The package programs against a small, explicit interface, so you can plug in
your own matrix representation (dense, HSS, GPU-style arrays such as
`JLArray`, ...) by defining a compression method and, if needed, a few
family-specific methods.

## The compression interface

A compression method is a callable subtype of `AbstractCompression`:

```julia
using NonEquilibriumGreenFunction

struct MyCompression <: AbstractCompression
    # any parameters you need
end
```

The package calls the following on it:

| Method | Required | Meaning |
|---|---|---|
| `c(axis, f; stationary=false)` | yes | Build the matrix of the kernel `f(t, tp)` (already causality-masked by the caller). `f` returns a square block of fixed size. `stationary=true` means `f` only depends on `t - tp`. |
| `c(m::AbstractMatrix)` | yes | (Re)compress a matrix into your family. `m` may be a dense CPU `Matrix` or a `SparseMatrixCSC` produced internally (block diagonals). Must be pure. |
| `c(axis, f, g)` | no | Separable kernel `f(t) * g(tp)`; a generic fallback exists, so this is an optimization hook. |
| `Base.:(==)` | yes | Between instances (free for plain immutable structs); needed for operator equality. |
| `recompress_inplace!(c, m)` | no | In-place recompression used by `compress!`; default is a no-op. |

## The matrix interface

The matrix family your compression produces must support:

| Operation | Notes |
|---|---|
| `size`, `eltype`, `getindex` (scalar or ranges) | Families that forbid scalar indexing (GPU arrays) must extend `same_time_blocks` (and `extract_blockdiag` uses it). |
| `+`, `-`, scalar `*`, `adjoint` | Kernel algebra. |
| matrix-matrix `*` within the family | Kernel products and dressing. |
| `LinearAlgebra.norm` | Operator norm. |
| `ldiv(left, right)` | Solve `left * X = right`; fallback `left \ right`. |
| `to_cpu(m)` | Materialize as CPU `Matrix`; fallback `Matrix(m)`. |
| `same_time_blocks(m, bs)` | One `bs×bs` CPU matrix per time step; generic fallback uses ranged `getindex`. |

Only these entry points are used by the kernel constructors (`RetardedKernel`,
`AdvancedKernel`, `AcausalKernel`, `InstantaneousKernel`), kernel algebra
(`+`, `-`, `*`, `adjoint`), `solve_dyson`, `same_time`/`keldysh_trace` and
`compress!`.

## Testing your implementation

`test_compression_interface(cpr; N=32, bs=2, atol=1e-8, types=(ComplexF64,))`
checks the whole contract against a dense reference: construction of all
kernel kinds (including `Stationary` maps, `InstantaneousKernel`,
`Separable`), the algebra, `solve_dyson`, `same_time`,
`keldysh_trace`, recompression, `make_similar` and `compress!`.

```julia
using Test
using NonEquilibriumGreenFunction

@testset "MyCompression" begin
    test_compression_interface(MyCompression())
end
```

## Example: GPU-style dense arrays with JLArrays.jl

`JLArrayCompression` (in the `JLArraysExt` package extension, available when
JLArrays.jl is loaded) stores kernels as dense `JLArray`s:

```julia
using NonEquilibriumGreenFunction
using JLArrays

cpr = Base.get_extension(NonEquilibriumGreenFunction, :JLArraysExt).JLArrayCompression()
g = RetardedKernel(axis, TwoTime(f); compression=cpr)
G = solve_dyson(g, K)
```

It defines exactly the interface above: the constructor builds a CPU matrix
blockwise and transfers it with `JLArray(r)`; `c(m)` wraps `to_cpu`; `ldiv`
round-trips through the CPU (JLArrays.jl has no GPU `\`); and
`same_time_blocks` uses ranged indexing to avoid GPU scalar indexing. Use it
as a template for other GPU backends (Metal.jl, CUDA.jl, ...).
