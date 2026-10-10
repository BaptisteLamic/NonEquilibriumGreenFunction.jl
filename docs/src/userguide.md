# User Guide

## The typical workflow

Every simulation with this package follows the same four steps. The two
[examples](generated/mqdm.md) are concrete instances of this workflow.

```julia
using NonEquilibriumGreenFunction

ax = 0:δt:T                                   # time axis

# 1. discretize the building blocks
g     = RetardedKernel(ax, TwoTime(g_func); compression=cpr)
Σ_R   = LocalKernel(ax, Σ_R_func; compression=cpr) + ...
ρ     = AcausalKernel(ax, Stationary(ρ_func); compression=cpr)

# 2. solve the retarded Dyson equation
G_R = solve_dyson(g, g * Σ_R)

# 3. dress the kinetic branch
Σ_K = -2im * coupling' * ρ * coupling
G_K = G_R * Σ_K * G_R'

# 4. extract observables
I_avr = ... # products of G_R, G_K, Σ_R, Σ_K
```

## Kernels and causality

A kernel is described by a **map** wrapping its time dependence:

- `TwoTime(f)`: general two-time function `f(t, t')`.
- `Stationary(f)`: time-translation invariant kernel `f(t - t')`; the one-argument
  signature guarantees the stationarity structurally and enables circulant compression.
- `Separable(f, g)`: separable kernel `f(t) * g(t')`, exploited by low-rank compression.

The map returns the **block matrix** of the operator at the given times. The block size
`bs` is inferred from the map sampled at the first axis point and is *global*: all
kernels combined in one simulation must return blocks of the same size. A scalar map
gives `bs = 1`; a map returning a 2×2 Nambu matrix gives `bs = 2`.

Three causalities are supported and tracked automatically through algebra:

| Causality | Meaning | Constructor |
|---|---|---|
| `Retarded` | zero for `t < t'` | `RetardedKernel` |
| `Advanced` | zero for `t > t'` | `AdvancedKernel` |
| `Acausal` | no constraint (e.g. equilibrium occupations) | `AcausalKernel` |

Contact terms `f(t) δ(t-t')` are *not* a causality: they satisfy every support
constraint simultaneously. They form a separate **locality** axis:

| Locality | Meaning | Constructor |
|---|---|---|
| `Local` | contact term `f(t) δ(t-t')`, applied exactly | `LocalKernel` |
| `Smooth` | regular kernel, integrated by the quadrature | the three above |

`Local` is the unit of the algebra: composing a `LocalKernel` with any other
operator leaves the result's causality and locality unchanged, and a `Local`
operator never enters the quadrature (no `δt` weight, no endpoint dressing) —
it is applied exactly.

Products and adjoints preserve causality (`Retarded × Retarded = Retarded`,
`Retarded' = Advanced`, ...), and the discretizations use it: retarded kernels are
compressed on their lower-triangular support only.

## Discretization

The time axis is discretized with the trapezoidal rule (`TrapzDiscretisation`). `N =
length(ax)` time steps produce a `bs*N × bs*N` block matrix. Continuous kernels are
integrated as

```math
\int^{t} g(t,t_1)\Sigma(t_1,t')\,dt_1 \;\to\; \delta t \sum_k g(t,t_k)\Sigma(t_k,t')
```

and products of kernels compose this quadrature automatically — the `*` operator on
kernels *is* the time integral, so `g * Σ` is ready to be passed to `solve_dyson`.

## Compression

The matrix of a discretized kernel is compressed to make the algebra quasi-linear:

- `HssCompression(atol, rtol, kest, leafsize)`: hierarchical semi-separable compression
  of the kernel matrix. General kernels cost ``\mathcal O(N^2)`` to compress and
  ``\mathcal O(N \log N)`` to multiply.
- `NONCompression()`: no compression, dense matrices. Multiply costs
  ``\mathcal O(N^3)`` — useful for testing and small systems.

Two knobs matter in practice:

- `quadrature`: the discretization's integration rule
  ([`AbstractQuadrature`](@ref)). `TrapezoidQuadrature()` (the default)
  gives half weights at the domain edges and converges second order for
  smooth kernels at any blocksize, including kernels that do not vanish
  at the boundary, and for every causality pairing (acausal × acausal,
  retarded × acausal, acausal × advanced; same-causality products are
  second order under both rules via the diagonal dressing).
  `RectangleQuadrature()` is the historical first-order rule, useful
  as a baseline. Pass a rule to any kernel constructor:
  `AcausalKernel(ax, map; compression, quadrature=RectangleQuadrature())`.

- `Singular` maps: kernels with an integrable singularity at ``\tau = 0``
  (e.g. the Keldysh thermal core ``-i/\beta \, \mathrm{csch}(\pi\tau/\beta)``)
  are discretized with product-integration weights
  (see [`singular_weights`](@ref)): the block-circulant matrix stores the exact
  hat-function integrals instead of sampled values, restoring second-order
  convergence of kernel products. The principal-value diagonal is exactly zero
  for odd cores. Use `Singular(f)` instead of `Stationary(f)` for such kernels.
  The core may be scalar-valued or matrix-valued, so finite-temperature
  multi-level systems (blocksize `> 1`) are supported; `thermal_kernel(t, β)`
  also provides the `T = 0` limit (`β = ∞`) where the core reduces to
  `-i/(πt)`.
- `Stationary` maps: the kernel depends only on `t-t'`, so the matrix is
  block-circulant. It is built through an FFT-accelerated circulant operator:
  ``\mathcal O(N \log N)`` construction and ``\mathcal O(N \log N)`` products.
- `leafsize` (HSS only): size of the HSS tree leaves. The examples use `leafsize=32`
  (bs=1) and `leafsize=64` (bs=2); adapt it to the rank structure of your kernel.

## Observables

Because the Keldysh trace is problem-dependent, observables are assembled by hand from
the operators. The current through lead `l` reduces to a combination of products of
`G_R`, `G_K`, `Σ_R_l` and `Σ_K_l` (see `compute_average_current` in either example);
`diag(matrix(op))` (bs=1) or `diag(matrix(op))[1:2:end] .- diag(matrix(op))[2:2:end]`
(bs=2, Keldysh trace) extracts the time-domain signal.

## Discretization error estimation

Every simulation above carries a *self-diagnostic*: after solving, the
time-discretization error of the whole flow can be estimated a posteriori at
negligible cost — no refined grid, no second problem, no interpolation — and
the estimate doubles as a **correction** that improves the solution.

The one-call entry point is
[`solve_keldysh_with_error_estimate`](@ref):

```julia
using NonEquilibriumGreenFunction, LinearAlgebra

# A single level coupled to leads: g(t) = -i e^{-iεt}, smooth self-energies
mk(ax) = (
    RetardedKernel(ax, TwoTime((t, tp) -> -1.0im * exp(-1im * 0.3 * (t - tp)));
    compression=NONCompression()),
    RetardedKernel(ax, TwoTime((t, tp) -> -0.5im * exp(-2(t - tp)));
    compression=NONCompression()),
    AcausalKernel(ax, TwoTime((t, tp) -> 1.0im * exp(-(t - tp)^2));
    compression=NONCompression()),
)

ax = range(0.0, 5.0; length=101)
g, Σ_R, Σ_K = mk(ax)
sol = solve_keldysh_with_error_estimate(g, Σ_R, Σ_K)

sol.error                    # KeldyshErrorEstimate (prints the estimate and bound)
sol.error.G_R.norm_estimate # estimated ‖G_R - G_R^exact‖_max
sol.error.G_K.norm_bound    # rigorous bound on the kinetic branch
```

The raw solution is `(sol.G_R, sol.G_K)`; the *corrected* one is
`sol.error.corrected`, a named tuple `(; G_R, G_K)` with the estimated error
added back. Feed it to the observables exactly like the raw one:

```julia
I_raw = current_signal(lead_current(sol.G_R, sol.G_K, Σ_R, Σ_K))
I_corr = current_signal(lead_current(sol.error.corrected.G_R,
                                     sol.error.corrected.G_K, Σ_R, Σ_K))
```

What the estimate covers, and how accurate it is on this demo (measured
against a 4×-refined reference solve on the same model):

| N | ‖G_R − G_R^ref‖ | estimate | corrected | ‖G_K − G_K^ref‖ | estimate | corrected | K bound |
|---|---|---|---|---|---|---|---|
| 101 | 3.4e-4 | 3.3e-4 | 2.4e-5 | 1.9e-3 | 2.5e-3 | 8.6e-4 | 1.8e-2 |
| 201 | 8.5e-5 | 8.6e-5 | 2.7e-6 | 4.9e-4 | 6.8e-4 | 2.6e-4 | 4.7e-3 |
| 401 | 2.1e-5 | 2.2e-5 | 8.9e-7 | 1.2e-4 | 1.8e-4 | 6.9e-5 | 1.2e-3 |

Three things to read off the table:

- **The retarded estimate is asymptotically exact** (effectivity ≈ 1). This is
  the `solve_dyson` scheme defect, recovered from the already computed matrices
  via the Euler–Maclaurin correction of the trapezoid rule.
- **The correction is worth a grid refinement or two**: adding
  `error.corrected` back reduces the error by an order of magnitude on both
  branches — for free, since the estimate is already computed.
- **Everything scales as O(δt²)**: both branches, the estimate and the bound.
  If the bound is not small enough for your accuracy target, halve δt (or just
  use the corrected solution).

The estimate covers every discretization step of the flow: the formation of the
kernel `K = g·Σ_R` (itself a quadrature product), the implicit-trapezoid
scheme of `solve_dyson`, and the two kernel products of the kinetic dressing
`G_K = G_R Σ_K G_R'`, with the retarded error propagated through the dressing at
first order. The per-branch *rigorous* bounds use a discrete-Gronwall constant,
so `norm_bound ≥ ‖error‖` is guaranteed (up to compression error, which must be
added separately when relevant).

If you already solved with [`solve_keldysh`](@ref), call
[`estimate_keldysh_error`](@ref) directly on the result; for a bare retarded
solve, [`estimate_discretization_error`](@ref) `(g, K, G)` estimates just the
`solve_dyson` step.

## Performance notes

- HSS compression does not benefit from multithreaded BLAS; large runs typically call
  `BLAS.set_num_threads(1)` and rely on Julia threads instead.
- Prefer `stationary=true` whenever the physics allows — the circulant path is the
  fastest construction.
- Kernel products allocate intermediate HSS matrices; long expression chains are faster
  when intermediate results are named and reused.
- Benchmarks and committed per-hardware baselines live in `benchmark/` (see
  `benchmark/README.md`).
