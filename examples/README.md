# Examples

The two worked examples live as Literate scripts in `docs/lit/` and are executed at
documentation build time (see `docs/make.jl`). The published pages therefore always show
code that runs against the current package.

- **MQDM junction** (`docs/lit/mqdm.jl`): Green function of a non-interacting quantum dot
  connected to two metal leads, complexity benchmark, and average current under bias.
- **SQDS junction** (`docs/lit/sqds.jl`): Green function of a quantum dot connected to two
  superconducting leads, and the transient current response to a voltage ramp.

Committed figures produced by the full-scale runs are kept here:

- `QD_benchmark.svg` — HSS vs no-compression scaling for the Metal–QD–Metal junction
- `average_current_QD.svg` — average current in the biased Metal–QD–Metal junction
- `transient_current_SQDS.svg` — transient current in the Superconductor–QD–Superconductor junction

To build the documentation locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path=dirname(pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```
