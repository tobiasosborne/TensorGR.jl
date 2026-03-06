# TensorGR.jl

**Abstract tensor algebra and general relativity in Julia.**

TensorGR.jl provides a complete symbolic tensor calculus system for general relativity, modeled on the capabilities of [xAct](http://www.xact.es/) (Mathematica). It features a typed expression tree, index canonicalization via the Butler-Portugal algorithm (xperm.c), covariant derivatives, perturbation theory, component calculations, exterior calculus, and vector bundle support.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Features

| Module | Capabilities |
|--------|-------------|
| **Abstract Algebra** | Typed AST (`TensorExpr` hierarchy), smart constructors, distributive expansion, term collection |
| **Canonicalization** | Slot symmetries via xperm.c (Butler-Portugal), Riemann symmetry, fully (anti)symmetric tensors |
| **Index Contraction** | Metric contraction engine, Kronecker delta elimination, automatic raising/lowering |
| **Covariant Derivatives** | `define_covd!`, Christoffel expansion, derivative commutation with Riemann curvature terms |
| **Curvature Algebra** | Riemann/Weyl decomposition, Schouten, trace-free Ricci, Einstein, `to_riemann`/`to_ricci` conversions |
| **Perturbation Theory** | Arbitrary-order metric perturbation (xPert-style), Leibniz partition recursion, gauge transformations |
| **Component Calculations** | `CTensor` arrays, Christoffel/Riemann/Ricci/Einstein/Weyl/Kretschmann from metric components |
| **Exterior Calculus** | Differential forms, wedge product, exterior derivative, Hodge dual, Cartan structure equations |
| **Vector Bundles** | `define_vbundle!`, per-index bundle tracking, cross-bundle contraction protection |
| **Rewrite Rules** | Pattern-matching rule engine, Bianchi identities, background field equations |
| **Display** | LaTeX and Unicode output |
| **Macros** | `@tensor`, `@manifold`, `@define_tensor`, `@covd` |

## Quick Start

```julia
using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    # Define a 4D manifold with metric g
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)

    # Metric contraction: g^{ab} g_{bc} = delta^a_c
    result = simplify(Tensor(:g, [up(:a), up(:b)]) * Tensor(:g, [down(:b), down(:c)]))
    println(to_unicode(result))  # delta^a_c

    # Riemann antisymmetry: R_{abcd} + R_{bacd} = 0
    R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
    R2 = Tensor(:Riem, [down(:b), down(:a), down(:c), down(:d)])
    println(simplify(R1 + R2))   # 0

    # Contracted Bianchi identity: nabla^a G_{ab} = 0
    for r in bianchi_rules(); register_rule!(reg, r); end
    bianchi = simplify(TDeriv(up(:a), Tensor(:Ein, [down(:a), down(:b)])))
    println(bianchi)             # 0
end
```

## Examples

Seven runnable example scripts are provided in [`examples/`](examples/):

| Script | Description |
|--------|-------------|
| [`01_getting_started.jl`](examples/01_getting_started.jl) | Manifold setup, metric contraction, Riemann symmetries |
| [`02_covariant_derivatives.jl`](examples/02_covariant_derivatives.jl) | CovD expansion, Christoffel symbols, Bianchi identity |
| [`03_curvature_decomposition.jl`](examples/03_curvature_decomposition.jl) | Weyl decomposition, Schouten, Gauss-Bonnet invariant |
| [`04_perturbation_theory.jl`](examples/04_perturbation_theory.jl) | Linearized gravity, metric perturbation, vacuum background |
| [`05_schwarzschild.jl`](examples/05_schwarzschild.jl) | Schwarzschild metric, Christoffel symbols, Kretschmann scalar |
| [`06_exterior_calculus.jl`](examples/06_exterior_calculus.jl) | Forms, wedge product, Hodge dual, Cartan formula |
| [`07_gauge_theory.jl`](examples/07_gauge_theory.jl) | SU(2) vector bundle, mixed indices, Yang-Mills structure |

Run any example:

```bash
julia --project examples/01_getting_started.jl
```

## Architecture

TensorGR uses a typed abstract syntax tree with five node types:

```
TensorExpr (abstract)
  |-- Tensor       # Named tensor with index slots: R_{abcd}
  |-- TProduct     # Product with rational coefficient: (1/2) g^{ab} T_{bc}
  |-- TSum         # Sum of expressions: R_{ab} + g_{ab} R
  |-- TDeriv       # Derivative operator: partial_a(T_{bc})
  |-- TScalar      # Scalar value: 3//2, symbolic
```

Each index (`TIndex`) carries a name, position (Up/Down), and vector bundle:

```julia
TIndex(:a, Up, :Tangent)    # spacetime index
TIndex(:A, Up, :SU2)        # gauge bundle index
```

### Simplification Pipeline

The `simplify` function runs a fixed-point loop:

1. **expand_products** -- distribute `*` over `+`
2. **contract_metrics** -- eliminate metric and delta contractions
3. **canonicalize** -- canonical index ordering via xperm.c
4. **collect_terms** -- combine like terms (with dummy normalization)
5. **apply_rules** -- registered rewrite rules

### Canonicalization

Index canonicalization uses Jose Martin-Garcia's [xperm.c](http://www.xact.es/xPerm/), implementing the Butler-Portugal algorithm for finding canonical representatives of index permutations under slot symmetry groups. The C library is compiled to `deps/libxperm.so` and called via `ccall`.

### Registry System

All tensor metadata (manifold, rank, symmetries) is stored in a `TensorRegistry` with context-based scoping via `with_registry`:

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    @define_tensor T on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    # ... computations use this registry
end
```

## Installation

TensorGR.jl is not yet registered in the Julia General registry. Install from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/tobiasosborne/TensorGR.jl")
```

Or in development mode:

```bash
git clone https://github.com/tobiasosborne/TensorGR.jl
cd TensorGR.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

**Requirements:** Julia 1.10+ and a C compiler (for building xperm.c on first use).

## Documentation

Full documentation is available in [`docs/`](docs/):

- [Tutorial](docs/src/tutorial.md) -- worked examples from basic to advanced
- [API Reference](docs/src/api/) -- complete function documentation
- [xperm Internals](docs/src/xperm_internals.md) -- how the canonicalization engine works

Build the docs locally:

```bash
julia --project=docs docs/make.jl
```

## Testing

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The test suite has **2,553 tests** covering all modules.

## Comparison with xAct

TensorGR.jl aims for feature parity with the core xAct packages:

| xAct Package | TensorGR.jl Equivalent |
|-------------|----------------------|
| xTensor | `TensorExpr` AST, `indices`, `free_indices`, `dummy_pairs` |
| xPerm | `xperm/` (compiled C library), `canonicalize` |
| xCoba | `components/` module, `CTensor`, `metric_compute` |
| xPert | `perturbation/` module, `expand_perturbation`, `MetricPerturbation` |
| Invar | `gr/conversions.jl`, `contract_curvature`, `kretschmann_expr` |

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.
