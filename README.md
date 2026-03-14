# TensorGR.jl

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-blue.svg)](https://julialang.org/)

**Abstract tensor algebra and general relativity in Julia.**

TensorGR.jl is a symbolic tensor calculus system for general relativity, providing capabilities comparable to [xAct](http://www.xact.es/) for Mathematica. It features a typed abstract syntax tree for tensor expressions, Butler-Portugal index canonicalization via xperm.c, a covariant derivative engine with curvature commutation, arbitrary-order metric perturbation theory, component computation, exterior calculus, 3+1 foliation with SVT decomposition, particle spectrum analysis via Barnes-Rivers spin projection, geodesic integration, stellar structure (TOV equations), and CAS integration through Symbolics.jl and SymEngine.jl extensions.

## Features

| Module | Capabilities |
|--------|-------------|
| **Abstract Algebra** | Typed AST (`TensorExpr` hierarchy), smart constructors, distributive expansion, term collection |
| **Canonicalization** | Slot symmetries via xperm.c (Butler-Portugal), Riemann symmetry, fully (anti)symmetric tensors |
| **Index Contraction** | Metric contraction engine, Kronecker delta elimination, automatic raising/lowering |
| **Covariant Derivatives** | `define_covd!`, Christoffel expansion, derivative commutation with Riemann curvature terms |
| **Curvature Algebra** | Riemann/Weyl decomposition, Schouten, trace-free Ricci, Einstein, Cotton tensor, `to_riemann`/`to_ricci` conversions |
| **Curvature Invariants** | Catalog of scalar invariants through cubic order (R, R^2, Ric^2, Kretschmann, Weyl^2, plus 6 cubic: I1-I6 including Goroff-Sagnotti) |
| **Curvature Syzygies** | Gauss-Bonnet identity (Riem^2 -> 4Ric^2 - R^2), Weyl vanishing (D<=3), dimension-dependent identities |
| **Perturbation Theory** | Arbitrary-order metric perturbation (xPert-style), Leibniz partition recursion, gauge transformations, Isaacson averaging |
| **Equation Solver** | `solve_tensors` -- solve linear tensor equations for unknowns, return `RewriteRule`s |
| **Component Calculations** | `CTensor` arrays with inverse/det/trace, Christoffel/Riemann/Ricci/Einstein/Weyl/Kretschmann from metric components |
| **Exterior Calculus** | Differential forms, wedge product, exterior derivative, Hodge dual, codifferential, Cartan structure equations |
| **3+1 Foliation** | `define_foliation!`, spacetime splitting, SVT decomposition, constraint engine, sector collection |
| **Quadratic Action** | `QuadraticForm`, spin projectors (Barnes-Rivers P2/P1/P0s/P0w), propagator analysis |
| **Kernel Extraction** | Position-space kernel extraction (`extract_kernel_direct`) with correct two-momentum physics, spin projection for particle spectrum |
| **Topological Invariants** | Pontryagin density, Euler density, Chern-Simons action |
| **Hypersurface / ADM** | `define_hypersurface!`, arbitrary-codimension submanifolds, induced metric, extrinsic curvature, projector |
| **Gauss-Codazzi** | Gauss equation, Codazzi-Mainardi equation, rewrite rules for intrinsic/extrinsic curvature relations |
| **GHY Boundary Term** | `ghy_boundary_term` for well-posed variational principle, `ibp_with_boundary` preserving surface terms |
| **Israel Junction** | `define_junction!`, `israel_equation`, `junction_stress_energy` for thin-shell matching conditions |
| **Product Manifolds** | `define_product_manifold!` for M1 x M2 with block-diagonal metric, additive scalar curvature, cross-scalar Einstein equations |
| **Smooth Maps** | `define_mapping!`, pullback/pushforward via Jacobian tensors, cross-manifold vector bundles |
| **Killing Vectors** | `define_killing!` with automatic registration of Killing equation as rewrite rules |
| **Matter / EOS** | Perfect fluid stress-energy tensor, `BarotropicEOS`, `PolytropicEOS`, `TabularEOS`, `PerfectFluid` coupling |
| **Symmetry Ansaetze** | `SphericalSymmetry`, `AxialSymmetry` (Lewis-Papapetrou), `StaticSymmetry`, `HomogeneousIsotropy` (FLRW) |
| **Geodesics** | `setup_geodesic`, `geodesic_rhs!`, `integrate_geodesic` with `GeodesicSolution` (DifferentialEquations.jl) |
| **TOV Solver** | `setup_tov`, `tov_rhs!` for neutron star structure with EOS coupling |
| **Spectrum Analysis** | Bueno-Cano parameters, `dS_spectrum_6deriv` for 6-derivative gravity on de Sitter, flat-space form factors |
| **Vector Bundles** | `define_vbundle!`, per-index bundle tracking, cross-bundle contraction protection |
| **Worldline / PN** | `Worldline` struct, PN order tracking, `truncate_pn` |
| **CAS Integration** | Symbolics.jl and SymEngine.jl extensions for scalar simplification and symbolic components |
| **Parallel Simplify** | `simplify(expr; parallel=true)` for TSum-level threading |
| **Rewrite Rules** | Pattern-matching rule engine with pattern indices, Bianchi identities, background field equations |
| **Display** | LaTeX and Unicode output, `parse_tex` / `@tex_str` input |
| **Macros** | `@tensor`, `@manifold`, `@define_tensor`, `@covd` |

## Quick Start

```julia
using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    # Define a 4D manifold with metric g and curvature tensors
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

**Optional dependencies:** `Symbolics.jl` (symbolic component computation, metric ansaetze), `SymEngine.jl` (scalar simplification), `DifferentialEquations.jl` (geodesic integration, TOV solver).

## Examples

The [`examples/`](examples/) directory contains runnable scripts demonstrating the major features:

| Script | Description |
|--------|-------------|
| [`01_getting_started.jl`](examples/01_getting_started.jl) | Manifold setup, metric contraction, Riemann symmetries |
| [`02_covariant_derivatives.jl`](examples/02_covariant_derivatives.jl) | CovD expansion, Christoffel symbols, Bianchi identity |
| [`03_curvature_decomposition.jl`](examples/03_curvature_decomposition.jl) | Weyl decomposition, Schouten, Gauss-Bonnet invariant |
| [`04_perturbation_theory.jl`](examples/04_perturbation_theory.jl) | Linearized gravity, metric perturbation, vacuum background |
| [`05_schwarzschild.jl`](examples/05_schwarzschild.jl) | Schwarzschild metric components, Christoffel symbols, Kretschmann scalar |
| [`06_exterior_calculus.jl`](examples/06_exterior_calculus.jl) | Forms, wedge product, Hodge dual, Cartan formula |
| [`07_gauge_theory.jl`](examples/07_gauge_theory.jl) | SU(2) vector bundle, mixed indices, Yang-Mills structure |
| [`08_postquantum_gravity.jl`](examples/08_postquantum_gravity.jl) | Post-quantum gravity action analysis |
| [`09_compare_with_reference.jl`](examples/09_compare_with_reference.jl) | Comparison with reference computations |
| [`10_onsager_machlup_R2_RicciSq.jl`](examples/10_onsager_machlup_R2_RicciSq.jl) | Onsager-Machlup functional with R^2 and Ricci-squared |
| [`11_6deriv_gravity_dS.jl`](examples/11_6deriv_gravity_dS.jl) | 6-derivative gravity on de Sitter (parallel simplify) |
| [`12_product_manifolds.jl`](examples/12_product_manifolds.jl) | Product manifold M1 x M2, block-diagonal metric, curvature decomposition |
| [`13_6deriv_particle_spectrum.jl`](examples/13_6deriv_particle_spectrum.jl) | Particle spectrum of 6-derivative gravity via Barnes-Rivers projection |
| [`14_cubic_bc_params.jl`](examples/14_cubic_bc_params.jl) | Bueno-Cano parameters for all 6 cubic curvature invariants |
| [`26_6deriv_spectrum_showcase.jl`](examples/26_6deriv_spectrum_showcase.jl) | Three independent paths for 6-derivative spectrum (spin projection, SVT, Bueno-Cano) |
| [`27_geodesic_orbits.jl`](examples/27_geodesic_orbits.jl) | Geodesic orbits in Schwarzschild spacetime with energy conservation check |

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
3. **contract_curvature** -- apply Riemann symmetry identities
4. **canonicalize** -- canonical index ordering via xperm.c
5. **collect_terms** -- combine like terms (with dummy normalization)
6. **apply_rules** -- registered rewrite rules

### Canonicalization

Index canonicalization uses Jose Martin-Garcia's [xperm.c](http://www.xact.es/xPerm/), implementing the Butler-Portugal algorithm for finding canonical representatives of index permutations under slot symmetry groups. The C library is compiled to `deps/libxperm.so` and called via `ccall`.

### Registry System

All tensor metadata (manifold, rank, symmetries) is stored in a `TensorRegistry` with thread-safe context-based scoping via `with_registry`:

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    @define_tensor T on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    # ... computations use this registry
end
```

For a complete description of the architecture, source layout, and implementation notes, see [CLAUDE.md](CLAUDE.md).

## Documentation

Full documentation is available in [`docs/`](docs/):

- [Tutorial](docs/src/tutorial.md) -- worked examples from basic to advanced
- [API Reference](docs/src/api/) -- complete function documentation
- [xperm Internals](docs/src/xperm_internals.md) -- how the canonicalization engine works

Build the docs locally:

```bash
julia --project=docs docs/make.jl
```

## Testing and Benchmarks

Run the full test suite:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The test suite covers ~3,500 tests across 50 test files.

Run benchmarks (tiered by complexity):

```bash
julia -t4 --project=benchmarks benchmarks/run_all.jl --tier 1   # fast sanity checks
julia -t4 --project=benchmarks benchmarks/run_all.jl --tier 2   # intermediate
julia -t4 --project=benchmarks benchmarks/run_all.jl --tier 3   # all benchmarks (including 6-derivative gravity on dS)
```

12 benchmarks with 152 benchmark tests cover expression sizes from single contractions to 6-derivative gravity Lagrangians with hundreds of terms.

## Comparison with xAct

TensorGR.jl aims for feature parity with the core xAct packages:

| xAct Package | TensorGR.jl Equivalent |
|-------------|----------------------|
| xTensor | `TensorExpr` AST, `indices`, `free_indices`, `dummy_pairs` |
| xPerm | `xperm/` (compiled C library), `canonicalize` |
| xCoba | `components/` module, `CTensor`, `metric_compute` |
| xPert | `perturbation/` module, `expand_perturbation`, `MetricPerturbation` |
| xTras | `solve_tensors`, `collect_tensors`, `make_ansatz`, `all_contractions` |
| Invar | `gr/invariants.jl`, `curvature_invariant`, `list_invariants`, `contract_curvature`, `kretschmann_expr` |

## Citation

If you use TensorGR.jl in your research, please cite:

```bibtex
@software{TensorGR.jl,
  title  = {TensorGR.jl: Abstract Tensor Algebra and General Relativity in Julia},
  author = {Tobias Osborne},
  url    = {https://github.com/tobiasosborne/TensorGR.jl},
  year   = {2025}
}
```

Key references for the algorithms used:

- J. M. Martin-Garcia, R. Portugal, L. R. U. Manssur, "The Invar tensor package", *Comp. Phys. Comm.* **177** (2007) 640-648. [xPerm canonicalization]
- J. M. Martin-Garcia, "xPerm: fast index canonicalization for tensor computer algebra", *Comp. Phys. Comm.* **179** (2008) 597-603. [Butler-Portugal algorithm]
- D. Brizuela, J. M. Martin-Garcia, G. A. Mena Marugan, "xPert: Computer algebra for metric perturbation theory", *Gen. Rel. Grav.* **41** (2009) 2415-2431. [Perturbation theory]

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.
