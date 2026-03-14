# TensorGR.jl

A Julia package for abstract tensor algebra and general relativity calculations, providing feature parity with the core functionality of [xAct](http://www.xact.es/) (Mathematica).

TensorGR.jl comprises approximately 12,100 lines of source code across 71 files, with 2,900+ tests and 12 benchmarks. It supports Julia 1.10 and 1.11.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/tobiasosborne/TensorGR.jl")
```

For symbolic component calculations with Symbolics.jl (optional):

```julia
Pkg.add("Symbolics")
```

## Quick Start

```julia
using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)

    # Metric contraction: g^{ab} g_{bc} = delta^a_c
    result = simplify(Tensor(:g, [up(:a), up(:b)]) * Tensor(:g, [down(:b), down(:c)]))
    println(to_unicode(result))  # delta^a_c

    # Riemann antisymmetry: R_{abcd} + R_{bacd} = 0
    R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
    R2 = Tensor(:Riem, [down(:b), down(:a), down(:c), down(:d)])
    println(simplify(R1 + R2))   # 0

    # Contracted Bianchi identity
    for r in bianchi_rules(); register_rule!(reg, r); end
    bianchi = simplify(TDeriv(up(:a), Tensor(:Ein, [down(:a), down(:b)])))
    println(bianchi)             # 0
end
```

## Feature Overview

### Abstract Tensor Algebra

- **Typed AST**: five node types (`Tensor`, `TProduct`, `TSum`, `TDeriv`, `TScalar`) with smart constructors that enforce normal forms
- **Index manipulation**: symmetry-aware canonicalization via xperm.c (Butler-Portugal algorithm), metric contraction, automatic dummy renaming
- **Rewrite rules**: pattern-matching rule engine with fixed-point iteration and automatic symmetry variant generation
- **Simplification pipeline**: expand products, contract metrics, canonicalize, collect terms, apply rules -- iterated to convergence
- **Parallel simplification**: `simplify(expr; parallel=true)` for multi-threaded TSum-level parallelism (threshold: 20 terms)
- **Symmetrization**: symmetrize/antisymmetrize over index sets, Young tableaux decomposition into irreducible representations
- **Tensor ansatz**: enumerate all independent contractions for a given index structure, build general linear combinations with symbolic coefficients
- **Equation solver**: solve linear tensor equations for unknown tensors, returning rewrite rules

### Differential Geometry & General Relativity

- **Metric engine**: one-call setup (`define_metric!`) for metric, inverse, delta, epsilon, Christoffel, curvature tensors, and Bianchi rules
- **Covariant derivatives**: Christoffel expansion, derivative commutation with Riemann terms, connection change
- **Curvature algebra**: Riemann-Weyl decomposition, Schouten, trace-free Ricci, Einstein, Kretschmann scalar, curvature basis conversions
- **Lie derivatives**: Lie derivative of arbitrary tensors, Lie bracket, conversion to covariant derivative form
- **Killing vectors**: define Killing fields with automatic Killing equation metadata
- **Hypersurface geometry**: induced metric, extrinsic curvature, projector, normal normalization
- **Smooth maps**: pullback, pushforward via explicit Jacobian contraction, induced metric
- **Product manifolds**: M1 x M2 with block-diagonal metric, curvature decomposition by factor
- **Topological invariants**: Pontryagin density, Euler (Gauss-Bonnet) density, Chern-Simons coupling
- **Differential operators**: box (d'Alembertian), gradient squared, derivative chains

### Perturbation Theory

- **Metric perturbation**: xPert-style arbitrary-order expansion via partition-based recursion
- **Curvature perturbations**: Christoffel, Riemann, Ricci, and scalar perturbations at any order
- **Background geometries**: maximally symmetric (de Sitter/AdS), vacuum (Ricci-flat), cosmological
- **Gauge transformations**: infinitesimal diffeomorphism gauge changes
- **Isaacson averaging**: short-wavelength averaging for gravitational wave stress-energy
- **Variational derivatives**: Euler-Lagrange equations, metric variation for deriving field equations

### Matter Fields

- **Perfect fluid**: stress-energy tensor with energy density, pressure, and 4-velocity, including normalization rules

### Component Calculations

- **Coordinate charts**: named coordinate systems on manifolds
- **CTensor arrays**: component tensors with contraction, trace, inverse, determinant, basis change
- **Metric curvature**: Christoffel, Riemann, Ricci, Einstein, Weyl, Kretschmann from metric components
- **Abstract-to-component**: convert abstract expressions to component arrays in a chart
- **Symbolic components**: Symbolics.jl extension for symbolic metric computations (weak dependency)

### Geodesics

- **Geodesic integration**: ODE right-hand side for numerical geodesic integration with DifferentialEquations.jl
- **Numerical Christoffel**: automatic finite-difference computation when no analytic form is available

### Exterior Calculus

- **Differential forms**: definition, wedge product, exterior derivative, interior product
- **Hodge dual**: star operator via Levi-Civita tensor
- **Codifferential**: adjoint of exterior derivative
- **Cartan structure equations**: connection and curvature forms in the vierbein/tetrad formalism

### Decomposition & Cosmological Perturbation Theory

- **3+1 foliation**: split spacetime indices into temporal and spatial, SVT substitution rules
- **SVT decomposition**: Fourier transforms, transverse/TT projectors
- **Barnes-Rivers projectors**: spin-2, spin-1, spin-0s, spin-0w projection operators for propagator analysis
- **Quadratic action**: extract kinetic matrix from Lagrangian, propagator inversion, determinant

### Other Features

- **Vector bundles**: per-index bundle tracking, cross-bundle contraction protection (gauge theory)
- **Worldline / PN**: point-particle EFT, post-Newtonian order counting, truncation
- **Integration by parts**: move derivatives off specified fields in product expressions
- **LaTeX parser**: `parse_tex`, `@tex_str` for reading LaTeX tensor notation
- **Display**: LaTeX (`to_latex`) and Unicode (`to_unicode`) output
- **Ergonomic macros**: `@tensor`, `@manifold`, `@define_tensor`, `@covd`
- **CAS integration**: Symbolics.jl and SymEngine.jl weak dependencies for scalar simplification

## Architecture

TensorGR uses a typed AST (`TensorExpr` hierarchy) with five node types:
- `Tensor` -- a single tensor with symbolic name and index slots
- `TProduct` -- a product of tensor expressions with rational coefficient
- `TSum` -- a sum of tensor expressions
- `TDeriv` -- a derivative operator applied to an expression
- `TScalar` -- a scalar value embedded in a tensor expression

Each index (`TIndex`) carries a name, position (Up/Down), and vector bundle (default `:Tangent`).

Canonicalization uses Jose Martin-Garcia's xperm.c (Butler-Portugal algorithm) compiled as a shared library and loaded via `ccall`. The `TensorRegistry` is the central mutable container holding manifold definitions, tensor metadata, rewrite rules, vector bundles, and foliations. Thread safety is provided via `task_local_storage` scoping for the registry and a `ReentrantLock` for xperm library loading.

The simplification pipeline runs a fixed-point loop: expand products over sums, contract metrics and deltas, detect curvature traces, canonicalize index ordering, optionally commute covariant derivatives, collect like terms, and apply registered rewrite rules.

## Contents

```@contents
Pages = [
    "tutorial.md",
    "api/types.md",
    "api/algebra.md",
    "api/gr.md",
    "api/matter.md",
    "api/perturbation.md",
    "api/components.md",
    "api/geodesics.md",
    "api/exterior.md",
    "api/foliation.md",
    "api/svt.md",
    "api/action.md",
    "api/ansatz.md",
    "api/advanced.md",
    "xperm_internals.md",
]
Depth = 2
```
