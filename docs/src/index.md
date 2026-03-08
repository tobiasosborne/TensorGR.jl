# TensorGR.jl

A Julia package for abstract tensor algebra and general relativity calculations, providing feature parity with the core functionality of [xAct](http://www.xact.es/) (Mathematica).

## Features

- **Abstract tensor algebra**: typed AST with smart constructors, canonicalization via xperm.c
- **Index manipulation**: symmetry-aware canonicalization, metric contraction, dummy renaming
- **Covariant derivatives**: DefCovD, Christoffel expansion, derivative commutation with Riemann terms
- **Curvature algebra**: Riemann-Weyl decomposition, Schouten, trace-free Ricci, Einstein, curvature conversions
- **Perturbation theory**: partition-based metric perturbation (xPert-style), inverse metric expansion, gauge transformations
- **Component calculations**: CTensor arrays, Christoffel/Riemann/Ricci/Einstein/Weyl/Kretschmann from metric components
- **Exterior calculus**: differential forms, wedge product, Hodge dual, exterior derivative, Cartan structure equations
- **Vector bundles**: define_vbundle!, per-index bundle tracking, cross-bundle contraction protection (gauge theory)
- **3+1 Foliation**: define_foliation!, split_spacetime, SVT substitution rules, constraint engine, sector collection
- **SVT decomposition**: Fourier transforms, transverse/TT projectors, Barnes-Rivers spin projectors (P2/P1/P0s/P0w)
- **Quadratic action**: extract kinetic matrix from Lagrangian, propagator inversion, symbolic determinant
- **Worldline / PN**: point-particle EFT, post-Newtonian order counting, truncation
- **CAS integration**: Symbolics.jl weak dependency for scalar simplification, symbolic quadratic forms, Fourier transforms
- **Parallel simplification**: `simplify(expr; parallel=true)` for multi-threaded TSum-level parallelism
- **Rewrite rules**: pattern-matching rule engine with fixed-point iteration
- **Display**: LaTeX and Unicode output
- **Ergonomic macros**: `@tensor`, `@manifold`, `@define_tensor`, `@covd`
- **LaTeX parser**: `parse_tex`, `@tex_str` for reading LaTeX tensor notation

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

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/tobiasosborne/TensorGR.jl")
```

## Architecture

TensorGR uses a typed AST (`TensorExpr` hierarchy) with five node types:
- `Tensor` -- a single tensor with symbolic name and index slots
- `TProduct` -- a product of tensor expressions with rational coefficient
- `TSum` -- a sum of tensor expressions
- `TDeriv` -- a derivative operator applied to an expression
- `TScalar` -- a scalar value embedded in a tensor expression

Each index (`TIndex`) carries a name, position (Up/Down), and vector bundle (default `:Tangent`).

Canonicalization uses Jose Martin-Garcia's xperm.c (Butler-Portugal algorithm) compiled as a shared library.

## Contents

```@contents
Pages = [
    "tutorial.md",
    "api/types.md",
    "api/algebra.md",
    "api/gr.md",
    "api/perturbation.md",
    "api/components.md",
    "api/exterior.md",
    "api/foliation.md",
    "api/svt.md",
    "api/action.md",
    "api/advanced.md",
    "xperm_internals.md",
]
Depth = 2
```
