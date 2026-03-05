# TensorGR.jl

A Julia package for abstract tensor algebra and general relativity calculations, providing full feature parity with the core functionality of xAct (Mathematica).

## Features

- **Abstract tensor algebra**: typed AST with smart constructors, canonicalization via xperm.c
- **Index manipulation**: symmetry-aware canonicalization, metric contraction, dummy renaming
- **Covariant derivatives**: DefCovD, Christoffel expansion, derivative commutation with Riemann terms
- **Perturbation theory**: partition-based metric perturbation (xPert-style), inverse metric expansion
- **Component calculations**: CTensor arrays, Christoffel/Riemann/Ricci from metric components
- **Curvature algebra**: Riemann-Weyl decomposition, Einstein/Ricci conversions, trace-free decomposition
- **Exterior calculus**: differential forms, wedge product, Hodge dual, exterior derivative
- **Rewrite rules**: pattern-matching rule engine with fixed-point iteration
- **Ergonomic macros**: `@tensor`, `@manifold`, `@define_tensor`, `@covd`

## Quick Start

```julia
using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)

    # Build tensor expressions
    R_ab = @tensor Ric[-a, -b]
    g_ab = @tensor g[-a, -b]
    R = @tensor RicScalar

    # Einstein tensor
    G = R_ab - (1//2) * g_ab * R

    # Simplify
    result = simplify(G)
end
```

## Installation

```julia
using Pkg
Pkg.add("TensorGR")
```

## Architecture

TensorGR uses a typed AST (`TensorExpr` hierarchy) with five node types:
- `Tensor` — a single tensor with symbolic name and index slots
- `TProduct` — a product of tensor expressions with rational coefficient
- `TSum` — a sum of tensor expressions
- `TDeriv` — a derivative operator applied to an expression
- `TScalar` — a scalar value embedded in a tensor expression

Canonicalization uses José Martin-Garcia's xperm.c (Butler-Portugal algorithm) compiled as a shared library, called via `ccall`.
