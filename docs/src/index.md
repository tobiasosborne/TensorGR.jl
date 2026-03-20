# TensorGR.jl

A Julia package for abstract tensor algebra and general relativity calculations, providing feature parity with the core functionality of [xAct](http://www.xact.es/) (Mathematica).

TensorGR.jl comprises approximately 39,000 lines of source code across 167 files in 30 modules, with ~360,000 tests across 172 test files and 12 benchmarks. It supports Julia 1.10 and 1.11.

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
- **Hypersurface geometry**: induced metric, extrinsic curvature, projector, normal normalization, Gauss-Codazzi equations, Israel junction conditions, GHY boundary term
- **Smooth maps**: pullback, pushforward via explicit Jacobian contraction, induced metric
- **Product manifolds**: M1 x M2 with block-diagonal metric, curvature decomposition by factor
- **Topological invariants**: Pontryagin density, Euler (Gauss-Bonnet) density, Chern-Simons coupling
- **Differential operators**: box (d'Alembertian), gradient squared, derivative chains

### Spinor Formalism (`spinors/`, `xideal/`)

- **Spinor bundles**: unprimed/primed (dotted) spinor indices, spin metric, soldering form (Infeld-van der Waerden symbols)
- **Curvature spinors**: Weyl spinor, Ricci spinor, lambda spinor, irreducible decomposition
- **Newman-Penrose**: null tetrads, 12 spin coefficients, NP field equations, NP Bianchi identities, directional derivatives
- **GHP formalism**: GHP-weighted quantities, GHP derivatives (edth, edth-bar, thorn, thorn-bar), GHP commutators, GHP field equations
- **Spacetime classification**: Weyl scalars, Petrov type (I/II/III/D/N/O), Segre classification, energy condition checks
- See [Spinors API](@ref) for details.

### Perturbation Theory

- **Metric perturbation**: xPert-style arbitrary-order expansion via partition-based recursion
- **Curvature perturbations**: Christoffel, Riemann, Ricci, and scalar perturbations at any order
- **Background geometries**: maximally symmetric (de Sitter/AdS), vacuum (Ricci-flat), cosmological
- **Gauge transformations**: infinitesimal diffeomorphism gauge changes
- **Isaacson averaging**: short-wavelength averaging for gravitational wave stress-energy
- **Variational derivatives**: Euler-Lagrange equations, metric variation for deriving field equations

### Scalar-Tensor Theories (`scalar_tensor/`)

- **Horndeski gravity**: L2-L5 Lagrangians, metric and scalar field equations, kinetic scalar X
- **Beyond Horndeski**: quartic and quintic extensions, alpha_H parameter
- **DHOST**: degenerate higher-order scalar-tensor theories, degeneracy classification (Class I/II/III), DOF counting
- **EFT of dark energy**: Bellini-Sawicki alpha parameters, observational constraints (GW170817), stability conditions
- **Multi-field Horndeski**: multi-scalar extension, kinetic matrix, reduction to single field
- See [Scalar-Tensor API](@ref) for details.

### Feynman Rules (`feynman/`)

- **Graviton propagator**: de Donder gauge graviton propagator with numerator structure
- **Vertices**: n-point graviton vertices from perturbation expansion, matter-graviton couplings
- **Gauge fixing**: de Donder gauge fixing action, Faddeev-Popov ghost sector
- **Diagrams**: tree-exchange diagrams, contraction engine, symmetry factors
- **Loop integrals**: Passarino-Veltman topology, dimensional regularization traces, superficial divergence
- **PN matching**: Fourier transform table for potential matching, Newton potential extraction
- See [Feynman Rules API](@ref) for details.

### Metric-Affine Gravity (`metric_affine/`)

- **Independent connection**: affine connection with torsion and nonmetricity
- **Torsion**: decomposition into vector, axial, and tensor parts; contortion tensor
- **Nonmetricity**: Weyl vector, second trace, disformation tensor
- **Distortion**: full distortion tensor relating affine and Levi-Civita connections
- **Curvature**: independent connection Riemann tensor, decomposition into Riemannian + torsion/nonmetricity parts
- **Brauer decomposition**: 11-piece irreducible decomposition of the curvature tensor
- See [Metric-Affine API](@ref) for details.

### Spherical Harmonics (`harmonics/`)

- **Scalar harmonics**: Y_lm with product rules, conjugation, inner products
- **Vector harmonics**: even and odd vector harmonics, divergence/curl eigenvalues
- **Tensor harmonics**: even (Y, Z) and odd tensor harmonics, completeness
- **Angular integrals**: Gaunt integrals, angular selection rules, vector and tensor Gaunt coefficients
- **Clebsch-Gordan**: Wigner 3j symbols, Clebsch-Gordan coefficients
- **Harmonic decomposition**: decompose scalars, vectors, and symmetric tensors into harmonic modes
- See [Spherical Harmonics API](@ref) for details.

### PPN Formalism (`ppn/`)

- **PPN parameters**: all 10 PPN parameters, GR values, conservation/preferred-frame tests
- **Metric ansatz**: PPN metric components with superpotentials
- **Field equations**: solve PPN field equations for GR and scalar-tensor theories
- **Observables**: perihelion precession, light deflection, Shapiro delay, Nordtvedt effect, geodetic precession
- See [PPN API](@ref) for details.

### Hamiltonian / ADM (`hamiltonian/`)

- **ADM decomposition**: lapse, shift, spatial metric decomposition
- **Constraints**: Hamiltonian and momentum constraints
- **Poisson brackets**: canonical pairs, fundamental brackets
- See [Hamiltonian API](@ref) for details.

### Covariant Phase Space (`phase_space/`)

- **Noether currents**: Noether current and charge for diffeomorphism invariance
- **Symplectic structure**: symplectic potential, symplectic current, boundary ambiguities
- **Wald entropy**: Wald entropy integrand for general diffeomorphism-invariant theories
- **First law**: Hamiltonian variation, first law of black hole mechanics
- See [Phase Space API](@ref) for details.

### Invariants (`invariants/`)

- **RInv/DualRInv**: canonical forms for Riemann polynomial invariants, left/right/double duals
- **Simplification**: 6-level simplification algorithm (Garcia-Parrado and Martin-Garcia)
- **DDI**: dimensionally dependent identities, generalized delta, Gauss-Bonnet DDI
- See [Invariants API](@ref) for details.

### Bimetric Gravity (`bimetric/`)

- **Hassan-Rosen potential**: elementary symmetric polynomials, interaction potential
- **Linearization**: bimetric perturbation, mass matrix, mass eigenstates (massless + massive)
- **Higuchi bound**: Higuchi bound check for massive spin-2 on de Sitter
- See [Bimetric API](@ref) for details.

### Fermions / Clifford Algebra (`fermions/`)

- **Gamma matrices**: Clifford algebra relation, gamma-5, slash notation
- **Traces**: Dirac traces (2-point, 4-point), gamma-5 traces
- **Fierz identities**: Fierz matrix, Fierz coefficients, identity verification
- **Charge conjugation**: charge conjugation properties, Majorana condition
- See [Fermions API](@ref) for details.

### Frame Bundle / Tetrads (`tetrads/`)

- **Frame indices**: Lorentz frame index bundle, frame metric
- **Tetrad formalism**: frame bundle registration for vierbein/tetrad computations
- See [Tetrads API](@ref) for details.

### Matter Fields

- **Perfect fluid**: stress-energy tensor with energy density, pressure, and 4-velocity, including normalization rules
- **Equations of state**: barotropic, polytropic, tabular EOS

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
- **Gauge theory**: algebra-valued forms, field strength, Yang-Mills equations, Bianchi identity, instanton density, Chern-Simons forms

### Decomposition & Cosmological Perturbation Theory

- **3+1 foliation**: split spacetime indices into temporal and spatial, SVT substitution rules, Bianchi type I backgrounds, Bianchi structure constants (types I-IX)
- **SVT decomposition**: Fourier transforms, transverse/TT projectors, gauge choices (synchronous, Newtonian, flat slicing, comoving, uniform density)
- **Barnes-Rivers projectors**: spin-2, spin-1, spin-0s, spin-0w projection operators for propagator analysis; extended to vector, antisymmetric-2, and rank-3 fields
- **Quadratic action**: extract kinetic matrix from Lagrangian, propagator inversion, determinant, unitarity analysis, source constraints

### Other Features

- **Vector bundles**: per-index bundle tracking, cross-bundle contraction protection (gauge theory)
- **Worldline / PN**: point-particle EFT, post-Newtonian order counting, truncation
- **TOV solver**: neutron star structure with EOS coupling, mass-radius curves
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
    "examples.md",
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
    "api/spinors.md",
    "api/scalar_tensor.md",
    "api/feynman.md",
    "api/metric_affine.md",
    "api/harmonics.md",
    "api/invariants.md",
    "api/bimetric.md",
    "api/ppn.md",
    "api/hamiltonian.md",
    "api/phase_space.md",
    "api/fermions.md",
    "api/tetrads.md",
    "api/advanced.md",
    "xperm_internals.md",
]
Depth = 2
```
