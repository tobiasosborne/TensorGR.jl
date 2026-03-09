# TensorGR.jl -- Agent Reference

## Project Overview

Abstract tensor algebra and general relativity in Julia, equivalent to xAct for Mathematica.
Typed AST for symbolic tensor expressions, Butler-Portugal canonicalization (xperm.c FFI),
covariant derivative engine, perturbation theory, SVT/foliation decomposition, exterior calculus,
component computation, and CAS integration via weak dependencies.

- ~12,000 lines src (69 files), ~7,500 lines test (40 files), 3,534 tests, 12 benchmarks (152 pass)
- Extensions: ~400 lines (Symbolics.jl + SymEngine.jl weak deps, symbolic component pipeline)
- Docs: 17 files (Documenter.jl setup + 10 API ref pages + tutorial + xperm internals + CLAUDE.md)
- CI/CD: GitHub Actions for Julia 1.10/1.11

## Quick Commands

```bash
julia --project -e 'using Pkg; Pkg.test()'                      # full test suite
julia -t4 --project=benchmarks benchmarks/run_all.jl --tier 3   # all benchmarks (tiers 1-3)
julia --project docs/make.jl                                     # build docs
```

## Architecture

### Core Type Hierarchy

```
TensorExpr (abstract)
  +-- Tensor(name::Symbol, indices::Vector{TIndex})      # single tensor
  +-- TProduct(scalar::Rational{Int}, factors::Vector{TensorExpr})  # product with coefficient
  +-- TSum(terms::Vector{TensorExpr})                    # sum
  +-- TDeriv(index::TIndex, arg::TensorExpr, covd::Symbol)  # derivative (default covd=:partial)
  +-- TScalar(val::Any)                                  # scalar value
```

- `TIndex(name::Symbol, position::IndexPosition, vbundle::Symbol)` -- Up/Down, default `:Tangent`
- Constructors: `up(:a)`, `down(:b)`, `up(:a, :SU2)`, `down(:b, :SU2)`

### Registry

`TensorRegistry` is the central mutable container holding all definitions:

```
TensorRegistry
  manifolds::Dict{Symbol, ManifoldProperties}
  tensors::Dict{Symbol, TensorProperties}
  rules::Vector{Any}               # RewriteRule instances
  vbundles::Dict{Symbol, VBundleProperties}
  foliations::Dict{Symbol, Any}    # FoliationProperties
  tex_aliases::Dict{Tuple{Symbol,Int}, Symbol}
  metric_cache::Dict{Symbol, Symbol}   # manifold => metric name
  delta_cache::Dict{Symbol, Symbol}    # manifold => delta name
```

Thread-safe via `task_local_storage`. Each task gets its own registry stack:
- `current_registry()` -- returns active registry (falls back to global)
- `with_registry(f, reg)` -- scoped registry context

### Simplify Pipeline

Fixed-point loop: `expand_products -> contract_metrics -> contract_curvature -> canonicalize -> [commute_covds] -> collect_terms -> apply_rules`

- `simplify(expr; registry=reg, maxiter=100, parallel=false, commute_covds_name=nothing)`
- `simplify(expr; parallel=true)` -- TSum-level threading (threshold: 20 terms)

### xperm.c (Butler-Portugal Canonicalization)

- Compiled C library at `deps/libxperm.so`, loaded via ccall
- `_ensure_lib_loaded` uses ReentrantLock for thread-safe dlopen
- `schreier_sims` reallocs -- must use `Libc.malloc` for buffer
- `canonical_perm` expects names (initial slot positions) for freeps/dummyps
- Perm uses `p.data[i]` not `p[i]` -- no indexing support on Perm struct
- All-free mode: all indices as free, dummies normalized separately

## Source Layout (layer by layer)

```
src/types.jl, registry.jl, show.jl          # Core types, registry, display
src/xperm/                                   # Butler-Portugal canonicalization (C FFI)
  permutations.jl, wrapper.jl
src/ast/                                     # AST primitives
  indices.jl, walk.jl
src/escape_hatch.jl                          # to_expr, from_expr, validate
src/rules.jl                                 # RewriteRule engine
src/gr/symmetries.jl                         # Symmetry types + generators
src/algebra/                                 # Core algebra
  arithmetic.jl                              #   +, -, *, scalar ops
  contraction.jl                             #   metric contraction
  canonicalize.jl                            #   xperm canonicalization
  derivatives.jl                             #   expand_derivatives
  simplify.jl                                #   simplify pipeline
  ibp.jl                                     #   integration by parts
  collect_tensors.jl                         #   collect/filter tensors
  ansatz.jl                                  #   tensor ansatz generation
  trace.jl                                   #   abstract trace
  symmetrize.jl                              #   symmetrize/antisymmetrize
  young.jl                                   #   Young tableaux
  solve.jl                                   #   solve tensor equations
src/scalar/                                  # Scalar algebra
  algebra.jl, functions.jl, simplify_cas.jl
src/gr/                                      # GR objects
  curvature.jl                               #   Riemann, Ricci, Weyl, Einstein
  bianchi.jl                                 #   Bianchi identity rules
  metric.jl                                  #   Metric engine (define, freeze, conformal)
  covd.jl                                    #   Covariant derivatives
  sort_covds.jl                              #   CovD sorting, commutation
  box.jl                                     #   Box, grad_squared, covd_chain
  lie.jl                                     #   Lie derivatives
  killing.jl                                 #   Killing vectors
  hypersurface.jl                            #   Hypersurface embedding
  mapping.jl                                 #   Smooth maps, pullback, pushforward
  topological.jl                             #   Pontryagin, Euler, Chern-Simons
  conversions.jl                             #   Curvature basis conversions
src/perturbation/                            # Perturbation theory
  partitions.jl, linearize.jl
  metric_perturbation.jl                     #   MetricPerturbation struct
  expand.jl                                  #   expand_perturbation (arbitrary order)
  gauge.jl                                   #   Gauge transformations
  variation.jl                               #   Functional/variational derivatives
  backgrounds.jl                             #   Max symmetric, vacuum backgrounds
  isaacson.jl                                #   GW stress-energy
src/svt/                                     # SVT decomposition
  fourier.jl, decompose.jl, projectors.jl
src/foliation/                               # 3+1 foliation
  foliation.jl, decompose.jl
  svt_rules.jl, constraints.jl, sectors.jl
src/action/                                  # Quadratic action analysis
  quadratic_action.jl, extract_quadratic.jl
  spin_projectors.jl                         #   Barnes-Rivers P2/P1/P0s/P0w
src/exterior/                                # Exterior calculus
  forms.jl, operations.jl, connection_forms.jl
src/components/                              # Component computation
  basis.jl, ctensor.jl, metric_compute.jl
  values.jl, to_basis.jl
src/worldline/                               # Worldline / PN
  worldline.jl
src/macros/                                  # Convenience macros
  tensor_macro.jl, definitions.jl
src/parser/                                  # LaTeX input parser
  latex_parser.jl
ext/                                         # Weak dependency extensions
  TensorGRSymbolicsExt.jl                    #   Symbolics.jl integration
  TensorGRSymEngineExt.jl                    #   SymEngine.jl integration
```

## Key API Functions (by module)

### Setup

- `@manifold M4 dim=4 metric=g` -- registers manifold + metric + delta + Tangent VBundle
- `define_curvature_tensors!(reg, :M4, :g)` -- registers Riem, Ric, RicScalar, Ein, Weyl
- `with_registry(reg) do ... end` -- scoped registry context
- `register_tensor!`, `register_manifold!`, `has_tensor`, `get_tensor`

### Building Expressions

- `Tensor(:name, [up(:a), down(:b)])` -- named tensor with indices
- `TProduct(scalar, factors)` / `tproduct(s, factors)` -- smart product (collapses single-factor)
- `TSum(terms)` / `tsum(terms)` -- smart sum
- `TDeriv(index, arg, covd)` -- derivative (default covd=:partial)
- `TScalar(val)` -- scalar value

### Simplification

- `simplify(expr; registry=reg, maxiter=100, parallel=false, commute_covds_name=nothing)`
- Pipeline: expand_products -> contract_metrics -> contract_curvature -> canonicalize -> collect_terms -> apply_rules
- `simplify(expr; parallel=true)` -- TSum-level threading (PARALLEL_THRESHOLD=20)

### Covariant Derivatives

- `@covd D on=M4 metric=g` or `define_covd!(reg, ...)`
- `covd_to_christoffel(expr, :D)` -- expand CovD into Christoffel symbols
- `commute_covds(expr, :D)` -- sort derivatives, produce Riemann commutator terms
- `change_covd(expr, :D1, :D2)` -- change connection (difference tensor)

### Curvature

- `riemann_to_weyl`, `weyl_to_riemann` -- Weyl decomposition
- `ricci_to_einstein`, `einstein_to_ricci` -- Einstein tensor
- `schouten_to_ricci`, `ricci_to_schouten` -- Schouten tensor
- `to_riemann(expr)`, `to_ricci(expr)` -- curvature basis conversion
- `contract_curvature(expr)` -- contract Riemann products using symmetries
- `kretschmann_expr`, `cotton_expr`, `tensor_norm`
- `pontryagin_density`, `euler_density`, `chern_simons_action` -- topological invariants

### Perturbation Theory

- `define_metric_perturbation!(reg, :g, :h)` -- g -> g + epsilon*h
- `expand_perturbation(expr, mp, order)` -- arbitrary-order perturbation expansion
- `delta_christoffel`, `delta_riemann`, `delta_ricci`, `delta_ricci_scalar` -- curvature perturbations
- `maximally_symmetric_background!`, `vacuum_background!` -- set backgrounds
- `variational_derivative`, `euler_lagrange` -- functional derivatives
- `gauge_transformation` -- gauge change of perturbation
- `isaacson_average` -- GW stress-energy (short-wavelength averaging)

### Equation Solver

- `solve_tensors(equation, unknowns; registry=reg, make_rules=true, take_traces=false)`
- Solves linear tensor equations, returns `Vector{RewriteRule}`
- Supports single/multiple unknowns, systems of equations

### Exterior Calculus

- `define_form!(reg, :A; manifold=:M4, degree=1)`
- `wedge`, `exterior_d`, `interior_product`, `hodge_dual`, `codifferential`
- `connection_form`, `curvature_form` -- Cartan structure equations

### Components

- `define_chart!(reg, :Schw; manifold=:M4, coords=[...])`
- `CTensor(data, metric_info)` -- component tensor with inverse/det/trace
- `metric_christoffel`, `metric_riemann`, `metric_ricci`, `metric_ricci_scalar`
- `metric_einstein`, `metric_weyl`, `metric_kretschmann`
- `to_basis(expr, chart)`, `component_array`

### Foliation (3+1 Decomposition)

- `define_foliation!(reg, :flat31; manifold=:M4, temporal=0, spatial=[1,2,3])`
- `split_spacetime`, `split_all_spacetime` -- decompose into temporal/spatial
- `svt_rules_bardeen`, `svt_rules_full` -- SVT substitution rules
- `collect_sectors` -- group by SVT field type
- `foliate_and_decompose(expr, :h; foliation=fol)` -> `Dict{Symbol,TensorExpr}`

### Quadratic Action

- `QuadraticForm`, `quadratic_form`, `propagator`, `determinant`
- `spin2_projector`, `spin1_projector`, `spin0s_projector`, `spin0w_projector` -- Barnes-Rivers

### Smooth Maps (Pullback/Pushforward)

- `define_mapping!(reg, :φ; domain=:M, codomain=:N)` -- register map + Jacobian tensor
- `pullback(T, :φ)` -- contract covariant indices with Jacobian dφ^i_a
- `pushforward(U, :φ)` -- contract contravariant indices with inverse Jacobian
- `pullback_metric(:φ, :g)` -- convenience for induced metric

### Symbolic Components (Symbolics.jl extension)

- `symbolic_diagonal_metric(coords, diag)` / `symbolic_metric(coords, g)` -- build SymbolicMetric
- `symbolic_christoffel`, `symbolic_riemann`, `symbolic_ricci`, `symbolic_ricci_scalar`
- `symbolic_einstein`, `symbolic_kretschmann`, `symbolic_curvature_from_metric`
- All require `using Symbolics` (weak dependency, loaded via extension)

### Rules and Display

- `make_rule(lhs, rhs; use_symmetries=true)` -- auto symmetry variants
- Pattern indices: `down(:a_)` matches any index; `RewriteRule(Tensor(:T,[down(:a_),down(:b_)]), ...)`
- `apply_rules`, `apply_rules_fixpoint`, `is_pattern_variable`
- `set_vanishing!(reg, :T)` -- set tensor to zero
- `to_latex(expr)`, `to_unicode(expr)`

### Other

- `define_vbundle!(reg, :SU2; manifold=:M4, dim=3, indices=[...])` -- vector bundles
- `define_hypersurface!`, `induced_metric_expr`, `extrinsic_curvature_expr`
- `define_killing!(reg, :xi; manifold=:M4, metric=:g)` -- Killing vectors
- `lie_derivative(v, expr)`, `lie_bracket`, `lie_to_covd`
- `box()`, `grad_squared()`, `covd_chain()`, `covd_product()`
- `Worldline`, `define_worldline!`, `pn_order`, `truncate_pn`
- `parse_tex`, `@tex_str` -- LaTeX input parsing

## Critical Implementation Notes

These are hard-won lessons. Violating them causes subtle bugs or test regressions.

### AST Construction

- `tproduct(1//1, [x])` collapses to `x` -- use multi-factor products to trigger xperm
- TDeriv has `.covd` field (default `:partial`) -- MUST propagate in ALL TDeriv reconstructions
- `_split_scalar(TScalar)`: non-Rational values (Symbol, Int) are opaque cores, NOT coefficients
- `_ImplodedObject` has `.covd` field (default `:partial` via compat constructor)

### Canonicalization

- Do NOT sort TSum terms or batch-rename in `_normalize_dummies` -- breaks benchmark term counts
- Do NOT sort deriv chains in `canonicalize(TDeriv)` -- fights xperm canonical ordering, causes non-convergence
- `_normalize_dummies` sorts partial derivative chains via `_sort_partial_chains` for d^2=0

### Registry

- Registry has `metric_cache`/`delta_cache` (Dict{Symbol,Symbol}); populated by `register_tensor!`
- `@manifold` auto-registers metric + delta + Tangent vbundle; do not re-register delta manually
- Thread-safety: task_local_storage for registry stack, ReentrantLock for xperm lib loading

### Perturbation Engine

- `_expand_pert(tensor, mp, 0)` returns the tensor itself (background value), NOT zero
- `delta_riemann`/`delta_ricci`/`delta_ricci_scalar` return ZERO at order 0 -- never call them for background
- Ric^{ab} (both up) handled via g^{ac}g^{bd}Ric_{cd} decomposition in expand.jl
- Riemann in perturbation engine requires first index Up (R^a_{bcd}) or all-down (R_{abcd})

### CAS Extensions

- Symbolics.jl is in [weakdeps] and test/Project.toml (NOT [deps])
- Extension overrides dispatch hooks (`_simplify_scalar_val`, `_try_simplify_entry`) -- cannot overwrite methods with same signature as base
- `_sym_mul/_sym_add/_sym_sub/_sym_neg/_sym_div` dispatch on `Symbolics.Num` in extension

### xperm.c FFI

- `schreier_sims` reallocs -- must use `Libc.malloc` for buffer
- `_ensure_lib_loaded` has ReentrantLock for thread-safe dlopen
- Perm uses `p.data[i]` not `p[i]` -- no indexing support on Perm struct
- Full algorithm docs at `docs/xperm_algorithm.md`

### Stubs (Not Yet Implemented)

- `sort_covds_to_box`, `sort_covds_to_div`, `lorentzian_contract` -- return input unchanged
- 3 `@test_skip` benchmarks: spherical harmonics, bitensors (stretch goals)

## Testing

- 40 test files in `test/` loaded by `test/runtests.jl`
- 12 benchmarks in `benchmarks/` (run_all.jl, `--tier 1/2/3`)
- 3 `@test_skip` benchmarks (stretch goals: spherical harmonics, bitensors)
- Benchmark ground truth: pinned term counts (especially bench_12 for 6-deriv gravity on dS)

## Git / Workflow

- Main branch: `master`
- Uses beads (`bd`) for issue tracking (see AGENTS.md)
- Session end: `bd sync && git push`
- LinearAlgebra is a stdlib dep (no compat entry needed)
- Unicode symbols like dagger cause ParseError -- use ASCII alternatives (e.g., `_dag`)
