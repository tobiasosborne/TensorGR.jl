# Advanced Features

## Product Manifolds

Define product manifolds `M = M_1 x M_2` with block-diagonal metric, additive scalar curvature, and factor-wise curvature decomposition. Factor curvature tensors use metric-suffixed names (e.g. `Riem_g1`, `Ric_g1`).

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M1 dim=2 metric=g1
    @manifold M2 dim=2 metric=g2
    define_product_manifold!(reg, :M; factors=[:M1, :M2])

    # Block-diagonal metric
    g = product_metric(:M)

    # Scalar curvature is additive: R = R_1 + R_2
    R = product_scalar_curvature(:M)

    # Factor Ricci (mixed components vanish)
    Ric1 = product_ricci(:M, :M1)

    # Factor Einstein with cross-scalar: G1_{ij} - (1/2) R2 g1_{ij}
    G1 = product_einstein(:M, :M1)

    # All factor Einstein equations
    eqs = product_einstein_equations(:M)
end
```

```@docs
ProductManifoldProperties
define_product_manifold!
has_product_manifold
get_product_manifold
product_metric
product_scalar_curvature
product_ricci
product_riemann
product_einstein
product_einstein_equations
```

## Smooth Maps (Pullback / Pushforward)

Define smooth maps between manifolds and compute pullbacks and pushforwards of tensor fields. The Jacobian tensor is automatically registered.

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M dim=3 metric=g
    @manifold N dim=4 metric=h

    # Register map phi: M -> N
    define_mapping!(reg, :phi; domain=:M, codomain=:N)

    # Pullback: contract covariant indices with Jacobian
    T = Tensor(:T, [down(:a), down(:b)])
    pb = pullback(T, :phi)

    # Pushforward: contract contravariant indices with inverse Jacobian
    U = Tensor(:U, [up(:a)])
    pf = pushforward(U, :phi)

    # Induced metric (pullback of codomain metric)
    gamma = pullback_metric(:phi, :h)
end
```

```@docs
MappingProperties
define_mapping!
get_mapping
has_mapping
pullback
pushforward
pullback_metric
```

## Hypersurface Boundaries and Junction Conditions

### Boundary Terms

The Gibbons-Hawking-York boundary term is needed for a well-posed variational principle. Integration by parts with boundary contributions is also supported.

```julia
# GHY boundary term: integral_Sigma K sqrt(gamma) d^3x
ghy = ghy_boundary_term(:K, :gamma)

# IBP with boundary: integral(d_a phi * psi) = -integral(phi * d_a psi) + boundary
result = ibp_with_boundary(expr, :phi)
```

### Junction Conditions

Israel junction conditions for thin shells relate the jump in extrinsic curvature across a hypersurface to the surface stress-energy tensor.

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_hypersurface!(reg, :Sigma; ambient=:M4, metric=:g, signature=-1)

    # Register junction condition tensors
    jd = define_junction!(reg, :Sigma)

    # Israel junction equation: [K_{ab}] - gamma_{ab} [K] + 8pi S_{ab} = 0
    eq = israel_equation(:Kp, :Km, :gamma_jc)

    # Solve for surface stress-energy
    S = junction_stress_energy(:Kp, :Km, :gamma_jc)
end
```

```@docs
JunctionData
define_junction!
israel_equation
junction_stress_energy
```

## Worldline / Post-Newtonian

The worldline module represents point-particle trajectories x^a(s) in
spacetime for the Effective Field Theory of Post-Newtonian Gravity
(EFTofPNG). Each worldline carries a particle label, curve parameter,
velocity tensor, and position tensor. The velocity is normalised:
g\_{ab} v^a v^b = -1 (timelike).

Post-Newtonian (PN) order counting tracks powers of the velocity tensor:
each v counts as O(epsilon) where epsilon ~ v/c. PN order n corresponds
to O(v^{2n}).

```@docs
Worldline
define_worldline!
pn_order
truncate_pn
```

## CAS Integration

TensorGR provides hooks for Computer Algebra System backends. When
Symbolics.jl is loaded (as a weak dependency), scalar simplification
dispatches through `Symbolics.simplify`, and the symbolic arithmetic
operators (`_sym_mul`, `_sym_add`, etc.) gain methods for `Symbolics.Num`.

Without a CAS backend, these functions are no-ops or operate on Julia
`Number` / `Expr` trees directly.

```@docs
simplify_scalar
simplify_quadratic_form
symbolic_quadratic_form
to_fourier_symbolic
```

### Extension Architecture

The base package defines stub functions and dispatch hooks:

- `_simplify_scalar_val(x)` -- identity by default, extended for `Expr` types
- `_try_simplify_entry(x)` -- identity by default, extended for matrix entries
- `_sym_mul`, `_sym_add`, `_sym_sub`, `_sym_neg`, `_sym_div` -- dispatch on `Number`
  by default, extended for `Symbolics.Num`

The Symbolics extension (`ext/TensorGRSymbolicsExt.jl`) adds methods that
convert `Expr` trees to `Symbolics.Num`, simplify, and convert back. It also
provides `to_symbolics` / `from_symbolics` for explicit conversion and
`sym_eval(::Symbolics.Num, vars)` for numeric evaluation.

```@docs
to_symbolics
from_symbolics
to_symengine
from_symengine
```

## Parallel Simplification

Pass `parallel=true` to `simplify` to parallelise TSum-level operations
across Julia threads:

```julia
result = simplify(expr; parallel=true)
```

**How it works**: When a `TSum` has at least `PARALLEL_THRESHOLD` (= 20)
terms and more than one Julia thread is available, each simplification step
(expand, contract, canonicalize) is applied to individual terms via
`Threads.@spawn`. Term collection uses a two-phase approach:
parallel canonicalization followed by serial dictionary merge.

**Thread safety**: The tensor registry uses `task_local_storage` scoping,
so each spawned task inherits the calling task's registry without data
races. The xperm.c shared library uses a `ReentrantLock` for thread-safe
`dlopen`.

## Macros

Ergonomic macros for common definitions and tensor construction.

```@docs
@tensor
@manifold
@define_tensor
@covd
```

## LaTeX Parser

Parse LaTeX tensor expressions into `TensorExpr` AST nodes. Supports
standard index notation with `^{}` and `_{}`, Greek letters, partial
derivatives (`\partial`), and common tensor names.

```@docs
parse_tex
@tex_str
```

## Display

Render tensor expressions as LaTeX source or Unicode strings.

```@docs
to_latex
to_unicode
```

## Escape Hatch

Convert between `TensorExpr` and Julia `Expr` for interop with other
packages, and validate expression well-formedness.

```@docs
to_expr
from_expr
is_well_formed
validate
```
