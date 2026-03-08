# Advanced Features

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

### Extension architecture

The base package defines stub functions and dispatch hooks:

- `_simplify_scalar_val(x)` -- identity by default, extended for `Expr` types
- `_try_simplify_entry(x)` -- identity by default, extended for matrix entries
- `_sym_mul`, `_sym_add`, `_sym_sub`, `_sym_neg`, `_sym_div` -- dispatch on `Number`
  by default, extended for `Symbolics.Num`

The Symbolics extension (`ext/TensorGRSymbolicsExt.jl`) adds methods that
convert `Expr` trees to `Symbolics.Num`, simplify, and convert back. It also
provides `to_symbolics` / `from_symbolics` for explicit conversion and
`sym_eval(::Symbolics.Num, vars)` for numeric evaluation.

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

**Performance**: On an 8-thread machine, the 6-derivative gravity benchmark
(6 cubic curvature invariants) drops from 164s serial to 66s parallel.

```@docs
simplify
```

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
