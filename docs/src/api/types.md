# Types & Registry

TensorGR.jl is built on a typed AST (abstract syntax tree) for tensor expressions, a global registry for manifold and tensor metadata, and a pattern-matching rewrite rule system. This page documents all core types, the registry API, and the rule engine.

## Index Types

Indices carry a symbolic name, a position (contravariant or covariant), and an optional vector bundle tag. The helper functions `up` and `down` provide a concise way to create indices.

```julia
a = up(:a)              # TIndex(:a, Up)
b = down(:b)            # TIndex(:b, Down)
c = up(:c, :SU2)        # TIndex(:c, Up, :SU2)  -- non-tangent bundle
```

```@docs
IndexPosition
Up
Down
TIndex
up
down
```

## Expression Types

Every tensor expression is a subtype of `TensorExpr`. The five concrete types form a complete AST: atomic tensors, products with rational coefficients, sums, derivatives, and embedded scalars.

```julia
R   = Tensor(:Ric, [down(:a), down(:b)])
g   = Tensor(:g, [down(:a), down(:b)])
EH  = R - (1//2) * g * Tensor(:RicScalar, TIndex[])
```

```@docs
TensorExpr
Tensor
TProduct
TSum
TDeriv
TScalar
```

## Symmetry Specifications

Symmetry metadata is attached to tensors via `SymmetrySpec` values. These drive the canonicalization engine (xperm.c) and rule generation.

```julia
# Pair symmetry in slots (1,2)
Symmetric(1, 2)

# Riemann-type symmetry: antisymmetric in (1,2), (3,4), symmetric under pair swap, first Bianchi
RiemannSymmetry()

# Fully antisymmetric in all listed slots
FullyAntiSymmetric(1, 2, 3, 4)
```

```@docs
Symmetric
AntiSymmetric
PairSymmetric
RiemannSymmetry
FullySymmetric
FullyAntiSymmetric
SymmetrySpec
```

## Registry

The `TensorRegistry` is a mutable container that stores manifold definitions, tensor metadata (rank, symmetries, options), rewrite rules, vector bundles, and foliations. Each task maintains its own registry stack via `task_local_storage`, making TensorGR.jl thread-safe.

### Registry Type and Properties

```@docs
TensorRegistry
TensorProperties
ManifoldProperties
VBundleProperties
```

### Manifold Registration

Register a manifold before defining tensors on it. The `@manifold` macro (see [Macros](@ref)) auto-registers a manifold with metric, delta, and tangent bundle.

```julia
reg = TensorRegistry()
mp = ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f])
register_manifold!(reg, mp)
```

```@docs
register_manifold!
has_manifold
get_manifold
unregister_manifold!
```

### Tensor Registration

```julia
register_tensor!(reg, TensorProperties(
    name=:T, manifold=:M4, rank=(0, 2),
    symmetries=SymmetrySpec[Symmetric(1, 2)]))
```

```@docs
register_tensor!
has_tensor
get_tensor
unregister_tensor!
set_vanishing!
tex_alias!
```

### Vector Bundles

Vector bundles generalize the tangent bundle. Indices carry a `vbundle` tag, and cross-bundle contractions are prevented.

```julia
define_vbundle!(reg, :SU2; manifold=:M4, dim=3, indices=[:i,:j,:k])
```

```@docs
define_vbundle!
```

### Registry Scoping

The global registry is accessed via `current_registry()`. Use `with_registry` to run code with a specific registry active -- particularly important for multi-threaded workloads.

```julia
reg = TensorRegistry()
with_registry(reg) do
    # all tensor operations here use `reg`
end
```

```@docs
current_registry
with_registry
```

### Rule Registration

Rules added to a registry are applied during every `simplify` pass.

```@docs
register_rule!
```

## Rewrite Rules

The rule engine provides pattern-matching rewrite rules analogous to Mathematica's `UpValues`/`DownValues` system used by xAct. Rules can match structurally (by equality) or functionally (via a predicate).

### Rule Types

```julia
# Structural rule: replace exact expression
rule = RewriteRule(old_expr, new_expr)

# Functional rule: match by predicate, transform by function
rule = RewriteRule(
    expr -> expr isa Tensor && expr.name == :Ric,
    expr -> TScalar(0//1)
)

# With condition
rule = RewriteRule(pattern, replacement, condition_fn)
```

```@docs
RewriteRule
```

### Rule Application

```@docs
apply_rules
apply_rules_fixpoint
```

### Rule Construction

`make_rule` auto-generates symmetry variants when the LHS is a tensor with known symmetries. `folded_rule` composes two rule sets sequentially. The `@rule` macro provides syntactic sugar.

```julia
rules = make_rule(
    Tensor(:T, [down(:a), down(:b)]),
    Tensor(:S, [down(:a), down(:b)]);
    use_symmetries=true
)
```

```@docs
make_rule
folded_rule
@rule
```
