# Types & Registry

## Core Types

```@docs
TensorExpr
Tensor
TProduct
TSum
TDeriv
TScalar
TIndex
IndexPosition
```

## Registry

```@docs
TensorRegistry
TensorProperties
ManifoldProperties
register_manifold!
register_tensor!
register_rule!
current_registry
with_registry
```

## Rules

```@docs
RewriteRule
apply_rules
apply_rules_fixpoint
```
