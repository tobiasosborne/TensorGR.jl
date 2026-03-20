# Curvature Invariants (Invar)

Canonical representation and simplification of scalar curvature invariants built from products of Riemann tensors. Implements the RInv contraction permutation encoding, dual (Hodge) invariants via DualRInv, the six-level Invar simplification algorithm (permutation symmetries, Bianchi identities, dimensionally-dependent identities, derivative commutation), and DDI rewrite rule generation.

## RInv Canonical Forms

Contraction permutation representation for scalar Riemann monomials. A degree-k invariant is encoded as a fixed-point-free involution on 4k slots, canonicalized under the combined Riemann slot-symmetry and inter-factor permutation group.

```@docs
RInv
canonicalize
canonicalize_rinv
are_equivalent
rinv_symmetry_group
to_tensor_expr
from_tensor_expr
```

## DualRInv

Curvature invariants involving the Levi-Civita tensor (Hodge duals of the Riemann tensor). Extends RInv with left, right, and double dual specifications.

```@docs
DualRInv
left_dual
right_dual
double_dual
pontryagin_rinv
```

## Simplify Levels (Invar Algorithm)

The six-level Invar simplification algorithm for scalar curvature invariants, following Garcia-Parrado and Martin-Garcia (2007).

```@docs
simplify_level1
simplify_level2
simplify_level3
simplify_level4
simplify_level5
is_riemann_monomial
count_riemann_degree
apply_bianchi_cyclic
bianchi_relation
differential_bianchi
contracted_bianchi
```

## DDI Rules

Dimensionally-dependent identity (DDI) generation and application. DDIs arise from the vanishing of the generalized Kronecker delta when its rank exceeds the manifold dimension.

```@docs
generate_ddi_rules
gauss_bonnet_ddi
register_ddi_rules!
has_ddi_rules
generate_riemann_ddi
riemann_ddi_expr
simplify_with_ddis
```
