# 3+1 Foliation

Split spacetime indices into temporal and spatial components for
Scalar-Vector-Tensor (SVT) decomposition of metric perturbations.

## Foliation Setup

Register a foliation on a manifold, specifying which component is temporal
and which are spatial. A standard 4D spacetime uses component 0 as temporal
and 1, 2, 3 as spatial.

```@docs
FoliationProperties
define_foliation!
get_foliation
has_foliation
```

## Component Classification

After registering a foliation, classify individual component indices as
temporal or spatial, and enumerate all component indices.

```@docs
classify_component
all_components
```

## Spacetime Splitting

Replace abstract spacetime indices with explicit sums over temporal and spatial
components. For a rank-n tensor in d dimensions, this produces up to d^n terms.

```@docs
split_spacetime
split_all_spacetime
```

## SVT Substitution Rules

Map tensor components to their SVT decomposition. In Bardeen gauge (B=E=F=0),
the metric perturbation decomposes as:

| Component | Bardeen gauge         | Full gauge                                              |
|:----------|:----------------------|:--------------------------------------------------------|
| h\_00     | 2\Phi                 | 2\Phi                                                   |
| h\_0i     | S\_i                  | d\_i B + S\_i                                           |
| h\_ij     | 2\psi delta\_ij + hTT | 2\psi delta\_ij + 2 d\_i d\_j E + d\_i F\_j + d\_j F\_i + hTT |

```@docs
SVTSubstitution
svt_rules_bardeen
svt_rules_full
apply_svt
```

## Constraint Rules

Enforce transversality and tracelessness constraints on SVT fields:
- Transverse vectors: k\_i S^i = 0 (or d\_i S^i = 0)
- Transverse tensors: k\_i hTT\_{ij} = 0
- Traceless tensors: delta^{ij} hTT\_{ij} = 0

```@docs
svt_constraint_rules
lorentzian_contract
```

## Sector Collection

After SVT substitution, group terms by their SO(3) spin content: scalar
(Phi, B, psi, E), vector (S, F), tensor (hTT), mixed (cross-sector,
should vanish by Schur orthogonality), or pure\_scalar (no SVT fields).

```@docs
collect_sectors
```

## End-to-End Pipeline

The `foliate_and_decompose` function chains all steps: split indices,
apply SVT rules, enforce constraints, and collect by sector.

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    fol = define_foliation!(reg, :flat31; manifold=:M4)
    sectors = foliate_and_decompose(expr, :h; foliation=fol, gauge=:bardeen)
    # sectors[:scalar], sectors[:vector], sectors[:tensor]
end
```

```@docs
foliate_and_decompose
```
