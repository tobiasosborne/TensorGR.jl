# Spinor Formalism

Two-component spinor calculus on Lorentzian 4-manifolds, including SL(2,C) spinor bundles, the soldering form (Infeld-van der Waerden symbol), curvature spinors (Weyl, Ricci, Lambda), the full Newman-Penrose null tetrad formalism with all 18 field equations and 11 Bianchi identities, and the Geroch-Held-Penrose (GHP) covariant formalism with spin/boost-weighted derivative operators.

## Spinor Setup

One-call setup of the complete spinor infrastructure: SL(2,C) bundles, spin metrics, and soldering form.

```@docs
define_spinor_structure!
```

## Spin Metric & Bundles

Register the SL(2,C) and conjugate SL(2,C)\_dot spinor vector bundles and the antisymmetric spin metric epsilon tensors.

```@docs
define_spinor_bundles!
spin_up
spin_down
spin_dot_up
spin_dot_down
is_spinor_index
is_dotted
conjugate_index
define_spin_metric!
spin_metric
```

## Soldering Form

The soldering form (Infeld-van der Waerden symbol) provides the isomorphism between the tangent bundle and the tensor product of the two spin bundles. Contraction rules for completeness and metric reconstruction are automatically registered.

```@docs
define_soldering_form!
to_spinor_indices
to_tensor_indices
```

## Curvature Spinors

The Riemann tensor decomposes into three irreducible spinor parts: the Weyl spinor Psi (self-dual Weyl), the Ricci spinor Phi (trace-free Ricci), and the scalar curvature spinor Lambda = R/24.

```@docs
define_curvature_spinors!
define_weyl_spinor!
define_ricci_spinor!
define_lambda_spinor!
lambda_spinor_expr
```

## Newman-Penrose Formalism

The Newman-Penrose formalism uses a null tetrad (l, n, m, mbar) to project all geometric quantities into complex scalars. Includes the 5 Weyl scalars, 9 Ricci scalars, 12 spin coefficients, 18 field equations (Ricci identities), and 11 Bianchi identities.

### Null Tetrad

```@docs
define_null_tetrad!
np_completeness
```

### Weyl & Ricci Scalars

```@docs
weyl_scalar
weyl_scalars
ricci_scalar_np
np_lambda
```

### Spin Coefficients

```@docs
spin_coefficient
all_spin_coefficients
```

### Directional Derivatives & Commutators

```@docs
np_directional_derivative
NPCommutatorRelation
np_commutator_table
```

### NP Field Equations (Ricci Identities)

```@docs
NPFieldEquation
np_field_equations
vacuum_np_field_equations
np_field_equation
```

### NP Bianchi Identities

```@docs
NPBianchiIdentity
np_bianchi_identities
vacuum_np_bianchi_identities
np_bianchi_identity
```

## GHP Formalism

The Geroch-Held-Penrose formalism replaces the NP directional derivatives with gauge-covariant operators that absorb the improper spin coefficients (epsilon, gamma, alpha, beta). Every GHP quantity carries a weight {p,q} under spin/boost transformations.

### Weights & Classification

```@docs
GHPWeight
spin_weight
boost_weight
ghp_weight
is_proper_ghp
```

### GHP Derivative Operators

```@docs
GHPDerivative
ghp_derivative
ghp_weight_shift
```

### GHP Field Equations

```@docs
GHPFieldEquation
ghp_field_equations
vacuum_ghp_field_equations
ghp_field_equation
ghp_field_equation_weight_consistent
```

### GHP Commutators

```@docs
GHPCommutatorRelation
ghp_commutator_table
ghp_commutator_weight_consistency
```
