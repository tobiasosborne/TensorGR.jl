# Scalar-Tensor Theories

Comprehensive scalar-tensor gravity framework covering Horndeski theory (the most general scalar-tensor theory with second-order equations of motion), beyond-Horndeski (GLPV) extensions, DHOST (Degenerate Higher-Order Scalar-Tensor) theories with degeneracy classification, the EFT of dark energy alpha parametrization, multi-field generalizations, and cosmological perturbation theory on FRW backgrounds.

## Horndeski Gravity

The four Horndeski Lagrangians L\_2 through L\_5 as abstract tensor expressions, with the scalar-tensor function G\_i(phi, X) and its derivatives.

```@docs
ScalarTensorFunction
g_tensor_name
differentiate_G
HorndeskiTheory
define_horndeski!
kinetic_X
horndeski_L2
horndeski_L3
horndeski_L4
horndeski_L5
horndeski_lagrangian
```

## Equations of Motion

Metric and scalar field equations obtained by varying the Horndeski action. Despite the Lagrangian containing second derivatives of phi, the EOMs are second order in both the metric and the scalar field (Horndeski's theorem).

```@docs
horndeski_metric_eom
horndeski_scalar_eom
horndeski_eom
```

## Alpha Parameters

The Bellini-Sawicki alpha parametrization for the EFT of dark energy on spatially flat FRW backgrounds.

```@docs
FRWBackground
define_frw_background
BelliniSawickiAlphas
compute_alphas
compute_alphas_numerical
```

## Beyond Horndeski

The Gleyzes-Langlois-Piazza-Vernizzi (GLPV) extension of Horndeski theory with two additional Lagrangians that propagate only 3 DOF via degeneracy conditions.

```@docs
BeyondHorndeskiTheory
define_beyond_horndeski!
beyond_horndeski_L4
beyond_horndeski_L5
beyond_horndeski_lagrangian
alpha_H
```

## DHOST

Degenerate Higher-Order Scalar-Tensor (DHOST) class I Lagrangian, quadratic in second covariant derivatives of the scalar field.

```@docs
DHOSTTheory
define_dhost!
dhost_L1
dhost_L2
dhost_L3
dhost_L4
dhost_L5
dhost_lagrangian
```

## DHOST Degeneracy

Algebraic constraints on DHOST coefficient functions ensuring 3 propagating degrees of freedom (no Ostrogradsky ghost), with class identification and Horndeski embedding.

```@docs
degeneracy_conditions
is_degenerate
dhost_class
horndeski_as_dhost
reduce_to_horndeski
dhost_dof_count
```

## EFT of Dark Energy

Unifying effective field theory parametrization of dark energy and modified gravity on FRW backgrounds via five time-dependent alpha functions.

```@docs
EFTDarkEnergy
eft_from_horndeski
eft_from_beyond_horndeski
eft_from_numerical
eft_stability
eft_observables
gw170817_constraint
eft_gr
eft_quintessence
eft_fR
apply_gw170817
```

## Multi-field

Multi-field Horndeski theory with N scalar fields, a field-space metric, and kinetic matrix.

```@docs
MultiScalarTensorFunction
multi_g_tensor_name
differentiate_MG
MultiHorndeskiTheory
define_multi_horndeski!
kinetic_matrix
kinetic_matrix_full
multi_horndeski_L2
multi_horndeski_L3
multi_horndeski_L4
to_single_field
```

## Quadratic Action

Quadratic action for scalar-tensor perturbations on FRW, with stability conditions and sound speeds.

```@docs
ScalarTensorQuadraticAction
StabilityConditions
tensor_sound_speed
scalar_sound_speed
quadratic_action_horndeski
stability_conditions
check_stability
to_quadratic_form
```
