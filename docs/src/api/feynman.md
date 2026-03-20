# Feynman Rules & EFT

Perturbative quantum gravity infrastructure: Feynman diagram types, graviton propagator and vertices from the expanded Einstein-Hilbert action, gauge-fixing and Faddeev-Popov ghosts, matter-graviton coupling vertices, tensor index contraction engine, loop integral representations with dimensional regularization, and post-Newtonian potential extraction via Fourier transform matching.

## Types

The core type hierarchy for Feynman diagrams: vertices, propagators, complete diagrams, and contracted amplitudes.

```@docs
TensorVertex
TensorPropagator
FeynmanDiagram
DiagramAmplitude
n_loops
build_diagram
tree_exchange_diagram
vertex_from_perturbation
contract_diagram
```

## Graviton Propagator

The graviton propagator in harmonic (de Donder) gauge with the standard numerator tensor structure.

```@docs
propagator_numerator
graviton_propagator
```

## Graviton Vertices

The n-point graviton vertices from expanding the Einstein-Hilbert action around flat space. The cubic vertex has 12 independent tensor structures after imposing Bose symmetry.

```@docs
graviton_3vertex
graviton_4vertex
graviton_vertex_n
```

## Gauge Fixing

De Donder (harmonic) gauge-fixing action and Faddeev-Popov ghost sector for perturbative gravity.

```@docs
gauge_fixing_condition
gauge_fixing_action
fp_operator
ghost_propagator
ghost_graviton_vertex
gauge_fixed_kinetic_operator
```

## Matter Vertices

Matter-graviton coupling vertices for point particles and minimally-coupled scalar fields, derived from expanding the worldline action and the scalar kinetic action.

```@docs
matter_graviton_vertex
scalar_matter_vertex
```

## Index Contraction

Extended contraction engine for Feynman diagrams: single-line contraction, loop momentum identification, momentum conservation, and symmetry factor computation.

```@docs
contract_line
find_loop_momenta
MomentumConstraint
momentum_constraints
impose_momentum_conservation
symmetry_factor
```

## Loop Integrals

Loop integral representations with dimensional regularization, Passarino-Veltman scalar master integrals, and topology identification.

```@docs
PropagatorDenom
MomentumIntegral
ScalarIntegral
integral_topology
pv_topology
to_momentum_integral
massless_bubble
massless_triangle
dimreg_trace
total_propagator_power
superficial_divergence
```

## PN Matching

Post-Newtonian potential extraction via Fourier transform of scattering amplitudes, with standard 3D Fourier transform tables.

```@docs
FourierEntry
fourier_transform_potential
newton_potential_coeff
PNPotentialTerm
classify_pn_order
```
