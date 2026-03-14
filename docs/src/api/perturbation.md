# Perturbation Theory

This module implements the xPert-style perturbation expansion engine for general relativity: first-order linearization on flat backgrounds, arbitrary-order metric perturbation expansion via partition-based recursion, gauge transformations, background geometry rules, Isaacson averaging for gravitational wave stress-energy, and variational derivatives for deriving field equations from Lagrangians.

## First-Order Linearization

Closed-form expressions for the linearized curvature tensors on a flat background. These are efficient hard-coded formulas suitable for gravitational wave and post-Newtonian calculations.

```julia
a, b, c, d = down(:a), down(:b), down(:c), down(:d)

# Linearized Riemann: delta R_{abcd}
dR = deltaRiemann(a, b, c, d, :h)

# Linearized Ricci: delta R_{ab}
dRic = deltaRicci(a, b, :h)

# Linearized Ricci scalar: delta R
dR_scalar = deltaRicciScalar(:h)

# Full linearization of an expression
lin = linearize(expr, :g => (:eta, :h); order=1)
```

```@docs
linearize
δRiemann
δRicci
δRicciScalar
```

## Metric Perturbation

Define a metric perturbation `g = g_0 + epsilon * h` and expand expressions to arbitrary perturbation order. The engine uses partition-based recursion for the inverse metric, Christoffel symbols, and curvature tensors at order `n`.

```julia
reg = TensorRegistry()
# ... register manifold and metric ...

# Define perturbation: g = g_bg + epsilon * h
mp = define_metric_perturbation!(reg, :g, :h)

# Perturb a tensor expression to order 1
perturbed = perturb(Tensor(:g, [down(:a), down(:b)]), mp, 1)
# => Tensor(:h, [down(:a), down(:b)])

# Inverse metric perturbation: delta(g^{ab}) = -g^{ac} g^{bd} h_{cd}
dinv = δinverse_metric(mp, up(:a), up(:b), 1)
```

```@docs
MetricPerturbation
define_metric_perturbation!
perturb
δinverse_metric
```

### Curvature Perturbations

Perturbation of the Christoffel symbol, Riemann tensor, Ricci tensor, and Ricci scalar at arbitrary order `n`. These use memoized partition-based recursion internally.

```julia
# First-order Christoffel perturbation
dGamma = δchristoffel(mp, up(:a), down(:b), down(:c), 1)

# First-order Riemann perturbation
dRiem = δriemann(mp, up(:a), down(:b), down(:c), down(:d), 1)

# Second-order Ricci perturbation
dRic2 = δricci(mp, down(:a), down(:b), 2)

# Second-order Ricci scalar perturbation
dR2 = δricci_scalar(mp, 2)
```

```@docs
δchristoffel
δriemann
δricci
δricci_scalar
```

### Full Expansion

Walk an expression tree and expand all curvature tensors at the given perturbation order. Dispatches on tensor name (Riem, Ric, RicScalar, Christoffel) and uses the Leibniz rule for products.

```julia
# Expand Einstein-Hilbert Lagrangian to second order
L_EH = Tensor(:RicScalar, TIndex[])
dL2 = expand_perturbation(L_EH, mp, 2)
```

```@docs
expand_perturbation
```

### Tensor Perturbations

Define perturbations of arbitrary (non-metric) tensors and query the perturbative order of an expression.

```julia
# Define a perturbation of a matter field
define_tensor_perturbation!(reg, :T, :deltaT)

# Count perturbation order
order = perturbation_order(expr, Set([:h, :deltaT]))
```

```@docs
define_tensor_perturbation!
perturbation_order
```

## Background Solutions

Register background field equations that set curvature tensors to specific values. These are applied as rewrite rules during simplification.

```julia
# Vacuum background: Ric = 0, R = 0 (Schwarzschild, Kerr)
vacuum_background!(reg, :M4)

# Maximally symmetric background (de Sitter / anti-de Sitter)
maximally_symmetric_background!(reg, :M4; cosmological_constant=:Lambda)

# Cosmological background (FRW)
cosmological_background!(reg, :M4)
```

```@docs
background_solution!
maximally_symmetric_background!
cosmological_background!
vacuum_background!
```

## Gauge Transformations

Apply infinitesimal diffeomorphism gauge transformations to perturbation expressions. At first order, the gauge change of the metric perturbation is the Lie derivative of the background metric along the gauge vector.

```julia
xi = Tensor(:xi, [up(:a)])
h_gauge = gauge_transformation(h_expr, xi, :g; order=1)
# h'_{ab} = h_{ab} + nabla_a xi_b + nabla_b xi_a
```

```@docs
gauge_transformation
```

## Isaacson Averaging

Short-wavelength averaging for gravitational wave effective stress-energy. Keeps only bilinear (quadratic) terms in the perturbation tensor, discarding linear and background terms.

```julia
# Average: keep only h*h terms, discard h and background
T_eff = isaacson_average(delta2_G_ab, :h)
```

```@docs
isaacson_average
```

## Variational Derivatives

Compute functional derivatives of Lagrangian densities with respect to fields, producing Euler-Lagrange equations. Also provides metric variation for deriving gravitational field equations.

```julia
# Euler-Lagrange equation: delta L / delta phi = 0
eom = variational_derivative(lagrangian, :phi)

# Euler-Lagrange for multiple fields
eoms = euler_lagrange(lagrangian, [:phi, :psi])

# Metric variation: delta L / delta g^{ab}
field_eq = metric_variation(lagrangian, :g, down(:a), down(:b))

# Vary a full Lagrangian density with respect to the metric
var = var_lagrangian(lagrangian, :g)
```

```@docs
variational_derivative
euler_lagrange
metric_variation
var_lagrangian
```

## Partition Combinatorics

Integer partition and composition utilities used internally by the perturbation engine. Exposed for advanced users building custom perturbation schemes.

```@docs
sorted_partitions
all_compositions
multinomial
```
