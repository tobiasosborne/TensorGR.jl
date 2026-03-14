# GR Objects

This module provides the differential geometry and general relativity infrastructure: metric and curvature tensor definitions, covariant derivatives with Christoffel symbols, Lie derivatives, Killing vectors, hypersurface geometry, curvature conversions, topological densities, curvature invariant catalog, and syzygies. Together these give a complete symbolic GR toolkit analogous to xAct's xTensor + xCoba.

## Metric Infrastructure

The `define_metric!` function is the primary entry point: it registers the metric tensor, Kronecker delta, Levi-Civita epsilon, all standard curvature tensors, the Levi-Civita covariant derivative with Christoffel symbols, and the Bianchi identity rules -- all in a single call.

```julia
reg = TensorRegistry()
mp = ManifoldProperties(:M4, 4, :g, :d, [:a,:b,:c,:d,:e,:f])
register_manifold!(reg, mp)
define_metric!(reg, :g; manifold=:M4, signature=lorentzian(4))
```

### Metric Signature

```@docs
MetricSignature
lorentzian
euclidean
sign_det
```

### Metric Definition and Control

```@docs
define_metric!
metric_signature
set_flat!
is_flat
freeze_metric!
unfreeze_metric!
is_frozen
```

### Metric Operations

Separate a metric from a tensor (insert explicit metric for raising/lowering), compute the metric determinant symbolically, and construct generalized Kronecker deltas.

```julia
# Insert explicit metric to separate index :a
result = separate_metric(T_expr, :a, :g)

# Generalized Kronecker delta: delta^{ab}_{cd} = det(delta matrix)
gd = gdelta([up(:a), up(:b)], [down(:c), down(:d)])
```

```@docs
separate_metric
metric_det_expr
sqrt_det_expr
gdelta
expand_gdelta
```

### Conformal Metrics

```@docs
set_conformal_to!
```

## Curvature Tensors

Register the standard curvature tensors (Riemann, Ricci, Ricci scalar, Einstein, Weyl, Schouten) and construct curvature expressions.

```julia
# Define all curvature tensors for a manifold
define_curvature_tensors!(reg, :M4, :g)

# Build Einstein tensor expression: G_{ab} = R_{ab} - (1/2) g_{ab} R
G = einstein_expr(down(:a), down(:b), :g)
```

```@docs
define_curvature_tensors!
einstein_expr
ricci_from_riemann
cotton_expr
tensor_norm
```

## Curvature Invariant Catalog

A structured catalog of scalar curvature invariants at orders 1 (linear), 2 (quadratic), and 3 (cubic in curvature). The catalog includes Ricci scalar, Kretschmann, Weyl squared, and all 6 independent cubic invariants (including the Goroff-Sagnotti invariant).

```julia
# Look up an invariant by name
K = curvature_invariant(:Kretschmann)
W2 = curvature_invariant(:Weyl_sq; manifold=:M4, metric=:g)

# List all available invariants
list_invariants()
list_invariants(order=3)  # only cubic invariants
```

```@docs
InvariantEntry
INVARIANT_CATALOG
curvature_invariant
list_invariants
```

## Curvature Conversions

Convert between different curvature tensor representations. These functions return the algebraic decomposition expressions.

### Weyl Decomposition

```julia
# Riemann in terms of Weyl + Ricci + scalar
decomp = riemann_to_weyl(down(:a), down(:b), down(:c), down(:d), :g; dim=4)

# Inverse: Weyl in terms of Riemann - Ricci terms
weyl = weyl_to_riemann(down(:a), down(:b), down(:c), down(:d), :g; dim=4)
```

```@docs
riemann_to_weyl
weyl_to_riemann
```

### Einstein / Ricci / Schouten Conversions

```@docs
ricci_to_einstein
einstein_to_ricci
schouten_to_ricci
ricci_to_schouten
```

### Trace-free Ricci

```@docs
tfricci_expr
ricci_to_tfricci
```

### Unified Conversion Functions

Walk an entire expression tree and replace all curvature tensors with their decomposition in terms of a target basis.

```julia
# Convert everything to Riemann + Ricci + scalar
result = to_riemann(expr; metric=:g, dim=4)

# Convert everything to Ricci + scalar (contract Riemann traces)
result = to_ricci(expr; metric=:g, dim=4)
```

```@docs
to_riemann
to_ricci
contract_curvature
```

### Riemann from Christoffel

```@docs
riemann_to_christoffel
kretschmann_expr
```

## Curvature Syzygies

Syzygies are algebraic identities between products of curvature tensors that hold in specific dimensions. They complement Bianchi identities (which are linear in curvature) with quadratic and higher-order relations.

```julia
# Gauss-Bonnet in 4D: Riem^2 -> 4 Ric^2 - R^2
rules = gauss_bonnet_rule(; metric=:g, drop_euler=true)

# All applicable syzygies for a given dimension
rules = syzygy_rules(; dim=4, metric=:g)
```

```@docs
gauss_bonnet_rule
weyl_vanishing_rule
ricci_trace_rule
riemann_vanishing_rule
syzygy_rules
```

## Bianchi Identities

Bianchi identity rules are auto-registered by `define_metric!`. The contracted Bianchi identities enforce divergence-freeness of the Einstein tensor and relate the divergence of Ricci to the gradient of the scalar curvature.

```@docs
bianchi_rules
```

## Covariant Derivatives

Define covariant derivatives with associated Christoffel symbols, expand them into partial derivatives plus connection terms, change between different connections, and express Christoffel symbols in terms of metric derivatives.

```julia
# Define a Levi-Civita connection (done automatically by define_metric!)
covd = define_covd!(reg, :nabla; manifold=:M4, metric=:g)

# Expand: nabla_a V^b = d_a V^b + Gamma^b_{ac} V^c
expanded = covd_to_christoffel(expr, :nabla)

# Switch connections: nabla1 -> nabla2 + (Gamma1 - Gamma2) terms
changed = change_covd(expr, :nabla1, :nabla2)
```

```@docs
CovDProperties
define_covd!
get_covd
unregister_covd!
covd_to_christoffel
change_covd
christoffel_to_grad_metric
grad_metric_to_christoffel
```

### Covariant Derivative Commutation

Commute covariant derivatives into canonical order, generating Riemann curvature terms from the commutator. Can also symmetrize derivative pairs.

```julia
# Sort all CovDs alphabetically, inserting [nabla_a, nabla_b] = Riemann terms
sorted = commute_covds(expr, :nabla)

# Detect box operator patterns: d_a(d^a(T)) -> g^{ab} d_a d_b T
boxed = sort_covds_to_box(expr; metric=:g)
```

```@docs
commute_covds
sort_covds_to_box
sort_covds_to_div
symmetrize_covds
```

## Box Operator and Differential Operators

Convenience constructors for common differential operators on scalar fields.

```julia
phi = Tensor(:phi, TIndex[])

# Box (d'Alembertian): box(phi) = g^{ab} d_a d_b phi
box_phi = box(phi, :g)

# Gradient squared: (nabla phi)^2 = g^{ab} d_a phi d_b phi
grad2 = grad_squared(phi, :g)

# Derivative chain: d_a d_b d_c phi
chain = covd_chain(phi, [down(:a), down(:b), down(:c)])

# Derivative product: (d_a phi)(d_b phi)
prod = covd_product(phi, down(:a), down(:b))
```

```@docs
box
grad_squared
covd_chain
covd_product
```

## Lie Derivatives

Compute Lie derivatives along vector fields, Lie brackets, and convert between Lie and covariant derivative representations.

```julia
v = Tensor(:v, [up(:a)])
T = Tensor(:T, [down(:a), down(:b)])

# Lie derivative of a tensor along v
lie_T = lie_derivative(v, T)

# Lie bracket [v, w]
w = Tensor(:w, [up(:a)])
bracket = lie_bracket(v, w)
```

```@docs
lie_derivative
lie_bracket
lie_to_covd
```

## Killing Vectors

Define Killing vector fields. The Killing equation is registered as metadata; use `check_killing` to verify the Killing condition.

```julia
define_killing!(reg, :xi; manifold=:M4, metric=:g)
```

```@docs
define_killing!
check_killing
```

## Hypersurface Geometry

Embed codimension-1 hypersurfaces in an ambient manifold. Registers the unit normal, induced metric, extrinsic curvature, and projector tensors.

```julia
# Define a spacelike hypersurface (timelike normal, signature=-1)
hs = define_hypersurface!(reg, :Sigma;
    ambient=:M4, metric=:g, signature=-1)

# Build the induced metric expression
gamma_ab = induced_metric_expr(down(:a), down(:b), :g, :n; signature=-1)

# Build the extrinsic curvature
K_ab = extrinsic_curvature_expr(down(:a), down(:b), :n, :g)

# Build the projector
P = projector_expr(up(:a), down(:b), :n; signature=-1)
```

```@docs
SubmanifoldProperties
HypersurfaceProperties
define_submanifold!
define_hypersurface!
extrinsic_curvature_expr
induced_metric_expr
projector_expr
```

### Boundary Terms

Gibbons-Hawking-York boundary term and integration by parts with boundary contributions.

```@docs
ghy_boundary_term
ibp_with_boundary
```

### Gauss-Codazzi Relations

Relate the intrinsic curvature of a hypersurface to the ambient curvature via the Gauss equation and Codazzi equation.

```@docs
gauss_equation
codazzi_equation
gauss_codazzi_rules
```

## Topological Densities

Construct topological invariants in 4D: the Pontryagin (Chern-Pontryagin) density, the Euler (Gauss-Bonnet) density, and the Chern-Simons gravitational coupling.

```julia
# Pontryagin density: epsilon^{abcd} R_{ab}^{ef} R_{cdef}
P4 = pontryagin_density(:g)

# Euler density: R^2 - 4 Ric^2 + Riem^2 (Gauss-Bonnet in 4D)
E4 = euler_density(:g; dim=4)

# Chern-Simons coupling: theta * *(R wedge R)
CS = chern_simons_action(Tensor(:theta, TIndex[]), :g)
```

```@docs
pontryagin_density
euler_density
chern_simons_action
```
