# Component Calculations

This module bridges abstract tensor algebra and concrete numerical/symbolic computation. It provides coordinate charts, component tensor arrays (`CTensor`), metric-based curvature computation from components, component value storage with symmetry awareness, basis definitions, conversion from abstract expressions to component representations, and the symbolic metric pipeline (via Symbolics.jl extension).

## Charts

A chart defines a coordinate system on a manifold with named coordinates. Charts are required for evaluating abstract tensor expressions to component arrays.

```julia
reg = TensorRegistry()
# ... register manifold :M4 ...
chart = define_chart!(reg, :spherical; manifold=:M4,
    coords=[:t, :r, :theta, :phi])
```

```@docs
ChartProperties
define_chart!
get_chart
```

## Component Tensors (CTensor)

`CTensor` is a multidimensional array tagged with a coordinate chart and index positions (Up/Down). It supports arithmetic, index contraction, trace, matrix inverse, determinant, and basis transformations.

```julia
# Create a 4x4 metric tensor in spherical coordinates
g_data = diagm([-1.0, 1.0, r^2, r^2*sin(theta)^2])
g = CTensor(g_data, :spherical, [Down, Down])

# Inverse metric
ginv = ctensor_inverse(g)

# Determinant
d = ctensor_det(g)

# Contract two CTensors on specified index slots
result = ctensor_contract(T, S, 2, 1)  # contract slot 2 of T with slot 1 of S

# Trace: contract two indices of the same tensor
tr = ctensor_trace(T, 1, 2)

# Basis change via Jacobian matrix
T_new = basis_change(T, jacobian_matrix)
```

```@docs
CTensor
ctensor_contract
ctensor_trace
ctensor_inverse
ctensor_det
```

## Metric Computations

Compute Christoffel symbols and curvature tensors from metric components. These functions operate on raw arrays and require a symbolic differentiation function (`deriv_fn`).

```julia
# Compute Christoffel symbols from metric components
Gamma = metric_christoffel(g_matrix, ginv_matrix, [:t,:r,:theta,:phi];
    deriv_fn=my_diff)

# Compute Riemann tensor from Christoffel
Riem = metric_riemann(Gamma, 4; coords=[:t,:r,:theta,:phi],
    deriv_fn=my_diff)

# Ricci tensor by contraction
Ric = metric_ricci(Riem, 4)

# Ricci scalar
R = metric_ricci_scalar(Ric, ginv_matrix, 4)

# Einstein tensor
G = metric_einstein(Ric, R, g_matrix, 4)

# Weyl tensor
C = metric_weyl(Riem, Ric, R, g_matrix, ginv_matrix, 4)

# Kretschmann scalar: R_{abcd} R^{abcd}
K = metric_kretschmann(Riem, g_matrix, ginv_matrix, 4)
```

```@docs
metric_christoffel
metric_riemann
metric_ricci
metric_ricci_scalar
metric_einstein
metric_weyl
metric_kretschmann
```

## Component Value Storage

Store and retrieve individual tensor component values with symmetry awareness. Only independent components need to be set; symmetry-related components are computed automatically.

```julia
store = ComponentStore(:g, :spherical, 4;
    symmetries=Any[Symmetric(1, 2)])

set_component!(store, [1, 1], -1.0)
set_component!(store, [2, 2],  1.0)

val = get_component(store, [1, 1])  # => -1.0

# List independent components (accounting for symmetries)
indep = independent_components(store)
```

```@docs
ComponentStore
set_component!
get_component
independent_components
```

## Bases

Define coordinate and non-coordinate bases (tetrads, vierbeins) on manifolds.

```julia
# Coordinate basis (default)
bp = define_basis!(reg, :coord; manifold=:M4)

# Tetrad basis
bp = define_basis!(reg, :tetrad; manifold=:M4, basis_type=:orthonormal)

# Retrieve basis properties
bp = get_basis(reg, :tetrad)

# Basis change
T_new = basis_change(T, jacobian_matrix)
```

```@docs
BasisProperties
define_basis!
get_basis
basis_change
```

## Abstract-to-Component Conversion

Convert abstract tensor expressions to component representations in a given chart. The `to_basis` function produces a `CTensor` where each entry is the abstract expression with free indices replaced by component numbers. The `to_ctensor` function additionally evaluates to numerical values using a dictionary of known components.

```julia
# Convert to component array (entries are TensorExpr)
ct = to_basis(Tensor(:Ric, [down(:a), down(:b)]), chart)

# Get a raw array of component expressions
arr = component_array(expr, chart, reg)

# Evaluate to numerical CTensor using known values
values = Dict(
    (:g, [1,1]) => -1.0,
    (:g, [2,2]) =>  1.0,
    # ...
)
ct_num = to_ctensor(expr, chart, values)
```

```@docs
to_basis
component_array
to_ctensor
```

## Symbolic Components (Symbolics.jl Extension)

Build metric and curvature components symbolically using Symbolics.jl. Requires `using Symbolics` (loaded as a weak dependency extension).

```julia
using Symbolics

# Diagonal metric (e.g. Schwarzschild)
@variables r theta
sm = symbolic_diagonal_metric([:t,:r,:theta,:phi],
    [-f(r), 1/f(r), r^2, r^2*sin(theta)^2])

# Full metric from matrix
sm = symbolic_metric([:t,:r,:theta,:phi], g_matrix)

# Compute curvature from symbolic metric
christoffel = symbolic_christoffel(sm)
riemann     = symbolic_riemann(sm)
ricci       = symbolic_ricci(sm)
R           = symbolic_ricci_scalar(sm)
einstein    = symbolic_einstein(sm)
K           = symbolic_kretschmann(sm)

# All curvature tensors at once
curv = symbolic_curvature_from_metric(sm)
```

```@docs
SymbolicMetric
symbolic_diagonal_metric
symbolic_metric
sym_deriv
symbolic_christoffel
symbolic_riemann
symbolic_ricci
symbolic_ricci_scalar
symbolic_einstein
symbolic_kretschmann
symbolic_curvature_from_metric
```
