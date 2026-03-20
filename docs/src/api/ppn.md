# Parametrized Post-Newtonian Formalism

The PPN framework parameterizes weak-field, slow-motion deviations from general relativity using 10 parameters. This module provides the PPN metric ansatz, potential registration, 3+1 decomposition, velocity-order tracking with truncation, field equation solving for named theories, and closed-form expressions for solar system observables (perihelion precession, light deflection, Shapiro delay, Nordtvedt effect, geodetic precession).

## PPN Metric Ansatz

The 10 PPN parameters and the standard PPN metric in isotropic coordinates. Supports named theory constructors for GR, Brans-Dicke, scalar-tensor, Nordtvedt, and Rosen bimetric theories.

```julia
params = ppn_gr()                                    # GR values
params = PPNParameters(:BransDicke; omega=40000)     # Brans-Dicke
metric = ppn_metric_ansatz(params, reg; order=2)     # full PPN metric
```

```@docs
PPNParameters
ppn_gr
is_gr
is_fully_conservative
is_preferred_frame_free
is_preferred_location_free
is_semi_conservative
define_ppn_potentials!
ppn_metric_ansatz
```

## Potential Decomposition

Decompose the PPN metric into its 3+1 components (g00, g0i, gij) and set up the PPN foliation with auxiliary tensors.

```julia
mc = ppn_decompose(params, reg; order=2)
mc.g00  # time-time component
mc.g0i  # time-space component
mc.gij  # space-space component
```

```@docs
PPNMetricComponents
ppn_decompose
ppn_compose
ppn_foliation!
PPNChristoffelComponents
ppn_christoffel_1pn
ppn_christoffel
```

## Velocity Order Tracking

Each PPN potential has a definite order in v/c. Products add orders; sums take the minimum. Truncation discards terms exceeding a specified order.

```@docs
PPN_ORDER_TABLE
ppn_order
ppn_max_order
truncate_ppn
PPN_METRIC_ORDERS
ppn_truncate_metric
```

## Field Equations

Solve PPN field equations for named gravitational theories, extracting the PPN parameters from the solution.

```julia
result = ppn_solve(:GR, reg; order=2)
result = ppn_solve(:BransDicke, reg; omega=40000)
table = ppn_parameter_table(result)
```

```@docs
PPNFieldEquationResult
ppn_solve_gr
ppn_solve_scalar_tensor
ppn_solve
extract_ppn_parameters
ppn_parameter_table
```

## Observables

Closed-form expressions for solar system tests as functions of PPN parameters.

### Perihelion Precession

```@docs
ppn_perihelion
ppn_perihelion_factor
```

### Light Deflection

```@docs
ppn_deflection
ppn_deflection_factor
```

### Shapiro Delay

```@docs
ppn_shapiro_delay
ppn_shapiro_factor
```

### Nordtvedt Effect

```@docs
ppn_nordtvedt_eta
```

### Geodetic Precession

```@docs
ppn_geodetic_precession
ppn_geodetic_factor
```

### Conservation and Bounds

```@docs
ppn_is_fully_conservative
ppn_observational_bounds
```
