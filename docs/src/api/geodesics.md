# Geodesics

Numerical integration of geodesic equations on curved spacetimes. Provides
the geodesic ODE right-hand side for use with DifferentialEquations.jl or
any ODE solver that accepts an in-place `f!(du, u, p, t)` signature.

## Geodesic Equation Setup

Construct a `GeodesicEquation` from a metric function. If no Christoffel
function is supplied, Christoffel symbols are computed numerically via
central finite differences at each evaluation point.

```julia
# Schwarzschild metric in spherical coordinates
function schw_metric(x)
    r, M = x[2], 1.0
    f = 1 - 2M/r
    g = diagm([-f, 1/f, r^2, r^2*sin(x[3])^2])
    ginv = diagm([-1/f, f, 1/r^2, 1/(r^2*sin(x[3])^2)])
    (g, ginv)
end

geq = setup_geodesic(schw_metric; dim=4, is_timelike=true)
```

```@docs
GeodesicEquation
setup_geodesic
```

## ODE Right-Hand Side

The state vector `u` has length `2*dim`: positions `u[1:dim]` followed by
velocities `u[dim+1:2*dim]`. The geodesic equation is:

    dx^mu/dtau = v^mu
    dv^mu/dtau = -Gamma^mu_{alpha beta} v^alpha v^beta

```julia
# Use with DifferentialEquations.jl
using DifferentialEquations
x0 = [0.0, 10.0, pi/2, 0.0]
v0 = [1.1, 0.0, 0.0, 0.02]
u0 = vcat(x0, v0)
prob = ODEProblem(geodesic_rhs!, u0, (0.0, 100.0), geq)
sol = solve(prob, Tsit5())
```

```@docs
geodesic_rhs!
```

## Integration Helper

The `integrate_geodesic` function wraps the ODE setup and returns a
structured `GeodesicSolution`. Requires DifferentialEquations.jl to be
loaded (weak dependency extension).

```julia
sol = integrate_geodesic(geq, x0, v0, (0.0, 100.0))
sol.t    # proper time / affine parameter values
sol.x    # position vectors at each time step
sol.v    # velocity vectors at each time step
```

```@docs
GeodesicSolution
integrate_geodesic
```

## Numerical Christoffel Symbols

When no analytic Christoffel function is provided, `setup_geodesic` uses
central finite differences internally. This function is also available
directly for debugging or validation.

```@docs
TensorGR._numerical_christoffel
```
