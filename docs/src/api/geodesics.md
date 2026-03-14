# Geodesic Integration

Numerical integration of geodesic equations on curved spacetimes. The module provides the ODE right-hand side for the geodesic equation, compatible with DifferentialEquations.jl for numerical solutions of both timelike and null geodesics.

## Overview

The geodesic equation describes the motion of free-falling particles in curved spacetime:

    d^2 x^mu / d tau^2 + Gamma^mu_{alpha beta} (dx^alpha / d tau)(dx^beta / d tau) = 0

This is converted to a first-order system of 2\*dim ODEs by introducing the velocity v^mu = dx^mu/dtau. The `GeodesicEquation` struct stores the metric and Christoffel symbol functions, and `geodesic_rhs!` computes the right-hand side for the ODE integrator.

## Setup

Construct a geodesic equation from a metric function. If no Christoffel function is provided, Christoffel symbols are computed automatically via central finite differences.

```julia
# Minkowski metric
function mink(x)
    g = diagm([-1.0, 1.0, 1.0, 1.0])
    ginv = diagm([-1.0, 1.0, 1.0, 1.0])
    (g, ginv)
end

geq = setup_geodesic(mink; dim=4)

# Schwarzschild metric
function schwarzschild(x; M=1.0)
    r, theta = x[2], x[3]
    f = 1 - 2M/r
    g = diagm([-f, 1/f, r^2, r^2*sin(theta)^2])
    ginv = diagm([-1/f, f, 1/r^2, 1/(r^2*sin(theta)^2)])
    (g, ginv)
end

geq = setup_geodesic(schwarzschild; dim=4, is_timelike=true)
```

```@docs
GeodesicEquation
setup_geodesic
```

## ODE Right-Hand Side

The `geodesic_rhs!` function computes the time derivatives in-place, following the DifferentialEquations.jl convention. The state vector `u` has length `2*dim`: positions in `u[1:dim]`, velocities in `u[dim+1:2*dim]`.

```julia
# Evaluate the RHS at a point (for manual stepping or testing)
du = zeros(8)
u = [0.0, 10.0, pi/2, 0.0,   # position: t, r, theta, phi
     1.1, 0.0, 0.0, 0.02]    # velocity: dt/dtau, dr/dtau, dtheta/dtau, dphi/dtau
geodesic_rhs!(du, u, geq, 0.0)

# Use with DifferentialEquations.jl
# using DifferentialEquations
# prob = ODEProblem(geodesic_rhs!, u0, (0.0, 100.0), geq)
# sol = solve(prob, Tsit5())
```

```@docs
geodesic_rhs!
```

## Integration

Integrate the geodesic equation numerically. Requires `DifferentialEquations.jl` to be loaded.

```julia
using DifferentialEquations

geq = setup_geodesic(schwarzschild; dim=4)
x0 = [0.0, 10.0, pi/2, 0.0]        # initial position
v0 = [1.1, 0.0, 0.0, 0.02]         # initial velocity
sol = integrate_geodesic(geq, x0, v0, (0.0, 100.0))

sol.t    # proper time values
sol.x    # positions at each time step
sol.v    # velocities at each time step
sol.retcode  # :Success if integration completed
```

```@docs
GeodesicSolution
integrate_geodesic
```

## Example: Schwarzschild Orbit

```julia
using TensorGR
using DifferentialEquations

# Schwarzschild metric with M = 1
function schw(x)
    r, theta = x[2], x[3]
    f = 1 - 2.0/r
    g = diagm([-f, 1/f, r^2, r^2*sin(theta)^2])
    ginv = diagm([-1/f, f, 1/r^2, 1/(r^2*sin(theta)^2)])
    (g, ginv)
end

geq = setup_geodesic(schw; dim=4, is_timelike=true)

# Circular orbit at r = 6M (ISCO for Schwarzschild)
r0 = 6.0
E = (1 - 2/r0) / sqrt(1 - 3/r0)
L = sqrt(r0) / sqrt(1 - 3/r0)
x0 = [0.0, r0, pi/2, 0.0]
v0 = [E/(1 - 2/r0), 0.0, 0.0, L/r0^2]

sol = integrate_geodesic(geq, x0, v0, (0.0, 200.0))
```
