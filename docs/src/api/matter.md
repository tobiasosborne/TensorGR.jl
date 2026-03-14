# Matter Coupling

Equations of state, perfect fluid stress-energy tensors, and the Tolman-Oppenheimer-Volkoff (TOV) system for hydrostatic equilibrium of spherically symmetric stars in general relativity.

## Equations of State

The `EquationOfState` hierarchy provides different relationships between pressure and energy density. Three concrete implementations are available: barotropic (linear), polytropic (power-law), and tabular (interpolated from data).

```julia
# Barotropic: p = w * rho
dust      = BarotropicEOS(0)       # w = 0 (pressureless)
radiation = BarotropicEOS(1//3)    # w = 1/3
de        = BarotropicEOS(-1)      # w = -1 (cosmological constant)

# Polytropic: p = K * rho^gamma
eos_ns = PolytropicEOS(1//10, 2//1)    # neutron star model
eos_wd = PolytropicEOS(1//10, 5//3)    # white dwarf model

# Tabular: linearly interpolated from data
eos_tab = TabularEOS([0.0, 1.0, 2.0], [0.0, 0.5, 1.5])
```

```@docs
EquationOfState
BarotropicEOS
PolytropicEOS
TabularEOS
```

### Evaluating the EOS

```julia
# Pressure from density
p = pressure(BarotropicEOS(1//3), 3)   # => 1//1

# Adiabatic sound speed squared: cs^2 = dp/drho
cs2 = sound_speed(BarotropicEOS(1//3), 1.0)  # => 1//3

# Tabular interpolation
eos = TabularEOS([0.0, 1.0, 2.0], [0.0, 0.5, 1.5])
pressure(eos, 0.5)       # => 0.25 (linear interpolation)
sound_speed(eos, 0.5)    # => 0.5 (finite difference slope)
```

```@docs
pressure
sound_speed
```

## Perfect Fluid

The perfect fluid stress-energy tensor is:

    T^{ab} = (rho + p) u^a u^b + p g^{ab}

where rho is the energy density, p is the pressure, and u^a is the 4-velocity with normalization g\_{ab} u^a u^b = -1.

### Defining a Perfect Fluid

`define_perfect_fluid!` registers the stress-energy tensor, energy density, pressure, and 4-velocity as tensors in the registry. It also registers the normalization rule `u_a u^a = -1`.

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4, signature=lorentzian(4))

    # Register the fluid and its component tensors
    fp = define_perfect_fluid!(reg, :T; manifold=:M4, metric=:g)

    # Build the stress-energy expression
    expr = perfect_fluid_expr(up(:a), up(:b), fp)
    # => (rho + p) * u^a * u^b + p * g^{ab}

    # Pair with an equation of state
    fluid = PerfectFluid(BarotropicEOS(1//3), fp)

    # Retrieve a previously defined fluid
    fp2 = get_perfect_fluid(reg, :T)
end
```

### Custom Field Names

```julia
fp = define_perfect_fluid!(reg, :Tmatter;
    manifold=:M4, metric=:g,
    rho=:epsilon, p=:P, u=:U)
```

```@docs
PerfectFluidProperties
define_perfect_fluid!
perfect_fluid_expr
get_perfect_fluid
PerfectFluid
```

## TOV System

The Tolman-Oppenheimer-Volkoff equations describe hydrostatic equilibrium of a spherically symmetric, non-rotating star in general relativity:

    dm/dr = 4 pi r^2 rho
    dp/dr = -(rho + p)(m + 4 pi r^3 p) / (r(r - 2m))

The integration starts at a small radius r0 (avoiding the r=0 singularity) using a regular series expansion for the initial data. Integration should be terminated when p <= 0 (stellar surface).

### Setup

```julia
# Polytropic neutron star model
eos = PolytropicEOS(1//10, 2//1)
tov = setup_tov(eos, 1.0)         # central density rho_c = 1.0

# The TOVSystem stores initial conditions
tov.u0    # [m(r0), p(r0)] from series expansion
tov.r0    # starting radius (default 1e-4)
tov.p_c   # central pressure
```

```@docs
TOVSystem
setup_tov
```

### ODE Right-Hand Side

The `tov_rhs!` function computes the TOV equations in-place, compatible with DifferentialEquations.jl.

```julia
# Manual evaluation (for testing)
du = zeros(2)
tov_rhs!(du, tov.u0, tov, tov.r0)
# du[1] = dm/dr, du[2] = dp/dr at r = r0

# Full integration with DifferentialEquations.jl:
# using DifferentialEquations
# prob = ODEProblem(tov_rhs!, tov.u0, (tov.r0, 20.0), tov)
# sol = solve(prob, Tsit5(); callback=ContinuousCallback(
#     (u,t,integrator) -> u[2], (integrator) -> terminate!(integrator)))
```

```@docs
tov_rhs!
```

## Example: Neutron Star Mass-Radius

```julia
using TensorGR

# Polytropic EOS: p = K rho^gamma
eos = PolytropicEOS(1//10, 2//1)

# Compute stellar models for a range of central densities
for rho_c in [0.5, 1.0, 2.0, 5.0]
    tov = setup_tov(eos, rho_c)

    # Evaluate the RHS at the starting point
    du = zeros(2)
    tov_rhs!(du, tov.u0, tov, tov.r0)

    println("rho_c = $rho_c: p_c = $(tov.p_c), dm/dr|_0 = $(du[1])")
end

# For full integration, use DifferentialEquations.jl:
# prob = ODEProblem(tov_rhs!, tov.u0, (tov.r0, 50.0), tov)
# sol = solve(prob, Tsit5(); callback=surface_callback)
# M_star = sol[1, end]  # total mass
# R_star = sol.t[end]   # stellar radius
```
