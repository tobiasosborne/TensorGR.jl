#= Equation of state types for matter coupling.

Provides abstract and concrete EOS types that pair with
PerfectFluidProperties (defined in gr/matter.jl) to close the
fluid equations.

Hierarchy:
    EquationOfState (abstract)
      BarotropicEOS    -- p = w * rho  (constant w)
      PolytropicEOS    -- p = K * rho^gamma
      TabularEOS       -- interpolated from data tables
=#

"""
    EquationOfState

Abstract supertype for all equations of state relating pressure to
energy density.
"""
abstract type EquationOfState end

# ---------------------------------------------------------------------------
# Barotropic EOS: p = w * rho
# ---------------------------------------------------------------------------

"""
    BarotropicEOS(w)

Linear (barotropic) equation of state `p = w * rho`.

Common values of `w`:
- `0`    -- dust (pressureless matter)
- `1//3` -- radiation
- `-1`   -- cosmological constant / dark energy
- `1`    -- stiff matter

# Example
```julia
dust      = BarotropicEOS(0)
radiation = BarotropicEOS(1//3)
de        = BarotropicEOS(-1)
```
"""
struct BarotropicEOS <: EquationOfState
    w::Rational{Int}
end

BarotropicEOS(w::Int) = BarotropicEOS(w // 1)

# ---------------------------------------------------------------------------
# Polytropic EOS: p = K * rho^gamma
# ---------------------------------------------------------------------------

"""
    PolytropicEOS(K, gamma)

Polytropic equation of state `p = K * rho^gamma`.

Reduces to barotropic when `gamma = 1` (with `w = K`).
Common in stellar structure (white dwarfs, neutron stars).

# Example
```julia
eos = PolytropicEOS(1//10, 5//3)   # non-relativistic degenerate gas
```
"""
struct PolytropicEOS <: EquationOfState
    K::Rational{Int}
    gamma::Rational{Int}
end

PolytropicEOS(K::Int, gamma) = PolytropicEOS(K // 1, Rational{Int}(gamma))
PolytropicEOS(K, gamma::Int) = PolytropicEOS(Rational{Int}(K), gamma // 1)
PolytropicEOS(K::Int, gamma::Int) = PolytropicEOS(K // 1, gamma // 1)

# ---------------------------------------------------------------------------
# Tabular EOS: interpolated from data
# ---------------------------------------------------------------------------

"""
    TabularEOS(rho_vals, p_vals)

Equation of state defined by tabulated (rho, p) data pairs.
Values are linearly interpolated between data points.

Both vectors must be the same length and `rho_vals` must be
sorted in ascending order.

# Example
```julia
eos = TabularEOS([0.0, 1.0, 2.0], [0.0, 0.5, 1.5])
pressure(eos, 0.5)  # => 0.25 (linear interpolation)
```
"""
struct TabularEOS <: EquationOfState
    rho_vals::Vector{Float64}
    p_vals::Vector{Float64}

    function TabularEOS(rho_vals::Vector{Float64}, p_vals::Vector{Float64})
        length(rho_vals) == length(p_vals) ||
            error("rho_vals and p_vals must have the same length")
        length(rho_vals) >= 2 ||
            error("TabularEOS requires at least 2 data points")
        issorted(rho_vals) ||
            error("rho_vals must be sorted in ascending order")
        new(rho_vals, p_vals)
    end
end

# ---------------------------------------------------------------------------
# pressure(eos, rho) -- evaluate p(rho)
# ---------------------------------------------------------------------------

"""
    pressure(eos::EquationOfState, rho)

Evaluate the pressure for a given energy density `rho`.
"""
function pressure end

pressure(eos::BarotropicEOS, rho) = eos.w * rho

pressure(eos::PolytropicEOS, rho) = eos.K * rho^eos.gamma

function pressure(eos::TabularEOS, rho::Real)
    rv = eos.rho_vals
    pv = eos.p_vals
    # Clamp to table bounds
    rho <= rv[1] && return pv[1]
    rho >= rv[end] && return pv[end]
    # Binary search for the bracketing interval
    i = searchsortedlast(rv, rho)
    i = clamp(i, 1, length(rv) - 1)
    # Linear interpolation
    t = (rho - rv[i]) / (rv[i+1] - rv[i])
    pv[i] + t * (pv[i+1] - pv[i])
end

# ---------------------------------------------------------------------------
# sound_speed(eos, rho) -- adiabatic sound speed  cs^2 = dp/drho
# ---------------------------------------------------------------------------

"""
    sound_speed(eos::EquationOfState, rho)

Adiabatic sound speed squared, `cs^2 = dp/drho`.

For a `BarotropicEOS`, this is simply `w`.
For a `PolytropicEOS`, `cs^2 = K * gamma * rho^(gamma - 1)`.
For a `TabularEOS`, the derivative is estimated by finite differences.
"""
function sound_speed end

sound_speed(eos::BarotropicEOS, _rho) = eos.w

sound_speed(eos::PolytropicEOS, rho) = eos.K * eos.gamma * rho^(eos.gamma - 1)

function sound_speed(eos::TabularEOS, rho::Real)
    rv = eos.rho_vals
    pv = eos.p_vals
    rho <= rv[1] && return (pv[2] - pv[1]) / (rv[2] - rv[1])
    rho >= rv[end] && return (pv[end] - pv[end-1]) / (rv[end] - rv[end-1])
    i = searchsortedlast(rv, rho)
    i = clamp(i, 1, length(rv) - 1)
    (pv[i+1] - pv[i]) / (rv[i+1] - rv[i])
end

# ---------------------------------------------------------------------------
# PerfectFluid -- pairs an EOS with a PerfectFluidProperties
# ---------------------------------------------------------------------------

"""
    PerfectFluid(eos, properties)

A perfect fluid combining an equation of state with the abstract tensor
definitions from `define_perfect_fluid!`.

# Fields
- `eos::EquationOfState` -- the equation of state
- `properties::PerfectFluidProperties` -- registry-linked tensor definitions

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    fp = define_perfect_fluid!(reg, :T; manifold=:M4, metric=:g)
    fluid = PerfectFluid(BarotropicEOS(1//3), fp)
    pressure(fluid.eos, 3)  # => 1//1
end
```
"""
struct PerfectFluid
    eos::EquationOfState
    properties::PerfectFluidProperties
end
