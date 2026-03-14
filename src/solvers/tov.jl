#= Tolman-Oppenheimer-Volkoff (TOV) equation ODE system.

Provides the TOV equations for hydrostatic equilibrium of a spherically
symmetric star in general relativity.  Designed for use with
DifferentialEquations.jl:

    tov = setup_tov(eos, rho_c)
    prob = ODEProblem(tov_rhs!, tov.u0, (tov.r0, R_max), tov)
    sol  = solve(prob, ...)

The state vector is u = [m, p] with independent variable r (areal radius).
=#

"""
    TOVSystem

Stores everything needed to integrate the TOV equations for a given
equation of state and central density.

# Fields
- `eos::EquationOfState` -- equation of state relating pressure to density.
- `rho_c::Float64` -- central energy density.
- `p_c::Float64` -- central pressure (computed from `eos` and `rho_c`).
- `u0::Vector{Float64}` -- initial state `[m(r0), p(r0)]` after the
  series-expansion start.
- `r0::Float64` -- starting radius (small but nonzero, avoiding the r=0
  singularity).
"""
struct TOVSystem
    eos::EquationOfState
    rho_c::Float64
    p_c::Float64
    u0::Vector{Float64}
    r0::Float64
end

# ---------------------------------------------------------------------------
# Internal: energy density from pressure (EOS inversion)
# ---------------------------------------------------------------------------

"""
    _density_from_pressure(eos, p)

Invert the EOS to find rho given p.  Analytic for Barotropic and Polytropic;
linear-interpolation search for Tabular.
"""
function _density_from_pressure(eos::BarotropicEOS, p)
    eos.w == 0 && return 0.0   # dust: p=0 always, rho is indeterminate
    Float64(p / eos.w)
end

function _density_from_pressure(eos::PolytropicEOS, p)
    K = Float64(eos.K)
    gamma = Float64(eos.gamma)
    K <= 0 && return 0.0
    p <= 0 && return 0.0
    (p / K)^(1.0 / gamma)
end

function _density_from_pressure(eos::TabularEOS, p_target::Real)
    pv = eos.p_vals
    rv = eos.rho_vals
    p_target <= pv[1] && return rv[1]
    p_target >= pv[end] && return rv[end]
    # Linear search (tables are typically small)
    for i in 1:(length(pv) - 1)
        if pv[i] <= p_target <= pv[i+1]
            t = (p_target - pv[i]) / (pv[i+1] - pv[i])
            return rv[i] + t * (rv[i+1] - rv[i])
        end
    end
    rv[end]
end

# ---------------------------------------------------------------------------
# Series expansion near r = 0 for regularity
# ---------------------------------------------------------------------------

"""
    _tov_series(eos, rho_c, r)

Evaluate the TOV solution at small radius `r` using the Taylor series
expansion around r=0, ensuring regularity:

    m(r) = (4/3) pi rho_c r^3 + O(r^5)
    p(r) = p_c - (2/3) pi (rho_c + p_c)(rho_c + 3 p_c) r^2 + O(r^4)

Returns `(m, p)`.
"""
function _tov_series(eos::EquationOfState, rho_c::Float64, r::Float64)
    p_c = Float64(pressure(eos, rho_c))
    m = (4.0 / 3.0) * pi * rho_c * r^3
    p = p_c - (2.0 / 3.0) * pi * (rho_c + p_c) * (rho_c + 3.0 * p_c) * r^2
    (m, p)
end

# ---------------------------------------------------------------------------
# setup_tov
# ---------------------------------------------------------------------------

"""
    setup_tov(eos, rho_c; r0=1e-4)

Construct a `TOVSystem` for the given equation of state and central density.

The integration starts at a small radius `r0` (default `1e-4`) where the
initial data are computed from the regular series expansion, avoiding the
coordinate singularity at r=0.

# Arguments
- `eos::EquationOfState` -- equation of state.
- `rho_c::Real` -- central energy density (geometric units, c = G = 1).
- `r0::Real` -- starting radius for integration (default `1e-4`).

# Returns
A `TOVSystem` ready for use with `tov_rhs!`.

# Example
```julia
eos = PolytropicEOS(1//10, 2//1)
tov = setup_tov(eos, 1.0)
# Use with DifferentialEquations.jl:
# prob = ODEProblem(tov_rhs!, tov.u0, (tov.r0, 20.0), tov)
# sol = solve(prob, Tsit5(); callback=surface_cb)
```
"""
function setup_tov(eos::EquationOfState, rho_c::Real; r0::Real=1e-4)
    rho_c_f = Float64(rho_c)
    rho_c_f > 0 || error("central density must be positive, got $rho_c")
    p_c = Float64(pressure(eos, rho_c_f))
    p_c > 0 || error("central pressure must be positive for a star; got p_c = $p_c")
    r0_f = Float64(r0)
    r0_f > 0 || error("starting radius must be positive, got $r0")

    m0, p0 = _tov_series(eos, rho_c_f, r0_f)
    TOVSystem(eos, rho_c_f, p_c, [m0, p0], r0_f)
end

# ---------------------------------------------------------------------------
# tov_rhs!  --  ODE right-hand side
# ---------------------------------------------------------------------------

"""
    tov_rhs!(du, u, p, r)

In-place ODE right-hand side for the TOV equations, compatible with
DifferentialEquations.jl.

The state vector `u = [m, p_pressure]` with independent variable `r`:

    dm/dr = 4 pi r^2 rho
    dp/dr = -(rho + p)(m + 4 pi r^3 p) / (r (r - 2m))

The parameter `p` (third argument) must be a `TOVSystem`.

Energy density `rho` is obtained from the pressure via EOS inversion.
Integration should be terminated when `p <= 0` (stellar surface).

# Example
```julia
tov = setup_tov(PolytropicEOS(1//10, 2//1), 1.0)
du = zeros(2)
tov_rhs!(du, tov.u0, tov, tov.r0)
# du now contains [dm/dr, dp/dr] at r = r0
```
"""
function tov_rhs!(du, u, p::TOVSystem, r)
    m = u[1]
    pres = u[2]

    # At the stellar surface or beyond, halt evolution
    if pres <= 0.0
        du[1] = 0.0
        du[2] = 0.0
        return nothing
    end

    rho = _density_from_pressure(p.eos, pres)

    # Mass equation
    du[1] = 4.0 * pi * r^2 * rho

    # TOV pressure equation
    # Guard against r -> 0 (should not happen if r0 > 0, but be safe)
    denom = r * (r - 2.0 * m)
    if abs(denom) < 1e-30
        # Fallback to Newtonian limit for safety
        du[2] = r > 0.0 ? -(rho * m) / r^2 : 0.0
    else
        du[2] = -(rho + pres) * (m + 4.0 * pi * r^3 * pres) / denom
    end

    nothing
end
