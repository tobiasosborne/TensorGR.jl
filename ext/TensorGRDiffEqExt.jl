module TensorGRDiffEqExt

using TensorGR
using DifferentialEquations

"""
    integrate_geodesic(geod::GeodesicEquation, x0, v0, tau_span; solver=Tsit5(), callback=nothing, kwargs...)

Integrate the geodesic equation using DifferentialEquations.jl.

Builds an `ODEProblem` from `geodesic_rhs!`, solves it, and wraps the result
in a `GeodesicSolution`.

# Arguments
- `geod::GeodesicEquation` -- geodesic equation from `setup_geodesic`.
- `x0::AbstractVector` -- initial position x^mu(0).
- `v0::AbstractVector` -- initial velocity dx^mu/dtau(0).
- `tau_span::Tuple{Real,Real}` -- integration interval `(tau_start, tau_end)`.

# Keyword Arguments
- `solver` -- ODE solver algorithm (default: `Tsit5()`).
- `callback` -- optional callback for event detection (e.g., `ContinuousCallback`).
- All other keyword arguments are forwarded to `solve`.

# Returns
A `GeodesicSolution` with fields `.t`, `.x`, `.v`, `.retcode`, `.raw`.
"""
function TensorGR.integrate_geodesic(geod::TensorGR.GeodesicEquation,
                                      x0::AbstractVector,
                                      v0::AbstractVector,
                                      tau_span::Tuple{Real,Real};
                                      solver=Tsit5(),
                                      callback=nothing,
                                      kwargs...)
    dim = geod.dim
    length(x0) == dim || throw(ArgumentError(
        "x0 has length $(length(x0)), expected $dim"))
    length(v0) == dim || throw(ArgumentError(
        "v0 has length $(length(v0)), expected $dim"))

    # Pack initial conditions: u = [x; v]
    u0 = vcat(Float64.(x0), Float64.(v0))
    tspan = (Float64(tau_span[1]), Float64(tau_span[2]))

    # Build and solve ODE problem
    prob = ODEProblem(geodesic_rhs!, u0, tspan, geod)

    solve_kwargs = Dict{Symbol,Any}(kwargs...)
    if callback !== nothing
        solve_kwargs[:callback] = callback
    end

    sol = solve(prob, solver; solve_kwargs...)

    # Unpack solution into positions and velocities
    t_vals = sol.t
    x_vals = [Float64.(sol.u[i][1:dim]) for i in eachindex(sol.u)]
    v_vals = [Float64.(sol.u[i][dim+1:2*dim]) for i in eachindex(sol.u)]

    retcode = Symbol(sol.retcode)

    TensorGR.GeodesicSolution(t_vals, x_vals, v_vals, retcode, sol)
end

end # module
