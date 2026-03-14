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

"""
    solve_tov(tov::TOVSystem, r_max::Real; solver=Tsit5(), kwargs...)

Integrate the TOV equations from the center outward until the pressure drops
to zero (stellar surface) or `r_max` is reached.

Returns a [`TensorGR.TOVSolution`](@ref).
"""
function TensorGR.solve_tov(tov::TensorGR.TOVSystem, r_max::Real;
                             solver=Tsit5(), kwargs...)
    # ContinuousCallback: stop when pressure <= 0
    condition(u, r, integrator) = u[2]  # pressure
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    # Build ODE problem
    tspan = (tov.r0, Float64(r_max))
    prob = ODEProblem(TensorGR.tov_rhs!, tov.u0, tspan, tov)

    # Merge user callbacks with surface callback
    solve_kwargs = Dict{Symbol,Any}(kwargs...)
    if haskey(solve_kwargs, :callback)
        solve_kwargs[:callback] = CallbackSet(cb, solve_kwargs[:callback])
    else
        solve_kwargs[:callback] = cb
    end

    sol = solve(prob, solver; solve_kwargs...)

    # Extract profiles
    r_vals = sol.t
    m_vals = [sol.u[i][1] for i in eachindex(sol.u)]
    p_vals = [sol.u[i][2] for i in eachindex(sol.u)]
    rho_vals = [TensorGR._density_from_pressure(tov.eos, pv) for pv in p_vals]

    r_surface = r_vals[end]
    M_total = m_vals[end]

    TensorGR.TOVSolution(r_surface, M_total, r_vals, m_vals, p_vals, rho_vals, sol)
end

"""
    mass_radius_curve(eos::EquationOfState, rho_c_range; r_max=50.0, kwargs...)

Compute a mass-radius curve by solving the TOV equations for each central
density in `rho_c_range`.

Returns a named tuple `(R=..., M=...)` of vectors.
"""
function TensorGR.mass_radius_curve(eos::TensorGR.EquationOfState,
                                     rho_c_range::AbstractVector;
                                     r_max=50.0, kwargs...)
    R_vals = Float64[]
    M_vals = Float64[]
    for rho_c in rho_c_range
        tov = TensorGR.setup_tov(eos, rho_c)
        sol = TensorGR.solve_tov(tov, r_max; kwargs...)
        push!(R_vals, sol.r_surface)
        push!(M_vals, sol.M_total)
    end
    (R=R_vals, M=M_vals)
end

end # module
