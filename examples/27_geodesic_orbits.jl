# ============================================================================
# TensorGR.jl -- Geodesic Orbits in Schwarzschild Spacetime
#
# Demonstrates:
#   1. Building the Schwarzschild metric as a numerical function
#   2. Optionally computing Christoffel symbols symbolically (Symbolics.jl)
#   3. Setting up a GeodesicEquation for numerical integration
#   4. Integrating a circular orbit at r = 10M for 5 orbital periods
#   5. Checking energy conservation along the trajectory
#
# Run: julia --project examples/27_geodesic_orbits.jl
#
# Requires DifferentialEquations.jl for full integration. Falls back to a
# simple Euler integrator if not available. Symbolics.jl is optional for
# symbolic Christoffel display.
# ============================================================================

using TensorGR
using Printf

println("=" ^ 70)
println("  Geodesic Orbits in Schwarzschild Spacetime")
println("=" ^ 70)

# --- 1. Schwarzschild metric function ---
# ds^2 = -(1 - 2M/r) dt^2 + dr^2/(1 - 2M/r) + r^2 d(theta)^2
#        + r^2 sin^2(theta) d(phi)^2
# We set M = 1 (geometric units).

# Helper: build a diagonal matrix
_diag4(v) = Float64[i == j ? v[i] : 0.0 for i in 1:4, j in 1:4]

function schwarzschild_metric(x)
    r_val = x[2]
    th_val = x[3]
    f_val = 1.0 - 2.0 / r_val
    sth2 = sin(th_val)^2
    g = _diag4([-f_val, 1.0 / f_val, r_val^2, r_val^2 * sth2])
    ginv = _diag4([-1.0 / f_val, f_val, 1.0 / r_val^2, 1.0 / (r_val^2 * sth2)])
    (g, ginv)
end

println("\nSchwarzschild metric (M = 1, geometric units):")
println("  g_tt          = -(1 - 2/r)")
println("  g_rr          = 1/(1 - 2/r)")
println("  g_theta,theta = r^2")
println("  g_phi,phi     = r^2 sin^2(theta)")

# --- 2. Symbolic Christoffel symbols (optional) ---

_has_symbolics = try
    @eval using Symbolics
    true
catch
    false
end

if _has_symbolics
    println("\n--- Symbolic Christoffel symbols (via Symbolics.jl) ---")
    @eval begin
        @variables _t _r _theta _phi
        _f_sym = 1 - 2 / _r
        _diag_entries = [-_f_sym, 1 / _f_sym, _r^2, _r^2 * sin(_theta)^2]
        _sm = symbolic_diagonal_metric([_t, _r, _theta, _phi], _diag_entries)
        _Gamma_sym = symbolic_christoffel(_sm)

        _labels = ["t", "r", "th", "ph"]
        for a in 1:4, b in 1:4, c in b:4
            val = Symbolics.simplify(_Gamma_sym[a, b, c])
            if !isequal(val, 0)
                println("  Gamma^$(_labels[a])_{$(_labels[b])$(_labels[c])} = ", val)
            end
        end
    end
else
    println("\n  (Symbolics.jl not available; skipping symbolic Christoffel display)")
    println("  Non-zero Christoffel symbols for Schwarzschild (M=1):")
    println("    Gamma^t_{tr}     = M / (r^2 f)")
    println("    Gamma^r_{tt}     = M f / r^2")
    println("    Gamma^r_{rr}     = -M / (r^2 f)")
    println("    Gamma^r_{th,th}  = -r f")
    println("    Gamma^r_{ph,ph}  = -r f sin^2(theta)")
    println("    Gamma^th_{r,th}  = 1/r")
    println("    Gamma^th_{ph,ph} = -sin(theta) cos(theta)")
    println("    Gamma^ph_{r,ph}  = 1/r")
    println("    Gamma^ph_{th,ph} = cos(theta) / sin(theta)")
end

# --- 3. Setup geodesic equation ---
# setup_geodesic computes Christoffel symbols numerically via finite differences
# when no christoffel_fn is provided.

geq = setup_geodesic(schwarzschild_metric; dim=4, is_timelike=true)

# --- 4. Circular orbit initial conditions at r = 10M ---
# For a circular orbit at radius r0 in Schwarzschild (M=1):
#   f(r0)  = 1 - 2/r0
#   Energy: E = f(r0) / sqrt(1 - 3/r0)
#   Ang. momentum: L = sqrt(r0) / sqrt(1 - 3/r0)
#   4-velocity: v^t = E/f, v^phi = L/r0^2, v^r = v^theta = 0

r0 = 10.0
f0 = 1.0 - 2.0 / r0    # = 0.8

E_orb = f0 / sqrt(1.0 - 3.0 / r0)
L_orb = sqrt(r0) / sqrt(1.0 - 3.0 / r0)
Omega = sqrt(1.0 / r0^3)

vt0 = E_orb / f0
vphi0 = L_orb / r0^2

x0 = [0.0, r0, pi / 2, 0.0]
v0 = [vt0, 0.0, 0.0, vphi0]

# Verify normalization: g_mu_nu v^mu v^nu = -1
function geodesic_norm(metric_fn, x, v)
    g, _ = metric_fn(x)
    sum(g[i, j] * v[i] * v[j] for i in 1:4, j in 1:4)
end

norm0 = geodesic_norm(schwarzschild_metric, x0, v0)

println("\n--- Circular orbit at r = $(r0)M ---")
println("  Specific energy E = ", round(E_orb, digits=8))
println("  Specific angular momentum L = ", round(L_orb, digits=8))
println("  Coordinate angular velocity Omega = ", round(Omega, digits=8))
println("  Initial 4-velocity norm (should be -1): ", round(norm0, digits=12))

# --- 5. Integrate for 5 orbital periods ---

T_coord = 2pi / Omega
T_proper = 2pi / vphi0
n_orbits = 5

println("\n  Coordinate period T = ", round(T_coord, digits=4))
println("  Proper-time period tau = ", round(T_proper, digits=4))
println("  Integrating for $n_orbits orbits...")

tau_end = n_orbits * T_proper

# Try DifferentialEquations.jl first, fall back to Euler
_has_diffeq = try
    @eval using DifferentialEquations
    true
catch
    false
end

local sol

if _has_diffeq
    println("  Using DifferentialEquations.jl (Tsit5 solver)")
    sol = integrate_geodesic(geq, x0, v0, (0.0, tau_end);
                             abstol=1e-12, reltol=1e-12)
    println("  Solver return code: ", sol.retcode)
    println("  Number of time steps: ", length(sol.t))
else
    println("  DifferentialEquations.jl not available; using Euler integration")
    dt = 0.01
    nsteps = round(Int, tau_end / dt)
    u = vcat(Float64.(x0), Float64.(v0))
    du = similar(u)
    t_vals = Float64[]
    x_vals = Vector{Float64}[]
    v_vals = Vector{Float64}[]
    local tau_cur = 0.0
    for step in 0:nsteps
        push!(t_vals, tau_cur)
        push!(x_vals, copy(u[1:4]))
        push!(v_vals, copy(u[5:8]))
        geodesic_rhs!(du, u, geq, tau_cur)
        u .+= dt .* du
        tau_cur += dt
    end
    sol = GeodesicSolution(t_vals, x_vals, v_vals, :Success, nothing)
    println("  Number of time steps: ", length(sol.t))
end

# --- 6. Conservation checks ---

norms = [geodesic_norm(schwarzschild_metric, sol.x[i], sol.v[i])
         for i in eachindex(sol.t)]
max_norm_err = maximum(abs.(norms .+ 1.0))

println("\n--- Conservation checks ---")
println("  Max |g_mu_nu v^mu v^nu + 1| = ", @sprintf("%.2e", max_norm_err))

# Radius should remain constant for circular orbit
radii = [sol.x[i][2] for i in eachindex(sol.t)]
max_r_err = maximum(abs.(radii .- r0))
println("  Max |r - r0| = ", @sprintf("%.2e", max_r_err))

# Theta should stay at pi/2 (equatorial plane)
thetas = [sol.x[i][3] for i in eachindex(sol.t)]
max_theta_err = maximum(abs.(thetas .- pi / 2))
println("  Max |theta - pi/2| = ", @sprintf("%.2e", max_theta_err))

# Conserved energy: E = f(r) v^t
energies = [(1.0 - 2.0 / sol.x[i][2]) * sol.v[i][1] for i in eachindex(sol.t)]
max_E_err = maximum(abs.(energies .- E_orb))
println("  Max |E - E0| = ", @sprintf("%.2e", max_E_err))

# Conserved angular momentum: L = r^2 sin^2(theta) v^phi
ang_mom = [sol.x[i][2]^2 * sin(sol.x[i][3])^2 * sol.v[i][4]
           for i in eachindex(sol.t)]
max_L_err = maximum(abs.(ang_mom .- L_orb))
println("  Max |L - L0| = ", @sprintf("%.2e", max_L_err))

# --- 7. Summary ---

phi_final = sol.x[end][4]
expected_phi = n_orbits * 2pi
phi_err = abs(phi_final - expected_phi)

println("\n--- Summary ---")
println("  Final phi = ", round(phi_final, digits=6),
        " (expected ", round(expected_phi, digits=6), ")")
println("  Azimuthal error: ", @sprintf("%.2e", phi_err), " rad")

if max_norm_err < 1e-6 && max_r_err < 1e-6
    println("\n  [PASS] All conservation checks within tolerance.")
else
    println("\n  [WARN] Conservation errors exceed tolerance.")
    println("         (Euler integration has limited accuracy; install")
    println("          DifferentialEquations.jl for high-precision results.)")
end

println("\nDone.")
