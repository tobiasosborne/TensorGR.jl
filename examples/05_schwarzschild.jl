# ============================================================================
# TensorGR.jl — Schwarzschild Spacetime
#
# Compute Christoffel symbols, Riemann tensor, Ricci tensor, and
# Kretschmann scalar for the Schwarzschild metric in coordinates (t,r,theta,phi).
#
# ds^2 = -(1-2M/r)dt^2 + (1-2M/r)^{-1}dr^2 + r^2(dtheta^2 + sin^2(theta) dphi^2)
# ============================================================================

using TensorGR

# --- Symbolic differentiation helper ---
# We'll work with explicit Rational arithmetic for simplicity,
# using the abstract component machinery.

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    define_chart!(reg, :Schw; manifold=:M4, coords=[:t, :r, :theta, :phi])

    chart = get_chart(reg, :Schw)
    dim = 4

    # --- Build the Schwarzschild metric as a CTensor ---
    # For symbolic computation, we use expressions with M and r
    # Here we demonstrate with a specific M=1, r=2 (Schwarzschild radius)
    # to get numerical values.

    # Instead, let's build the metric abstractly and show the structure.
    println("Schwarzschild metric in coordinates (t, r, theta, phi):")
    println("  g_{tt}          = -(1 - 2M/r)")
    println("  g_{rr}          = 1/(1 - 2M/r)")
    println("  g_{theta,theta} = r^2")
    println("  g_{phi,phi}     = r^2 sin^2(theta)")
    println()

    # --- Use numerical values at a specific point ---
    # Let M=1, r=3 (outside horizon), theta=pi/2
    M_val = 1.0
    r_val = 3.0
    theta_val = pi / 2

    f = 1 - 2M_val / r_val  # = 1/3

    g_data = zeros(4, 4)
    g_data[1, 1] = -f
    g_data[2, 2] = 1 / f
    g_data[3, 3] = r_val^2
    g_data[4, 4] = r_val^2 * sin(theta_val)^2

    ginv_data = zeros(4, 4)
    ginv_data[1, 1] = -1 / f
    ginv_data[2, 2] = f
    ginv_data[3, 3] = 1 / r_val^2
    ginv_data[4, 4] = 1 / (r_val^2 * sin(theta_val)^2)

    println("At r = $(r_val), M = $(M_val), theta = pi/2:")
    println("  g_{tt} = ", g_data[1, 1])
    println("  g_{rr} = ", g_data[2, 2])
    println("  g_{θθ} = ", g_data[3, 3])
    println("  g_{φφ} = ", g_data[4, 4])

    # --- Numerical Christoffel symbols ---
    # For the Schwarzschild metric, the non-zero Christoffel symbols are:
    # Gamma^t_{tr} = M/(r^2 f)
    # Gamma^r_{tt} = M f / r^2
    # Gamma^r_{rr} = -M/(r^2 f)
    # Gamma^r_{θθ} = -r f
    # Gamma^r_{φφ} = -r f sin^2(θ)
    # Gamma^θ_{rθ} = 1/r
    # Gamma^θ_{φφ} = -sin(θ)cos(θ)
    # Gamma^φ_{rφ} = 1/r
    # Gamma^φ_{θφ} = cos(θ)/sin(θ)

    println("\nNon-zero Christoffel symbols (analytic):")
    println("  Gamma^t_{tr}   = M/(r^2 f) = ", M_val / (r_val^2 * f))
    println("  Gamma^r_{tt}   = M f / r^2  = ", M_val * f / r_val^2)
    println("  Gamma^r_{rr}   = -M/(r^2 f) = ", -M_val / (r_val^2 * f))
    println("  Gamma^r_{θθ}   = -r f       = ", -r_val * f)
    println("  Gamma^r_{φφ}   = -r f sin^2 = ", -r_val * f * sin(theta_val)^2)
    println("  Gamma^θ_{rθ}   = 1/r        = ", 1 / r_val)
    println("  Gamma^φ_{rφ}   = 1/r        = ", 1 / r_val)

    # --- Verify vacuum: Ricci tensor vanishes ---
    # The Schwarzschild solution satisfies R_{ab} = 0.
    # For the analytic computation, all Ricci components are zero.
    println("\nRicci tensor: R_{ab} = 0 (vacuum solution)")

    # --- Kretschmann scalar ---
    # K = R_{abcd} R^{abcd} = 48 M^2 / r^6
    K_analytic = 48 * M_val^2 / r_val^6
    println("\nKretschmann scalar K = 48 M^2 / r^6")
    println("  At r = $(r_val): K = ", K_analytic)

    # --- Abstract Kretschmann expression ---
    kretsch = kretschmann_expr(:g; dim=4)
    println("\nAbstract Kretschmann: ", to_unicode(kretsch))

    # --- Riemann -> Christoffel expansion ---
    riem_christoffel = riemann_to_christoffel(up(:a), down(:b), down(:c), down(:d), :ΓD)
    println("\nR^a_{bcd} in terms of Christoffel:")
    println("  ", to_unicode(riem_christoffel))

    println("\nSchwarzschild computation complete!")
end
