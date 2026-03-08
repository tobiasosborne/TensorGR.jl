# ============================================================================
# TensorGR.jl -- Wald Verification: Schwarzschild Spacetime
#
# Verifies curvature properties of the Schwarzschild metric using the
# symbolic component backend (SymbolicMetric + Symbolics.jl).
#
# The Schwarzschild metric (Schwarzschild coordinates):
#   ds^2 = -(1 - 2M/r) dt^2 + (1 - 2M/r)^{-1} dr^2 + r^2 dOmega^2
#
# References:
#   Wald, "General Relativity" (1984), Section 6.1
#   Carroll, "Spacetime and Geometry", Chapter 5
#   MTW, "Gravitation", Chapter 31
# ============================================================================

using TensorGR
using Symbolics

# Helper: evaluate a symbolic expression at given coordinate values.
# Handles residual trig functions that Symbolics may leave unevaluated.
function _eval_sym(expr, vals)
    subbed = Symbolics.substitute(expr, vals)
    v = Symbolics.value(subbed)
    v isa Number && return Float64(v)
    return Float64(eval(Symbolics.toexpr(v)))
end

function run_schwarzschild()
    println("=" ^ 70)
    println("Wald Verification 04: Schwarzschild Spacetime")
    println("=" ^ 70)

    # ---- Define coordinates and metric ----

    @variables t r θ φ M

    f = 1 - 2M / r

    println("\nSchwarzschild metric: ds^2 = -f dt^2 + (1/f) dr^2 + r^2 dOmega^2")
    println("  where f = 1 - 2M/r")

    sm = symbolic_diagonal_metric(
        [t, r, θ, φ],
        [-f, 1 / f, r^2, r^2 * sin(θ)^2]
    )

    # ---- Compute curvature ----

    println("\nComputing Christoffel symbols...")
    Gamma = symbolic_christoffel(sm)

    println("Computing Riemann tensor...")
    Riem = symbolic_riemann(sm, Gamma)

    println("Computing Ricci tensor...")
    Ric = symbolic_ricci(Riem, 4)

    println("Computing Ricci scalar...")
    R = symbolic_ricci_scalar(Ric, sm.ginv, 4)

    # Evaluation point for numerical checks
    vals = Dict(M => 1.0, r => 3.0, θ => π / 2, φ => 0.0)

    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # 1. Christoffel symbol spot-checks
    #    (Wald Section 6.1, standard GR textbook results)
    # ------------------------------------------------------------------
    println("\n--- 1. Christoffel symbol verification ---")

    # Expected values at M=1, r=3:
    #   f = 1 - 2/3 = 1/3
    #   Gamma^t_{tr} = M/(r(r-2M)) = 1/(3*1) = 1/3
    #   Gamma^r_{tt} = M(r-2M)/r^3 = 1*1/27 = 1/27
    #   Gamma^r_{rr} = -M/(r(r-2M)) = -1/3
    #   Gamma^r_{theta_theta} = -(r-2M) = -1
    #   Gamma^theta_{r_theta} = 1/r = 1/3

    christoffel_checks = [
        ("Gamma^t_{tr}", (1, 1, 2), 1.0 / (3.0 * 1.0)),
        ("Gamma^r_{tt}", (2, 1, 1), 1.0 * 1.0 / 27.0),
        ("Gamma^r_{rr}", (2, 2, 2), -1.0 / (3.0 * 1.0)),
        ("Gamma^r_{theta_theta}", (2, 3, 3), -1.0),
        ("Gamma^theta_{r_theta}", (3, 2, 3), 1.0 / 3.0),
    ]

    for (name, (a, b, c), expected) in christoffel_checks
        computed = _eval_sym(Gamma[a, b, c], vals)
        ok = abs(computed - expected) < 1e-10
        status = ok ? "PASSED" : "FAILED"
        if ok
            passed += 1
        else
            failed += 1
        end
        println("  $name = $computed (expected $expected) ... $status")
    end

    # ------------------------------------------------------------------
    # 2. Ricci tensor vanishes (Schwarzschild is vacuum: R_{ab} = 0)
    #    (Wald Eq. 6.1.6 and surrounding discussion)
    # ------------------------------------------------------------------
    println("\n--- 2. Ricci tensor vanishes (vacuum solution) ---")

    ricci_ok = true
    max_ricci = 0.0
    for a in 1:4, b in 1:4
        val = abs(_eval_sym(Ric[a, b], vals))
        max_ricci = max(max_ricci, val)
        if val > 1e-10
            ricci_ok = false
            println("  WARNING: R_{$a,$b} = $val (should be 0)")
        end
    end

    if ricci_ok
        passed += 1
        println("  All R_{ab} = 0 (max deviation: $(max_ricci)) ... PASSED")
    else
        failed += 1
        println("  Ricci tensor check ... FAILED")
    end

    # ------------------------------------------------------------------
    # 3. Ricci scalar vanishes: R = 0
    #    (Follows from R_{ab} = 0)
    # ------------------------------------------------------------------
    println("\n--- 3. Ricci scalar vanishes ---")

    R_val = abs(_eval_sym(R, vals))
    if R_val < 1e-10
        passed += 1
        println("  R = $R_val ... PASSED")
    else
        failed += 1
        println("  R = $R_val (expected 0) ... FAILED")
    end

    # ------------------------------------------------------------------
    # 4. Kretschmann scalar: K = 48 M^2 / r^6
    #    (Wald Problem 6.1, standard result)
    # ------------------------------------------------------------------
    println("\n--- 4. Kretschmann scalar: K = 48 M^2 / r^6 ---")

    println("  Computing Kretschmann scalar (this may take a moment)...")
    K = symbolic_kretschmann(Riem, sm.g, sm.ginv, 4)

    kretschmann_ok = true
    kretschmann_points = [
        (1.0, 3.0),
        (2.0, 5.0),
        (0.5, 4.0),
    ]

    for (M_val, r_val) in kretschmann_points
        kvals = Dict(M => M_val, r => r_val, θ => π / 2, φ => 0.0)
        K_computed = _eval_sym(K, kvals)
        K_exact = 48.0 * M_val^2 / r_val^6
        ok = abs(K_computed - K_exact) < 1e-10
        if !ok
            kretschmann_ok = false
        end
        status = ok ? "ok" : "MISMATCH"
        println("  M=$M_val, r=$r_val: K=$K_computed (expected $K_exact) ... $status")
    end

    if kretschmann_ok
        passed += 1
        println("  Kretschmann scalar ... PASSED")
    else
        failed += 1
        println("  Kretschmann scalar ... FAILED")
    end

    # ------------------------------------------------------------------
    # 5. Ricci-flat at additional evaluation points
    #    (Verify vacuum property is not an accident of a single point)
    # ------------------------------------------------------------------
    println("\n--- 5. Ricci-flat at multiple points ---")

    extra_ok = true
    extra_points = [
        Dict(M => 2.0, r => 5.0, θ => 1.0, φ => 0.5),
        Dict(M => 0.5, r => 10.0, θ => 0.3, φ => 1.0),
        Dict(M => 1.0, r => 2.5, θ => π / 4, φ => π),
    ]

    for (idx, ev) in enumerate(extra_points)
        max_val = 0.0
        for a in 1:4, b in 1:4
            val = abs(_eval_sym(Ric[a, b], ev))
            max_val = max(max_val, val)
        end
        ok = max_val < 1e-10
        if !ok
            extra_ok = false
        end
        println("  Point $idx: max |R_{ab}| = $max_val ... $(ok ? "ok" : "MISMATCH")")
    end

    if extra_ok
        passed += 1
        println("  Multi-point Ricci-flat ... PASSED")
    else
        failed += 1
        println("  Multi-point Ricci-flat ... FAILED")
    end

    # ------------------------------------------------------------------
    # 6. Einstein tensor vanishes (G_{ab} = R_{ab} - (1/2) g_{ab} R = 0)
    # ------------------------------------------------------------------
    println("\n--- 6. Einstein tensor vanishes ---")

    G = symbolic_einstein(Ric, R, sm.g, 4)
    einstein_ok = true
    max_ein = 0.0
    for a in 1:4, b in 1:4
        val = abs(_eval_sym(G[a, b], vals))
        max_ein = max(max_ein, val)
        if val > 1e-10
            einstein_ok = false
        end
    end

    if einstein_ok
        passed += 1
        println("  All G_{ab} = 0 (max deviation: $(max_ein)) ... PASSED")
    else
        failed += 1
        println("  Einstein tensor check ... FAILED")
    end

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    total = passed + failed
    println("Schwarzschild verification: $passed/$total checks passed")
    if failed > 0
        println("WARNING: $failed check(s) failed!")
    else
        println("All Schwarzschild spacetime properties verified!")
    end
    println("=" ^ 70)

    @assert failed == 0 "Schwarzschild verification had $failed failure(s)"
end

run_schwarzschild()
