# ============================================================================
# TensorGR.jl -- Course Verification: Lecture 19 -- Schwarzschild Solution
#
# Verifies curvature properties of the Schwarzschild metric using the
# symbolic component backend (SymbolicMetric + Symbolics.jl).
#
# The Schwarzschild metric (Schwarzschild coordinates):
#   ds^2 = -(1 - 2M/r) dt^2 + (1 - 2M/r)^{-1} dr^2 + r^2 dOmega^2
#
# Topics:
#   1. Schwarzschild metric and Christoffel symbols
#   2. Ricci tensor = 0 (vacuum solution)
#   3. Kretschmann scalar K = 48 M^2 / r^6
#   4. The fh=1 identity: R_{tt}/f + R_{rr}*f = 0
#   5. Einstein tensor vanishes
#
# References:
#   Carroll, "Spacetime and Geometry", Chapter 5
#   Wald, "General Relativity" (1984), Section 6.1
# ============================================================================

using TensorGR
using Symbolics

# Helper: evaluate a symbolic expression at given coordinate values.
function _eval_sym(expr, vals)
    subbed = Symbolics.substitute(expr, vals)
    v = Symbolics.value(subbed)
    v isa Number && return Float64(v)
    return Float64(eval(Symbolics.toexpr(v)))
end

function run_schwarzschild()
    println("="^70)
    println("Lecture 19: Schwarzschild Solution")
    println("="^70)

    @variables t r theta phi M

    f = 1 - 2M / r

    println("\nSchwarzschild metric: ds^2 = -f dt^2 + (1/f) dr^2 + r^2 dOmega^2")
    println("  where f = 1 - 2M/r")

    sm = symbolic_diagonal_metric(
        [t, r, theta, phi],
        [-f, 1 / f, r^2, r^2 * sin(theta)^2]
    )

    # ---- Compute curvature pipeline ----
    println("\nComputing Christoffel symbols...")
    Gamma = symbolic_christoffel(sm)

    println("Computing Riemann tensor...")
    Riem = symbolic_riemann(sm, Gamma)

    println("Computing Ricci tensor...")
    Ric = symbolic_ricci(Riem, 4)

    println("Computing Ricci scalar...")
    R = symbolic_ricci_scalar(Ric, sm.ginv, 4)

    println("Computing Einstein tensor...")
    G = symbolic_einstein(Ric, R, sm.g, 4)

    # Evaluation points
    vals = Dict(M => 1.0, r => 3.0, theta => pi / 2, phi => 0.0)

    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # 1. Christoffel symbol spot-checks
    # ------------------------------------------------------------------
    println("\n--- 1. Christoffel symbol verification ---")

    # At M=1, r=3: f = 1/3
    # Gamma^t_{tr} = M/(r(r-2M)) = 1/(3*1) = 1/3
    # Gamma^r_{tt} = M(r-2M)/r^3 = 1*1/27
    # Gamma^r_{rr} = -M/(r(r-2M)) = -1/3
    # Gamma^r_{theta_theta} = -(r-2M) = -1
    # Gamma^theta_{r_theta} = 1/r = 1/3
    # Gamma^phi_{r_phi} = 1/r = 1/3
    # Gamma^r_{phi_phi} = -(r-2M)*sin^2(theta) = -1
    # Gamma^theta_{phi_phi} = -sin(theta)*cos(theta) = 0 at theta=pi/2

    christoffel_checks = [
        ("Gamma^t_{tr}",           (1,1,2), 1.0/3.0),
        ("Gamma^r_{tt}",           (2,1,1), 1.0/27.0),
        ("Gamma^r_{rr}",           (2,2,2), -1.0/3.0),
        ("Gamma^r_{theta,theta}",  (2,3,3), -1.0),
        ("Gamma^r_{phi,phi}",      (2,4,4), -1.0),
        ("Gamma^theta_{r,theta}",  (3,2,3), 1.0/3.0),
        ("Gamma^phi_{r,phi}",      (4,2,4), 1.0/3.0),
        ("Gamma^theta_{phi,phi}",  (3,4,4), 0.0),
    ]

    for (name, (a,b,c), expected) in christoffel_checks
        computed = _eval_sym(Gamma[a,b,c], vals)
        ok = abs(computed - expected) < 1e-10
        if ok; passed += 1; else; failed += 1; end
        println("  $name = $computed (expected $expected) ... $(ok ? "PASSED" : "FAILED")")
    end

    # ------------------------------------------------------------------
    # 2. Ricci tensor vanishes (vacuum: R_{ab} = 0)
    # ------------------------------------------------------------------
    println("\n--- 2. Ricci tensor vanishes (vacuum solution) ---")

    ricci_ok = true
    max_ricci = 0.0
    for a in 1:4, b in 1:4
        val = abs(_eval_sym(Ric[a,b], vals))
        max_ricci = max(max_ricci, val)
        if val > 1e-10
            ricci_ok = false
            println("  WARNING: R_{$a,$b} = $val")
        end
    end
    if ricci_ok; passed += 1; else; failed += 1; end
    println("  All R_{ab} = 0 (max deviation: $max_ricci) ... $(ricci_ok ? "PASSED" : "FAILED")")

    # Verify at additional points
    extra_points = [
        Dict(M => 2.0, r => 5.0, theta => 1.0, phi => 0.5),
        Dict(M => 0.5, r => 10.0, theta => 0.3, phi => 1.0),
        Dict(M => 1.0, r => 2.5, theta => pi/4, phi => pi),
    ]

    extra_ok = true
    for (idx, ev) in enumerate(extra_points)
        for a in 1:4, b in 1:4
            val = abs(_eval_sym(Ric[a,b], ev))
            if val > 1e-10
                extra_ok = false
            end
        end
    end
    if extra_ok; passed += 1; else; failed += 1; end
    println("  Ricci-flat at 3 additional points ... $(extra_ok ? "PASSED" : "FAILED")")

    # ------------------------------------------------------------------
    # 3. Ricci scalar vanishes
    # ------------------------------------------------------------------
    println("\n--- 3. Ricci scalar vanishes ---")

    R_val = abs(_eval_sym(R, vals))
    ok = R_val < 1e-10
    if ok; passed += 1; else; failed += 1; end
    println("  R = $R_val ... $(ok ? "PASSED" : "FAILED")")

    # ------------------------------------------------------------------
    # 4. Kretschmann scalar: K = 48 M^2 / r^6
    # ------------------------------------------------------------------
    println("\n--- 4. Kretschmann scalar: K = 48 M^2 / r^6 ---")

    println("  Computing Kretschmann scalar...")
    K = symbolic_kretschmann(Riem, sm.g, sm.ginv, 4)

    kretschmann_ok = true
    kretschmann_points = [
        (1.0, 3.0),
        (2.0, 5.0),
        (0.5, 4.0),
    ]

    for (M_val, r_val) in kretschmann_points
        kvals = Dict(M => M_val, r => r_val, theta => pi/2, phi => 0.0)
        K_computed = _eval_sym(K, kvals)
        K_exact = 48.0 * M_val^2 / r_val^6
        ok_k = abs(K_computed - K_exact) < 1e-10
        if !ok_k; kretschmann_ok = false; end
        println("  M=$M_val, r=$r_val: K=$K_computed (expected $K_exact) ... $(ok_k ? "ok" : "MISMATCH")")
    end
    if kretschmann_ok; passed += 1; else; failed += 1; end
    println("  Kretschmann scalar ... $(kretschmann_ok ? "PASSED" : "FAILED")")

    # ------------------------------------------------------------------
    # 5. fh=1 identity: R_{tt}/f + R_{rr}*f simplifies to 0
    #    (Here h = 1/f, and the combination eliminates second derivatives of f)
    # ------------------------------------------------------------------
    println("\n--- 5. fh=1 identity: R_{tt}/f + R_{rr}*f = 0 ---")

    # Since both R_{tt} and R_{rr} are zero for Schwarzschild, verify
    # numerically at a non-vacuum perturbation: use a modified metric
    # and check the structural relation.
    # For exact Schwarzschild, this is trivially 0+0=0.
    # We verify that R_{tt} * (1/f) + R_{rr} * f evaluates to 0.
    fh_val = _eval_sym(Ric[1,1], vals) / (1.0 - 2.0/3.0) +
             _eval_sym(Ric[2,2], vals) * (1.0 - 2.0/3.0)
    ok = abs(fh_val) < 1e-10
    if ok; passed += 1; else; failed += 1; end
    println("  R_{tt}/f + R_{rr}*f = $fh_val ... $(ok ? "PASSED" : "FAILED")")

    # ------------------------------------------------------------------
    # 6. Einstein tensor vanishes
    # ------------------------------------------------------------------
    println("\n--- 6. Einstein tensor vanishes ---")

    einstein_ok = true
    max_ein = 0.0
    for a in 1:4, b in 1:4
        val = abs(_eval_sym(G[a,b], vals))
        max_ein = max(max_ein, val)
        if val > 1e-10
            einstein_ok = false
        end
    end
    if einstein_ok; passed += 1; else; failed += 1; end
    println("  All G_{ab} = 0 (max deviation: $max_ein) ... $(einstein_ok ? "PASSED" : "FAILED")")

    # ------------------------------------------------------------------
    # 7. Kretschmann scales as 1/r^6 at large r
    #    Verify K(r2)/K(r1) = (r1/r2)^6 for two large-r points.
    # ------------------------------------------------------------------
    println("\n--- 7. Kretschmann 1/r^6 scaling ---")

    r1 = 5.0; r2 = 10.0
    kv1 = Dict(M => 1.0, r => r1, theta => pi/2, phi => 0.0)
    kv2 = Dict(M => 1.0, r => r2, theta => pi/2, phi => 0.0)
    K1 = _eval_sym(K, kv1)
    K2 = _eval_sym(K, kv2)
    ratio = K1 / K2
    expected_ratio = (r2 / r1)^6
    ok = abs(ratio - expected_ratio) < 1e-8
    if ok; passed += 1; else; failed += 1; end
    println("  K($r1)/K($r2) = $ratio (expected $(expected_ratio)) ... $(ok ? "PASSED" : "FAILED")")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    println("\n" * "="^70)
    total = passed + failed
    println("Lecture 19 (Schwarzschild) verification: $passed/$total checks passed")
    if failed > 0
        println("WARNING: $failed check(s) failed!")
    else
        println("All Schwarzschild solution properties verified!")
    end
    println("="^70)

    @assert failed == 0 "Lecture 19 verification had $failed failure(s)"
end

run_schwarzschild()
