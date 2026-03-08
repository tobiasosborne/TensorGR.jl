# ============================================================================
# TensorGR.jl -- Wald Verification: FLRW Cosmology
#
# Verifies curvature properties of the Friedmann-Lemaitre-Robertson-Walker
# metric using the symbolic component backend (SymbolicMetric + Symbolics.jl).
#
# The FLRW metric (comoving coordinates):
#   ds^2 = -dtau^2 + a(tau)^2 [ dchi^2/(1-k*chi^2) + chi^2 dOmega^2 ]
#
# We declare a as a function of tau: @variables a(tau), so that
# Symbolics.Differential(tau)(a(tau)) correctly produces da/dtau.
# After computing curvature, we substitute:
#   Differential(tau)(a(tau)) -> adot
#   Differential(tau,2)(a(tau)) -> addot
#   a(tau) -> a_bare (plain variable for evaluation)
#
# References:
#   Wald, "General Relativity" (1984), Appendix F (FLRW cosmology)
#   Carroll, "Spacetime and Geometry", Chapter 8
#   Weinberg, "Cosmology", Chapter 1
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

function run_flrw()
    println("=" ^ 70)
    println("Wald Verification 06: FLRW Cosmology")
    println("=" ^ 70)

    # ---- Define coordinates and variables ----

    @variables tau chi theta_f phi_f k
    @variables a_func(tau)         # scale factor as a function of tau
    @variables a_bare adot addot   # plain variables for substitution and evaluation

    println("\nFLRW metric:")
    println("  ds^2 = -dtau^2 + a^2 [dchi^2/(1-k*chi^2) + chi^2 dOmega^2]")
    println("  a(tau) declared as function; derivatives produce da/dtau automatically")

    sm = symbolic_diagonal_metric(
        [tau, chi, theta_f, phi_f],
        [Symbolics.Num(-1),
         a_func^2 / (1 - k * chi^2),
         a_func^2 * chi^2,
         a_func^2 * chi^2 * sin(theta_f)^2]
    )

    # ---- Compute curvature pipeline ----

    println("\nComputing Christoffel symbols...")
    Gamma_raw = symbolic_christoffel(sm)

    println("Computing Riemann tensor...")
    Riem_raw = symbolic_riemann(sm, Gamma_raw)

    println("Computing Ricci tensor...")
    Ric_raw = symbolic_ricci(Riem_raw, 4)

    println("Computing Ricci scalar...")
    R_raw = symbolic_ricci_scalar(Ric_raw, sm.ginv, 4)

    # ---- Build substitution dictionary ----
    # Replace derivative terms and functional form with plain variables

    D = Symbolics.Differential(tau)
    da = Symbolics.expand_derivatives(D(a_func))
    dda = Symbolics.expand_derivatives(D(da))

    sub_dict = Dict(da => adot, dda => addot, a_func => a_bare)

    # Apply substitution to Christoffel symbols
    Gamma = Array{Any}(undef, 4, 4, 4)
    for i in 1:4, j in 1:4, l in 1:4
        g2 = Symbolics.substitute(Gamma_raw[i, j, l], sub_dict)
        Gamma[i, j, l] = Symbolics.simplify(g2)
    end

    # Apply substitution to Ricci tensor
    Ric = Matrix{Any}(undef, 4, 4)
    for i in 1:4, j in 1:4
        r2 = Symbolics.substitute(Ric_raw[i, j], sub_dict)
        Ric[i, j] = Symbolics.simplify(r2)
    end

    # Apply substitution to Ricci scalar
    R_scalar = Symbolics.simplify(Symbolics.substitute(R_raw, sub_dict))

    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # 1. Christoffel symbol spot-checks (standard FLRW results)
    #
    # Convention: coords = (tau, chi, theta, phi) = indices (1, 2, 3, 4)
    # ------------------------------------------------------------------
    println("\n--- 1. Christoffel symbol verification ---")

    # Evaluation point
    vals = Dict(
        a_bare => 2.0, adot => 0.5, addot => -0.1,
        k => 0.0, chi => 0.5, theta_f => π / 2, phi_f => 0.0, tau => 0.0
    )

    a_v = 2.0
    ad_v = 0.5
    k_v = 0.0
    chi_v = 0.5
    th_v = π / 2

    christoffel_checks = [
        ("Gamma^tau_{chi,chi}", (1, 2, 2), a_v * ad_v / (1 - k_v * chi_v^2)),
        ("Gamma^tau_{theta,theta}", (1, 3, 3), a_v * ad_v * chi_v^2),
        ("Gamma^tau_{phi,phi}", (1, 4, 4), a_v * ad_v * chi_v^2 * sin(th_v)^2),
        ("Gamma^chi_{tau,chi}", (2, 1, 2), ad_v / a_v),
        ("Gamma^chi_{chi,chi}", (2, 2, 2), k_v * chi_v / (1 - k_v * chi_v^2)),
        ("Gamma^chi_{theta,theta}", (2, 3, 3), -chi_v * (1 - k_v * chi_v^2)),
        ("Gamma^theta_{tau,theta}", (3, 1, 3), ad_v / a_v),
        ("Gamma^theta_{chi,theta}", (3, 2, 3), 1.0 / chi_v),
        ("Gamma^theta_{phi,phi}", (3, 4, 4), -sin(th_v) * cos(th_v)),
        ("Gamma^phi_{tau,phi}", (4, 1, 4), ad_v / a_v),
        ("Gamma^phi_{chi,phi}", (4, 2, 4), 1.0 / chi_v),
        ("Gamma^phi_{theta,phi}", (4, 3, 4), cos(th_v) / sin(th_v)),
    ]

    for (name, (i, j, l), expected) in christoffel_checks
        computed = _eval_sym(Gamma[i, j, l], vals)
        ok = abs(computed - expected) < 1e-10
        if ok
            passed += 1
        else
            failed += 1
        end
        status = ok ? "PASSED" : "FAILED"
        println("  $name = $computed (expected $expected) ... $status")
    end

    # ------------------------------------------------------------------
    # 2. Vanishing Christoffels: Gamma^tau_{tau,*} should all be zero
    #    (since g_{tau tau} = -1 is constant)
    # ------------------------------------------------------------------
    println("\n--- 2. Vanishing temporal Christoffel components ---")

    vanish_ok = true
    for j in 1:4
        v1 = abs(_eval_sym(Gamma[1, 1, j], vals))
        if v1 > 1e-10
            vanish_ok = false
            println("  WARNING: Gamma^tau_{tau,$j} = $v1")
        end
    end

    if vanish_ok
        passed += 1
        println("  All Gamma^tau_{tau,*} = 0 ... PASSED")
    else
        failed += 1
        println("  Vanishing temporal Christoffels ... FAILED")
    end

    # ------------------------------------------------------------------
    # 3. Ricci tensor: R_{tau tau} = -3 addot / a
    #    (Carroll Eq. 8.38)
    # ------------------------------------------------------------------
    println("\n--- 3. Ricci tensor R_{tau,tau} = -3 addot / a ---")

    R_tt_expected = -3.0 * vals[addot] / vals[a_bare]
    R_tt_computed = _eval_sym(Ric[1, 1], vals)

    ok = abs(R_tt_computed - R_tt_expected) < 1e-10
    if ok
        passed += 1
    else
        failed += 1
    end
    status = ok ? "PASSED" : "FAILED"
    println("  R_{tau,tau} = $R_tt_computed (expected $R_tt_expected) ... $status")

    # ------------------------------------------------------------------
    # 4. Spatial Ricci component: R_{chi chi}
    #    R_{chi chi} = (a*addot + 2*adot^2 + 2k) / (1-k*chi^2)
    #    (Carroll Eq. 8.39, adapted for comoving coords)
    # ------------------------------------------------------------------
    println("\n--- 4. Spatial Ricci component R_{chi,chi} ---")

    R_xx_expected = (vals[a_bare] * vals[addot] + 2.0 * vals[adot]^2 + 2.0 * vals[k]) /
                    (1 - vals[k] * vals[chi]^2)
    R_xx_computed = _eval_sym(Ric[2, 2], vals)

    ok = abs(R_xx_computed - R_xx_expected) < 1e-10
    if ok
        passed += 1
    else
        failed += 1
    end
    status = ok ? "PASSED" : "FAILED"
    println("  R_{chi,chi} = $R_xx_computed (expected $R_xx_expected) ... $status")

    # ------------------------------------------------------------------
    # 5. Off-diagonal Ricci components vanish (FLRW is diagonal)
    # ------------------------------------------------------------------
    println("\n--- 5. Off-diagonal Ricci components vanish ---")

    offdiag_ok = true
    max_offdiag = 0.0
    for i in 1:4, j in 1:4
        if i != j
            val = abs(_eval_sym(Ric[i, j], vals))
            max_offdiag = max(max_offdiag, val)
            if val > 1e-10
                offdiag_ok = false
                println("  WARNING: R_{$i,$j} = $val")
            end
        end
    end

    if offdiag_ok
        passed += 1
        println("  All off-diagonal R_{ab} = 0 (max: $max_offdiag) ... PASSED")
    else
        failed += 1
        println("  Off-diagonal Ricci ... FAILED")
    end

    # ------------------------------------------------------------------
    # 6. Ricci scalar: R = 6(addot/a + adot^2/a^2 + k/a^2)
    #    (Carroll Eq. 8.40)
    # ------------------------------------------------------------------
    println("\n--- 6. Ricci scalar ---")

    R_expected = 6.0 * (vals[addot] / vals[a_bare] +
                        vals[adot]^2 / vals[a_bare]^2 +
                        vals[k] / vals[a_bare]^2)
    R_computed = _eval_sym(R_scalar, vals)

    ok = abs(R_computed - R_expected) < 1e-10
    if ok
        passed += 1
    else
        failed += 1
    end
    status = ok ? "PASSED" : "FAILED"
    println("  R = $R_computed (expected $R_expected = 6*(addot/a + adot^2/a^2 + k/a^2)) ... $status")

    # ------------------------------------------------------------------
    # 7. Multi-point verification (non-zero curvature, k != 0)
    # ------------------------------------------------------------------
    println("\n--- 7. Multi-point Ricci scalar verification ---")

    test_points = [
        Dict(a_bare => 1.0, adot => 1.0, addot => -0.5, k => 1.0,
             chi => 0.3, theta_f => 1.0, phi_f => 0.5, tau => 0.0),
        Dict(a_bare => 3.0, adot => 0.2, addot => 0.0, k => -1.0,
             chi => 0.4, theta_f => π / 3, phi_f => 1.0, tau => 0.0),
        Dict(a_bare => 0.5, adot => 2.0, addot => -1.0, k => 0.0,
             chi => 1.0, theta_f => π / 4, phi_f => 0.0, tau => 0.0),
    ]

    multi_ok = true
    for (idx, ev) in enumerate(test_points)
        R_exp = 6.0 * (ev[addot] / ev[a_bare] +
                        ev[adot]^2 / ev[a_bare]^2 +
                        ev[k] / ev[a_bare]^2)
        R_comp = _eval_sym(R_scalar, ev)
        ok_pt = abs(R_comp - R_exp) < 1e-10
        if !ok_pt
            multi_ok = false
        end
        println("  Point $idx (a=$(ev[a_bare]), adot=$(ev[adot]), " *
                "addot=$(ev[addot]), k=$(ev[k])): " *
                "R=$R_comp (expected $R_exp) ... $(ok_pt ? "ok" : "MISMATCH")")
    end

    if multi_ok
        passed += 1
        println("  Multi-point Ricci scalar ... PASSED")
    else
        failed += 1
        println("  Multi-point Ricci scalar ... FAILED")
    end

    # ------------------------------------------------------------------
    # 8. R_{tau,tau} at multiple points
    # ------------------------------------------------------------------
    println("\n--- 8. R_{tau,tau} = -3*addot/a at multiple points ---")

    rtt_ok = true
    for (idx, ev) in enumerate(test_points)
        R_tt_exp = -3.0 * ev[addot] / ev[a_bare]
        R_tt_comp = _eval_sym(Ric[1, 1], ev)
        ok_pt = abs(R_tt_comp - R_tt_exp) < 1e-10
        if !ok_pt
            rtt_ok = false
        end
        println("  Point $idx: R_{tt}=$R_tt_comp (expected $R_tt_exp) ... " *
                "$(ok_pt ? "ok" : "MISMATCH")")
    end

    if rtt_ok
        passed += 1
        println("  Multi-point R_{tau,tau} ... PASSED")
    else
        failed += 1
        println("  Multi-point R_{tau,tau} ... FAILED")
    end

    # ------------------------------------------------------------------
    # 9. Flat space limit: k=0, a=const (adot=addot=0) gives R=0
    # ------------------------------------------------------------------
    println("\n--- 9. Flat space limit (k=0, a=const) ---")

    flat_vals = Dict(a_bare => 1.0, adot => 0.0, addot => 0.0, k => 0.0,
                     chi => 0.5, theta_f => π / 2, phi_f => 0.0, tau => 0.0)

    R_flat = _eval_sym(R_scalar, flat_vals)
    ok = abs(R_flat) < 1e-10
    if ok
        passed += 1
    else
        failed += 1
    end
    println("  R(flat) = $R_flat ... $(ok ? "PASSED" : "FAILED")")

    # Also check all Ricci components vanish in flat limit
    flat_ricci_ok = true
    for i in 1:4, j in 1:4
        val = abs(_eval_sym(Ric[i, j], flat_vals))
        if val > 1e-10
            flat_ricci_ok = false
        end
    end

    if flat_ricci_ok
        passed += 1
        println("  All R_{ab}(flat) = 0 ... PASSED")
    else
        failed += 1
        println("  Flat Ricci check ... FAILED")
    end

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    total = passed + failed
    println("FLRW verification: $passed/$total checks passed")
    if failed > 0
        println("WARNING: $failed check(s) failed!")
    else
        println("All FLRW cosmology properties verified!")
    end
    println("=" ^ 70)

    @assert failed == 0 "FLRW verification had $failed failure(s)"
end

run_flrw()
