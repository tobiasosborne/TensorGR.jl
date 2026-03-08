# ============================================================================
# TensorGR.jl -- Course Verification: Lecture 17 -- Friedmann Equations
#
# Verifies FLRW cosmology using SYMBOLIC COMPONENTS (Symbolics.jl).
# Follows the functional-derivative technique from the existing 06_flrw.jl
# Wald verification example.
#
# The FLRW metric (comoving coordinates):
#   ds^2 = -dtau^2 + a(tau)^2 [ dchi^2/(1-k*chi^2) + chi^2 dOmega^2 ]
#
# Topics:
#   1. FLRW Christoffel symbols -- all 12 independent components
#   2. FLRW Ricci tensor components
#   3. Ricci scalar: R = 6(addot/a + adot^2/a^2 + k/a^2)
#   4. Einstein tensor: G_{tau tau} = 3(adot^2/a^2 + k/a^2)
#   5. Friedmann equations from Einstein equation
#
# References:
#   Carroll, "Spacetime and Geometry", Chapter 8
#   Wald, "General Relativity" (1984), Appendix F
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

function run_friedmann()
    println("="^70)
    println("Lecture 17: Friedmann Equations (FLRW Cosmology)")
    println("="^70)

    # ---- Define coordinates and variables ----
    @variables tau chi theta_f phi_f k
    @variables a_func(tau)         # scale factor as a function of tau
    @variables a_bare adot addot   # plain variables for substitution

    println("\nFLRW metric:")
    println("  ds^2 = -dtau^2 + a(tau)^2 [dchi^2/(1-k*chi^2) + chi^2 dOmega^2]")

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

    println("Computing Einstein tensor...")
    G_raw = symbolic_einstein(Ric_raw, R_raw, sm.g, 4)

    # ---- Substitution: functional derivatives -> plain variables ----
    D = Symbolics.Differential(tau)
    da = Symbolics.expand_derivatives(D(a_func))
    dda = Symbolics.expand_derivatives(D(da))
    sub_dict = Dict(da => adot, dda => addot, a_func => a_bare)

    # Apply to Christoffels
    Gamma = Array{Any}(undef, 4, 4, 4)
    for i in 1:4, j in 1:4, l in 1:4
        Gamma[i, j, l] = Symbolics.simplify(Symbolics.substitute(Gamma_raw[i, j, l], sub_dict))
    end

    # Apply to Ricci
    Ric = Matrix{Any}(undef, 4, 4)
    for i in 1:4, j in 1:4
        Ric[i, j] = Symbolics.simplify(Symbolics.substitute(Ric_raw[i, j], sub_dict))
    end

    # Apply to Ricci scalar
    R_scalar = Symbolics.simplify(Symbolics.substitute(R_raw, sub_dict))

    # Apply to Einstein tensor
    G = Matrix{Any}(undef, 4, 4)
    for i in 1:4, j in 1:4
        G[i, j] = Symbolics.simplify(Symbolics.substitute(G_raw[i, j], sub_dict))
    end

    passed = 0
    failed = 0

    # Evaluation point
    vals = Dict(
        a_bare => 2.0, adot => 0.5, addot => -0.1,
        k => 0.0, chi => 0.5, theta_f => pi / 2, phi_f => 0.0, tau => 0.0
    )
    a_v = 2.0; ad_v = 0.5; add_v = -0.1; k_v = 0.0; chi_v = 0.5; th_v = pi / 2

    # ------------------------------------------------------------------
    # 1. Christoffel symbols -- all 12 independent non-zero components
    # ------------------------------------------------------------------
    println("\n--- 1. FLRW Christoffel symbols (12 independent) ---")

    christoffel_checks = [
        # Gamma^tau_{spatial,spatial} = a * adot * (spatial metric factor)
        ("Gamma^tau_{chi,chi}",   (1,2,2), a_v * ad_v / (1 - k_v * chi_v^2)),
        ("Gamma^tau_{th,th}",     (1,3,3), a_v * ad_v * chi_v^2),
        ("Gamma^tau_{phi,phi}",   (1,4,4), a_v * ad_v * chi_v^2 * sin(th_v)^2),
        # Gamma^spatial_{tau,spatial} = adot/a
        ("Gamma^chi_{tau,chi}",   (2,1,2), ad_v / a_v),
        ("Gamma^th_{tau,th}",     (3,1,3), ad_v / a_v),
        ("Gamma^phi_{tau,phi}",   (4,1,4), ad_v / a_v),
        # Pure spatial Christoffels
        ("Gamma^chi_{chi,chi}",   (2,2,2), k_v * chi_v / (1 - k_v * chi_v^2)),
        ("Gamma^chi_{th,th}",     (2,3,3), -chi_v * (1 - k_v * chi_v^2)),
        ("Gamma^chi_{phi,phi}",   (2,4,4), -chi_v * (1 - k_v * chi_v^2) * sin(th_v)^2),
        ("Gamma^th_{chi,th}",     (3,2,3), 1.0 / chi_v),
        ("Gamma^th_{phi,phi}",    (3,4,4), -sin(th_v) * cos(th_v)),
        ("Gamma^phi_{chi,phi}",   (4,2,4), 1.0 / chi_v),
        ("Gamma^phi_{th,phi}",    (4,3,4), cos(th_v) / sin(th_v)),
    ]

    for (name, (i,j,l), expected) in christoffel_checks
        computed = _eval_sym(Gamma[i,j,l], vals)
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
    # 2. Ricci tensor components
    #    R_{tau tau} = -3 addot / a
    #    R_{chi chi} = (a*addot + 2*adot^2 + 2k) / (1-k*chi^2)
    #    R_{th th}   = (a*addot + 2*adot^2 + 2k) * chi^2
    #    R_{phi phi} = (a*addot + 2*adot^2 + 2k) * chi^2 * sin^2(th)
    # ------------------------------------------------------------------
    println("\n--- 2. FLRW Ricci tensor components ---")

    # R_{tau,tau}
    R_tt_expected = -3.0 * add_v / a_v
    R_tt_computed = _eval_sym(Ric[1,1], vals)
    ok = abs(R_tt_computed - R_tt_expected) < 1e-10
    if ok; passed += 1; else; failed += 1; end
    println("  R_{tau,tau} = $R_tt_computed (expected $R_tt_expected = -3*addot/a) ... $(ok ? "PASSED" : "FAILED")")

    # R_{chi,chi}
    spatial_factor = a_v * add_v + 2.0 * ad_v^2 + 2.0 * k_v
    R_xx_expected = spatial_factor / (1 - k_v * chi_v^2)
    R_xx_computed = _eval_sym(Ric[2,2], vals)
    ok = abs(R_xx_computed - R_xx_expected) < 1e-10
    if ok; passed += 1; else; failed += 1; end
    println("  R_{chi,chi} = $R_xx_computed (expected $R_xx_expected) ... $(ok ? "PASSED" : "FAILED")")

    # R_{th,th}
    R_thth_expected = spatial_factor * chi_v^2
    R_thth_computed = _eval_sym(Ric[3,3], vals)
    ok = abs(R_thth_computed - R_thth_expected) < 1e-10
    if ok; passed += 1; else; failed += 1; end
    println("  R_{th,th} = $R_thth_computed (expected $R_thth_expected) ... $(ok ? "PASSED" : "FAILED")")

    # R_{phi,phi}
    R_pp_expected = spatial_factor * chi_v^2 * sin(th_v)^2
    R_pp_computed = _eval_sym(Ric[4,4], vals)
    ok = abs(R_pp_computed - R_pp_expected) < 1e-10
    if ok; passed += 1; else; failed += 1; end
    println("  R_{phi,phi} = $R_pp_computed (expected $R_pp_expected) ... $(ok ? "PASSED" : "FAILED")")

    # Off-diagonal vanishes
    offdiag_ok = true
    for i in 1:4, j in 1:4
        i == j && continue
        val = abs(_eval_sym(Ric[i,j], vals))
        if val > 1e-10
            offdiag_ok = false
        end
    end
    if offdiag_ok; passed += 1; else; failed += 1; end
    println("  All off-diagonal R_{ab} = 0 ... $(offdiag_ok ? "PASSED" : "FAILED")")

    # ------------------------------------------------------------------
    # 3. Ricci scalar: R = 6(addot/a + adot^2/a^2 + k/a^2)
    # ------------------------------------------------------------------
    println("\n--- 3. Ricci scalar ---")

    R_expected = 6.0 * (add_v / a_v + ad_v^2 / a_v^2 + k_v / a_v^2)
    R_computed = _eval_sym(R_scalar, vals)
    ok = abs(R_computed - R_expected) < 1e-10
    if ok; passed += 1; else; failed += 1; end
    println("  R = $R_computed (expected $R_expected) ... $(ok ? "PASSED" : "FAILED")")

    # Verify at k=1 point
    vals_k1 = Dict(a_bare => 1.0, adot => 1.0, addot => -0.5, k => 1.0,
                   chi => 0.3, theta_f => 1.0, phi_f => 0.5, tau => 0.0)
    R_k1_exp = 6.0 * (-0.5/1.0 + 1.0/1.0 + 1.0/1.0)
    R_k1_comp = _eval_sym(R_scalar, vals_k1)
    ok = abs(R_k1_comp - R_k1_exp) < 1e-10
    if ok; passed += 1; else; failed += 1; end
    println("  R(k=1) = $R_k1_comp (expected $R_k1_exp) ... $(ok ? "PASSED" : "FAILED")")

    # ------------------------------------------------------------------
    # 4. Einstein tensor: G_{tau tau} = 3(adot^2/a^2 + k/a^2)
    #    This is the Friedmann equation when set equal to 8pi*rho.
    # ------------------------------------------------------------------
    println("\n--- 4. Einstein tensor G_{tau,tau} (first Friedmann eq) ---")

    G_tt_expected = 3.0 * (ad_v^2 / a_v^2 + k_v / a_v^2)
    G_tt_computed = _eval_sym(G[1,1], vals)
    ok = abs(G_tt_computed - G_tt_expected) < 1e-10
    if ok; passed += 1; else; failed += 1; end
    println("  G_{tau,tau} = $G_tt_computed (expected $G_tt_expected = 3*(H^2+k/a^2)) ... $(ok ? "PASSED" : "FAILED")")

    # Verify at k=1 point
    G_tt_k1_exp = 3.0 * (1.0 / 1.0 + 1.0 / 1.0)
    G_tt_k1_comp = _eval_sym(G[1,1], vals_k1)
    ok = abs(G_tt_k1_comp - G_tt_k1_exp) < 1e-10
    if ok; passed += 1; else; failed += 1; end
    println("  G_{tau,tau}(k=1) = $G_tt_k1_comp (expected $G_tt_k1_exp) ... $(ok ? "PASSED" : "FAILED")")

    # ------------------------------------------------------------------
    # 5. Friedmann equations
    #    First:  H^2 + k/a^2 = (8pi/3) rho   <->  G_{tt} = 8pi rho
    #    Second: addot/a = -(4pi/3)(rho + 3p) <->  from G_{ii}
    #
    #    Verify the spatial Einstein component:
    #    G_{chi chi} = -(2*a*addot + adot^2 + k) / (1-k*chi^2)
    # ------------------------------------------------------------------
    println("\n--- 5. Spatial Einstein component (second Friedmann eq) ---")

    G_xx_expected = -(2.0 * a_v * add_v + ad_v^2 + k_v) / (1 - k_v * chi_v^2)
    G_xx_computed = _eval_sym(G[2,2], vals)
    ok = abs(G_xx_computed - G_xx_expected) < 1e-10
    if ok; passed += 1; else; failed += 1; end
    println("  G_{chi,chi} = $G_xx_computed (expected $G_xx_expected) ... $(ok ? "PASSED" : "FAILED")")

    # Flat space limit: k=0, a=const -> all curvature vanishes
    println("\n--- Flat space limit (k=0, a=const) ---")
    flat_vals = Dict(a_bare => 1.0, adot => 0.0, addot => 0.0, k => 0.0,
                     chi => 0.5, theta_f => pi/2, phi_f => 0.0, tau => 0.0)
    R_flat = _eval_sym(R_scalar, flat_vals)
    ok = abs(R_flat) < 1e-10
    if ok; passed += 1; else; failed += 1; end
    println("  R(flat) = $R_flat ... $(ok ? "PASSED" : "FAILED")")

    flat_ein_ok = true
    for i in 1:4, j in 1:4
        val = abs(_eval_sym(G[i,j], flat_vals))
        if val > 1e-10
            flat_ein_ok = false
        end
    end
    if flat_ein_ok; passed += 1; else; failed += 1; end
    println("  All G_{ab}(flat) = 0 ... $(flat_ein_ok ? "PASSED" : "FAILED")")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    println("\n" * "="^70)
    total = passed + failed
    println("Lecture 17 (Friedmann) verification: $passed/$total checks passed")
    if failed > 0
        println("WARNING: $failed check(s) failed!")
    else
        println("All Friedmann equation properties verified!")
    end
    println("="^70)

    @assert failed == 0 "Lecture 17 verification had $failed failure(s)"
end

run_friedmann()
