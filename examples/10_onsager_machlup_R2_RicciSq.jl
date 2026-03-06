# ============================================================================
# Onsager-Machlup Action: alpha R^2 - beta R_{mu nu} R^{mu nu}
#
# Compute the two-point functions (propagators) for all degrees of freedom
# of linearized gravity around flat space with this quadratic-in-curvature
# action, relevant for stochastic classical gravity.
#
# Degrees of freedom (Bardeen gauge, SVT decomposition):
#   - Tensor:  h^TT_{ij}  (2 polarizations, transverse-traceless)
#   - Vector:  V_i        (2 components, transverse)
#   - Scalar:  (Phi, psi) (2 coupled scalars)
#
# Convention: h_{00} = 2Phi, h_{0i} = V_i, h_{ij} = 2 psi delta_{ij} + h^TT_{ij}
# Signature (-,+,+,+), p^2 = omega^2 - k^2
# ============================================================================

using TensorGR

# --- Part 1: Verify linearized curvature from TensorGR perturbation engine ---

println("=" ^ 72)
println("ONSAGER-MACHLUP ACTION: alpha R^2 - beta R_{mu nu} R^{mu nu}")
println("=" ^ 72)

println("\n--- Part 1: Linearized curvature (TensorGR perturbation engine) ---\n")

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp = define_metric_perturbation!(reg, :g, :h)

    delta_Ric = δricci(mp, down(:a), down(:b), 1)
    println("delta R_{ab} = ", to_unicode(delta_Ric))

    delta_R = δricci_scalar(mp, 1)
    println("delta R      = ", to_unicode(delta_R))
end

# --- Part 2: SVT decomposition ---

println("\n--- Part 2: Bardeen gauge SVT decomposition ---\n")
println("h_{00} = 2 Phi,  h_{0i} = V_i (transverse),  h_{ij} = 2 psi delta_{ij} + h^TT_{ij}")
println("h = eta^{mu nu} h_{mu nu} = -2 Phi + 6 psi")
println()

println("Linearized Ricci in Fourier space (p^2 = omega^2 - k^2):")
println()
println("  R_{00}       = k^2 Phi + 3 omega^2 psi")
println("  R_{0i}|_V    = (k^2 / 2) V_i")
println("  R_{0i}|_S    = 2 omega k_i psi")
println("  R_{ij}|_S    = -k_i k_j (Phi - psi) - p^2 psi delta_{ij}")
println("  R_{ij}|_TT   = -(p^2 / 2) h^TT_{ij}")
println("  R            = -2 k^2 Phi + (4 k^2 - 6 omega^2) psi")

# --- Part 3: Kinetic matrices for each sector ---

println("\n--- Part 3: Quadratic action by sector ---\n")

println("Action: S = alpha R^2 - beta R_{mu nu} R^{mu nu}")
println()

# ---- Tensor sector ----
println("TENSOR sector (h^TT_{ij}):")
println("  R_{mu nu} R^{mu nu}|_TT = (p^4 / 4) h^TT_{ij} h^TT_{ij}")
println("  R^2|_TT = 0   (traceless mode)")
println("  => S_TT = -beta (p^4 / 4) h^TT_{ij} h^TT_{ij}")
println("  => M_TT = -beta p^4 / 4")
println()

# ---- Vector sector ----
println("VECTOR sector (V_i, transverse):")
println("  R_{mu nu} R^{mu nu}|_V = (k^2 p^2 / 2) V_i V_i")
println("  R^2|_V = 0   (traceless mode)")
println("  => S_V = -beta (k^2 p^2 / 2) V_i V_i")
println("  => M_V = -beta k^2 p^2 / 2")
println()

# ---- Scalar sector ----
println("SCALAR sector (Phi, psi coupled):")
println("  R_{mu nu} R^{mu nu}|_S computed from:")
println("    R_{00}^2, -2 R_{0i} R_{0i}, R_{ij} R_{ij}")
println()

# --- Part 4: Symbolic computation with Symbolics.jl ---

println("--- Part 4: Symbolic scalar kinetic matrix ---\n")

symbolics_available = try
    @eval using Symbolics
    true
catch
    println("Symbolics.jl not available; using Expr-tree fallback")
    false
end

if symbolics_available
    @eval begin
        @variables k_var omega_var alpha_var beta_var

        p_sq = omega_var^2 - k_var^2  # p^2 = omega^2 - k^2

        # ---- Tensor propagator ----
        M_TT = -beta_var * p_sq^2 / 4
        println("TENSOR:")
        println("  M_TT = -beta p^4 / 4")
        println("  <h^TT_{ij} h^TT_{kl}> = (-4 / (beta p^4)) Pi^TT_{ijkl}")
        println()

        # Build TT projector
        Pi_TT = tt_projector(down(:i), down(:j), down(:k), down(:l))
        println("  Pi^TT_{ijkl} = ", to_unicode(Pi_TT))
        println()

        # ---- Vector propagator ----
        M_V = -beta_var * k_var^2 * p_sq / 2
        println("VECTOR:")
        println("  M_V = -beta k^2 p^2 / 2")
        println("  <V_i V_j> = (-2 / (beta k^2 p^2)) P^T_{ij}")
        println()

        P_T = transverse_projector(down(:i), down(:j))
        println("  P^T_{ij} = ", to_unicode(P_T))
        println()

        # ---- Scalar kinetic matrix ----
        # From the linearized Ricci components in Fourier space:
        #
        # R_{mu nu} R^{mu nu}|_S contributions:
        #   Phi^2: R_{00}^2|_{Phi^2} + R_{ij}R_{ij}|_{Phi^2} = k^4 + k^4 = 2k^4
        #   Phi*psi cross-term (full): 6k^2 omega^2 + 2(-k^4 + k^2 p^2) = 8k^2 omega^2 - 4k^4
        #   psi^2: 9omega^4 - 8omega^2 k^2 + k^4 - 2k^2 p^2 + 3p^4
        #        = 12omega^4 - 16omega^2 k^2 + 6k^4
        #
        # R^2|_S = (-2k^2 Phi + (4k^2 - 6omega^2) psi)^2:
        #   Phi^2: 4k^4
        #   Phi*psi cross-term (full): -4k^2(4k^2 - 6omega^2)
        #   psi^2: (4k^2 - 6omega^2)^2

        # Kinetic matrix for S = alpha R^2 - beta R_{mu nu} R^{mu nu}:
        # M_{IJ} = alpha * (R^2 coefficient) - beta * (R_{mu nu} R^{mu nu} coefficient)
        # where the "coefficient" of Phi_I Phi_J means M_{IJ} with S = Phi^I M_{IJ} Phi^J

        # Phi^2 coefficient in R^2 = 4k^4, in R_mn R^mn = 2k^4
        M_PP = alpha_var * 4k_var^4 - beta_var * 2k_var^4

        # Cross-term: M_{Phi psi} = half the Phi*psi coefficient in S
        # Phi*psi coefficient in R^2: -4k^2(4k^2 - 6omega^2) = -16k^4 + 24k^2 omega^2
        # Phi*psi coefficient in R_mn R^mn: 8k^2 omega^2 - 4k^4
        # So full cross-term = alpha(-16k^4+24k^2 omega^2) - beta(8k^2 omega^2-4k^4)
        # M_{Phi psi} = half of that
        cross_R2 = -4k_var^2 * (4k_var^2 - 6omega_var^2)
        cross_RR = 8k_var^2 * omega_var^2 - 4k_var^4
        M_Pp = (alpha_var * cross_R2 - beta_var * cross_RR) / 2

        # psi^2 coefficient in R^2: (4k^2-6omega^2)^2 = 16k^4-48k^2 omega^2+36omega^4
        # psi^2 coefficient in R_mn R^mn: 12omega^4-16omega^2 k^2+6k^4
        M_pp = alpha_var * (4k_var^2 - 6omega_var^2)^2 - beta_var * (12omega_var^4 - 16omega_var^2 * k_var^2 + 6k_var^4)

        println("SCALAR kinetic matrix M (S = Phi^I M_{IJ} Phi^J):")
        println()
        M_PP_s = Symbolics.simplify(M_PP; expand=true)
        M_Pp_s = Symbolics.simplify(M_Pp; expand=true)
        M_pp_s = Symbolics.simplify(M_pp; expand=true)
        println("  M_{Phi Phi} = ", M_PP_s)
        println("  M_{Phi psi} = ", M_Pp_s)
        println("  M_{psi psi} = ", M_pp_s)
        println()

        # Compact form using p^2 and the combinations A = 2alpha - beta, B = 3alpha - beta
        println("Compact form (A = 2alpha - beta, B = 3alpha - beta):")
        println("  M_{Phi Phi} = 2A k^4")
        println("  M_{Phi psi} = 4B k^2 p^2 + 2A k^4")
        println("  M_{psi psi} = 12B p^4 + 8B k^2 p^2 + 2A k^4")
        println()

        # Build symbolic QuadraticForm
        entries_scalar = Dict(
            (:Phi, :Phi) => M_PP,
            (:Phi, :psi) => M_Pp,
            (:psi, :psi) => M_pp
        )
        qf = symbolic_quadratic_form(entries_scalar, [:Phi, :psi];
                                      variables=[:k_var, :omega_var, :alpha_var, :beta_var])

        # ---- Determinant ----
        println("--- Determinant ---\n")

        det_expr = determinant(qf)
        det_simplified = Symbolics.simplify(det_expr; expand=true)
        println("  det(M) = ", det_simplified)
        println()
        println("  Analytic: det(M) = -8 beta (3alpha - beta) k^4 p^4")
        println()

        # Verify determinant formula numerically
        println("  Numerical verification:")
        test_points = [
            (alpha=0.5, beta=0.25, omega=2.0, k=1.0),
            (alpha=1.0, beta=0.0,  omega=1.5, k=2.0),
            (alpha=0.3, beta=1.0,  omega=3.0, k=1.0),
            (alpha=2.0, beta=0.5,  omega=1.0, k=1.0),
        ]

        local all_pass = true
        for pt in test_points
            vars = Dict(:alpha_var => pt.alpha, :beta_var => pt.beta,
                        :omega_var => pt.omega, :k_var => pt.k)
            p2 = pt.omega^2 - pt.k^2

            det_numeric = sym_eval(det_expr, vars)
            det_analytic = -8 * pt.beta * (3pt.alpha - pt.beta) * pt.k^4 * p2^2

            match = abs(det_numeric - det_analytic) < 1e-8
            all_pass &= match
            status = match ? "pass" : "FAIL"
            println("    alpha=$(pt.alpha), beta=$(pt.beta), omega=$(pt.omega), k=$(pt.k): " *
                    "num=$(round(det_numeric, digits=4)), ana=$(round(det_analytic, digits=4))  $status")
        end
        println("  All determinant checks: ", all_pass ? "PASS" : "FAIL")

        # ---- Propagators ----
        println("\n--- Part 5: Two-point functions (propagators = M^{-1}) ---\n")

        prop = propagator(qf)

        println("SCALAR propagator matrix G = M^{-1}:")
        for i in 1:2, j in i:2
            entry = Symbolics.simplify(prop.matrix[i, j]; expand=true)
            println("  G[$(qf.fields[i]),$(qf.fields[j])] = ", entry)
        end
        println()

        # Analytic propagator via Cramer's rule:
        # G_{Phi Phi} = M_{psi psi} / det
        # G_{Phi psi} = -M_{Phi psi} / det
        # G_{psi psi} = M_{Phi Phi} / det
        #
        # With det = -8 beta B k^4 p^4 (B = 3alpha - beta, A = 2alpha - beta):
        #
        # G_{Phi Phi} = (12B p^4 + 8B k^2 p^2 + 2A k^4) / (-8 beta B k^4 p^4)
        #             = -3/(2 beta k^4) - 1/(beta k^2 p^2) - A/(4 beta B p^4)
        #
        # G_{Phi psi} = -(4B k^2 p^2 + 2A k^4) / (-8 beta B k^4 p^4)
        #             = 1/(2 beta k^2 p^2) + A/(4 beta B p^4)
        #
        # G_{psi psi} = 2A k^4 / (-8 beta B k^4 p^4)
        #             = -A/(4 beta B p^4) = -(2alpha - beta)/(4 beta (3alpha - beta) p^4)

        println("Partial-fraction decomposition (A = 2alpha - beta, B = 3alpha - beta):")
        println()
        println("  <Phi Phi> = -3/(2 beta k^4) - 1/(beta k^2 p^2) - A/(4 beta B p^4)")
        println("  <Phi psi> =  1/(2 beta k^2 p^2) + A/(4 beta B p^4)")
        println("  <psi psi> = -A/(4 beta B p^4)")
        println()
        println("  <V_i V_j>               = -2/(beta k^2 p^2) P^T_{ij}")
        println("  <h^TT_{ij} h^TT_{kl}>   = -4/(beta p^4) Pi^TT_{ijkl}")
        println()

        # Numerical verification of propagators
        println("--- Numerical verification of propagators ---\n")

        for pt in test_points
            p2 = pt.omega^2 - pt.k^2
            abs(p2) < 1e-10 && continue
            abs(pt.beta) < 1e-10 && continue              # beta=0: no propagator
            abs(3pt.alpha - pt.beta) < 1e-10 && continue  # conformal point

            vars = Dict(:alpha_var => pt.alpha, :beta_var => pt.beta,
                        :omega_var => pt.omega, :k_var => pt.k)

            A = 2pt.alpha - pt.beta
            B = 3pt.alpha - pt.beta

            # From M^{-1}
            G_PP_tgr = sym_eval(prop.matrix[1,1], vars)
            G_Pp_tgr = sym_eval(prop.matrix[1,2], vars)
            G_pp_tgr = sym_eval(prop.matrix[2,2], vars)

            # Analytic partial fractions
            G_PP_ref = -3/(2pt.beta*pt.k^4) - 1/(pt.beta*pt.k^2*p2) - A/(4pt.beta*B*p2^2)
            G_Pp_ref = 1/(2pt.beta*pt.k^2*p2) + A/(4pt.beta*B*p2^2)
            G_pp_ref = -A/(4pt.beta*B*p2^2)

            match_PP = abs(G_PP_tgr - G_PP_ref) < 1e-8
            match_Pp = abs(G_Pp_tgr - G_Pp_ref) < 1e-8
            match_pp = abs(G_pp_tgr - G_pp_ref) < 1e-8

            println("  alpha=$(pt.alpha), beta=$(pt.beta), omega=$(pt.omega), k=$(pt.k), p^2=$(round(p2,digits=2)):")
            println("    <PhiPhi>: M^-1=$(round(G_PP_tgr,digits=6)), analytic=$(round(G_PP_ref,digits=6))  $(match_PP ? "pass" : "FAIL")")
            println("    <Phipsi>: M^-1=$(round(G_Pp_tgr,digits=6)), analytic=$(round(G_Pp_ref,digits=6))  $(match_Pp ? "pass" : "FAIL")")
            println("    <psipsi>: M^-1=$(round(G_pp_tgr,digits=6)), analytic=$(round(G_pp_ref,digits=6))  $(match_pp ? "pass" : "FAIL")")
        end

        # ---- Special limits ----
        println("\n--- Part 6: Special limits and physics ---\n")

        # beta -> 3alpha: B = 0, conformal gravity (Weyl^2)
        # Gauss-Bonnet: R^2 - 4 R_mn R^mn + R_mnrs R^mnrs is topological
        # In 4D linearized: R_mnrs R^mnrs = 4 R_mn R^mn - R^2, so
        # Weyl^2 = R_mn R^mn - (1/3) R^2 (linearized, 4D).
        # Our action with alpha = beta/3 gives: (beta/3) R^2 - beta R_mn R^mn = -beta Weyl^2
        println("Conformal point: beta = 3 alpha  (B = 3alpha - beta = 0)")
        println("  Action becomes: alpha R^2 - 3alpha R_{mu nu} R^{mu nu} = -3alpha Weyl^2")
        println("  det(M) = 0 => scalar sector is degenerate (conformal symmetry)")
        println("  Weyl tensor is traceless => scalar dof drops out")
        println()

        vars_conf = Dict(:alpha_var => 1.0/3.0, :beta_var => 1.0,
                         :omega_var => 2.0, :k_var => 1.0)
        det_conf = sym_eval(det_expr, vars_conf)
        println("  Verify: alpha=1/3, beta=1 => det = $(det_conf)  $(abs(det_conf) < 1e-10 ? "= 0 (degenerate)" : "!= 0")")
        println()

        # alpha = 0: pure -beta R_mn R^mn
        println("Pure Ricci-squared: alpha = 0")
        println("  A = -beta, B = -beta")
        println("  <psi psi> = -(-beta)/(4 beta (-beta) p^4) = -1/(4 beta^2 p^4)")
        println("  All propagators scale as 1/beta or 1/beta^2")
        println()

        # beta = 0: pure alpha R^2 (scalar gravity, no tensor/vector propagators)
        println("Pure R-squared: beta = 0")
        println("  M_TT = 0, M_V = 0 => tensor and vector modes have NO kinetic term")
        println("  Only scalar modes propagate (conformally coupled scalar)")
        println("  det(M_scalar) = -8*0*(3alpha-0)*k^4*p^4 = 0 => also degenerate!")
        println("  This reflects the enhanced gauge symmetry of pure R^2 theory")
        println()

        # alpha = beta/2: A = 0, psi decouples
        println("Trace-decoupling point: alpha = beta/2  (A = 2alpha - beta = 0)")
        println("  <psi psi> = 0 => trace mode psi has no two-point function")
        println("  M_{Phi Phi} = 0, M_{Phi psi} = 4B k^2 p^2")
        println("  Scalar sector reduces to off-diagonal coupling only")
    end
else
    # Fallback using Expr trees
    @eval begin
        println("Using Expr-tree algebra (Symbolics.jl not found)\n")

        # Kinetic matrix entries using Expr trees
        M_PP = :((4alpha - 2beta) * k^4)
        M_Pp = :(((12alpha - 4beta) * k^2 * omega^2 + (-8alpha + 2beta) * k^4) / 2)
        M_pp = :((36alpha - 12beta) * omega^4 + (-48alpha + 16beta) * omega^2 * k^2 + (16alpha - 6beta) * k^4)

        entries = Dict((:Phi, :Phi) => M_PP, (:Phi, :psi) => M_Pp, (:psi, :psi) => M_pp)
        qf = quadratic_form(entries, [:Phi, :psi])
        println("Scalar kinetic matrix:")
        println(qf)
        println()

        det_val = determinant(qf)
        println("det(M) = ", det_val)
        println()
        println("Analytic: det(M) = -8 beta (3alpha - beta) k^4 p^4  where p^2 = omega^2 - k^2")
        println()

        prop = propagator(qf)
        println("Scalar propagator G = M^{-1}:")
        println(prop)
    end
end

# --- Summary ---

println("\n" * "=" ^ 72)
println("SUMMARY: Two-point functions for S = alpha R^2 - beta R_{mu nu} R^{mu nu}")
println("=" ^ 72)
println()
println("Define A = 2alpha - beta,  B = 3alpha - beta,  p^2 = omega^2 - k^2")
println()
println("TENSOR (2 dof, transverse-traceless):")
println("  <h^TT_{ij}(p) h^TT_{kl}(-p)> = (-4 / (beta p^4)) Pi^TT_{ijkl}")
println()
println("VECTOR (2 dof, transverse):")
println("  <V_i(p) V_j(-p)> = (-2 / (beta k^2 p^2)) P^T_{ij}")
println()
println("SCALAR (2 coupled dof):")
println("  <Phi(p) Phi(-p)> = -3/(2 beta k^4) - 1/(beta k^2 p^2) - A/(4 beta B p^4)")
println("  <Phi(p) psi(-p)> =  1/(2 beta k^2 p^2) + A/(4 beta B p^4)")
println("  <psi(p) psi(-p)> = -A/(4 beta B p^4)")
println()
println("Pole structure:")
println("  1/k^4       : instantaneous (Coulomb-like)")
println("  1/(k^2 p^2) : mixed (one spatial, one covariant propagation)")
println("  1/p^4       : fourth-derivative propagator (improved UV, stochastic DOF)")
println()
println("Special points:")
println("  B = 0 (beta = 3alpha): conformal / Weyl^2 point, scalar sector degenerate")
println("  A = 0 (beta = 2alpha): trace mode psi decouples")
println("  beta = 0:              tensor & vector nonpropagating (pure R^2)")
println()
println("The 1/p^4 poles (vs 1/p^2 in GR) reflect the fourth-derivative nature.")
println("In the Onsager-Machlup framework, these are noise-dressed propagators")
println("with convergent path integrals when beta < 0 or alpha, beta chosen so M > 0.")
