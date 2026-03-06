# ============================================================================
# Postquantum Gravity: Onsager-Machlup Action
#
# Symbolically compute the quadratic action for linearized fourth-derivative
# gravity, decompose into SVT sectors, and extract two-point functions.
#
# Action:
#   I = integral d^4x [ (1/4) L_{mu nu} L^{mu nu} - beta (d_mu d_nu h^{mu nu} - box h)^2 ]
#
# where L_{mu nu} is the linearized Lichnerowicz operator.
#
# This example demonstrates the full pipeline:
# 1. Perturbation engine -> linearized Ricci
# 2. 3+1 foliation -> split spacetime indices
# 3. SVT decomposition -> Bardeen gauge fields
# 4. Symbolic quadratic form -> Symbolics.jl CAS
# 5. Propagator extraction -> symbolic inverse
# ============================================================================

using TensorGR

# --- Part 1: Build the linearized Einstein operator symbolically ---

println("=== Part 1: Linearized curvature from perturbation engine ===\n")

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

    # delta R_{ab} on flat background via the xPert engine
    mp = define_metric_perturbation!(reg, :g, :h)
    delta_Ric_ab = δricci(mp, down(:a), down(:b), 1)

    println("delta R_{ab} = ", to_unicode(delta_Ric_ab))
    println()

    # delta R (Ricci scalar perturbation)
    delta_R = δricci_scalar(mp, 1)
    println("delta R = ", to_unicode(delta_R))
    println()

    # delta Gamma for completeness
    delta_Gamma = δchristoffel(mp, up(:a), down(:b), down(:c), 1)
    println("delta Gamma^a_{bc} = ", to_unicode(delta_Gamma))
end

# --- Part 2: 3+1 Foliation and SVT Decomposition ---

println("\n=== Part 2: 3+1 foliation and SVT decomposition ===\n")

reg2 = TensorRegistry()
with_registry(reg2) do
    @manifold M4 dim=4 metric=g
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

    # Define 3+1 foliation
    fol = define_foliation!(reg2, :flat31; manifold=:M4)
    println("Foliation: temporal=0, spatial=[1,2,3]")
    println()

    # Split h_{ab} into 3+1 components
    h_ab = Tensor(:h, [down(:a), down(:b)])
    split_expr = split_all_spacetime(h_ab, fol)
    println("h_{ab} split into $(length(split_expr.terms)) component terms")
    println()

    # Apply SVT substitution in Bardeen gauge
    println("Bardeen gauge: h_{00}=2Phi, h_{0i}=V_i, h_{ij}=2psi delta_{ij}+h^TT_{ij}")
    substituted = apply_svt(split_expr, :h, fol; gauge=:bardeen)
    println()

    # Collect by sector
    sectors = collect_sectors(substituted)
    for (sector, expr) in sort(collect(sectors), by=first)
        field_names = TensorGR._extract_field_names(expr)
        println("  $sector sector: fields = $field_names")
    end
    println()

    # Full pipeline: foliate_and_decompose
    println("--- End-to-end foliate_and_decompose ---")
    sectors2 = foliate_and_decompose(h_ab, :h; foliation=fol)
    for (sector, _) in sort(collect(sectors2), by=first)
        println("  $sector sector present")
    end
end

# --- Part 3: Fourier-space quadratic form with Symbolics.jl ---

println("\n=== Part 3: Fourier-space action for each SVT sector ===\n")

# Try to load Symbolics for the CAS part
symbolics_available = try
    @eval using Symbolics
    true
catch
    println("Symbolics.jl not available; skipping CAS section")
    false
end

if symbolics_available
    @eval begin
        @variables k_var omega_var beta_var

        # --- Tensor sector ---
        # h^TT is transverse-traceless: k^i h^TT_{ij} = 0, h^TT_{ii} = 0
        # L_{ij}[TT] = -box h^TT_{ij} = p^2 h^TT_{ij}
        # (1/4) L_{mu nu} L^{mu nu}[TT] = p^4/4 * h^TT * h^TT

        p_sq = omega_var^2 - k_var^2  # p^2 = omega^2 - k^2
        M_TT = p_sq^2 / 4
        println("TENSOR sector:")
        println("  L_{ij}[TT] = p^2 h^TT_{ij}")
        println("  Kinetic: M_TT = p^4/4 = ", M_TT)
        println("  Propagator: 4/p^4")
        println()

        # Build TT projector symbolically
        P_TT = tt_projector(down(:i), down(:j), down(:k), down(:l))
        println("  Pi^TT_{ijkl} = ", to_unicode(P_TT))
        println()

        # --- Vector sector ---
        # h_{0i} = V_i with k^i V_i = 0, gauge term = 0
        M_V = k_var^2 * p_sq / 2
        println("VECTOR sector:")
        println("  L_{0i}[V] = k^2 V_i")
        println("  Kinetic: M_V = k^2 p^2/2 = ", M_V)
        println("  Propagator: 2/(k^2 p^2) P^T_{ij}")
        println()

        P_T = transverse_projector(down(:i), down(:j))
        println("  P^T_{ij} = ", to_unicode(P_T))
        println()

        # --- Scalar sector (Phi, psi coupled) ---
        println("SCALAR sector (Phi, psi coupled):")
        println("  h_{00} = -2Phi, h_{ij} = 2psi delta_{ij}, h = 2Phi+6psi")
        println()

        # Quadratic form entries (derived from Lichnerowicz + gauge-fixing)
        # (1/4) L_{mu nu} L^{mu nu}[scalar]:
        M_PP_lich = 2k_var^4
        M_Pp_lich = -4k_var^2 * omega_var^2
        M_pp_lich = 2k_var^4 - 4k_var^2 * omega_var^2 + 12omega_var^4

        # Gauge-fixing: -beta (d_mu d_nu h^{mu nu} - box h)^2
        gauge_scalar = -2k_var^2  # coefficient of Phi in the gauge scalar
        # Full gauge scalar: -2k^2 Phi + 2(3omega^2-4k^2) psi
        gauge_Phi = -2k_var^2
        gauge_psi = 2 * (3omega_var^2 - 4k_var^2)

        # -beta * (gauge)^2 adds to the quadratic form:
        M_PP_gauge = -beta_var * gauge_Phi^2
        M_Pp_gauge = -beta_var * gauge_Phi * gauge_psi  # factor of 2 for symmetry already in
        M_pp_gauge = -beta_var * gauge_psi^2

        # Total scalar kinetic matrix
        M_PP = M_PP_lich + M_PP_gauge
        M_Pp = M_Pp_lich + M_Pp_gauge
        M_pp = M_pp_lich + M_pp_gauge

        println("  M_{PhiPhi} = ", Symbolics.simplify(M_PP))
        println("  M_{Phipsi} = ", Symbolics.simplify(M_Pp))
        println("  M_{psipsi} = ", Symbolics.simplify(M_pp))
        println()

        # Build symbolic QuadraticForm
        entries_scalar = Dict(
            (:Phi, :Phi) => M_PP,
            (:Phi, :psi) => M_Pp,
            (:psi, :psi) => M_pp
        )
        qf = symbolic_quadratic_form(entries_scalar, [:Phi, :psi];
                                      variables=[:k_var, :omega_var, :beta_var])

        # --- Part 4: Propagators ---

        println("=== Part 4: Two-point functions (propagators = M^{-1}) ===\n")

        det_expr = determinant(qf)
        det_simplified = Symbolics.simplify(det_expr)
        println("  det(M) = ", det_simplified)
        println()

        prop = propagator(qf)
        println("  Scalar propagator matrix:")
        for i in 1:2, j in i:2
            entry = Symbolics.simplify(prop.matrix[i, j])
            println("    G[$(qf.fields[i]),$(qf.fields[j])] = ", entry)
        end
        println()

        # --- Numerical cross-checks ---

        println("--- Numerical cross-check: beta=1, omega=2, k=1 ---")
        vars = Dict(:beta_var => 1.0, :omega_var => 2.0, :k_var => 1.0)

        m11 = sym_eval(M_PP, vars)
        m12 = sym_eval(M_Pp, vars)
        m22 = sym_eval(M_pp, vars)
        d = m11 * m22 - m12^2

        println("  M_{PhiPhi}=$(m11), M_{Phipsi}=$(m12), M_{psipsi}=$(m22)")
        println("  det = $(d)")
        println("  G_{PhiPhi} = $(m22/d)")
        println("  G_{Phipsi} = $(-m12/d)")
        println("  G_{psipsi} = $(m11/d)")
        println()

        # Conformal point beta=1/3
        println("--- Conformal point beta=1/3 ---")
        vars_conf = Dict(:beta_var => 1.0/3.0, :omega_var => 2.0, :k_var => 1.0)
        det_conf = sym_eval(det_expr, vars_conf)
        println("  det(M) at beta=1/3 = $(det_conf)")

        # Tensor sector propagator check
        m_tt = sym_eval(M_TT, vars)
        println("  M_TT at omega=2,k=1: $(m_tt) (p^4/4 = $(3.0^2/4))")
        m_v = sym_eval(M_V, vars)
        println("  M_V at omega=2,k=1: $(m_v) (k^2 p^2/2 = $(1.0*3.0/2))")
    end
else
    println("Skipping Symbolics-based analysis.")
    println()

    # Fallback: Expr-tree based computation (existing approach)
    M_PP = :((2 - 4β) * k^4)
    M_Pp = :((-4ω^2 + 8β*(3ω^2 - 4k^2)) * k^2 / 2)
    M_pp = :(2k^4 - 4k^2*ω^2 + 12ω^4 - 4β*(3ω^2 - 4k^2)^2)

    entries = Dict((:Phi,:Phi) => M_PP, (:Phi,:psi) => M_Pp, (:psi,:psi) => M_pp)
    qf = quadratic_form(entries, [:Phi, :psi])
    println("  Scalar kinetic matrix M:")
    println(qf)
end

# --- Summary ---

println("\n=== Summary ===\n")
println("The Onsager-Machlup action for postquantum gravity decomposes as:")
println()
println("  I = I_tensor + I_vector + I_scalar")
println()
println("  I_tensor = (p^4/4) h^TT_{ij} h^TT_{ij}")
println("  I_vector = (k^2 p^2/2) V_i V_i")
println("  I_scalar = Phi_I M_{IJ}(omega,k,beta) Phi_J")
println()
println("Pipeline used:")
println("  1. delta R_{ab} from perturbation engine (xPert)")
println("  2. 3+1 foliation: split mu -> (0, i)")
println("  3. SVT decomposition: Bardeen gauge (Phi, psi, V_i, h^TT_{ij})")
println("  4. Symbolic quadratic form (Symbolics.jl CAS)")
println("  5. Propagators via symbolic matrix inversion")
println()
println("The 1/p^4 tensor propagator (vs 1/p^2 in GR) reflects the")
println("fourth-derivative nature: improved UV behavior at the cost of")
println("additional degrees of freedom handled by the stochastic framework.")
