#= Cross-check: verify Bueno-Cano parameters via TGR perturbation engine on MSS.
#
# Computes δ²(√g·L) for L = κ(R - 2Λ) + γ·I_i via the perturbation engine
# and compares spin-projected form factors against BC mass predictions.
#
# The full quadratic Lagrangian including √g determinant is:
#   Q = δ²L + ½h·δL + (⅛h² - ¼hh)·L₀
# where h = g^{ab}h_{ab} (trace), L₀ = L|_{MSS}, δL = first-order perturbation.
#
# CRITICAL: The cosmological constant -2Λ is needed for gauge invariance on MSS.
#   L₀(EH) = R₀ - 2Λ = 4Λ - 2Λ = 2Λ  (NOT 4Λ!)
#
# Pipeline: expand_perturbation → commute_covds → to_fourier → extract_kernel
#           → spin_project → compare mass poles vs dS_spectrum_6deriv
#
# Reference: Bueno & Cano (1607.06463) Eqs. (17)-(19)
=#

using TensorGR

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

"""Substitute Tensor(:Λ, []) → TScalar(Λ_val) for numeric evaluation."""
function _subst_lambda(expr::Tensor, Λ_val)
    expr.name == :Λ && isempty(expr.indices) && return TScalar(Λ_val)
    expr
end
_subst_lambda(s::TScalar, _) = s
function _subst_lambda(p::TProduct, Λ_val)
    tproduct(p.scalar, TensorExpr[_subst_lambda(f, Λ_val) for f in p.factors])
end
function _subst_lambda(s::TSum, Λ_val)
    tsum(TensorExpr[_subst_lambda(t, Λ_val) for t in s.terms])
end
function _subst_lambda(d::TDeriv, Λ_val)
    TDeriv(d.index, _subst_lambda(d.arg, Λ_val), d.covd)
end

"""Build √g correction: ½h·δL + (⅛h² - ¼h_{ab}h^{ab})·L₀.
Uses indices :z1-:z8 to avoid clashes."""
function sqrt_g_correction(L0::TensorExpr, δL::TensorExpr, metric::Symbol)
    h_A = Tensor(metric, [up(:z1), up(:z2)]) * Tensor(:h, [down(:z1), down(:z2)])
    h_B = Tensor(metric, [up(:z3), up(:z4)]) * Tensor(:h, [down(:z3), down(:z4)])
    hh  = Tensor(:h, [down(:z5), down(:z6)]) * Tensor(:h, [up(:z5), up(:z6)])
    h_C = Tensor(metric, [up(:z7), up(:z8)]) * Tensor(:h, [down(:z7), down(:z8)])

    tproduct(1 // 8, TensorExpr[h_A * h_B]) * L0 +
    tproduct(-1 // 4, TensorExpr[hh]) * L0 +
    tproduct(1 // 2, TensorExpr[h_C]) * δL
end

# ═══════════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════════

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    maximally_symmetric_background!(reg, :M4; metric=:g, cosmological_constant=:Λ)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    @define_tensor k on=M4 rank=(0,1)

    mp = define_metric_perturbation!(reg, :g, :h; curved=true, covariant_output=true)
    Λ_tensor = Tensor(:Λ, TIndex[])

    println("=" ^ 70)
    println("  BC Parameter Cross-Check via Perturbation Engine")
    println("  Action: S = κ∫√g(R - 2Λ) + γ∫√g·I_i")
    println("=" ^ 70)

    # ═══════════════════════════════════════════════════════════════════
    # Step 1: EH action L = R - 2Λ on MSS (Λ→0 sanity check: FP kernel)
    # ═══════════════════════════════════════════════════════════════════

    println("\n── Step 1: EH + Λ on MSS ──")
    R_expr = Tensor(:RicScalar, TIndex[])

    # EH perturbation orders (Λ is constant, so δΛ=0, only R contributes)
    println("  Computing δR (order 1)...")
    δR = simplify(expand_perturbation(R_expr, mp, 1); registry=reg, commute_covds_name=:∇g, maxiter=200)
    n_δR = δR isa TSum ? length(δR.terms) : 1
    println("  δR: $n_δR terms")

    println("  Computing δ²R (order 2)...")
    δ2R = simplify(δricci_scalar(mp, 2); registry=reg, commute_covds_name=:∇g, maxiter=200)
    n_δ2R = δ2R isa TSum ? length(δ2R.terms) : 1
    println("  δ²R: $n_δ2R terms")

    # Background: L₀ = R₀ - 2Λ = 4Λ - 2Λ = 2Λ (ON-SHELL condition!)
    L0_EH = tproduct(2 // 1, TensorExpr[Λ_tensor])
    println("  L₀(EH) = R₀ - 2Λ = 2Λ  (on-shell)")

    # Full quadratic EH Lagrangian including √g
    println("  Assembling Q_EH = δ²R + √g correction(2Λ, δR)...")
    corr_EH = sqrt_g_correction(L0_EH, δR, :g)
    Q_EH = δ2R + corr_EH
    Q_EH = simplify(Q_EH; registry=reg, commute_covds_name=:∇g, maxiter=200)
    n_Q = Q_EH isa TSum ? length(Q_EH.terms) : 1
    println("  Q_EH: $n_Q terms")

    # Fourier → kernel → spin project
    println("  Fourier transforming...")
    Qf = to_fourier(Q_EH; covd_names=Set([:∇g]))
    Qf = simplify(Qf; registry=reg)
    Qf = fix_dummy_positions(Qf)
    K_EH = extract_kernel(Qf, :h; registry=reg)
    println("  Kernel: $(length(K_EH.terms)) bilinear terms")

    println("  Spin projecting...")
    s2_EH  = spin_project(K_EH, :spin2;  registry=reg)
    s1_EH  = spin_project(K_EH, :spin1;  registry=reg)
    s0s_EH = spin_project(K_EH, :spin0s; registry=reg)
    s0w_EH = spin_project(K_EH, :spin0w; registry=reg)
    println("  ✓ All 4 sectors projected")

    # At Λ→0, should recover FP: Tr(K·P²)=(5/2)k², Tr(K·P¹)=0, Tr(K·P⁰ˢ)=-k², Tr(K·P⁰ʷ)=0
    Λ_flat = 1e-12
    k2 = 1.7
    v2  = _eval_spin_scalar(_subst_lambda(s2_EH,  Λ_flat), k2)
    v1  = _eval_spin_scalar(_subst_lambda(s1_EH,  Λ_flat), k2)
    v0s = _eval_spin_scalar(_subst_lambda(s0s_EH, Λ_flat), k2)
    v0w = _eval_spin_scalar(_subst_lambda(s0w_EH, Λ_flat), k2)

    println("\n  Λ→0 limit (k²=$k2):")
    println("    Tr(K·P²)  = $(round(v2; digits=6)),  expected $(round(2.5*k2; digits=6))")
    println("    Tr(K·P¹)  = $(round(v1; digits=6)),  expected 0")
    println("    Tr(K·P⁰ˢ) = $(round(v0s; digits=6)), expected $(round(-k2; digits=6))")
    println("    Tr(K·P⁰ʷ) = $(round(v0w; digits=6)), expected 0")

    fp_pass = abs(v2 - 2.5*k2) < 0.01 && abs(v1) < 0.01 &&
              abs(v0s - (-k2)) < 0.01 && abs(v0w) < 0.01
    println("  FP kernel: $(fp_pass ? "✓ PASS" : "✗ FAIL")")
    if !fp_pass
        println("    Ratios: spin-2/expected=$(round(v2/(2.5*k2); digits=4)), spin-0s/expected=$(round(v0s/(-k2); digits=4))")
    end

    # Gauge check at finite Λ
    Λ_test = 0.3
    v1_dS  = _eval_spin_scalar(_subst_lambda(s1_EH,  Λ_test), k2)
    v0w_dS = _eval_spin_scalar(_subst_lambda(s0w_EH, Λ_test), k2)
    println("\n  Gauge sectors at Λ=$Λ_test:")
    println("    Tr(K·P¹)  = $(round(v1_dS; digits=6))  (should be 0)")
    println("    Tr(K·P⁰ʷ) = $(round(v0w_dS; digits=6))  (should be 0)")

    # ═══════════════════════════════════════════════════════════════════
    # Step 2: R³ (if EH passes, extend to cubic)
    # ═══════════════════════════════════════════════════════════════════

    if fp_pass
        println("\n── Step 2: Adding R³ cubic ──")
        R3_expr = R_expr * R_expr * R_expr

        println("  Computing δ(R³) (order 1)...")
        δR3 = simplify(expand_perturbation(R3_expr, mp, 1); registry=reg, commute_covds_name=:∇g, maxiter=200)
        n_δR3 = δR3 isa TSum ? length(δR3.terms) : 1
        println("  δ(R³): $n_δR3 terms")

        println("  Computing δ²(R³) (order 2)...")
        t0 = time()
        δ2R3 = simplify(expand_perturbation(R3_expr, mp, 2); registry=reg, commute_covds_name=:∇g, maxiter=200)
        n_δ2R3 = δ2R3 isa TSum ? length(δ2R3.terms) : 1
        dt = round(time() - t0; digits=1)
        println("  δ²(R³): $n_δ2R3 terms ($(dt)s)")

        # R³|_{MSS} = (4Λ)³ = 64Λ³
        L0_R3 = tproduct(64 // 1, TensorExpr[Λ_tensor, Λ_tensor, Λ_tensor])

        corr_R3 = sqrt_g_correction(L0_R3, δR3, :g)
        Q_R3 = δ2R3 + corr_R3
        Q_R3 = simplify(Q_R3; registry=reg, commute_covds_name=:∇g, maxiter=200)
        n_QR3 = Q_R3 isa TSum ? length(Q_R3.terms) : 1
        println("  Q_R3: $n_QR3 terms")

        Qf3 = to_fourier(Q_R3; covd_names=Set([:∇g]))
        Qf3 = simplify(Qf3; registry=reg)
        Qf3 = fix_dummy_positions(Qf3)
        K_R3 = extract_kernel(Qf3, :h; registry=reg)
        println("  R³ Kernel: $(length(K_R3.terms)) bilinear terms")

        s2_R3  = spin_project(K_R3, :spin2;  registry=reg)
        s0s_R3 = spin_project(K_R3, :spin0s; registry=reg)
        println("  ✓ R³ spin-2, spin-0s projected")

        # BC prediction for κR - 2κΛ + γ₁R³
        κ = 1.0; γ₁ = 0.01
        bc_pred = dS_spectrum_6deriv(κ=κ, γ₁=γ₁, Λ=Λ_test)
        println("\n  BC predictions (κ=$κ, γ₁=$γ₁, Λ=$Λ_test):")
        println("    m²_g = $(bc_pred.m2_graviton)")
        println("    m²_s = $(round(bc_pred.m2_scalar; digits=6))")

        # Evaluate combined form factors
        k2_vals = [0.5, 1.0, 2.0, 4.0]
        println("\n  Combined f₀(k²) = κ·f₀_EH + γ₁·f₀_R³:")
        f0_vals = Float64[]
        for k2v in k2_vals
            eh  = _eval_spin_scalar(_subst_lambda(s0s_EH, Λ_test), k2v)
            r3  = _eval_spin_scalar(_subst_lambda(s0s_R3, Λ_test), k2v)
            total = κ * eh + γ₁ * r3
            push!(f0_vals, total)
            println("    k²=$(rpad(k2v, 4)) : $(round(total; digits=6))")
        end

        # Linear fit: f₀(k²) = A·k² + B → mass pole at k² = -B/A
        A = (f0_vals[2] - f0_vals[1]) / (k2_vals[2] - k2_vals[1])
        B = f0_vals[1] - A * k2_vals[1]
        m2_s_meas = -B / A
        println("\n  Spin-0 mass from linear fit:")
        println("    f₀ ≈ $(round(A; digits=4))·k² + $(round(B; digits=4))")
        println("    m²_s (measured)  = $(round(m2_s_meas; digits=6))")
        println("    m²_s (predicted) = $(round(bc_pred.m2_scalar; digits=6))")
        if isfinite(bc_pred.m2_scalar)
            rel = abs(m2_s_meas - bc_pred.m2_scalar) / abs(bc_pred.m2_scalar)
            println("    $(rel < 0.05 ? "✓" : "✗") Relative error: $(round(100*rel; digits=1))%")
        end
    else
        println("\n  Skipping Step 2 (R³): EH sanity check failed, debug needed")
        println("  See HANDOFF-next-session.md for diagnosis notes")
    end

    println("\n" * "=" ^ 70)
    println("  DONE")
    println("=" ^ 70)
end
