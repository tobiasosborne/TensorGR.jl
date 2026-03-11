#= Priority 1+2 diagnostic: verify to_fourier on nested TDeriv,
   and build analytic Y = η^{ab}δ²Ric_{ab} for comparison.

   Session 18: Following HANDOFF priorities from session 17.
=#

using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    @define_tensor k on=M4 rank=(0,1)

    mp = define_metric_perturbation!(reg, :g, :h)
    set_vanishing!(reg, :Riem)
    set_vanishing!(reg, :Ric)
    set_vanishing!(reg, :RicScalar)

    # ═══════════════════════════════════════════════════════════════
    # Priority 1: Verify to_fourier on nested TDeriv
    # ═══════════════════════════════════════════════════════════════

    println("═══ Priority 1: to_fourier on nested TDeriv ═══\n")

    # Get raw δ²R (no simplification yet — see what the engine gives us)
    δ2R_raw = δricci_scalar(mp, 2)
    println("δ²R raw: $(δ2R_raw isa TSum ? length(δ2R_raw.terms) : 1) terms")

    # Check which terms have nested TDeriv(∂, TSum)
    function has_nested_deriv_sum(e::TensorExpr)
        e isa TDeriv && e.arg isa TSum && return true
        e isa TDeriv && return has_nested_deriv_sum(e.arg)
        e isa TProduct && return any(has_nested_deriv_sum, e.factors)
        e isa TSum && return any(has_nested_deriv_sum, e.terms)
        return false
    end

    if δ2R_raw isa TSum
        for (i, t) in enumerate(δ2R_raw.terms)
            if has_nested_deriv_sum(t)
                println("  term $i has nested TDeriv(∂, TSum): ", string(t)[1:min(end,100)])
            end
        end
    end

    # Method A: Fourier transform of the raw expression (as pipeline does it)
    δ2R_fA = to_fourier(δ2R_raw)
    δ2R_fA = simplify(δ2R_fA; registry=reg)
    δ2R_fA = fix_dummy_positions(δ2R_fA)

    # Method B: expand_products first (to distribute TDeriv over TSum),
    # then Fourier transform
    δ2R_expanded = TensorGR.expand_products(δ2R_raw)
    δ2R_fB = to_fourier(δ2R_expanded)
    δ2R_fB = simplify(δ2R_fB; registry=reg)
    δ2R_fB = fix_dummy_positions(δ2R_fB)

    # Method C: simplify first, then Fourier
    δ2R_simpl = simplify(δ2R_raw; registry=reg)
    δ2R_fC = to_fourier(δ2R_simpl)
    δ2R_fC = simplify(δ2R_fC; registry=reg)
    δ2R_fC = fix_dummy_positions(δ2R_fC)

    println("\nMethod A (Fourier raw):        $(δ2R_fA isa TSum ? length(δ2R_fA.terms) : 1) Fourier terms")
    println("Method B (expand → Fourier):   $(δ2R_fB isa TSum ? length(δ2R_fB.terms) : 1) Fourier terms")
    println("Method C (simplify → Fourier): $(δ2R_fC isa TSum ? length(δ2R_fC.terms) : 1) Fourier terms")

    # Extract kernels and compare spin projections
    K_A = extract_kernel(δ2R_fA, :h; registry=reg)
    K_B = extract_kernel(δ2R_fB, :h; registry=reg)
    K_C = extract_kernel(δ2R_fC, :h; registry=reg)
    K_FP = build_FP_momentum_kernel(reg)

    k2 = 1.7
    function spin_all(K)
        s2  = _eval_spin_scalar(spin_project(K, :spin2;  registry=reg), k2)
        s1  = _eval_spin_scalar(spin_project(K, :spin1;  registry=reg), k2)
        s0s = _eval_spin_scalar(spin_project(K, :spin0s; registry=reg), k2)
        s0w = _eval_spin_scalar(spin_project(K, :spin0w; registry=reg), k2)
        (s2, s1, s0s, s0w)
    end

    function show_spin(label, sp)
        println("  $label: spin2=$(round(sp[1];digits=4)), spin1=$(round(sp[2];digits=4)), " *
                "spin0s=$(round(sp[3];digits=4)), spin0w=$(round(sp[4];digits=4))")
    end

    sp_A  = spin_all(K_A)
    sp_B  = spin_all(K_B)
    sp_C  = spin_all(K_C)
    sp_FP = spin_all(K_FP)

    println("\nSpin projections of δ²R (at k²=$k2):")
    show_spin("Method A (Fourier raw)      ", sp_A)
    show_spin("Method B (expand → Fourier) ", sp_B)
    show_spin("Method C (simplify → Fourier)", sp_C)
    show_spin("FP reference                ", sp_FP)

    # Check: are methods A/B/C identical?
    A_eq_B = all(isapprox(a, b; atol=1e-10) for (a,b) in zip(sp_A, sp_B))
    A_eq_C = all(isapprox(a, b; atol=1e-10) for (a,b) in zip(sp_A, sp_C))
    println("\nMethods A=B? $A_eq_B    A=C? $A_eq_C")
    if A_eq_B && A_eq_C
        println("✓ to_fourier gives same result regardless of pre-expansion → NOT the bug")
    else
        println("✗ to_fourier DIFFERS depending on expansion order → THIS IS THE BUG")
    end

    # ═══════════════════════════════════════════════════════════════
    # Priority 2: Build analytic Y = η^{ab}δ²Ric_{ab} from δΓ·δΓ
    # ═══════════════════════════════════════════════════════════════

    println("\n═══ Priority 2: Analytic Y = η^{ab}δ²Ric_{ab} (δΓ·δΓ terms) ═══\n")

    # In Fourier space (flat background), the first-order Christoffel perturbation is:
    #   δΓ^c_{ab} = ½η^{cd}(k_a h_{bd} + k_b h_{ad} - k_d h_{ab})
    #
    # The second-order Ricci tensor (quadratic part only, from Γ·Γ) is:
    #   δ²Ric_{ab}|_{ΓΓ} = δΓ^c_{ad} δΓ^d_{bc} - δΓ^c_{ab} δΓ^d_{cd}
    #
    # Y = η^{ab} δ²Ric_{ab}|_{ΓΓ} = η^{ab}(δΓ^c_{ad} δΓ^d_{bc} - δΓ^c_{ab} δΓ^d_{cd})
    #
    # We build this in momentum space as explicit tensor products.

    # Helper: build δΓ^c_{ab} in Fourier space with specific indices
    # δΓ^c_{ab} = ½(k_a h^c_b + k_b h^c_a - k^c h_{ab})
    # where h^c_b = η^{cd}h_{db}
    function fourier_δΓ(c_up::Symbol, a_dn::Symbol, b_dn::Symbol;
                         dummy::Symbol=:_none)
        # Term 1: ½ k_a h^c_b = ½ k_a η^{cd} h_{db}
        # Term 2: ½ k_b h^c_a = ½ k_b η^{cd} h_{da}
        # Term 3: -½ k^c h_{ab} = -½ η^{ce} k_e h_{ab}
        # We use 'd' as the contraction dummy in terms 1,2 and 'e' in term 3
        # Caller must provide disjoint dummy names

        # For simplicity, we'll use mixed-index h with metric contractions explicit
        # Actually let's keep it clean: raise c with the metric explicitly

        # δΓ^c_{ab} = ½η^{cd}(k_a h_{db} + k_b h_{da} - k_d h_{ab})
        d = dummy
        t1 = tproduct(1//2, TensorExpr[
            Tensor(:g, [up(c_up), up(d)]),
            Tensor(:k, [down(a_dn)]),
            Tensor(:h, [down(d), down(b_dn)])])
        t2 = tproduct(1//2, TensorExpr[
            Tensor(:g, [up(c_up), up(d)]),
            Tensor(:k, [down(b_dn)]),
            Tensor(:h, [down(d), down(a_dn)])])
        t3 = tproduct(-1//2, TensorExpr[
            Tensor(:g, [up(c_up), up(d)]),
            Tensor(:k, [down(d)]),
            Tensor(:h, [down(a_dn), down(b_dn)])])
        tsum(TensorExpr[t1, t2, t3])
    end

    # Y = η^{ab}(δΓ^c_{ad} δΓ^d_{bc} - δΓ^c_{ab} δΓ^d_{cd})
    # We need to pick concrete dummy names carefully:
    # Free indices: none (Y is a scalar bilinear in h)
    # Dummies: a,b (from η^{ab} trace), c,d (from Ricci contraction), plus internal dummies

    # Term 1: η^{ab} δΓ^c_{ad} δΓ^d_{bc}
    # δΓ^c_{ad} with dummy e: ½g^{ce}(k_a h_{ed} + k_d h_{ea} - k_e h_{ad})
    # δΓ^d_{bc} with dummy f: ½g^{df}(k_b h_{fc} + k_c h_{fb} - k_f h_{bc})
    # Full: g^{ab} × [½g^{ce}(k_a h_{ed} + k_d h_{ea} - k_e h_{ad})] × [½g^{df}(k_b h_{fc} + k_c h_{fb} - k_f h_{bc})]

    # This is 3×3 = 9 terms for Term 1, and 9 for Term 2.
    # Better to let TensorGR handle the algebra. Build as products, then extract kernel.

    Γ1_cad = fourier_δΓ(:c, :a, :d; dummy=:e)
    Γ1_dbc = fourier_δΓ(:d, :b, :c; dummy=:f)

    Γ2_cab = fourier_δΓ(:c, :a, :b; dummy=:e)
    Γ2_dcd = fourier_δΓ(:d, :c, :d; dummy=:f)

    # Ensure no dummy clash between the two Γ factors
    Γ1_dbc_safe = ensure_no_dummy_clash(Γ1_cad, Γ1_dbc)
    Γ2_dcd_safe = ensure_no_dummy_clash(Γ2_cab, Γ2_dcd)

    # η^{ab} δΓ^c_{ad} δΓ^d_{bc}
    Y1 = Tensor(:g, [up(:a), up(:b)]) * Γ1_cad * Γ1_dbc_safe

    # -η^{ab} δΓ^c_{ab} δΓ^d_{cd}
    Y2 = tproduct(-1//1, TensorExpr[Tensor(:g, [up(:a), up(:b)]), Γ2_cab, Γ2_dcd_safe])

    Y_analytic = Y1 + Y2
    Y_analytic = simplify(Y_analytic; registry=reg)
    Y_analytic = fix_dummy_positions(Y_analytic)

    println("Y analytic: $(Y_analytic isa TSum ? length(Y_analytic.terms) : 1) terms after simplify")

    K_Y = extract_kernel(Y_analytic, :h; registry=reg)
    sp_Y = spin_all(K_Y)
    println("\nAnalytic Y spin projections:")
    show_spin("Y analytic (δΓ·δΓ)         ", sp_Y)

    # Engine Y = engine δ²R - analytic X
    # X = -h^{ab}δRic_{ab} (from session 17, example 20)
    # Build analytic X again:
    X1 = tproduct(1//1, TensorExpr[
        Tensor(:k, [down(:c)]), Tensor(:k, [down(:a)]),
        Tensor(:h, [up(:a), up(:b)]), Tensor(:h, [up(:c), down(:b)])])
    X2 = tproduct(-1//2, TensorExpr[
        TScalar(:k²),
        Tensor(:h, [up(:a), up(:b)]), Tensor(:h, [down(:a), down(:b)])])
    X3 = tproduct(-1//2, TensorExpr[
        Tensor(:k, [down(:a)]), Tensor(:k, [down(:b)]),
        Tensor(:h, [up(:a), up(:b)]),
        Tensor(:g, [up(:c), up(:d)]), Tensor(:h, [down(:c), down(:d)])])
    X = X1 + X2 + X3
    K_X = extract_kernel(X, :h; registry=reg)
    sp_X = spin_all(K_X)

    # Engine δ²R
    sp_eng = sp_C  # Method C (simplify → Fourier), same as session 17
    eng_Y = (sp_eng[1] - sp_X[1], sp_eng[2] - sp_X[2],
             sp_eng[3] - sp_X[3], sp_eng[4] - sp_X[4])

    println("\nComparison: Y from engine vs Y analytic:")
    show_spin("Y engine (δ²R - X)         ", eng_Y)
    show_spin("Y analytic (δΓ·δΓ)         ", sp_Y)
    Y_match = all(isapprox(a, b; atol=1e-10) for (a,b) in zip(eng_Y, sp_Y))
    println("Y engine = Y analytic? $Y_match")

    # ═══════════════════════════════════════════════════════════════
    # Additional check: does δ²R = X + Y at the kernel level?
    # ═══════════════════════════════════════════════════════════════
    println("\n═══ Consistency: δ²R =? X + Y ═══\n")
    sp_XY = (sp_X[1] + sp_Y[1], sp_X[2] + sp_Y[2],
             sp_X[3] + sp_Y[3], sp_X[4] + sp_Y[4])
    show_spin("X + Y analytic             ", sp_XY)
    show_spin("δ²R engine (method C)      ", sp_C)
    XY_match = all(isapprox(a, b; atol=1e-10) for (a,b) in zip(sp_XY, sp_C))
    println("X + Y = δ²R? $XY_match")

    # ═══════════════════════════════════════════════════════════════
    # Final comparison with FP
    # ═══════════════════════════════════════════════════════════════
    println("\n═══ Final: what should δ²R be for EH match? ═══\n")
    # For L_EH = ½√g R, δ²L_EH = ½(δ²R + ½h·δR - ...) on flat
    # The FP kernel IS the correct quadratic Lagrangian.
    # δ²R should relate to FP via: FP = δ²R + ½h·δR + total derivs (+ √g terms)
    # But ½h·δR on flat:
    # δR = k_ak_b h^{ab} - k²h  (linearized Ricci scalar)
    # ½h·δR = ½ h (k_ak_b h^{ab} - k²h) where h = η^{ab}h_{ab}
    δR1 = tproduct(1//2, TensorExpr[
        Tensor(:g, [up(:c), up(:d)]), Tensor(:h, [down(:c), down(:d)]),
        Tensor(:k, [down(:a)]), Tensor(:k, [down(:b)]),
        Tensor(:h, [up(:a), up(:b)])])
    δR2 = tproduct(-1//2, TensorExpr[
        Tensor(:g, [up(:c), up(:d)]), Tensor(:h, [down(:c), down(:d)]),
        TScalar(:k²),
        Tensor(:g, [up(:e), up(:f)]), Tensor(:h, [down(:e), down(:f)])])
    hdR = δR1 + δR2
    hdR = simplify(hdR; registry=reg)
    hdR = fix_dummy_positions(hdR)

    K_hdR = extract_kernel(hdR, :h; registry=reg)
    sp_hdR = spin_all(K_hdR)

    # L₂ = δ²R + ½h·δR
    sp_L2 = (sp_C[1] + sp_hdR[1], sp_C[2] + sp_hdR[2],
             sp_C[3] + sp_hdR[3], sp_C[4] + sp_hdR[4])

    show_spin("FP reference               ", sp_FP)
    show_spin("δ²R engine                 ", sp_C)
    show_spin("½h·δR                      ", sp_hdR)
    show_spin("L₂ = δ²R + ½h·δR          ", sp_L2)
    show_spin("L₂ - FP                    ", (sp_L2[1]-sp_FP[1], sp_L2[2]-sp_FP[2],
                                               sp_L2[3]-sp_FP[3], sp_L2[4]-sp_FP[4]))
    println("\n  If L₂ = FP + total_derivs, then L₂ - FP should spin-project to 0.")
    println("  (Total derivs ↔ terms proportional to k² that vanish on-shell)")
end
