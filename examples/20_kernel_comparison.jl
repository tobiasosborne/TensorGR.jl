#= Direct kernel comparison: perturbation engine vs FP =#

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

    # ─── Perturbation engine path (NO expand_derivatives) ───
    δ2R = simplify(δricci_scalar(mp, 2); registry=reg)

    # Print the raw terms BEFORE Fourier
    println("── δ²R position space ($(δ2R isa TSum ? length(δ2R.terms) : 1) terms) ──")
    if δ2R isa TSum
        for (i, t) in enumerate(δ2R.terms)
            s = string(t)
            println("  $i: $(length(s) > 120 ? s[1:120]*"..." : s)")
        end
    end

    # Fourier
    δ2R_f = to_fourier(δ2R)
    δ2R_f = simplify(δ2R_f; registry=reg)
    δ2R_f = fix_dummy_positions(δ2R_f)

    println("\n── δ²R Fourier space ($(δ2R_f isa TSum ? length(δ2R_f.terms) : 1) terms) ──")
    if δ2R_f isa TSum
        for (i, t) in enumerate(δ2R_f.terms)
            s = string(t)
            println("  $i: $(length(s) > 120 ? s[1:120]*"..." : s)")
        end
    end

    K_pert = extract_kernel(δ2R_f, :h; registry=reg)
    println("\n── δ²R KineticKernel ($(length(K_pert.terms)) terms) ──")
    for (i, bt) in enumerate(K_pert.terms)
        println("  $i: coeff=$(bt.coeff), L=$(bt.left), R=$(bt.right)")
    end

    # ─── FP kernel ───
    K_FP = build_FP_momentum_kernel(reg)
    println("\n── FP KineticKernel ($(length(K_FP.terms)) terms) ──")
    for (i, bt) in enumerate(K_FP.terms)
        println("  $i: coeff=$(bt.coeff), L=$(bt.left), R=$(bt.right)")
    end

    # ─── Build X = -h^{ab}δRic_{ab} analytically in Fourier space ───
    # δRic_{ab} = -½(k_ck_a h^c_b + k_ck_b h^c_a - k²h_{ab} - k_ak_b h)
    # (sign from Fourier convention: each ∂ → k, (ik)² = -k² → we drop i, so ∂²h → k²h)
    # X = -h^{ab}·δRic_{ab}
    #   = h^{ab}·½(k_ck_a h^c_b + k_ck_b h^c_a - k²h_{ab} - k_ak_b h)
    #   = k_ck_a h^{ab}h^c_b - ½k²h^{ab}h_{ab} - ½k_ak_b h^{ab}h
    println("\n── Analytic X = -h^{ab}δRic_{ab} in Fourier ──")
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
    println("  K_X: $(length(K_X.terms)) terms")

    k2 = 1.7
    s2_X  = _eval_spin_scalar(spin_project(K_X, :spin2;  registry=reg), k2)
    s1_X  = _eval_spin_scalar(spin_project(K_X, :spin1;  registry=reg), k2)
    s0s_X = _eval_spin_scalar(spin_project(K_X, :spin0s; registry=reg), k2)
    s0w_X = _eval_spin_scalar(spin_project(K_X, :spin0w; registry=reg), k2)
    println("  spin-2=$(round(s2_X;digits=4)), spin-1=$(round(s1_X;digits=4)), spin-0s=$(round(s0s_X;digits=4)), spin-0w=$(round(s0w_X;digits=4))")

    # Pert engine should give δ²R = X + η^{ab}δ²Ric_{ab}
    # The Y = η^{ab}δ²Ric_{ab} part involves δΓ·δΓ terms
    # Check: do the pert engine spin projections match X?
    s2_P  = _eval_spin_scalar(spin_project(K_pert, :spin2;  registry=reg), k2)
    s1_P  = _eval_spin_scalar(spin_project(K_pert, :spin1;  registry=reg), k2)
    s0s_P = _eval_spin_scalar(spin_project(K_pert, :spin0s; registry=reg), k2)
    s0w_P = _eval_spin_scalar(spin_project(K_pert, :spin0w; registry=reg), k2)

    println("\n── Comparison ──")
    println("  Pert engine δ²R: spin-2=$(round(s2_P;digits=4)), spin-1=$(round(s1_P;digits=4)), spin-0s=$(round(s0s_P;digits=4))")
    println("  Analytic X only: spin-2=$(round(s2_X;digits=4)), spin-1=$(round(s1_X;digits=4)), spin-0s=$(round(s0s_X;digits=4))")
    println("  Y = δ²R - X:     spin-2=$(round(s2_P-s2_X;digits=4)), spin-1=$(round(s1_P-s1_X;digits=4)), spin-0s=$(round(s0s_P-s0s_X;digits=4))")

    # FP reference
    s2_FP = _eval_spin_scalar(spin_project(K_FP, :spin2; registry=reg), k2)
    s0s_FP = _eval_spin_scalar(spin_project(K_FP, :spin0s; registry=reg), k2)
    println("  FP reference:    spin-2=$(round(s2_FP;digits=4)), spin-0s=$(round(s0s_FP;digits=4))")

    # ½h·δR contribution
    println("\n  Expected: δ²R + ½h·δR = FP + total_deriv")
    println("  ½h·δR spin-2 = 0 (pure scalar term, no spin-2)")
    println("  So δ²R spin-2 should = FP spin-2 + total_deriv spin-2")
    println("  If total_derivs spin-project to zero: δ²R spin-2 should = FP spin-2 = $(round(s2_FP;digits=4))")
    println("  Actual δ²R spin-2 = $(round(s2_P;digits=4))")
    println("  Discrepancy = $(round(s2_P - s2_FP;digits=4))")
end
