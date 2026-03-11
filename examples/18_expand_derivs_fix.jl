#= Fix test: expand_derivatives before to_fourier
#
# The perturbation engine produces terms like:
#   -h × ∂(g·∂h + g·∂h - g·∂h)
# where the outer ∂ wraps a TSum. extract_kernel can't find the second h
# inside TDeriv(∂, TSum(...)). Solution: expand_derivatives first.
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

    k2 = 1.7

    # ─── δ²R: WITHOUT expand_derivatives ───
    println("── Without expand_derivatives ──")
    δ2R = simplify(δricci_scalar(mp, 2); registry=reg)
    δ2R_f = to_fourier(δ2R)
    δ2R_f = simplify(δ2R_f; registry=reg)
    δ2R_f = fix_dummy_positions(δ2R_f)
    K1 = extract_kernel(δ2R_f, :h; registry=reg)
    println("  Kernel: $(length(K1.terms)) bilinear terms (from $(δ2R isa TSum ? length(δ2R.terms) : 1) terms)")

    s2_1 = _eval_spin_scalar(spin_project(K1, :spin2; registry=reg), k2)
    s1_1 = _eval_spin_scalar(spin_project(K1, :spin1; registry=reg), k2)
    s0s_1 = _eval_spin_scalar(spin_project(K1, :spin0s; registry=reg), k2)
    println("  spin-2=$(round(s2_1;digits=4)), spin-1=$(round(s1_1;digits=4)), spin-0s=$(round(s0s_1;digits=4))")

    # ─── δ²R: WITH expand_derivatives ───
    println("\n── With expand_derivatives ──")
    δ2R_exp = expand_derivatives(δ2R)
    δ2R_exp = simplify(δ2R_exp; registry=reg)
    n_exp = δ2R_exp isa TSum ? length(δ2R_exp.terms) : 1
    println("  δ²R expanded: $n_exp terms")

    δ2R_exp_f = to_fourier(δ2R_exp)
    δ2R_exp_f = simplify(δ2R_exp_f; registry=reg)
    δ2R_exp_f = fix_dummy_positions(δ2R_exp_f)
    K2 = extract_kernel(δ2R_exp_f, :h; registry=reg)
    println("  Kernel: $(length(K2.terms)) bilinear terms")

    s2_2 = _eval_spin_scalar(spin_project(K2, :spin2; registry=reg), k2)
    s1_2 = _eval_spin_scalar(spin_project(K2, :spin1; registry=reg), k2)
    s0s_2 = _eval_spin_scalar(spin_project(K2, :spin0s; registry=reg), k2)
    s0w_2 = _eval_spin_scalar(spin_project(K2, :spin0w; registry=reg), k2)
    println("  spin-2=$(round(s2_2;digits=4)), spin-1=$(round(s1_2;digits=4)), spin-0s=$(round(s0s_2;digits=4)), spin-0w=$(round(s0w_2;digits=4))")

    # ─── δ²R + ½h·δR (full L_FP on flat), with expand_derivatives ───
    println("\n── Full L_FP = δ²R + ½h·δR (expanded) ──")
    δR = simplify(expand_perturbation(Tensor(:RicScalar, TIndex[]), mp, 1); registry=reg)
    δR = expand_derivatives(δR)
    δR = simplify(δR; registry=reg)

    h_trace = Tensor(:g, [up(:z1), up(:z2)]) * Tensor(:h, [down(:z1), down(:z2)])
    half_h_δR = tproduct(1//2, TensorExpr[h_trace]) * δR
    half_h_δR = expand_derivatives(half_h_δR)
    half_h_δR = simplify(half_h_δR; registry=reg)

    Q = δ2R_exp + half_h_δR
    Q = simplify(Q; registry=reg)
    nQ = Q isa TSum ? length(Q.terms) : 1
    println("  L_FP: $nQ terms")

    Q_f = to_fourier(Q)
    Q_f = simplify(Q_f; registry=reg)
    Q_f = fix_dummy_positions(Q_f)
    K3 = extract_kernel(Q_f, :h; registry=reg)
    println("  Kernel: $(length(K3.terms)) bilinear terms")

    s2_3 = _eval_spin_scalar(spin_project(K3, :spin2; registry=reg), k2)
    s1_3 = _eval_spin_scalar(spin_project(K3, :spin1; registry=reg), k2)
    s0s_3 = _eval_spin_scalar(spin_project(K3, :spin0s; registry=reg), k2)
    s0w_3 = _eval_spin_scalar(spin_project(K3, :spin0w; registry=reg), k2)
    println("  spin-2=$(round(s2_3;digits=4)), spin-1=$(round(s1_3;digits=4)), spin-0s=$(round(s0s_3;digits=4)), spin-0w=$(round(s0w_3;digits=4))")

    # ─── Reference ───
    println("\n── Reference FP ──")
    K_FP = build_FP_momentum_kernel(reg)
    v2_FP = _eval_spin_scalar(spin_project(K_FP, :spin2; registry=reg), k2)
    v0s_FP = _eval_spin_scalar(spin_project(K_FP, :spin0s; registry=reg), k2)
    println("  spin-2=$(round(v2_FP;digits=4)), spin-0s=$(round(v0s_FP;digits=4))")

    println("\n── Summary ──")
    println("  Without expand_derivs: spin-2=$(round(s2_1;digits=4)) ($(round(s2_1/v2_FP;digits=3))× FP)")
    println("  With expand_derivs:    spin-2=$(round(s2_2;digits=4)) ($(round(s2_2/v2_FP;digits=3))× FP)")
    println("  Full L_FP (expanded):  spin-2=$(round(s2_3;digits=4)) ($(round(s2_3/v2_FP;digits=3))× FP)")

    fp_match = abs(s2_3 - v2_FP) < 0.01 && abs(s0s_3 - v0s_FP) < 0.01 && abs(s1_3) < 0.01
    println("\n  Full L_FP matches FP? $(fp_match ? "✓ YES" : "✗ NO")")

    d2r_match = abs(s2_2 - v2_FP) < 0.01 && abs(s0s_2 - v0s_FP) < 0.01 && abs(s1_2) < 0.01
    println("  δ²R alone matches FP? $(d2r_match ? "✓ YES" : "✗ NO (need √g correction)")")
end
