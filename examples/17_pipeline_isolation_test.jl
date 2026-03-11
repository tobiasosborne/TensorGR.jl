#= Pipeline isolation test:
# Build FP Lagrangian in POSITION space (with ∂ derivatives),
# run through to_fourier + extract_kernel + spin_project,
# and compare with the direct FP kernel builder.
#
# If this matches → pipeline is fine, perturbation engine is wrong.
# If this doesn't match → pipeline has a bug.
=#

using TensorGR

println("=" ^ 70)
println("  Pipeline Isolation: Position-space FP → Fourier → spin project")
println("=" ^ 70)

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    @define_tensor k on=M4 rank=(0,1)

    k2 = 1.7

    # ─── Build FP Lagrangian in position space ───
    # L_FP = ½(∂_c h_{ab})(∂^c h^{ab}) - (∂_b h^{ab})(∂_c h^c_a)
    #      + (∂_a h)(∂_b h^{ab}) - ½(∂_a h)(∂^a h)
    # where h = g^{cd}h_{cd} = trace

    # Term 1: ½ (∂_c h_{ab})(∂^c h^{ab})
    t1 = tproduct(1//2, TensorExpr[
        TDeriv(down(:c), Tensor(:h, [down(:a), down(:b)])),
        TDeriv(up(:c), Tensor(:h, [up(:a), up(:b)]))])

    # Term 2: -(∂_b h^{ab})(∂_c h^c_a)
    t2 = tproduct(-1//1, TensorExpr[
        TDeriv(down(:b), Tensor(:h, [up(:a), up(:b)])),
        TDeriv(down(:c), Tensor(:h, [up(:c), down(:a)]))])

    # Term 3: +(∂_a h)(∂_b h^{ab})
    # h = g^{cd}h_{cd}, so ∂_a h = ∂_a(g^{cd}h_{cd})
    # On flat background, g is constant, so ∂_a(g^{cd}h_{cd}) = g^{cd}∂_a h_{cd}
    # But in the AST, we can write this as TDeriv on the product, or just use the
    # expanded form: TDeriv(∂_a, h^c_c) = ∂_a(h^c_c) where h^c_c = h with trace indices
    # Actually, let's use g^{cd} ∂_a h_{cd} directly
    t3 = tproduct(1//1, TensorExpr[
        Tensor(:g, [up(:c), up(:d)]),
        TDeriv(down(:a), Tensor(:h, [down(:c), down(:d)])),
        TDeriv(down(:b), Tensor(:h, [up(:a), up(:b)]))])

    # Term 4: -½(∂_a h)(∂^a h) = -½ g^{cd}(∂_a h_{cd}) g^{ef}(∂^a h_{ef})
    t4 = tproduct(-1//2, TensorExpr[
        Tensor(:g, [up(:c), up(:d)]),
        TDeriv(down(:a), Tensor(:h, [down(:c), down(:d)])),
        Tensor(:g, [up(:e), up(:f)]),
        TDeriv(up(:a), Tensor(:h, [down(:e), down(:f)]))])

    L_FP_pos = t1 + t2 + t3 + t4
    L_FP_pos = simplify(L_FP_pos; registry=reg)
    n_pos = L_FP_pos isa TSum ? length(L_FP_pos.terms) : 1
    println("\n  L_FP (position space, simplified): $n_pos terms")

    # ─── Fourier transform ───
    L_FP_k = to_fourier(L_FP_pos)
    L_FP_k = simplify(L_FP_k; registry=reg)
    L_FP_k = fix_dummy_positions(L_FP_k)
    K_pos = extract_kernel(L_FP_k, :h; registry=reg)
    println("  Kernel (from position space): $(length(K_pos.terms)) bilinear terms")

    s2_pos  = spin_project(K_pos, :spin2;  registry=reg)
    s1_pos  = spin_project(K_pos, :spin1;  registry=reg)
    s0s_pos = spin_project(K_pos, :spin0s; registry=reg)
    s0w_pos = spin_project(K_pos, :spin0w; registry=reg)

    v2_pos  = _eval_spin_scalar(s2_pos,  k2)
    v1_pos  = _eval_spin_scalar(s1_pos,  k2)
    v0s_pos = _eval_spin_scalar(s0s_pos, k2)
    v0w_pos = _eval_spin_scalar(s0w_pos, k2)

    println("  Position→Fourier: spin-2=$(round(v2_pos;digits=4)), spin-1=$(round(v1_pos;digits=4)), spin-0s=$(round(v0s_pos;digits=4)), spin-0w=$(round(v0w_pos;digits=4))")

    # ─── Direct FP kernel ───
    K_FP = build_FP_momentum_kernel(reg)
    s2_D  = spin_project(K_FP, :spin2;  registry=reg)
    s1_D  = spin_project(K_FP, :spin1;  registry=reg)
    s0s_D = spin_project(K_FP, :spin0s; registry=reg)
    s0w_D = spin_project(K_FP, :spin0w; registry=reg)

    v2_D  = _eval_spin_scalar(s2_D,  k2)
    v1_D  = _eval_spin_scalar(s1_D,  k2)
    v0s_D = _eval_spin_scalar(s0s_D, k2)
    v0w_D = _eval_spin_scalar(s0w_D, k2)

    println("  Direct FP:        spin-2=$(round(v2_D;digits=4)), spin-1=$(round(v1_D;digits=4)), spin-0s=$(round(v0s_D;digits=4)), spin-0w=$(round(v0w_D;digits=4))")

    match = abs(v2_pos - v2_D) < 0.01 && abs(v1_pos - v1_D) < 0.01 &&
            abs(v0s_pos - v0s_D) < 0.01 && abs(v0w_pos - v0w_D) < 0.01
    println("\n  Pipeline check: $(match ? "✓ MATCH — pipeline is correct" : "✗ MISMATCH — pipeline has a bug")")

    if !match
        println("\n  Printing both kernels for comparison:")
        println("  --- Position-space FP kernel ---")
        for (i, bt) in enumerate(K_pos.terms)
            println("    $i: coeff=$(bt.coeff), left=$(bt.left), right=$(bt.right)")
        end
        println("  --- Direct FP kernel ---")
        for (i, bt) in enumerate(K_FP.terms)
            println("    $i: coeff=$(bt.coeff), left=$(bt.left), right=$(bt.right)")
        end
    end

    # ─── Also test: print the perturbation engine δ²R terms ───
    println("\n── Perturbation engine δ²R (first few terms, position space) ──")
    mp = define_metric_perturbation!(reg, :g, :h)
    set_vanishing!(reg, :Riem)
    set_vanishing!(reg, :Ric)
    set_vanishing!(reg, :RicScalar)
    δ2R = simplify(δricci_scalar(mp, 2); registry=reg)
    if δ2R isa TSum
        for (i, t) in enumerate(δ2R.terms[1:min(5, length(δ2R.terms))])
            println("  term $i: $t")
        end
        println("  ... ($(length(δ2R.terms)) terms total)")
    else
        println("  $δ2R")
    end

    println("\n" * "=" ^ 70)
end
