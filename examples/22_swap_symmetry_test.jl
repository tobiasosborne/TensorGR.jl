#= Test: spin_project must be symmetric under left↔right h swap.
# For symmetric projectors P^{μν,ρσ} = P^{ρσ,μν}, the result of
# Tr(K·P) must be independent of which h is "left" and which is "right".
=#

using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    @define_tensor k on=M4 rank=(0,1)

    k2 = 1.7

    # ─── Test 1: k_a k_b h^{ab} × h_trace ───
    # Build with h^{ab} first (= "left")
    expr_A = tproduct(1//1, TensorExpr[
        Tensor(:k, [down(:a)]), Tensor(:k, [down(:b)]),
        Tensor(:h, [up(:a), up(:b)]),
        Tensor(:g, [up(:c), up(:d)]), Tensor(:h, [down(:c), down(:d)])])

    # Build with h_{cd} first (= "left"), h^{ab} second (= "right")
    expr_B = tproduct(1//1, TensorExpr[
        Tensor(:k, [down(:a)]), Tensor(:k, [down(:b)]),
        Tensor(:g, [up(:c), up(:d)]), Tensor(:h, [down(:c), down(:d)]),
        Tensor(:h, [up(:a), up(:b)])])

    K_A = extract_kernel(expr_A, :h; registry=reg)
    K_B = extract_kernel(expr_B, :h; registry=reg)

    println("── Test 1: k_a k_b h^{ab} h_trace ──")
    println("  K_A: left=$(K_A.terms[1].left), right=$(K_A.terms[1].right)")
    println("  K_B: left=$(K_B.terms[1].left), right=$(K_B.terms[1].right)")

    for spin in [:spin2, :spin1, :spin0s, :spin0w]
        vA = _eval_spin_scalar(spin_project(K_A, spin; registry=reg), k2)
        vB = _eval_spin_scalar(spin_project(K_B, spin; registry=reg), k2)
        status = abs(vA - vB) < 1e-10 ? "✓" : "✗ BUG"
        println("  $spin: left-first=$(round(vA;digits=6)), right-first=$(round(vB;digits=6)) $status")
    end

    # ─── Test 2: k² h_{ab} h^{ab} ───
    println("\n── Test 2: k² h_{ab} h^{ab} ──")
    expr_C = tproduct(1//1, TensorExpr[
        TScalar(:k²),
        Tensor(:h, [down(:a), down(:b)]),
        Tensor(:h, [up(:a), up(:b)])])

    expr_D = tproduct(1//1, TensorExpr[
        TScalar(:k²),
        Tensor(:h, [up(:a), up(:b)]),
        Tensor(:h, [down(:a), down(:b)])])

    K_C = extract_kernel(expr_C, :h; registry=reg)
    K_D = extract_kernel(expr_D, :h; registry=reg)

    println("  K_C: left=$(K_C.terms[1].left), right=$(K_C.terms[1].right)")
    println("  K_D: left=$(K_D.terms[1].left), right=$(K_D.terms[1].right)")

    for spin in [:spin2, :spin1, :spin0s, :spin0w]
        vC = _eval_spin_scalar(spin_project(K_C, spin; registry=reg), k2)
        vD = _eval_spin_scalar(spin_project(K_D, spin; registry=reg), k2)
        status = abs(vC - vD) < 1e-10 ? "✓" : "✗ BUG"
        println("  $spin: C=$(round(vC;digits=6)), D=$(round(vD;digits=6)) $status")
    end

    # ─── Test 3: k_b k_c h^{ab} h^c_a  (FP term 2) ───
    println("\n── Test 3: k_b k_c h^{ab} h^c_a ──")
    expr_E = tproduct(-1//1, TensorExpr[
        Tensor(:k, [down(:b)]), Tensor(:k, [down(:c)]),
        Tensor(:h, [up(:a), up(:b)]),
        Tensor(:h, [up(:c), down(:a)])])

    expr_F = tproduct(-1//1, TensorExpr[
        Tensor(:k, [down(:b)]), Tensor(:k, [down(:c)]),
        Tensor(:h, [up(:c), down(:a)]),
        Tensor(:h, [up(:a), up(:b)])])

    K_E = extract_kernel(expr_E, :h; registry=reg)
    K_F = extract_kernel(expr_F, :h; registry=reg)

    println("  K_E: left=$(K_E.terms[1].left), right=$(K_E.terms[1].right)")
    println("  K_F: left=$(K_F.terms[1].left), right=$(K_F.terms[1].right)")

    for spin in [:spin2, :spin1, :spin0s, :spin0w]
        vE = _eval_spin_scalar(spin_project(K_E, spin; registry=reg), k2)
        vF = _eval_spin_scalar(spin_project(K_F, spin; registry=reg), k2)
        status = abs(vE - vF) < 1e-10 ? "✓" : "✗ BUG"
        println("  $spin: E=$(round(vE;digits=6)), F=$(round(vF;digits=6)) $status")
    end

    println("\n── Summary ──")
    println("  If any test shows BUG, spin_project is not symmetric under h swap.")
    println("  This means total derivative terms (antisymmetric kernel) do NOT")
    println("  cancel, which is the root cause of the δ²R mismatch.")
end
