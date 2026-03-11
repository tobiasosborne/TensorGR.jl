#= Minimal reproducer: does Fourier commute with simplify? =#

using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    @define_tensor k on=M4 rank=(0,1)

    # Build: ∂_e(g^{ef} ∂_d h_{cf})
    # This appears inside δΓ and hence inside δ²R
    inner = TDeriv(down(:d), Tensor(:h, [down(:c), down(:f)]))
    mid = tproduct(1//1, TensorExpr[Tensor(:g, [up(:e), up(:f)]), inner])
    expr = TDeriv(down(:e), mid)

    println("Expression: ", string(expr))

    # Method A: Fourier then simplify
    fA = to_fourier(expr)
    fA = simplify(fA; registry=reg)
    fA = fix_dummy_positions(fA)
    println("\nMethod A (Fourier→simplify): ", string(fA))

    # Method C: simplify then Fourier then simplify
    sC = simplify(expr; registry=reg)
    println("Simplified position: ", string(sC))
    fC = to_fourier(sC)
    fC = simplify(fC; registry=reg)
    fC = fix_dummy_positions(fC)
    println("Method C (simplify→Fourier→simplify): ", string(fC))

    println("\nA == C? ", fA == fC)

    # ── Now test with a product: h * ∂(g * ∂h + g * ∂h - g * ∂h) ──
    # This mimics term 2 of raw δ²R: h^{cd} * (∂_e δΓ^e_{cd} - ∂_d δΓ^e_{ce})
    println("\n═══ More complex: h * ∂(sum of g·∂h) ═══")

    t1 = tproduct(1//2, TensorExpr[Tensor(:g, [up(:e), up(:f)]),
         TDeriv(down(:d), Tensor(:h, [down(:c), down(:f)]))])
    t2 = tproduct(1//2, TensorExpr[Tensor(:g, [up(:e), up(:f)]),
         TDeriv(down(:c), Tensor(:h, [down(:d), down(:f)]))])
    t3 = tproduct(-1//2, TensorExpr[Tensor(:g, [up(:e), up(:f)]),
         TDeriv(down(:f), Tensor(:h, [down(:d), down(:c)]))])
    δΓ_sum = tsum(TensorExpr[t1, t2, t3])  # δΓ^e_{dc} bracket

    # ∂_e(δΓ_sum) — one piece of δRic
    dδΓ = TDeriv(down(:e), δΓ_sum)

    # h^{cd} * ∂_e(δΓ^e_{dc})
    expr2 = tproduct(1//1, TensorExpr[
        Tensor(:h, [up(:c), up(:d)]), dδΓ])

    println("Expression: h^{cd} * ∂_e(δΓ^e_{dc})")

    fA2 = to_fourier(expr2)
    fA2 = simplify(fA2; registry=reg)
    fA2 = fix_dummy_positions(fA2)
    println("\nMethod A: $(fA2 isa TSum ? length(fA2.terms) : 1) terms")
    if fA2 isa TSum
        for (i,t) in enumerate(fA2.terms); println("  $i: ", string(t)); end
    else
        println("  ", string(fA2))
    end

    sC2 = simplify(expr2; registry=reg)
    fC2 = to_fourier(sC2)
    fC2 = simplify(fC2; registry=reg)
    fC2 = fix_dummy_positions(fC2)
    println("\nMethod C: $(fC2 isa TSum ? length(fC2.terms) : 1) terms")
    if fC2 isa TSum
        for (i,t) in enumerate(fC2.terms); println("  $i: ", string(t)); end
    else
        println("  ", string(fC2))
    end

    println("\nA == C? ", fA2 == fC2)

    # ── Even simpler: just ∂_a(g^{ab} T_b) vs g^{ab} ∂_a T_b ──
    println("\n═══ Simplest: ∂_a(g^{ab} h_{bc}) ═══")
    e3 = TDeriv(down(:a), tproduct(1//1, TensorExpr[
        Tensor(:g, [up(:a), up(:b)]), Tensor(:h, [down(:b), down(:c)])]))

    fA3 = simplify(to_fourier(e3); registry=reg)
    fC3 = to_fourier(simplify(e3; registry=reg))
    fC3 = simplify(fC3; registry=reg)
    println("∂_a(g^{ab} h_{bc}):")
    println("  A (Fourier first): ", string(fA3))
    println("  C (simplify first): ", string(fC3))
    println("  equal? ", fA3 == fC3)
end
