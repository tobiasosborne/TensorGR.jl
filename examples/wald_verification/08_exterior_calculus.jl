# ============================================================================
# TensorGR.jl -- Wald Verification: Exterior Calculus
#
# Verifies fundamental exterior calculus identities: d^2 = 0 and
# the Maxwell-Bianchi identity dF = 0.
#
# References:
#   Wald, "General Relativity" (1984), Appendix B
#   Carroll, "Spacetime and Geometry", Appendix E
# ============================================================================

using TensorGR

println("="^70)
println("Wald Verification 08: Exterior Calculus")
println("="^70)

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    # ------------------------------------------------------------------
    # 1. d^2 = 0 for a 0-form (scalar):
    #    d(df) = d_a(d_b f) is antisymmetric in (a,b), hence
    #    the antisymmetric part vanishes by symmetry of partials.
    #
    #    We verify: d_a(d_b f) - d_b(d_a f) = 0 (partials commute).
    # ------------------------------------------------------------------
    println("\n--- 1. d^2 = 0 for a scalar (partials commute) ---")

    @define_tensor f on=M4 rank=(0,0)
    scalar_f = Tensor(:f, TIndex[])

    ddf_ab = TDeriv(down(:a), TDeriv(down(:b), scalar_f))
    ddf_ba = TDeriv(down(:b), TDeriv(down(:a), scalar_f))
    d2f = simplify(ddf_ab - ddf_ba)
    println("  d_a(d_b f) - d_b(d_a f) = ", to_unicode(d2f))
    @assert d2f == TScalar(0 // 1) "d^2 f != 0 (partials should commute)"
    println("  PASSED")

    # ------------------------------------------------------------------
    # 2. d^2 = 0 for a 1-form:
    #    For a 1-form A_b, (dA)_{ab} = d_a A_b (exterior derivative).
    #    Then d(dA)_{cab} = d_c(d_a A_b).
    #    Antisymmetrizing over (c,a,b) should give zero.
    #
    #    d(dA)_{[cab]} = d_c d_a A_b + d_a d_b A_c + d_b d_c A_a
    #                   - d_a d_c A_b - d_c d_b A_a - d_b d_a A_c = 0
    #    (Each pair cancels by commutativity of partial derivatives.)
    # ------------------------------------------------------------------
    println("\n--- 2. d^2 = 0 for a 1-form ---")

    define_form!(reg, :A; manifold=:M4, degree=1)

    # Full antisymmetric sum: all 3! = 6 permutations of (c, a, b)
    term1 = TDeriv(down(:c), TDeriv(down(:a), Tensor(:A, [down(:b)])))
    term2 = TDeriv(down(:a), TDeriv(down(:b), Tensor(:A, [down(:c)])))
    term3 = TDeriv(down(:b), TDeriv(down(:c), Tensor(:A, [down(:a)])))
    term4 = TDeriv(down(:a), TDeriv(down(:c), Tensor(:A, [down(:b)])))
    term5 = TDeriv(down(:c), TDeriv(down(:b), Tensor(:A, [down(:a)])))
    term6 = TDeriv(down(:b), TDeriv(down(:a), Tensor(:A, [down(:c)])))

    antisym_ddA = simplify(term1 + term2 + term3 - term4 - term5 - term6)
    println("  d(dA)_{[cab]} = ", to_unicode(antisym_ddA))
    @assert antisym_ddA == TScalar(0 // 1) "d^2 A != 0"
    println("  PASSED")

    # ------------------------------------------------------------------
    # 3. Maxwell-Bianchi: dF = 0
    #    If F = dA (the electromagnetic field strength), then dF = d(dA) = 0.
    #    This is the Bianchi identity for electromagnetism:
    #      d_{[a} F_{bc]} = 0
    #
    #    We build F_{ab} = d_a A_b - d_b A_a and verify the cyclic identity
    #      d_c F_{ab} + d_a F_{bc} + d_b F_{ca} = 0
    #    by first using expand_derivatives to distribute d through F.
    # ------------------------------------------------------------------
    println("\n--- 3. Maxwell-Bianchi identity: d_{[c} F_{ab]} = 0 ---")

    # F_{ab} = d_a A_b - d_b A_a (field strength of a 1-form)
    F_ab = TDeriv(down(:a), Tensor(:A, [down(:b)])) -
           TDeriv(down(:b), Tensor(:A, [down(:a)]))
    println("  F_{ab} = ", to_unicode(F_ab))

    # d_c F_{ab} + d_a F_{bc} + d_b F_{ca}
    # We must expand_derivatives to distribute d through the sum F
    F_bc = TDeriv(down(:b), Tensor(:A, [down(:c)])) -
           TDeriv(down(:c), Tensor(:A, [down(:b)]))
    F_ca = TDeriv(down(:c), Tensor(:A, [down(:a)])) -
           TDeriv(down(:a), Tensor(:A, [down(:c)]))

    dF_cab = expand_derivatives(TDeriv(down(:c), F_ab))
    dF_abc = expand_derivatives(TDeriv(down(:a), F_bc))
    dF_bca = expand_derivatives(TDeriv(down(:b), F_ca))

    bianchi = simplify(dF_cab + dF_abc + dF_bca)
    println("  d_c F_{ab} + d_a F_{bc} + d_b F_{ca} = ", to_unicode(bianchi))
    @assert bianchi == TScalar(0 // 1) "Maxwell-Bianchi dF = 0 failed"
    println("  PASSED")

    # ------------------------------------------------------------------
    # 4. Exterior derivative structure
    #    exterior_d(A, 1, d_a) = d_a A_b (a TDeriv)
    # ------------------------------------------------------------------
    println("\n--- 4. Exterior derivative structure ---")

    dA = exterior_d(Tensor(:A, [down(:b)]), 1, down(:a))
    println("  dA = ", to_unicode(dA))
    @assert dA isa TDeriv "Exterior derivative should produce TDeriv"
    @assert dA.index == down(:a) "Derivative index should be a"
    println("  PASSED: dA is a TDeriv with correct structure")

    println("\n" * "="^70)
    println("All exterior calculus identities verified!")
    println("="^70)
end
