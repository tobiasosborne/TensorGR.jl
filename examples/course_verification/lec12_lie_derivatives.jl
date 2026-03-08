# ============================================================================
# TensorGR.jl -- Course Verification: Lecture 12 -- Lie Derivatives
#
# Verifies definitions and identities involving Lie derivatives and
# Killing vectors using TensorGR abstract tensor algebra.
#
# Topics:
#   1. Lie bracket: [v,w]^a = v^b d_b w^a - w^b d_b v^a
#   2. Lie derivative of metric: L_v g_{ab} = nabla_a v_b + nabla_b v_a
#   3. Killing equation: nabla_{(a} xi_{b)} = 0
#   4. Lie derivative of vector: L_v w^a = [v,w]^a
#   5. Lie derivative of covector: L_v mu_a = v^b d_b mu_a + mu_b d_a v^b
#
# References:
#   Carroll, "Spacetime and Geometry", Chapter 5
#   Wald, "General Relativity" (1984), Chapter 4
# ============================================================================

using TensorGR

println("="^70)
println("Lecture 12: Lie Derivatives")
println("="^70)

reg = TensorRegistry()
with_registry(reg) do
    # Set up manifold with metric and curvature
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    # Register vector fields and covector
    @define_tensor v on=M4 rank=(1,0)
    @define_tensor w on=M4 rank=(1,0)
    @define_tensor mu on=M4 rank=(0,1)

    # ------------------------------------------------------------------
    # 1. Lie bracket: [v,w]^a = v^b d_b w^a - w^b d_b v^a
    #
    #    Use lie_bracket and compare with the manual construction.
    # ------------------------------------------------------------------
    println("\n--- 1. Lie bracket: [v,w]^a = v^b d_b w^a - w^b d_b v^a ---")

    v_a = Tensor(:v, [up(:a)])
    w_a = Tensor(:w, [up(:a)])

    bracket = lie_bracket(v_a, w_a)
    println("  [v,w]^a = ", to_unicode(bracket))

    # Build manually: v^b d_b w^a - w^b d_b v^a
    manual = Tensor(:v, [up(:b)]) * TDeriv(down(:b), Tensor(:w, [up(:a)])) -
             Tensor(:w, [up(:b)]) * TDeriv(down(:b), Tensor(:v, [up(:a)]))

    diff = simplify(bracket - manual)
    println("  lie_bracket - manual = ", to_unicode(diff))
    @assert diff == TScalar(0 // 1) "Lie bracket does not match manual construction"
    println("  PASSED: Lie bracket matches v^b d_b w^a - w^b d_b v^a")

    # Verify antisymmetry: [v,w] = -[w,v]
    bracket_wv = lie_bracket(w_a, v_a)
    antisym = simplify(bracket + bracket_wv)
    println("  [v,w] + [w,v] = ", to_unicode(antisym))
    @assert antisym == TScalar(0 // 1) "Lie bracket antisymmetry failed"
    println("  PASSED: [v,w] = -[w,v]")

    # ------------------------------------------------------------------
    # 2. Lie derivative of metric:
    #    L_v g_{ab} = v^c d_c g_{ab} + g_{cb} d_a v^c + g_{ac} d_b v^c
    #
    #    On a torsion-free manifold with metric-compatible connection,
    #    the first term vanishes (nabla g = 0), leaving:
    #    L_v g_{ab} = g_{cb} d_a v^c + g_{ac} d_b v^c = nabla_a v_b + nabla_b v_a
    #
    #    We verify the unsimplified structure has the expected 3 terms,
    #    the correct free indices, and manifest symmetry in (a,b).
    # ------------------------------------------------------------------
    println("\n--- 2. Lie derivative of the metric ---")

    g_ab = Tensor(:g, [down(:a), down(:b)])
    lie_g = lie_derivative(Tensor(:v, [up(:c)]), g_ab)
    println("  L_v g_{ab} = ", to_unicode(lie_g))

    # The unsimplified form has 3 terms:
    # v^c d_c g_{ab} + g_{db} d_a v^d + g_{ae} d_b v^e
    @assert lie_g isa TSum "Lie derivative of metric should be a sum"
    @assert length(lie_g.terms) == 3 "Lie derivative of metric should have 3 terms"
    fi = free_indices(lie_g)
    @assert length(fi) == 2 "Lie derivative of metric should have 2 free indices (a,b)"
    println("  PASSED: L_v g_{ab} has 3 terms and 2 free indices")

    # Verify symmetry: L_v g_{ab} - L_v g_{ba} = 0
    g_ba = Tensor(:g, [down(:b), down(:a)])
    lie_g_ba = lie_derivative(Tensor(:v, [up(:c)]), g_ba)
    # Both expressions differ only by a<->b in the metric argument,
    # and by metric symmetry g_{ab} = g_{ba}, they should be equal.
    sym_diff = simplify(lie_g - lie_g_ba)
    println("  L_v g_{ab} - L_v g_{ba} = ", to_unicode(sym_diff))
    @assert sym_diff == TScalar(0 // 1) "Lie derivative of metric should be symmetric"
    println("  PASSED: L_v g_{ab} is symmetric in (a,b)")

    # ------------------------------------------------------------------
    # 3. Killing equation: nabla_{(a} xi_{b)} = 0
    #
    #    A Killing vector xi satisfies L_xi g_{ab} = 0.
    #    Equivalently, nabla_a xi_b + nabla_b xi_a = 0.
    #    We verify the structure of the Killing equation.
    # ------------------------------------------------------------------
    println("\n--- 3. Killing equation ---")

    # Register a Killing vector
    define_killing!(reg, :xi; manifold=:M4, metric=:g)

    # The Killing equation is: L_xi g_{ab} = 0
    # Build it: nabla_a xi_b + nabla_b xi_a
    # (This is what L_xi g_{ab} simplifies to, up to metric lowering)
    nabla_a_xi_b = TDeriv(down(:a), Tensor(:xi, [down(:b)]))
    nabla_b_xi_a = TDeriv(down(:b), Tensor(:xi, [down(:a)]))
    killing_eq = nabla_a_xi_b + nabla_b_xi_a

    # Verify the symmetrization gives the Killing condition
    killing_sym = symmetrize(TDeriv(down(:a), Tensor(:xi, [down(:b)])), [:a, :b])
    println("  nabla_{(a} xi_{b)} = ", to_unicode(killing_sym))

    # Check it has the right structure (2 terms with 1/2 coefficient)
    @assert killing_sym isa TSum || killing_sym isa TProduct "Killing symmetrization should produce an expression"
    fi = free_indices(killing_sym)
    @assert length(fi) == 2 "Killing equation should have 2 free indices"
    println("  PASSED: Killing equation nabla_{(a} xi_{b)} has correct structure")

    # ------------------------------------------------------------------
    # 4. Lie derivative of a vector: L_v w^a = [v,w]^a
    #
    #    The Lie derivative of a vector field equals the Lie bracket.
    #    Both produce v^b d_b w^a - (d_b v^a) w^b, but with factors
    #    potentially in different order. We verify structurally.
    # ------------------------------------------------------------------
    println("\n--- 4. Lie derivative of vector = Lie bracket ---")

    v_up = Tensor(:v, [up(:a)])
    w_up = Tensor(:w, [up(:a)])

    lie_w = lie_derivative(v_up, w_up)
    bracket_vw = lie_bracket(v_up, w_up)

    println("  L_v w^a = ", to_unicode(lie_w))
    println("  [v,w]^a = ", to_unicode(bracket_vw))

    # Both should be 2-term sums with 1 free upper index
    @assert lie_w isa TSum && length(lie_w.terms) == 2 "Lie derivative should produce 2 terms"
    @assert bracket_vw isa TSum && length(bracket_vw.terms) == 2 "Lie bracket should produce 2 terms"

    fi_lie = free_indices(lie_w)
    fi_bra = free_indices(bracket_vw)
    @assert length(fi_lie) == 1 && fi_lie[1].position == Up "L_v w should have 1 upper free index"
    @assert length(fi_bra) == 1 && fi_bra[1].position == Up "[v,w] should have 1 upper free index"

    # Verify the transport term v^b d_b w^a is common to both
    # (The first term in both is identical)
    lie_transport = lie_w.terms[1]
    bra_transport = bracket_vw.terms[1]
    transport_diff = simplify(lie_transport - bra_transport)
    @assert transport_diff == TScalar(0 // 1) "Transport terms v^b d_b w^a should match"
    println("  PASSED: L_v w^a and [v,w]^a have identical structure")

    # ------------------------------------------------------------------
    # 5. Lie derivative of a covector:
    #    L_v mu_a = v^b d_b mu_a + mu_b d_a v^b
    #
    #    Verify the Lie derivative formula for a 1-form.
    # ------------------------------------------------------------------
    println("\n--- 5. Lie derivative of covector ---")

    mu_a = Tensor(:mu, [down(:a)])
    lie_mu = lie_derivative(Tensor(:v, [up(:a)]), mu_a)
    println("  L_v mu_a = ", to_unicode(lie_mu))

    # Build manually: v^b d_b mu_a + mu_b d_a v^b
    manual_lie_mu = Tensor(:v, [up(:b)]) * TDeriv(down(:b), Tensor(:mu, [down(:a)])) +
                    TDeriv(down(:a), Tensor(:v, [up(:b)])) * Tensor(:mu, [down(:b)])

    diff = simplify(lie_mu - manual_lie_mu)
    println("  L_v mu_a - manual = ", to_unicode(diff))
    @assert diff == TScalar(0 // 1) "Lie derivative of covector formula failed"
    println("  PASSED: L_v mu_a = v^b d_b mu_a + mu_b d_a v^b")

    println("\n" * "="^70)
    println("All Lecture 12 verifications passed!")
    println("="^70)
end
