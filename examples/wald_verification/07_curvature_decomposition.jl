# ============================================================================
# TensorGR.jl -- Wald Verification: Curvature Decomposition
#
# Verifies Weyl decomposition, curvature conversion roundtrips,
# and trace properties.
#
# References:
#   Wald, "General Relativity" (1984), Section 3.2
#   Carroll, "Spacetime and Geometry", Section 3.7
# ============================================================================

using TensorGR

println("="^70)
println("Wald Verification 07: Curvature Decomposition")
println("="^70)

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    # ------------------------------------------------------------------
    # 1. Weyl decomposition roundtrip:
    #    R_{abcd} = C_{abcd} + (Ricci terms)
    #    So: riemann_to_weyl gives R as C + f(Ric, R, g)
    #        weyl_to_riemann gives C as R - f(Ric, R, g)
    #    Their difference should give: R_{abcd} - R_{abcd} = 0
    #    i.e., riemann_to_weyl - weyl_to_riemann should leave only
    #    2*f(Ric, R, g) (the Ricci decomposition terms, doubled).
    #
    #    More precisely: riemann_to_weyl = C + f, weyl_to_riemann = R - f
    #    So: riemann_to_weyl(R) - (R itself) should be f (Ricci terms)
    #    and adding back weyl_to_riemann should cancel.
    # ------------------------------------------------------------------
    println("\n--- 1. Weyl decomposition: to_riemann(to_weyl) roundtrip ---")

    a, b, c, d = down(:a), down(:b), down(:c), down(:d)

    # Start with the Weyl tensor (it is the Riemann minus Ricci parts)
    # weyl_to_riemann: C_{abcd} -> R_{abcd} - f(Ric, R, g)
    # Substituting into riemann_to_weyl should give C back
    weyl_as_riemann = weyl_to_riemann(a, b, c, d, :g; dim=4)
    println("  C_{abcd} expressed via Riemann:")
    println("    ", to_unicode(weyl_as_riemann))

    # Now apply riemann_to_weyl to the Riemann part.
    # We build: Riem = C + f(Ric, R, g), then substitute C = R - f -> R - f + f = R
    # The direct check: R_{abcd} = riemann_to_weyl, which gives C + f.
    # Take riemann_to_weyl and substitute C = weyl_to_riemann:
    riem_decomposed = riemann_to_weyl(a, b, c, d, :g; dim=4)
    println("\n  R_{abcd} decomposed:")
    println("    ", to_unicode(riem_decomposed))

    # Verify: riemann_to_weyl and weyl_to_riemann are consistent
    # riemann_to_weyl gives: C + f(Ric, R, g)
    # The actual Riemann is just Tensor(:Riem, ...), so:
    # C + f - R_{abcd} should be zero (it IS the decomposition)
    Riem_abcd = Tensor(:Riem, [a, b, c, d])
    check = simplify(riem_decomposed - Riem_abcd)
    println("\n  Decomposition - R_{abcd} = ", to_unicode(check))
    # This should NOT be zero -- it's the decomposition definition, not an identity.
    # Instead, it should be: C + f - R = 0, but only if C = R - f.
    # Actually riemann_to_weyl returns: C + f (the full decomposition, not R).
    # It should equal R when C is defined as R - f.

    # Correct check: weyl_to_riemann returns R - f, and riemann_to_weyl returns C + f.
    # Their sum should be R + C (no cross-cancellation needed).
    # The identity: R = C + f, so R - (C + f) = 0
    identity_check = simplify(Riem_abcd - riem_decomposed)
    println("  R_{abcd} - (C + Ricci terms) = ", to_unicode(identity_check))
    # This is a tautological identity only when C = R - f, which is
    # verified by checking that substituting weyl_to_riemann for C gives 0.

    # Substitute Weyl -> (R - f) in the decomposition formula (C + f)
    # This should yield R - f + f = R, and then R - R = 0
    expr_with_weyl_expanded = to_riemann(riem_decomposed; metric=:g, dim=4)
    round_trip = simplify(Riem_abcd - expr_with_weyl_expanded)
    println("  R_{abcd} - to_riemann(decomposition) = ", to_unicode(round_trip))
    @assert round_trip == TScalar(0 // 1) "Weyl decomposition roundtrip failed"
    println("  PASSED")

    # ------------------------------------------------------------------
    # 2. Weyl trace-freeness: C^a_{bac} = 0
    #    (Wald Section 3.2, defining property of Weyl tensor)
    #
    #    The Weyl tensor is traceless on every pair of indices.
    # ------------------------------------------------------------------
    println("\n--- 2. Weyl trace-freeness: C^a_{bac} = 0 ---")

    # Contracting first and third indices of Weyl
    Weyl_contracted = Tensor(:Weyl, [up(:a), down(:b), down(:a), down(:c)])
    result = simplify(Weyl_contracted)
    println("  C^a_{bac} = ", to_unicode(result))
    # Weyl has RiemannSymmetry, and traceless flag -- check via canonicalize
    # The contraction of Weyl should vanish if we express it via the decomposition
    # Let's use the conversion approach
    weyl_as_riem = weyl_to_riemann(up(:a), down(:b), down(:a), down(:c), :g; dim=4)
    weyl_trace = simplify(weyl_as_riem)
    println("  C^a_{bac} via Riemann decomposition = ", to_unicode(weyl_trace))
    # R^a_{bac} = R_{bc} (Ricci), minus the Ricci terms from the decomposition
    # which should exactly cancel, giving 0
    result_final = simplify(contract_curvature(weyl_as_riem))
    println("  After contract_curvature: ", to_unicode(result_final))
    @assert result_final == TScalar(0 // 1) "Weyl trace-freeness failed"
    println("  PASSED")

    # ------------------------------------------------------------------
    # 3. Einstein-Ricci roundtrip:
    #    einstein_to_ricci(a, b) = R_{ab} - (1/2) g_{ab} R
    #    ricci_to_einstein(a, b) = G_{ab} + (1/2) g_{ab} R
    #    Composing: ricci_to_einstein(einstein_to_ricci) = identity
    # ------------------------------------------------------------------
    println("\n--- 3. Einstein-Ricci roundtrip ---")

    # Start with G_{ab}, convert to Ricci form, then back
    Ein_ab = Tensor(:Ein, [a, b])
    as_ricci = einstein_to_ricci(a, b, :g)
    println("  G_{ab} as Ricci: ", to_unicode(as_ricci))

    # Now convert back: in the Ricci expression, replace Ric -> G + (1/2)gR
    back_to_ein = to_ricci(ricci_to_einstein(a, b, :g); metric=:g)
    println("  Roundtrip Ric -> Ein -> Ric: ", to_unicode(back_to_ein))

    # Direct check: einstein_to_ricci gives R_{ab} - (1/2)g_{ab}R
    # ricci_to_einstein gives G_{ab} + (1/2)g_{ab}R
    # Substituting G = R - (1/2)gR into the second:
    #   (R - (1/2)gR) + (1/2)gR = R  -- so it should give Ric_{ab}
    Ric_ab = Tensor(:Ric, [a, b])
    diff = simplify(back_to_ein - Ric_ab)
    println("  Roundtrip - R_{ab} = ", to_unicode(diff))
    @assert diff == TScalar(0 // 1) "Einstein-Ricci roundtrip failed"
    println("  PASSED")

    # ------------------------------------------------------------------
    # 4. Einstein tensor definition:
    #    G_{ab} = R_{ab} - (1/2) g_{ab} R  (Wald Eq. 3.2.28)
    # ------------------------------------------------------------------
    println("\n--- 4. Einstein tensor definition verification ---")

    ein_expr = einstein_expr(a, b, :g)
    ein_conv = einstein_to_ricci(a, b, :g)
    diff = simplify(ein_expr - ein_conv)
    println("  einstein_expr - einstein_to_ricci = ", to_unicode(diff))
    @assert diff == TScalar(0 // 1) "Einstein definition mismatch"
    println("  PASSED")

    # ------------------------------------------------------------------
    # 5. Riemann symmetry count check
    #    In 4D, Riemann has 20 independent components.
    #    Weyl has 10, Ricci has 10, scalar has 1.
    #    Verify structural decomposition has correct number of
    #    distinct tensor types.
    # ------------------------------------------------------------------
    println("\n--- 5. Decomposition structure check ---")

    decomp = riemann_to_weyl(a, b, c, d, :g; dim=4)
    # Count distinct tensor names in the decomposition
    tensor_names = Set{Symbol}()
    walk(decomp) do node
        if node isa Tensor
            push!(tensor_names, node.name)
        end
        node
    end
    println("  Tensors in Weyl decomposition: ", tensor_names)
    @assert :Weyl in tensor_names "Decomposition should contain Weyl"
    @assert :Ric in tensor_names "Decomposition should contain Ricci"
    @assert :RicScalar in tensor_names "Decomposition should contain Ricci scalar"
    @assert :g in tensor_names "Decomposition should contain metric"
    println("  PASSED: All expected tensors present")

    println("\n" * "="^70)
    println("All curvature decomposition identities verified!")
    println("="^70)
end
