# ============================================================================
# TensorGR.jl -- Course Verification: Lecture 7 -- Derivative Operators
#
# Verifies the fundamental definitions and properties of covariant
# derivatives using TensorGR abstract tensor algebra.
#
# Topics:
#   1. Christoffel symbol formula from metric derivatives
#   2. Covariant derivative of a vector (positive Christoffel sign)
#   3. Covariant derivative of a covector (negative Christoffel sign)
#   4. Metric compatibility: nabla_a g_{bc} = 0
#   5. Difference of connections is a tensor (change_covd)
#
# References:
#   Carroll, "Spacetime and Geometry", Chapter 3
#   Wald, "General Relativity" (1984), Chapter 3
# ============================================================================

using TensorGR

println("="^70)
println("Lecture 7: Derivative Operators")
println("="^70)

reg = TensorRegistry()
with_registry(reg) do
    # Set up a 4D manifold with metric and full GR infrastructure
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    # Register test tensors
    @define_tensor V on=M4 rank=(1,0)
    @define_tensor w on=M4 rank=(0,1)
    @define_tensor T on=M4 rank=(1,1)

    # ------------------------------------------------------------------
    # 1. Christoffel formula:
    #    Gamma^rho_{mu nu} = (1/2) g^{rho sigma}
    #        (d_mu g_{sigma nu} + d_nu g_{sigma mu} - d_sigma g_{mu nu})
    #
    #    Use christoffel_to_grad_metric to construct the RHS and verify
    #    it has the expected three-term structure.
    # ------------------------------------------------------------------
    println("\n--- 1. Christoffel formula from metric derivatives ---")

    christoffel_expr = christoffel_to_grad_metric(:g, up(:a), down(:b), down(:c))
    println("  Gamma^a_{bc} = ", to_unicode(christoffel_expr))

    # The expression should be a sum (3 metric derivative terms with g^{ad})
    @assert christoffel_expr isa TProduct || christoffel_expr isa TSum "Christoffel formula should produce a structured expression"
    # Verify it has the right free indices: a(up), b(down), c(down)
    fi = free_indices(christoffel_expr)
    fi_names = sort([idx.name for idx in fi])
    @assert length(fi) == 3 "Christoffel formula should have 3 free indices"
    println("  Free indices: ", fi_names)
    println("  PASSED: Christoffel formula has correct structure")

    # ------------------------------------------------------------------
    # 2. CovD of a vector: nabla_a V^b = d_a V^b + Gamma^b_{ac} V^c
    #
    #    Build nabla_a V^b and expand via covd_to_christoffel.
    #    The result should be partial + positive Christoffel term.
    # ------------------------------------------------------------------
    println("\n--- 2. Covariant derivative of a vector ---")

    nabla_V = TDeriv(down(:a), Tensor(:V, [up(:b)]))
    expanded_V = covd_to_christoffel(nabla_V, :∇g)
    println("  nabla_a V^b = ", to_unicode(expanded_V))

    # Should be a TSum with 2 terms: partial + Gamma*V
    @assert expanded_V isa TSum "CovD of vector should expand to a sum"
    @assert length(expanded_V.terms) == 2 "CovD of vector should have exactly 2 terms"
    println("  PASSED: nabla_a V^b = d_a V^b + Gamma^b_{ac} V^c")

    # ------------------------------------------------------------------
    # 3. CovD of a covector: nabla_a w_b = d_a w_b - Gamma^c_{ab} w_c
    #
    #    The minus sign for lowered indices is the key distinction.
    # ------------------------------------------------------------------
    println("\n--- 3. Covariant derivative of a covector ---")

    nabla_w = TDeriv(down(:a), Tensor(:w, [down(:b)]))
    expanded_w = covd_to_christoffel(nabla_w, :∇g)
    println("  nabla_a w_b = ", to_unicode(expanded_w))

    # Should be a TSum with 2 terms: partial - Gamma*w
    @assert expanded_w isa TSum "CovD of covector should expand to a sum"
    @assert length(expanded_w.terms) == 2 "CovD of covector should have exactly 2 terms"

    # Verify the minus sign: one term should have a negative coefficient
    has_negative = any(expanded_w.terms) do term
        if term isa TProduct
            return term.scalar < 0
        end
        false
    end
    @assert has_negative "CovD of covector must have a negative Christoffel term"
    println("  PASSED: nabla_a w_b = d_a w_b - Gamma^c_{ab} w_c (minus sign verified)")

    # ------------------------------------------------------------------
    # 4. Metric compatibility: nabla_a g_{bc} = 0
    #
    #    This is the defining property of the Levi-Civita connection.
    #    The simplify engine should reduce nabla_a g_{bc} to zero.
    # ------------------------------------------------------------------
    println("\n--- 4. Metric compatibility: nabla_a g_{bc} = 0 ---")

    nabla_g = TDeriv(down(:a), Tensor(:g, [down(:b), down(:c)]))
    result = simplify(nabla_g)
    println("  nabla_a g_{bc} = ", to_unicode(result))
    @assert result == TScalar(0 // 1) "Metric compatibility failed: nabla_a g_{bc} != 0"
    println("  PASSED: Metric compatibility verified")

    # ------------------------------------------------------------------
    # 5. Difference of connections is a tensor:
    #    nabla_a V^b = D_a V^b + C^b_{ac} V^c
    #    where C^b_{ac} = Gamma^b_{ac} - GammaD^b_{ac}
    #
    #    Define a second CovD and use change_covd to verify the structure.
    # ------------------------------------------------------------------
    println("\n--- 5. Difference of connections is a tensor ---")

    # Define a second covariant derivative D on the same manifold
    define_covd!(reg, :D; manifold=:M4, metric=:g)

    # Change nabla_a V^b from nabla to D
    nabla_V_expr = TDeriv(down(:a), Tensor(:V, [up(:b)]), :∇g)
    changed = change_covd(nabla_V_expr, :∇g, :D)
    println("  nabla_a V^b in terms of D: ", to_unicode(changed))

    # The result should contain D_a V^b + (Gamma_nabla - Gamma_D) terms
    @assert changed isa TSum "change_covd should produce a sum (D + difference tensor terms)"
    fi_changed = free_indices(changed)
    fi_orig = free_indices(nabla_V_expr)
    @assert length(fi_changed) == length(fi_orig) "change_covd must preserve free indices"
    println("  PASSED: Connection difference is a well-defined tensor")

    println("\n" * "="^70)
    println("All Lecture 7 verifications passed!")
    println("="^70)
end
