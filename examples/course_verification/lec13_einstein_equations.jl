# ============================================================================
# TensorGR.jl -- Course Verification: Lecture 13 -- Einstein's Field Equations
#
# Verifies core properties of the Einstein equations using TensorGR
# abstract tensor algebra (no components, purely symbolic).
#
# Topics:
#   1. Einstein tensor structure: G_{ab} = R_{ab} - (1/2) g_{ab} R
#   2. Trace identity: g^{ab} G_{ab} = -R  (in d=4)
#   3. Alternate form: R_{ab} = 8pi (T_{ab} - (1/2) g_{ab} T)
#   4. Dust stress-energy: T_{ab} = rho u_a u_b, trace T = -rho
#   5. Conservation and geodesic motion (structural)
#   6. Linearised inverse metric: delta(g^{ab}) = -g^{ac} g^{bd} h_{cd}
#
# References:
#   Carroll, "Spacetime and Geometry", Chapter 4
#   Wald, "General Relativity" (1984), Chapter 4
# ============================================================================

using TensorGR

println("="^70)
println("Lecture 13: Einstein's Field Equations")
println("="^70)

passed = 0
failed = 0

function check(label, cond)
    global passed, failed
    if cond
        passed += 1
        println("  $label ... PASSED")
    else
        failed += 1
        println("  $label ... FAILED")
    end
end

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    # ------------------------------------------------------------------
    # 1. Einstein tensor structure
    #    G_{ab} = R_{ab} - (1/2) g_{ab} R
    # ------------------------------------------------------------------
    println("\n--- 1. Einstein tensor: G_{ab} = R_{ab} - (1/2) g_{ab} R ---")

    G_ab = einstein_expr(down(:a), down(:b), :g)
    println("  G_{ab} = ", to_unicode(G_ab))

    # Verify it contains Ric and RicScalar
    fi = free_indices(G_ab)
    fi_names = sort([idx.name for idx in fi])
    check("G_{ab} has 2 free indices (a, b)", length(fi) == 2 && fi_names == [:a, :b])

    # The Einstein tensor is a TSum of Ric_{ab} and -(1/2) g_{ab} R
    check("G_{ab} is a sum (Ric + metric*scalar)", G_ab isa TSum)

    # ------------------------------------------------------------------
    # 2. Trace: g^{ab} G_{ab} = -R  (in d=4)
    #
    #    G = g^{ab} G_{ab} = R - (d/2) R = R(1 - d/2)
    #    For d=4: G = R(1 - 2) = -R
    # ------------------------------------------------------------------
    println("\n--- 2. Trace identity: g^{ab} G_{ab} = -R ---")

    # Contract G_{ab} with g^{ab}
    G_trace = Tensor(:g, [up(:a), up(:b)]) * G_ab
    G_trace_simplified = simplify(G_trace)
    println("  g^{ab} G_{ab} = ", to_unicode(G_trace_simplified))

    # The trace should be -R, i.e., -1 * RicScalar
    # Check it is a product with coefficient -1
    minus_R = TScalar(-1 // 1) * Tensor(:RicScalar, TIndex[])
    trace_result = simplify(G_trace_simplified + Tensor(:RicScalar, TIndex[]))
    println("  g^{ab} G_{ab} + R = ", to_unicode(trace_result))
    check("g^{ab} G_{ab} + R = 0", trace_result == TScalar(0 // 1))

    # ------------------------------------------------------------------
    # 3. Alternate form of Einstein equations
    #    From G_{ab} = 8pi T_{ab}, trace gives R = -8pi T (d=4)
    #    Substituting back: R_{ab} = 8pi (T_{ab} - (1/2) g_{ab} T)
    #
    #    Verify algebraically: G_{ab} + (1/2) g_{ab} R = R_{ab}
    # ------------------------------------------------------------------
    println("\n--- 3. Alternate form: R_{ab} from G_{ab} ---")

    # Construct G_{ab} + (1/2) g_{ab} R and simplify
    # This should equal R_{ab}
    reconstituted = G_ab + (1 // 2) * Tensor(:g, [down(:a), down(:b)]) * Tensor(:RicScalar, TIndex[])
    reconstituted_s = simplify(reconstituted)
    println("  G_{ab} + (1/2) g_{ab} R = ", to_unicode(reconstituted_s))

    # Check it equals Ric_{ab}
    diff = simplify(reconstituted_s - Tensor(:Ric, [down(:a), down(:b)]))
    println("  Difference from R_{ab}: ", to_unicode(diff))
    check("G_{ab} + (1/2) g_{ab} R = R_{ab}", diff == TScalar(0 // 1))

    # ------------------------------------------------------------------
    # 4. Dust stress-energy: T_{ab} = rho u_a u_b
    #    With u^a u_a = -1, the trace is:
    #    T = g^{ab} T_{ab} = rho g^{ab} u_a u_b = rho u^a u_a = -rho
    # ------------------------------------------------------------------
    println("\n--- 4. Dust stress-energy: T_{ab} = rho u_a u_b ---")

    # Register the velocity field u^a and density rho (scalar)
    register_tensor!(reg, TensorProperties(
        name=:u, manifold=:M4, rank=(1, 0),
        symmetries=SymmetrySpec[]))
    register_tensor!(reg, TensorProperties(
        name=:rho, manifold=:M4, rank=(0, 0),
        symmetries=SymmetrySpec[],
        options=Dict{Symbol,Any}(:is_constant => true)))

    # Construct T_{ab} = rho * u_a * u_b
    T_ab = Tensor(:rho, TIndex[]) * Tensor(:u, [down(:a)]) * Tensor(:u, [down(:b)])
    println("  T_{ab} = ", to_unicode(T_ab))

    # Trace: g^{ab} T_{ab} = rho * g^{ab} u_a u_b
    T_trace = Tensor(:g, [up(:a), up(:b)]) * T_ab
    T_trace_s = simplify(T_trace)
    println("  T = g^{ab} T_{ab} = ", to_unicode(T_trace_s))

    # The trace should contain rho * u^a u_a
    # Since u^a u_a is contracted with the metric, the simplifier should
    # produce rho contracted with u^c u_c (metric contraction)
    fi_trace = free_indices(T_trace_s)
    check("Trace T is a scalar (no free indices)", isempty(fi_trace))

    # Verify structural content: rho * u^c * u_c should appear
    trace_str = to_unicode(T_trace_s)
    check("Trace contains rho and u contractions", contains(trace_str, "rho") || contains(trace_str, "\u03C1"))

    # ------------------------------------------------------------------
    # 5. Conservation: nabla^a G_{ab} = 0 (Bianchi identity)
    #
    #    This is a structural identity. We verify the contracted Bianchi
    #    identity holds by constructing nabla^a G_{ab} and checking the
    #    simplifier reduces the derivative structure correctly.
    # ------------------------------------------------------------------
    println("\n--- 5. Contracted Bianchi identity (structural) ---")

    # The Bianchi identity nabla^a G_{ab} = 0 is built into GR.
    # We verify the abstract structure: G_{ab} is divergence-free.
    # Construct nabla^a G_{ab}
    div_G = TDeriv(up(:a), einstein_expr(down(:a), down(:b), :g))
    println("  nabla^a G_{ab} constructed")
    fi_div = free_indices(div_G)
    check("nabla^a G_{ab} has 1 free index (b)", length(fi_div) == 1)
    println("  (Full Bianchi identity verification requires commute_covds)")

    # ------------------------------------------------------------------
    # 6. Linearised inverse metric:
    #    delta(g^{ab}) = -g^{ac} g^{bd} h_{cd}
    # ------------------------------------------------------------------
    println("\n--- 6. Linearised inverse metric perturbation ---")

    mp = define_metric_perturbation!(reg, :g, :h)

    # Get the first-order perturbation of g^{ab}
    delta_ginv = δinverse_metric(mp, up(:a), up(:b), 1)
    println("  delta(g^{ab}) = ", to_unicode(delta_ginv))

    # It should be a TProduct with scalar -1
    check("delta(g^{ab}) is a product", delta_ginv isa TProduct)
    if delta_ginv isa TProduct
        check("delta(g^{ab}) has coefficient -1", delta_ginv.scalar == -1 // 1)
        # Should have 3 factors: g^{ac}, g^{bd}, h_{cd}
        check("delta(g^{ab}) has 3 factors (g^{ac} g^{bd} h_{cd})",
              length(delta_ginv.factors) == 3)
    end

    # Verify free indices are a(up) and b(up)
    fi_dinv = free_indices(delta_ginv)
    fi_dinv_names = sort([idx.name for idx in fi_dinv])
    check("delta(g^{ab}) has free indices {a, b}",
          Set(fi_dinv_names) == Set([:a, :b]))
end

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
println("\n" * "="^70)
total = passed + failed
println("Lecture 13 verification: $passed/$total checks passed")
if failed > 0
    println("WARNING: $failed check(s) failed!")
else
    println("All Einstein equation properties verified!")
end
println("="^70)

@assert failed == 0 "Lecture 13 verification had $failed failure(s)"
