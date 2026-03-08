# ============================================================================
# TensorGR.jl -- Course Verification: Lecture 14 -- Gravitational Radiation
#
# Verifies linearised gravity identities using the TensorGR perturbation
# engine (abstract tensor algebra, no components).
#
# Topics:
#   1. Linearised Riemann tensor (first-order perturbation)
#   2. Linearised Ricci tensor
#   3. Trace-reversed perturbation: gamma_bar = gamma - (1/2) eta gamma
#   4. Gauge transformation: gamma -> gamma + d_a xi_b + d_b xi_a
#
# References:
#   Carroll, "Spacetime and Geometry", Chapter 7
#   Wald, "General Relativity" (1984), Section 4.4
# ============================================================================

using TensorGR

println("="^70)
println("Lecture 14: Gravitational Radiation (Linearised Gravity)")
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

    # Define metric perturbation: g = g + epsilon * h
    mp = define_metric_perturbation!(reg, :g, :h)

    # ------------------------------------------------------------------
    # 1. Linearised Riemann tensor
    #    delta(R^a_{bcd}) at first order around flat background
    #    Should be non-zero and have the derivative-of-h structure.
    # ------------------------------------------------------------------
    println("\n--- 1. Linearised Riemann tensor ---")

    delta_Riem = δriemann(mp, up(:a), down(:b), down(:c), down(:d), 1)
    println("  delta(R^a_{bcd}) = ", to_unicode(delta_Riem))

    check("Linearised Riemann is non-zero", delta_Riem != TScalar(0 // 1))

    # Should have free indices a(up), b(down), c(down), d(down)
    fi = free_indices(delta_Riem)
    fi_names = sort([idx.name for idx in fi])
    check("Linearised Riemann has 4 free indices",
          length(fi) == 4 && Set(fi_names) == Set([:a, :b, :c, :d]))

    # ------------------------------------------------------------------
    # 2. Linearised Ricci tensor
    #    delta(R_{ab}) = delta(R^c_{acb})
    #    Non-zero with correct index structure.
    # ------------------------------------------------------------------
    println("\n--- 2. Linearised Ricci tensor ---")

    delta_Ric = δricci(mp, down(:a), down(:b), 1)
    println("  delta(R_{ab}) = ", to_unicode(delta_Ric))

    check("Linearised Ricci is non-zero", delta_Ric != TScalar(0 // 1))

    fi_ric = free_indices(delta_Ric)
    fi_ric_names = sort([idx.name for idx in fi_ric])
    check("Linearised Ricci has 2 free indices {a, b}",
          length(fi_ric) == 2 && Set(fi_ric_names) == Set([:a, :b]))

    # Symmetry check: delta(R_{ab}) and delta(R_{ba}) should have same structure
    delta_Ric_ba = δricci(mp, down(:b), down(:a), 1)
    n_ab = delta_Ric isa TSum ? length(delta_Ric.terms) : 1
    n_ba = delta_Ric_ba isa TSum ? length(delta_Ric_ba.terms) : 1
    check("Linearised Ricci symmetric: same term count for (a,b) and (b,a)",
          n_ab == n_ba)

    # ------------------------------------------------------------------
    # 3. Trace-reversed perturbation
    #    gamma_bar_{ab} = h_{ab} - (1/2) g_{ab} h
    #    where h = g^{cd} h_{cd} is the trace
    #
    #    Trace of gamma_bar: g^{ab} gamma_bar_{ab} = h - (d/2) h = -h
    #    In d=4: trace(gamma_bar) = -trace(h)
    # ------------------------------------------------------------------
    println("\n--- 3. Trace-reversed perturbation ---")

    # Register a trace scalar h_tr for clarity
    # h_{ab}
    h_ab = Tensor(:h, [down(:a), down(:b)])

    # h = g^{cd} h_{cd}  (the trace)
    h_trace = Tensor(:g, [up(:c), up(:d)]) * Tensor(:h, [down(:c), down(:d)])

    # gamma_bar_{ab} = h_{ab} - (1/2) g_{ab} h
    gamma_bar = h_ab - (1 // 2) * Tensor(:g, [down(:a), down(:b)]) * h_trace
    println("  gamma_bar_{ab} = h_{ab} - (1/2) g_{ab} (g^{cd} h_{cd})")

    # Compute trace of gamma_bar: g^{ab} gamma_bar_{ab}
    gamma_bar_trace = Tensor(:g, [up(:a), up(:b)]) * gamma_bar
    gamma_bar_trace_s = simplify(gamma_bar_trace)
    println("  Tr(gamma_bar) = ", to_unicode(gamma_bar_trace_s))

    # Also compute trace of h: g^{ab} h_{ab}
    h_trace_direct = Tensor(:g, [up(:a), up(:b)]) * h_ab
    h_trace_s = simplify(h_trace_direct)
    println("  Tr(h) = ", to_unicode(h_trace_s))

    # Verify: Tr(gamma_bar) + Tr(h) = 0
    sum_traces = simplify(gamma_bar_trace_s + h_trace_s)
    println("  Tr(gamma_bar) + Tr(h) = ", to_unicode(sum_traces))
    check("Tr(gamma_bar) = -Tr(h)", sum_traces == TScalar(0 // 1))

    # ------------------------------------------------------------------
    # 4. Gauge transformation
    #    Under xi^a, the metric perturbation transforms as:
    #    h'_{ab} = h_{ab} + nabla_a xi_b + nabla_b xi_a
    #
    #    Use the gauge_transformation function from the perturbation module.
    # ------------------------------------------------------------------
    println("\n--- 4. Gauge transformation ---")

    # Register a gauge vector xi^a
    register_tensor!(reg, TensorProperties(
        name=:xi, manifold=:M4, rank=(1, 0),
        symmetries=SymmetrySpec[]))

    xi = Tensor(:xi, [up(:c)])

    # Apply gauge transformation to h_{ab}
    h_gauge = gauge_transformation(h_ab, xi, :g; order=1)
    println("  h'_{ab} = ", to_unicode(h_gauge))

    # The result should be h_{ab} + Lie_xi(g_{ab})
    # Lie_xi(g_{ab}) = nabla_a xi_b + nabla_b xi_a (for metric)
    check("Gauge-transformed h_{ab} is a sum", h_gauge isa TSum)
    if h_gauge isa TSum
        # Should have more terms than just h_{ab}
        check("Gauge transformation adds derivative terms",
              length(h_gauge.terms) > 1)
    end

    # Verify free indices are preserved
    fi_gauge = free_indices(h_gauge)
    fi_gauge_names = sort([idx.name for idx in fi_gauge])
    check("Gauge transformation preserves free indices {a, b}",
          Set(fi_gauge_names) == Set([:a, :b]))

    # ------------------------------------------------------------------
    # 5. Linearised Einstein tensor
    #    delta(G_{ab}) = delta(R_{ab}) - (1/2) g_{ab} delta(R)
    # ------------------------------------------------------------------
    println("\n--- 5. Linearised Einstein tensor ---")

    delta_R = δricci_scalar(mp, 1)
    delta_Ein = δricci(mp, down(:a), down(:b), 1) -
                (1 // 2) * Tensor(:g, [down(:a), down(:b)]) * delta_R
    result = simplify(delta_Ein)
    println("  delta(G_{ab}) = ", to_unicode(result))
    check("Linearised Einstein tensor is non-zero", result != TScalar(0 // 1))

    fi_ein = free_indices(result)
    check("Linearised Einstein has free indices {a, b}",
          Set([idx.name for idx in fi_ein]) == Set([:a, :b]))
end

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
println("\n" * "="^70)
total = passed + failed
println("Lecture 14 verification: $passed/$total checks passed")
if failed > 0
    println("WARNING: $failed check(s) failed!")
else
    println("All linearised gravity identities verified!")
end
println("="^70)

@assert failed == 0 "Lecture 14 verification had $failed failure(s)"
