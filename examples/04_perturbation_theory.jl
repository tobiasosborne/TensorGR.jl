# ============================================================================
# TensorGR.jl — Perturbation Theory
#
# Linearized gravity: expand metric perturbation g = g0 + eps*h,
# compute delta(g^{ab}), delta(Christoffel), delta(Ricci), and
# verify the structure of linearized Einstein equations.
# ============================================================================

using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)

    # --- (1) Define metric perturbation g -> g + eps*h ---
    mp = define_metric_perturbation!(reg, :g, :h)

    # --- (2) First-order perturbation of inverse metric ---
    # delta(g^{ab}) = -g^{ac} g^{bd} h_{cd}
    delta_ginv = δinverse_metric(mp, up(:a), up(:b), 1)
    println("delta(g^{ab}) = ", to_unicode(delta_ginv))
    @assert delta_ginv isa TProduct
    @assert delta_ginv.scalar == -1 // 1
    println("  (confirmed: -g^{ac} g^{bd} h_{cd})")

    # --- (3) Second-order perturbation of inverse metric ---
    delta2_ginv = δinverse_metric(mp, up(:a), up(:b), 2)
    println("\ndelta^2(g^{ab}) = ", to_unicode(delta2_ginv))

    # --- (4) First-order Christoffel perturbation ---
    delta_gamma = δchristoffel(mp, up(:a), down(:b), down(:c), 1)
    println("\ndelta(Gamma^a_{bc}) = ", to_unicode(delta_gamma))
    # Should be: (1/2) g^{ad} (d_b h_{cd} + d_c h_{bd} - d_d h_{bc})

    # --- (5) First-order Ricci perturbation ---
    delta_ricci = δricci(mp, down(:a), down(:b), 1)
    println("\ndelta(R_{ab}) at order 1:")
    println("  ", to_unicode(delta_ricci))

    # --- (6) Perturbation of the Ricci scalar ---
    delta_R = δricci_scalar(mp, 1)
    println("\ndelta(R) at order 1:")
    println("  ", to_unicode(delta_R))

    # --- (7) Expand perturbation of a general tensor expression ---
    # Perturb g_{ab} at order 1: should give h_{ab}
    g_ab = Tensor(:g, [down(:a), down(:b)])
    pert_g = expand_perturbation(g_ab, mp, 1)
    println("\ndelta(g_{ab}) = ", to_unicode(pert_g))
    @assert pert_g isa Tensor && pert_g.name == :h

    # Perturb g_{ab} at order 0: should give g_{ab}
    bg_g = expand_perturbation(g_ab, mp, 0)
    @assert bg_g == g_ab

    # --- (8) Background field equation: set Ric=0 (vacuum) ---
    background_solution!(reg, [:Ric, :RicScalar, :Ein])

    # After setting vacuum background, Ric simplifies to zero
    ric_expr = Tensor(:Ric, [down(:a), down(:b)])
    ric_simplified = simplify(ric_expr)
    println("\nOn vacuum background: R_{ab} = ", to_unicode(ric_simplified))
    @assert ric_simplified == TScalar(0 // 1)

    println("\nAll perturbation theory checks passed!")
end
