# ============================================================================
# TensorGR.jl — Covariant Derivatives
#
# Define a covariant derivative, expand into Christoffel symbols,
# commute derivatives to produce Riemann curvature, and verify
# the contracted Bianchi identity: nabla^a G_{ab} = 0.
# ============================================================================

using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor V on=M4 rank=(1,0)
    @define_tensor W on=M4 rank=(0,1)
    @covd D on=M4 metric=g

    # --- (1) Expand nabla_a V^b into partial + Christoffel ---
    nabla_V = TDeriv(down(:a), Tensor(:V, [up(:b)]))
    expanded = covd_to_christoffel(nabla_V, :D)
    println("nabla_a V^b = ", to_unicode(expanded))
    # Result: d_a V^b + Gamma^b_{ac} V^c

    # --- (2) Expand nabla_a W_b ---
    nabla_W = TDeriv(down(:a), Tensor(:W, [down(:b)]))
    expanded_W = covd_to_christoffel(nabla_W, :D)
    println("nabla_a W_b = ", to_unicode(expanded_W))
    # Result: d_a W_b - Gamma^c_{ab} W_c

    # --- (3) Christoffel in terms of metric gradients ---
    christoffel_expr = christoffel_to_grad_metric(:g, up(:a), down(:b), down(:c))
    println("\nGamma^a_{bc} = ", to_unicode(christoffel_expr))

    # --- (4) Commutator of covariant derivatives yields Riemann ---
    # [nabla_a, nabla_b] V^c = R^c_{dab} V^d (schematic)
    # Using commute_covds to sort nested derivatives:
    double_deriv = TDeriv(down(:b), TDeriv(down(:a), Tensor(:V, [up(:c)])))
    sorted = commute_covds(double_deriv, :D)
    println("\nAfter commuting nabla_b nabla_a V^c:")
    println("  ", to_unicode(sorted))

    # --- (5) Contracted Bianchi: nabla^a G_{ab} = 0 ---
    for r in bianchi_rules()
        register_rule!(reg, r)
    end
    bianchi_expr = TDeriv(up(:a), Tensor(:Ein, [down(:a), down(:b)]))
    bianchi_result = simplify(bianchi_expr)
    println("\nnabla^a G_{ab} = ", to_unicode(bianchi_result))
    @assert bianchi_result == TScalar(0 // 1)

    # --- (6) Contracted Bianchi for Ricci: nabla^a R_{ab} = (1/2) nabla_b R ---
    ricci_bianchi = TDeriv(up(:a), Tensor(:Ric, [down(:a), down(:b)]))
    ricci_result = simplify(ricci_bianchi)
    println("nabla^a R_{ab} = ", to_unicode(ricci_result))

    println("\nAll checks passed!")
end
