# ============================================================================
# TensorGR.jl — Exterior Calculus
#
# Differential forms, wedge products, exterior derivatives,
# interior products, Hodge dual, and Cartan's magic formula.
# ============================================================================

using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g

    # --- (1) Define differential forms ---
    define_form!(reg, :A; manifold=:M4, degree=1)   # 1-form (e.g., gauge potential)
    define_form!(reg, :F; manifold=:M4, degree=2)   # 2-form (e.g., field strength)
    @define_tensor V on=M4 rank=(1,0)               # vector field

    println("Registered forms: A (1-form), F (2-form)")

    # --- (2) 2-form antisymmetry: F_{ba} = -F_{ab} ---
    F_ab = Tensor(:F, [down(:a), down(:b)])
    F_ba = Tensor(:F, [down(:b), down(:a)])
    antisym = simplify(F_ab + F_ba)
    println("\nF_{ab} + F_{ba} = ", to_unicode(antisym))
    @assert antisym == TScalar(0 // 1)

    # --- (3) Wedge product of two 1-forms ---
    A1 = Tensor(:A, [down(:a)])
    A2 = Tensor(:A, [down(:b)])
    w = wedge(A1, A2, 1, 1)
    println("\nA_a wedge A_b = ", to_unicode(w))
    # Result has coefficient (1+1)!/(1!1!) = 2

    # --- (4) Exterior derivative ---
    dA = exterior_d(A1, 1, down(:b))
    println("dA = ", to_unicode(dA))
    # dA = d_b(A_a) — the exterior derivative of a 1-form

    # --- (5) Interior product: iota_v alpha ---
    v = Tensor(:V, [up(:a)])
    alpha = Tensor(:F, [down(:a), down(:b)])
    iv_alpha = interior_product(v, alpha)
    println("\niota_V F = ", to_unicode(iv_alpha))
    # Contracts V^a with F_{ab} => V^a F_{ab}

    # --- (6) Cartan's magic formula ---
    # Lie derivative: L_v omega = d(iota_v omega) + iota_v(d omega)
    omega = Tensor(:A, [down(:a)])
    cartan = cartan_lie_d(v, omega, 1, down(:b))
    println("\nCartan formula L_V A = d(iota_V A) + iota_V(dA):")
    println("  ", to_unicode(cartan))

    # --- (7) Connection 1-form ---
    @covd D on=M4 metric=g
    omega_form = connection_form(:ΓD, up(:a), down(:b), down(:c))
    println("\nConnection 1-form omega^a_b = Gamma^a_{cb} dx^c:")
    println("  ", to_unicode(omega_form))

    # --- (8) Curvature 2-form via second structure equation ---
    curv_form = curvature_form(:ΓD, up(:a), down(:b), down(:c), down(:d))
    println("\nCurvature 2-form Omega^a_b:")
    println("  ", to_unicode(curv_form))

    # --- (9) First Cartan structure equation (torsion) ---
    @define_tensor theta on=M4 rank=(1,1)  # coframe
    torsion = cartan_first_structure(:T, :ΓD, :theta, up(:a), down(:b), down(:c))
    println("\nTorsion 2-form T^a = d(theta^a) + omega^a_b ^ theta^b:")
    println("  ", to_unicode(torsion))

    println("\nAll exterior calculus examples completed!")
end
