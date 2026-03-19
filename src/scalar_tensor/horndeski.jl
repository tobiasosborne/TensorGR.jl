#= Horndeski scalar-tensor theory Lagrangian construction.

Implements the four Horndeski Lagrangians L_2 through L_5 as abstract tensor
expressions, following Kobayashi (2019) arXiv:1901.04778, Eqs 5-8:
  L_2 = G_2(phi, X)
  L_3 = -G_3(phi, X) Box(phi)
  L_4 = G_4 R + G_{4,X} [(Box phi)^2 - (nabla_a nabla_b phi)^2]
  L_5 = G_5 G_{ab} nabla^a nabla^b phi
        - (1/6) G_{5,X} [(Box phi)^3 - 3 Box(phi)(nabla_a nabla_b phi)^2
                          + 2(nabla_a nabla_b phi)^3]

where X = -(1/2) g^{ab} d_a phi d_b phi is the kinetic term.
=#

# ── ScalarTensorFunction ────────────────────────────────────────────

"""
    ScalarTensorFunction(name, phi_derivs, X_derivs)

Represents an abstract function G_i(phi, X) and its derivatives.
`phi_derivs` and `X_derivs` count the number of differentiations w.r.t.
phi and X respectively.
"""
struct ScalarTensorFunction
    name::Symbol        # base name, e.g. :G2, :G3, :G4, :G5
    phi_derivs::Int     # number of phi-derivatives taken
    X_derivs::Int       # number of X-derivatives taken
end

"""
    g_tensor_name(stf::ScalarTensorFunction) -> Symbol

Return the registry tensor name for this function, e.g. :G4_X, :G4_phiX.
"""
function g_tensor_name(stf::ScalarTensorFunction)
    base = string(stf.name)
    suffix = ""
    if stf.phi_derivs > 0
        suffix *= join(fill("phi", stf.phi_derivs))
    end
    if stf.X_derivs > 0
        suffix *= join(fill("X", stf.X_derivs))
    end
    isempty(suffix) ? Symbol(base) : Symbol(base, "_", suffix)
end

"""
    differentiate_G(stf::ScalarTensorFunction, var::Symbol) -> ScalarTensorFunction

Differentiate the scalar-tensor function with respect to :phi or :X.
"""
function differentiate_G(stf::ScalarTensorFunction, var::Symbol)
    if var == :X
        ScalarTensorFunction(stf.name, stf.phi_derivs, stf.X_derivs + 1)
    elseif var == :phi
        ScalarTensorFunction(stf.name, stf.phi_derivs + 1, stf.X_derivs)
    else
        error("differentiate_G: variable must be :phi or :X, got $var")
    end
end

# ── HorndeskiTheory ─────────────────────────────────────────────────

"""
    HorndeskiTheory

Container linking manifold, metric, scalar field, and the four G-functions.
"""
struct HorndeskiTheory
    manifold::Symbol
    metric::Symbol
    scalar_field::Symbol
    G_functions::NTuple{4, ScalarTensorFunction}
    covd::Symbol
end

# ── Registration ────────────────────────────────────────────────────

"""
    define_horndeski!(reg; manifold, metric, scalar_field=:phi, covd=:nabla)

Register the scalar field and all required G-functions as rank-0 tensors.
Returns a `HorndeskiTheory` struct.
"""
function define_horndeski!(reg::TensorRegistry;
                           manifold::Symbol, metric::Symbol,
                           scalar_field::Symbol=:phi,
                           covd::Symbol=:nabla)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    # Register scalar field as rank-0 tensor (if not already present)
    if !has_tensor(reg, scalar_field)
        register_tensor!(reg, TensorProperties(
            name=scalar_field, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}()))
    end

    # Base G-functions
    G2 = ScalarTensorFunction(:G2, 0, 0)
    G3 = ScalarTensorFunction(:G3, 0, 0)
    G4 = ScalarTensorFunction(:G4, 0, 0)
    G5 = ScalarTensorFunction(:G5, 0, 0)

    # All needed derivatives: G2, G3, G3_phi, G4, G4_X, G4_phi,
    #                         G5, G5_X, G5_phi, G5_XX
    needed = [
        G2,
        G3, differentiate_G(G3, :phi),
        G4, differentiate_G(G4, :X), differentiate_G(G4, :phi),
        G5, differentiate_G(G5, :X), differentiate_G(G5, :phi),
        differentiate_G(differentiate_G(G5, :X), :X),
    ]

    for stf in needed
        tname = g_tensor_name(stf)
        if !has_tensor(reg, tname)
            register_tensor!(reg, TensorProperties(
                name=tname, manifold=manifold, rank=(0, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:is_scalar_tensor_function => true,
                                         :stf_base => stf.name,
                                         :stf_phi_derivs => stf.phi_derivs,
                                         :stf_X_derivs => stf.X_derivs)))
        end
    end

    # Define covariant derivative if not already present
    if !has_tensor(reg, covd)
        define_covd!(reg, covd; manifold=manifold, metric=metric)
    end

    HorndeskiTheory(manifold, metric, scalar_field,
                    (G2, G3, G4, G5), covd)
end

# ── Kinetic term ────────────────────────────────────────────────────

"""
    kinetic_X(scalar_field, metric; registry) -> TensorExpr

Kinetic term X = -(1/2) g^{ab} d_a(phi) d_b(phi).
"""
function kinetic_X(scalar_field::Symbol, metric::Symbol;
                   registry::TensorRegistry=current_registry())
    phi = Tensor(scalar_field, TIndex[])
    (-1 // 2) * grad_squared(phi, metric; registry=registry)
end

# ── Lagrangian building blocks ──────────────────────────────────────

"""
    horndeski_L2(ht::HorndeskiTheory; registry) -> TensorExpr

L_2 = G_2(phi, X).  Kobayashi (2019) Eq 5.
"""
function horndeski_L2(ht::HorndeskiTheory;
                      registry::TensorRegistry=current_registry())
    Tensor(:G2, TIndex[])
end

"""
    horndeski_L3(ht::HorndeskiTheory; registry) -> TensorExpr

L_3 = -G_3(phi, X) Box(phi).  Kobayashi (2019) Eq 6.
"""
function horndeski_L3(ht::HorndeskiTheory;
                      registry::TensorRegistry=current_registry())
    phi = Tensor(ht.scalar_field, TIndex[])
    box_phi = box(phi, ht.metric; registry=registry)
    (-1 // 1) * Tensor(:G3, TIndex[]) * box_phi
end

"""
    horndeski_L4(ht::HorndeskiTheory; registry) -> TensorExpr

L_4 = G_4 R + G_{4,X} [(Box phi)^2 - (nabla_a nabla_b phi)(nabla^a nabla^b phi)].
Kobayashi (2019) Eq 7.
"""
function horndeski_L4(ht::HorndeskiTheory;
                      registry::TensorRegistry=current_registry())
    with_registry(registry) do
        phi = Tensor(ht.scalar_field, TIndex[])
        R = Tensor(:RicScalar, TIndex[])
        G4 = Tensor(:G4, TIndex[])
        G4X = Tensor(:G4_X, TIndex[])

        # Box(phi)
        box_phi = box(phi, ht.metric; registry=registry)

        # (Box phi)^2 = box_phi * box_phi (need fresh indices for second copy)
        box_phi2 = box(phi, ht.metric; registry=registry)

        # (nabla_a nabla_b phi)(nabla^a nabla^b phi)
        # = g^{ac} g^{bd} (d_c d_d phi)(d_a d_b phi)
        # We build two copies of d_a d_b phi and contract with metrics
        used = Set{Symbol}()
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)

        dd_phi_1 = covd_chain(phi, [down(a), down(b)])
        dd_phi_2 = covd_chain(phi, [down(c), down(d)])
        g_ac = Tensor(ht.metric, [up(a), up(c)])
        g_bd = Tensor(ht.metric, [up(b), up(d)])
        nabla_sq = g_ac * g_bd * dd_phi_1 * dd_phi_2

        # L_4 = G4 * R + G4_X * [(Box phi)^2 - nabla_sq]
        G4 * R + G4X * (box_phi * box_phi2 - nabla_sq)
    end
end

"""
    horndeski_L5(ht::HorndeskiTheory; registry) -> TensorExpr

L_5 = G_5 G_{ab} nabla^a nabla^b phi
      - (1/6) G_{5,X} [(Box phi)^3 - 3 Box(phi)(nabla_a nabla_b phi)^2
                        + 2(nabla_a nabla_b phi)^3].
Kobayashi (2019) Eq 8.
"""
function horndeski_L5(ht::HorndeskiTheory;
                      registry::TensorRegistry=current_registry())
    with_registry(registry) do
        phi = Tensor(ht.scalar_field, TIndex[])
        G5 = Tensor(:G5, TIndex[])
        G5X = Tensor(:G5_X, TIndex[])

        # --- First term: G_5 G_{ab} nabla^a nabla^b phi ---
        used = Set{Symbol}()
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)

        Ein_ab = Tensor(:Ein, [up(a), up(b)])
        dd_phi_ab = covd_chain(phi, [down(a), down(b)])
        term1 = G5 * Ein_ab * dd_phi_ab

        # --- Second term: -(1/6) G_{5,X} * [cubic combination] ---

        # (Box phi)^3: need 3 independent copies
        box1 = box(phi, ht.metric; registry=registry)
        box2 = box(phi, ht.metric; registry=registry)
        box3 = box(phi, ht.metric; registry=registry)
        cube_box = box1 * box2 * box3

        # Box(phi) * (nabla_a nabla_b phi)^2
        box_for_sq = box(phi, ht.metric; registry=registry)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)
        e = fresh_index(used); push!(used, e)
        f = fresh_index(used); push!(used, f)
        dd1 = covd_chain(phi, [down(c), down(d)])
        dd2 = covd_chain(phi, [down(e), down(f)])
        g_ce = Tensor(ht.metric, [up(c), up(e)])
        g_df = Tensor(ht.metric, [up(d), up(f)])
        sq_term = box_for_sq * g_ce * g_df * dd1 * dd2

        # (nabla_a nabla_b phi)^3
        # = nabla_a nabla^b phi nabla_b nabla^c phi nabla_c nabla^a phi
        # = g^{ad} g^{be} g^{cf} (d_d d_b phi)(d_e d_c phi)(d_f d_a phi)
        # We need 6 fresh indices for the triple contraction
        i1 = fresh_index(used); push!(used, i1)
        i2 = fresh_index(used); push!(used, i2)
        i3 = fresh_index(used); push!(used, i3)
        i4 = fresh_index(used); push!(used, i4)
        i5 = fresh_index(used); push!(used, i5)
        i6 = fresh_index(used); push!(used, i6)

        # nabla_{i1} nabla_{i2} phi  (with i1 contracted up via g^{i4,i1})
        # nabla_{i2} nabla_{i3} phi  ...  → trace structure
        # Actually: (∇_a ∇^b φ)(∇_b ∇^c φ)(∇_c ∇^a φ)
        # = g^{i1,i4} g^{i2,i5} g^{i3,i6} (∂_{i4}∂_{i2} φ)(∂_{i5}∂_{i3} φ)(∂_{i6}∂_{i1} φ)
        dd_A = covd_chain(phi, [down(i4), down(i2)])
        dd_B = covd_chain(phi, [down(i5), down(i3)])
        dd_C = covd_chain(phi, [down(i6), down(i1)])
        g1 = Tensor(ht.metric, [up(i1), up(i4)])
        g2 = Tensor(ht.metric, [up(i2), up(i5)])
        g3 = Tensor(ht.metric, [up(i3), up(i6)])
        triple_term = g1 * g2 * g3 * dd_A * dd_B * dd_C

        term2 = (-1 // 6) * G5X * (cube_box - (3 // 1) * sq_term + (2 // 1) * triple_term)

        term1 + term2
    end
end

"""
    horndeski_lagrangian(ht::HorndeskiTheory; registry) -> TensorExpr

Full Horndeski Lagrangian L = L_2 + L_3 + L_4 + L_5.
"""
function horndeski_lagrangian(ht::HorndeskiTheory;
                              registry::TensorRegistry=current_registry())
    horndeski_L2(ht; registry=registry) +
    horndeski_L3(ht; registry=registry) +
    horndeski_L4(ht; registry=registry) +
    horndeski_L5(ht; registry=registry)
end
