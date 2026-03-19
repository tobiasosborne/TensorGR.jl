#= Beyond-Horndeski (GLPV) scalar-tensor theory extensions.

Gleyzes-Langlois-Piazza-Vernizzi (GLPV, 2015) extended Horndeski theory with
two additional Lagrangians that yield higher-order EOMs but still propagate
only 3 DOF (1 scalar + 2 tensor) via degeneracy conditions:

  L_4^{bH} = F_4(phi, X) epsilon^{abce} epsilon^{dfg}_{e} nabla_a(phi)
              nabla_d(phi) nabla_b nabla_f(phi) nabla_c nabla_g(phi)

  L_5^{bH} = F_5(phi, X) epsilon^{abcd} epsilon^{efgh} nabla_a(phi)
              nabla_e(phi) nabla_b nabla_f(phi) nabla_c nabla_g(phi)
              nabla_d nabla_h(phi)

Using the epsilon identity these become the expanded forms (Kobayashi 2019,
Eqs 2.13-2.14):

  L_4^{bH} = F_4 [ (Box phi)^2 - (dd phi)^2 ] X
            - F_4 [ Box phi  dphi^a dphi^b dd_ab phi
                   - dphi^a dd_ab phi dphi^c dd^b_c phi ]
            (expanded determinant form; see implementation below)

Ground truth: Gleyzes et al, PRL 114, 211101 (2015);
              Kobayashi arXiv:1901.04778, Sec 2.2, Eqs 2.13-2.14.
=#

# ── BeyondHorndeskiTheory ──────────────────────────────────────────

"""
    BeyondHorndeskiTheory

Container extending `HorndeskiTheory` with the two beyond-Horndeski
free functions F_4(phi, X) and F_5(phi, X).
"""
struct BeyondHorndeskiTheory
    horndeski::HorndeskiTheory
    F4::ScalarTensorFunction
    F5::ScalarTensorFunction
end

# ── Registration ────────────────────────────────────────────────────

"""
    define_beyond_horndeski!(reg, ht::HorndeskiTheory;
                              F4=:F4, F5=:F5) -> BeyondHorndeskiTheory

Register the beyond-Horndeski free functions F_4 and F_5 (and their
X-derivatives F4_X, F5_X) as rank-0 tensors, then return a
`BeyondHorndeskiTheory` wrapping the given Horndeski theory.
"""
function define_beyond_horndeski!(reg::TensorRegistry,
                                   ht::HorndeskiTheory;
                                   F4::Symbol=:F4,
                                   F5::Symbol=:F5)
    F4_stf = ScalarTensorFunction(F4, 0, 0)
    F5_stf = ScalarTensorFunction(F5, 0, 0)

    needed = [
        F4_stf,
        differentiate_G(F4_stf, :X),   # F4_X
        F5_stf,
        differentiate_G(F5_stf, :X),   # F5_X
    ]

    for stf in needed
        tname = g_tensor_name(stf)
        if !has_tensor(reg, tname)
            register_tensor!(reg, TensorProperties(
                name=tname, manifold=ht.manifold, rank=(0, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:is_scalar_tensor_function => true,
                                         :stf_base => stf.name,
                                         :stf_phi_derivs => stf.phi_derivs,
                                         :stf_X_derivs => stf.X_derivs)))
        end
    end

    BeyondHorndeskiTheory(ht, F4_stf, F5_stf)
end

# ── L_4^{bH} Lagrangian ────────────────────────────────────────────

"""
    beyond_horndeski_L4(bht::BeyondHorndeskiTheory; registry) -> TensorExpr

Construct the beyond-Horndeski L_4^{bH} Lagrangian using the expanded
determinant form (Kobayashi 2019, Eq 2.13):

  L_4^{bH} = F_4 * epsilon^{abce} epsilon_{dfge}
              nabla_a phi nabla^d phi nabla_b nabla^f phi nabla_c nabla^g phi

Using epsilon^{abce} epsilon_{dfge} = -3! delta^{[a}_d delta^b_f delta^{c]}_g,
the expanded form is:

  L_4^{bH} = F_4 [ 2(Box phi)^2 (dphi)^2
              - 2(Box phi)(dphi^a dd_{ab} dphi^b)
              - (dd phi)^2 (dphi)^2
              + 2(dphi^a dd_{ac} dd^{cb} dphi_b) ]

where (dphi)^2 = nabla_a phi nabla^a phi = -2X.
"""
function beyond_horndeski_L4(bht::BeyondHorndeskiTheory;
                              registry::TensorRegistry=current_registry())
    ht = bht.horndeski
    with_registry(registry) do
        phi = Tensor(ht.scalar_field, TIndex[])
        F4t = Tensor(g_tensor_name(bht.F4), TIndex[])

        used = Set{Symbol}()
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)
        e = fresh_index(used); push!(used, e)
        f = fresh_index(used); push!(used, f)

        # (dphi)^2 = g^{ab} d_a phi d_b phi = -2X
        dphi_sq = Tensor(ht.metric, [up(a), up(b)]) *
                  TDeriv(down(a), phi) * TDeriv(down(b), phi)

        # Box phi
        box_phi1 = box(phi, ht.metric; registry=registry)
        box_phi2 = box(phi, ht.metric; registry=registry)

        # (Box phi)^2 (dphi)^2
        term1 = (2 // 1) * box_phi1 * box_phi2 * dphi_sq

        # (dd phi)^2 = g^{ac} g^{bd} dd_{ab} dd_{cd}
        dd_ab = covd_chain(phi, [down(c), down(d)])
        dd_cd = covd_chain(phi, [down(e), down(f)])
        nabla_sq = Tensor(ht.metric, [up(c), up(e)]) *
                   Tensor(ht.metric, [up(d), up(f)]) * dd_ab * dd_cd

        # -(dd phi)^2 (dphi)^2
        # Need fresh indices for dphi_sq copy
        a2 = fresh_index(used); push!(used, a2)
        b2 = fresh_index(used); push!(used, b2)
        dphi_sq2 = Tensor(ht.metric, [up(a2), up(b2)]) *
                   TDeriv(down(a2), phi) * TDeriv(down(b2), phi)
        term2 = (-1 // 1) * nabla_sq * dphi_sq2

        # Box(phi) dphi^a dd_{ab} dphi^b
        # = Box(phi) g^{ac} d_c phi dd_{ab} g^{bd} d_d phi
        c2 = fresh_index(used); push!(used, c2)
        d2 = fresh_index(used); push!(used, d2)
        e2 = fresh_index(used); push!(used, e2)
        f2 = fresh_index(used); push!(used, f2)
        box_phi3 = box(phi, ht.metric; registry=registry)
        dphi_up_c2 = Tensor(ht.metric, [up(c2), up(d2)]) * TDeriv(down(d2), phi)
        dphi_up_f2 = Tensor(ht.metric, [up(e2), up(f2)]) * TDeriv(down(f2), phi)
        dd_ce = covd_chain(phi, [down(c2), down(e2)])
        term3 = (-2 // 1) * box_phi3 * dphi_up_c2 * dd_ce * dphi_up_f2

        # dphi^a dd_{ac} dd^{cb} dphi_b
        # = g^{ag} d_g phi  dd_{ac}  g^{ch} dd_{hb}  g^{bi} d_i phi
        g1 = fresh_index(used); push!(used, g1)
        h1 = fresh_index(used); push!(used, h1)
        i1 = fresh_index(used); push!(used, i1)
        j1 = fresh_index(used); push!(used, j1)
        k1 = fresh_index(used); push!(used, k1)
        l1 = fresh_index(used); push!(used, l1)

        dphi_up_g = Tensor(ht.metric, [up(g1), up(h1)]) * TDeriv(down(h1), phi)
        dd_gk = covd_chain(phi, [down(g1), down(i1)])
        dd_jl = covd_chain(phi, [down(j1), down(k1)])
        g_ij = Tensor(ht.metric, [up(i1), up(j1)])
        dphi_up_l = Tensor(ht.metric, [up(k1), up(l1)]) * TDeriv(down(l1), phi)
        term4 = (2 // 1) * dphi_up_g * dd_gk * g_ij * dd_jl * dphi_up_l

        F4t * (term1 + term2 + term3 + term4)
    end
end

# ── L_5^{bH} Lagrangian ────────────────────────────────────────────

"""
    beyond_horndeski_L5(bht::BeyondHorndeskiTheory; registry) -> TensorExpr

Construct the beyond-Horndeski L_5^{bH} Lagrangian using the expanded
determinant form (Kobayashi 2019, Eq 2.14):

  L_5^{bH} = F_5 * epsilon^{abcd} epsilon_{efgh}
              nabla^e phi nabla_a phi nabla_b nabla^f phi
              nabla_c nabla^g phi nabla_d nabla^h phi

Using epsilon^{abcd} epsilon_{efgh} = -4! delta^{[a}_e delta^b_f delta^c_g delta^{d]}_h,
the expanded form is:

  L_5^{bH} = F_5 (dphi)^2 [ (Box phi)^3 - 3 Box(phi)(dd phi)^2 + 2(dd phi)^3 ]
            - 3 F_5 [ (Box phi)^2 dphi^a dd_{ab} dphi^b
                      - (dd phi)^2 dphi^a dd_{ab} dphi^b
                      - 2 Box(phi) dphi^a dd_{ac} dd^{cb} dphi_b
                      + 2 dphi^a dd_{ac} dd^{cd} dd_{db} dphi^b ]
"""
function beyond_horndeski_L5(bht::BeyondHorndeskiTheory;
                              registry::TensorRegistry=current_registry())
    ht = bht.horndeski
    with_registry(registry) do
        phi = Tensor(ht.scalar_field, TIndex[])
        F5t = Tensor(g_tensor_name(bht.F5), TIndex[])

        used = Set{Symbol}()

        # --- (dphi)^2 piece times cubic combination ---

        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        dphi_sq = Tensor(ht.metric, [up(a), up(b)]) *
                  TDeriv(down(a), phi) * TDeriv(down(b), phi)

        # (Box phi)^3
        box1 = box(phi, ht.metric; registry=registry)
        box2 = box(phi, ht.metric; registry=registry)
        box3 = box(phi, ht.metric; registry=registry)
        cube_box = box1 * box2 * box3

        # (dd phi)^2 with fresh indices
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)
        e = fresh_index(used); push!(used, e)
        f = fresh_index(used); push!(used, f)
        dd_cd = covd_chain(phi, [down(c), down(d)])
        dd_ef = covd_chain(phi, [down(e), down(f)])
        nabla_sq = Tensor(ht.metric, [up(c), up(e)]) *
                   Tensor(ht.metric, [up(d), up(f)]) * dd_cd * dd_ef

        box_for_sq = box(phi, ht.metric; registry=registry)

        # (dd phi)^3 = dd_a^b dd_b^c dd_c^a
        i1 = fresh_index(used); push!(used, i1)
        i2 = fresh_index(used); push!(used, i2)
        i3 = fresh_index(used); push!(used, i3)
        i4 = fresh_index(used); push!(used, i4)
        i5 = fresh_index(used); push!(used, i5)
        i6 = fresh_index(used); push!(used, i6)
        dd_A = covd_chain(phi, [down(i4), down(i2)])
        dd_B = covd_chain(phi, [down(i5), down(i3)])
        dd_C = covd_chain(phi, [down(i6), down(i1)])
        triple_term = Tensor(ht.metric, [up(i1), up(i4)]) *
                      Tensor(ht.metric, [up(i2), up(i5)]) *
                      Tensor(ht.metric, [up(i3), up(i6)]) *
                      dd_A * dd_B * dd_C

        cubic = cube_box - (3 // 1) * box_for_sq * nabla_sq + (2 // 1) * triple_term

        # First group: (dphi)^2 * cubic
        group1 = dphi_sq * cubic

        # --- Terms with dphi^a (...) dphi_b structure ---
        # These arise from the cross-terms in the 4! antisymmetrization.
        # For a concise implementation, we use the structure:
        #   -3 [(Box phi)^2 - (dd phi)^2] dphi^a dd_{ab} dphi^b
        #   +6 [Box(phi) dphi^a dd_{ac} dd^{cb} dphi_b
        #       - dphi^a dd_{ac} dd^{cd} dd_{db} dphi^b]

        # [(Box phi)^2 - (dd phi)^2] dphi^a dd_{ab} dphi^b
        j1 = fresh_index(used); push!(used, j1)
        j2 = fresh_index(used); push!(used, j2)
        j3 = fresh_index(used); push!(used, j3)
        j4 = fresh_index(used); push!(used, j4)
        j5 = fresh_index(used); push!(used, j5)
        j6 = fresh_index(used); push!(used, j6)

        box4 = box(phi, ht.metric; registry=registry)
        box5 = box(phi, ht.metric; registry=registry)

        dphi_up_j1 = Tensor(ht.metric, [up(j1), up(j2)]) * TDeriv(down(j2), phi)
        dphi_up_j3 = Tensor(ht.metric, [up(j3), up(j4)]) * TDeriv(down(j4), phi)
        dd_j1j3 = covd_chain(phi, [down(j1), down(j3)])

        # (dd phi)^2 fresh copy for this group
        dd_j5j6_a = covd_chain(phi, [down(j5), down(j6)])
        k1 = fresh_index(used); push!(used, k1)
        k2 = fresh_index(used); push!(used, k2)
        dd_k1k2 = covd_chain(phi, [down(k1), down(k2)])
        nabla_sq2 = Tensor(ht.metric, [up(j5), up(k1)]) *
                    Tensor(ht.metric, [up(j6), up(k2)]) * dd_j5j6_a * dd_k1k2

        cross1 = (box4 * box5 - nabla_sq2) * dphi_up_j1 * dd_j1j3 * dphi_up_j3

        # Box(phi) dphi^a dd_{ac} dd^{cb} dphi_b
        m1 = fresh_index(used); push!(used, m1)
        m2 = fresh_index(used); push!(used, m2)
        m3 = fresh_index(used); push!(used, m3)
        m4 = fresh_index(used); push!(used, m4)
        m5 = fresh_index(used); push!(used, m5)
        m6 = fresh_index(used); push!(used, m6)

        box6 = box(phi, ht.metric; registry=registry)
        dphi_up_m1 = Tensor(ht.metric, [up(m1), up(m2)]) * TDeriv(down(m2), phi)
        dd_m1m3 = covd_chain(phi, [down(m1), down(m3)])
        dd_m4m5 = covd_chain(phi, [down(m4), down(m5)])
        g_m3m4 = Tensor(ht.metric, [up(m3), up(m4)])
        dphi_up_m5 = Tensor(ht.metric, [up(m5), up(m6)]) * TDeriv(down(m6), phi)

        cross2 = box6 * dphi_up_m1 * dd_m1m3 * g_m3m4 * dd_m4m5 * dphi_up_m5

        # dphi^a dd_{ac} dd^{cd} dd_{db} dphi^b
        n1 = fresh_index(used); push!(used, n1)
        n2 = fresh_index(used); push!(used, n2)
        n3 = fresh_index(used); push!(used, n3)
        n4 = fresh_index(used); push!(used, n4)
        n5 = fresh_index(used); push!(used, n5)
        n6 = fresh_index(used); push!(used, n6)
        n7 = fresh_index(used); push!(used, n7)
        n8 = fresh_index(used); push!(used, n8)

        dphi_up_n1 = Tensor(ht.metric, [up(n1), up(n2)]) * TDeriv(down(n2), phi)
        dd_n1n3 = covd_chain(phi, [down(n1), down(n3)])
        g_n3n4 = Tensor(ht.metric, [up(n3), up(n4)])
        dd_n4n5 = covd_chain(phi, [down(n4), down(n5)])
        g_n5n6 = Tensor(ht.metric, [up(n5), up(n6)])
        dd_n6n7 = covd_chain(phi, [down(n6), down(n7)])
        dphi_up_n7 = Tensor(ht.metric, [up(n7), up(n8)]) * TDeriv(down(n8), phi)

        cross3 = dphi_up_n1 * dd_n1n3 * g_n3n4 * dd_n4n5 * g_n5n6 * dd_n6n7 * dphi_up_n7

        group2 = (-3 // 1) * cross1 + (6 // 1) * cross2 - (6 // 1) * cross3

        F5t * (group1 + group2)
    end
end

# ── Full beyond-Horndeski Lagrangian ───────────────────────────────

"""
    beyond_horndeski_lagrangian(bht::BeyondHorndeskiTheory; registry) -> TensorExpr

Full beyond-Horndeski Lagrangian: L_Horn + L_4^{bH} + L_5^{bH}.
"""
function beyond_horndeski_lagrangian(bht::BeyondHorndeskiTheory;
                                      registry::TensorRegistry=current_registry())
    horndeski_lagrangian(bht.horndeski; registry=registry) +
    beyond_horndeski_L4(bht; registry=registry) +
    beyond_horndeski_L5(bht; registry=registry)
end

# ── alpha_H for beyond-Horndeski ───────────────────────────────────

"""
    alpha_H(bht::BeyondHorndeskiTheory, bg::FRWBackground;
            registry=current_registry()) -> Any

Compute the beyond-Horndeski parameter alpha_H on an FRW background.

alpha_H is nonzero only for beyond-Horndeski theories (F_4 != 0 or F_5 != 0).
For pure Horndeski, alpha_H = 0.

Ground truth: Gleyzes et al (2015); Kobayashi (2019) Sec 5.3, Eqs 5.12-5.13:

  alpha_H = (2X / M_*^2) * [ 2 F_4 dot{phi}^2 + F_5 dot{phi}^3 H ]

where M_*^2 includes the beyond-Horndeski correction:

  M_*^2 = M_*^2|_{Horn} + 2X [ 4 F_4 X - F_5 dot{phi} H X ]
"""
function alpha_H(bht::BeyondHorndeskiTheory, bg::FRWBackground;
                  registry::TensorRegistry=current_registry())
    # Register needed functions
    _register_alpha_functions!(registry, bht.horndeski)

    H = bg.H
    pd = bg.phi_dot
    X = bg.X_bg

    mul = _sym_mul
    add = _sym_add
    sub = _sym_sub

    F4_name = g_tensor_name(bht.F4)
    F5_name = g_tensor_name(bht.F5)

    # M_*^2 including beyond-Horndeski corrections:
    # M_*^2 = M_*^2|_Horn + 2X [4 F_4 X - F_5 pd H X]
    # First get Horndeski M_*^2
    alphas_horn = compute_alphas(bht.horndeski, bg; registry=registry)
    M_star_sq_horn = alphas_horn.M_star_sq

    # Beyond-Horndeski correction
    bh_correction = mul(mul(2, X),
        sub(mul(4, mul(_G(F4_name), X)),
            mul(_G(F5_name), mul(pd, mul(H, X)))))
    M_star_sq = add(M_star_sq_horn, bh_correction)

    # alpha_H = (2X / M_*^2) * [2 F_4 pd^2 + F_5 pd^3 H]
    pd2 = mul(pd, pd)
    pd3 = mul(pd2, pd)
    bracket = add(mul(2, mul(_G(F4_name), pd2)),
                  mul(_G(F5_name), mul(pd3, H)))
    _sym_div(mul(mul(2, X), bracket), M_star_sq)
end
