#= Horndeski field equations (metric + scalar EOMs).

Implements the equations of motion obtained by varying the Horndeski action
  S = integral sqrt(-g) (L_2 + L_3 + L_4 + L_5) d^4x
with respect to g^{ab} (metric EOM) and phi (scalar EOM).

Ground truth: Kobayashi (2019) arXiv:1901.04778, Eqs 2.5-2.7.
Key property: despite L containing second derivatives of phi, the EOMs
are SECOND ORDER in both g_{ab} and phi (Horndeski's theorem).

Approach: direct construction from the known closed-form expressions
(Approach 2 from the spec), since variational derivatives of L_4 and L_5
w.r.t. the metric involve intricate Gauss-Bonnet type terms.
=#

# ── Additional G-function derivatives for EOM ──────────────────────

"""
    _register_eom_functions!(reg, ht::HorndeskiTheory)

Register the additional G-function derivatives needed by the field equations
but not by the Lagrangian construction. The EOM require:
  G2_X, G2_phi, G3_X, G4_XX, G4_phiX, G5_phiX
"""
function _register_eom_functions!(reg::TensorRegistry, ht::HorndeskiTheory)
    G2, G3, G4, G5 = ht.G_functions
    extra = [
        differentiate_G(G2, :X),                          # G2_X
        differentiate_G(G2, :phi),                        # G2_phi
        differentiate_G(G3, :X),                          # G3_X
        differentiate_G(differentiate_G(G4, :X), :X),     # G4_XX
        differentiate_G(differentiate_G(G4, :phi), :X),   # G4_phiX
        differentiate_G(differentiate_G(G5, :phi), :X),   # G5_phiX
    ]
    for stf in extra
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
end

# ── Metric EOM: E_{ab} = 0 ─────────────────────────────────────────

"""
    horndeski_metric_eom(ht::HorndeskiTheory; idx_a=down(:a), idx_b=down(:b),
                         registry=current_registry()) -> TensorExpr

Metric field equation E_{ab} = 0 for Horndeski gravity.
Returns E_{ab} = E^{(2)}_{ab} + E^{(3)}_{ab} + E^{(4)}_{ab} + E^{(5)}_{ab}.

Ground truth: Kobayashi (2019) Eqs 2.5-2.6.

In the GR limit (G2=G3=G5=0, G4=const), this reduces to G4 * Ein_{ab}.
"""
function horndeski_metric_eom(ht::HorndeskiTheory;
                               idx_a::TIndex=down(:a), idx_b::TIndex=down(:b),
                               registry::TensorRegistry=current_registry())
    _register_eom_functions!(registry, ht)

    with_registry(registry) do
        E2 = _metric_eom_L2(ht, idx_a, idx_b; registry=registry)
        E3 = _metric_eom_L3(ht, idx_a, idx_b; registry=registry)
        E4 = _metric_eom_L4(ht, idx_a, idx_b; registry=registry)
        E5 = _metric_eom_L5(ht, idx_a, idx_b; registry=registry)
        E2 + E3 + E4 + E5
    end
end

# E^{(2)}_{ab} = (1/2) G_{2,X} nabla_a phi nabla_b phi + (1/2) G_2 g_{ab}
function _metric_eom_L2(ht::HorndeskiTheory, a::TIndex, b::TIndex;
                         registry::TensorRegistry=current_registry())
    phi = Tensor(ht.scalar_field, TIndex[])
    (1 // 2) * Tensor(:G2_X, TIndex[]) * TDeriv(a, phi) * TDeriv(b, phi) +
    (1 // 2) * Tensor(:G2, TIndex[]) * Tensor(ht.metric, [a, b])
end

# E^{(3)}_{ab}: cubic Galileon contribution.
# Kobayashi (2019) Eq 2.5, L3 row:
#   E^{(3)}_{ab} = (1/2) G_{3,phi} (dd_ab - g_ab Box)
#                  + (1/2) G_{3,X} [Box(phi) dphi_a dphi_b
#                    - dphi_(a dphi^c dd_{b)c}]
function _metric_eom_L3(ht::HorndeskiTheory, a::TIndex, b::TIndex;
                         registry::TensorRegistry=current_registry())
    phi = Tensor(ht.scalar_field, TIndex[])
    G3X = Tensor(:G3_X, TIndex[])
    G3phi = Tensor(:G3_phi, TIndex[])
    g_ab = Tensor(ht.metric, [a, b])

    used = Set{Symbol}([a.name, b.name])
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)

    box_phi = box(phi, ht.metric; registry=registry)
    dd_ab = covd_chain(phi, [a, b])
    dphi_a = TDeriv(a, phi)
    dphi_b = TDeriv(b, phi)

    # Symmetrized term: dphi_(a dphi^c dd_{b)c}
    # = (1/2)[dphi_a dphi^c dd_bc + dphi_b dphi^c dd_ac]
    # where dphi^c = g^{cd} dphi_d
    dphi_up_c = Tensor(ht.metric, [up(c), up(d)]) * TDeriv(down(d), phi)
    dd_bc = covd_chain(phi, [b, down(c)])
    dd_ac = covd_chain(phi, [a, down(c)])
    sym_term = (1 // 2) * (dphi_a * dphi_up_c * dd_bc + dphi_b * dphi_up_c * dd_ac)

    (1 // 2) * G3phi * (dd_ab - g_ab * box_phi) +
    (1 // 2) * G3X * (box_phi * dphi_a * dphi_b - sym_term)
end

# E^{(4)}_{ab}: G4 contribution -- contains Einstein tensor and higher-order terms.
# Kobayashi (2019) Eq 2.5, L4 rows:
#   G_4 G_{ab}
#   - G_{4,phi} (dd_ab - g_ab Box)
#   + G_{4,X} [Box dd_ab - dd_ac dd^c_b + (1/2) g_ab ((Box)^2 - (dd)^2) + R_{acbd} dphi^c dphi^d]
#   + G_{4,XX} [(1/2) dphi_a dphi_b ((Box)^2 - (dd)^2) + Box dphi_(a dd_{b)c} dphi^c
#               - dphi^c dphi^d dd_ca dd_db]
function _metric_eom_L4(ht::HorndeskiTheory, a::TIndex, b::TIndex;
                         registry::TensorRegistry=current_registry())
    phi = Tensor(ht.scalar_field, TIndex[])
    G4 = Tensor(:G4, TIndex[])
    G4X = Tensor(:G4_X, TIndex[])
    G4XX = Tensor(:G4_XX, TIndex[])
    G4phi = Tensor(:G4_phi, TIndex[])

    used = Set{Symbol}([a.name, b.name])
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)

    Ein_ab = Tensor(:Ein, [a, b])
    g_ab = Tensor(ht.metric, [a, b])
    dphi_a = TDeriv(a, phi)
    dphi_b = TDeriv(b, phi)
    dd_ab = covd_chain(phi, [a, b])
    box_phi = box(phi, ht.metric; registry=registry)

    # Term 1: G_4 G_{ab}
    term1 = G4 * Ein_ab

    # Term 2: -G_{4,phi} (dd_ab - g_ab Box)
    term2 = (-1 // 1) * G4phi * (dd_ab - g_ab * box_phi)

    # Term 3: G_{4,X} terms
    # dd_ac dd^c_b = dd_{ac} g^{cd} dd_{db}
    dd_ac = covd_chain(phi, [a, down(c)])
    dd_db = covd_chain(phi, [down(d), b])
    g_cd_up = Tensor(ht.metric, [up(c), up(d)])
    dd_ac_dd_cb = dd_ac * g_cd_up * dd_db

    box_phi2 = box(phi, ht.metric; registry=registry)

    # (dd phi)^2 = g^{ec} g^{fd} dd_{ef} dd_{cd}
    dd_ef = covd_chain(phi, [down(e), down(f)])
    dd_cd = covd_chain(phi, [down(c), down(d)])
    nabla_sq = Tensor(ht.metric, [up(e), up(c)]) * Tensor(ht.metric, [up(f), up(d)]) *
               dd_ef * dd_cd

    # R_{acbd} dphi^c dphi^d
    riem_dphi = Tensor(:Riem, [a, up(c), b, up(d)]) * TDeriv(down(c), phi) * TDeriv(down(d), phi)

    term3 = G4X * (box_phi * dd_ab - dd_ac_dd_cb +
                   (1 // 2) * g_ab * (box_phi * box_phi2 - nabla_sq) +
                   riem_dphi)

    # Term 4: G_{4,XX} terms
    p = fresh_index(used); push!(used, p)
    q = fresh_index(used); push!(used, q)

    box_phi3 = box(phi, ht.metric; registry=registry)
    box_phi4 = box(phi, ht.metric; registry=registry)

    # (dd phi)^2 fresh copy
    dd_pq = covd_chain(phi, [down(p), down(q)])
    dd_cd2 = covd_chain(phi, [down(c), down(d)])
    nabla_sq2 = Tensor(ht.metric, [up(p), up(c)]) * Tensor(ht.metric, [up(q), up(d)]) *
                dd_pq * dd_cd2

    # Box dphi_(a dd_{b)c} dphi^c (symmetrized)
    # = (1/2) Box [dphi_a dd_{bc} dphi^c + dphi_b dd_{ac} dphi^c]
    dphi_up_c = Tensor(ht.metric, [up(c), up(d)]) * TDeriv(down(d), phi)
    dd_bc2 = covd_chain(phi, [b, down(c)])
    dd_ac2 = covd_chain(phi, [a, down(c)])
    box_sym = box_phi3 * (1 // 2) * (dphi_a * dd_bc2 * dphi_up_c +
                                      dphi_b * dd_ac2 * dphi_up_c)

    # dphi^c dphi^d dd_ca dd_db
    dphi_up_p = Tensor(ht.metric, [up(p), up(c)]) * TDeriv(down(c), phi)
    dphi_up_q = Tensor(ht.metric, [up(q), up(d)]) * TDeriv(down(d), phi)
    dd_pa = covd_chain(phi, [down(p), a])
    dd_qb = covd_chain(phi, [down(q), b])
    cross_dd = dphi_up_p * dphi_up_q * dd_pa * dd_qb

    term4 = G4XX * ((1 // 2) * dphi_a * dphi_b * (box_phi4 * box(phi, ht.metric; registry=registry) - nabla_sq2) +
                    box_sym - cross_dd)

    term1 + term2 + term3 + term4
end

# E^{(5)}_{ab}: G5 contribution.
# Kobayashi (2019) Eq 2.6:
#   G_{5,phi} terms (same structure as E^{(4)} G_{4,X} terms but with G_{5,phi}/2)
#   G_{5,X} terms (Ein-coupled + Riemann terms)
function _metric_eom_L5(ht::HorndeskiTheory, a::TIndex, b::TIndex;
                         registry::TensorRegistry=current_registry())
    phi = Tensor(ht.scalar_field, TIndex[])
    G5X = Tensor(:G5_X, TIndex[])
    G5phi = Tensor(:G5_phi, TIndex[])

    used = Set{Symbol}([a.name, b.name])
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)

    g_ab = Tensor(ht.metric, [a, b])
    Ein_ab = Tensor(:Ein, [a, b])
    dd_ab = covd_chain(phi, [a, b])
    box_phi = box(phi, ht.metric; registry=registry)

    # G_{5,phi} term: same structure as G_{4,X} in E^{(4)} but with coefficient -(1/2)
    # -(1/2) G_{5,phi} [Box dd_ab - dd_ac dd^c_b + (1/2) g_ab ((Box)^2 - (dd)^2) + R_{acbd} dphi^c dphi^d]
    dd_ac = covd_chain(phi, [a, down(c)])
    dd_db = covd_chain(phi, [down(d), b])
    g_cd = Tensor(ht.metric, [up(c), up(d)])
    dd_ac_dd_cb = dd_ac * g_cd * dd_db

    dd_ef = covd_chain(phi, [down(e), down(f)])
    dd_cd = covd_chain(phi, [down(c), down(d)])
    nabla_sq = Tensor(ht.metric, [up(e), up(c)]) * Tensor(ht.metric, [up(f), up(d)]) *
               dd_ef * dd_cd

    box_phi2 = box(phi, ht.metric; registry=registry)
    riem_dphi = Tensor(:Riem, [a, up(c), b, up(d)]) * TDeriv(down(c), phi) * TDeriv(down(d), phi)

    term_phi = (-1 // 2) * G5phi * (box_phi * dd_ab - dd_ac_dd_cb +
                                     (1 // 2) * g_ab * (box_phi * box_phi2 - nabla_sq) +
                                     riem_dphi)

    # G_{5,X} terms: Ein-coupled and Riemann-coupled
    p = fresh_index(used); push!(used, p)
    q = fresh_index(used); push!(used, q)
    r = fresh_index(used); push!(used, r)
    s = fresh_index(used); push!(used, s)

    box_phi3 = box(phi, ht.metric; registry=registry)
    box_phi4 = box(phi, ht.metric; registry=registry)

    dd_pq = covd_chain(phi, [down(p), down(q)])
    dd_cd2 = covd_chain(phi, [down(c), down(d)])
    nabla_sq2 = Tensor(ht.metric, [up(p), up(c)]) * Tensor(ht.metric, [up(q), up(d)]) *
                dd_pq * dd_cd2

    # R_{acbd} dd^{cd} Box
    riem_dd_box = Tensor(:Riem, [a, up(c), b, up(d)]) *
                  covd_chain(phi, [down(c), down(d)]) * box_phi3

    # R_{acbd} dd^{ce} dd_e^d  (chain contraction)
    riem_dd_dd = Tensor(:Riem, [a, up(p), b, up(q)]) *
                 covd_chain(phi, [down(p), down(r)]) *
                 Tensor(ht.metric, [up(r), up(s)]) *
                 covd_chain(phi, [down(s), down(q)])

    term_X = G5X * ((-1 // 2) * Ein_ab * (box_phi4 * box(phi, ht.metric; registry=registry) - nabla_sq2) +
                    riem_dd_box - riem_dd_dd)

    term_phi + term_X
end

# ── Scalar EOM: E_phi = 0 ──────────────────────────────────────────

"""
    horndeski_scalar_eom(ht::HorndeskiTheory; registry=current_registry()) -> TensorExpr

Scalar field equation E_phi = 0 for Horndeski gravity.
Returns E_phi = J^{(2)} + J^{(3)} + J^{(4)} + J^{(5)}.

Ground truth: Kobayashi (2019) Eq 2.7.

In the quintessence limit (G2 = X - V(phi), G3=G5=0, G4=const),
this reduces to the Klein-Gordon equation: Box(phi) + V'(phi) = 0.
"""
function horndeski_scalar_eom(ht::HorndeskiTheory;
                               registry::TensorRegistry=current_registry())
    _register_eom_functions!(registry, ht)

    with_registry(registry) do
        J2 = _scalar_eom_L2(ht; registry=registry)
        J3 = _scalar_eom_L3(ht; registry=registry)
        J4 = _scalar_eom_L4(ht; registry=registry)
        J5 = _scalar_eom_L5(ht; registry=registry)
        J2 + J3 + J4 + J5
    end
end

# J^{(2)} = -G_{2,phi} + G_{2,X} Box(phi)
function _scalar_eom_L2(ht::HorndeskiTheory;
                         registry::TensorRegistry=current_registry())
    phi = Tensor(ht.scalar_field, TIndex[])
    box_phi = box(phi, ht.metric; registry=registry)
    (-1 // 1) * Tensor(:G2_phi, TIndex[]) + Tensor(:G2_X, TIndex[]) * box_phi
end

# J^{(3)} = G_{3,phi} Box(phi) + G_{3,X} [R_{ab} dphi^a dphi^b + (Box phi)^2 - (dd phi)^2]
# Kobayashi (2019) Eq 2.7.
function _scalar_eom_L3(ht::HorndeskiTheory;
                         registry::TensorRegistry=current_registry())
    phi = Tensor(ht.scalar_field, TIndex[])
    G3phi = Tensor(:G3_phi, TIndex[])
    G3X = Tensor(:G3_X, TIndex[])

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)

    box_phi = box(phi, ht.metric; registry=registry)
    box_phi2 = box(phi, ht.metric; registry=registry)

    # (dd phi)^2
    dd_ab = covd_chain(phi, [down(a), down(b)])
    dd_cd = covd_chain(phi, [down(c), down(d)])
    nabla_sq = Tensor(ht.metric, [up(a), up(c)]) * Tensor(ht.metric, [up(b), up(d)]) *
               dd_ab * dd_cd

    # R_{ab} dphi^a dphi^b
    ric_dphi = Tensor(:Ric, [up(a), up(b)]) * TDeriv(down(a), phi) * TDeriv(down(b), phi)

    G3phi * box_phi + G3X * (ric_dphi + box_phi2 * box(phi, ht.metric; registry=registry) - nabla_sq)
end

# J^{(4)} = -G_{4,phi} R - G_{4,phiX} [(Box phi)^2 - (dd phi)^2]
# Kobayashi (2019) Eq 2.7.
function _scalar_eom_L4(ht::HorndeskiTheory;
                         registry::TensorRegistry=current_registry())
    phi = Tensor(ht.scalar_field, TIndex[])
    R = Tensor(:RicScalar, TIndex[])

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)

    box_phi = box(phi, ht.metric; registry=registry)
    box_phi2 = box(phi, ht.metric; registry=registry)

    dd_ab = covd_chain(phi, [down(a), down(b)])
    dd_cd = covd_chain(phi, [down(c), down(d)])
    nabla_sq = Tensor(ht.metric, [up(a), up(c)]) * Tensor(ht.metric, [up(b), up(d)]) *
               dd_ab * dd_cd

    (-1 // 1) * Tensor(:G4_phi, TIndex[]) * R -
    Tensor(:G4_phiX, TIndex[]) * (box_phi * box_phi2 - nabla_sq)
end

# J^{(5)} = -G_{5,phi} G_{ab} dd^{ab} phi
#          + (1/6) G_{5,phiX} [(Box phi)^3 - 3 Box(dd phi)^2 + 2(dd phi)^3]
# Kobayashi (2019) Eq 2.7.
function _scalar_eom_L5(ht::HorndeskiTheory;
                         registry::TensorRegistry=current_registry())
    phi = Tensor(ht.scalar_field, TIndex[])
    G5phi = Tensor(:G5_phi, TIndex[])
    G5phiX = Tensor(:G5_phiX, TIndex[])

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)

    # G_{ab} dd^{ab} phi
    ein_dd = Tensor(:Ein, [up(a), up(b)]) * covd_chain(phi, [down(a), down(b)])

    # Cubic structure (same as L5 Lagrangian but with G_{5,phi_X})
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)

    box1 = box(phi, ht.metric; registry=registry)
    box2 = box(phi, ht.metric; registry=registry)
    box3 = box(phi, ht.metric; registry=registry)
    cube_box = box1 * box2 * box3

    # Box(phi) * (dd phi)^2
    box_sq = box(phi, ht.metric; registry=registry)
    dd_cd = covd_chain(phi, [down(c), down(d)])
    dd_ef = covd_chain(phi, [down(e), down(f)])
    sq_term = box_sq * Tensor(ht.metric, [up(c), up(e)]) *
              Tensor(ht.metric, [up(d), up(f)]) * dd_cd * dd_ef

    # (dd phi)^3 trace
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

    (-1 // 1) * G5phi * ein_dd +
    (1 // 6) * G5phiX * (cube_box - (3 // 1) * sq_term + (2 // 1) * triple_term)
end

# ── Combined EOM ────────────────────────────────────────────────────

"""
    horndeski_eom(ht::HorndeskiTheory; idx_a=down(:a), idx_b=down(:b),
                  registry=current_registry()) -> (E_ab, E_phi)

Return both field equations as a tuple (metric EOM, scalar EOM).
"""
function horndeski_eom(ht::HorndeskiTheory;
                        idx_a::TIndex=down(:a), idx_b::TIndex=down(:b),
                        registry::TensorRegistry=current_registry())
    E_ab = horndeski_metric_eom(ht; idx_a=idx_a, idx_b=idx_b, registry=registry)
    E_phi = horndeski_scalar_eom(ht; registry=registry)
    (E_ab, E_phi)
end
