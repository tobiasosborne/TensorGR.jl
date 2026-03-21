#= Differential invariant database: scalar invariants with 4 covariant
#  derivatives of curvature tensors (order 6).
#
#  Extends the order-4 catalog from differential_2.jl to include order-6
#  differential invariants.
#
#  Order 6 comes in two sectors:
#    Case {4}: 4 covariant derivatives + 1 curvature factor (n_derivs=4, n_riemann=1)
#    Case {0,2}: 2 curvature factors + 2 derivatives distributed (n_derivs=2, n_riemann=2)
#
#  Case {4} invariants:
#    box2_R          : Box^2 R = nabla^a nabla_a nabla^b nabla_b R       (total derivative)
#    hessian_R_sq    : (nabla_a nabla_b R)(nabla^a nabla^b R)
#    hessian_Ric_sq  : (nabla_a nabla_b R_{cd})(nabla^a nabla^b R^{cd})
#
#  Case {0,2} invariants:
#    R_box_R         : R nabla^a nabla_a R = R Box R
#    Ric_box_Ric     : R_{ab} nabla^c nabla_c R^{ab} = R_{ab} Box R^{ab}
#    Riem_box_Riem   : R_{abcd} nabla^e nabla_e R^{abcd} = R_{abcd} Box R^{abcd}
#
#  Ground truth: Fulling, King, Wybourne & Cummins (1992);
#                Garcia-Parrado & Martin-Garcia (2007), Sec 6.
=#

# ---- Expression builders (private) ------------------------------------------

# Box^2 R = nabla^a nabla_a nabla^b nabla_b R = g^{ab} D_a D_b (g^{cd} D_c D_d R)
function _build_box2_R(reg::TensorRegistry, manifold::Symbol,
                       metric::Symbol, covd::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)
    R = Tensor(:RicScalar, TIndex[])
    # Inner Box: g^{cd} D_c D_d R
    inner = Tensor(metric, [up(c), up(d)]) * TDeriv(down(c), TDeriv(down(d), R, covd), covd)
    # Outer Box: g^{ab} D_a D_b (inner)
    Tensor(metric, [up(a), up(b)]) * TDeriv(down(a), TDeriv(down(b), inner, covd), covd)
end

# (nabla_a nabla_b R)(nabla^a nabla^b R)
# = g^{ac} g^{bd} (D_a D_b R)(D_c D_d R)
function _build_hessian_R_sq(reg::TensorRegistry, manifold::Symbol,
                             metric::Symbol, covd::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)
    R1 = Tensor(:RicScalar, TIndex[])
    R2 = Tensor(:RicScalar, TIndex[])
    hess1 = TDeriv(down(a), TDeriv(down(b), R1, covd), covd)
    hess2 = TDeriv(down(c), TDeriv(down(d), R2, covd), covd)
    Tensor(metric, [up(a), up(c)]) *
        Tensor(metric, [up(b), up(d)]) *
        hess1 * hess2
end

# (nabla_a nabla_b R_{cd})(nabla^a nabla^b R^{cd})
# = g^{ae} g^{bf} g^{cg} g^{dh} (D_a D_b Ric_{cd})(D_e D_f Ric_{gh})
function _build_hessian_Ric_sq(reg::TensorRegistry, manifold::Symbol,
                               metric::Symbol, covd::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    g_idx = fresh_index(used); push!(used, g_idx)
    h = fresh_index(used)

    Ric1 = Tensor(:Ric, [down(c), down(d)])
    Ric2 = Tensor(:Ric, [down(g_idx), down(h)])
    hess1 = TDeriv(down(a), TDeriv(down(b), Ric1, covd), covd)
    hess2 = TDeriv(down(e), TDeriv(down(f), Ric2, covd), covd)

    Tensor(metric, [up(a), up(e)]) *
        Tensor(metric, [up(b), up(f)]) *
        Tensor(metric, [up(c), up(g_idx)]) *
        Tensor(metric, [up(d), up(h)]) *
        hess1 * hess2
end

# R Box R = R nabla^a nabla_a R = R g^{ab} D_a D_b R
function _build_R_box_R(reg::TensorRegistry, manifold::Symbol,
                        metric::Symbol, covd::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)
    R_scalar = Tensor(:RicScalar, TIndex[])
    R_scalar2 = Tensor(:RicScalar, TIndex[])
    R_scalar * Tensor(metric, [up(a), up(b)]) *
        TDeriv(down(a), TDeriv(down(b), R_scalar2, covd), covd)
end

# R_{ab} Box R^{ab} = R_{ab} g^{cd} D_c D_d R^{ab}
# = g^{cd} g^{ae} g^{bf} R_{ab} D_c D_d Ric_{ef}
function _build_Ric_box_Ric(reg::TensorRegistry, manifold::Symbol,
                            metric::Symbol, covd::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used)

    Ric1 = Tensor(:Ric, [down(a), down(b)])
    Ric2 = Tensor(:Ric, [down(e), down(f)])
    box_Ric = TDeriv(down(c), TDeriv(down(d), Ric2, covd), covd)

    Tensor(metric, [up(a), up(e)]) *
        Tensor(metric, [up(b), up(f)]) *
        Tensor(metric, [up(c), up(d)]) *
        Ric1 * box_Ric
end

# R_{abcd} Box R^{abcd} = R_{abcd} g^{ef} D_e D_f R^{abcd}
# = g^{ef} g^{ag} g^{bh} g^{ci} g^{dj} R_{abcd} D_e D_f Riem_{ghij}
function _build_Riem_box_Riem(reg::TensorRegistry, manifold::Symbol,
                              metric::Symbol, covd::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    g_idx = fresh_index(used); push!(used, g_idx)
    h = fresh_index(used); push!(used, h)
    i_idx = fresh_index(used); push!(used, i_idx)
    j = fresh_index(used)

    Riem1 = Tensor(:Riem, [down(a), down(b), down(c), down(d)])
    Riem2 = Tensor(:Riem, [down(g_idx), down(h), down(i_idx), down(j)])
    box_Riem = TDeriv(down(e), TDeriv(down(f), Riem2, covd), covd)

    Tensor(metric, [up(e), up(f)]) *
        Tensor(metric, [up(a), up(g_idx)]) *
        Tensor(metric, [up(b), up(h)]) *
        Tensor(metric, [up(c), up(i_idx)]) *
        Tensor(metric, [up(d), up(j)]) *
        Riem1 * box_Riem
end

# ---- Extend catalog with order-6 entries -----------------------------------

# Case {4}: 4 derivatives + 1 curvature factor
_DIFF_INVAR_CATALOG[:box2_R] = DiffInvariantEntry(
    :box2_R, 4, 1, 6,
    "Box^2 R = nabla^a nabla_a nabla^b nabla_b R (iterated d'Alembertian of Ricci scalar)",
    _build_box2_R,
    true,  # is total derivative (nabla_a J^a for J^a = nabla^a Box R)
)

_DIFF_INVAR_CATALOG[:hessian_R_sq] = DiffInvariantEntry(
    :hessian_R_sq, 4, 1, 6,
    "(nabla_a nabla_b R)(nabla^a nabla^b R) (Hessian of scalar curvature squared)",
    _build_hessian_R_sq,
    false,
)

_DIFF_INVAR_CATALOG[:hessian_Ric_sq] = DiffInvariantEntry(
    :hessian_Ric_sq, 4, 1, 6,
    "(nabla_a nabla_b R_{cd})(nabla^a nabla^b R^{cd}) (Hessian of Ricci squared)",
    _build_hessian_Ric_sq,
    false,
)

# Case {0,2}: 2 curvature factors + 2 derivatives
_DIFF_INVAR_CATALOG[:R_box_R] = DiffInvariantEntry(
    :R_box_R, 2, 2, 6,
    "R Box R = R nabla^a nabla_a R (scalar curvature times Box R)",
    _build_R_box_R,
    false,
)

_DIFF_INVAR_CATALOG[:Ric_box_Ric] = DiffInvariantEntry(
    :Ric_box_Ric, 2, 2, 6,
    "R_{ab} Box R^{ab} = R_{ab} nabla^c nabla_c R^{ab} (Ricci times Box Ricci)",
    _build_Ric_box_Ric,
    false,
)

_DIFF_INVAR_CATALOG[:Riem_box_Riem] = DiffInvariantEntry(
    :Riem_box_Riem, 2, 2, 6,
    "R_{abcd} Box R^{abcd} = R_{abcd} nabla^e nabla_e R^{abcd} (Riemann times Box Riemann)",
    _build_Riem_box_Riem,
    false,
)
