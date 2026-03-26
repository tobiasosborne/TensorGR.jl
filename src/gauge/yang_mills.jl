#= Yang-Mills field strength and equations of motion (indexed tensor form).
#
# Provides the Yang-Mills field strength, gauge-covariant derivative, Bianchi
# identity, Lagrangian density, and field equations in the indexed tensor
# representation used by the BRST module.  These complement the exterior-calculus
# versions in src/exterior/algebra_forms.jl (which operate on AlgValuedForm).
#
# Convention:
#   F^I_{ab} = ∂_a A^I_b − ∂_b A^I_a + f^I_{JK} A^J_a A^K_b
#   D_a X^I  = ∂_a X^I + f^I_{JK} A^J_a X^K
#
# References:
#   Peskin & Schroeder (1995), Ch 15.
#   Weinberg, QFT Vol II (1996), Ch 15.
#   Nakahara (2003), Sec 11.1.
=#

# ── Field strength ───────────────────────────────────────────────────

"""
    yang_mills_field_strength(ggp, I, a, b; registry=current_registry())

Yang-Mills field strength with explicit indices:

    F^I_{ab} = ∂_a A^I_b − ∂_b A^I_a + f^I_{JK} A^J_a A^K_b

# Arguments
- `ggp::GaugeGroupProperties` — gauge group
- `I::TIndex` — Up gauge index
- `a::TIndex`, `b::TIndex` — Down Tangent indices
"""
function yang_mills_field_strength(ggp::GaugeGroupProperties,
                                   I::TIndex, a::TIndex, b::TIndex;
                                   registry::TensorRegistry=current_registry())
    I.position === Up   || error("yang_mills_field_strength: I must be Up")
    a.position === Down || error("yang_mills_field_strength: a must be Down")
    b.position === Down || error("yang_mills_field_strength: b must be Down")

    vb = ggp.vbundle
    used = Set{Symbol}([I.name, a.name, b.name])
    J_sym = fresh_index(used; vbundle=vb)
    push!(used, J_sym)
    K_sym = fresh_index(used; vbundle=vb)

    J_up   = TIndex(J_sym, Up, vb)
    J_down = TIndex(J_sym, Down, vb)
    K_up   = TIndex(K_sym, Up, vb)
    K_down = TIndex(K_sym, Down, vb)

    # ∂_a A^I_b
    dA_ab = TDeriv(a, Tensor(ggp.gauge_field, [I, b]), :partial)
    # ∂_b A^I_a
    dA_ba = TDeriv(b, Tensor(ggp.gauge_field, [I, a]), :partial)
    # f^I_{JK} A^J_a A^K_b
    fAA = TProduct(1 // 1, TensorExpr[
        Tensor(ggp.structure_constants, [I, J_down, K_down]),
        Tensor(ggp.gauge_field, [J_up, a]),
        Tensor(ggp.gauge_field, [K_up, b])
    ])

    TSum(TensorExpr[dA_ab, TProduct(-1 // 1, TensorExpr[dA_ba]), fAA])
end

# ── Gauge-covariant derivative ───────────────────────────────────────

"""
    gauge_covariant_deriv(ggp, expr, I, a; registry=current_registry())

Gauge-covariant derivative of a gauge-algebra–valued tensor:

    D_a X^I = ∂_a X^I + f^I_{JK} A^J_a X^K

# Arguments
- `ggp::GaugeGroupProperties` — gauge group
- `expr::TensorExpr` — expression carrying gauge index (must carry one Up gauge index)
- `I::TIndex` — Up gauge index on the result
- `a::TIndex` — Down Tangent derivative index
"""
function gauge_covariant_deriv(ggp::GaugeGroupProperties,
                                expr::TensorExpr,
                                I::TIndex, a::TIndex;
                                registry::TensorRegistry=current_registry())
    I.position === Up   || error("gauge_covariant_deriv: I must be Up")
    a.position === Down || error("gauge_covariant_deriv: a must be Down")

    vb = ggp.vbundle

    # Find the gauge-algebra index in expr to contract with structure constants
    expr_idxs = indices(expr)
    gauge_idx = nothing
    for idx in expr_idxs
        if idx.vbundle === vb && idx.position === Up
            gauge_idx = idx
            break
        end
    end
    gauge_idx !== nothing ||
        error("gauge_covariant_deriv: expression must carry an Up gauge index in vbundle $vb")

    used = Set{Symbol}([I.name, a.name])
    for idx in expr_idxs
        push!(used, idx.name)
    end
    J_sym = fresh_index(used; vbundle=vb)
    push!(used, J_sym)
    K_sym = fresh_index(used; vbundle=vb)

    J_up   = TIndex(J_sym, Up, vb)
    J_down = TIndex(J_sym, Down, vb)
    K_up   = TIndex(K_sym, Up, vb)
    K_down = TIndex(K_sym, Down, vb)

    # ∂_a X^I  (we relabel expr's gauge index I → K for the interaction term)
    term1 = TDeriv(a, expr, :partial)

    # f^I_{JK} A^J_a X^K — contract K with the gauge index of expr
    # Replace the gauge index in expr with K_up
    expr_K = _replace_gauge_index(expr, gauge_idx, K_up)

    term2 = TProduct(1 // 1, TensorExpr[
        Tensor(ggp.structure_constants, [I, J_down, K_down]),
        Tensor(ggp.gauge_field, [J_up, a]),
        expr_K
    ])

    TSum(TensorExpr[term1, term2])
end

"""Replace the gauge-algebra index `old` with `new_idx` in an expression."""
function _replace_gauge_index(t::Tensor, old::TIndex, new_idx::TIndex)
    new_idxs = TIndex[]
    for idx in t.indices
        if idx.name === old.name && idx.vbundle === old.vbundle && idx.position === old.position
            push!(new_idxs, new_idx)
        else
            push!(new_idxs, idx)
        end
    end
    Tensor(t.name, new_idxs)
end

function _replace_gauge_index(p::TProduct, old::TIndex, new_idx::TIndex)
    TProduct(p.scalar, TensorExpr[_replace_gauge_index(f, old, new_idx) for f in p.factors])
end

function _replace_gauge_index(s::TSum, old::TIndex, new_idx::TIndex)
    TSum(TensorExpr[_replace_gauge_index(t, old, new_idx) for t in s.terms])
end

function _replace_gauge_index(d::TDeriv, old::TIndex, new_idx::TIndex)
    TDeriv(d.index, _replace_gauge_index(d.arg, old, new_idx), d.covd)
end

function _replace_gauge_index(s::TScalar, ::TIndex, ::TIndex)
    s
end

function _replace_gauge_index(expr::TensorExpr, ::TIndex, ::TIndex)
    expr  # fallback: leave unchanged
end

# ── Bianchi identity ─────────────────────────────────────────────────

"""
    yang_mills_bianchi(ggp, I, a, b, c; registry=current_registry())

Yang-Mills Bianchi identity D_{[a} F^I_{bc]}:

    D_a F^I_{bc} + D_b F^I_{ca} + D_c F^I_{ab} = 0

Returns the expression (should simplify to zero).

# Arguments
- `ggp::GaugeGroupProperties` — gauge group
- `I::TIndex` — Up gauge index
- `a, b, c::TIndex` — Down Tangent indices
"""
function yang_mills_bianchi(ggp::GaugeGroupProperties,
                             I::TIndex, a::TIndex, b::TIndex, c::TIndex;
                             registry::TensorRegistry=current_registry())
    F_bc = yang_mills_field_strength(ggp, I, b, c; registry=registry)
    F_ca = yang_mills_field_strength(ggp, I, c, a; registry=registry)
    F_ab = yang_mills_field_strength(ggp, I, a, b; registry=registry)

    t1 = gauge_covariant_deriv(ggp, F_bc, I, a; registry=registry)
    t2 = gauge_covariant_deriv(ggp, F_ca, I, b; registry=registry)
    t3 = gauge_covariant_deriv(ggp, F_ab, I, c; registry=registry)

    TSum(TensorExpr[t1, t2, t3])
end

# ── Lagrangian and field equations ───────────────────────────────────

"""
    yang_mills_lagrangian(ggp; metric=nothing, registry=current_registry())

Yang-Mills Lagrangian density:

    L = −(1/4) F^I_{ab} F_I^{ab}

Returns the symbolic expression with metric contractions to raise indices.
If `metric` is not specified, uses the metric from the gauge group's manifold.
"""
function yang_mills_lagrangian(ggp::GaugeGroupProperties;
                                metric::Union{Symbol,Nothing}=nothing,
                                registry::TensorRegistry=current_registry())
    vb = ggp.vbundle
    mfld = ggp.manifold

    # Determine metric
    met = if metric !== nothing
        metric
    else
        haskey(registry.metric_cache, mfld) ||
            error("yang_mills_lagrangian: no metric registered for manifold $mfld")
        registry.metric_cache[mfld]
    end

    used = Set{Symbol}()

    # Indices for first F^I_{ab}
    I_sym = fresh_index(used; vbundle=vb)
    push!(used, I_sym)
    a_sym = fresh_index(used)
    push!(used, a_sym)
    b_sym = fresh_index(used)
    push!(used, b_sym)

    # Indices for second F_I^{cd} (need metric contractions)
    c_sym = fresh_index(used)
    push!(used, c_sym)
    d_sym = fresh_index(used)
    push!(used, d_sym)

    I_up   = TIndex(I_sym, Up, vb)
    I_down = TIndex(I_sym, Down, vb)
    a_down = down(a_sym)
    b_down = down(b_sym)
    a_up   = up(a_sym)
    b_up   = up(b_sym)
    c_down = down(c_sym)
    d_down = down(d_sym)
    c_up   = up(c_sym)
    d_up   = up(d_sym)

    # F^I_{ab}
    F1 = yang_mills_field_strength(ggp, I_up, a_down, b_down; registry=registry)

    # F_I^{cd} = delta_{IJ} g^{ca} g^{db} F^J_{ab}
    # For simplicity, build F^I_{cd} and contract with g^{ac} g^{bd}
    J_sym = fresh_index(used; vbundle=vb)
    push!(used, J_sym)
    J_up   = TIndex(J_sym, Up, vb)
    J_down = TIndex(J_sym, Down, vb)

    F2 = yang_mills_field_strength(ggp, J_up, c_down, d_down; registry=registry)

    # −(1/4) δ_{IJ} g^{ac} g^{bd} F^I_{ab} F^J_{cd}
    delta_name = haskey(registry.delta_cache, mfld) ? registry.delta_cache[mfld] : :delta
    # For gauge indices we use the Killing form δ_{IJ} (compact semisimple)
    delta_gauge = Tensor(:delta, [I_down, J_down])

    g_ac = Tensor(met, [a_up, c_up])
    g_bd = Tensor(met, [b_up, d_up])

    TProduct(-1 // 4, TensorExpr[delta_gauge, g_ac, g_bd, F1, F2])
end

"""
    yang_mills_field_equations(ggp, I, b; metric=nothing, registry=current_registry())

Sourceless Yang-Mills field equation:

    D_a F^{Ia}_{  b} = 0

Returns the expression (should be set to zero).
"""
function yang_mills_field_equations(ggp::GaugeGroupProperties,
                                    I::TIndex, b::TIndex;
                                    metric::Union{Symbol,Nothing}=nothing,
                                    registry::TensorRegistry=current_registry())
    I.position === Up   || error("yang_mills_field_equations: I must be Up")
    b.position === Down || error("yang_mills_field_equations: b must be Down")

    vb = ggp.vbundle
    mfld = ggp.manifold

    met = if metric !== nothing
        metric
    else
        haskey(registry.metric_cache, mfld) ||
            error("yang_mills_field_equations: no metric registered for manifold $mfld")
        registry.metric_cache[mfld]
    end

    used = Set{Symbol}([I.name, b.name])

    # D_a F^{Ia}_b : need dummy a for contraction
    a_sym = fresh_index(used)
    push!(used, a_sym)
    c_sym = fresh_index(used)
    push!(used, c_sym)

    a_down = down(a_sym)
    c_down = down(c_sym)
    a_up   = up(a_sym)

    # F^I_{cb} with c dummy
    F = yang_mills_field_strength(ggp, I, c_down, b; registry=registry)

    # Raise first spacetime index: F^{Ia}_b = g^{ac} F^I_{cb}
    g_ac = Tensor(met, [a_up, up(c_sym)])
    F_raised = TProduct(1 // 1, TensorExpr[g_ac, F])

    # D_a (F^{Ia}_b)
    gauge_covariant_deriv(ggp, F_raised, I, a_down; registry=registry)
end
