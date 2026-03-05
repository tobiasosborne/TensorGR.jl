#= Curvature tensor conversions.

Riemann decomposition: R_{abcd} = C_{abcd} + (Weyl decomposition terms)
where C is the Weyl tensor and the remaining terms involve Ricci/scalar.

In d dimensions:
  R_{abcd} = C_{abcd}
    + 2/(d-2) (g_{a[c} R_{d]b} - g_{b[c} R_{d]a})
    - 2/((d-1)(d-2)) R g_{a[c} g_{d]b}
=#

"""
    riemann_to_weyl(a, b, c, d, metric; dim=4) -> TensorExpr

Express Riemann in terms of Weyl + Ricci decomposition:
  R_{abcd} = C_{abcd} + 2/(d-2)(g_{a[c}R_{d]b} - g_{b[c}R_{d]a})
           - 2/((d-1)(d-2)) R g_{a[c}g_{d]b}
"""
function riemann_to_weyl(a::TIndex, b::TIndex, c::TIndex, d::TIndex,
                         metric::Symbol; dim::Int=4)
    Riem = Tensor(:Riem, [a, b, c, d])
    Weyl = Tensor(:Weyl, [a, b, c, d])
    Ric_db = Tensor(:Ric, [d, b])
    Ric_cb = Tensor(:Ric, [c, b])
    Ric_da = Tensor(:Ric, [d, a])
    Ric_ca = Tensor(:Ric, [c, a])
    g_ac = Tensor(metric, [a, c])
    g_ad = Tensor(metric, [a, d])
    g_bc = Tensor(metric, [b, c])
    g_bd = Tensor(metric, [b, d])
    R = Tensor(:RicScalar, TIndex[])

    coeff1 = 1 // (dim - 2)
    coeff2 = 1 // ((dim - 1) * (dim - 2))

    # Antisymmetrize: g_{a[c} R_{d]b} = (1/2)(g_{ac}R_{db} - g_{ad}R_{cb})
    term1 = coeff1 * (g_ac * Ric_db - g_ad * Ric_cb -
                       g_bc * Ric_da + g_bd * Ric_ca)

    # g_{a[c} g_{d]b} = (1/2)(g_{ac}g_{db} - g_{ad}g_{cb})
    term2 = coeff2 * R * (g_ac * g_bd - g_ad * g_bc)

    Weyl + term1 - term2
end

"""
    weyl_to_riemann(a, b, c, d, metric; dim=4) -> TensorExpr

Express Weyl in terms of Riemann - Ricci decomposition (inverse of above).
"""
function weyl_to_riemann(a::TIndex, b::TIndex, c::TIndex, d::TIndex,
                         metric::Symbol; dim::Int=4)
    Riem = Tensor(:Riem, [a, b, c, d])
    Ric_db = Tensor(:Ric, [d, b])
    Ric_cb = Tensor(:Ric, [c, b])
    Ric_da = Tensor(:Ric, [d, a])
    Ric_ca = Tensor(:Ric, [c, a])
    g_ac = Tensor(metric, [a, c])
    g_ad = Tensor(metric, [a, d])
    g_bc = Tensor(metric, [b, c])
    g_bd = Tensor(metric, [b, d])
    R = Tensor(:RicScalar, TIndex[])

    coeff1 = 1 // (dim - 2)
    coeff2 = 1 // ((dim - 1) * (dim - 2))

    term1 = coeff1 * (g_ac * Ric_db - g_ad * Ric_cb -
                       g_bc * Ric_da + g_bd * Ric_ca)
    term2 = coeff2 * R * (g_ac * g_bd - g_ad * g_bc)

    Riem - term1 + term2
end

"""
    ricci_to_einstein(a, b, metric; dim=4) -> TensorExpr

R_{ab} in terms of Einstein tensor: R_{ab} = G_{ab} + (1/2)g_{ab}R
"""
function ricci_to_einstein(a::TIndex, b::TIndex, metric::Symbol)
    Tensor(:Ein, [a, b]) +
        (1 // 2) * Tensor(metric, [a, b]) * Tensor(:RicScalar, TIndex[])
end

"""
    einstein_to_ricci(a, b, metric; dim=4) -> TensorExpr

G_{ab} in terms of Ricci: G_{ab} = R_{ab} - (1/2)g_{ab}R
"""
function einstein_to_ricci(a::TIndex, b::TIndex, metric::Symbol)
    Tensor(:Ric, [a, b]) -
        (1 // 2) * Tensor(metric, [a, b]) * Tensor(:RicScalar, TIndex[])
end

# ─── Contraction detection ──────────────────────────────────────────

"""
    contract_curvature(expr::TensorExpr) -> TensorExpr

Walk the expression bottom-up and replace:
- Riemann with a pair of contracted indices → Ricci
- Ricci with a pair of contracted indices → RicciScalar

Contraction patterns for Riemann R^a{}_{bcd}:
- indices 1 & 3 contracted (R^a{}_{bad}) → -Ric_{bd}
- indices 1 & 4 contracted (R^a{}_{bca}) → +Ric_{bc}
"""
function contract_curvature(expr::TensorExpr)
    walk(expr) do node
        if node isa TProduct
            _contract_curvature_product(node)
        elseif node isa Tensor
            _contract_curvature_bare(node)
        else
            node
        end
    end
end

function _contract_curvature_bare(t::Tensor)
    if t.name == :Riem && length(t.indices) == 4
        replacement = _try_contract_riemann(t)
        replacement !== nothing && return replacement
    elseif t.name == :Ric && length(t.indices) == 2
        replacement = _try_contract_ricci(t)
        replacement !== nothing && return replacement
    end
    t
end

function _contract_curvature_product(p::TProduct)
    # Look for a single Riemann or Ricci tensor in the product factors
    for (i, f) in enumerate(p.factors)
        f isa Tensor || continue

        if f.name == :Riem && length(f.indices) == 4
            replacement = _try_contract_riemann(f)
            if replacement !== nothing
                new_factors = TensorExpr[p.factors[j] for j in eachindex(p.factors) if j != i]
                push!(new_factors, replacement)
                return tproduct(p.scalar, new_factors)
            end
        elseif f.name == :Ric && length(f.indices) == 2
            replacement = _try_contract_ricci(f)
            if replacement !== nothing
                new_factors = TensorExpr[p.factors[j] for j in eachindex(p.factors) if j != i]
                push!(new_factors, replacement)
                return tproduct(p.scalar, new_factors)
            end
        end
    end

    # Also handle bare tensors that appear as the only factor
    p
end

function _try_contract_riemann(t::Tensor)
    # Check all pairs for contraction (same name, opposite position)
    idxs = t.indices
    for i in 1:4
        for j in (i+1):4
            if idxs[i].name == idxs[j].name &&
               idxs[i].position != idxs[j].position
                # Found a contraction between slots i and j
                remaining = TIndex[idxs[k] for k in 1:4 if k != i && k != j]
                # Determine the sign based on which slots are contracted
                # R^a_{bac} with (1,3) contracted: R^a_{bac} = +R_{bc} (standard convention)
                # R^a_{bca} with (1,4) contracted: R^a_{bca} = -R_{bc}
                # For general slot positions, use the parity of bringing contracted
                # indices adjacent:
                #   (1,3): contract slots 1,3 → Ric from slots 2,4, sign = -1
                #   (1,4): contract slots 1,4 → Ric from slots 2,3, sign = +1
                #   (2,4): contract slots 2,4 → Ric from slots 1,3, sign = +1
                #   etc.
                # Standard: R^a_{bcd}, contraction of first with third = R^a_{bac} = R_{bc}
                sign = _riemann_contraction_sign(i, j)
                if sign == 1
                    return Tensor(:Ric, remaining)
                else
                    return tproduct(sign // 1, TensorExpr[Tensor(:Ric, remaining)])
                end
            end
        end
    end
    nothing
end

function _riemann_contraction_sign(i::Int, j::Int)
    # Riemann symmetries: R_{abcd} = -R_{bacd} = -R_{abdc} = R_{cdab}
    # Standard trace: R_{abcd} g^{ac} = R_{bd}  (slots 1,3)
    # So contraction of slots (1,3) → +1, yielding Ricci with remaining (2,4)
    # Contraction of slots (1,4) → -1 (by antisymmetry in cd)
    # Contraction of slots (2,3) → -1 (by antisymmetry in ab)
    # Contraction of slots (2,4) → +1 (double antisymmetry)
    # Contraction of slots (1,2) or (3,4) → 0 by antisymmetry, but we still
    # return a sign; the result will be zero by the caller's context.
    if (i, j) == (1, 3) || (i, j) == (2, 4)
        return 1
    elseif (i, j) == (1, 4) || (i, j) == (2, 3)
        return -1
    else
        # (1,2) or (3,4): antisymmetric → trace is zero
        return 0
    end
end

function _try_contract_ricci(t::Tensor)
    idxs = t.indices
    if length(idxs) == 2 &&
       idxs[1].name == idxs[2].name &&
       idxs[1].position != idxs[2].position
        return Tensor(:RicScalar, TIndex[])
    end
    nothing
end

# ─── Schouten tensor ────────────────────────────────────────────────

"""
    schouten_to_ricci(a, b; dim=4) -> TensorExpr

Schouten tensor in terms of Ricci and scalar curvature:
  P_{ab} = 1/(d-2) (R_{ab} - R g_{ab} / (2(d-1)))
"""
function schouten_to_ricci(a::TIndex, b::TIndex; dim::Int=4)
    coeff1 = 1 // (dim - 2)
    coeff2 = 1 // (2 * (dim - 1))
    # P_{ab} = (1/(d-2)) * R_{ab} - (1/(d-2)) * (1/(2(d-1))) * R * g_{ab}
    # But we don't know metric name here — Schouten is metric-independent at this level
    # Convention: use the expression-level form
    # Actually, we need the metric. Let's add a metric kwarg.
    error("Use schouten_to_ricci(a, b, metric; dim) instead")
end

"""
    schouten_to_ricci(a, b, metric; dim=4) -> TensorExpr

Schouten tensor in terms of Ricci and scalar curvature:
  P_{ab} = 1/(d-2) (R_{ab} - R g_{ab} / (2(d-1)))
"""
function schouten_to_ricci(a::TIndex, b::TIndex, metric::Symbol; dim::Int=4)
    coeff_outer = 1 // (dim - 2)
    coeff_scalar = 1 // (2 * (dim - 1))
    Ric = Tensor(:Ric, [a, b])
    R = Tensor(:RicScalar, TIndex[])
    g = Tensor(metric, [a, b])
    coeff_outer * (Ric - coeff_scalar * R * g)
end

"""
    ricci_to_schouten(a, b, metric; dim=4) -> TensorExpr

Ricci tensor in terms of Schouten tensor:
  R_{ab} = (d-2) P_{ab} + g_{ab} R / (2(d-1))
"""
function ricci_to_schouten(a::TIndex, b::TIndex, metric::Symbol; dim::Int=4)
    coeff_sch = (dim - 2) // 1
    coeff_scalar = 1 // (2 * (dim - 1))
    Sch = Tensor(:Sch, [a, b])
    R = Tensor(:RicScalar, TIndex[])
    g = Tensor(metric, [a, b])
    coeff_sch * Sch + coeff_scalar * R * g
end

# ─── Trace-free Ricci ───────────────────────────────────────────────

"""
    tfricci_expr(a, b, metric; dim=4) -> TensorExpr

Trace-free Ricci tensor:
  S_{ab} = R_{ab} - (1/d) g_{ab} R
"""
function tfricci_expr(a::TIndex, b::TIndex, metric::Symbol; dim::Int=4)
    Ric = Tensor(:Ric, [a, b])
    R = Tensor(:RicScalar, TIndex[])
    g = Tensor(metric, [a, b])
    Ric - (1 // dim) * g * R
end

"""
    ricci_to_tfricci(a, b, metric; dim=4) -> TensorExpr

Ricci tensor in terms of trace-free Ricci:
  R_{ab} = S_{ab} + (1/d) g_{ab} R
"""
function ricci_to_tfricci(a::TIndex, b::TIndex, metric::Symbol; dim::Int=4)
    S = Tensor(:TFRic, [a, b])
    R = Tensor(:RicScalar, TIndex[])
    g = Tensor(metric, [a, b])
    S + (1 // dim) * g * R
end

# ─── Unified conversion functions ───────────────────────────────────

"""
    to_riemann(expr; metric=:g, dim=4) -> TensorExpr

Convert all curvature tensors (Weyl, Schouten, Einstein, TFRicci) to
Riemann + Ricci + RicciScalar + metric form.

Walks the expression and replaces each known curvature tensor with its
Riemann-based expression.
"""
function to_riemann(expr::TensorExpr; metric::Symbol=:g, dim::Int=4)
    walk(expr) do node
        node isa Tensor || return node
        if node.name == :Weyl && length(node.indices) == 4
            a, b, c, d = node.indices
            # C_{abcd} = R_{abcd} - (Ricci decomposition terms)
            weyl_to_riemann(a, b, c, d, metric; dim=dim)
        elseif node.name == :Sch && length(node.indices) == 2
            a, b = node.indices
            # P_{ab} = 1/(d-2)(R_{ab} - ...) — express as Ricci
            schouten_to_ricci(a, b, metric; dim=dim)
        elseif node.name == :Ein && length(node.indices) == 2
            a, b = node.indices
            einstein_to_ricci(a, b, metric)
        elseif node.name == :TFRic && length(node.indices) == 2
            a, b = node.indices
            # S_{ab} = R_{ab} - (1/d) g_{ab} R
            tfricci_expr(a, b, metric; dim=dim)
        else
            node
        end
    end
end

"""
    to_ricci(expr; metric=:g, dim=4) -> TensorExpr

Convert all curvature tensors to Ricci + RicciScalar + metric form.

Replaces Weyl, Riemann (decomposed), Schouten, Einstein, and TFRicci
with expressions involving only Ricci tensor, Ricci scalar, and the metric.
Also applies `contract_curvature` to detect traces of Riemann → Ricci.
"""
function to_ricci(expr::TensorExpr; metric::Symbol=:g, dim::Int=4)
    # First pass: replace non-Riemann curvature tensors
    result = walk(expr) do node
        node isa Tensor || return node
        if node.name == :Weyl && length(node.indices) == 4
            a, b, c, d = node.indices
            # Express Weyl in terms of Riemann, then Riemann → Ricci decomposition
            # C = R - (Ricci terms), so R = C + (Ricci terms)
            # But we want Ricci, so Weyl = R - (Ricci terms)
            # We need to express Weyl directly in Ricci form:
            # From riemann_to_weyl: R = C + f(Ric, R, g)
            # So C = R - f(Ric, R, g), but R_{abcd} itself cannot be reduced
            # to Ricci without additional info. Weyl IS the traceless part.
            # In to_ricci, Weyl stays unless it has traces — but typically
            # Weyl is traceless so we just return it as-is here and let
            # contract_curvature handle any traced products later.
            node
        elseif node.name == :Sch && length(node.indices) == 2
            a, b = node.indices
            schouten_to_ricci(a, b, metric; dim=dim)
        elseif node.name == :Ein && length(node.indices) == 2
            a, b = node.indices
            einstein_to_ricci(a, b, metric)
        elseif node.name == :TFRic && length(node.indices) == 2
            a, b = node.indices
            tfricci_expr(a, b, metric; dim=dim)
        else
            node
        end
    end

    # Second pass: contract traces of Riemann → Ricci, Ricci → RicciScalar
    contract_curvature(result)
end

# ─── Riemann ↔ Christoffel ─────────────────────────────────────────

"""
    riemann_to_christoffel(a::TIndex, b::TIndex, c::TIndex, d::TIndex,
                           christoffel::Symbol) -> TensorExpr

Express R^a_{bcd} in terms of Christoffel symbols:
R^a_{bcd} = ∂_c Γ^a_{db} - ∂_d Γ^a_{cb} + Γ^a_{ce} Γ^e_{db} - Γ^a_{de} Γ^e_{cb}
"""
function riemann_to_christoffel(a::TIndex, b::TIndex, c::TIndex, d::TIndex,
                                 christoffel::Symbol)
    @assert a.position == Up
    used = Set{Symbol}([a.name, b.name, c.name, d.name])
    e = fresh_index(used)

    Γ_adb = Tensor(christoffel, [a, d, b])
    Γ_acb = Tensor(christoffel, [a, c, b])
    Γ_ace = Tensor(christoffel, [a, c, down(e)])
    Γ_edb = Tensor(christoffel, [up(e), d, b])
    Γ_ade = Tensor(christoffel, [a, d, down(e)])
    Γ_ecb = Tensor(christoffel, [up(e), c, b])

    TDeriv(c, Γ_adb) - TDeriv(d, Γ_acb) + Γ_ace * Γ_edb - Γ_ade * Γ_ecb
end

"""
    kretschmann_expr(metric::Symbol; dim::Int=4) -> TensorExpr

Create the Kretschmann scalar K = R_{abcd} R^{abcd} as an abstract expression.
"""
function kretschmann_expr(metric::Symbol; dim::Int=4)
    a, b, c, d = down(:a), down(:b), down(:c), down(:d)
    e, f, g, h = up(:e), up(:f), up(:g), up(:h)
    Riem_down = Tensor(:Riem, [a, b, c, d])
    Riem_up = Tensor(:Riem, [e, f, g, h])
    g_ae = Tensor(metric, [a, e])
    g_bf = Tensor(metric, [b, f])
    g_cg = Tensor(metric, [c, g])
    g_dh = Tensor(metric, [d, h])
    Riem_down * g_ae * g_bf * g_cg * g_dh * Riem_up
end
