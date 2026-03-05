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
