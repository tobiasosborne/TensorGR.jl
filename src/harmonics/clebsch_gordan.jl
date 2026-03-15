# ── Clebsch-Gordan coefficients and harmonic product algebra ──────────────

"""
    _bigfact(n::Integer) -> BigInt

Factorial using BigInt arithmetic for exact intermediate precision.
"""
function _bigfact(n::Integer)
    n < 0 && return zero(BigInt)
    factorial(BigInt(n))
end

"""
    _triangle_coeff(j1, j2, j3) -> Rational{BigInt}

Triangle coefficient Delta(j1,j2,j3) = (j1+j2-j3)!(j1-j2+j3)!(-j1+j2+j3)! / (j1+j2+j3+1)!
Returns 0 if the triangle inequality is violated.
"""
function _triangle_coeff(j1::Integer, j2::Integer, j3::Integer)
    a, b, c = j1 + j2 - j3, j1 - j2 + j3, -j1 + j2 + j3
    (a < 0 || b < 0 || c < 0) && return Rational{BigInt}(0)
    _bigfact(a) * _bigfact(b) * _bigfact(c) // _bigfact(j1 + j2 + j3 + 1)
end

"""
    wigner3j(j1, j2, j3, m1, m2, m3) -> Float64

Compute the Wigner 3j symbol using the Racah formula.
Uses BigInt factorials internally for precision, returns Float64.

Selection rules (return 0 immediately if violated):
- m1 + m2 + m3 != 0
- Triangle inequality: |j1 - j2| <= j3 <= j1 + j2
- |mi| > ji for any i
"""
function wigner3j(j1::Integer, j2::Integer, j3::Integer,
                  m1::Integer, m2::Integer, m3::Integer)
    # Selection rules
    m1 + m2 + m3 != 0 && return 0.0
    abs(m1) > j1 && return 0.0
    abs(m2) > j2 && return 0.0
    abs(m3) > j3 && return 0.0
    j3 < abs(j1 - j2) && return 0.0
    j3 > j1 + j2 && return 0.0
    any(j -> j < 0, (j1, j2, j3)) && return 0.0

    # Triangle coefficient
    tri = _triangle_coeff(j1, j2, j3)
    tri == 0 && return 0.0

    # Prefactor squared: Delta * product of (ji+mi)!(ji-mi)!
    pf_sq = tri
    for (j, m) in ((j1, m1), (j2, m2), (j3, m3))
        pf_sq *= _bigfact(j + m) * _bigfact(j - m)
    end

    # Racah sum over t
    t_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    t_min > t_max && return 0.0

    s = Rational{BigInt}(0)
    for t in t_min:t_max
        denom = _bigfact(t) *
                _bigfact(j3 - j2 + t + m1) *
                _bigfact(j3 - j1 + t - m2) *
                _bigfact(j1 + j2 - j3 - t) *
                _bigfact(j1 - t - m1) *
                _bigfact(j2 - t + m2)
        s += (iseven(t) ? 1 : -1) // denom
    end

    phase = iseven(j1 - j2 + m3) ? 1 : -1
    Float64(phase) * sqrt(Float64(pf_sq)) * Float64(s)
end

"""
    clebsch_gordan(j1, m1, j2, m2, J, M) -> Float64

Clebsch-Gordan coefficient <j1,m1; j2,m2 | J,M>.
Related to the Wigner 3j symbol by:
    CG = (-1)^(j1 - j2 + M) * sqrt(2J + 1) * (j1 j2 J; m1 m2 -M)
"""
function clebsch_gordan(j1::Integer, m1::Integer, j2::Integer, m2::Integer,
                        J::Integer, M::Integer)
    m1 + m2 != M && return 0.0
    phase = iseven(j1 - j2 + M) ? 1.0 : -1.0
    phase * sqrt(2J + 1) * wigner3j(j1, j2, J, m1, m2, -M)
end

"""
    harmonic_product(Y1::ScalarHarmonic, Y2::ScalarHarmonic) -> TensorExpr

Expand Y_{l1,m1} * Y_{l2,m2} as a sum over Y_{l3,m3} using Gaunt coefficients.

The Gaunt integral gives:
    Y_{l1,m1} Y_{l2,m2} = sum_{l3} c_{l3} Y_{l3,m3}

where m3 = m1 + m2 and
    c_{l3} = (-1)^m3 * sqrt((2l1+1)(2l2+1)(2l3+1) / (4pi))
             * (l1 l2 l3; 0 0 0) * (l1 l2 l3; m1 m2 -m3)

The (-1)^m3 phase arises from conj(Y_{l3,m3}) = (-1)^m3 Y_{l3,-m3} in the
projection integral c_{l3} = integral Y_{l1,m1} Y_{l2,m2} conj(Y_{l3,m3}) dOmega.

Selection rules:
- m3 = m1 + m2
- |l1 - l2| <= l3 <= l1 + l2
- l1 + l2 + l3 must be even (parity)
"""
function harmonic_product(Y1::ScalarHarmonic, Y2::ScalarHarmonic)
    l1, m1 = Y1.l, Y1.m
    l2, m2 = Y2.l, Y2.m
    m3 = m1 + m2

    terms = TensorExpr[]
    for l3 in abs(l1 - l2):l1+l2
        # Parity selection: l1 + l2 + l3 must be even
        isodd(l1 + l2 + l3) && continue
        # m3 must satisfy |m3| <= l3
        abs(m3) > l3 && continue

        # Gaunt coefficient with conjugation phase
        phase = iseven(m3) ? 1.0 : -1.0
        prefactor = sqrt((2l1 + 1) * (2l2 + 1) * (2l3 + 1) / (4pi))
        w3j_zero = wigner3j(l1, l2, l3, 0, 0, 0)
        w3j_m = wigner3j(l1, l2, l3, m1, m2, -m3)
        coeff = phase * prefactor * w3j_zero * w3j_m

        abs(coeff) < 1e-15 && continue
        push!(terms, tproduct(1 // 1, TensorExpr[TScalar(coeff), ScalarHarmonic(l3, m3)]))
    end

    isempty(terms) && return TScalar(0)
    length(terms) == 1 && return terms[1]
    tsum(terms)
end
