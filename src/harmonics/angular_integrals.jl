# -- Angular integration engine for products of tensor spherical harmonics ------
#
# Computes integrals of the form integral H1 * H2 * H3* dOmega for arbitrary
# combinations of scalar, vector, and tensor harmonics on S^2.
#
# All integrals reduce to products of Wigner 3j symbols (Gaunt coefficients)
# times normalization factors from the harmonic definitions.
#
# Ground truth: Martel & Poisson, PRD 71, 104003 (2005), Appendix A;
#               Brizuela, Martin-Garcia & Tiglio, PRD 80, 024021 (2009), App B;
#               Gleiser, Nicasio, Price & Pullin, Phys. Rep. 325, 41 (2000), App B.

# ---- Gaunt integral (scalar triple) -----------------------------------------

"""
    gaunt_integral(l1, m1, l2, m2, l3, m3) -> Float64

Gaunt integral: integral Y_{l1,m1} Y_{l2,m2} Y*_{l3,m3} dOmega.

This is the fundamental building block for all angular integrals. Using
Y*_{lm} = (-1)^m Y_{l,-m}, the integral becomes:

    (-1)^{m3} sqrt((2l1+1)(2l2+1)(2l3+1)/(4pi))
        * (l1 l2 l3; 0 0 0) * (l1 l2 l3; m1 m2 -m3)

Selection rules (returns 0 if violated):
- m1 + m2 = m3
- |l1 - l2| <= l3 <= l1 + l2
- l1 + l2 + l3 must be even (parity)
"""
function gaunt_integral(l1::Integer, m1::Integer, l2::Integer, m2::Integer,
                        l3::Integer, m3::Integer)
    # Selection rule: m1 + m2 = m3 (from conjugation of Y*_{l3,m3})
    m1 + m2 != m3 && return 0.0
    # Triangle inequality
    l3 < abs(l1 - l2) && return 0.0
    l3 > l1 + l2 && return 0.0
    # Parity: l1 + l2 + l3 must be even
    isodd(l1 + l2 + l3) && return 0.0

    phase = iseven(m3) ? 1.0 : -1.0
    prefactor = sqrt((2l1 + 1) * (2l2 + 1) * (2l3 + 1) / (4pi))
    w3j_zero = wigner3j(l1, l2, l3, 0, 0, 0)
    w3j_m = wigner3j(l1, l2, l3, m1, m2, -m3)

    phase * prefactor * w3j_zero * w3j_m
end

# ---- Vector Gaunt integral ---------------------------------------------------

"""
    vector_gaunt(l1, m1, l2, m2, l3, m3) -> Float64

Vector harmonic triple integral:
    integral Y^A_{l1,m1} Y_{A,l2,m2} Y*_{l3,m3} dOmega

where Y^A_{lm} = D^A Y_{lm} is the even-parity vector harmonic.

By identity D^A Y_{l1} D_A Y_{l2} = 1/2 [l1(l1+1) + l2(l2+1) - l3(l3+1)] Y_{l1} Y_{l2}
(after integration by parts on S^2 and using the addition theorem), the integral
reduces to:

    1/2 [l1(l1+1) + l2(l2+1) - l3(l3+1)] * gaunt_integral(l1,m1,l2,m2,l3,m3)

Selection rules: same as gaunt_integral plus l1 >= 1 and l2 >= 1.

Ground truth: Martel & Poisson (2005) Eq A.1; Brizuela et al. (2006) Eq B.2.
"""
function vector_gaunt(l1::Integer, m1::Integer, l2::Integer, m2::Integer,
                      l3::Integer, m3::Integer)
    # Vector harmonics require l >= 1
    (l1 < 1 || l2 < 1) && return 0.0

    g = gaunt_integral(l1, m1, l2, m2, l3, m3)
    abs(g) < 1e-15 && return 0.0

    # Angular momentum coupling coefficient
    coupling = (l1 * (l1 + 1) + l2 * (l2 + 1) - l3 * (l3 + 1)) / 2
    coupling * g
end

# ---- Tensor Gaunt integral ---------------------------------------------------

"""
    tensor_gaunt(l1, m1, l2, m2, l3, m3, type1::Symbol, type2::Symbol) -> Float64

Tensor harmonic triple integral:
    integral T1^{AB}_{l1,m1} T2_{AB,l2,m2} Y*_{l3,m3} dOmega

where T1, T2 are tensor harmonics of specified type (:Y, :Z, or :X).

The reduction formulas are:
- Y-Y (metric x metric): Omega^{AB} Omega_{AB} = 2, so
    integral Y^{AB}_{l1} Y_{AB,l2} Y*_{l3} = 2 * gaunt(l1,m1,l2,m2,l3,m3)

- Z-Z (STF x STF): reduces via the identity for D^A D^B products to
    integral Z^{AB}_{l1} Z_{AB,l2} Y*_{l3} = Q(l1,l2,l3) * gaunt(l1,m1,l2,m2,l3,m3)
  where Q = 1/2 [(l1(l1+1) + l2(l2+1) - l3(l3+1))^2/2
              - l1(l1+1) - l2(l2+1)] + 1/4 l1(l1+1)*l2(l2+1)

  which simplifies to:
    Q = 1/4 * [ (L1 + L2 - L3)^2 - 2(L1 + L2) + 2*L1*L2 ]
  with L_i = l_i(l_i+1).

  Equivalently (Gleiser et al. Eq B.12):
    Q = 1/4 * [(L1+L2-L3)^2/2 + L1*L2 - L1 - L2]
      = 1/8 * (L1+L2-L3)^2 + 1/4*(L1*L2 - L1 - L2)

- X-X (odd STF x odd STF): same norm as Z-Z by parity symmetry, so
    integral X^{AB}_{l1} X_{AB,l2} Y*_{l3} = Q(l1,l2,l3) * gaunt(l1,m1,l2,m2,l3,m3)

- Cross-type (Y-Z, Y-X, Z-X): vanish by trace/parity orthogonality.

Ground truth: Gleiser et al. (2000) Appendix B, Eq B.12;
              Brizuela et al. (2009) Appendix A.
"""
function tensor_gaunt(l1::Integer, m1::Integer, l2::Integer, m2::Integer,
                      l3::Integer, m3::Integer,
                      type1::Symbol, type2::Symbol)
    # Cross-type integrals vanish
    # Y-Z and Z-Y: trace-free Z contracted with metric Y gives trace of Z = 0
    if (type1 == :Y && type2 == :Z) || (type1 == :Z && type2 == :Y)
        return 0.0
    end
    # Y-X and X-Y: different parity
    if (type1 == :Y && type2 == :X) || (type1 == :X && type2 == :Y)
        return 0.0
    end
    # Z-X and X-Z: different parity
    if (type1 == :Z && type2 == :X) || (type1 == :X && type2 == :Z)
        return 0.0
    end

    # Y^{AB} (metric type) requires l >= 0; Z^{AB}, X^{AB} require l >= 2
    if type1 == :Y
        l1 < 0 && return 0.0
    else
        l1 < 2 && return 0.0
    end
    if type2 == :Y
        l2 < 0 && return 0.0
    else
        l2 < 2 && return 0.0
    end

    g = gaunt_integral(l1, m1, l2, m2, l3, m3)
    abs(g) < 1e-15 && return 0.0

    if type1 == :Y && type2 == :Y
        # metric x metric: factor of 2 from Omega^{AB} Omega_{AB} = 2
        return 2.0 * g
    end

    # Z-Z or X-X: STF coupling coefficient
    # Q(l1,l2,l3) from Gleiser et al. Eq B.12
    L1 = l1 * (l1 + 1)
    L2 = l2 * (l2 + 1)
    L3 = l3 * (l3 + 1)
    dL = L1 + L2 - L3  # = 2 * vector_coupling
    Q = dL^2 / 8 + (L1 * L2 - L1 - L2) / 4

    Q * g
end

# ---- Selection rules ---------------------------------------------------------

"""
    angular_selection_rule(l1, l2, l3) -> Bool

Check whether the triangle inequality and parity selection rule are satisfied
for a triple integral of harmonics with angular momenta l1, l2, l3.

Returns `true` if the integral can be nonzero:
- Triangle inequality: |l1 - l2| <= l3 <= l1 + l2
- Parity: l1 + l2 + l3 must be even
"""
function angular_selection_rule(l1::Integer, l2::Integer, l3::Integer)
    l3 < abs(l1 - l2) && return false
    l3 > l1 + l2 && return false
    isodd(l1 + l2 + l3) && return false
    true
end

# ---- Generic angular_integral dispatch ---------------------------------------

"""
    angular_integral(H1, H2) -> TScalar

Two-harmonic angular integral (inner product on S^2).
Delegates to `inner_product` for scalar harmonics and
`vector_inner_product`/`tensor_inner_product` for higher-rank harmonics.

Returns TScalar with the integral value.
"""
angular_integral(y1::ScalarHarmonic, y2::ScalarHarmonic) = inner_product(y1, y2)
angular_integral(y1::EvenVectorHarmonic, y2::EvenVectorHarmonic) = vector_inner_product(y1, y2)
angular_integral(y1::OddVectorHarmonic, y2::OddVectorHarmonic) = vector_inner_product(y1, y2)
angular_integral(y1::EvenVectorHarmonic, y2::OddVectorHarmonic) = TScalar(0)
angular_integral(y1::OddVectorHarmonic, y2::EvenVectorHarmonic) = TScalar(0)
angular_integral(t1::T, t2::T) where {T<:_TensorHarmonic} = tensor_inner_product(t1, t2)
angular_integral(t1::EvenTensorHarmonicY, t2::EvenTensorHarmonicZ) = TScalar(0)
angular_integral(t1::EvenTensorHarmonicZ, t2::EvenTensorHarmonicY) = TScalar(0)
angular_integral(t1::EvenTensorHarmonicY, t2::OddTensorHarmonic) = TScalar(0)
angular_integral(t1::OddTensorHarmonic, t2::EvenTensorHarmonicY) = TScalar(0)
angular_integral(t1::EvenTensorHarmonicZ, t2::OddTensorHarmonic) = TScalar(0)
angular_integral(t1::OddTensorHarmonic, t2::EvenTensorHarmonicZ) = TScalar(0)

"""
    angular_integral(H1, H2, H3) -> Float64

Triple angular integral integral H1 * H2 * H3* dOmega for products of
scalar, vector, and tensor spherical harmonics.

Dispatches to `gaunt_integral`, `vector_gaunt`, or `tensor_gaunt`
depending on the harmonic types. The third argument H3 is conjugated
(consistent with projection integral conventions).

Supported combinations:
- Three scalars: gaunt_integral
- Two vectors + one scalar: vector_gaunt (index-contracted vectors)
- Two tensors + one scalar: tensor_gaunt (index-contracted tensors)
- Scalar-vector mixed (no index contraction): vanishes by parity
"""
# Scalar-Scalar-Scalar
function angular_integral(y1::ScalarHarmonic, y2::ScalarHarmonic, y3::ScalarHarmonic)
    gaunt_integral(y1.l, y1.m, y2.l, y2.m, y3.l, y3.m)
end

# Vector-Vector-Scalar (Y^A Y_A Y* -- contracted vector pair)
function angular_integral(y1::EvenVectorHarmonic, y2::EvenVectorHarmonic,
                           y3::ScalarHarmonic)
    vector_gaunt(y1.l, y1.m, y2.l, y2.m, y3.l, y3.m)
end

# Odd vector pair contracted with scalar
function angular_integral(x1::OddVectorHarmonic, x2::OddVectorHarmonic,
                           y3::ScalarHarmonic)
    # X^A X_A has same coupling as Y^A Y_A by parity symmetry on S^2
    vector_gaunt(x1.l, x1.m, x2.l, x2.m, y3.l, y3.m)
end

# Cross-parity vector pair: Y^A X_A Y* vanishes
function angular_integral(::EvenVectorHarmonic, ::OddVectorHarmonic, ::ScalarHarmonic)
    0.0
end
function angular_integral(::OddVectorHarmonic, ::EvenVectorHarmonic, ::ScalarHarmonic)
    0.0
end

# Scalar-Vector (no contraction) vanishes by parity on S^2
function angular_integral(::ScalarHarmonic, ::_VectorHarmonic, ::ScalarHarmonic)
    0.0
end
function angular_integral(::_VectorHarmonic, ::ScalarHarmonic, ::ScalarHarmonic)
    0.0
end

# Tensor-Tensor-Scalar (T^{AB} T_{AB} Y* -- contracted tensor pair)
function angular_integral(t1::EvenTensorHarmonicY, t2::EvenTensorHarmonicY,
                           y3::ScalarHarmonic)
    tensor_gaunt(t1.l, t1.m, t2.l, t2.m, y3.l, y3.m, :Y, :Y)
end

function angular_integral(t1::EvenTensorHarmonicZ, t2::EvenTensorHarmonicZ,
                           y3::ScalarHarmonic)
    tensor_gaunt(t1.l, t1.m, t2.l, t2.m, y3.l, y3.m, :Z, :Z)
end

function angular_integral(t1::OddTensorHarmonic, t2::OddTensorHarmonic,
                           y3::ScalarHarmonic)
    tensor_gaunt(t1.l, t1.m, t2.l, t2.m, y3.l, y3.m, :X, :X)
end

# Cross-type tensor pairs vanish
function angular_integral(::EvenTensorHarmonicY, ::EvenTensorHarmonicZ, ::ScalarHarmonic)
    0.0
end
function angular_integral(::EvenTensorHarmonicZ, ::EvenTensorHarmonicY, ::ScalarHarmonic)
    0.0
end
function angular_integral(::EvenTensorHarmonicY, ::OddTensorHarmonic, ::ScalarHarmonic)
    0.0
end
function angular_integral(::OddTensorHarmonic, ::EvenTensorHarmonicY, ::ScalarHarmonic)
    0.0
end
function angular_integral(::EvenTensorHarmonicZ, ::OddTensorHarmonic, ::ScalarHarmonic)
    0.0
end
function angular_integral(::OddTensorHarmonic, ::EvenTensorHarmonicZ, ::ScalarHarmonic)
    0.0
end
