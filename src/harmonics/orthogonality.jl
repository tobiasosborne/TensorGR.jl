# ── Vector harmonic orthogonality relations ──────────────────────────────
#
# Ground truth: Martel & Poisson, Phys. Rev. D 71, 104003 (2005),
#               arXiv:gr-qc/0502028, Section II.C, Eqs 2.13--2.15.
#
#   Eq 2.13: integral Y_bar^A_{lm} Y_A^{l'm'} dOmega = l(l+1) delta_{ll'} delta_{mm'}
#   Eq 2.14: integral X_bar^A_{lm} X_A^{l'm'} dOmega = l(l+1) delta_{ll'} delta_{mm'}
#   Eq 2.15: integral Y_bar^A_{lm} X_A^{l'm'} dOmega = 0

"""
    vector_inner_product(Y1, Y2) -> TScalar

Compute the S^2 inner product integral Y1^a Y2*_a dOmega for vector harmonics.
Returns TScalar with the result.

Ground truth: Martel & Poisson (2005) Sec II.C, Eqs 2.13-2.15:
  integral Y^a_{lm} Y*_{a,l'm'} dOmega = l(l+1) delta_{ll'} delta_{mm'}
  integral X^a_{lm} X*_{a,l'm'} dOmega = l(l+1) delta_{ll'} delta_{mm'}
  integral Y^a_{lm} X*_{a,l'm'} dOmega = 0  (even-odd orthogonal)
"""
function vector_inner_product end

# Eq 2.13: even-even orthogonality
function vector_inner_product(y1::EvenVectorHarmonic, y2::EvenVectorHarmonic)
    (y1.l == y2.l && y1.m == y2.m) ? TScalar(y1.l * (y1.l + 1)) : TScalar(0)
end

# Eq 2.14: odd-odd orthogonality
function vector_inner_product(x1::OddVectorHarmonic, x2::OddVectorHarmonic)
    (x1.l == x2.l && x1.m == x2.m) ? TScalar(x1.l * (x1.l + 1)) : TScalar(0)
end

# Eq 2.15: even-odd cross-orthogonality (always zero)
vector_inner_product(::EvenVectorHarmonic, ::OddVectorHarmonic) = TScalar(0)
vector_inner_product(::OddVectorHarmonic, ::EvenVectorHarmonic) = TScalar(0)

# ── Extend inner_product to vector harmonics ─────────────────────────────

"""
    inner_product(Y1::EvenVectorHarmonic, Y2::EvenVectorHarmonic) -> TScalar

S^2 inner product for even vector harmonics.
Martel & Poisson (2005) Eq 2.13: l(l+1) delta_{ll'} delta_{mm'}.
"""
inner_product(y1::EvenVectorHarmonic, y2::EvenVectorHarmonic) = vector_inner_product(y1, y2)

"""
    inner_product(X1::OddVectorHarmonic, X2::OddVectorHarmonic) -> TScalar

S^2 inner product for odd vector harmonics.
Martel & Poisson (2005) Eq 2.14: l(l+1) delta_{ll'} delta_{mm'}.
"""
inner_product(x1::OddVectorHarmonic, x2::OddVectorHarmonic) = vector_inner_product(x1, x2)

"""
    inner_product(::EvenVectorHarmonic, ::OddVectorHarmonic) -> TScalar(0)

Even-odd cross inner product vanishes identically.
Martel & Poisson (2005) Eq 2.15.
"""
inner_product(y::EvenVectorHarmonic, x::OddVectorHarmonic) = TScalar(0)

"""
    inner_product(::OddVectorHarmonic, ::EvenVectorHarmonic) -> TScalar(0)

Odd-even cross inner product vanishes identically.
Martel & Poisson (2005) Eq 2.15.
"""
inner_product(x::OddVectorHarmonic, y::EvenVectorHarmonic) = TScalar(0)
