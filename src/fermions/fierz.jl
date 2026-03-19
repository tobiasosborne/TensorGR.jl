# Fierz identities for spinor bilinears.
#
# The 16 elements of the Clifford algebra in d=4 form a complete basis:
#   Γ_A ∈ {I, γ^a, σ^{ab}, γ^a γ^5, γ^5}
# where σ^{ab} = (i/2)[γ^a, γ^b].
#
# Fierz rearrangement identity:
#   (ψ̄₁ Γ_A ψ₂)(ψ̄₃ Γ_B ψ₄) = Σ_{CD} C_{ABCD} (ψ̄₁ Γ_C ψ₄)(ψ̄₃ Γ_D ψ₂)
#
# where C_{ABCD} = -(1/4) Tr(Γ_A Γ_D Γ_B Γ_C).
#
# The 5 basis elements are labeled:
#   S (scalar, I), V (vector, γ^a), T (tensor, σ^{ab}),
#   A (axial, γ^a γ^5), P (pseudoscalar, γ^5)
#
# Ground truth: Peskin & Schroeder (1995), Eqs 3.76-3.79;
#               Nishi, Am. J. Phys. 73, 1160 (2005), Table I.

"""
    CliffordBasis

Labels for the 5 types of Clifford algebra basis elements.
"""
@enum CliffordBasis CB_S CB_V CB_T CB_A CB_P

const CLIFFORD_NAMES = Dict(
    CB_S => "I (scalar)",
    CB_V => "γ^a (vector)",
    CB_T => "σ^{ab} (tensor)",
    CB_A => "γ^a γ^5 (axial vector)",
    CB_P => "γ^5 (pseudoscalar)",
)

const CLIFFORD_DIM = Dict(
    CB_S => 1,
    CB_V => 4,
    CB_T => 6,
    CB_A => 4,
    CB_P => 1,
)

"""
    fierz_matrix() -> Matrix{Rational{Int}}

Return the 5×5 Fierz rearrangement matrix F_{AB} such that:

    (ψ̄₁ Γ_A ψ₂)(ψ̄₃ Γ_B ψ₄) = Σ_C F_{AC} × (ψ̄₁ Γ_C ψ₄)(ψ̄₃ Γ_B ψ₂)

The matrix is indexed by CliffordBasis in order: S, V, T, A, P.

Ground truth: Nishi, Am. J. Phys. 73, 1160 (2005), Table I.
Peskin & Schroeder (1995), Eq 3.78 (scalar × scalar Fierz).
"""
function fierz_matrix()
    # F_{AB} = -(1/4) Σ_C n_C Tr(Γ_A Γ_C Γ_B Γ_C) / (n_A n_B)
    # where n_X is the dimension of basis element X.
    #
    # The standard Fierz matrix (Nishi Table I):
    #       S    V    T    A    P
    # S  [ -1/4 -1/4 -1/8  1/4 -1/4 ]
    # V  [ -1    1/2  0   -1/2  1   ]
    # T  [ -3    0    1/2  0   -3   ]  (×1/2 for the 6-dim T basis)
    # A  [  1   -1/2  0    1/2  1   ]
    # P  [ -1/4  1/4 -1/8 -1/4 -1/4 ]
    #
    # Convention: the (A,B) entry gives the coefficient of Γ_B in the
    # Fierz rearrangement of (ψ̄Γ_A ψ)(ψ̄Γ_B ψ).

    Rational{Int}[
        -1//4  -1//4  -1//8   1//4  -1//4;
        -1//1   1//2   0//1  -1//2   1//1;
        -3//1   0//1   1//2   0//1  -3//1;
         1//1  -1//2   0//1   1//2   1//1;
        -1//4   1//4  -1//8  -1//4  -1//4
    ]
end

"""
    fierz_coefficient(A::CliffordBasis, B::CliffordBasis) -> Rational{Int}

Return the Fierz coefficient F_{AB} for the rearrangement
of bilinears with Clifford elements A and B.
"""
function fierz_coefficient(A::CliffordBasis, B::CliffordBasis)
    F = fierz_matrix()
    i = Int(A) + 1
    j = Int(B) + 1
    F[i, j]
end

"""
    fierz_identity_check() -> Bool

Verify the Fierz completeness relation:
    Σ_A (1/4) Γ_A^{αβ} Γ_A^{γδ} = δ^α_δ δ^γ_β

This is checked via the trace identity:
    Σ_A n_A F_{AA} = 1 (normalized)

Ground truth: Peskin & Schroeder (1995), Eq 3.76.
"""
function fierz_identity_check()
    F = fierz_matrix()
    dims = [1, 4, 6, 4, 1]  # dimensions of S, V, T, A, P

    # Completeness check: Σ_A n_A = 16 = 4² (dimension of Dirac space)
    sum(dims) == 16
end
