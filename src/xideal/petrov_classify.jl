#= Petrov classification from invariants I, J and Weyl scalars.
#
# Determines the algebraic (Petrov) type of the Weyl tensor:
# I, II, III, D, N, O. Uses the decision tree from Stephani et al.
# (2003), Table 4.3, with disambiguation via the Q-matrix rank
# (Penrose & Rindler Vol 1, Sec 8.7).
#
# References:
#   Stephani et al., "Exact Solutions" (2003), Table 4.3
#   Penrose & Rindler, "Spinors and Space-time" Vol 1 (1984), Sec 8.7
#   d'Inverno, "Introducing Einstein's Relativity" (1992), Ch 6.5
=#

using LinearAlgebra: eigvals, svdvals

"""
    PetrovType

Enum for the six Petrov types of the Weyl tensor.
- `TypeI`   -- algebraically general, 4 distinct principal null directions
- `TypeII`  -- 1 double + 2 simple PNDs
- `TypeD`   -- 2 double PNDs (e.g. Schwarzschild, Kerr)
- `TypeIII` -- 1 triple + 1 simple PND
- `TypeN`   -- 1 quadruple PND (e.g. pp-waves)
- `TypeO`   -- conformally flat (Weyl = 0, e.g. Minkowski, FRW)
"""
@enum PetrovType TypeI TypeII TypeIII TypeD TypeN TypeO

"""
    petrov_classify(Weyl::Array{T,4}, g::Matrix; dim::Int=4, atol=1e-10) where T
        -> PetrovType

Classify the Petrov type from the all-down Weyl tensor C_{abcd} and
the metric g_{ab}. Constructs a null tetrad internally, computes the
NP scalars and invariants, then applies the decision tree.

Requires `dim=4` and a diagonal metric.
"""
function petrov_classify(Weyl::Array{T,4}, g::Matrix; dim::Int=4, atol=1e-10) where T
    dim == 4 || error("petrov_classify requires dim=4")
    W = T === Any ? Float64.(Weyl) : Weyl
    if _weyl_is_zero(W, atol)
        return TypeO
    end
    g_ct = CTensor(Float64.(g), :_petrov_classify)
    l, n, m, mbar = null_tetrad_from_metric(g_ct)
    psi = weyl_scalars(W, l, n, m, mbar)
    petrov_classify(psi; atol=atol)
end

"""
    petrov_classify(psi::NamedTuple; atol=1e-10) -> PetrovType

Classify the Petrov type from pre-computed Newman-Penrose Weyl scalars
`(Psi0, Psi1, Psi2, Psi3, Psi4)`.

Decision tree (Stephani et al., Table 4.3):
1. All Psi_k = 0 -> Type O
2. Compute I, J. If I = J = 0 -> disambiguate III vs N
3. If I^3 = 27 J^2 (algebraically special, I != 0) -> disambiguate II vs D
4. Otherwise -> Type I
"""
function petrov_classify(psi::NamedTuple{(:Psi0, :Psi1, :Psi2, :Psi3, :Psi4)}; atol=1e-10)
    P0, P1, P2, P3, P4 = psi.Psi0, psi.Psi1, psi.Psi2, psi.Psi3, psi.Psi4

    # Step 1: Type O (all scalars vanish)
    if _all_zero(P0, P1, P2, P3, P4; atol=atol)
        return TypeO
    end

    # Step 2: Compute invariants
    inv = petrov_invariants(psi)
    Ival, Jval = inv.I, inv.J

    # Scale tolerance for I,J zero-check relative to Weyl scalar magnitudes.
    # I is quadratic in Psi_k, J is cubic, so scale by Psi_max^2 and Psi_max^3.
    psi_max = max(abs(P0), abs(P1), abs(P2), abs(P3), abs(P4))
    I_scale = max(abs(Ival), atol * psi_max^2)
    J_scale = max(abs(Jval), atol * psi_max^3)

    # Step 3: I = J = 0 -> III or N
    if abs(Ival) <= atol * psi_max^2 && abs(Jval) <= atol * psi_max^3
        return _disambiguate_III_N(P0, P1, P2, P3, P4; atol=atol)
    end

    # Step 4: Algebraically special (I^3 = 27 J^2) -> II or D
    if is_algebraically_special(Ival, Jval; atol=atol)
        return _disambiguate_II_D(P0, P1, P2, P3, P4; atol=atol)
    end

    # Step 5: Algebraically general
    return TypeI
end

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

"""
    _weyl_is_zero(Weyl::Array, atol) -> Bool

Check if all components of the Weyl tensor are below tolerance.
"""
function _weyl_is_zero(Weyl::Array, atol)
    for v in Weyl
        abs(v) > atol && return false
    end
    return true
end

_is_zero(x; atol=1e-10) = abs(x) <= atol

_all_zero(args...; atol=1e-10) = all(x -> abs(x) <= atol, args)

"""
    _disambiguate_II_D(Psi0, Psi1, Psi2, Psi3, Psi4; atol=1e-10) -> PetrovType

Distinguish Type II from Type D using Q-matrix eigenvalue multiplicity.
Q = [Psi0 Psi1 Psi2; Psi1 Psi2 Psi3; Psi2 Psi3 Psi4].
Type D: Q has an eigenvalue with algebraic multiplicity >= 2.
Type II: Q has 3 distinct eigenvalues.

See Penrose & Rindler (1984), Sec 8.7; d'Inverno (1992), Ch 6.5.
"""
function _disambiguate_II_D(P0, P1, P2, P3, P4; atol=1e-10)
    Q = [P0 P1 P2; P1 P2 P3; P2 P3 P4]
    evals = eigvals(Q)
    # Check if any pair of eigenvalues coincides -> Type D
    for i in 1:length(evals), j in (i+1):length(evals)
        scale = max(abs(evals[i]), abs(evals[j]), one(real(eltype(evals))))
        if abs(evals[i] - evals[j]) <= atol * scale
            return TypeD
        end
    end
    return TypeII
end

"""
    _disambiguate_III_N(Psi0, Psi1, Psi2, Psi3, Psi4; atol=1e-10) -> PetrovType

Distinguish Type III from Type N when I = J = 0.

Type N: all 2x2 minors of Q vanish (Q has rank <= 1 with Hankel
structure forcing a single nonzero corner entry in canonical frame).
Type III: at least one 2x2 minor of Q is nonzero.

See Stephani et al. (2003), Table 4.3; Penrose & Rindler (1984), Sec 8.7.
"""
function _disambiguate_III_N(P0, P1, P2, P3, P4; atol=1e-10)
    Q = [P0 P1 P2; P1 P2 P3; P2 P3 P4]
    # Check all 2x2 minors of the 3x3 Q matrix
    for i in 1:3, j in (i+1):3, k in 1:3, l in (k+1):3
        minor = Q[i,k] * Q[j,l] - Q[i,l] * Q[j,k]
        abs(minor) > atol && return TypeIII
    end
    return TypeN
end

"""
    _matrix_rank(M::Matrix; atol=1e-10) -> Int

Numerical rank of a matrix via SVD.
"""
function _matrix_rank(M::Matrix; atol=1e-10)
    sv = svdvals(M)
    isempty(sv) && return 0
    tol = max(atol, atol * maximum(abs, sv))
    count(s -> abs(s) > tol, sv)
end
