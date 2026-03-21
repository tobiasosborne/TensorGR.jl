#= Degree-3 algebraic invariant database for the Invar pipeline.
#
# Precomputed reduction relations for scalar cubic Riemann invariants:
# products of 3 Riemann tensors with all 12 indices contracted pairwise.
#
# Slot layout: factor 1 occupies slots 1-4, factor 2 occupies slots 5-8,
# factor 3 occupies slots 9-12.  The contraction is a fixed-point-free
# involution on [1..12]: sigma(i) is the slot paired with slot i via the metric.
#
# Raw count: (12-1)!! = 10395 pairings.
# After Level 1 (Riemann permutation symmetries): 13 non-vanishing canonical forms.
# After Level 2 (first Bianchi identity): 8 independent cubic invariants.
#
# The 13 non-vanishing canonical forms (Level 1):
#
#   I1:  [3,4,1,2, 7,8,5,6, 11,12,9,10]   R^3
#        Each factor self-contracts (1<->3, 2<->4) to give R.
#
#   I2:  [3,4,1,2, 7,9,5,11, 6,12,8,10]   R * R_{ab}R^{ab}
#        Factor 1: R (self-contracted).  Factors 2,3: Ric^2 (mixed contraction).
#
#   I3:  [3,4,1,2, 9,10,11,12, 5,6,7,8]   R * R_{abcd}R^{abcd}
#        Factor 1: R.  Factors 2,3: Kretschmann (full cross-contraction).
#
#   I4:  [3,4,1,2, 9,11,10,12, 5,7,6,8]   R * R_{acbd}R^{abcd}
#        Factor 1: R.  Factors 2,3: degree-2 cross contraction.
#        Dependent at Level 2: I4 = (1/2) I3.
#
#   I5:  [3,5,1,7, 2,9,4,11, 6,12,8,10]   R_{ab}R_{cd}R^{acbd}
#        Two Ricci traces with a Riemann cross contraction.
#
#   I6:  [3,5,1,9, 2,7,6,11, 4,12,8,10]   R_a^b R_b^c R_c^a  (Ricci cube trace)
#        Three self-contracted factors in a cyclic chain.
#
#   I7:  [3,5,1,9, 2,10,11,12, 4,6,7,8]   R_{ab}R^{acde}R^b_{cde}
#        One Ricci trace, two Riemanns cross-contracted.
#
#   I8:  [3,5,1,9, 2,11,10,12, 4,7,6,8]   R_{ab}R^{acef}R^{bd}_{ef}  (cross)
#        One Ricci trace, two Riemanns with mixed contraction.
#        Dependent at Level 2: I8 = (1/2) I7.
#
#   I9:  [5,6,9,10, 1,2,11,12, 3,4,7,8]   R_{ab}^{cd}R_{cd}^{ef}R_{ef}^{ab}
#        Full cross-contraction cycle (Goroff-Sagnotti invariant).
#
#   I10: [5,6,9,11, 1,2,10,12, 3,7,4,8]   R_{ab}^{ce}R_{cd}^{af}R_{ef}^{bd}
#        Riem^3 with one pair-swap in cross contraction.
#        Dependent at Level 2: I10 = (1/2) I9.
#
#   I11: [5,7,9,11, 1,10,2,12, 3,6,4,8]   R_{a}^{c}_{b}^{e}R_{c}^{a}_{d}^{f}R_{e}^{b}_{f}^{d}
#        Riem^3 with two pair-swaps.
#        Dependent at Level 2: I11 = (1/4) I9.
#
#   I12: [5,9,7,11, 1,10,3,12, 2,6,4,8]   R_{a}^{c}_{b}^{e}R_{c}^{d}_{e}^{f}R_{d}^{a}_{f}^{b}
#        Riem^3 cross pattern.
#        Dependent at Level 2: I12 = (1/4) I9 + I13.
#
#   I13: [5,9,7,11, 1,12,3,10, 2,8,4,6]   R_{a}^{c}_{b}^{e}R_{c}^{f}_{e}^{d}R_{d}^{a}_{f}^{b}
#        Second independent pure-Riemann cubic invariant.
#
# The 8 independent invariants at Level 2 (Bianchi):
#   I1, I2, I3, I5, I6, I7, I9, I13
# i.e., R^3, R*Ric^2, R*K, Ric_{ab}Ric_{cd}R^{acbd}, Ric^3,
# R_{ab}R^{acde}R^b_{cde}, Riem^3 (Goroff-Sagnotti),
# and a second pure-Riemann cubic.
#
# The 5 Bianchi reduction relations:
#   I4  = (1/2) I3          (R * degree-2 Bianchi relation)
#   I8  = (1/2) I7          (Ric * degree-2 Bianchi relation)
#   I10 = (1/2) I9          (Riem^3 Bianchi on any factor)
#   I11 = (1/4) I9          (chained from I10 = (1/2) I9, I11 = (1/2) I10)
#   I12 = (1/4) I9 + I13    (from I11 + I13 - I12 = 0 and I11 = (1/4) I9)
#
# Derivation: Bianchi identity R_{a[bcd]} = 0 applied to one factor of each
# cubic monomial.  The cyclic permutation of the last 3 slots produces
# contraction patterns in the same orbit (with sign) or in other canonical
# orbits, yielding linear relations.
#
# Ground truth: Fulling, King, Wybourne & Cummins (1992), CQG 9:1151, Table 2;
#               Garcia-Parrado & Martin-Garcia (2007), Sec 4, Levels 1-2.
=#

# ---- Level 1: Permutation symmetries ------------------------------------------
# 13 non-vanishing canonical forms, all independent (no relations).

const _DEGREE3_STEP1 = CaseRelations(
    3,        # degree
    "0_0_0",  # case_key (algebraic, no derivatives)
    1,        # step (Level 1: permutation symmetries)
    nothing,  # dim (dimension-independent)
    13,       # n_independent: I1..I13 all independent
    0,        # n_dependent: none at Level 1
    InvarRelation[]  # no relations (all are independent)
)

# ---- Level 2: First Bianchi identity ------------------------------------------
# 8 independent (I1, I2, I3, I5, I6, I7, I9, I13),
# 5 dependent (I4, I8, I10, I11, I12).

const _DEGREE3_STEP2 = CaseRelations(
    3,        # degree
    "0_0_0",  # case_key
    2,        # step (Level 2: cyclic / first Bianchi)
    nothing,  # dim (dimension-independent)
    8,        # n_independent
    5,        # n_dependent
    InvarRelation[
        # I4 = (1/2) * I3   [R * degree-2 Bianchi]
        InvarRelation(
            [3, 4, 1, 2, 9, 11, 10, 12, 5, 7, 6, 8],          # LHS: I4
            [(1//2, [3, 4, 1, 2, 9, 10, 11, 12, 5, 6, 7, 8])]  # RHS: (1/2) * I3
        ),
        # I8 = (1/2) * I7   [Ric * degree-2 Bianchi]
        InvarRelation(
            [3, 5, 1, 9, 2, 11, 10, 12, 4, 7, 6, 8],          # LHS: I8
            [(1//2, [3, 5, 1, 9, 2, 10, 11, 12, 4, 6, 7, 8])]  # RHS: (1/2) * I7
        ),
        # I10 = (1/2) * I9  [Riem^3 Bianchi]
        InvarRelation(
            [5, 6, 9, 11, 1, 2, 10, 12, 3, 7, 4, 8],          # LHS: I10
            [(1//2, [5, 6, 9, 10, 1, 2, 11, 12, 3, 4, 7, 8])]  # RHS: (1/2) * I9
        ),
        # I11 = (1/4) * I9  [chained: I11 = (1/2)*I10 = (1/4)*I9]
        InvarRelation(
            [5, 7, 9, 11, 1, 10, 2, 12, 3, 6, 4, 8],          # LHS: I11
            [(1//4, [5, 6, 9, 10, 1, 2, 11, 12, 3, 4, 7, 8])]  # RHS: (1/4) * I9
        ),
        # I12 = (1/4) * I9 + I13  [from I11 + I13 - I12 = 0 and I11 = (1/4)*I9]
        InvarRelation(
            [5, 9, 7, 11, 1, 10, 3, 12, 2, 6, 4, 8],          # LHS: I12
            [(1//4, [5, 6, 9, 10, 1, 2, 11, 12, 3, 4, 7, 8]),  # (1/4) * I9
             (1//1, [5, 9, 7, 11, 1, 12, 3, 10, 2, 8, 4, 6])]  # + I13
        ),
    ]
)

# ---- Register in database ------------------------------------------------------

_register_invar_case!(_DEGREE3_STEP1)
_register_invar_case!(_DEGREE3_STEP2)

# ---- Named accessors for degree-3 canonical invariants --------------------------

"""
    degree3_canonical_rinvs() -> Vector{RInv}

Return the 13 non-vanishing canonical RInv forms at degree 3 (Level 1).
These are the distinct contraction patterns after applying all Riemann
permutation symmetries.

Ordered by lexicographic contraction vector.  See the header comment
in `src/invariants/db/degree3.jl` for the physical identification of
each form.
"""
function degree3_canonical_rinvs()
    RInv[
        RInv(3, [3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10], true),   # I1: R^3
        RInv(3, [3, 4, 1, 2, 7, 9, 5, 11, 6, 12, 8, 10], true),   # I2: R*Ric^2
        RInv(3, [3, 4, 1, 2, 9, 10, 11, 12, 5, 6, 7, 8], true),   # I3: R*K
        RInv(3, [3, 4, 1, 2, 9, 11, 10, 12, 5, 7, 6, 8], true),   # I4: R*cross
        RInv(3, [3, 5, 1, 7, 2, 9, 4, 11, 6, 12, 8, 10], true),   # I5: Ric*Ric*Riem
        RInv(3, [3, 5, 1, 9, 2, 7, 6, 11, 4, 12, 8, 10], true),   # I6: Ric^3
        RInv(3, [3, 5, 1, 9, 2, 10, 11, 12, 4, 6, 7, 8], true),   # I7: Ric*Riem^2
        RInv(3, [3, 5, 1, 9, 2, 11, 10, 12, 4, 7, 6, 8], true),   # I8: Ric*Riem cross
        RInv(3, [5, 6, 9, 10, 1, 2, 11, 12, 3, 4, 7, 8], true),   # I9: Riem^3 (GS)
        RInv(3, [5, 6, 9, 11, 1, 2, 10, 12, 3, 7, 4, 8], true),   # I10: Riem^3 cross
        RInv(3, [5, 7, 9, 11, 1, 10, 2, 12, 3, 6, 4, 8], true),   # I11: Riem^3 cross 2
        RInv(3, [5, 9, 7, 11, 1, 10, 3, 12, 2, 6, 4, 8], true),   # I12: Riem^3 cross 3
        RInv(3, [5, 9, 7, 11, 1, 12, 3, 10, 2, 8, 4, 6], true),   # I13: Riem^3 indep 2
    ]
end

"""
    degree3_independent_rinvs() -> Vector{RInv}

Return the 8 independent RInv forms at degree 3 after Level 2 (Bianchi).
These span the full space of algebraic cubic curvature invariants
in any dimension.

Returns `[R^3, R*Ric^2, R*K, Ric*Ric*Riem, Ric^3, Ric*Riem^2, Riem^3, Riem^3_alt]`.

Ground truth: Fulling et al. (1992), Table 2, order p=3.
"""
function degree3_independent_rinvs()
    RInv[
        RInv(3, [3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10], true),   # I1: R^3
        RInv(3, [3, 4, 1, 2, 7, 9, 5, 11, 6, 12, 8, 10], true),   # I2: R*Ric^2
        RInv(3, [3, 4, 1, 2, 9, 10, 11, 12, 5, 6, 7, 8], true),   # I3: R*K
        RInv(3, [3, 5, 1, 7, 2, 9, 4, 11, 6, 12, 8, 10], true),   # I5: Ric*Ric*Riem
        RInv(3, [3, 5, 1, 9, 2, 7, 6, 11, 4, 12, 8, 10], true),   # I6: Ric^3
        RInv(3, [3, 5, 1, 9, 2, 10, 11, 12, 4, 6, 7, 8], true),   # I7: Ric*Riem^2
        RInv(3, [5, 6, 9, 10, 1, 2, 11, 12, 3, 4, 7, 8], true),   # I9: Riem^3 (GS)
        RInv(3, [5, 9, 7, 11, 1, 12, 3, 10, 2, 8, 4, 6], true),   # I13: Riem^3 alt
    ]
end
