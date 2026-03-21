#= Degree-4 algebraic invariant database for the Invar pipeline.
#
# Precomputed reduction data for scalar quartic Riemann invariants:
# products of 4 Riemann tensors with all 16 indices contracted pairwise.
#
# Slot layout: factor i (1-indexed) occupies slots 4(i-1)+1 .. 4i.
# Factor 1 = slots 1-4, factor 2 = slots 5-8, factor 3 = slots 9-12,
# factor 4 = slots 13-16.  The contraction is a fixed-point-free involution
# on [1..16]: sigma(i) is the slot paired with slot i via the metric.
#
# Raw count: (16-1)!! = 2,027,025 pairings.
# After Level 1 (Riemann permutation symmetries): 57 non-vanishing canonical forms.
# After Level 2 (first Bianchi identity): 26 independent invariants.
#
# Of the 57 canonical forms:
#   - 19 are products of lower-degree invariants (d2*d2 or R*d3)
#   - 38 are genuinely quartic (non-product)
#
# Of the 26 independent invariants after Bianchi:
#   - 11 are product-type (automatically independent from lower-degree bases)
#   - 15 are genuinely quartic (Invar Table 2, step B)
#
# The 57 canonical forms are organized by lexicographic contraction vector.
# Product forms (I1-I16, I35, I36, I46) are identified by their factor-level
# graph being disconnected: some subset of factors contracts only among
# themselves with no cross-contraction to the remaining factors.
#
# Key named invariants:
#   I1:  R^4 = (R_{abcd}g^{ac}g^{bd})^4
#   I2:  R^2 * Ric^2 = (RicScalar)^2 * R_{ab}R^{ab}
#   I3:  R^2 * K = (RicScalar)^2 * R_{abcd}R^{abcd}  (R^2 * Kretschmann)
#   I14: (Ric^2)^2 = (R_{ab}R^{ab})^2
#   I15: Ric^2 * K = (R_{ab}R^{ab})(R_{cdef}R^{cdef})
#   I35: K^2 = (R_{abcd}R^{abcd})^2
#   I39: R^{ab}_{cd}R^{cd}_{ef}R^{ef}_{gh}R^{gh}_{ab}  (Riem^4 4-cycle)
#
# Bianchi relations at Level 2:
#   Product-type relations (inherited from degree-2 and degree-3 Bianchi):
#     I4  = (1/2) I3     [from d2: I4_d2 = (1/2) K, times R^2]
#     I8  = (1/2) I7     [from d3: I8_d3 = (1/2) I7_d3, times R]
#     I10 = (1/2) I9     [from d3: I10_d3 = (1/2) I9_d3, times R]
#     I11 = (1/4) I9     [from d3: I11_d3 = (1/4) I9_d3, times R]
#     I12 = (1/4) I9 + I13  [from d3: I12_d3 = (1/4) I9_d3 + I13_d3, times R]
#     I16 = (1/2) I15    [from d2: I4_d2 = (1/2) K, times Ric^2]
#     I36 = (1/2) I35    [from d2: I4_d2 = (1/2) K, times K]
#     I46 = (1/4) I35    [from d2: I4_d2 = (1/2) K, squared]
#   Non-product Bianchi relations: 23 relations reducing 38 forms to 15
#     (not enumerated; requires full Bianchi orbit analysis on 16-slot
#      contractions, which is computationally expensive)
#
# Ground truth: Martin-Garcia, Portugal & Manssur (2007), CPC 177:640,
#               Table 1 (57 canonical) and Table 2 (step A=38, step B=15);
#               Fulling, King, Wybourne & Cummins (1992), CQG 9:1151.
=#

# ---- Level 1: Permutation symmetries ------------------------------------------
# 57 non-vanishing canonical forms, all independent (no relations at Level 1).

const _DEGREE4_STEP1 = CaseRelations(
    4,            # degree
    "0_0_0_0",    # case_key (algebraic, no derivatives)
    1,            # step (Level 1: permutation symmetries)
    nothing,      # dim (dimension-independent)
    57,           # n_independent: all 57 are independent at Level 1
    0,            # n_dependent: none at Level 1
    InvarRelation[]  # no relations (all are independent)
)

# ---- Level 2: First Bianchi identity ------------------------------------------
# 26 independent, 31 dependent.
# The 8 product-type Bianchi relations are stored explicitly.
# The 23 non-product Bianchi relations are not enumerated (stubbed).

const _DEGREE4_STEP2 = CaseRelations(
    4,            # degree
    "0_0_0_0",    # case_key
    2,            # step (Level 2: cyclic / first Bianchi)
    nothing,      # dim (dimension-independent)
    26,           # n_independent: 11 product + 15 non-product
    31,           # n_dependent: 8 product + 23 non-product
    InvarRelation[
        # --- Product-type Bianchi relations (inherited from lower-degree) ---

        # I4 = (1/2) * I3   [R^2 * (I4_d2 = (1/2)*K)]
        InvarRelation(
            [3, 4, 1, 2, 7, 8, 5, 6, 13, 15, 14, 16, 9, 11, 10, 12],   # I4
            [(1//2, [3, 4, 1, 2, 7, 8, 5, 6, 13, 14, 15, 16, 9, 10, 11, 12])]  # (1/2)*I3
        ),
        # I8 = (1/2) * I7   [R * (I8_d3 = (1/2)*I7_d3)]
        InvarRelation(
            [3, 4, 1, 2, 7, 9, 5, 13, 6, 15, 14, 16, 8, 11, 10, 12],   # I8
            [(1//2, [3, 4, 1, 2, 7, 9, 5, 13, 6, 14, 15, 16, 8, 10, 11, 12])]  # (1/2)*I7
        ),
        # I10 = (1/2) * I9   [R * (I10_d3 = (1/2)*I9_d3)]
        InvarRelation(
            [3, 4, 1, 2, 9, 10, 13, 15, 5, 6, 14, 16, 7, 11, 8, 12],   # I10
            [(1//2, [3, 4, 1, 2, 9, 10, 13, 14, 5, 6, 15, 16, 7, 8, 11, 12])]  # (1/2)*I9
        ),
        # I11 = (1/4) * I9   [R * (I11_d3 = (1/4)*I9_d3)]
        InvarRelation(
            [3, 4, 1, 2, 9, 11, 13, 15, 5, 14, 6, 16, 7, 10, 8, 12],   # I11
            [(1//4, [3, 4, 1, 2, 9, 10, 13, 14, 5, 6, 15, 16, 7, 8, 11, 12])]  # (1/4)*I9
        ),
        # I12 = (1/4) * I9 + I13   [R * (I12_d3 = (1/4)*I9_d3 + I13_d3)]
        InvarRelation(
            [3, 4, 1, 2, 9, 13, 11, 15, 5, 14, 7, 16, 6, 10, 8, 12],   # I12
            [(1//4, [3, 4, 1, 2, 9, 10, 13, 14, 5, 6, 15, 16, 7, 8, 11, 12]),  # (1/4)*I9
             (1//1, [3, 4, 1, 2, 9, 13, 11, 15, 5, 16, 7, 14, 6, 12, 8, 10])]  # + I13
        ),
        # I16 = (1/2) * I15   [Ric^2 * (I4_d2 = (1/2)*K)]
        InvarRelation(
            [3, 5, 1, 7, 2, 8, 4, 6, 13, 15, 14, 16, 9, 11, 10, 12],   # I16
            [(1//2, [3, 5, 1, 7, 2, 8, 4, 6, 13, 14, 15, 16, 9, 10, 11, 12])]  # (1/2)*I15
        ),
        # I36 = (1/2) * I35   [K * (I4_d2 = (1/2)*K)]
        InvarRelation(
            [5, 6, 7, 8, 1, 2, 3, 4, 13, 15, 14, 16, 9, 11, 10, 12],   # I36
            [(1//2, [5, 6, 7, 8, 1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12])]  # (1/2)*I35
        ),
        # I46 = (1/4) * I35   [I4_d2^2 = (1/2 K)^2 = (1/4)*K^2]
        InvarRelation(
            [5, 7, 6, 8, 1, 3, 2, 4, 13, 15, 14, 16, 9, 11, 10, 12],   # I46
            [(1//4, [5, 6, 7, 8, 1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12])]  # (1/4)*I35
        ),
    ]
)

# ---- Register in database ------------------------------------------------------

_register_invar_case!(_DEGREE4_STEP1)
_register_invar_case!(_DEGREE4_STEP2)

# ---- Named accessors for degree-4 canonical invariants --------------------------

"""
    degree4_canonical_rinvs() -> Vector{RInv}

Return the 57 non-vanishing canonical RInv forms at degree 4 (Level 1).
These are the distinct contraction patterns after applying all Riemann
permutation symmetries.

Ordered by lexicographic contraction vector.  The first 16 forms (I1-I16)
and I35, I36, I46 are products of lower-degree invariants; the remaining
38 are genuinely quartic.

Ground truth: Martin-Garcia, Portugal & Manssur (2007), CPC 177:640,
Table 1, degree 4: 57 canonical forms.
"""
function degree4_canonical_rinvs()
    RInv[
        RInv(4, [3,4,1,2, 7,8,5,6, 11,12,9,10, 15,16,13,14], true),   # I1:  R^4
        RInv(4, [3,4,1,2, 7,8,5,6, 11,13,9,15, 10,16,12,14], true),   # I2:  R^2*Ric^2
        RInv(4, [3,4,1,2, 7,8,5,6, 13,14,15,16, 9,10,11,12], true),   # I3:  R^2*K
        RInv(4, [3,4,1,2, 7,8,5,6, 13,15,14,16, 9,11,10,12], true),   # I4:  R^2*I4_d2
        RInv(4, [3,4,1,2, 7,9,5,11, 6,13,8,15, 10,16,12,14], true),   # I5:  R*Ric*Ric*Riem
        RInv(4, [3,4,1,2, 7,9,5,13, 6,11,10,15, 8,16,12,14], true),   # I6:  R*Ric^3
        RInv(4, [3,4,1,2, 7,9,5,13, 6,14,15,16, 8,10,11,12], true),   # I7:  R*Ric*Riem^2
        RInv(4, [3,4,1,2, 7,9,5,13, 6,15,14,16, 8,11,10,12], true),   # I8:  R*Ric*Riem_x
        RInv(4, [3,4,1,2, 9,10,13,14, 5,6,15,16, 7,8,11,12], true),   # I9:  R*Riem^3_GS
        RInv(4, [3,4,1,2, 9,10,13,15, 5,6,14,16, 7,11,8,12], true),   # I10: R*Riem^3_x
        RInv(4, [3,4,1,2, 9,11,13,15, 5,14,6,16, 7,10,8,12], true),   # I11: R*Riem^3_x2
        RInv(4, [3,4,1,2, 9,13,11,15, 5,14,7,16, 6,10,8,12], true),   # I12: R*Riem^3_x3
        RInv(4, [3,4,1,2, 9,13,11,15, 5,16,7,14, 6,12,8,10], true),   # I13: R*Riem^3_alt
        RInv(4, [3,5,1,7, 2,8,4,6, 11,13,9,15, 10,16,12,14], true),   # I14: (Ric^2)^2
        RInv(4, [3,5,1,7, 2,8,4,6, 13,14,15,16, 9,10,11,12], true),   # I15: Ric^2*K
        RInv(4, [3,5,1,7, 2,8,4,6, 13,15,14,16, 9,11,10,12], true),   # I16: Ric^2*I4_d2
        RInv(4, [3,5,1,7, 2,9,4,11, 6,13,8,15, 10,16,12,14], true),   # I17: Ric^2*Riem A
        RInv(4, [3,5,1,7, 2,9,4,13, 6,11,10,15, 8,16,12,14], true),   # I18: Ric^2*Riem B
        RInv(4, [3,5,1,7, 2,9,4,13, 6,14,15,16, 8,10,11,12], true),   # I19: Ric^2*Riem C
        RInv(4, [3,5,1,7, 2,9,4,13, 6,15,14,16, 8,11,10,12], true),   # I20: Ric^2*Riem D
        RInv(4, [3,5,1,9, 2,7,6,13, 4,11,10,15, 8,16,12,14], true),   # I21: Ric*Ric_x A
        RInv(4, [3,5,1,9, 2,7,6,13, 4,14,15,16, 8,10,11,12], true),   # I22: Ric*Ric_x B
        RInv(4, [3,5,1,9, 2,7,6,13, 4,15,14,16, 8,11,10,12], true),   # I23: Ric*Ric_x C
        RInv(4, [3,5,1,9, 2,10,11,13, 4,6,7,15, 8,16,12,14], true),   # I24: Ric*Riem^2 A
        RInv(4, [3,5,1,9, 2,10,13,14, 4,6,15,16, 7,8,11,12], true),   # I25: Ric*Riem^2 B
        RInv(4, [3,5,1,9, 2,10,13,15, 4,6,14,16, 7,11,8,12], true),   # I26: Ric*Riem^2 C
        RInv(4, [3,5,1,9, 2,11,10,13, 4,7,6,15, 8,16,12,14], true),   # I27: Ric*Riem^2 D
        RInv(4, [3,5,1,9, 2,11,12,13, 4,15,6,7, 8,16,10,14], true),   # I28: Ric*Riem^2 E
        RInv(4, [3,5,1,9, 2,11,13,14, 4,15,6,16, 7,8,10,12], true),   # I29: Ric*Riem^2 F
        RInv(4, [3,5,1,9, 2,11,13,15, 4,14,6,16, 7,10,8,12], true),   # I30: Ric*Riem^2 G
        RInv(4, [3,5,1,9, 2,13,11,12, 4,15,7,8, 6,16,10,14], true),   # I31: Ric*Riem^2 H
        RInv(4, [3,5,1,9, 2,13,11,14, 4,15,7,16, 6,8,10,12], true),   # I32: Ric*Riem^2 I
        RInv(4, [3,5,1,9, 2,13,11,15, 4,14,7,16, 6,10,8,12], true),   # I33: Ric*Riem^2 J
        RInv(4, [3,5,1,9, 2,13,11,15, 4,16,7,14, 6,12,8,10], true),   # I34: Ric*Riem^2 K
        RInv(4, [5,6,7,8, 1,2,3,4, 13,14,15,16, 9,10,11,12], true),   # I35: K^2
        RInv(4, [5,6,7,8, 1,2,3,4, 13,15,14,16, 9,11,10,12], true),   # I36: K*I4_d2
        RInv(4, [5,6,7,9, 1,2,3,13, 4,14,15,16, 8,10,11,12], true),   # I37: Riem^3*Ric A
        RInv(4, [5,6,7,9, 1,2,3,13, 4,15,14,16, 8,11,10,12], true),   # I38: Riem^3*Ric B
        RInv(4, [5,6,9,10, 1,2,13,14, 3,4,15,16, 7,8,11,12], true),   # I39: Riem^4 chain
        RInv(4, [5,6,9,10, 1,2,13,15, 3,4,14,16, 7,11,8,12], true),   # I40: Riem^4 A
        RInv(4, [5,6,9,11, 1,2,13,15, 3,14,4,16, 7,10,8,12], true),   # I41: Riem^4 B
        RInv(4, [5,6,9,13, 1,2,10,14, 3,7,15,16, 4,8,11,12], true),   # I42: Riem^4 C
        RInv(4, [5,6,9,13, 1,2,10,15, 3,7,14,16, 4,11,8,12], true),   # I43: Riem^4 D
        RInv(4, [5,6,9,13, 1,2,11,15, 3,14,7,16, 4,10,8,12], true),   # I44: Riem^4 E
        RInv(4, [5,6,9,13, 1,2,11,15, 3,16,7,14, 4,12,8,10], true),   # I45: Riem^4 F
        RInv(4, [5,7,6,8, 1,3,2,4, 13,15,14,16, 9,11,10,12], true),   # I46: I4_d2^2
        RInv(4, [5,7,6,9, 1,3,2,13, 4,15,14,16, 8,11,10,12], true),   # I47: Riem^4 G
        RInv(4, [5,7,9,11, 1,13,2,14, 3,15,4,16, 6,8,10,12], true),   # I48: Riem^4 H
        RInv(4, [5,7,9,11, 1,13,2,15, 3,14,4,16, 6,10,8,12], true),   # I49: Riem^4 I
        RInv(4, [5,7,9,13, 1,10,2,15, 3,6,14,16, 4,11,8,12], true),   # I50: Riem^4 J
        RInv(4, [5,7,9,13, 1,11,2,15, 3,14,6,16, 4,10,8,12], true),   # I51: Riem^4 K
        RInv(4, [5,7,9,13, 1,11,2,15, 3,16,6,14, 4,12,8,10], true),   # I52: Riem^4 L
        RInv(4, [5,9,7,11, 1,13,3,15, 2,14,4,16, 6,10,8,12], true),   # I53: Riem^4 M
        RInv(4, [5,9,7,11, 1,13,3,15, 2,16,4,14, 6,12,8,10], true),   # I54: Riem^4 N
        RInv(4, [5,9,7,13, 1,11,3,15, 2,14,6,16, 4,10,8,12], true),   # I55: Riem^4 O
        RInv(4, [5,9,7,13, 1,11,3,15, 2,16,6,14, 4,12,8,10], true),   # I56: Riem^4 P
        RInv(4, [5,9,7,13, 1,15,3,11, 2,16,8,14, 4,12,6,10], true),   # I57: Riem^4 Q
    ]
end

"""
    degree4_independent_rinvs() -> Vector{RInv}

Return the 26 independent RInv forms at degree 4 after Level 2 (Bianchi).
These span the full space of algebraic quartic curvature invariants
in generic dimension (d >= 8).

The 26 independent forms consist of:
- 11 product-type invariants (from products of degree-2 and degree-3 bases)
- 15 genuinely quartic invariants (non-product, surviving Bianchi reduction)

Only the 11 product-type independent forms are listed explicitly (they are
determined unambiguously from the degree-2 and degree-3 independent bases).
The 15 non-product independent forms are not individually identified here
because the full Bianchi orbit analysis for 16-slot contractions has not
been performed; the count 15 is verified against Martin-Garcia et al. (2007).

Ground truth: Martin-Garcia, Portugal & Manssur (2007), CPC 177:640,
Table 2, degree 4: step B = 15 non-product independent;
Fulling et al. (1992), CQG 9:1151.
"""
function degree4_independent_rinvs()
    # The 11 product-type independent invariants are:
    # I1  = R^4          (d1*d1*d1*d1)
    # I2  = R^2*Ric^2    (d1*d1 * d2)
    # I3  = R^2*K        (d1*d1 * d2)
    # I5  = R*Ric*Ric*Riem  (d1 * d3_I5)
    # I6  = R*Ric^3      (d1 * d3_I6)
    # I7  = R*Ric*Riem^2 (d1 * d3_I7)
    # I9  = R*Riem^3_GS  (d1 * d3_I9)
    # I13 = R*Riem^3_alt (d1 * d3_I13)
    # I14 = (Ric^2)^2    (d2 * d2)
    # I15 = Ric^2*K      (d2 * d2)
    # I35 = K^2           (d2 * d2)
    RInv[
        RInv(4, [3,4,1,2, 7,8,5,6, 11,12,9,10, 15,16,13,14], true),   # I1:  R^4
        RInv(4, [3,4,1,2, 7,8,5,6, 11,13,9,15, 10,16,12,14], true),   # I2:  R^2*Ric^2
        RInv(4, [3,4,1,2, 7,8,5,6, 13,14,15,16, 9,10,11,12], true),   # I3:  R^2*K
        RInv(4, [3,4,1,2, 7,9,5,11, 6,13,8,15, 10,16,12,14], true),   # I5:  R*Ric^2_Riem
        RInv(4, [3,4,1,2, 7,9,5,13, 6,11,10,15, 8,16,12,14], true),   # I6:  R*Ric^3
        RInv(4, [3,4,1,2, 7,9,5,13, 6,14,15,16, 8,10,11,12], true),   # I7:  R*Ric*Riem^2
        RInv(4, [3,4,1,2, 9,10,13,14, 5,6,15,16, 7,8,11,12], true),   # I9:  R*Riem^3_GS
        RInv(4, [3,4,1,2, 9,13,11,15, 5,16,7,14, 6,12,8,10], true),   # I13: R*Riem^3_alt
        RInv(4, [3,5,1,7, 2,8,4,6, 11,13,9,15, 10,16,12,14], true),   # I14: (Ric^2)^2
        RInv(4, [3,5,1,7, 2,8,4,6, 13,14,15,16, 9,10,11,12], true),   # I15: Ric^2*K
        RInv(4, [5,6,7,8, 1,2,3,4, 13,14,15,16, 9,10,11,12], true),   # I35: K^2
    ]
end
