#= Degree-2 algebraic invariant database for the Invar pipeline.
#
# Precomputed reduction relations for scalar quadratic Riemann invariants:
# products of 2 Riemann tensors with all 8 indices contracted pairwise.
#
# Slot layout: factor 1 occupies slots 1-4, factor 2 occupies slots 5-8.
# The contraction is a fixed-point-free involution on [1..8]: sigma(i) is
# the slot paired with slot i via the metric.
#
# Raw count: (8-1)!! = 105 pairings.
# After Level 1 (Riemann permutation symmetries): 4 non-vanishing canonical forms.
# After Level 2 (first Bianchi identity): 3 independent invariants.
#
# The 3 independent invariants (dimension-independent) are:
#   I1: [3,4,1,2,7,8,5,6] = R^2  (Ricci scalar squared)
#       Each factor self-contracts (1<->3, 2<->4) to give R, then R*R.
#   I2: [3,5,1,7,2,8,4,6] = R_{ab}R^{ab}  (Ricci tensor norm)
#       Mixed self/cross contraction yielding Ric*Ric.
#   I3: [5,6,7,8,1,2,3,4] = R_{abcd}R^{abcd}  (Kretschmann scalar)
#       Full cross-contraction between factors.
#
# The 1 dependent invariant at Level 2 (Bianchi reduction):
#   I4: [5,7,6,8,1,3,2,4] = R_{acbd}R^{abcd}  (cross contraction)
#       Reduces via the first Bianchi identity R_{a[bcd]} = 0 to:
#       I4 = (1/2) * I3
#
# Derivation: R_{acbd} = R_{abcd} + R_{adbc} (from Bianchi + antisymmetry).
# Contracting with R^{abcd}: I4 = I3 + R_{adbc}R^{abcd}.
# By orbit sign analysis, R_{adbc}R^{abcd} has sign -1 relative to I4
# in the canonical orbit, so R_{adbc}R^{abcd} = -I4.
# Hence I4 = I3 - I4, giving 2*I4 = I3, i.e., I4 = (1/2)*I3.
#
# Ground truth: Fulling, King, Wybourne & Cummins (1992), CQG 9:1151, Table 1;
#               Garcia-Parrado & Martin-Garcia (2007), Sec 4, Levels 1-2.
=#

# ---- Level 1: Permutation symmetries ------------------------------------------
# 4 non-vanishing canonical forms, all independent (no relations).

const _DEGREE2_STEP1 = CaseRelations(
    2,        # degree
    "0_0",    # case_key (algebraic, no derivatives)
    1,        # step (Level 1: permutation symmetries)
    nothing,  # dim (dimension-independent)
    4,        # n_independent: I1, I2, I3, I4
    0,        # n_dependent: none at Level 1
    InvarRelation[]  # no relations (all are independent)
)

# ---- Level 2: First Bianchi identity ------------------------------------------
# 3 independent (I1, I2, I3), 1 dependent (I4 -> (1/2)*I3).

const _DEGREE2_STEP2 = CaseRelations(
    2,        # degree
    "0_0",    # case_key
    2,        # step (Level 2: cyclic / first Bianchi)
    nothing,  # dim (dimension-independent)
    3,        # n_independent: R^2, Ric^2, Kretschmann
    1,        # n_dependent: I4 reduces to I3
    InvarRelation[
        InvarRelation(
            [5, 7, 6, 8, 1, 3, 2, 4],          # LHS: I4 (dependent)
            [(1//2, [5, 6, 7, 8, 1, 2, 3, 4])]  # RHS: (1/2) * I3 (Kretschmann)
        )
    ]
)

# ---- Register in database ------------------------------------------------------

_register_invar_case!(_DEGREE2_STEP1)
_register_invar_case!(_DEGREE2_STEP2)

# ---- Named accessors for degree-2 canonical invariants --------------------------

"""
    degree2_canonical_rinvs() -> Vector{RInv}

Return the 4 non-vanishing canonical RInv forms at degree 2 (Level 1).
These are the distinct contraction patterns after applying all Riemann
permutation symmetries.

Returns `[R^2, Ric^2, Kretschmann, I4_cross]`.
"""
function degree2_canonical_rinvs()
    RInv[
        RInv(2, [3, 4, 1, 2, 7, 8, 5, 6], true),   # R^2
        RInv(2, [3, 5, 1, 7, 2, 8, 4, 6], true),   # Ric^2
        RInv(2, [5, 6, 7, 8, 1, 2, 3, 4], true),   # Kretschmann
        RInv(2, [5, 7, 6, 8, 1, 3, 2, 4], true),   # cross contraction
    ]
end

"""
    degree2_independent_rinvs() -> Vector{RInv}

Return the 3 independent RInv forms at degree 2 after Level 2 (Bianchi).
These span the full space of algebraic quadratic curvature invariants
in any dimension.

Returns `[R^2, Ric^2, Kretschmann]`.

Ground truth: Fulling et al. (1992), Table 1, order p=2.
"""
function degree2_independent_rinvs()
    RInv[
        RInv(2, [3, 4, 1, 2, 7, 8, 5, 6], true),   # R^2
        RInv(2, [3, 5, 1, 7, 2, 8, 4, 6], true),   # Ric^2
        RInv(2, [5, 6, 7, 8, 1, 2, 3, 4], true),   # Kretschmann
    ]
end
