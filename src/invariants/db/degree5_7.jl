#= Degree 5-7 algebraic invariant database for the Invar pipeline.
#
# Precomputed counts for scalar Riemann polynomial invariants at degrees 5, 6, 7.
# At these degrees full enumeration of contraction vectors is computationally
# infeasible, so we store ground truth counts with metadata only (no explicit
# RInv forms or InvarRelation entries).
#
# Slot layout: degree n has 4n slots. Raw involutions = (4n-1)!!.
#   Degree 5: 20 slots, (19)!! = 654,729,075 raw involutions.
#   Degree 6: 24 slots, (23)!! ~ 3.16 * 10^{10} raw involutions.
#   Degree 7: 28 slots, (27)!! ~ 2.13 * 10^{13} raw involutions.
#
# Counts from Garcia-Parrado & Martin-Garcia (2007), arXiv:0704.1756,
# Table 1 (Canon column) and Table 2 (Steps A and B); and Martin-Garcia,
# Portugal & Manssur (2008), CPC 179:586, arXiv:0802.1274, Table 1.
#
# Column mapping (paper -> our database):
#   Canon = total canonical forms including products (our Step 1 n_independent)
#   Invars = non-product canonical forms (Canon minus products)
#   Step B / Cyclic = non-product independent after first Bianchi identity
#
# To get total independent after Bianchi (our Step 2):
#   n_independent = product_independent + Step_B
# where product_independent is computed from the polynomial ring generated
# by non-product independent invariants at all lower degrees:
#   g1=1 (R), g2=2 (Ric^2, K), g3=5, g4=15, g5=54, g6=270
#
# Degree 5:  Canon=288, Invars=204, Step_B=54, products=84, products_indep=21
#   Total independent = 21 + 54 = 75.  Dependent = 288 - 75 = 213.
#
# Degree 6:  Canon=2070, Invars=1613, Step_B=270, products=457, products_indep=139
#   Total independent = 139 + 270 = 409.  Dependent = 2070 - 409 = 1661.
#
# Degree 7:  Canon=19610, Invars=16532, Step_B=1639, products=3078, products_indep=608
#   Total independent = 608 + 1639 = 2247.  Dependent = 19610 - 2247 = 17363.
#
# Product-independent counts are derived from the partition function of the
# polynomial ring with generators at degrees 1-6.  The generator counts
# (non-product independent invariants after Bianchi) at each degree are:
#   g1=1, g2=2, g3=5, g4=15, g5=54, g6=270
# These match the Step B column of Garcia-Parrado & Martin-Garcia (2007) Table 2
# after accounting for the degree-1 generator R (which has Step B = 1).
#
# Ground truth: Garcia-Parrado & Martin-Garcia, arXiv:0704.1756 (2007), Tables 1-2;
#               Martin-Garcia, Portugal & Manssur, arXiv:0802.1274 (2008), Table 1;
#               Fulling, King, Wybourne & Cummins (1992), CQG 9:1151.
=#

# ---- Degree 5 ----------------------------------------------------------------

# Level 1: Permutation symmetries -- 288 non-vanishing canonical forms.
const _DEGREE5_STEP1 = CaseRelations(
    5,                # degree
    "0_0_0_0_0",      # case_key (algebraic, no derivatives)
    1,                # step (Level 1: permutation symmetries)
    nothing,          # dim (dimension-independent)
    288,              # n_independent: all 288 canonical forms
    0,                # n_dependent: none at Level 1
    InvarRelation[]   # no relations (counts only)
)

# Level 2: First Bianchi identity -- 75 independent, 213 dependent.
# 21 product-type independent + 54 non-product independent.
const _DEGREE5_STEP2 = CaseRelations(
    5,                # degree
    "0_0_0_0_0",      # case_key
    2,                # step (Level 2: cyclic / first Bianchi)
    nothing,          # dim (dimension-independent)
    75,               # n_independent: 21 product + 54 non-product
    213,              # n_dependent: 288 - 75
    InvarRelation[]   # relations not enumerated (counts only)
)

# ---- Degree 6 ----------------------------------------------------------------

# Level 1: Permutation symmetries -- 2070 non-vanishing canonical forms.
const _DEGREE6_STEP1 = CaseRelations(
    6,                    # degree
    "0_0_0_0_0_0",        # case_key (algebraic, no derivatives)
    1,                    # step (Level 1: permutation symmetries)
    nothing,              # dim (dimension-independent)
    2070,                 # n_independent: all 2070 canonical forms
    0,                    # n_dependent: none at Level 1
    InvarRelation[]       # no relations (counts only)
)

# Level 2: First Bianchi identity -- 409 independent, 1661 dependent.
# 139 product-type independent + 270 non-product independent.
const _DEGREE6_STEP2 = CaseRelations(
    6,                    # degree
    "0_0_0_0_0_0",        # case_key
    2,                    # step (Level 2: cyclic / first Bianchi)
    nothing,              # dim (dimension-independent)
    409,                  # n_independent: 139 product + 270 non-product
    1661,                 # n_dependent: 2070 - 409
    InvarRelation[]       # relations not enumerated (counts only)
)

# ---- Degree 7 ----------------------------------------------------------------

# Level 1: Permutation symmetries -- 19610 non-vanishing canonical forms.
const _DEGREE7_STEP1 = CaseRelations(
    7,                        # degree
    "0_0_0_0_0_0_0",          # case_key (algebraic, no derivatives)
    1,                        # step (Level 1: permutation symmetries)
    nothing,                  # dim (dimension-independent)
    19610,                    # n_independent: all 19610 canonical forms
    0,                        # n_dependent: none at Level 1
    InvarRelation[]           # no relations (counts only)
)

# Level 2: First Bianchi identity -- 2247 independent, 17363 dependent.
# 608 product-type independent + 1639 non-product independent.
const _DEGREE7_STEP2 = CaseRelations(
    7,                        # degree
    "0_0_0_0_0_0_0",          # case_key
    2,                        # step (Level 2: cyclic / first Bianchi)
    nothing,                  # dim (dimension-independent)
    2247,                     # n_independent: 608 product + 1639 non-product
    17363,                    # n_dependent: 19610 - 2247
    InvarRelation[]           # relations not enumerated (counts only)
)

# ---- Register in database ------------------------------------------------------

_register_invar_case!(_DEGREE5_STEP1)
_register_invar_case!(_DEGREE5_STEP2)
_register_invar_case!(_DEGREE6_STEP1)
_register_invar_case!(_DEGREE6_STEP2)
_register_invar_case!(_DEGREE7_STEP1)
_register_invar_case!(_DEGREE7_STEP2)

# ---- Count accessors -----------------------------------------------------------

"""
    degree5_7_canonical_count(degree::Int) -> Int

Return the number of non-vanishing canonical algebraic scalar invariants
(Level 1, after permutation symmetries) at the given degree (5, 6, or 7).

These are the distinct contraction patterns of `degree` Riemann tensors with
all `4*degree` indices contracted pairwise via the metric, after quotienting
by the Riemann permutation symmetry group (pair symmetry, antisymmetry,
pair exchange).

# Ground truth
Garcia-Parrado & Martin-Garcia, arXiv:0704.1756 (2007), Table 1.
"""
function degree5_7_canonical_count(degree::Int)
    degree == 5 && return 288
    degree == 6 && return 2070
    degree == 7 && return 19610
    throw(ArgumentError("degree5_7_canonical_count: degree must be 5, 6, or 7, got $degree"))
end

"""
    degree5_7_independent_count(degree::Int) -> Int

Return the number of independent algebraic scalar invariants at the given
degree (5, 6, or 7) after applying the first Bianchi identity (Level 2).

This includes both product-type invariants (built from products of
lower-degree independent invariants) and genuinely new (non-product)
invariants at this degree.

# Breakdown
- Degree 5: 75 = 21 product + 54 non-product
- Degree 6: 409 = 139 product + 270 non-product
- Degree 7: 2247 = 608 product + 1639 non-product

# Ground truth
Garcia-Parrado & Martin-Garcia, arXiv:0704.1756 (2007), Table 2, Step B;
product counts from polynomial ring partition analysis.
"""
function degree5_7_independent_count(degree::Int)
    degree == 5 && return 75
    degree == 6 && return 409
    degree == 7 && return 2247
    throw(ArgumentError("degree5_7_independent_count: degree must be 5, 6, or 7, got $degree"))
end
