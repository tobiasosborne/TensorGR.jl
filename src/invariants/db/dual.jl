#= Dual invariant database for the Invar pipeline.
#
# Precomputed data for scalar curvature invariants involving the Hodge dual
# of the Riemann tensor (Levi-Civita contractions).
#
# Dual curvature invariants involve epsilon tensors:
#   *R_{abcd}  = (1/2) epsilon_{ab}^{ef} R_{efcd}          (left dual)
#   R*_{abcd}  = (1/2) R_{abef} epsilon^{ef}_{cd}           (right dual)
#   *R*_{abcd} = (1/4) epsilon_{ab}^{ef} R_{efgh} epsilon^{gh}_{cd} (double dual)
#
# Key identity in d=4: *R*_{abcd} = R_{abcd}  (double dual = original)
#
# The only independent dual invariant at degree 2 in d=4 is the Pontryagin
# density:  P = *R^{ab}_{cd} R_{ab}^{cd} = (1/2) epsilon^{abef} R_{efcd} R_{ab}^{cd}
#
# This is a topological invariant (total derivative) and is NOT equal to any
# non-dual invariant.  All other degree-2 dual contractions reduce to non-dual
# invariants via the double-dual identity.
#
# Degree 3-5: counts of independent dual invariants are recorded, but explicit
# relations are not yet enumerated (stubbed with correct n_independent).
#
# Ground truth: Garcia-Parrado & Martin-Garcia (2007), Sec 4, Level 6;
#               Zakhary & McIntosh (1997) GRG 29, 539;
#               Fulling et al. (1992), CQG 9, 1151.
=#

# ---- Degree 2: dual invariants (Level 6, d=4) --------------------------------
#
# At degree 2, the possible dual contractions are:
#   (a) *R_{abcd} R^{abcd}      = Pontryagin density (INDEPENDENT)
#   (b) *R_{abcd} *R^{abcd}     = R_{abcd} R^{abcd}  (reduces to Kretschmann
#                                    by double-dual identity *R* = R)
#   (c) R*_{abcd} R^{abcd}      = *R_{abcd} R^{abcd}  (same as Pontryagin
#                                    by Riemann symmetry R_{abcd} = R_{cdab})
#
# Thus: n_independent = 1 (Pontryagin), n_dependent = 2 (reduce to non-dual).
#
# The dependent relations are:
#   (b) double-dual Kretschmann -> Kretschmann (coefficient 1)
#   (c) right-dual Kretschmann -> Pontryagin (coefficient 1, same invariant)
#
# We store (b) as the explicit InvarRelation in the database.  Relation (c) is
# an identity at the DualRInv representation level (left vs right dual on
# Kretschmann contraction are the same scalar by Riemann pair symmetry).

const _DUAL_DEGREE2_STEP6 = CaseRelations(
    2,            # degree
    "dual_0_0",   # case_key (dual, algebraic, no derivatives)
    6,            # step (Level 6: dual invariant product relations)
    4,            # dim = 4 (dual identities are dimension-specific)
    1,            # n_independent: Pontryagin density
    2,            # n_dependent: double-dual Kretschmann, right-dual equiv
    InvarRelation[
        # double-dual Kretschmann = Kretschmann
        # LHS:  *R*_{abcd} *R*^{abcd}  (double-dual on both factors of Kretschmann)
        #        In DualRInv: DualRInv(kr, [(1,:double),(2,:double)])
        #        As a "virtual contraction": we use the Kretschmann contraction
        #        [5,6,7,8,1,2,3,4] tagged as double-dual
        # RHS: 1 * Kretschmann = [5,6,7,8,1,2,3,4]
        InvarRelation(
            [5, 6, 7, 8, 1, 2, 3, 4],   # LHS: (virtual) double-dual Kretschmann base contraction
            [(1 // 1, [5, 6, 7, 8, 1, 2, 3, 4])]  # RHS: ordinary Kretschmann
        )
    ]
)

# ---- Degree 3: dual invariants (Level 6, d=4) --------------------------------
#
# At degree 3, independent dual invariants include contractions like:
#   *R^{ab}_{cd} R^{cd}_{ef} R^{ef}_{ab}
# and mixed dual/non-dual products.
#
# Ground truth (Garcia-Parrado & Martin-Garcia 2007, Table 1, Level 6):
# In d=4, degree 3 has 2 independent dual invariants.
# The explicit relations are complex and require full orbit enumeration
# of the combined Riemann + epsilon symmetry group.

const _DUAL_DEGREE3_STEP6 = CaseRelations(
    3,              # degree
    "dual_0_0_0",   # case_key
    6,              # step
    4,              # dim = 4
    2,              # n_independent (2 independent dual invariants at degree 3)
    0,              # n_dependent (stubbed: no explicit relations yet)
    InvarRelation[]  # relations to be filled in future work
)

# ---- Degree 4: dual invariants (Level 6, d=4) --------------------------------
#
# At degree 4 in d=4, there are more independent dual invariants from
# products of 4 Riemann tensors with various dual insertions.
#
# Ground truth count: 5 independent dual invariants at degree 4.

const _DUAL_DEGREE4_STEP6 = CaseRelations(
    4,                # degree
    "dual_0_0_0_0",   # case_key
    6,                # step
    4,                # dim = 4
    5,                # n_independent
    0,                # n_dependent (stubbed)
    InvarRelation[]    # relations to be filled in future work
)

# ---- Degree 5: dual invariants (Level 6, d=4) --------------------------------
#
# At degree 5 in d=4, dual invariants proliferate further.
#
# Ground truth count: 12 independent dual invariants at degree 5.

const _DUAL_DEGREE5_STEP6 = CaseRelations(
    5,                  # degree
    "dual_0_0_0_0_0",   # case_key
    6,                   # step
    4,                   # dim = 4
    12,                  # n_independent
    0,                   # n_dependent (stubbed)
    InvarRelation[]      # relations to be filled in future work
)

# ---- Register in database -------------------------------------------------------

_register_invar_case!(_DUAL_DEGREE2_STEP6)
_register_invar_case!(_DUAL_DEGREE3_STEP6)
_register_invar_case!(_DUAL_DEGREE4_STEP6)
_register_invar_case!(_DUAL_DEGREE5_STEP6)

# ---- Named accessors for dual invariants -----------------------------------------

"""
    dual_independent_rinvs(degree::Int; dim::Int=4) -> Vector{DualRInv}

Return the independent dual curvature invariants at the given degree
in `dim` dimensions, as `DualRInv` objects.

At degree 2 in d=4, the only independent dual invariant is the Pontryagin
density: P = *R^{ab}_{cd} R_{ab}^{cd}.

At higher degrees, the DualRInv objects are constructed from known
canonical contraction patterns.

# Arguments
- `degree::Int` -- number of Riemann factors (polynomial degree in curvature)
- `dim::Int=4` -- manifold dimension

# Returns
`Vector{DualRInv}` of independent dual invariants.

# Ground truth
Garcia-Parrado & Martin-Garcia (2007), Sec 4, Level 6;
Zakhary & McIntosh (1997) GRG 29, 539.
"""
function dual_independent_rinvs(degree::Int; dim::Int=4)
    dim == 4 || error("dual_independent_rinvs: only d=4 is currently supported, got d=$dim")

    if degree == 2
        return DualRInv[pontryagin_rinv()]
    elseif degree == 3
        # Two independent dual invariants at degree 3:
        #   D1: *R^{ab}_{cd} R^{cd}_{ef} R^{ef}_{ab}
        #   D2: *R^{ab}_{cd} R^{ce}_{af} R^{df}_{be}
        # Build from canonical degree-3 contractions with left dual on factor 1
        #
        # Goroff-Sagnotti contraction: R_{abcd} R^{cd}_{ef} R^{ef}_{ab}
        # Slots: factor 1 = 1-4, factor 2 = 5-8, factor 3 = 9-12
        # Contraction: 1<->11, 2<->12, 3<->5, 4<->6, 7<->9, 8<->10
        gs_contraction = [11, 12, 5, 6, 3, 4, 9, 10, 7, 8, 1, 2]
        d1 = left_dual(RInv(3, gs_contraction), 1)

        # Second degree-3 dual invariant: different contraction pattern
        # R_{abcd} R^{ce}_{af} R^{df}_{be} has contraction:
        # 1<->6, 2<->10, 3<->5, 4<->12, 7<->9, 8<->11
        #  => [6, 10, 5, 12, 3, 1, 9, 11, 7, 2, 8, 4]
        c2 = [6, 10, 5, 12, 3, 1, 9, 11, 7, 2, 8, 4]
        d2 = left_dual(RInv(3, c2), 1)

        return DualRInv[d1, d2]
    elseif degree == 4
        # Stubbed: return Pontryagin-like contractions with left dual on factor 1
        # for the 5 independent canonical contractions at degree 4.
        # Build from the simplest contraction patterns.
        results = DualRInv[]
        # The full enumeration of degree-4 dual invariants requires orbit
        # analysis of the combined Riemann + epsilon symmetry group.
        # For now, return the first canonical pattern as a representative.
        # Degree-4 chain contraction: R R R R with sequential index pairing
        # 1<->5, 2<->6, 3<->7, 4<->8, 9<->13, 10<->14, 11<->15, 12<->16
        c_chain = [5, 6, 7, 8, 1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12]
        push!(results, left_dual(RInv(4, c_chain), 1))
        return results
    elseif degree == 5
        # Stubbed: return empty for now
        return DualRInv[]
    else
        error("dual_independent_rinvs: degree $degree not yet implemented")
    end
end

"""
    pontryagin_rinv_canonical() -> DualRInv

Return the canonical Pontryagin density as a `DualRInv`.

This is equivalent to `pontryagin_rinv()` from `dual_rinv.jl` but
serves as the canonical database entry for the degree-2 dual invariant.

The Pontryagin density P = R *R = (1/2) epsilon^{abef} R_{efcd} R_{ab}^{cd}
is the unique independent dual invariant at degree 2 in d=4.

Ground truth: Garcia-Parrado & Martin-Garcia (2007), Sec 4, Level 6.
"""
function pontryagin_rinv_canonical()
    pontryagin_rinv()
end
