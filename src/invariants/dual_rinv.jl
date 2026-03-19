#= DualRInv: Curvature invariants involving the Levi-Civita tensor.

Dual curvature invariants involve the Hodge dual of the Riemann tensor:
  *R_{abcd} = (1/2) epsilon_{ab}^{ef} R_{efcd}   (left dual)
  R*_{abcd} = (1/2) R_{abef} epsilon^{ef}_{cd}    (right dual)
  *R*_{abcd} = (1/4) epsilon_{ab}^{ef} R_{efgh} epsilon^{gh}_{cd} (double dual)

The representation extends RInv by treating each epsilon tensor as an additional
"factor" with 4 slots and fully antisymmetric symmetry.  The contraction
permutation covers all slots (Riemann + epsilon), and canonicalization handles
the combined system.

Slot layout:
  - Riemann factors:  slots 1..4k  (4 slots each, k factors)
  - Epsilon factors:  slots 4k+1..4k+4*n_eps  (4 slots each, n_eps epsilons)

Reference: Garcia-Parrado & Martin-Garcia, Comp. Phys. Comm. 176 (2007) 246, Sec 2.3;
           Zakhary & McIntosh (1997) GRG 29, 539.
=#

"""
    DualRInv

Represents a scalar curvature invariant involving Levi-Civita tensors.
Extends RInv with dual insertion points.

# Fields
- `base::RInv` -- underlying Riemann contraction (degree k)
- `dual_positions::Vector{Tuple{Int,Symbol}}` -- (factor_index, :left/:right/:double)
  Each entry specifies that Riemann factor `factor_index` has a Hodge dual applied.
  `:left` dualizes the first index pair (ab), `:right` dualizes the second pair (cd),
  `:double` dualizes both pairs.
"""
struct DualRInv
    base::RInv
    dual_positions::Vector{Tuple{Int,Symbol}}

    function DualRInv(base::RInv, dual_positions::Vector{Tuple{Int,Symbol}})
        for (fi, side) in dual_positions
            (1 <= fi <= base.degree) ||
                error("DualRInv: factor index $fi out of range 1:$(base.degree)")
            side in (:left, :right, :double) ||
                error("DualRInv: invalid dual side :$side; must be :left, :right, or :double")
        end
        # Check no duplicate factor indices (a factor can only appear once)
        factors_seen = Set{Int}()
        for (fi, _) in dual_positions
            fi in factors_seen &&
                error("DualRInv: duplicate dual specification for factor $fi")
            push!(factors_seen, fi)
        end
        new(base, dual_positions)
    end
end

function Base.:(==)(a::DualRInv, b::DualRInv)
    a.base == b.base || return false
    # Normalize dual_positions by sorting for comparison
    sort(a.dual_positions) == sort(b.dual_positions)
end

function Base.hash(d::DualRInv, h::UInt)
    hash(sort(d.dual_positions), hash(d.base, hash(:DualRInv, h)))
end

# ---- Number of epsilon tensors ------------------------------------------------

"""
    _n_epsilons(drinv::DualRInv) -> Int

Count the number of epsilon tensors needed.  Each `:left` or `:right` dual
contributes one epsilon; each `:double` dual contributes two.
"""
function _n_epsilons(drinv::DualRInv)
    n = 0
    for (_, side) in drinv.dual_positions
        n += side == :double ? 2 : 1
    end
    n
end

# ---- Construction helpers ----------------------------------------------------

"""
    left_dual(rinv::RInv, factor::Int) -> DualRInv

Apply left Hodge dual to the specified Riemann factor.
The left dual replaces R_{abcd} with *R_{abcd} = (1/2) epsilon_{ab}^{ef} R_{efcd}.
"""
function left_dual(rinv::RInv, factor::Int)
    DualRInv(rinv, [(factor, :left)])
end

"""
    right_dual(rinv::RInv, factor::Int) -> DualRInv

Apply right Hodge dual to the specified Riemann factor.
The right dual replaces R_{abcd} with R*_{abcd} = (1/2) R_{abef} epsilon^{ef}_{cd}.
"""
function right_dual(rinv::RInv, factor::Int)
    DualRInv(rinv, [(factor, :right)])
end

"""
    double_dual(rinv::RInv, factor::Int) -> DualRInv

Apply double Hodge dual to the specified Riemann factor.
The double dual replaces R_{abcd} with
*R*_{abcd} = (1/4) epsilon_{ab}^{ef} R_{efgh} epsilon^{gh}_{cd}.
"""
function double_dual(rinv::RInv, factor::Int)
    DualRInv(rinv, [(factor, :double)])
end

"""
    left_dual(drinv::DualRInv, factor::Int) -> DualRInv

Apply left Hodge dual to an additional Riemann factor of an existing DualRInv.
"""
function left_dual(drinv::DualRInv, factor::Int)
    # Check if this factor already has a dual specification
    for (fi, side) in drinv.dual_positions
        if fi == factor
            if side == :right
                # right + left = double
                new_pos = [(f, s) for (f, s) in drinv.dual_positions if f != factor]
                push!(new_pos, (factor, :double))
                return DualRInv(drinv.base, new_pos)
            else
                error("left_dual: factor $factor already has :$side dual")
            end
        end
    end
    DualRInv(drinv.base, vcat(drinv.dual_positions, [(factor, :left)]))
end

"""
    right_dual(drinv::DualRInv, factor::Int) -> DualRInv

Apply right Hodge dual to an additional Riemann factor of an existing DualRInv.
"""
function right_dual(drinv::DualRInv, factor::Int)
    for (fi, side) in drinv.dual_positions
        if fi == factor
            if side == :left
                # left + right = double
                new_pos = [(f, s) for (f, s) in drinv.dual_positions if f != factor]
                push!(new_pos, (factor, :double))
                return DualRInv(drinv.base, new_pos)
            else
                error("right_dual: factor $factor already has :$side dual")
            end
        end
    end
    DualRInv(drinv.base, vcat(drinv.dual_positions, [(factor, :right)]))
end

# ---- Pontryagin density as DualRInv -----------------------------------------

"""
    pontryagin_rinv() -> DualRInv

The Pontryagin density R *R as a DualRInv.

This is epsilon^{abef} R_{abcd} R_{ef}^{cd}, represented as a degree-2
RInv with a right dual on the second factor (equivalently, left dual on first).
The contraction pattern has the first factor contracted with the epsilon via
the epsilon's contraction with the second factor.

Contraction: Kretschmann-type (1<->5, 2<->6, 3<->7, 4<->8) with right dual
on factor 2.
"""
function pontryagin_rinv()
    # R_{abcd} *R^{abcd} = R_{abcd} (1/2) epsilon^{cdef} R_{ef}^{ab}
    # Base: degree-2 Kretschmann contraction
    kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
    right_dual(kr, 2)
end

# ---- Conversion to TensorExpr -----------------------------------------------

"""
    to_tensor_expr(drinv::DualRInv;
                    registry=current_registry(), metric=:g) -> TensorExpr

Convert DualRInv to tensor expression with explicit Levi-Civita tensors.

Each dualized pair of Riemann indices gets replaced by a contraction with
an epsilon tensor.  Left dual acts on the first index pair (slots 1,2),
right dual on the second pair (slots 3,4), and double dual on both.

The resulting expression includes the appropriate 1/2 prefactors from the
dual definition.
"""
function to_tensor_expr(drinv::DualRInv;
                         registry::TensorRegistry=current_registry(),
                         metric::Symbol=:g)
    k = drinv.base.degree
    nslots_riem = 4k
    n_eps = _n_epsilons(drinv)

    eps_name = Symbol(:ε, metric)

    used = Set{Symbol}()

    # Allocate index names for Riemann slots
    riem_idx = Vector{Symbol}(undef, nslots_riem)
    for i in 1:nslots_riem
        nm = fresh_index(used)
        push!(used, nm)
        riem_idx[i] = nm
    end

    # Build a map: for each dualized pair, we need fresh "internal" indices
    # that contract between the epsilon and the Riemann.
    # The original Riemann slots get replaced by the epsilon's external indices,
    # and the epsilon's internal indices contract with the Riemann.

    # dual_map[factor] = (side, [(orig_slot, new_riem_idx, eps_ext_idx), ...])
    dual_info = Dict{Int, Vector{Tuple{Int,Symbol,Symbol}}}()
    # For each dual, we create fresh indices for the internal contraction
    for (fi, side) in drinv.dual_positions
        off = 4(fi - 1)
        info = Tuple{Int,Symbol,Symbol}[]
        if side == :left || side == :double
            # Left dual: epsilon on slots off+1, off+2
            for j in 1:2
                slot = off + j
                new_riem = fresh_index(used); push!(used, new_riem)
                push!(info, (slot, new_riem, riem_idx[slot]))
            end
        end
        if side == :right || side == :double
            # Right dual: epsilon on slots off+3, off+4
            for j in 3:4
                slot = off + j
                new_riem = fresh_index(used); push!(used, new_riem)
                push!(info, (slot, new_riem, riem_idx[slot]))
            end
        end
        dual_info[fi] = info
    end

    # Build modified Riemann index arrays: replace dualized slots with
    # internal contraction indices
    modified_riem_idx = copy(riem_idx)
    for (fi, entries) in dual_info
        for (slot, new_riem, _) in entries
            modified_riem_idx[slot] = new_riem
        end
    end

    factors = TensorExpr[]

    # Riemann factors with all-down indices (using modified indices)
    for f in 1:k
        off = 4(f - 1)
        idxs = [down(modified_riem_idx[off+j]) for j in 1:4]
        push!(factors, Tensor(:Riem, idxs))
    end

    # Epsilon factors for each dualization.
    #
    # Left dual:  *R_{abcd} = (1/2) epsilon_{ab}^{ef} R_{efcd}
    #   epsilon has external Down indices (a,b) that sit in the original
    #   contraction slots, and internal Up indices (e,f) that contract
    #   with the Riemann's Down indices.
    #
    # Right dual: R*_{abcd} = (1/2) R_{abef} epsilon^{ef}_{cd}
    #   Same structure: external Down (c,d), internal Up (e,f).
    #
    for (fi, side) in drinv.dual_positions
        entries = dual_info[fi]
        if side == :left
            e1_ext = entries[1][3]  # original index at slot (becomes external)
            e2_ext = entries[2][3]
            e1_int = entries[1][2]  # new index (contracts with Riemann)
            e2_int = entries[2][2]
            # epsilon_{ext1, ext2}^{int1, int2}
            push!(factors, Tensor(eps_name, [down(e1_ext), down(e2_ext),
                                              up(e1_int), up(e2_int)]))
        elseif side == :right
            e1_ext = entries[1][3]
            e2_ext = entries[2][3]
            e1_int = entries[1][2]
            e2_int = entries[2][2]
            # epsilon^{int1, int2}_{ext1, ext2}
            push!(factors, Tensor(eps_name, [up(e1_int), up(e2_int),
                                              down(e1_ext), down(e2_ext)]))
        elseif side == :double
            # Left epsilon (first pair): entries 1,2
            le1_ext = entries[1][3]
            le2_ext = entries[2][3]
            le1_int = entries[1][2]
            le2_int = entries[2][2]
            push!(factors, Tensor(eps_name, [down(le1_ext), down(le2_ext),
                                              up(le1_int), up(le2_int)]))
            # Right epsilon (second pair): entries 3,4
            re1_ext = entries[3][3]
            re2_ext = entries[4][3]
            re1_int = entries[3][2]
            re2_int = entries[4][2]
            push!(factors, Tensor(eps_name, [up(re1_int), up(re2_int),
                                              down(re1_ext), down(re2_ext)]))
        end
    end

    # Metric contractions from the base RInv contraction
    visited = falses(nslots_riem)
    for i in 1:nslots_riem
        visited[i] && continue
        j = drinv.base.contraction[i]
        visited[i] = true
        visited[j] = true
        push!(factors, Tensor(metric, [up(riem_idx[i]), up(riem_idx[j])]))
    end

    # Prefactor: (1/2)^n_eps from the dual definition
    prefactor = 1 // (2^n_eps)

    tproduct(prefactor, factors)
end

# ---- Slot generators for canonicalization ------------------------------------

"""
    _dual_rinv_slot_generators(k, n_eps, dual_positions) -> Vector{Tuple{Vector{Int}, Int}}

Build symmetry generators for the combined Riemann + epsilon slot system.

Slot layout: 4k Riemann slots + 4*n_eps epsilon slots.
"""
function _dual_rinv_slot_generators(k::Int, n_eps::Int,
                                     dual_positions::Vector{Tuple{Int,Symbol}})
    n = 4k + 4n_eps
    gens = Tuple{Vector{Int}, Int}[]

    # Per-factor Riemann symmetries (same as RInv)
    for f in 1:k
        off = 4(f - 1)

        # Anti-symmetry in first pair
        p = collect(1:n)
        p[off+1], p[off+2] = p[off+2], p[off+1]
        push!(gens, (p, -1))

        # Anti-symmetry in second pair
        p = collect(1:n)
        p[off+3], p[off+4] = p[off+4], p[off+3]
        push!(gens, (p, -1))

        # Pair symmetry
        p = collect(1:n)
        p[off+1], p[off+3] = p[off+3], p[off+1]
        p[off+2], p[off+4] = p[off+4], p[off+2]
        push!(gens, (p, +1))
    end

    # Epsilon symmetries: fully antisymmetric in all 4 slots
    eps_base = 4k
    for e in 1:n_eps
        eoff = eps_base + 4(e - 1)
        # Adjacent transpositions with sign -1 (antisymmetry)
        for j in 1:3
            p = collect(1:n)
            p[eoff+j], p[eoff+j+1] = p[eoff+j+1], p[eoff+j]
            push!(gens, (p, -1))
        end
    end

    # Inter-factor transpositions for non-dualized Riemann factors only
    # (identical Riem tensors can be exchanged)
    non_dual = Set(fi for (fi, _) in dual_positions)
    undual_factors = [f for f in 1:k if !(f in non_dual)]
    for i in 1:length(undual_factors)-1
        f1 = undual_factors[i]
        f2 = undual_factors[i+1]
        off1 = 4(f1 - 1)
        off2 = 4(f2 - 1)
        p = collect(1:n)
        for j in 1:4
            p[off1+j], p[off2+j] = p[off2+j], p[off1+j]
        end
        push!(gens, (p, +1))
    end

    gens
end
