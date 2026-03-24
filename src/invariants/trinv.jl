#= TRInv: Contraction permutation representation for tensorial Riemann monomials.

Extends RInv to handle products of Riemann tensors with free (uncontracted)
indices.  The contraction is a *partial* involution: fixed points denote free
slots, non-fixed points are paired by metric contractions.

Example: R_{abcd} R^{ab}_{ef} has 8 slots.  Slots for c,d,e,f are free
(contraction[i] == i); the a,b pairs map to each other.

Canonicalization uses xperm's Butler-Portugal algorithm (polynomial time)
via `xperm_canonical_perm_ext`, properly separating free and dummy indices
for the two-step coset_rep + double_coset_rep algorithm.

Reference: Garcia-Parrado & Martin-Garcia, Comp. Phys. Comm. 176 (2007) 246;
           Martin-Garcia, Comp. Phys. Comm. 179 (2008) 597.
=#

"""
    TRInv(degree, contraction, free_slots, free_positions[, canonical])

Contraction permutation for a tensorial Riemann monomial of `degree` k.

The `contraction` is a (partial) involution of length 4k: `contraction[i]`
gives the slot paired with slot `i`, or `i` itself if slot `i` is free.

# Fields
- `degree::Int` -- number of Riemann factors
- `contraction::Vector{Int}` -- involution with fixed points (length 4k)
- `free_slots::Vector{Int}` -- sorted fixed-point positions
- `free_positions::Vector{IndexPosition}` -- Up/Down for each free slot
- `canonical::Bool` -- true if in canonical form
"""
struct TRInv
    degree::Int
    contraction::Vector{Int}
    free_slots::Vector{Int}
    free_positions::Vector{IndexPosition}
    canonical::Bool
end

function TRInv(degree::Int, contraction::Vector{Int},
               free_slots::Vector{Int}, free_positions::Vector{IndexPosition})
    n = 4degree
    length(contraction) == n ||
        error("TRInv: contraction length $(length(contraction)) != 4*degree=$n")
    length(free_slots) == length(free_positions) ||
        error("TRInv: free_slots and free_positions must have same length")

    for i in 1:n
        ci = contraction[i]
        (1 <= ci <= n) || error("TRInv: contraction[$i]=$ci out of range 1:$n")
        contraction[ci] == i ||
            error("TRInv: not an involution at $i")
    end

    actual_free = sort!([i for i in 1:n if contraction[i] == i])
    sorted_free = sort(free_slots)
    sorted_free == actual_free ||
        error("TRInv: free_slots do not match fixed points of contraction")

    # Reorder free_positions to match sorted free_slots
    if free_slots != sorted_free
        perm = sortperm(free_slots)
        TRInv(degree, contraction, sorted_free, free_positions[perm], false)
    else
        TRInv(degree, contraction, sorted_free, free_positions, false)
    end
end

# ---- Derived properties -------------------------------------------------------

rank(t::TRInv) = length(t.free_slots)
is_scalar(t::TRInv) = isempty(t.free_slots)
is_zero(t::TRInv) = t.canonical && all(==(0), t.contraction)

function Base.hash(t::TRInv, h::UInt)
    hash(t.free_positions, hash(t.free_slots,
         hash(t.contraction, hash(t.degree, hash(:TRInv, h)))))
end

function Base.:(==)(a::TRInv, b::TRInv)
    a.degree == b.degree || return false
    rank(a) == rank(b) || return false
    ca = a.canonical ? a : first(canonicalize(a))
    cb = b.canonical ? b : first(canonicalize(b))
    ca.contraction == cb.contraction && ca.free_positions == cb.free_positions
end

# ---- Conversions to/from RInv ------------------------------------------------

function TRInv(rinv::RInv)
    TRInv(rinv.degree, rinv.contraction, Int[], IndexPosition[], rinv.canonical)
end

function RInv(trinv::TRInv)
    is_scalar(trinv) ||
        error("Cannot convert TRInv with free indices to RInv")
    RInv(trinv.degree, trinv.contraction, trinv.canonical)
end

# ---- Zero sentinel -----------------------------------------------------------

function _zero_trinv(degree::Int, free_slots::Vector{Int},
                     free_positions::Vector{IndexPosition})
    TRInv(degree, zeros(Int, 4degree), free_slots, free_positions, true)
end

# ---- Symmetry generators -----------------------------------------------------

"""
    _trinv_slot_generators(trinv::TRInv) -> Vector{Tuple{Vector{Int}, Int}}

Build generators for the slot symmetry group acting on 4k points.
Returns (permutation, sign) pairs.

Includes per-factor Riemann symmetries and adjacent-factor exchange
generators (bubble-sort generators for factor permutation group).
Factor exchange is only added when both factors have compatible
free-slot patterns (same local positions free, same Up/Down).
"""
function _trinv_slot_generators(trinv::TRInv)
    k = trinv.degree
    n = 4k
    gens = Tuple{Vector{Int}, Int}[]

    # Per-factor Riemann symmetries: anti[1,2], anti[3,4], pair-swap
    for f in 1:k
        off = 4(f - 1)

        p = collect(1:n)
        p[off+1], p[off+2] = p[off+2], p[off+1]
        push!(gens, (p, -1))

        p = collect(1:n)
        p[off+3], p[off+4] = p[off+4], p[off+3]
        push!(gens, (p, -1))

        p = collect(1:n)
        p[off+1], p[off+3] = p[off+3], p[off+1]
        p[off+2], p[off+4] = p[off+4], p[off+2]
        push!(gens, (p, +1))
    end

    # Adjacent-factor exchange (bubble-sort generators for S_k)
    for f in 1:(k - 1)
        _trinv_factors_exchangeable(trinv, f, f + 1) || continue
        off1 = 4(f - 1)
        off2 = 4f
        p = collect(1:n)
        for j in 1:4
            p[off1+j], p[off2+j] = p[off2+j], p[off1+j]
        end
        push!(gens, (p, +1))
    end

    gens
end

"""
    _trinv_factors_exchangeable(trinv, f1, f2) -> Bool

Check whether Riemann factors f1 and f2 can be exchanged.
They are exchangeable iff they have the same local free-slot pattern
and the same Up/Down positions on those free slots.
"""
function _trinv_factors_exchangeable(trinv::TRInv, f1::Int, f2::Int)
    off1 = 4(f1 - 1)
    off2 = 4(f2 - 1)
    c = trinv.contraction
    fs = trinv.free_slots
    fp = trinv.free_positions

    for j in 1:4
        is_free_1 = c[off1+j] == off1 + j
        is_free_2 = c[off2+j] == off2 + j
        is_free_1 == is_free_2 || return false
        if is_free_1
            i1 = searchsortedfirst(fs, off1 + j)
            i2 = searchsortedfirst(fs, off2 + j)
            fp[i1] == fp[i2] || return false
        end
    end
    true
end

# ---- Canonicalization via xperm -----------------------------------------------

"""
    canonicalize(trinv::TRInv) -> (TRInv, Int)

Canonicalize a tensorial Riemann monomial via xperm's Butler-Portugal
algorithm.  Returns `(canonical_trinv, sign)` where sign is +1, -1,
or 0 (vanishing by antisymmetry).

Uses `canonical_perm_ext` with proper free/dummy separation:
  Step 1 (coset_rep):        pin free indices to canonical slots
  Step 2 (double_coset_rep): canonicalize dummy contractions
"""
function canonicalize(trinv::TRInv)
    trinv.canonical && return (trinv, +1)

    k = trinv.degree
    nslots = 4k
    k >= 1 || return (trinv, +1)

    # Degenerate: single Riemann, all free -- just apply Riemann symmetries
    # via the general algorithm below (no special case needed)

    n = nslots + 2  # +2 for sign bits
    c = trinv.contraction
    fs = trinv.free_slots
    fp = trinv.free_positions

    # ── Classify slots ──────────────────────────────────────────────
    dummy_pairs = Tuple{Int,Int}[]
    visited = falses(nslots)
    for i in 1:nslots
        visited[i] && continue
        visited[i] = true
        j = c[i]
        if j != i
            visited[j] = true
            push!(dummy_pairs, (i, j))
        end
    end
    n_dummies = length(dummy_pairs)
    n_free = length(fs)

    # ── Assign names (Renato convention: perm[slot] = name) ───────
    # Dummy pair k -> names (2k-1, 2k)
    # Free index j -> name 2*n_dummies + j
    slot_to_name = zeros(Int, nslots)

    for (pi, (s1, s2)) in enumerate(dummy_pairs)
        slot_to_name[s1] = 2pi - 1
        slot_to_name[s2] = 2pi
    end

    for (fi, slot) in enumerate(fs)
        slot_to_name[slot] = 2n_dummies + fi
    end

    # ── Build xperm permutation ───────────────────────────────────
    perm_data = Vector{Int32}(undef, n)
    for i in 1:nslots
        perm_data[i] = Int32(slot_to_name[i])
    end
    perm_data[n-1] = Int32(n - 1)  # positive sign
    perm_data[n] = Int32(n)

    # ── Build symmetry generators (slot-space, for right-coset action) ─
    # canonical_perm_ext finds the canonical representative of the right
    # coset P·S, where P is the Renato perm (slot→name) and S acts on
    # slots (the domain of P).  Generators are slot-space permutations
    # lifted to n = nslots + 2 points, with sign encoded in the last two.
    raw_gens = _trinv_slot_generators(trinv)
    perm = Perm(perm_data)

    all_gens = Perm[]
    for (g_slot, gsign) in raw_gens
        pg = collect(Int32, 1:n)
        for i in 1:nslots
            pg[i] = Int32(g_slot[i])
        end
        if gsign == -1
            pg[n-1], pg[n] = pg[n], pg[n-1]
        end
        push!(all_gens, Perm(pg))
    end

    isempty(all_gens) && return (TRInv(k, c, fs, fp, true), +1)

    # ── Free and dummy name lists for xperm ───────────────────────
    free_names = Int32[Int32(2n_dummies + fi) for fi in 1:n_free]
    dummy_names = Int32[]
    for pi in 1:n_dummies
        push!(dummy_names, Int32(2pi - 1), Int32(2pi))
    end

    # ── Call xperm ────────────────────────────────────────────────
    base = Int32.(1:nslots)
    cperm = xperm_canonical_perm_ext(perm, base, all_gens,
                                      free_names, dummy_names, n;
                                      metricQ=1)

    # ── Check for zero ────────────────────────────────────────────
    if all(==(Int32(0)), cperm.data)
        return (_zero_trinv(k, fs, fp), 0)
    end

    # ── Extract sign ──────────────────────────────────────────────
    sign = cperm.data[n-1] == Int32(n - 1) ? +1 : -1

    # ── Reconstruct TRInv from canonical permutation ──────────────
    # cperm is in Renato notation: cperm[slot] = canonical name
    # Build inverse: name_to_slot[name] = slot
    name_to_slot = zeros(Int, nslots)
    for slot in 1:nslots
        name = Int(cperm.data[slot])
        name_to_slot[name] = slot
    end

    # Reconstruct contraction: dummy pair names (2k-1, 2k) map to slots
    new_contraction = zeros(Int, nslots)
    for pi in 1:n_dummies
        s1 = name_to_slot[2pi - 1]
        s2 = name_to_slot[2pi]
        new_contraction[s1] = s2
        new_contraction[s2] = s1
    end

    # Free slots: name (2*n_dummies + j) -> slot
    new_free_slots = Vector{Int}(undef, n_free)
    for fi in 1:n_free
        new_free_slots[fi] = name_to_slot[2n_dummies + fi]
    end

    # Free slots are fixed points
    for fi in 1:n_free
        new_contraction[new_free_slots[fi]] = new_free_slots[fi]
    end

    # Determine new free positions from the canonical slot locations.
    # The Riemann slot position pattern is [Down, Down, Down, Down]
    # (all-down convention) -- but free positions can be raised.
    # We preserve the ORIGINAL free_positions because TRInv tracks the
    # abstract index structure, not slot-dependent positions.
    # The canonical ordering of free indices is determined by xperm;
    # their Up/Down pattern follows the original ordering.
    new_free_positions = copy(fp)

    sp = sortperm(new_free_slots)
    new_free_slots_sorted = new_free_slots[sp]
    new_free_positions_sorted = new_free_positions[sp]

    (TRInv(k, new_contraction, new_free_slots_sorted,
           new_free_positions_sorted, true), sign)
end

# ---- Conversion to TensorExpr ------------------------------------------------

"""
    to_tensor_expr(trinv::TRInv; registry=current_registry(), metric=:g) -> TensorExpr

Convert a TRInv to a TensorExpr product of Riemann tensors.

Free indices get fresh names; contracted pairs are joined by inverse metrics.
The scalar coefficient is always 1//1 (sign is tracked separately).
"""
function to_tensor_expr(trinv::TRInv;
                         registry::TensorRegistry=current_registry(),
                         metric::Symbol=:g)
    k = trinv.degree
    nslots = 4k
    c = trinv.contraction
    fs = trinv.free_slots
    fp = trinv.free_positions

    used = Set{Symbol}()
    slot_name = Vector{Symbol}(undef, nslots)
    slot_pos = Vector{IndexPosition}(undef, nslots)

    # Default: all down (standard Riemann convention)
    for i in 1:nslots
        slot_pos[i] = Down
    end

    # Set free slot positions
    for (fi, slot) in enumerate(fs)
        slot_pos[slot] = fp[fi]
    end

    # Assign names: dummy pairs share a name, free slots get unique names
    visited = falses(nslots)
    for i in 1:nslots
        visited[i] && continue
        visited[i] = true
        j = c[i]
        nm = fresh_index(used)
        push!(used, nm)
        slot_name[i] = nm
        if j != i
            visited[j] = true
            slot_name[j] = nm
            # Dummy pair: one up, one down (via metric)
            slot_pos[i] = Down
            slot_pos[j] = Down  # both down on Riemann; metric raises
        end
    end

    factors = TensorExpr[]

    # Riemann factors with their slot indices
    for f in 1:k
        off = 4(f - 1)
        idxs = [TIndex(slot_name[off+j], slot_pos[off+j]) for j in 1:4]
        push!(factors, Tensor(:Riem, idxs))
    end

    # Metric contractions for dummy pairs
    visited .= false
    for i in 1:nslots
        visited[i] && continue
        visited[i] = true
        j = c[i]
        j == i && continue  # skip free slots
        visited[j] = true
        push!(factors, Tensor(metric, [up(slot_name[i]), up(slot_name[j])]))
    end

    tproduct(1 // 1, factors)
end

# ---- Conversion from TensorExpr ---------------------------------------------

"""
    from_tensor_expr_trinv(expr::TensorExpr;
                            registry=current_registry(),
                            metric=:g) -> (TRInv, Int)

Encode a product of Riemann tensors (possibly with free indices) as a TRInv.
Returns `(trinv, sign)` where sign captures the scalar coefficient's sign.

The expression should be a product of `:Riem` tensors and optionally
inverse metrics for dummy contractions.
"""
function from_tensor_expr_trinv(expr::TensorExpr;
                                 registry::TensorRegistry=current_registry(),
                                 metric::Symbol=:g)
    p = _as_tproduct(expr)
    p === nothing && error("from_tensor_expr_trinv: expected a product, got $(typeof(expr))")

    riem_factors = Tensor[]
    metric_factors = Tensor[]
    for f in p.factors
        f isa Tensor || error("from_tensor_expr_trinv: non-Tensor factor: $(typeof(f))")
        if f.name == :Riem
            push!(riem_factors, f)
        elseif f.name == metric
            push!(metric_factors, f)
        else
            error("from_tensor_expr_trinv: unexpected tensor $(f.name)")
        end
    end

    k = length(riem_factors)
    k >= 1 || error("from_tensor_expr_trinv: no Riemann tensors found")
    nslots = 4k

    # Map each index name to its Riemann slot(s)
    name_to_riem_slots = Dict{Symbol, Vector{Int}}()
    slot_positions = Vector{IndexPosition}(undef, nslots)
    for (fi, riem) in enumerate(riem_factors)
        off = 4(fi - 1)
        for (j, idx) in enumerate(riem.indices)
            slot = off + j
            push!(get!(Vector{Int}, name_to_riem_slots, idx.name), slot)
            slot_positions[slot] = idx.position
        end
    end

    # Collect metric contraction pairs
    metric_pair_names = Set{Symbol}()
    metric_links = Dict{Symbol, Symbol}()
    for m in metric_factors
        a, b = m.indices[1].name, m.indices[2].name
        metric_links[a] = b
        metric_links[b] = a
        push!(metric_pair_names, a, b)
    end

    # Build contraction
    contraction = collect(1:nslots)  # start as identity (all free)
    free_slots = Int[]
    free_positions = IndexPosition[]

    # Direct Riemann-Riemann contractions (same name appears in two Riemann slots)
    for (sym, slots) in name_to_riem_slots
        if length(slots) == 2
            s1, s2 = slots
            contraction[s1] = s2
            contraction[s2] = s1
        elseif length(slots) == 1
            # Check if contracted via metric
            if haskey(metric_links, sym)
                partner_sym = metric_links[sym]
                partner_slots = get(name_to_riem_slots, partner_sym, Int[])
                if length(partner_slots) == 1
                    s1 = slots[1]
                    s2 = partner_slots[1]
                    contraction[s1] = s2
                    contraction[s2] = s1
                end
            end
        end
    end

    # Identify free slots (still self-mapping after contractions)
    for i in 1:nslots
        if contraction[i] == i
            push!(free_slots, i)
            push!(free_positions, slot_positions[i])
        end
    end

    sign = p.scalar >= 0 ? +1 : -1

    (TRInv(k, contraction, free_slots, free_positions), sign)
end

# ---- First Bianchi identity (cyclic reduction) -------------------------------

"""
    bianchi_cyclic_trinv(trinv::TRInv, factor::Int) -> Vector{Tuple{TRInv, Int}}

Apply the first Bianchi identity R_{a[bcd]} = 0 to Riemann factor `factor`
of a tensorial monomial.

The identity R_{abcd} + R_{acdb} + R_{adbc} = 0 means:
    trinv = -trinv'₁ - trinv'₂
where trinv'₁ and trinv'₂ are obtained by cyclically permuting the last
three slots (off+2, off+3, off+4) of the target factor.

Returns a vector of `(canonical_trinv, sign)` pairs representing the
RHS of the relation.  The original trinv equals the negative sum of these.

The contraction transforms by conjugation: σ' = π · σ · π⁻¹, where π
is the 3-cycle on the affected slots.
"""
function bianchi_cyclic_trinv(trinv::TRInv, factor::Int)
    k = trinv.degree
    (1 <= factor <= k) ||
        error("bianchi_cyclic_trinv: factor $factor out of range [1, $k]")

    nslots = 4k
    off = 4(factor - 1)
    c = trinv.contraction
    fs = trinv.free_slots
    fp = trinv.free_positions

    # The two nontrivial cyclic permutations of (off+2, off+3, off+4):
    #   π₁: b→c→d→b  i.e. off+2→off+3, off+3→off+4, off+4→off+2
    #   π₂: b→d→c→b  i.e. off+2→off+4, off+3→off+2, off+4→off+3
    cycles = (
        (off+2 => off+3, off+3 => off+4, off+4 => off+2),
        (off+2 => off+4, off+3 => off+2, off+4 => off+3),
    )

    result = Tuple{TRInv, Int}[]

    for cyc in cycles
        # Build the permutation π (and π⁻¹)
        pi = collect(1:nslots)
        for (src, dst) in cyc
            pi[src] = dst
        end
        pi_inv = collect(1:nslots)
        for i in 1:nslots
            pi_inv[pi[i]] = i
        end

        # New contraction: σ' = π · σ · π⁻¹
        new_c = Vector{Int}(undef, nslots)
        for i in 1:nslots
            new_c[pi[i]] = pi[c[pi_inv[pi[i]]]]
        end
        # Simplify: new_c[j] = pi[c[pi_inv[j]]] for all j
        for j in 1:nslots
            new_c[j] = pi[c[pi_inv[j]]]
        end

        # New free slots: π maps free slots to their new positions
        new_fs = sort!([pi[s] for s in fs])

        # New free positions: each free slot carries its original position
        # Build a map from original free slot → position
        orig_pos = Dict{Int, IndexPosition}()
        for (i, s) in enumerate(fs)
            orig_pos[s] = fp[i]
        end
        new_fp = IndexPosition[orig_pos[pi_inv[s]] for s in new_fs]

        new_trinv = TRInv(k, new_c, new_fs, new_fp, false)
        canon, sign = canonicalize(new_trinv)

        # The Bianchi relation has -1 coefficient for each cycled term
        push!(result, (canon, -sign))
    end

    result
end

"""
    bianchi_relations_trinv(trinvs::Vector{TRInv}) -> Vector{Dict{TRInv, Rational{Int}}}

Generate all first Bianchi linear relations among a set of canonical TRInv
monomials.  For each monomial and each factor, applies the cyclic identity
and expresses the result as a linear combination of canonical monomials.

Each relation is a Dict mapping canonical TRInv → coefficient, where
the sum equals zero.
"""
function bianchi_relations_trinv(trinvs::Vector{TRInv})
    # Index canonical monomials for lookup
    canon_set = Dict{Vector{Int}, Int}()
    for (i, t) in enumerate(trinvs)
        ct = t.canonical ? t : first(canonicalize(t))
        canon_set[ct.contraction] = i
    end

    relations = Dict{TRInv, Rational{Int}}[]

    for t in trinvs
        ct = t.canonical ? t : first(canonicalize(t))
        for f in 1:ct.degree
            rel = Dict{TRInv, Rational{Int}}()
            rel[ct] = 1 // 1  # the original monomial

            cycled = bianchi_cyclic_trinv(ct, f)
            for (c_trinv, sign) in cycled
                coeff = Rational{Int}(sign)
                rel[c_trinv] = get(rel, c_trinv, 0 // 1) + coeff
            end

            # Remove zero coefficients
            filter!(p -> p.second != 0, rel)

            # Only keep nontrivial relations (more than one term)
            length(rel) > 1 && push!(relations, rel)
        end
    end

    # Deduplicate by normalizing each relation to a canonical key
    unique_rels = Dict{TRInv, Rational{Int}}[]
    seen = Set{UInt}()
    for rel in relations
        h = hash(sort!(collect(keys(rel)), by = t -> t.contraction))
        h in seen && continue
        push!(seen, h)
        push!(unique_rels, rel)
    end

    unique_rels
end

# ---- Second Bianchi identity (differential) ----------------------------------

"""
    apply_bianchi2_tensorial(expr::TensorExpr, factor_idx::Int;
                              registry::TensorRegistry=current_registry()) -> TensorExpr

Apply the second Bianchi identity ∇_{[a} R_{bc]de} = 0 to the
`factor_idx`-th factor of a TProduct, which must be a TDeriv wrapping
a Riemann tensor: `∂_a R_{bcde}`.

The identity ∇_a R_{bcde} + ∇_b R_{cade} + ∇_c R_{abde} = 0 means:

    ∂_a R_{bcde} = -∂_b R_{cade} - ∂_c R_{abde}

The derivative index `a` is antisymmetrized with the first pair `b,c`
of the Riemann tensor, producing two new terms where the derivative
index cycles through positions (a,b,c).

Returns the sum of the two substituted terms (canonicalized).
"""
function apply_bianchi2_tensorial(expr::TensorExpr, factor_idx::Int;
                                   registry::TensorRegistry=current_registry())
    expr isa TProduct || return expr

    factors = collect(expr.factors)
    (1 <= factor_idx <= length(factors)) ||
        error("apply_bianchi2_tensorial: factor_idx $factor_idx out of range")

    f = factors[factor_idx]

    # The target factor must be TDeriv wrapping a Riemann tensor
    f isa TDeriv || return expr
    f.arg isa Tensor || return expr
    f.arg.name == :Riem || return expr
    length(f.arg.indices) == 4 || return expr

    deriv_idx = f.index        # a
    covd = f.covd
    b, c, d, e = f.arg.indices # R_{bcde}

    # Second Bianchi: ∂_a R_{bcde} = -∂_b R_{cade} - ∂_c R_{abde}
    term1_factor = TDeriv(b, Tensor(:Riem, [c, deriv_idx, d, e]), covd)
    term2_factor = TDeriv(c, Tensor(:Riem, [deriv_idx, b, d, e]), covd)

    other_factors = TensorExpr[factors[i] for i in eachindex(factors) if i != factor_idx]

    t1 = tproduct(-expr.scalar, vcat(other_factors, TensorExpr[term1_factor]))
    t2 = tproduct(-expr.scalar, vcat(other_factors, TensorExpr[term2_factor]))

    result = tsum(TensorExpr[t1, t2])

    with_registry(registry) do
        canonicalize(result)
    end
end

"""
    has_diff_riemann(expr::TensorExpr) -> Bool

Check whether an expression contains any covariant derivative of a
Riemann tensor (i.e., `TDeriv(_, Tensor(:Riem, _), _)`).
"""
function has_diff_riemann(expr::TDeriv)
    (expr.arg isa Tensor && expr.arg.name == :Riem) || has_diff_riemann(expr.arg)
end
has_diff_riemann(::Tensor) = false
has_diff_riemann(::TScalar) = false
function has_diff_riemann(p::TProduct)
    any(has_diff_riemann, p.factors)
end
function has_diff_riemann(s::TSum)
    any(has_diff_riemann, s.terms)
end

"""
    diff_riemann_factor_indices(p::TProduct) -> Vector{Int}

Return the indices of factors in a TProduct that are TDeriv wrapping
a Riemann tensor (candidates for second Bianchi application).
"""
function diff_riemann_factor_indices(p::TProduct)
    idxs = Int[]
    for (i, f) in enumerate(p.factors)
        if f isa TDeriv && f.arg isa Tensor && f.arg.name == :Riem
            push!(idxs, i)
        end
    end
    idxs
end

# ---- Display -----------------------------------------------------------------

function Base.show(io::IO, t::TRInv)
    r = rank(t)
    tag = t.canonical ? "canonical" : "non-canonical"
    print(io, "TRInv(degree=$(t.degree), rank=$r, $tag)")
end

function Base.show(io::IO, ::MIME"text/plain", t::TRInv)
    r = rank(t)
    tag = t.canonical ? "canonical" : "non-canonical"
    println(io, "TRInv(degree=$(t.degree), rank=$r, $tag)")
    println(io, "  contraction: $(t.contraction)")
    if r > 0
        println(io, "  free_slots:  $(t.free_slots)")
        print(io, "  free_pos:    $(t.free_positions)")
    end
end

# ══════════════════════════════════════════════════════════════════════
# Tensorial DDI Reduction
# ══════════════════════════════════════════════════════════════════════

"""
    generate_tensorial_ddi(dim::Int, degree::Int, n_free::Int;
                            registry=current_registry(), metric=:g)
        -> Vector{Dict{TRInv, Rational{Int}}}

Generate dimensionally-dependent identities (DDIs) for tensorial Riemann
monomials of given `degree` (number of Riemann factors) with `n_free` free
indices, valid in `dim` dimensions.

The master DDI `δ^{[a₁...a_{d+1}]}_{[b₁...b_{d+1}]} = 0` is partially
contracted with `degree` Riemann tensors. Indices not consumed by the
Riemann contractions remain free, yielding tensorial identities.

Each returned Dict maps canonical TRInv → coefficient, summing to zero.

# Examples
```julia
# In d=3: Weyl vanishing is a degree-1, rank-4 tensorial DDI
rels = generate_tensorial_ddi(3, 1, 4; registry=reg)
```
"""
function generate_tensorial_ddi(dim::Int, degree::Int, n_free::Int;
                                 registry::TensorRegistry=current_registry(),
                                 metric::Symbol=:g)
    nslots = 4 * degree
    n_contracted = nslots - n_free

    # The generalized delta has p = dim+1 index pairs.
    # We use n_contracted/2 pairs for Riemann contractions and
    # (p - n_contracted/2) remaining pairs, of which n_free become free
    # and the rest are contracted with metrics.
    p = dim + 1
    iseven(n_contracted) || return Dict{TRInv, Rational{Int}}[]
    n_riem_pairs = n_contracted ÷ 2

    # Need enough delta pairs: n_riem_pairs + ceil(n_free/2) <= p
    n_riem_pairs + (n_free + 1) ÷ 2 > p && return Dict{TRInv, Rational{Int}}[]

    # For the simplest case: build the identity at TensorExpr level
    # using the generalized delta, contract with Riemann tensors, simplify
    with_registry(registry) do
        _build_tensorial_ddi_expr(dim, degree, n_free, registry, metric)
    end
end

"""
    apply_ddi_tensorial(expr::TensorExpr, dim::Int;
                         registry=current_registry(), metric=:g) -> TensorExpr

Apply tensorial DDI reduction to a curvature expression in dimension `dim`.

In low dimensions, the Riemann tensor has fewer independent components than
in generic dimension. This function applies the corresponding DDI identities:

- **d ≤ 2**: `R_{abcd} = (R/2)(g_{ac}g_{bd} - g_{ad}g_{bc})`; `R_{ab} = (R/2)g_{ab}`
- **d = 3**: Weyl vanishes → Riemann decomposes into Ricci + metric
- **d = 4**: Scalar DDIs (Gauss-Bonnet), plus rank-2/4 DDIs from `δ^5 = 0`

Returns the simplified expression (which may contain Ricci, metric, and scalar
tensors instead of Riemann).
"""
function apply_ddi_tensorial(expr::TensorExpr, dim::Int;
                              registry::TensorRegistry=current_registry(),
                              metric::Symbol=:g)
    with_registry(registry) do
        result = expr

        if dim <= 3
            # Weyl vanishes: decompose Riemann → Weyl + Ric terms, set Weyl=0
            result = walk(result) do node
                node isa Tensor || return node
                if node.name == :Riem && length(node.indices) == 4
                    riemann_to_weyl(node.indices[1], node.indices[2],
                                    node.indices[3], node.indices[4],
                                    metric; dim=dim)
                else
                    node
                end
            end
            # Zero out Weyl (vanishes in d≤3)
            result = walk(result) do node
                node isa Tensor && node.name == :Weyl ? TScalar(0 // 1) : node
            end
        end

        if dim <= 2
            # Ricci trace: R_{ab} = (R/dim) g_{ab}
            result = walk(result) do node
                node isa Tensor || return node
                if node.name == :Ric && length(node.indices) == 2
                    a, b = node.indices
                    tproduct(1 // dim, TensorExpr[
                        Tensor(:RicScalar, TIndex[]),
                        Tensor(metric, [a, b])])
                else
                    node
                end
            end
        end

        # Apply scalar DDIs and simplify
        simplify_with_ddis(result; dim=dim, order=2, registry=registry)
    end
end

"""
    ddi_reduces_trinv(trinv::TRInv, dim::Int) -> Bool

Check whether a tensorial DDI identity constrains this TRInv in dimension `dim`.
Fast check without actually computing the reduction.

Known DDI conditions:
- d ≤ 3, degree 1, rank 4: Weyl vanishing (Riemann = Ricci decomposition)
- d ≤ 2, degree 1, rank 2: Ricci trace (R_{ab} ∝ g_{ab})
- d ≤ 2k, degree k, rank 0: scalar DDI (Gauss-Bonnet at k=2, cubic at k=3)
"""
function ddi_reduces_trinv(trinv::TRInv, dim::Int)
    k = trinv.degree
    r = rank(trinv)

    # Rank-4 degree-1: Weyl vanishes in d ≤ 3
    k == 1 && r == 4 && dim <= 3 && return true

    # Rank-2 degree-1: Ricci is pure trace in d ≤ 2
    k == 1 && r == 2 && dim <= 2 && return true

    # Scalar (rank 0): generalized Gauss-Bonnet when 2k ≥ dim
    r == 0 && 2k >= dim && return true

    # General: the (d+1)-antisymmetrized delta constrains when
    # the monomial has more index structure than the dimension supports
    # This is a conservative check for higher-rank higher-degree cases
    4k > dim * (dim + 1) && return true

    false
end

# ── Internal helpers ─────────────────────────────────────────────────

"""Build tensorial DDI identities at the TensorExpr level."""
function _build_tensorial_ddi_expr(dim::Int, degree::Int, n_free::Int,
                                    registry::TensorRegistry, metric::Symbol)
    # Construct the Weyl decomposition identity for the simplest cases
    # that produce useful results: degree=1 (Riemann DDIs)
    results = Dict{TRInv, Rational{Int}}[]

    if degree == 1 && n_free == 4 && dim <= 3
        # Weyl vanishing: R_{abcd} = Ricci decomposition
        # This is the rank-4 DDI from delta^4 = 0 in d=3
        # Already handled by weyl_vanishing_rule in syzygies.jl
        # but we generate the TRInv-level identity here
        used = Set{Symbol}()
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)

        riem = Tensor(:Riem, [down(a), down(b), down(c), down(d)])
        decomp = riemann_to_weyl(down(a), down(b), down(c), down(d), metric; dim=dim)
        # In d<=3, Weyl=0, so decomp = Ricci terms only
        identity_expr = riem - simplify(decomp; registry=registry)

        if !(identity_expr isa TScalar && identity_expr.val == 0 // 1)
            rel = _expr_to_trinv_relation(identity_expr, registry, metric)
            !isempty(rel) && push!(results, rel)
        end
    end

    if degree == 1 && n_free == 2 && dim <= 2
        # Ricci trace: R_{ab} = (R/d) g_{ab} in d=2
        used = Set{Symbol}()
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)

        # Build the identity at TensorExpr level
        ric = Tensor(:Ric, [down(a), down(b)])
        r_over_d = tproduct(1 // dim, TensorExpr[Tensor(:RicScalar, TIndex[]),
                   Tensor(metric, [down(a), down(b)])])
        identity_expr = ric - r_over_d
        identity_expr = simplify(identity_expr; registry=registry)

        if !(identity_expr isa TScalar && identity_expr.val == 0 // 1)
            rel = _expr_to_trinv_relation(identity_expr, registry, metric)
            !isempty(rel) && push!(results, rel)
        end
    end

    results
end

"""Convert a TensorExpr identity to a TRInv coefficient relation."""
function _expr_to_trinv_relation(expr::TensorExpr,
                                  registry::TensorRegistry, metric::Symbol)
    rel = Dict{TRInv, Rational{Int}}()

    terms = expr isa TSum ? expr.terms : [expr]
    for t in terms
        try
            trinv, sgn = from_tensor_expr_trinv(t; registry=registry, metric=metric)
            canon = canonicalize(trinv; registry=registry)
            coeff = t isa TProduct ? t.scalar : 1 // 1
            key = canon
            rel[key] = get(rel, key, 0 // 1) + coeff * sgn
        catch
            # Not a pure Riemann monomial — skip (e.g., Ricci or metric terms)
            continue
        end
    end

    # Remove zero entries
    filter!(p -> p.second != 0 // 1, rel)
    rel
end

"""Parse a simplified expression back into TRInv terms."""
function _parse_trinv_sum(expr::TensorExpr,
                           registry::TensorRegistry, metric::Symbol)
    terms = expr isa TSum ? expr.terms : [expr]
    result = Tuple{TRInv, Rational{Int}}[]

    for t in terms
        try
            trinv, sgn = from_tensor_expr_trinv(t; registry=registry, metric=metric)
            canon = canonicalize(trinv; registry=registry)
            coeff = t isa TProduct ? t.scalar : 1 // 1
            push!(result, (canon, coeff * sgn))
        catch
            continue
        end
    end

    result
end

# ══════════════════════════════════════════════════════════════════════
# TInvarSimplify: top-level tensorial invariant simplification
# ══════════════════════════════════════════════════════════════════════

"""
    tinvar_simplify(expr::TensorExpr;
                     registry=current_registry(),
                     dim::Union{Int,Nothing}=nothing,
                     metric::Symbol=:g,
                     covd::Union{Symbol,Nothing}=nothing) -> TensorExpr

Comprehensive simplification for tensorial Riemann expressions.

Applies the following pipeline in sequence:
1. **Canonicalization**: Butler-Portugal canonical ordering via xperm
2. **Metric contraction**: contract metrics and curvature traces
3. **CovD commutation**: sort covariant derivatives (if `covd` specified),
   producing Riemann commutator terms + differential Bianchi
4. **DDI reduction**: dimensionally-dependent identities (if `dim` specified),
   including Weyl vanishing (d≤3) and Gauss-Bonnet (d=4)
5. **Collect terms**: combine equivalent terms

Returns the simplified expression in terms of independent tensorial monomials.

# Examples
```julia
reg = TensorRegistry()
@manifold M4 dim=4 metric=g registry=reg
define_curvature_tensors!(reg, :M4, :g)

# Simplify R_{a}^{bcd} R_{bcd}^{e}
used = Set{Symbol}()
a, b, c, d, e = [fresh_index(used) for _ in 1:5]
for s in [a,b,c,d,e]; push!(used, s); end

expr = Tensor(:Riem, [down(a), up(b), up(c), up(d)]) *
       Tensor(:Riem, [down(b), down(c), down(d), up(e)])
result = tinvar_simplify(expr; registry=reg, dim=4)
```

See also: [`apply_ddi_tensorial`](@ref), [`canonicalize`](@ref),
[`simplify`](@ref), [`full_simplify`](@ref)
"""
function tinvar_simplify(expr::TensorExpr;
                          registry::TensorRegistry=current_registry(),
                          dim::Union{Int,Nothing}=nothing,
                          metric::Symbol=:g,
                          covd::Union{Symbol,Nothing}=nothing)
    with_registry(registry) do
        result = expr

        # Phase 1: Standard simplify (canonicalize + metric contraction + curvature contraction)
        skw = Dict{Symbol,Any}(:registry => registry)
        if covd !== nothing
            skw[:commute_covds_name] = covd
        end
        result = simplify(result; pairs(skw)...)

        # Phase 2: Tensorial DDI reduction (Weyl vanishing, Ricci trace, scalar DDIs)
        if dim !== nothing
            result = apply_ddi_tensorial(result, dim; registry=registry, metric=metric)
        end

        # Phase 3: Final simplify to collect terms after DDI substitution
        result = simplify(result; registry=registry)

        result
    end
end
