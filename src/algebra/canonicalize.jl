#= Canonicalization via xperm.c.

For a product of tensors with total k index slots, the permutation domain
has n = k + 2 points.  Points 1..k are index slots; points k+1 and k+2
carry the sign (+/-).

xperm convention:
  - Permutation perm[slot] = name: a bijection mapping slot positions to
    canonical integer "names". Each slot gets a unique name.
  - Dummy pairs get TWO consecutive names: pair k gets (2k-1, 2k).
  - Free indices get names after all dummies.
  - freeps/dummyps passed to canonical_perm are the names of the indices
    (in the identity config, name i sits in slot i, so names = initial slots).
  - The perm MUST be a proper bijection (no repeated values).

For products containing derivatives (TDeriv), we use an "implode/explode"
strategy: derivatives are flattened into compound indexed objects with
their derivative indices prepended, canonicalized together, then split back.
=#

"""
    canonicalize(expr::TensorExpr) -> TensorExpr

Canonicalize index ordering using slot symmetries (via xperm.c).
"""
canonicalize(s::TScalar) = s

function canonicalize(t::Tensor)
    reg = current_registry()
    has_tensor(reg, t.name) || return t
    props = get_tensor(reg, t.name)
    isempty(props.symmetries) && return t
    _canonicalize_product(TProduct(1 // 1, TensorExpr[t]))
end

function canonicalize(s::TSum)
    tsum(TensorExpr[canonicalize(t) for t in s.terms])
end

function canonicalize(d::TDeriv)
    TDeriv(d.index, canonicalize(d.arg), d.covd)
end

function canonicalize(p::TProduct)
    sum_idx = findfirst(f -> f isa TSum, p.factors)
    if sum_idx !== nothing
        s = p.factors[sum_idx]::TSum
        other = TensorExpr[p.factors[i] for i in eachindex(p.factors) if i != sum_idx]
        terms = TensorExpr[canonicalize(tproduct(p.scalar, vcat(other, [t])))
                           for t in s.terms]
        return tsum(terms)
    end
    _canonicalize_product(p)
end

# ─── Implode/Explode: flatten derivatives into compound objects ──────

struct _ImplodedObject
    tensor_name::Symbol
    deriv_indices::Vector{TIndex}
    tensor_indices::Vector{TIndex}
    deriv_covds::Vector{Symbol}  # one covd per derivative index
end

_ImplodedObject(name::Symbol, derivs::Vector{TIndex}, indices::Vector{TIndex}) =
    _ImplodedObject(name, derivs, indices, fill(:partial, length(derivs)))

_ImplodedObject(name::Symbol, derivs::Vector{TIndex}, indices::Vector{TIndex}, covd::Symbol) =
    _ImplodedObject(name, derivs, indices, fill(covd, length(derivs)))

function _all_indices(obj::_ImplodedObject)
    vcat(obj.deriv_indices, obj.tensor_indices)
end

function _implode(expr::Tensor)
    _ImplodedObject(expr.name, TIndex[], copy(expr.indices), Symbol[])
end

function _implode(expr::TDeriv)
    derivs = TIndex[]
    covds = Symbol[]
    inner = expr
    while inner isa TDeriv
        push!(derivs, inner.index)
        push!(covds, inner.covd)
        inner = inner.arg
    end
    inner isa Tensor || return nothing
    # Refuse to implode mixed-covd chains: canonicalization would scramble
    # derivative indices across different covd types
    if length(covds) >= 2 && !all(==(covds[1]), covds)
        return nothing
    end
    _ImplodedObject(inner.name, derivs, copy(inner.indices), covds)
end

_implode(::TScalar) = nothing
_implode(::TProduct) = nothing
_implode(::TSum) = nothing

function _explode(obj::_ImplodedObject)
    result = Tensor(obj.tensor_name, obj.tensor_indices)
    for i in length(obj.deriv_indices):-1:1
        covd = i <= length(obj.deriv_covds) ? obj.deriv_covds[i] : :partial
        result = TDeriv(obj.deriv_indices[i], result, covd)
    end
    result
end

# ─── Core engine ─────────────────────────────────────────────────────

function _canonicalize_product(p::TProduct)
    reg = current_registry()

    # Try to implode all factors into indexed objects
    imploded = _ImplodedObject[]
    non_tensor_factors = TensorExpr[]
    for f in p.factors
        obj = _implode(f)
        if obj !== nothing
            push!(imploded, obj)
        elseif f isa TScalar
            push!(non_tensor_factors, f)
        else
            return p
        end
    end

    isempty(imploded) && return p

    # Gather all indices and their slot positions
    all_indices = TIndex[]
    slot_ranges = UnitRange{Int}[]
    pos = 1
    for obj in imploded
        idxs = _all_indices(obj)
        r = pos:(pos + length(idxs) - 1)
        push!(slot_ranges, r)
        append!(all_indices, idxs)
        pos += length(idxs)
    end

    nslots = length(all_indices)
    nslots < 2 && return p
    n = nslots + 2

    # ── Classify free and dummy indices ──────────────────────────────
    name_slots = Dict{Symbol, Vector{Tuple{Int, IndexPosition}}}()
    for (slot, idx) in enumerate(all_indices)
        push!(get!(Vector{Tuple{Int,IndexPosition}}, name_slots, idx.name), (slot, idx.position))
    end

    dummy_info = Vector{Tuple{Symbol, Int, Int}}()  # (sym, up_slot, down_slot)
    free_slots_list = Vector{Tuple{Symbol, Int}}()

    for (sym, occurrences) in name_slots
        ups   = [(s, p) for (s, p) in occurrences if p == Up]
        downs = [(s, p) for (s, p) in occurrences if p == Down]

        if length(ups) == 1 && length(downs) == 1
            push!(dummy_info, (sym, ups[1][1], downs[1][1]))
        else
            for (s, _) in occurrences
                push!(free_slots_list, (sym, s))
            end
        end
    end

    # ── Name assignment ──────────────────────────────────────────────
    # xperm requires a BIJECTIVE perm. Every slot gets a unique name.
    # We treat ALL indices as free for xperm (dummies get unique names
    # but are listed as free). Dummy renaming is handled separately in
    # collect_terms via _normalize_dummies.
    #
    # This avoids xperm's double-coset algorithm moving dummy names
    # between structurally different slots (e.g., derivative vs tensor).

    # Sort all slots by symbol for deterministic name assignment
    all_slots_sorted = sort(collect(1:nslots), by = i -> all_indices[i].name)

    slot_to_name = Dict{Int, Int}()
    for (name, slot) in enumerate(all_slots_sorted)
        slot_to_name[slot] = name
    end

    # All indices treated as free for xperm
    freeps = Int32.(1:nslots)
    dummyps = Int32[]

    perm_data = Vector{Int32}(undef, n)
    for i in 1:nslots
        perm_data[i] = Int32(slot_to_name[i])
    end
    perm_data[n - 1] = Int32(n - 1)
    perm_data[n]     = Int32(n)

    # ── Symmetry generators ──────────────────────────────────────────
    # xperm uses left-action: cperm = g ∘ perm. Slot generators must be
    # conjugated so that left-action produces the physical slot swap:
    #   (perm ∘ gen_slot ∘ perm⁻¹) ∘ perm = perm ∘ gen_slot
    perm = Perm(perm_data)
    perm_inv_data = perm_inverse(perm)

    all_gens = Perm[]
    for (oi, obj) in enumerate(imploded)
        has_tensor(reg, obj.tensor_name) || continue
        props = get_tensor(reg, obj.tensor_name)
        isempty(props.symmetries) && continue

        nderiv = length(obj.deriv_indices)
        nslots_t = length(obj.tensor_indices)

        local_gens = symmetry_generators(props.symmetries, nslots_t)
        offset = slot_ranges[oi][1] - 1 + nderiv
        for lg in local_gens
            pg = collect(Int32, 1:n)
            for j in 1:nslots_t
                pg[offset + j] = Int32(offset + lg.data[j])
            end
            if lg.data[nslots_t + 1] != nslots_t + 1
                pg[n - 1], pg[n] = pg[n], pg[n - 1]
            end
            # Conjugate: g_conj = perm ∘ gen_slot ∘ perm⁻¹
            conj = Vector{Int32}(undef, n)
            for i in 1:n
                conj[i] = perm.data[Int(pg[Int(perm_inv_data.data[i])])]
            end
            push!(all_gens, Perm(conj))
        end

        # Commuting partial derivatives: ∂_a ∂_b = ∂_b ∂_a
        if nderiv >= 2
            deriv_offset = slot_ranges[oi][1] - 1
            for k in 1:(nderiv - 1)
                pg = collect(Int32, 1:n)
                pg[deriv_offset + k] = Int32(deriv_offset + k + 1)
                pg[deriv_offset + k + 1] = Int32(deriv_offset + k)
                # Conjugate derivative swap too
                conj = Vector{Int32}(undef, n)
                for i in 1:n
                    conj[i] = perm.data[Int(pg[Int(perm_inv_data.data[i])])]
                end
                push!(all_gens, Perm(conj))
            end
        end
    end

    isempty(all_gens) && isempty(dummyps) && return p

    # ── Call xperm.c ─────────────────────────────────────────────────
    base = Int32.(1:nslots)

    cperm = xperm_canonical_perm(perm, base, all_gens, freeps, dummyps, n)

    # Zero?
    all(==(Int32(0)), cperm.data) && return ZERO

    # Sign
    sign = cperm.data[n - 1] == Int32(n - 1) ? 1 : -1

    # ── Reconstruct indices from canonical permutation ───────────────
    # With conjugated generators, cperm = perm ∘ π (physical slot perm).
    # cperm[slot] = name at slot in canonical config.
    name_to_sym = Dict{Int, Symbol}()
    for (slot, name) in slot_to_name
        name_to_sym[name] = all_indices[slot].name
    end

    new_all_indices = Vector{TIndex}(undef, nslots)
    for slot in 1:nslots
        cname = Int(cperm.data[slot])
        sym = name_to_sym[cname]
        new_all_indices[slot] = TIndex(sym, all_indices[slot].position, all_indices[slot].vbundle)
    end

    # Explode back into TDeriv chains
    new_factors = copy(non_tensor_factors)
    for (oi, obj) in enumerate(imploded)
        range = slot_ranges[oi]
        new_idxs = new_all_indices[range]
        nderiv = length(obj.deriv_indices)
        new_obj = _ImplodedObject(
            obj.tensor_name,
            new_idxs[1:nderiv],
            new_idxs[nderiv+1:end],
            obj.deriv_covds
        )
        push!(new_factors, _explode(new_obj))
    end

    # Sort factors by canonical key so that e.g. RicScalar*g and g*RicScalar
    # produce the same TProduct, enabling collect_terms to merge them.
    sort!(new_factors, by=_factor_sort_key)

    tproduct(p.scalar * (sign // 1), new_factors)
end

# ─── Factor sort key for canonical product ordering ──────────────────

function _factor_sort_key(f::TensorExpr)
    if f isa TScalar
        return (0, "", "")
    elseif f isa Tensor
        idx_str = join([string(idx.name, idx.position == Up ? "^" : "_") for idx in f.indices])
        return (1, string(f.name), idx_str)
    elseif f isa TDeriv
        inner = f
        depth = 0
        while inner isa TDeriv
            depth += 1
            inner = inner.arg
        end
        name = inner isa Tensor ? string(inner.name) : ""
        idx_str = join([string(idx.name, idx.position == Up ? "^" : "_") for idx in indices(f)])
        return (2, name * "D$depth", idx_str)
    else
        return (3, "", "")
    end
end

# ─── Fix same-position dummy pairs ───────────────────────────────────
#
# The all-free xperm mode can produce expressions where a dummy index
# name appears twice with the same position (e.g. both Down), when a
# symmetry generator swaps names between slots of different positions.
# This is a valid canonical form for merging purposes, but the resulting
# expression has invalid dummy structure.
#
# fix_dummy_positions repairs this by flipping the position of one
# occurrence so that each dummy pair has one Up and one Down.

"""
    fix_dummy_positions(expr::TensorExpr) -> TensorExpr

Fix same-position dummy pairs in an expression. When canonicalization
produces a term where a dummy index appears twice with the same position
(both Up or both Down), flip one occurrence to restore valid (Up, Down)
pairing. This preserves the mathematical meaning because the affected
slots are always related by a tensor symmetry.

Use this after simplification when downstream operations (e.g. Fourier
transform, spin projection) require valid dummy pairs.
"""
fix_dummy_positions(s::TScalar) = s
fix_dummy_positions(t::Tensor) = t
fix_dummy_positions(d::TDeriv) = TDeriv(d.index, fix_dummy_positions(d.arg), d.covd)

function fix_dummy_positions(s::TSum)
    tsum(TensorExpr[fix_dummy_positions(t) for t in s.terms])
end

function fix_dummy_positions(p::TProduct)
    # Collect all indices with their slot locations
    all_idxs = TIndex[]
    slot_offsets = Int[]  # which factor each slot belongs to
    slot_locals = Int[]   # local index within the factor
    for (fi, f) in enumerate(p.factors)
        idxs = indices(f)
        for (li, idx) in enumerate(idxs)
            push!(all_idxs, idx)
            push!(slot_offsets, fi)
            push!(slot_locals, li)
        end
    end

    # Find same-position pairs
    name_slots = Dict{Symbol, Vector{Tuple{Int, IndexPosition}}}()
    for (slot, idx) in enumerate(all_idxs)
        push!(get!(Vector{Tuple{Int,IndexPosition}}, name_slots, idx.name), (slot, idx.position))
    end

    # Identify violated slots and the position they should have
    fixes = Dict{Int, IndexPosition}()  # slot => corrected position
    for (sym, occs) in name_slots
        length(occs) == 2 || continue
        s1, p1 = occs[1]
        s2, p2 = occs[2]
        p1 == p2 || continue  # only fix same-position violations
        # Flip the second occurrence
        fixes[s2] = (p2 == Up ? Down : Up)
    end

    isempty(fixes) && return p

    # Apply fixes: rebuild the factors with corrected positions
    new_factors = TensorExpr[]
    slot = 1
    for (fi, f) in enumerate(p.factors)
        idxs = indices(f)
        needs_fix = false
        for li in 1:length(idxs)
            if haskey(fixes, slot + li - 1)
                needs_fix = true
                break
            end
        end
        if needs_fix
            push!(new_factors, _apply_position_fixes(f, slot, fixes))
        else
            push!(new_factors, f)
        end
        slot += length(idxs)
    end

    tproduct(p.scalar, new_factors)
end

function _apply_position_fixes(t::Tensor, base_slot::Int, fixes::Dict{Int, IndexPosition})
    new_indices = TIndex[]
    for (li, idx) in enumerate(t.indices)
        slot = base_slot + li - 1
        if haskey(fixes, slot)
            push!(new_indices, TIndex(idx.name, fixes[slot], idx.vbundle))
        else
            push!(new_indices, idx)
        end
    end
    Tensor(t.name, new_indices)
end

function _apply_position_fixes(d::TDeriv, base_slot::Int, fixes::Dict{Int, IndexPosition})
    if haskey(fixes, base_slot)
        new_idx = TIndex(d.index.name, fixes[base_slot], d.index.vbundle)
        TDeriv(new_idx, _apply_position_fixes(d.arg, base_slot + 1, fixes), d.covd)
    else
        TDeriv(d.index, _apply_position_fixes(d.arg, base_slot + 1, fixes), d.covd)
    end
end

function _apply_position_fixes(s::TScalar, base_slot::Int, fixes::Dict{Int, IndexPosition})
    s
end

"""
    normalize_field_positions(expr, field; metric=:g) -> TensorExpr

Lower all Up indices on `field` tensors to Down by inserting inverse metric
connectors (g^{old_up, fresh_up}). The metrics become part of the product,
ensuring that when `extract_kernel` separates h factors from the coefficient,
the h factors are all-Down with disjoint index names.

IMPORTANT: Do NOT call `simplify` after this — `contract_metrics` will undo
the lowering. Pass the result directly to `extract_kernel`.
"""
normalize_field_positions(s::TScalar, field::Symbol; metric::Symbol=:g) = s
normalize_field_positions(d::TDeriv, field::Symbol; metric::Symbol=:g) =
    TDeriv(d.index, normalize_field_positions(d.arg, field; metric), d.covd)

function normalize_field_positions(t::Tensor, field::Symbol; metric::Symbol=:g)
    t.name != field && return t
    any(idx -> idx.position == Up, t.indices) || return t
    normalize_field_positions(tproduct(1 // 1, TensorExpr[t]), field; metric)
end

function normalize_field_positions(s::TSum, field::Symbol; metric::Symbol=:g)
    tsum(TensorExpr[normalize_field_positions(t, field; metric) for t in s.terms])
end

function normalize_field_positions(p::TProduct, field::Symbol; metric::Symbol=:g)
    all_names = Set{Symbol}()
    for f in p.factors
        for idx in indices(f)
            push!(all_names, idx.name)
        end
    end

    new_factors = TensorExpr[]
    for f in p.factors
        if f isa Tensor && f.name == field
            lowered_idxs = TIndex[]
            connectors = TensorExpr[]
            for idx in f.indices
                if idx.position == Up
                    fn = fresh_index(all_names)
                    push!(all_names, fn)
                    push!(lowered_idxs, TIndex(fn, Down, idx.vbundle))
                    push!(connectors, Tensor(metric, [idx, TIndex(fn, Up, idx.vbundle)]))
                else
                    push!(lowered_idxs, idx)
                end
            end
            push!(new_factors, Tensor(field, lowered_idxs))
            append!(new_factors, connectors)
        else
            push!(new_factors, f)
        end
    end

    tproduct(p.scalar, new_factors)
end
