#= Index-free tensor notation.
#
# An IndexFree object captures the contraction topology of a tensor product
# without explicit index names. Each tensor factor occupies a block of
# numbered slots, and contractions are represented as pairs of slot indices.
#
# This is the general-purpose version of the domain-specific RInv system
# (which handles only Riemann monomials). It follows the design of xTras's
# IndexFree/ToIndexFree/FromIndexFree, but additionally records the
# contraction pattern (not just tensor names).
#
# Usage:
#   indexed → IndexFree:  to_index_free(expr)
#   IndexFree → indexed:  from_index_free(ifree; free_names=[...])
#   Round-trip: from_index_free(to_index_free(expr)) ≈ expr (up to dummy names)
#
# References:
#   Nutma, *xTras: A field-theory inspired xAct package* (2014), Sec 5.
=#

"""
    IndexFree

Index-free representation of a tensor product, capturing the contraction
topology without explicit index names.

# Fields
- `scalar::Rational{Int}` -- overall scalar coefficient
- `factors::Vector{Symbol}` -- tensor names in order
- `slot_count::Int` -- total number of index slots across all factors
- `contractions::Vector{Tuple{Int,Int}}` -- pairs of global slot indices
  that are contracted (dummy pairs). Each pair `(i,j)` with `i < j`.
- `free_slots::Vector{Int}` -- global slot indices that are free (uncontracted)
- `free_positions::Vector{IndexPosition}` -- Up/Down for each free slot
- `slot_vbundles::Vector{Symbol}` -- vector bundle for each slot
"""
struct IndexFree
    scalar::Rational{Int}
    factors::Vector{Symbol}
    slot_count::Int
    contractions::Vector{Tuple{Int,Int}}
    free_slots::Vector{Int}
    free_positions::Vector{IndexPosition}
    slot_vbundles::Vector{Symbol}
end

function Base.:(==)(a::IndexFree, b::IndexFree)
    a.scalar == b.scalar &&
    a.factors == b.factors &&
    a.slot_count == b.slot_count &&
    a.contractions == b.contractions &&
    a.free_slots == b.free_slots &&
    a.free_positions == b.free_positions &&
    a.slot_vbundles == b.slot_vbundles
end

function Base.hash(a::IndexFree, h::UInt)
    hash(a.slot_vbundles,
        hash(a.free_positions,
            hash(a.free_slots,
                hash(a.contractions,
                    hash(a.slot_count,
                        hash(a.factors,
                            hash(a.scalar, hash(:IndexFree, h))))))))
end

function Base.show(io::IO, ifree::IndexFree)
    if ifree.scalar != 1 // 1
        print(io, ifree.scalar, " * ")
    end
    for (i, f) in enumerate(ifree.factors)
        i > 1 && print(io, " ⊗ ")
        print(io, f)
    end
    if !isempty(ifree.contractions)
        print(io, " [")
        for (i, (s1, s2)) in enumerate(ifree.contractions)
            i > 1 && print(io, ", ")
            print(io, s1, "↔", s2)
        end
        print(io, "]")
    end
    if !isempty(ifree.free_slots)
        print(io, " {free: ", join(ifree.free_slots, ","), "}")
    end
end

# ── ToIndexFree ────────────────────────────────────────────────────────

"""
    to_index_free(expr::TensorExpr; registry=current_registry()) -> IndexFree

Convert an indexed tensor expression to index-free form.

Extracts the contraction topology: which slots of which factors are
contracted (dummy pairs), and which are free.

Handles `Tensor`, `TProduct`, and `TScalar` expressions. `TSum` and
`TDeriv` are not supported (derivatives add implicit slots that
complicate the topology).

# Example
```julia
# R_{ab} R^{ab} → IndexFree with Ric⊗Ric and one contraction pair
expr = TProduct(1//1, [Tensor(:Ric, [down(:a), down(:b)]),
                        Tensor(:Ric, [up(:a), up(:b)])])
ifree = to_index_free(expr)
```
"""
function to_index_free(expr::TensorExpr;
                        registry::TensorRegistry=current_registry())
    _to_index_free(expr, registry)
end

function _to_index_free(t::Tensor, reg::TensorRegistry)
    n = length(t.indices)
    positions = [idx.position for idx in t.indices]
    vbundles = [idx.vbundle for idx in t.indices]

    # A single tensor: no contractions, all slots are free
    IndexFree(
        1 // 1,
        Symbol[t.name],
        n,
        Tuple{Int,Int}[],
        collect(1:n),
        positions,
        vbundles
    )
end

function _to_index_free(p::TProduct, reg::TensorRegistry)
    # Collect factors and their slot information
    factor_names = Symbol[]
    all_indices = TIndex[]
    slot_vbundles = Symbol[]
    slot_positions = IndexPosition[]

    for f in p.factors
        if f isa Tensor
            push!(factor_names, f.name)
            for idx in f.indices
                push!(all_indices, idx)
                push!(slot_vbundles, idx.vbundle)
                push!(slot_positions, idx.position)
            end
        elseif f isa TScalar
            # Scalars contribute no slots
            continue
        elseif f isa GammaMatrix
            push!(factor_names, :gamma)
            push!(all_indices, f.index)
            push!(slot_vbundles, f.index.vbundle)
            push!(slot_positions, f.index.position)
        else
            error("to_index_free: unsupported factor type $(typeof(f))")
        end
    end

    n = length(all_indices)

    # Find dummy pairs: indices that appear twice with matching (name, vbundle)
    # Build a map from (name, vbundle) → list of slot positions
    idx_map = Dict{Tuple{Symbol,Symbol}, Vector{Int}}()
    for (i, idx) in enumerate(all_indices)
        key = (idx.name, idx.vbundle)
        push!(get!(idx_map, key, Int[]), i)
    end

    contractions = Tuple{Int,Int}[]
    free_slots_set = Set(1:n)
    for (_, slots) in idx_map
        if length(slots) == 2
            s1, s2 = minmax(slots[1], slots[2])
            push!(contractions, (s1, s2))
            delete!(free_slots_set, s1)
            delete!(free_slots_set, s2)
        end
        # Indices appearing once are free; ≥3 is an error we don't check here
    end

    # Sort contractions for canonical ordering
    sort!(contractions)

    free_slots = sort!(collect(free_slots_set))
    free_pos = [slot_positions[s] for s in free_slots]

    IndexFree(
        p.scalar,
        factor_names,
        n,
        contractions,
        free_slots,
        free_pos,
        slot_vbundles
    )
end

function _to_index_free(s::TScalar, reg::TensorRegistry)
    IndexFree(
        s.val isa Rational ? s.val : 1 // 1,
        Symbol[],
        0,
        Tuple{Int,Int}[],
        Int[],
        IndexPosition[],
        Symbol[]
    )
end

# ── FromIndexFree ──────────────────────────────────────────────────────

"""
    from_index_free(ifree::IndexFree;
                     free_names::Vector{Symbol}=Symbol[],
                     registry=current_registry()) -> TensorExpr

Convert an IndexFree object back to an indexed TensorExpr by assigning
fresh index names.

Dummy indices are named `:_d1, :_d2, ...` for each contraction pair.
Free indices use `free_names` if provided, otherwise `:_f1, :_f2, ...`.

# Example
```julia
expr = from_index_free(ifree; free_names=[:a, :b])
```
"""
function from_index_free(ifree::IndexFree;
                          free_names::Vector{Symbol}=Symbol[],
                          registry::TensorRegistry=current_registry())
    if ifree.slot_count == 0
        return TScalar(ifree.scalar)
    end

    # Build the index assignment: slot → TIndex
    slot_indices = Vector{Union{TIndex,Nothing}}(nothing, ifree.slot_count)

    # Assign dummy indices for contractions
    for (k, (s1, s2)) in enumerate(ifree.contractions)
        dummy_name = Symbol(:_d, k)
        vb = ifree.slot_vbundles[s1]
        # Determine Up/Down: one must be Up and the other Down
        # Use the original positions stored in slot_vbundles context
        # For proper contraction: assign Up to the first slot's original position
        # and Down to the second, but we need actual positions.
        # Since we're reconstructing, use Up for s1 and Down for s2
        # (the contraction engine handles position matching).
        slot_indices[s1] = TIndex(dummy_name, Up, vb)
        slot_indices[s2] = TIndex(dummy_name, Down, vb)
    end

    # Assign free indices
    for (k, s) in enumerate(ifree.free_slots)
        if k <= length(free_names)
            fname = free_names[k]
        else
            fname = Symbol(:_f, k)
        end
        vb = ifree.slot_vbundles[s]
        pos = ifree.free_positions[k]
        slot_indices[s] = TIndex(fname, pos, vb)
    end

    # Build tensor factors
    factors = TensorExpr[]
    slot_offset = 0
    for fname in ifree.factors
        if fname === :gamma
            # GammaMatrix: 1 slot
            idx = slot_indices[slot_offset + 1]
            push!(factors, GammaMatrix(idx))
            slot_offset += 1
        else
            # Look up rank from registry to determine slot count
            if has_tensor(registry, fname)
                props = get_tensor(registry, fname)
                nslots = sum(props.rank)
            else
                # Count consecutive slots belonging to this factor
                # Fallback: count until next factor's slots begin
                nslots = _count_factor_slots(ifree, slot_offset, fname)
            end
            idxs = TIndex[slot_indices[slot_offset + i] for i in 1:nslots]
            push!(factors, Tensor(fname, idxs))
            slot_offset += nslots
        end
    end

    if length(factors) == 1 && ifree.scalar == 1 // 1
        return factors[1]
    end

    TProduct(ifree.scalar, factors)
end

# Count how many slots a factor occupies based on the IndexFree structure
function _count_factor_slots(ifree::IndexFree, offset::Int, name::Symbol)
    # Use vbundle transitions to detect factor boundaries
    # Fallback: count remaining slots divided by remaining factors
    remaining_slots = ifree.slot_count - offset
    remaining_factors = count(==(name), ifree.factors)
    remaining_factors > 0 ? remaining_slots ÷ remaining_factors : remaining_slots
end

# ── Utility functions ─────────────────────────────────────────────────

"""
    index_free_structure(expr::TensorExpr;
                          registry=current_registry()) -> Vector{Symbol}

Return just the tensor factor names (without indices), useful for
identifying the tensor structure of an expression.

Equivalent to xTras's `TermsOf` for a single term.
"""
function index_free_structure(expr::TensorExpr;
                               registry::TensorRegistry=current_registry())
    ifree = to_index_free(expr; registry=registry)
    ifree.factors
end

"""
    same_tensor_structure(a::IndexFree, b::IndexFree) -> Bool

Check if two IndexFree objects have the same tensor factors and
contraction topology (ignoring scalar coefficients).
"""
function same_tensor_structure(a::IndexFree, b::IndexFree)
    a.factors == b.factors &&
    a.contractions == b.contractions &&
    a.free_slots == b.free_slots &&
    a.free_positions == b.free_positions
end
