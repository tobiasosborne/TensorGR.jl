#= Ansatz construction.

Utilities for constructing the most general tensor expression with given
index structure and symmetry properties.
=#

"""
    make_ansatz(terms::Vector{TensorExpr}, coeffs::Vector{Symbol}) -> TensorExpr

Create a linear combination: c₁ T₁ + c₂ T₂ + ...
where cᵢ are symbolic scalar coefficients.
"""
function make_ansatz(terms::Vector{TensorExpr}, coeffs::Vector{Symbol})
    @assert length(terms) == length(coeffs)
    result = TensorExpr[]
    for (t, c) in zip(terms, coeffs)
        push!(result, TScalar(c) * t)
    end
    tsum(result)
end

"""
    make_ansatz(terms::Vector{TensorExpr}) -> TensorExpr

Create a linear combination with auto-generated coefficient names c1, c2, ...
"""
function make_ansatz(terms::Vector{TensorExpr})
    coeffs = [Symbol(:c, i) for i in eachindex(terms)]
    make_ansatz(terms, coeffs)
end

"""
    all_contractions(tensors::Vector{Tensor}, free_idxs::Vector{TIndex}) -> Vector{TensorExpr}

Enumerate all independent index contractions of the given tensors that produce
expressions with the specified free indices.

Algorithm: rename indices to avoid clashes, generate all perfect matchings of
the contractible slots, assign dummy names, canonicalize, and deduplicate.
"""
function all_contractions(tensors::Vector{Tensor}, free_idxs::Vector{TIndex})
    isempty(tensors) && return TensorExpr[]

    # Step 1: Rename indices on each tensor to avoid clashes
    renamed = _rename_all_indices(tensors)

    # Step 2: Collect all index slots (position in flat list → index)
    all_idxs = TIndex[]
    # Track which tensor owns each slot
    tensor_owners = Int[]
    slot_in_tensor = Int[]
    for (ti, t) in enumerate(renamed)
        for (si, idx) in enumerate(t.indices)
            push!(all_idxs, idx)
            push!(tensor_owners, ti)
            push!(slot_in_tensor, si)
        end
    end

    n = length(all_idxs)
    free_names = Set(idx.name for idx in free_idxs)

    # Step 3: Identify free vs contractible slots
    free_slots = Int[]
    contract_slots = Int[]
    for i in 1:n
        if all_idxs[i].name in free_names
            push!(free_slots, i)
        else
            push!(contract_slots, i)
        end
    end

    # Must have even number of contractible slots
    if isodd(length(contract_slots))
        return TensorExpr[]
    end

    if isempty(contract_slots)
        return [tproduct(1 // 1, TensorExpr[t for t in renamed])]
    end

    # Step 4: Generate all perfect matchings of contractible slots
    matchings = _perfect_matchings(contract_slots)

    # Step 5: For each matching, build expression with dummy contractions
    results = TensorExpr[]
    seen_canonical = Set{TensorExpr}()

    used_names = Set{Symbol}(idx.name for idx in all_idxs)
    for m in matchings
        # Assign fresh dummy name to each pair, set one Up and one Down
        new_indices = copy(all_idxs)
        for (s1, s2) in m
            dname = fresh_index(used_names)
            push!(used_names, dname)
            vb = all_idxs[s1].vbundle
            new_indices[s1] = TIndex(dname, Up, vb)
            new_indices[s2] = TIndex(dname, Down, vb)
        end

        # Rebuild tensors with new indices
        new_tensors = TensorExpr[]
        offset = 0
        for (ti, t) in enumerate(renamed)
            ni = length(t.indices)
            tidxs = new_indices[offset+1:offset+ni]
            push!(new_tensors, Tensor(t.name, tidxs))
            offset += ni
        end

        expr = tproduct(1 // 1, new_tensors)
        cexpr = canonicalize(expr)

        # Skip zero expressions
        cexpr == TScalar(0 // 1) && continue

        # Normalize dummy names for reliable deduplication
        nexpr = _normalize_for_dedup(cexpr)

        if nexpr ∉ seen_canonical
            # Also check negation (canonicalize may produce sign differences)
            neg_expr = _negate_expr(cexpr)
            nneg = _normalize_for_dedup(canonicalize(neg_expr))
            if nneg ∉ seen_canonical
                push!(seen_canonical, nexpr)
                push!(results, cexpr)
            end
        end
    end

    results
end

"""Normalize dummy names to canonical set for deduplication."""
function _normalize_for_dedup(expr::TensorExpr)
    pairs = dummy_pairs(expr)
    isempty(pairs) && return expr
    all_idxs = indices(expr)
    first_occ = Dict{Symbol, Int}()
    for (i, idx) in enumerate(all_idxs)
        haskey(first_occ, idx.name) || (first_occ[idx.name] = i)
    end
    dummy_names = sort!([p[1].name for p in pairs], by = n -> get(first_occ, n, 0))
    canonical = [:_p, :_q, :_r, :_s, :_t, :_u, :_v, :_w]
    result = expr
    for (i, old) in enumerate(dummy_names)
        new = i <= length(canonical) ? canonical[i] : Symbol("_d", i)
        old != new && (result = rename_dummy(result, old, new))
    end
    result
end

"""Negate an expression for sign-aware deduplication."""
function _negate_expr(expr::TProduct)
    tproduct(-expr.scalar, copy(expr.factors))
end
_negate_expr(expr::TensorExpr) = tproduct(-1 // 1, TensorExpr[expr])

"""Rename all indices across tensors to unique names, avoiding clashes."""
function _rename_all_indices(tensors::Vector{Tensor})
    used = Set{Symbol}()
    result = Tensor[]
    for t in tensors
        new_indices = TIndex[]
        for idx in t.indices
            new_name = fresh_index(used)
            push!(used, new_name)
            push!(new_indices, TIndex(new_name, idx.position, idx.vbundle))
        end
        push!(result, Tensor(t.name, new_indices))
    end
    result
end

"""
Generate all perfect matchings of items into pairs.
For 2k items, returns (2k-1)!! matchings.
"""
function _perfect_matchings(items::Vector{Int})
    n = length(items)
    n == 0 && return [Tuple{Int,Int}[]]
    n == 2 && return [[(items[1], items[2])]]
    first = items[1]
    rest = items[2:end]
    result = Vector{Vector{Tuple{Int,Int}}}()
    for (i, partner) in enumerate(rest)
        remaining = [rest[j] for j in eachindex(rest) if j != i]
        for sub in _perfect_matchings(remaining)
            push!(result, [(first, partner); sub])
        end
    end
    result
end

"""
    contraction_ansatz(tensors::Vector{Tensor}, free_idxs::Vector{TIndex};
                       coeffs::Union{Vector{Symbol}, Nothing}=nothing) -> TensorExpr

Build the most general linear combination of contractions of the given tensors
with the specified free indices.
"""
function contraction_ansatz(tensors::Vector{Tensor}, free_idxs::Vector{TIndex};
                            coeffs::Union{Vector{Symbol}, Nothing}=nothing)
    contractions = all_contractions(tensors, free_idxs)
    if coeffs === nothing
        coeffs_actual = [Symbol(:c, i) for i in eachindex(contractions)]
    else
        coeffs_actual = coeffs
    end
    make_ansatz(contractions, coeffs_actual)
end
