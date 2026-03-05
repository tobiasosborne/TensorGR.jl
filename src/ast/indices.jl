"""
    indices(expr::TensorExpr) -> Vector{TIndex}

Return all indices in the expression (both free and dummy).
"""
indices(t::Tensor) = copy(t.indices)
indices(::TScalar) = TIndex[]
indices(d::TDeriv) = vcat(TIndex[d.index], indices(d.arg))

function indices(p::TProduct)
    result = TIndex[]
    for f in p.factors
        append!(result, indices(f))
    end
    result
end

function indices(s::TSum)
    # All terms should have the same free indices; return indices of first term
    isempty(s.terms) ? TIndex[] : indices(s.terms[1])
end

"""
    free_indices(expr::TensorExpr) -> Vector{TIndex}

Return the free (uncontracted) indices. A free index appears exactly once.
A dummy index appears as a pair (one Up, one Down, same name).
"""
function free_indices(expr::TensorExpr)
    all_idxs = indices(expr)
    # Count each (name, position) pair. An index is dummy if the same name
    # appears with both Up and Down positions.
    name_counts = Dict{Symbol, Vector{TIndex}}()
    for idx in all_idxs
        push!(get!(Vector{TIndex}, name_counts, idx.name), idx)
    end

    free = TIndex[]
    for (name, idxs) in name_counts
        has_up = any(i -> i.position == Up, idxs)
        has_down = any(i -> i.position == Down, idxs)
        if has_up && has_down
            # Dummy — skip paired indices, keep any unpaired
            up_count = count(i -> i.position == Up, idxs)
            down_count = count(i -> i.position == Down, idxs)
            # Standard case: one up, one down → fully dummy
            # If counts differ, the extras are free
            paired = min(up_count, down_count)
            for _ in 1:(up_count - paired)
                push!(free, TIndex(name, Up))
            end
            for _ in 1:(down_count - paired)
                push!(free, TIndex(name, Down))
            end
        else
            append!(free, idxs)
        end
    end
    free
end

"""
    dummy_pairs(expr::TensorExpr) -> Vector{Tuple{TIndex, TIndex}}

Return pairs of contracted indices (one Up, one Down, same name).
"""
function dummy_pairs(expr::TensorExpr)
    all_idxs = indices(expr)
    name_groups = Dict{Symbol, Vector{TIndex}}()
    for idx in all_idxs
        push!(get!(Vector{TIndex}, name_groups, idx.name), idx)
    end

    pairs = Tuple{TIndex, TIndex}[]
    for (name, idxs) in name_groups
        ups = filter(i -> i.position == Up, idxs)
        downs = filter(i -> i.position == Down, idxs)
        for i in 1:min(length(ups), length(downs))
            push!(pairs, (ups[i], downs[i]))
        end
    end
    pairs
end

"""
    fresh_index(used::Set{Symbol}) -> Symbol

Generate a fresh index name not in `used`.
Tries the standard alphabet a-z first, then a1, b1, ... etc.
"""
function fresh_index(used::Set{Symbol})
    for c in 'a':'z'
        s = Symbol(c)
        s in used || return s
    end
    # Extended names
    for n in 1:100
        for c in 'a':'z'
            s = Symbol(c, n)
            s in used || return s
        end
    end
    error("Could not generate fresh index (exhausted 2600+ names)")
end

"""
    rename_dummy(expr::TensorExpr, old::Symbol, new::Symbol) -> TensorExpr

Rename all occurrences of index name `old` to `new` (both Up and Down).
"""
function rename_dummy(expr::Tensor, old::Symbol, new::Symbol)
    new_indices = map(expr.indices) do idx
        idx.name == old ? TIndex(new, idx.position) : idx
    end
    Tensor(expr.name, new_indices)
end

function rename_dummy(expr::TProduct, old::Symbol, new::Symbol)
    TProduct(expr.scalar, TensorExpr[rename_dummy(f, old, new) for f in expr.factors])
end

function rename_dummy(expr::TSum, old::Symbol, new::Symbol)
    TSum(TensorExpr[rename_dummy(t, old, new) for t in expr.terms])
end

function rename_dummy(expr::TDeriv, old::Symbol, new::Symbol)
    new_idx = expr.index.name == old ? TIndex(new, expr.index.position) : expr.index
    TDeriv(new_idx, rename_dummy(expr.arg, old, new))
end

rename_dummy(expr::TScalar, ::Symbol, ::Symbol) = expr

"""
    ensure_no_dummy_clash(a::TensorExpr, b::TensorExpr) -> TensorExpr

Return a modified version of `b` where any dummy index names that clash
with dummy names in `a` have been renamed to fresh names.
"""
function ensure_no_dummy_clash(a::TensorExpr, b::TensorExpr)
    dummies_a = Set(pair[1].name for pair in dummy_pairs(a))
    dummies_b = Set(pair[1].name for pair in dummy_pairs(b))
    clashing = intersect(dummies_a, dummies_b)

    isempty(clashing) && return b

    all_names = Set{Symbol}()
    for idx in indices(a)
        push!(all_names, idx.name)
    end
    for idx in indices(b)
        push!(all_names, idx.name)
    end

    result = b
    for name in clashing
        new_name = fresh_index(all_names)
        push!(all_names, new_name)
        result = rename_dummy(result, name, new_name)
    end
    result
end
