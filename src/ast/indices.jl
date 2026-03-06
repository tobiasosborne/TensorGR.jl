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
    # Group by (name, vbundle). An index is dummy if the same (name, vbundle)
    # appears with both Up and Down positions.
    key_counts = Dict{Tuple{Symbol,Symbol}, Vector{TIndex}}()
    for idx in all_idxs
        push!(get!(Vector{TIndex}, key_counts, (idx.name, idx.vbundle)), idx)
    end

    free = TIndex[]
    for ((name, vb), idxs) in key_counts
        has_up = any(i -> i.position == Up, idxs)
        has_down = any(i -> i.position == Down, idxs)
        if has_up && has_down
            up_count = count(i -> i.position == Up, idxs)
            down_count = count(i -> i.position == Down, idxs)
            paired = min(up_count, down_count)
            for _ in 1:(up_count - paired)
                push!(free, TIndex(name, Up, vb))
            end
            for _ in 1:(down_count - paired)
                push!(free, TIndex(name, Down, vb))
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
    key_groups = Dict{Tuple{Symbol,Symbol}, Vector{TIndex}}()
    for idx in all_idxs
        push!(get!(Vector{TIndex}, key_groups, (idx.name, idx.vbundle)), idx)
    end

    pairs = Tuple{TIndex, TIndex}[]
    for ((name, vb), idxs) in key_groups
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
        idx.name == old ? TIndex(new, idx.position, idx.vbundle) : idx
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
    new_idx = expr.index.name == old ? TIndex(new, expr.index.position, expr.index.vbundle) : expr.index
    TDeriv(new_idx, rename_dummy(expr.arg, old, new))
end

rename_dummy(expr::TScalar, ::Symbol, ::Symbol) = expr

"""
    index_sort(idxs::Vector{TIndex}; by=:name) -> Vector{TIndex}

Sort indices canonically: by name alphabetically, preserving position.
"""
function index_sort(idxs::Vector{TIndex}; by::Symbol=:name)
    if by == :name
        sort(idxs, by = idx -> idx.name)
    elseif by == :position
        sort(idxs, by = idx -> (idx.position == Up ? 0 : 1, idx.name))
    else
        sort(idxs, by = idx -> idx.name)
    end
end

"""
    same_dummies(expr::TSum) -> TSum

Rename dummies in all terms of a sum to use the same canonical names.
Improves readability without changing mathematical meaning.
"""
function same_dummies(expr::TSum)
    isempty(expr.terms) && return expr
    TSum(TensorExpr[_normalize_dummies_for_display(t) for t in expr.terms])
end

same_dummies(expr::TensorExpr) = expr

function _normalize_dummies_for_display(expr::TensorExpr)
    pairs = dummy_pairs(expr)
    isempty(pairs) && return expr

    all_idxs = indices(expr)
    first_occurrence = Dict{Symbol, Int}()
    for (i, idx) in enumerate(all_idxs)
        haskey(first_occurrence, idx.name) || (first_occurrence[idx.name] = i)
    end

    dummy_names = [p[1].name for p in pairs]
    sort!(dummy_names, by = n -> get(first_occurrence, n, 0))

    # Rename to standard dummy names: p, q, r, s, ...
    canonical_dummy_names = [:p, :q, :r, :s, :t, :u, :v, :w]
    result = expr
    used = Set(idx.name for idx in free_indices(expr))
    for n in canonical_dummy_names
        push!(used, n)
    end

    for (i, old_name) in enumerate(dummy_names)
        if i <= length(canonical_dummy_names)
            new_name = canonical_dummy_names[i]
        else
            new_name = Symbol("_d", i)
        end
        if old_name != new_name && new_name ∉ Set(idx.name for idx in free_indices(result))
            result = rename_dummy(result, old_name, new_name)
        end
    end
    result
end

"""
    split_index(expr::TensorExpr, idx::Symbol, dim::Int) -> TensorExpr

Replace an abstract index with a sum over component indices.
T_{a} → Σ_{μ=1}^{dim} T_{μ}  (returns a sum of expressions with component markers)
"""
function split_index(expr::TensorExpr, idx::Symbol, dim::Int)
    terms = TensorExpr[]
    for μ in 1:dim
        # Replace the abstract index with a component marker
        component_sym = Symbol(:_, idx, :_, μ)
        push!(terms, _replace_index_name(expr, idx, component_sym))
    end
    tsum(terms)
end

function _replace_index_name(t::Tensor, old::Symbol, new::Symbol)
    new_indices = map(t.indices) do idx
        idx.name == old ? TIndex(new, idx.position, idx.vbundle) : idx
    end
    Tensor(t.name, new_indices)
end

function _replace_index_name(p::TProduct, old::Symbol, new::Symbol)
    TProduct(p.scalar, TensorExpr[_replace_index_name(f, old, new) for f in p.factors])
end

function _replace_index_name(s::TSum, old::Symbol, new::Symbol)
    TSum(TensorExpr[_replace_index_name(t, old, new) for t in s.terms])
end

function _replace_index_name(d::TDeriv, old::Symbol, new::Symbol)
    new_idx = d.index.name == old ? TIndex(new, d.index.position, d.index.vbundle) : d.index
    TDeriv(new_idx, _replace_index_name(d.arg, old, new))
end

_replace_index_name(s::TScalar, ::Symbol, ::Symbol) = s

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
