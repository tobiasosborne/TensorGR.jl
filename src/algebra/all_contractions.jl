#= Full contraction enumeration via metric.

Given a tensor expression with 2n free indices and a metric, generate all
(2n-1)!! distinct full contractions by pairing indices with metric tensors.
=#

"""
    all_contractions(expr::TensorExpr, metric_name::Symbol;
                     registry::TensorRegistry=current_registry()) -> Vector{TensorExpr}

Generate all distinct full contractions of `expr` using metric `metric_name`.
For a rank-2n expression with 2n free indices, returns up to (2n-1)!! contractions
(fewer after symmetry deduplication).

Odd-rank expressions cannot be fully contracted and throw an error.
Scalar expressions (no free indices) return `[expr]`.

# Examples
```julia
# Trace of a rank-2 tensor: 1 contraction
results = all_contractions(T_ab, :g)  # -> [g^{ab} T_{ab}]

# Rank-4 tensor: up to 3 = 3!! contractions
results = all_contractions(R_abcd, :g)  # -> 3 results
```
"""
function all_contractions(expr::TensorExpr, metric_name::Symbol;
                          registry::TensorRegistry=current_registry())
    fi = free_indices(expr)
    n = length(fi)

    # Scalar: already fully contracted
    n == 0 && return TensorExpr[expr]

    # Odd rank: impossible to fully contract
    if isodd(n)
        throw(ArgumentError("Cannot fully contract odd-rank expression (rank $n)"))
    end

    # Generate all pairings of free indices
    pairings = _all_pairings(collect(1:n))

    results = TensorExpr[]
    seen = Set{TensorExpr}()

    for pairing in pairings
        # Build metric product: one g^{..} per pair
        contracted = expr
        used_names = Set{Symbol}(idx.name for idx in indices(expr))

        for (i, j) in pairing
            idx_i = fi[i]
            idx_j = fi[j]

            # Generate fresh dummy names to avoid clashes
            d1 = fresh_index(used_names)
            push!(used_names, d1)
            d2 = fresh_index(used_names)
            push!(used_names, d2)

            # Build metric with indices opposite to the free indices
            # If free index is Down, metric index must be Up (to contract)
            pos_i = idx_i.position == Down ? Up : Down
            pos_j = idx_j.position == Down ? Up : Down
            vb = idx_i.vbundle

            g = Tensor(metric_name, [TIndex(d1, pos_i, vb), TIndex(d2, pos_j, vb)])

            # Rename the free indices in expr to match the dummy names
            contracted = rename_dummy(contracted, idx_i.name, d1)
            contracted = rename_dummy(contracted, idx_j.name, d2)
            contracted = contracted * g
        end

        # Simplify and canonicalize
        result = simplify(contracted; registry=registry)

        # Skip zeros
        result == TScalar(0 // 1) && continue

        # Deduplicate via normalized dummy names
        normed = _normalize_for_dedup(result)
        if normed in seen
            continue
        end
        # Also check negation
        neg_normed = _normalize_for_dedup(_negate_expr(result))
        if neg_normed in seen
            continue
        end

        push!(seen, normed)
        push!(results, result)
    end

    results
end

"""
Generate all ways to partition indices 1:2n into n pairs.
Returns vector of vectors of (i,j) tuples.
For 2n items, produces (2n-1)!! = 1*3*5*...*(2n-1) pairings.
"""
function _all_pairings(items::Vector{Int})
    n = length(items)
    n == 0 && return [Tuple{Int,Int}[]]
    n == 2 && return [[(items[1], items[2])]]

    first = items[1]
    rest = items[2:end]
    result = Vector{Vector{Tuple{Int,Int}}}()
    for (i, partner) in enumerate(rest)
        remaining = [rest[j] for j in eachindex(rest) if j != i]
        for sub in _all_pairings(remaining)
            push!(result, [(first, partner); sub])
        end
    end
    result
end
