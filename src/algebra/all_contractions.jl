#= Full contraction enumeration via metric.

Given a tensor expression with 2n free indices and a metric, generate all
(2n-1)!! distinct full contractions by pairing indices with metric tensors.
=#

"""
    all_contractions(expr::TensorExpr, metric_name::Symbol;
                     registry::TensorRegistry=current_registry(),
                     filter::Bool=true) -> Vector{TensorExpr}

Generate all distinct full contractions of `expr` using metric `metric_name`.
For a rank-2n expression with 2n free indices, returns up to (2n-1)!! contractions
(fewer after symmetry deduplication).

When `filter=true` (default), applies `filter_independent_contractions` to remove
contractions that are linearly dependent after full canonicalization (e.g., due to
Riemann symmetries or Bianchi identity).

Odd-rank expressions cannot be fully contracted and throw an error.
Scalar expressions (no free indices) return `[expr]`.

# Examples
```julia
# Trace of a rank-2 tensor: 1 contraction
results = all_contractions(T_ab, :g)  # -> [g^{ab} T_{ab}]

# Rank-4 tensor: up to 3 = 3!! contractions
results = all_contractions(R_abcd, :g)  # -> 3 results (fewer with symmetries)
```
"""
function all_contractions(expr::TensorExpr, metric_name::Symbol;
                          registry::TensorRegistry=current_registry(),
                          filter::Bool=true)
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

    filter ? filter_independent_contractions(results; registry=registry) : results
end

"""
    contraction_ansatz(tensor_names::Vector{Symbol}, metric_name::Symbol;
                       registry::TensorRegistry=current_registry(),
                       coeff_prefix::Symbol=:c) -> TSum

Build the most general scalar formed by fully contracting products of the given tensors.
Returns a `TSum` where each term is `cᵢ * (fully contracted product)`, with `cᵢ` as
symbolic `TScalar` coefficients.

Each tensor is looked up in the registry to determine its rank and index structure.
Fresh index names (`:_c1, :_c2, ...`) are used to avoid collisions with user indices.

# Examples
```julia
# Two independent quadratic Ricci invariants: R_{ab}R^{ab} and R²
contraction_ansatz([:Ric, :Ric], :g)  # -> c1 * R_{ab}R^{ab} + c2 * R²
```
"""
function contraction_ansatz(tensor_names::Vector{Symbol}, metric_name::Symbol;
                            registry::TensorRegistry=current_registry(),
                            coeff_prefix::Symbol=:c)
    # Build tensor expressions with fresh indices (all down)
    used = Set{Symbol}()
    tensors = TensorExpr[]
    for tname in tensor_names
        tp = get_tensor(registry, tname)
        total_rank = tp.rank[1] + tp.rank[2]
        idxs = TIndex[]
        for _ in 1:total_rank
            name = _fresh_ansatz_index(used)
            push!(used, name)
            push!(idxs, TIndex(name, Down, :Tangent))
        end
        push!(tensors, Tensor(tname, idxs))
    end

    # Form the product
    product = tproduct(1 // 1, tensors)

    # Generate all fully-contracted scalars
    contractions = all_contractions(product, metric_name; registry=registry)

    # Assign symbolic coefficients
    coeffs = [Symbol(coeff_prefix, i) for i in eachindex(contractions)]
    make_ansatz(contractions, coeffs)
end

# ─── Contraction filtering by symmetry ────────────────────────────────────

"""
    _canonical_structure(expr::TensorExpr) -> (normalized::TensorExpr, coeff::Rational{Int})

Extract the scalar coefficient and normalized tensor structure of a fully-contracted
expression. Two expressions with the same normalized structure are linearly dependent
(differ only by a scalar factor).

Returns `(structure, coefficient)` where `structure` has coefficient 1 and
canonically renamed dummy indices.
"""
function _canonical_structure(expr::TensorExpr)
    scalar, core = _split_scalar(expr)
    normalized = _normalize_dummies(core)
    (normalized, scalar)
end

"""
    filter_independent_contractions(contractions::Vector{TensorExpr};
                                    registry::TensorRegistry=current_registry()) -> Vector{TensorExpr}

Filter a list of fully-contracted expressions to keep only linearly independent ones.
Two contractions are equivalent if they simplify to the same canonical form
(possibly differing by a scalar factor).

Uses `simplify()` to canonicalize and `_canonical_structure` to extract the structural
part. Returns one representative from each equivalence class.

# Examples
```julia
# Riemann R_{abcd} has 3 pairings but only 2 independent contractions
raw = all_contractions(R, :g; filter=false)   # 3 contractions
filtered = filter_independent_contractions(raw)  # <= 3
```
"""
function filter_independent_contractions(contractions::Vector{TensorExpr};
                                          registry::TensorRegistry=current_registry())
    isempty(contractions) && return contractions

    # Simplify each contraction and extract canonical structure
    seen = Dict{TensorExpr, Int}()  # normalized structure => index of representative
    result = TensorExpr[]

    for c in contractions
        simplified = simplify(c; registry=registry)

        # Skip zeros
        simplified == TScalar(0 // 1) && continue

        structure, _ = _canonical_structure(simplified)

        # Check both the structure and its negation (sign equivalence)
        neg_structure, _ = _canonical_structure(_negate_expr(simplified))

        if !haskey(seen, structure) && !haskey(seen, neg_structure)
            seen[structure] = length(result) + 1
            push!(result, c)
        end
    end

    result
end

"""
Generate a fresh index name for ansatz construction, using the `_c` prefix
to avoid clashing with user-level index names.
"""
function _fresh_ansatz_index(used::Set{Symbol})
    for n in 1:1000
        s = Symbol(:_c, n)
        s in used || return s
    end
    error("Could not generate fresh ansatz index (exhausted 1000 names)")
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
