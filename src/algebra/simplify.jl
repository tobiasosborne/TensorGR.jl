#= Expression simplification: expand products over sums, collect like terms.

These are the workhorses that turn a raw expression tree into something
where canonicalization and contraction can do their jobs.
=#

"""
    expand_products(expr::TensorExpr) -> TensorExpr

Distribute products over sums: A*(B+C) → A*B + A*C.
Recursively expands until no TSum appears inside any TProduct.
"""
expand_products(t::Tensor) = t
expand_products(s::TScalar) = s
expand_products(d::TDeriv) = TDeriv(d.index, expand_products(d.arg), d.covd)

function expand_products(s::TSum)
    tsum(TensorExpr[expand_products(t) for t in s.terms])
end

function expand_products(p::TProduct)
    # First expand children
    expanded = TensorExpr[expand_products(f) for f in p.factors]

    # Find a sum factor to distribute over
    sum_idx = findfirst(f -> f isa TSum, expanded)
    sum_idx === nothing && return TProduct(p.scalar, expanded)

    s = expanded[sum_idx]::TSum
    others = TensorExpr[expanded[i] for i in eachindex(expanded) if i != sum_idx]
    terms = TensorExpr[]
    for term in s.terms
        new_p = tproduct(p.scalar, vcat(others, [term]))
        push!(terms, expand_products(new_p))  # recurse: there may be more sums
    end
    tsum(terms)
end

"""
    collect_terms(expr::TSum) -> TensorExpr

Combine terms that differ only by scalar coefficient.
`3 R_{ab} + (-2) R_{ab}` → `R_{ab}`.

Before comparing, each term's dummy indices are renamed to a canonical
alphabet so that `T_{ab} X^{ab}` and `T_{cd} X^{cd}` are recognized as equal.
"""
function collect_terms(expr::TSum)
    buckets = Dict{TensorExpr, Rational{Int}}()

    for term in expr.terms
        scalar, core = _split_scalar(term)
        # Canonicalize structure, then rename dummies for comparison
        canonical_core = _normalize_dummies(canonicalize(core))
        buckets[canonical_core] = get(buckets, canonical_core, 0 // 1) + scalar
    end

    terms = TensorExpr[]
    for (core, coeff) in buckets
        coeff == 0 && continue
        push!(terms, tproduct(coeff, TensorExpr[core]))
    end

    tsum(terms)
end

collect_terms(expr::TensorExpr) = expr

"""
    _normalize_dummies(expr) -> TensorExpr

Rename all dummy indices in `expr` to a canonical alphabet (`_d1`, `_d2`, ...)
so that terms differing only in dummy names compare as equal.
"""
function _normalize_dummies(expr::TensorExpr)
    pairs = dummy_pairs(expr)
    isempty(pairs) && return expr

    # Sort dummy pairs by first occurrence to get deterministic ordering
    all_idxs = indices(expr)
    first_occurrence = Dict{Symbol, Int}()
    for (i, idx) in enumerate(all_idxs)
        haskey(first_occurrence, idx.name) || (first_occurrence[idx.name] = i)
    end

    dummy_names = [p[1].name for p in pairs]
    sort!(dummy_names, by = n -> get(first_occurrence, n, 0))

    # Rename to canonical names
    result = expr
    used = Set(idx.name for idx in free_indices(expr))
    for (i, old_name) in enumerate(dummy_names)
        new_name = Symbol("_d", i)
        if old_name != new_name && new_name ∉ used
            result = rename_dummy(result, old_name, new_name)
        end
    end
    result
end

"""Split a TensorExpr into (scalar coefficient, tensor part)."""
function _split_scalar(expr::TProduct)
    if length(expr.factors) == 1
        return expr.scalar, expr.factors[1]
    end
    return expr.scalar, TProduct(1 // 1, expr.factors)
end
_split_scalar(expr::Tensor) = (1 // 1, expr)
_split_scalar(expr::TScalar) = (expr.val isa Rational ? expr.val : 1 // 1, TScalar(1))
_split_scalar(expr::TDeriv) = (1 // 1, expr)
_split_scalar(expr::TSum) = (1 // 1, expr)

"""
    simplify(expr; registry=current_registry(), maxiter=20) -> TensorExpr

Full simplification pipeline applied to fixed point:
1. expand_products — distribute * over +
2. contract_metrics — eliminate g and δ contractions
3. contract_curvature — detect Riemann/Ricci traces
4. canonicalize — canonical index ordering via xperm
5. commute_covds (optional) — sort covariant derivatives, insert Riemann terms
6. collect_terms — combine like terms
7. apply_rules — registered rewrite rules

Each step is applied, then the whole pipeline repeats until the expression
stabilizes or `maxiter` is reached.

Pass `commute_covds_name=:∇` to include covariant derivative commutation
in the simplify loop (off by default).
"""
function simplify(expr::TensorExpr;
                  registry::TensorRegistry=current_registry(),
                  maxiter::Int=20,
                  commute_covds_name::Union{Symbol,Nothing}=nothing)
    with_registry(registry) do
        _simplify_fixpoint(expr, registry, maxiter, commute_covds_name)
    end
end

function _simplify_fixpoint(expr::TensorExpr, reg::TensorRegistry, maxiter::Int,
                            covd_name::Union{Symbol,Nothing}=nothing)
    current = expr
    for _ in 1:maxiter
        next = _simplify_one_pass(current, reg, covd_name)
        next == current && return current
        current = next
    end
    current
end

function _simplify_one_pass(expr::TensorExpr, reg::TensorRegistry,
                            covd_name::Union{Symbol,Nothing}=nothing)
    result = expand_products(expr)
    result = contract_metrics(result)
    result = contract_curvature(result)
    result = canonicalize(result)

    if covd_name !== nothing
        result = commute_covds(result, covd_name; registry=reg)
        result = contract_curvature(result)
        result = canonicalize(result)
    end

    result = collect_terms(result)

    rules = get_rules(reg)
    if !isempty(rules)
        typed_rules = RewriteRule[r for r in rules]
        result = apply_rules(result, typed_rules)
    end

    result
end
