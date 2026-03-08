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
    _collect_terms_impl(expr, true)
end

collect_terms(expr::TensorExpr) = expr

"""Skip re-canonicalization when terms are already canonical (called from simplify pipeline)."""
function _collect_terms_impl(expr::TSum, do_canonicalize::Bool)
    buckets = Dict{TensorExpr, Rational{Int}}()

    for term in expr.terms
        scalar, core = _split_scalar(term)
        normalized = do_canonicalize ?
            _normalize_dummies(canonicalize(core)) :
            _normalize_dummies(core)
        buckets[normalized] = get(buckets, normalized, 0 // 1) + scalar
    end

    terms = TensorExpr[]
    for (core, coeff) in buckets
        coeff == 0 && continue
        push!(terms, tproduct(coeff, TensorExpr[core]))
    end

    tsum(terms)
end

"""
    _normalize_dummies(expr) -> TensorExpr

Rename all dummy indices in `expr` to a canonical alphabet (`_d1`, `_d2`, ...)
so that terms differing only in dummy names compare as equal.
"""
function _normalize_dummies(expr::TensorExpr)
    # Sort commuting partial derivative chains so that
    # ∂_b(∂_a(T)) and ∂_a(∂_b(T)) get the same normalized form.
    normalized = _sort_partial_chains(expr)

    # Single-pass index analysis (replaces 3 separate tree walks)
    all_idxs, free, pairs = _analyze_indices(normalized)
    isempty(pairs) && return normalized

    # Sort dummy pairs by first occurrence to get deterministic ordering
    first_occurrence = Dict{Symbol, Int}()
    for (i, idx) in enumerate(all_idxs)
        haskey(first_occurrence, idx.name) || (first_occurrence[idx.name] = i)
    end

    dummy_names = [p[1].name for p in pairs]
    sort!(dummy_names, by = n -> get(first_occurrence, n, 0))

    # Rename to canonical names
    result = normalized
    used = Set(idx.name for idx in free)
    for (i, old_name) in enumerate(dummy_names)
        new_name = Symbol("_d", i)
        if old_name != new_name && new_name ∉ used
            result = rename_dummy(result, old_name, new_name)
        end
    end
    result
end

"""Sort partial derivative chains for normalization (partials commute)."""
function _sort_partial_chains(expr::TDeriv)
    inner = _sort_partial_chains(expr.arg)
    d = TDeriv(expr.index, inner, expr.covd)
    d.covd == :partial || return d
    d.arg isa TDeriv || return d
    d.arg.covd == :partial || return d

    # Collect chain of commuting partial derivatives
    chain_idxs = TIndex[]
    current = d
    while current isa TDeriv && current.covd == :partial
        push!(chain_idxs, current.index)
        current = current.arg
    end
    length(chain_idxs) < 2 && return d

    sorted_idxs = sort(chain_idxs, by = idx -> idx.name)
    sorted_idxs == chain_idxs && return d

    result = current
    for i in length(sorted_idxs):-1:1
        result = TDeriv(sorted_idxs[i], result, :partial)
    end
    result
end

function _sort_partial_chains(expr::TProduct)
    TProduct(expr.scalar, TensorExpr[_sort_partial_chains(f) for f in expr.factors])
end
function _sort_partial_chains(expr::TSum)
    TSum(TensorExpr[_sort_partial_chains(t) for t in expr.terms])
end
_sort_partial_chains(expr::Tensor) = expr
_sort_partial_chains(expr::TScalar) = expr

"""Split a TensorExpr into (scalar coefficient, tensor part)."""
function _split_scalar(expr::TProduct)
    if length(expr.factors) == 1
        return expr.scalar, expr.factors[1]
    end
    return expr.scalar, TProduct(1 // 1, expr.factors)
end
_split_scalar(expr::Tensor) = (1 // 1, expr)
_split_scalar(expr::TScalar) = expr.val isa Rational ? (expr.val, TScalar(1)) : (1 // 1, expr)
_split_scalar(expr::TDeriv) = (1 // 1, expr)
_split_scalar(expr::TSum) = (1 // 1, expr)

# ── Parallel helpers ─────────────────────────────────────────────────────────

const PARALLEL_THRESHOLD = 20

"""
    _pmap_over_tsum(f, expr::TSum)

Apply `f` to each term of a TSum in parallel (via `Threads.@spawn`), reassembling
with `tsum`. Falls back to serial `f(expr)` when the term count is below
`PARALLEL_THRESHOLD` or only one thread is available.
"""
function _pmap_over_tsum(f, expr::TSum)
    n = length(expr.terms)
    if n < PARALLEL_THRESHOLD || Threads.nthreads() == 1
        return f(expr)
    end
    reg = current_registry()
    results = Vector{TensorExpr}(undef, n)
    @sync for i in 1:n
        let i=i
            Threads.@spawn begin
                with_registry(reg) do
                    results[i] = f(expr.terms[i])
                end
            end
        end
    end
    tsum(results)
end
_pmap_over_tsum(f, expr::TensorExpr) = f(expr)

"""
    _collect_terms_parallel(expr::TSum)

Two-phase parallel collect_terms: parallel canonicalize + serial Dict merge.
"""
function _collect_terms_parallel(expr::TSum)
    n = length(expr.terms)
    reg = current_registry()
    # Phase 1: parallel canonicalize
    pairs = Vector{Tuple{Rational{Int}, TensorExpr}}(undef, n)
    @sync for i in 1:n
        let i=i
            Threads.@spawn begin
                with_registry(reg) do
                    scalar, core = _split_scalar(expr.terms[i])
                    pairs[i] = (scalar, _normalize_dummies(canonicalize(core)))
                end
            end
        end
    end
    # Phase 2: serial Dict merge
    buckets = Dict{TensorExpr, Rational{Int}}()
    for (scalar, core) in pairs
        buckets[core] = get(buckets, core, 0 // 1) + scalar
    end
    terms = TensorExpr[tproduct(c, TensorExpr[k]) for (k, c) in buckets if c != 0]
    tsum(terms)
end

# ── Main simplify pipeline ──────────────────────────────────────────────────

"""
    simplify(expr; registry=current_registry(), maxiter=20, parallel=false) -> TensorExpr

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

Pass `parallel=true` to parallelize TSum operations across threads.
"""
function simplify(expr::TensorExpr;
                  registry::TensorRegistry=current_registry(),
                  maxiter::Int=20,
                  commute_covds_name::Union{Symbol,Nothing}=nothing,
                  parallel::Bool=false)
    with_registry(registry) do
        _simplify_fixpoint(expr, registry, maxiter, commute_covds_name, parallel)
    end
end

function _simplify_fixpoint(expr::TensorExpr, reg::TensorRegistry, maxiter::Int,
                            covd_name::Union{Symbol,Nothing}=nothing,
                            parallel::Bool=false)
    current = expr
    h_current = hash(current)
    for _ in 1:maxiter
        next = _simplify_one_pass(current, reg, covd_name, parallel)
        h_next = hash(next)
        # Short-circuit: if hashes differ, expressions differ (skip deep ==)
        if h_next == h_current && next == current
            return current
        end
        current = next
        h_current = h_next
    end
    @warn "simplify did not converge after $maxiter iterations"
    current
end

function _simplify_one_pass(expr::TensorExpr, reg::TensorRegistry,
                            covd_name::Union{Symbol,Nothing}=nothing,
                            parallel::Bool=false)
    if parallel
        result = _pmap_over_tsum(expand_products, expr)
        result = _pmap_over_tsum(contract_metrics, result)
        result = _pmap_over_tsum(contract_curvature, result)
        result = _pmap_over_tsum(canonicalize, result)
    else
        result = expand_products(expr)
        result = contract_metrics(result)
        result = contract_curvature(result)
        result = canonicalize(result)
    end

    if covd_name !== nothing
        result = commute_covds(result, covd_name; registry=reg)
        if parallel
            result = _pmap_over_tsum(contract_curvature, result)
            result = _pmap_over_tsum(canonicalize, result)
        else
            result = contract_curvature(result)
            result = canonicalize(result)
        end
    end

    if parallel && result isa TSum && length(result.terms) >= PARALLEL_THRESHOLD
        result = _collect_terms_parallel(result)
    else
        result = collect_terms(result)
    end

    rules = get_rules(reg)
    if !isempty(rules)
        typed_rules = RewriteRule[r for r in rules]
        result = apply_rules(result, typed_rules)
    end

    result
end
