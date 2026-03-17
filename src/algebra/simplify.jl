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
    sum_idx === nothing && return tproduct(p.scalar, expanded)

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
    distribute_derivs_over_sums(expr::TensorExpr) -> TensorExpr

Distribute derivatives over sums (linearity): ∂(A+B) → ∂A + ∂B.
Also pull scalars out: ∂(c*T) → c*∂(T).

This matches xAct's eager derivative linearity (`covdL[expr_Plus] := Map[covdR, expr]`).
"""
distribute_derivs_over_sums(t::Tensor) = t
distribute_derivs_over_sums(s::TScalar) = s

function distribute_derivs_over_sums(s::TSum)
    tsum(TensorExpr[distribute_derivs_over_sums(t) for t in s.terms])
end

function distribute_derivs_over_sums(p::TProduct)
    TProduct(p.scalar, TensorExpr[distribute_derivs_over_sums(f) for f in p.factors])
end

function distribute_derivs_over_sums(d::TDeriv)
    inner = distribute_derivs_over_sums(d.arg)
    # Expand products inside the derivative to expose nested sums.
    # E.g., ∂(g*(A+B)) → ∂(g*A + g*B) → ∂(g*A) + ∂(g*B)
    inner = expand_products(inner)
    if inner isa TSum
        # ∂(A + B) → ∂A + ∂B
        tsum(TensorExpr[distribute_derivs_over_sums(TDeriv(d.index, t, d.covd)) for t in inner.terms])
    elseif inner isa TProduct && inner.scalar != 1 // 1
        # ∂(c*T) → c*∂(T) (linearity), then recurse to distribute over
        # any TSum that was hidden under the scalar
        core = tproduct(1 // 1, inner.factors)
        distributed = distribute_derivs_over_sums(TDeriv(d.index, core, d.covd))
        if distributed isa TSum
            tsum(TensorExpr[tproduct(inner.scalar, TensorExpr[t]) for t in distributed.terms])
        else
            tproduct(inner.scalar, TensorExpr[distributed])
        end
    else
        TDeriv(d.index, inner, d.covd)
    end
end

"""
    flatten_metric_derivs(expr, metric::Symbol) -> TensorExpr

Apply the Leibniz rule to derivatives of products containing the metric,
then drop the ∂(metric) terms (metric compatibility on flat background).

This converts `∂_a(g^{cd} * X)` → `g^{cd} * ∂_a(X)` by:
1. Leibniz: `∂(g*X) → ∂g*X + g*∂X`
2. Metric compatibility: `∂g = 0` → drop the first term
3. Result: `g*∂X`

Use as a post-processing step before kernel extraction when the background
is flat (all curvature vanishing). NOT for the general simplify pipeline.
"""
flatten_metric_derivs(t::Tensor, metric::Symbol) = t
flatten_metric_derivs(s::TScalar, metric::Symbol) = s

function flatten_metric_derivs(s::TSum, metric::Symbol)
    tsum(TensorExpr[flatten_metric_derivs(t, metric) for t in s.terms])
end

function flatten_metric_derivs(p::TProduct, metric::Symbol)
    TProduct(p.scalar, TensorExpr[flatten_metric_derivs(f, metric) for f in p.factors])
end

function flatten_metric_derivs(d::TDeriv, metric::Symbol)
    inner = flatten_metric_derivs(d.arg, metric)
    if inner isa TSum
        # Distribute derivative over sum first: ∂(A+B) → ∂A + ∂B
        # Then recurse to flatten each term
        terms = TensorExpr[flatten_metric_derivs(TDeriv(d.index, t, d.covd), metric)
                           for t in inner.terms]
        return tsum(terms)
    elseif inner isa TProduct
        if inner.scalar != 1 // 1
            # Pull scalar out: ∂(c*X) → c*∂(X), then recurse
            core = tproduct(1 // 1, inner.factors)
            flat = flatten_metric_derivs(TDeriv(d.index, core, d.covd), metric)
            return flat isa TSum ?
                tsum(TensorExpr[tproduct(inner.scalar, TensorExpr[t]) for t in flat.terms]) :
                tproduct(inner.scalar, TensorExpr[flat])
        end
        # Check if any factor is the metric tensor
        metric_idx = findfirst(f -> f isa Tensor && f.name == metric, inner.factors)
        if metric_idx !== nothing
            # Leibniz + ∂g=0: ∂(g*X) → g*∂X (drop ∂g*X term)
            metric_tensor = inner.factors[metric_idx]
            other_factors = TensorExpr[inner.factors[i] for i in eachindex(inner.factors) if i != metric_idx]
            other = tproduct(inner.scalar, other_factors)
            new_deriv = flatten_metric_derivs(TDeriv(d.index, other, d.covd), metric)
            return tproduct(1 // 1, TensorExpr[metric_tensor, new_deriv])
        end
    end
    TDeriv(d.index, inner, d.covd)
end

"""
    collect_inner_sums(expr::TensorExpr) -> TensorExpr

Recursively simplify TSum nodes inside TDeriv arguments.
The top-level `collect_terms` only merges terms at the outermost TSum;
this function reaches inner sums that are trapped inside derivative wrappers.
"""
collect_inner_sums(t::Tensor) = t
collect_inner_sums(s::TScalar) = s

function collect_inner_sums(s::TSum)
    tsum(TensorExpr[collect_inner_sums(t) for t in s.terms])
end

function collect_inner_sums(p::TProduct)
    TProduct(p.scalar, TensorExpr[collect_inner_sums(f) for f in p.factors])
end

function collect_inner_sums(d::TDeriv)
    inner = collect_inner_sums(d.arg)
    # Simplify any TSum inside the derivative argument.
    # IMPORTANT: do NOT use collect_terms here — it renormalizes dummy
    # names without knowledge of the outer context, causing clashes.
    # Instead, merge only structurally identical terms (same core, no renaming).
    if inner isa TSum
        inner = _merge_identical_terms(inner)
    elseif inner isa TProduct
        new_factors = TensorExpr[]
        for f in inner.factors
            if f isa TSum
                push!(new_factors, _merge_identical_terms(f))
            else
                push!(new_factors, f)
            end
        end
        inner = TProduct(inner.scalar, new_factors)
    end
    TDeriv(d.index, inner, d.covd)
end

"""Merge terms with structurally identical cores (no dummy renaming).
Used for inner sums where dummy normalization would clash with outer context."""
function _merge_identical_terms(s::TSum)
    buckets = Dict{TensorExpr, Rational{Int}}()
    for term in s.terms
        scalar, core = _split_scalar(term)
        buckets[core] = get(buckets, core, 0 // 1) + scalar
    end
    terms = TensorExpr[]
    for (core, coeff) in buckets
        coeff == 0 && continue
        push!(terms, tproduct(coeff, TensorExpr[core]))
    end
    sort!(terms, by = t -> hash(t))
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

    # Sort terms by hash for deterministic ordering across Dict iterations.
    # This prevents spurious simplify iterations caused by Dict randomization.
    sort!(terms, by = t -> hash(t))
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

    # Two-phase renaming to avoid collision when canonical names (_d1, _d2, ...)
    # overlap with existing dummy names. Phase 1: old → temp. Phase 2: temp → canonical.
    phase1 = Dict{Symbol,Symbol}()
    for (i, old_name) in enumerate(dummy_names)
        tmp_name = Symbol("__ndtmp", i)
        old_name != tmp_name && (phase1[old_name] = tmp_name)
    end
    result = isempty(phase1) ? normalized : rename_dummies(normalized, phase1)

    phase2 = Dict{Symbol,Symbol}()
    for (i, _) in enumerate(dummy_names)
        tmp_name = Symbol("__ndtmp", i)
        new_name = Symbol("_d", i)
        tmp_name != new_name && (phase2[tmp_name] = new_name)
    end
    isempty(phase2) ? result : rename_dummies(result, phase2)
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
    sort!(terms, by = t -> hash(t))
    tsum(terms)
end

# ── Scalar simplification ───────────────────────────────────────────────────

"""
    _simplify_scalars(expr) -> TensorExpr

Consolidate non-Rational TScalar factors in TProduct terms and simplify via
CAS hooks. Multiple TScalar values are combined using `_sym_mul` and passed
through `_simplify_scalar_val` (which dispatches to Symbolics.simplify when
that extension is loaded).
"""
function _simplify_scalars(expr::TProduct)
    scalar_indices = Int[]
    for (i, f) in enumerate(expr.factors)
        if f isa TScalar && !(f.val isa Rational)
            push!(scalar_indices, i)
        end
    end
    isempty(scalar_indices) && return expr

    vals = [expr.factors[i].val for i in scalar_indices]
    combined = length(vals) == 1 ? vals[1] : reduce(_sym_mul, vals)
    simplified = _simplify_scalar_val(combined)

    # Fast path: single scalar, no change
    if length(scalar_indices) == 1 && simplified === combined
        return expr
    end

    # Rebuild factors: replace all non-Rational TScalars with single simplified
    new_factors = TensorExpr[]
    first_done = false
    for (i, f) in enumerate(expr.factors)
        if i ∈ scalar_indices
            if !first_done
                push!(new_factors, TScalar(simplified))
                first_done = true
            end
        else
            push!(new_factors, f)
        end
    end
    tproduct(expr.scalar, new_factors)
end

function _simplify_scalars(s::TSum)
    tsum(TensorExpr[_simplify_scalars(t) for t in s.terms])
end

function _simplify_scalars(d::TDeriv)
    TDeriv(d.index, _simplify_scalars(d.arg), d.covd)
end

_simplify_scalars(t::Tensor) = t
_simplify_scalars(s::TScalar) = s

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
        result = _pmap_over_tsum(_simplify_scalars, result)
    else
        result = expand_products(expr)
        result = contract_metrics(result)
        result = contract_curvature(result)
        result = canonicalize(result)
        result = _simplify_scalars(result)
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

    # Simplify inner sums trapped inside TDeriv arguments, then collect top-level
    result = collect_inner_sums(result)
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
