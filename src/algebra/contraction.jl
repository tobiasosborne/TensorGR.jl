#= Metric contraction engine.

The contraction engine eliminates metric tensors and Kronecker deltas from
products by raising/lowering indices on partner tensors. It runs to fixed
point: each contraction may expose new contractions.

Design: the engine walks through factors looking for metrics (g) and deltas (δ)
that share a dummy index with another factor. When found, it removes the
metric/delta and transfers the index.
=#

"""
    contract_metrics(expr::TensorExpr) -> TensorExpr

Eliminate all metric and delta contractions, raising/lowering indices.
Distributes over sums, recurses into derivatives, runs to fixed point on products.
"""
function contract_metrics(t::Tensor)
    reg = current_registry()
    if has_tensor(reg, t.name)
        props = get_tensor(reg, t.name)
        if props.is_delta && length(t.indices) == 2
            # Self-traced delta: δ^a_a → dim
            if t.indices[1].name == t.indices[2].name &&
               t.indices[1].position != t.indices[2].position
                dim = _tensor_dim(reg, props)
                return TScalar(dim // 1)
            end
            # Same-position delta → metric: δ_{ab} = g_{ab}
            if t.indices[1].position == t.indices[2].position
                vb = t.indices[1].vbundle
                metric_name = _find_metric_for_vbundle(reg, vb, props.manifold)
                if metric_name !== nothing
                    return Tensor(metric_name, copy(t.indices))
                end
            end
        end
        if props.is_metric && length(t.indices) == 2
            if t.indices[1].name == t.indices[2].name &&
               t.indices[1].position != t.indices[2].position &&
               t.indices[1].vbundle == t.indices[2].vbundle
                dim = _tensor_dim(reg, props)
                return TScalar(dim // 1)
            end
        end
    end
    t
end
contract_metrics(s::TScalar) = s

function contract_metrics(s::TSum)
    tsum(TensorExpr[contract_metrics(t) for t in s.terms])
end

function contract_metrics(d::TDeriv)
    TDeriv(d.index, contract_metrics(d.arg), d.covd)
end

function contract_metrics(p::TProduct)
    # First, distribute over any sum factors
    sum_idx = findfirst(f -> f isa TSum, p.factors)
    if sum_idx !== nothing
        return _distribute_and_contract(p, sum_idx)
    end

    # Fixed-point contraction loop
    result = p
    while true
        next = _contract_one(result)
        next == result && break
        # If result collapsed to non-product, done
        next isa TProduct || return contract_metrics(next)
        result = next
    end
    result
end

function _distribute_and_contract(p::TProduct, sum_idx::Int)
    s = p.factors[sum_idx]::TSum
    other = TensorExpr[p.factors[i] for i in eachindex(p.factors) if i != sum_idx]
    terms = TensorExpr[]
    for term in s.terms
        new_factors = copy(other)
        push!(new_factors, term)
        push!(terms, contract_metrics(tproduct(p.scalar, new_factors)))
    end
    tsum(terms)
end

"""
Try to contract one metric or delta from the product. Returns the product
unchanged if no contraction is possible.
"""
function _contract_one(p::TProduct)
    reg = current_registry()
    factors = p.factors

    for (i, fi) in enumerate(factors)
        fi isa Tensor || continue
        has_tensor(reg, fi.name) || continue
        props = get_tensor(reg, fi.name)

        if props.is_metric && !props.frozen
            result = _try_metric_contraction(p, i, fi, reg)
            result !== nothing && return result

        elseif props.is_delta
            # Same-position delta → metric: δ_{ab} = g_{ab}, δ^{ab} = g^{ab}
            if length(fi.indices) == 2 && fi.indices[1].position == fi.indices[2].position
                vb = fi.indices[1].vbundle
                metric_name = _find_metric_for_vbundle(reg, vb, props.manifold)
                if metric_name !== nothing
                    new_tensor = Tensor(metric_name, copy(fi.indices))
                    new_factors = TensorExpr[k == i ? new_tensor : fk for (k, fk) in enumerate(factors)]
                    return tproduct(p.scalar, new_factors)
                end
            end
            result = _try_delta_contraction(p, i, fi, reg)
            result !== nothing && return result
        end
    end

    p  # no contraction found
end

function _try_metric_contraction(p::TProduct, metric_idx::Int, metric::Tensor, reg)
    # A metric g has two indices, both up or both down.
    # It contracts with another factor sharing a dummy index.
    midxs = metric.indices
    length(midxs) == 2 || return nothing

    for (j, fj) in enumerate(p.factors)
        j == metric_idx && continue
        fj isa Tensor || continue

        for (mi, midx) in enumerate(midxs)
            for (ti, tidx) in enumerate(fj.indices)
                if midx.name == tidx.name && midx.position != tidx.position && midx.vbundle == tidx.vbundle
                    # Found contraction: metric index midx pairs with fj index tidx
                    other_midx = midxs[3 - mi]

                    # Special case: metric × metric → delta
                    fj_props = has_tensor(reg, fj.name) ? get_tensor(reg, fj.name) : nothing
                    is_partner_metric = fj_props !== nothing && fj_props.is_metric

                    if is_partner_metric
                        # g^{ab} g_{bc} → δ^a_c
                        other_fidx = fj.indices[3 - ti]
                        mprops = get_tensor(reg, metric.name)
                        vb = midx.vbundle
                        delta_name = _find_delta_for_vbundle(reg, vb, mprops.manifold)
                        new_tensor = Tensor(delta_name,
                            [other_midx, TIndex(other_fidx.name, other_fidx.position, other_fidx.vbundle)])
                    else
                        # g^{ab} T_{bc} → T^a_c  (raise/lower index)
                        new_indices = copy(fj.indices)
                        new_indices[ti] = TIndex(other_midx.name, other_midx.position, other_midx.vbundle)
                        new_tensor = Tensor(fj.name, new_indices)
                    end

                    new_factors = TensorExpr[]
                    for (k, fk) in enumerate(p.factors)
                        if k == metric_idx
                            continue
                        elseif k == j
                            push!(new_factors, new_tensor)
                        else
                            push!(new_factors, fk)
                        end
                    end
                    return tproduct(p.scalar, new_factors)
                end
            end
        end
    end

    # Self-trace in product: g^a_a * ... → dim * ...
    if midxs[1].name == midxs[2].name &&
       midxs[1].position != midxs[2].position &&
       midxs[1].vbundle == midxs[2].vbundle
        mprops = get_tensor(reg, metric.name)
        dim = _tensor_dim(reg, mprops)
        new_factors = TensorExpr[f for (k, f) in enumerate(p.factors) if k != metric_idx]
        return tproduct(p.scalar * dim, new_factors)
    end

    nothing
end

"""Find the delta tensor name for a manifold (or vbundle), or :δ by default."""
function _find_delta(reg::TensorRegistry, manifold::Symbol)
    get(reg.delta_cache, manifold, :δ)
end

"""Find the delta for a specific vbundle, falling back to manifold lookup."""
function _find_delta_for_vbundle(reg::TensorRegistry, vbundle::Symbol, manifold::Symbol)
    get(reg.delta_cache, vbundle, _find_delta(reg, manifold))
end

"""Find the metric tensor name for a manifold (or vbundle), or nothing."""
function _find_metric(reg::TensorRegistry, manifold::Symbol)
    get(reg.metric_cache, manifold, nothing)
end

"""Find the metric for a specific vbundle, falling back to manifold lookup."""
function _find_metric_for_vbundle(reg::TensorRegistry, vbundle::Symbol, manifold::Symbol)
    get(reg.metric_cache, vbundle, _find_metric(reg, manifold))
end

"""
Effective dimension for a metric/delta tensor. Uses `:vbundle_dim` option
if present (for spinor metrics on 2-dim bundles), otherwise falls back to
the manifold dimension.
"""
function _tensor_dim(reg::TensorRegistry, props::TensorProperties)
    vdim = get(props.options, :vbundle_dim, nothing)
    vdim !== nothing && return Int(vdim)
    mp = get_manifold(reg, props.manifold)
    mp.dim
end

function _try_delta_contraction(p::TProduct, delta_idx::Int, delta::Tensor, reg)
    # Delta δ^a_b: has one up, one down index.
    # Contracts by replacing the dummy name in the partner with the free name from delta.
    didxs = delta.indices
    length(didxs) == 2 || return nothing

    for (j, fj) in enumerate(p.factors)
        j == delta_idx && continue
        fj isa Tensor || continue

        for (di, didx) in enumerate(didxs)
            for (ti, tidx) in enumerate(fj.indices)
                if didx.name == tidx.name && didx.position != tidx.position && didx.vbundle == tidx.vbundle
                    # The other delta index replaces the contracted index
                    other_didx = didxs[3 - di]
                    new_indices = copy(fj.indices)
                    new_indices[ti] = TIndex(other_didx.name, tidx.position, tidx.vbundle)
                    new_tensor = Tensor(fj.name, new_indices)

                    new_factors = TensorExpr[]
                    for (k, fk) in enumerate(p.factors)
                        if k == delta_idx
                            continue
                        elseif k == j
                            push!(new_factors, new_tensor)
                        else
                            push!(new_factors, fk)
                        end
                    end
                    return tproduct(p.scalar, new_factors)
                end
            end
        end
    end

    # Self-trace: δ^a_a → dimension
    if didxs[1].name == didxs[2].name && didxs[1].position != didxs[2].position && didxs[1].vbundle == didxs[2].vbundle
        dprops = get_tensor(reg, delta.name)
        dim = _tensor_dim(reg, dprops)
        new_factors = TensorExpr[f for (k, f) in enumerate(p.factors) if k != delta_idx]
        return tproduct(p.scalar * dim, new_factors)
    end

    nothing
end

# ── Trace-free enforcement ───────────────────────────────────────────────────

"""
    enforce_tracefree(expr; registry=current_registry()) -> TensorExpr

Replace any tensor with a self-contracted trace-free pair by zero.
A tensor T registered with `tracefree_pairs` containing `(i,j)` vanishes
when indices at slots `i` and `j` are contracted (same name, opposite position).
"""
function enforce_tracefree(expr::TensorExpr; registry::TensorRegistry=current_registry())
    _enforce_tracefree(expr, registry)
end

function _enforce_tracefree(t::Tensor, reg::TensorRegistry)
    has_tensor(reg, t.name) || return t
    props = get_tensor(reg, t.name)
    isempty(props.tracefree_pairs) && return t
    idxs = t.indices
    for (i, j) in props.tracefree_pairs
        (i <= length(idxs) && j <= length(idxs)) || continue
        if idxs[i].name == idxs[j].name &&
           idxs[i].position != idxs[j].position &&
           idxs[i].vbundle == idxs[j].vbundle
            return TScalar(0 // 1)
        end
    end
    t
end

_enforce_tracefree(s::TScalar, ::TensorRegistry) = s

function _enforce_tracefree(d::TDeriv, reg::TensorRegistry)
    inner = _enforce_tracefree(d.arg, reg)
    inner isa TScalar && inner.val == 0 && return TScalar(0 // 1)
    TDeriv(d.index, inner, d.covd)
end

function _enforce_tracefree(s::TSum, reg::TensorRegistry)
    tsum(TensorExpr[_enforce_tracefree(t, reg) for t in s.terms])
end

function _enforce_tracefree(p::TProduct, reg::TensorRegistry)
    new_factors = TensorExpr[]
    for f in p.factors
        result = _enforce_tracefree(f, reg)
        result isa TScalar && result.val == 0 && return TScalar(0 // 1)
        push!(new_factors, result)
    end
    tproduct(p.scalar, new_factors)
end

# ── Divergence-free enforcement ──────────────────────────────────────────────

"""
    enforce_divfree(expr; registry=current_registry()) -> TensorExpr

Replace any divergence of a divergence-free tensor by zero.
A tensor T registered with `divfree_indices` containing `(covd, slot)` vanishes
when derivative index contracts with the tensor index at `slot` (same name,
opposite position) and the derivative uses the matching `covd`.
"""
function enforce_divfree(expr::TensorExpr; registry::TensorRegistry=current_registry())
    _enforce_divfree(expr, registry)
end

function _enforce_divfree(d::TDeriv, reg::TensorRegistry)
    inner = _enforce_divfree(d.arg, reg)
    inner isa TScalar && inner.val == 0 && return TScalar(0 // 1)
    # Check: is inner a tensor with divfree_indices matching this derivative?
    if inner isa Tensor
        has_tensor(reg, inner.name) || return TDeriv(d.index, inner, d.covd)
        props = get_tensor(reg, inner.name)
        isempty(props.divfree_indices) && return TDeriv(d.index, inner, d.covd)
        for (covd_name, slot) in props.divfree_indices
            d.covd == covd_name || continue
            slot <= length(inner.indices) || continue
            tidx = inner.indices[slot]
            if d.index.name == tidx.name &&
               d.index.position != tidx.position &&
               d.index.vbundle == tidx.vbundle
                return TScalar(0 // 1)
            end
        end
    end
    TDeriv(d.index, inner, d.covd)
end

_enforce_divfree(t::Tensor, ::TensorRegistry) = t
_enforce_divfree(s::TScalar, ::TensorRegistry) = s

function _enforce_divfree(s::TSum, reg::TensorRegistry)
    tsum(TensorExpr[_enforce_divfree(t, reg) for t in s.terms])
end

function _enforce_divfree(p::TProduct, reg::TensorRegistry)
    new_factors = TensorExpr[]
    for f in p.factors
        result = _enforce_divfree(f, reg)
        result isa TScalar && result.val == 0 && return TScalar(0 // 1)
        push!(new_factors, result)
    end
    tproduct(p.scalar, new_factors)
end
