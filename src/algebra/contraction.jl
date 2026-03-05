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
    # Check for self-traced delta: δ^a_a → dim
    reg = current_registry()
    if has_tensor(reg, t.name)
        props = get_tensor(reg, t.name)
        if get(props.options, :is_delta, false) && length(t.indices) == 2
            if t.indices[1].name == t.indices[2].name &&
               t.indices[1].position != t.indices[2].position
                mp = get_manifold(reg, props.manifold)
                return TScalar(mp.dim // 1)
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
    TDeriv(d.index, contract_metrics(d.arg))
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

        if get(props.options, :is_metric, false) && !get(props.options, :frozen, false)
            result = _try_metric_contraction(p, i, fi, reg)
            result !== nothing && return result

        elseif get(props.options, :is_delta, false)
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
                if midx.name == tidx.name && midx.position != tidx.position
                    # Found contraction: metric index midx pairs with fj index tidx
                    other_midx = midxs[3 - mi]

                    # Special case: metric × metric → delta
                    fj_props = has_tensor(reg, fj.name) ? get_tensor(reg, fj.name) : nothing
                    is_partner_metric = fj_props !== nothing &&
                        get(fj_props.options, :is_metric, false)

                    if is_partner_metric
                        # g^{ab} g_{bc} → δ^a_c
                        other_fidx = fj.indices[3 - ti]
                        delta_name = _find_delta(reg, get_tensor(reg, metric.name).manifold)
                        new_tensor = Tensor(delta_name,
                            [other_midx, TIndex(other_fidx.name, other_fidx.position)])
                    else
                        # g^{ab} T_{bc} → T^a_c  (raise/lower index)
                        new_indices = copy(fj.indices)
                        new_indices[ti] = TIndex(other_midx.name, other_midx.position)
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

    nothing
end

"""Find the delta tensor name for a manifold, or :δ by default."""
function _find_delta(reg::TensorRegistry, manifold::Symbol)
    for (name, tp) in reg.tensors
        if tp.manifold == manifold && get(tp.options, :is_delta, false)
            return name
        end
    end
    :δ
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
                if didx.name == tidx.name && didx.position != tidx.position
                    # The other delta index replaces the contracted index
                    other_didx = didxs[3 - di]
                    new_indices = copy(fj.indices)
                    new_indices[ti] = TIndex(other_didx.name, tidx.position)
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
    if didxs[1].name == didxs[2].name && didxs[1].position != didxs[2].position
        mp = get_manifold(reg, get_tensor(reg, delta.name).manifold)
        dim = mp.dim
        new_factors = TensorExpr[f for (k, f) in enumerate(p.factors) if k != delta_idx]
        return tproduct(p.scalar * dim, new_factors)
    end

    nothing
end
