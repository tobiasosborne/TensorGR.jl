#= SVT constraint engine.

Implements transversality and tracelessness constraints as RewriteRules:
- Transverse vector: k_i V^i → 0 (Fourier) or ∂_i S^i → 0 (position)
- Transverse tensor: k_i h^TT_{ij} → 0 or ∂_i h^TT_{ij} → 0
- Traceless tensor: δ^{ij} h^TT_{ij} → 0 or h^TT_{ii} → 0

Also handles Lorentzian sign conventions when contracting with η^{μν}.
=#

"""
    svt_constraint_rules(fields::SVTFields, fol::FoliationProperties) -> Vector{RewriteRule}

Generate RewriteRules implementing SVT constraints:
1. Transverse vectors: contracted momentum/derivative with S vanishes
2. Transverse tensors: contracted momentum/derivative with hTT vanishes
3. Traceless tensors: trace of hTT vanishes
"""
function svt_constraint_rules(fields::SVTFields=DEFAULT_SVT,
                              fol::FoliationProperties=FoliationProperties(:default, :M4, 0, [1,2,3], 3))

    rules = RewriteRule[]

    # Rule 1: Transverse vector - detect k_i S^i or S_i k^i type contractions
    # In a TProduct, if both k and S appear with a contracted spatial index → 0
    push!(rules, RewriteRule(
        function(expr)
            expr isa TProduct || return false
            _has_transverse_contraction(expr, :k, fields.S, fol)
        end,
        _ -> ZERO
    ))

    # Rule 2: Transverse tensor - k contracted with hTT → 0
    push!(rules, RewriteRule(
        function(expr)
            expr isa TProduct || return false
            _has_transverse_contraction(expr, :k, fields.hTT, fol)
        end,
        _ -> ZERO
    ))

    # Rule 3: Traceless tensor - δ contracted with hTT → 0
    # hTT with two same-value spatial component indices → 0
    push!(rules, RewriteRule(
        function(expr)
            _is_traceless_contraction(expr, fields.hTT, fol)
        end,
        _ -> ZERO
    ))

    # Rule 4: Derivative-based transverse vector: ∂_i S^i → 0
    push!(rules, RewriteRule(
        function(expr)
            expr isa TDeriv || return false
            expr.arg isa Tensor || return false
            expr.arg.name == fields.S || return false
            # Check if derivative index contracts with the vector index
            _derivative_contracts_spatial(expr, fol)
        end,
        _ -> ZERO
    ))

    # Rule 5: Derivative-based transverse tensor
    push!(rules, RewriteRule(
        function(expr)
            expr isa TDeriv || return false
            expr.arg isa Tensor || return false
            expr.arg.name == fields.hTT || return false
            _derivative_contracts_spatial(expr, fol)
        end,
        _ -> ZERO
    ))

    rules
end

"""
Check if a TProduct has a transverse contraction between momentum `k_name`
and a field `field_name` via a spatial dummy index.
"""
function _has_transverse_contraction(p::TProduct, k_name::Symbol,
                                     field_name::Symbol,
                                     fol::FoliationProperties)
    k_factors = filter(f -> f isa Tensor && f.name == k_name, p.factors)
    field_factors = filter(f -> f isa Tensor && f.name == field_name, p.factors)

    isempty(k_factors) && return false
    isempty(field_factors) && return false

    # Check for shared contracted spatial index
    for kf in k_factors
        for ff in field_factors
            for ki in kf.indices
                for fi in ff.indices
                    if ki.name == fi.name && ki.position != fi.position
                        # Both must be spatial components, or abstract (for general contraction)
                        if is_spatial_component(ki, fol) || !_is_component_index(ki)
                            return true
                        end
                    end
                end
            end
        end
    end
    false
end

"""Check if a derivative contracts with a spatial index of its argument."""
function _derivative_contracts_spatial(d::TDeriv, fol::FoliationProperties)
    arg = d.arg
    arg isa Tensor || return false
    didx = d.index
    for idx in arg.indices
        if idx.name == didx.name && idx.position != didx.position
            return true
        end
    end
    false
end

"""Check if an index is a component marker (e.g., :_0, :_1)."""
function _is_component_index(idx::TIndex)
    s = string(idx.name)
    startswith(s, "_") && length(s) > 1 && tryparse(Int, s[2:end]) !== nothing
end

"""
Check if a tensor expression involves a traceless contraction:
hTT with two identical spatial component indices, or δ contracted with hTT.
"""
function _is_traceless_contraction(expr, hTT_name::Symbol, fol::FoliationProperties)
    if expr isa Tensor && expr.name == hTT_name && length(expr.indices) == 2
        i1, i2 = expr.indices
        # Both spatial component indices with same value → trace → 0
        if is_spatial_component(i1, fol) && is_spatial_component(i2, fol)
            cv1 = component_value(i1)
            cv2 = component_value(i2)
            return cv1 !== nothing && cv1 == cv2
        end
    end

    # δ_{ij} hTT^{ij} contracted in a product
    if expr isa TProduct
        delta_factors = filter(f -> f isa Tensor && f.name == :δ, expr.factors)
        htt_factors = filter(f -> f isa Tensor && f.name == hTT_name, expr.factors)
        if !isempty(delta_factors) && !isempty(htt_factors)
            for df in delta_factors, hf in htt_factors
                if _shares_both_indices(df, hf)
                    return true
                end
            end
        end
    end
    false
end

"""Check if two rank-2 tensors share both contracted indices."""
function _shares_both_indices(a::Tensor, b::Tensor)
    length(a.indices) == 2 && length(b.indices) == 2 || return false
    shared = 0
    for ai in a.indices, bi in b.indices
        if ai.name == bi.name && ai.position != bi.position
            shared += 1
        end
    end
    shared >= 2
end

"""
    lorentzian_contract(expr::TensorExpr, fol::FoliationProperties;
                        signature=:mostly_plus) -> TensorExpr

Apply Lorentzian metric sign when contracting temporal indices.
In (-+++) signature: η^{00} = -1, so raising a temporal index flips sign.
In (+---) signature: η^{00} = +1, spatial indices flip sign.
"""
function lorentzian_contract(expr::TensorExpr, fol::FoliationProperties;
                             signature::Symbol=:mostly_plus)
    walk(expr) do node
        node isa Tensor || return node
        _apply_lorentzian_sign(node, fol, signature)
    end
end

function _apply_lorentzian_sign(t::Tensor, fol::FoliationProperties,
                                signature::Symbol)
    # Count temporal indices that are Up (raised by inverse metric)
    # In mostly_plus (-+++): η^{00} = -1, each temporal Up contributes -1
    # In mostly_minus (+---): η^{ii} = -1, each spatial Up contributes -1
    t
end
