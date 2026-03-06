#= SVT field substitution rules for 3+1 decomposition.

Maps tensor components to their SVT decomposition:
  h_{00} → 2Φ
  h_{0i} → ∂_i B + S_i
  h_{ij} → 2ψδ_{ij} + 2∂_i∂_j E + ∂_i F_j + ∂_j F_i + h^TT_{ij}

In Bardeen gauge (B=E=F=0):
  h_{00} → 2Φ
  h_{0i} → S_i  (or V_i for transverse vector)
  h_{ij} → 2ψδ_{ij} + h^TT_{ij}
=#

"""
    SVTSubstitution(tensor_name, pattern, replacement)

A substitution rule mapping a tensor with a specific component pattern
to its SVT replacement expression.

- `tensor_name`: name of the tensor to match (e.g., `:h`)
- `pattern`: vector of `:temporal`/`:spatial` labels for each index slot
- `replacement`: function `(indices) -> TensorExpr` producing the replacement
"""
struct SVTSubstitution
    tensor_name::Symbol
    pattern::Vector{Symbol}  # :temporal or :spatial for each slot
    replacement::Any         # Function(Vector{TIndex}) -> TensorExpr
end

"""
    svt_rules_bardeen(fields::SVTFields) -> Vector{SVTSubstitution}

SVT substitution rules in Bardeen gauge (B=E=F=0):
- h_{00} → 2Φ
- h_{0i} → V_i  (transverse vector, using S field)
- h_{i0} → V_i  (symmetric)
- h_{ij} → 2ψδ_{ij} + h^TT_{ij}
"""
function svt_rules_bardeen(fields::SVTFields=DEFAULT_SVT)
    rules = SVTSubstitution[]

    # h_{00} → 2Φ (scalar)
    push!(rules, SVTSubstitution(:h, [:temporal, :temporal],
        _ -> tproduct(2 // 1, TensorExpr[Tensor(fields.ϕ, TIndex[])])))

    # h_{0i} → V_i (transverse vector)
    push!(rules, SVTSubstitution(:h, [:temporal, :spatial],
        idxs -> Tensor(fields.S, TIndex[idxs[2]])))

    # h_{i0} → V_i (symmetric)
    push!(rules, SVTSubstitution(:h, [:spatial, :temporal],
        idxs -> Tensor(fields.S, TIndex[idxs[1]])))

    # h_{ij} → 2ψδ_{ij} + h^TT_{ij}
    push!(rules, SVTSubstitution(:h, [:spatial, :spatial],
        idxs -> tsum(TensorExpr[
            tproduct(2 // 1, TensorExpr[
                Tensor(fields.ψ, TIndex[]),
                Tensor(:δ, TIndex[idxs[1], idxs[2]])
            ]),
            Tensor(fields.hTT, TIndex[idxs[1], idxs[2]])
        ])))

    rules
end

"""
    svt_rules_full(fields::SVTFields) -> Vector{SVTSubstitution}

Full SVT substitution rules (all 7 SVT fields):
- h_{00} → 2Φ
- h_{0i} → ∂_i B + S_i
- h_{ij} → 2ψδ_{ij} + 2∂_i∂_j E + ∂_i F_j + ∂_j F_i + h^TT_{ij}
"""
function svt_rules_full(fields::SVTFields=DEFAULT_SVT)
    rules = SVTSubstitution[]

    # h_{00} → 2Φ
    push!(rules, SVTSubstitution(:h, [:temporal, :temporal],
        _ -> tproduct(2 // 1, TensorExpr[Tensor(fields.ϕ, TIndex[])])))

    # h_{0i} → ∂_i B + S_i
    push!(rules, SVTSubstitution(:h, [:temporal, :spatial],
        idxs -> tsum(TensorExpr[
            TDeriv(idxs[2], Tensor(fields.B, TIndex[])),
            Tensor(fields.S, TIndex[idxs[2]])
        ])))

    # h_{i0} → ∂_i B + S_i (symmetric)
    push!(rules, SVTSubstitution(:h, [:spatial, :temporal],
        idxs -> tsum(TensorExpr[
            TDeriv(idxs[1], Tensor(fields.B, TIndex[])),
            Tensor(fields.S, TIndex[idxs[1]])
        ])))

    # h_{ij} → 2ψδ_{ij} + 2∂_i∂_j E + ∂_i F_j + ∂_j F_i + h^TT_{ij}
    push!(rules, SVTSubstitution(:h, [:spatial, :spatial],
        idxs -> tsum(TensorExpr[
            tproduct(2 // 1, TensorExpr[
                Tensor(fields.ψ, TIndex[]),
                Tensor(:δ, TIndex[idxs[1], idxs[2]])
            ]),
            tproduct(2 // 1, TensorExpr[
                TDeriv(idxs[1], TDeriv(idxs[2], Tensor(fields.E, TIndex[])))
            ]),
            TDeriv(idxs[1], Tensor(fields.F, TIndex[idxs[2]])),
            TDeriv(idxs[2], Tensor(fields.F, TIndex[idxs[1]])),
            Tensor(fields.hTT, TIndex[idxs[1], idxs[2]])
        ])))

    rules
end

"""
    apply_svt(expr::TensorExpr, rules::Vector{SVTSubstitution},
              fol::FoliationProperties) -> TensorExpr

Walk the expression tree and apply SVT substitution rules to tensors
whose indices match the component patterns.
"""
function apply_svt(expr::TensorExpr, rules::Vector{SVTSubstitution},
                   fol::FoliationProperties)
    walk(expr) do node
        node isa Tensor || return node
        _try_svt_match(node, rules, fol)
    end
end

function _try_svt_match(t::Tensor, rules::Vector{SVTSubstitution},
                        fol::FoliationProperties)
    for rule in rules
        t.name == rule.tensor_name || continue
        length(t.indices) == length(rule.pattern) || continue

        # Check if all indices match the pattern
        match = true
        for (idx, pat) in zip(t.indices, rule.pattern)
            if pat == :temporal
                is_temporal_component(idx, fol) || (match = false; break)
            elseif pat == :spatial
                is_spatial_component(idx, fol) || (match = false; break)
            end
        end

        match && return rule.replacement(t.indices)
    end
    t  # no match
end

"""
    apply_svt(expr::TensorExpr, h_name::Symbol, fol::FoliationProperties;
              gauge=:bardeen, fields=DEFAULT_SVT) -> TensorExpr

Convenience method: apply SVT rules for a given perturbation tensor and gauge.
"""
function apply_svt(expr::TensorExpr, h_name::Symbol, fol::FoliationProperties;
                   gauge::Symbol=:bardeen, fields::SVTFields=DEFAULT_SVT)
    rules = gauge == :bardeen ? svt_rules_bardeen(fields) : svt_rules_full(fields)
    # Adapt rules for the specific tensor name
    adapted = SVTSubstitution[
        SVTSubstitution(h_name, r.pattern, r.replacement) for r in rules
    ]
    apply_svt(expr, adapted, fol)
end
