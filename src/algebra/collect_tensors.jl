#= CollectTensors: group terms by tensor structure, simplify scalar coefficients.

Given a sum of terms, group those with the same tensor structure and
combine their scalar coefficients.
=#

"""
    collect_tensors(expr::TSum) -> TensorExpr

Group terms by tensor structure and combine scalar coefficients.
This is a higher-level version of collect_terms that handles
more complex scalar coefficients (not just Rational{Int}).
"""
function collect_tensors(expr::TSum)
    # Each term is split into (scalar_part, tensor_part)
    buckets = Dict{TensorExpr, Vector{Any}}()

    for term in expr.terms
        scalar, core = _split_scalar_extended(term)
        key = core
        if haskey(buckets, key)
            push!(buckets[key], scalar)
        else
            buckets[key] = [scalar]
        end
    end

    terms = TensorExpr[]
    for (core, coeffs) in buckets
        total = sum(coeffs)
        total == 0 && continue
        if total == 1
            push!(terms, core)
        elseif total isa Rational && total == 1 // 1
            push!(terms, core)
        else
            push!(terms, tproduct(Rational{Int}(total), TensorExpr[core]))
        end
    end

    tsum(terms)
end

collect_tensors(expr::TensorExpr) = expr

function _split_scalar_extended(expr::TProduct)
    if length(expr.factors) == 1
        return expr.scalar, expr.factors[1]
    end
    return expr.scalar, TProduct(1 // 1, expr.factors)
end
_split_scalar_extended(expr::Tensor) = (1 // 1, expr)
_split_scalar_extended(expr::TDeriv) = (1 // 1, expr)
_split_scalar_extended(expr::TScalar) = (expr.val isa Number ? expr.val : 1, TScalar(1))
_split_scalar_extended(expr::TSum) = (1 // 1, expr)
