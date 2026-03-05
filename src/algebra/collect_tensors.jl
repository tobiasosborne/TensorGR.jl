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

"""
    remove_constants(expr::TensorExpr) -> TensorExpr

Strip scalar constants from terms, keeping only tensor factors.
"""
remove_constants(t::Tensor) = t
remove_constants(::TScalar) = TScalar(1 // 1)
remove_constants(d::TDeriv) = d
function remove_constants(p::TProduct)
    tproduct(1 // 1, TensorExpr[f for f in p.factors if !(f isa TScalar)])
end
function remove_constants(s::TSum)
    tsum(TensorExpr[remove_constants(t) for t in s.terms])
end

"""
    remove_tensors(expr::TensorExpr) -> TensorExpr

Strip tensor factors from terms, keeping only scalar parts.
"""
remove_tensors(::Tensor) = TScalar(1 // 1)
remove_tensors(s::TScalar) = s
remove_tensors(::TDeriv) = TScalar(1 // 1)
function remove_tensors(p::TProduct)
    scalars = TensorExpr[f for f in p.factors if f isa TScalar]
    tproduct(p.scalar, isempty(scalars) ? TensorExpr[TScalar(1 // 1)] : scalars)
end
function remove_tensors(s::TSum)
    tsum(TensorExpr[remove_tensors(t) for t in s.terms])
end

"""
    index_collect(expr::TSum, tensor_name::Symbol) -> Dict{TensorExpr, Any}

Collect terms by tensor structure involving the named tensor,
returning a dictionary mapping tensor structures to scalar coefficients.
"""
function index_collect(expr::TSum, tensor_name::Symbol)
    result = Dict{TensorExpr, Any}()
    for term in expr.terms
        scalar, core = _split_scalar_extended(term)
        if _contains_tensor(core, tensor_name)
            key = core
            result[key] = get(result, key, 0) + scalar
        end
    end
    result
end

function _contains_tensor(expr::Tensor, name::Symbol)
    expr.name == name
end
function _contains_tensor(expr::TProduct, name::Symbol)
    any(f -> _contains_tensor(f, name), expr.factors)
end
function _contains_tensor(expr::TSum, name::Symbol)
    any(t -> _contains_tensor(t, name), expr.terms)
end
function _contains_tensor(expr::TDeriv, name::Symbol)
    _contains_tensor(expr.arg, name)
end
_contains_tensor(::TScalar, ::Symbol) = false
