"""
    children(expr::TensorExpr) -> Vector{TensorExpr}

Return the immediate sub-expressions of `expr`.
"""
children(::Tensor) = TensorExpr[]
children(::TScalar) = TensorExpr[]
children(p::TProduct) = p.factors
children(s::TSum) = s.terms
children(d::TDeriv) = TensorExpr[d.arg]

"""
    walk(f, expr::TensorExpr) -> TensorExpr

Recursively apply `f` to all sub-expressions bottom-up.
First walks children, then applies `f` to the rebuilt parent.
"""
function walk(f, expr::Tensor)
    f(expr)
end

function walk(f, expr::TScalar)
    f(expr)
end

function walk(f, expr::TProduct)
    new_factors = TensorExpr[walk(f, factor) for factor in expr.factors]
    f(TProduct(expr.scalar, new_factors))
end

function walk(f, expr::TSum)
    new_terms = TensorExpr[walk(f, term) for term in expr.terms]
    f(TSum(new_terms))
end

function walk(f, expr::TDeriv)
    new_arg = walk(f, expr.arg)
    f(TDeriv(expr.index, new_arg))
end

"""
    substitute(expr::TensorExpr, rule::Pair{<:TensorExpr, <:TensorExpr}) -> TensorExpr

Replace all occurrences of `rule.first` with `rule.second` in `expr`.
"""
function substitute(expr::TensorExpr, rule::Pair{<:TensorExpr, <:TensorExpr})
    old, new = rule
    walk(expr) do node
        node == old ? new : node
    end
end
