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

"""
    dagger(expr::TensorExpr) -> TensorExpr

Complex conjugation of a tensor expression.
Swaps Up↔Down on all indices (Hermitian conjugation).
For real tensors, this is the identity.
"""
function dagger(t::Tensor)
    new_indices = [TIndex(idx.name, idx.position == Up ? Down : Up, idx.vbundle) for idx in t.indices]
    Tensor(Symbol(t.name, :_dag), new_indices)
end
dagger(s::TScalar) = s  # scalars are assumed real
function dagger(p::TProduct)
    # Reverse factor order for non-commutative tensors
    TProduct(p.scalar, TensorExpr[dagger(f) for f in reverse(p.factors)])
end
function dagger(s::TSum)
    tsum(TensorExpr[dagger(t) for t in s.terms])
end
function dagger(d::TDeriv)
    TDeriv(TIndex(d.index.name, d.index.position == Up ? Down : Up, d.index.vbundle), dagger(d.arg))
end

"""
    derivative_order(expr::TensorExpr) -> Int

Count the total number of derivatives acting in an expression.
"""
derivative_order(::Tensor) = 0
derivative_order(::TScalar) = 0
derivative_order(d::TDeriv) = 1 + derivative_order(d.arg)
function derivative_order(p::TProduct)
    isempty(p.factors) ? 0 : maximum(derivative_order(f) for f in p.factors)
end
function derivative_order(s::TSum)
    isempty(s.terms) ? 0 : maximum(derivative_order(t) for t in s.terms)
end

"""
    is_constant(expr::TensorExpr) -> Bool

Check if expression has no free or dummy indices (i.e., is a scalar constant).
"""
is_constant(::TScalar) = true
is_constant(t::Tensor) = isempty(t.indices)
is_constant(d::TDeriv) = false
function is_constant(p::TProduct)
    all(is_constant, p.factors)
end
function is_constant(s::TSum)
    all(is_constant, s.terms)
end

"""
    is_sorted_covds(expr::TensorExpr) -> Bool

Check if covariant derivatives in the expression are in canonical (alphabetical) order.
"""
is_sorted_covds(::Tensor) = true
is_sorted_covds(::TScalar) = true
function is_sorted_covds(d::TDeriv)
    if d.arg isa TDeriv
        d.index.name <= d.arg.index.name || return false
    end
    is_sorted_covds(d.arg)
end
function is_sorted_covds(p::TProduct)
    all(is_sorted_covds, p.factors)
end
function is_sorted_covds(s::TSum)
    all(is_sorted_covds, s.terms)
end
