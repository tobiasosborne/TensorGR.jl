"""
    to_expr(t::TensorExpr) -> Expr

Convert a TensorExpr to a Julia Expr for escape-hatch manipulation.
"""
function to_expr(idx::TIndex)
    Expr(:call, idx.position == Up ? :up : :down, QuoteNode(idx.name))
end

function to_expr(t::Tensor)
    Expr(:call, :Tensor, QuoteNode(t.name),
         Expr(:vect, map(to_expr, t.indices)...))
end

function to_expr(p::TProduct)
    Expr(:call, :TProduct, p.scalar,
         Expr(:vect, map(to_expr, p.factors)...))
end

function to_expr(s::TSum)
    Expr(:call, :TSum,
         Expr(:vect, map(to_expr, s.terms)...))
end

function to_expr(d::TDeriv)
    Expr(:call, :TDeriv, to_expr(d.index), to_expr(d.arg))
end

function to_expr(s::TScalar)
    Expr(:call, :TScalar, s.val isa Symbol ? QuoteNode(s.val) : s.val)
end

"""
    from_expr(ex::Expr) -> TensorExpr

Convert a Julia Expr (produced by `to_expr`) back to a TensorExpr.
"""
function from_expr(ex::Expr)
    ex.head == :call || error("Expected :call expression, got $(ex.head)")
    tag = ex.args[1]

    if tag == :Tensor
        name = ex.args[2].value
        idxs = TIndex[_idx_from_expr(a) for a in ex.args[3].args]
        return Tensor(name, idxs)
    elseif tag == :TProduct
        scalar = ex.args[2]
        factors = TensorExpr[from_expr(a) for a in ex.args[3].args]
        return TProduct(scalar, factors)
    elseif tag == :TSum
        terms = TensorExpr[from_expr(a) for a in ex.args[2].args]
        return TSum(terms)
    elseif tag == :TDeriv
        idx = _idx_from_expr(ex.args[2])
        arg = from_expr(ex.args[3])
        return TDeriv(idx, arg)
    elseif tag == :TScalar
        val = ex.args[2]
        val = val isa QuoteNode ? val.value : val
        return TScalar(val)
    else
        error("Unknown TensorExpr tag: $tag")
    end
end

function _idx_from_expr(ex::Expr)
    ex.head == :call || error("Expected :call for TIndex, got $(ex.head)")
    pos_sym = ex.args[1]
    name = ex.args[2].value
    pos = pos_sym == :up ? Up : Down
    TIndex(name, pos)
end

"""
    is_well_formed(expr::TensorExpr) -> Bool

Check that the expression satisfies basic structural invariants:
- No duplicate dummy names with wrong pairing (must be one Up, one Down)
- All TProduct factors are TensorExpr subtypes
"""
function is_well_formed(expr::Tensor)
    true
end

function is_well_formed(expr::TScalar)
    true
end

function is_well_formed(expr::TProduct)
    all(is_well_formed, expr.factors) || return false
    # Check no clashing dummies within a single factor
    # (dummies should only exist across factors in a product)
    all_idxs = indices(expr)
    name_groups = Dict{Symbol, Vector{TIndex}}()
    for idx in all_idxs
        push!(get!(Vector{TIndex}, name_groups, idx.name), idx)
    end
    for (name, idxs) in name_groups
        up_count = count(i -> i.position == Up, idxs)
        down_count = count(i -> i.position == Down, idxs)
        # Each name should appear at most once as Up and once as Down
        (up_count <= 1 && down_count <= 1) || return false
    end
    true
end

function is_well_formed(expr::TSum)
    all(is_well_formed, expr.terms)
end

function is_well_formed(expr::TDeriv)
    is_well_formed(expr.arg)
end
