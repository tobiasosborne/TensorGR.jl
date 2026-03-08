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
    Expr(:call, :TDeriv, to_expr(d.index), to_expr(d.arg), QuoteNode(d.covd))
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
        covd = length(ex.args) >= 4 ? ex.args[4].value : :partial
        return TDeriv(idx, arg, covd)
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
            # Each (name, vbundle) should appear at most once as Up and once as Down
        (up_count <= 1 && down_count <= 1) || return false
    end
    true
end

function is_well_formed(expr::TSum)
    all(is_well_formed, expr.terms) || return false
    # Check that all terms have the same free indices
    isempty(expr.terms) && return true
    ref_free = sort(free_indices(expr.terms[1]), by=idx -> (idx.name, idx.position))
    for i in 2:length(expr.terms)
        term_free = sort(free_indices(expr.terms[i]), by=idx -> (idx.name, idx.position))
        length(ref_free) == length(term_free) || return false
        for (a, b) in zip(ref_free, term_free)
            a.name == b.name && a.position == b.position || return false
        end
    end
    true
end

function is_well_formed(expr::TDeriv)
    is_well_formed(expr.arg)
end

"""
    validate(expr::TensorExpr; registry=current_registry()) -> Vector{String}

Deep validation of an expression. Returns a list of issues found.
Checks:
- Well-formedness (index pairing)
- Free index consistency in sums
- Tensor rank consistency with registry
"""
function validate(expr::TensorExpr; registry::TensorRegistry=current_registry())
    issues = String[]
    _validate_walk(expr, registry, issues)
    issues
end

function _validate_walk(t::Tensor, reg::TensorRegistry, issues::Vector{String})
    if has_tensor(reg, t.name)
        props = get_tensor(reg, t.name)
        expected_rank = sum(props.rank)
        actual_rank = length(t.indices)
        if actual_rank != expected_rank
            push!(issues, "Tensor $(t.name): expected $(expected_rank) indices, got $(actual_rank)")
        end
    end
end

function _validate_walk(s::TScalar, ::TensorRegistry, ::Vector{String})
end

function _validate_walk(p::TProduct, reg::TensorRegistry, issues::Vector{String})
    for f in p.factors
        _validate_walk(f, reg, issues)
    end
    if !is_well_formed(p)
        push!(issues, "Product has malformed index structure")
    end
end

function _validate_walk(s::TSum, reg::TensorRegistry, issues::Vector{String})
    for t in s.terms
        _validate_walk(t, reg, issues)
    end
    if !is_well_formed(s)
        push!(issues, "Sum has inconsistent free indices across terms")
    end
end

function _validate_walk(d::TDeriv, reg::TensorRegistry, issues::Vector{String})
    _validate_walk(d.arg, reg, issues)
end
