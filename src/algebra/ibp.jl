#= Integration by parts under an implicit ∫ d⁴x.

ibp moves a derivative off one field onto the others in a product,
picking up a sign flip and discarding the boundary term:

  ∫ (∂_a Φ) Ψ d⁴x = -∫ Φ (∂_a Ψ) d⁴x
=#

"""
    ibp(expr::TensorExpr, field::Symbol) -> TensorExpr

Integrate by parts: move all derivatives off tensors named `field`.
Operates under an implicit ∫ d⁴x, discarding boundary terms.
"""
ibp(t::Tensor, ::Symbol) = t
ibp(s::TScalar, ::Symbol) = s

function ibp(s::TSum, field::Symbol)
    tsum(TensorExpr[ibp(t, field) for t in s.terms])
end

function ibp(p::TProduct, field::Symbol)
    tproduct(p.scalar, TensorExpr[ibp(f, field) for f in p.factors])
end

function ibp(d::TDeriv, field::Symbol)
    # If this derivative acts (directly or through nesting) on `field`,
    # flip the sign and move the derivative to act on the "rest" of the
    # expression.  In a standalone context (not inside a product), IBP
    # is a no-op — the partner to absorb the derivative doesn't exist here.
    #
    # The useful case is handled by `ibp_product` which sees the full
    # product structure.
    TDeriv(d.index, ibp(d.arg, field))
end

"""
    ibp_product(expr::TProduct, field::Symbol) -> TensorExpr

Integration by parts inside a product: for each factor that is a chain of
derivatives acting on `field`, peel ALL derivatives at once and transfer them
to the remaining factors.

For a factor ∂_a...∂_n(field), this gives:
  (-1)^n * field * ∂_a...∂_n(rest)
where rest is the product of all other factors.
"""
function ibp_product(p::TProduct, field::Symbol)
    factors = p.factors
    for (i, fi) in enumerate(factors)
        base, idxs = _peel_all_derivs_of(fi, field)
        isempty(idxs) && continue

        # Build the "rest" product (everything except factor i)
        rest_factors = TensorExpr[factors[j] for j in eachindex(factors) if j != i]
        rest = isempty(rest_factors) ? TScalar(1 // 1) : tproduct(1 // 1, rest_factors)

        # Apply all n derivatives to the rest
        d_rest = rest
        for idx in idxs
            d_rest = TDeriv(idx, d_rest)
        end
        d_rest = expand_derivatives(d_rest)

        # Reassemble: (-1)^n * scalar * base * d_rest
        sign = iseven(length(idxs)) ? 1 : -1
        return tproduct(Rational{Int}(sign) * p.scalar, TensorExpr[base, d_rest])
    end
    p  # no derivative of `field` found
end

"""
Peel ALL nested derivatives off an expression that ultimately acts on `field`.
Returns (base_field, [idx_outermost, ..., idx_innermost]).
If the expression does not act on `field`, returns (expr, TIndex[]).
"""
function _peel_all_derivs_of(expr::TDeriv, field::Symbol)
    if _acts_on(expr, field)
        inner, idxs = _peel_all_derivs_of(expr.arg, field)
        return (inner, TIndex[expr.index; idxs])
    end
    (expr, TIndex[])
end

_peel_all_derivs_of(expr::TensorExpr, ::Symbol) = (expr, TIndex[])

"""Check if a derivative expression ultimately acts on a tensor named `field`."""
_acts_on(d::TDeriv, field::Symbol) = _acts_on(d.arg, field)
_acts_on(t::Tensor, field::Symbol) = t.name == field
_acts_on(::TensorExpr, ::Symbol) = false
