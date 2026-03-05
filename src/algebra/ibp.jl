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

Integration by parts inside a product: for each factor that is a derivative
of `field`, peel off the outermost derivative, flip sign, and apply it to
the remaining factors via the Leibniz rule.

Runs one step — call repeatedly or use `ibp_all` for fixed point.
"""
function ibp_product(p::TProduct, field::Symbol)
    factors = p.factors
    for (i, fi) in enumerate(factors)
        stripped, idx = _peel_deriv_of(fi, field)
        idx === nothing && continue

        # Found: factor i has a derivative acting on `field`.
        # IBP: remove the derivative from factor i, apply it to everything else,
        # flip sign.  ∫ (∂_a Φ) Ψ₁ Ψ₂ = -∫ Φ (∂_a(Ψ₁ Ψ₂))
        others = TensorExpr[j == i ? stripped : factors[j] for j in eachindex(factors)]
        # Build the "rest" product (everything except the field that lost its derivative)
        rest_factors = TensorExpr[others[j] for j in eachindex(others) if j != i]
        rest = isempty(rest_factors) ? TScalar(1 // 1) : tproduct(1 // 1, rest_factors)

        # Apply the derivative to the rest via Leibniz
        d_rest = expand_derivatives(TDeriv(idx, rest))

        # Reassemble: -scalar * stripped * d_rest
        return tproduct(-p.scalar, TensorExpr[stripped, d_rest])
    end
    p  # no derivative of `field` found
end

"""
Peel the outermost derivative off an expression, if it ultimately acts on `field`.
Returns (inner_without_deriv, derivative_index) or (expr, nothing).
"""
function _peel_deriv_of(expr::TDeriv, field::Symbol)
    if _acts_on(expr, field)
        return (expr.arg, expr.index)
    end
    (expr, nothing)
end

_peel_deriv_of(expr::TensorExpr, ::Symbol) = (expr, nothing)

"""Check if a derivative expression ultimately acts on a tensor named `field`."""
_acts_on(d::TDeriv, field::Symbol) = _acts_on(d.arg, field)
_acts_on(t::Tensor, field::Symbol) = t.name == field
_acts_on(::TensorExpr, ::Symbol) = false
