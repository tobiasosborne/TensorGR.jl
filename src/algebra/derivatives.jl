#= Derivative expansion: Leibniz rule and linearity.

expand_derivatives applies the Leibniz rule to derivatives of products
and distributes derivatives over sums. It does NOT introduce connection
terms (Christoffel symbols) — that belongs to the covariant derivative
layer in Layer 3.
=#

"""
    expand_derivatives(expr::TensorExpr) -> TensorExpr

Expand derivatives using the Leibniz rule (product rule) and linearity.
- ∂(A * B) = (∂A) * B + A * (∂B)
- ∂(A + B) = ∂A + ∂B
- ∂(scalar) = 0
- ∂(c * A) = c * ∂A  for constant scalar c
"""
expand_derivatives(t::Tensor) = t
expand_derivatives(s::TScalar) = s

function expand_derivatives(p::TProduct)
    TProduct(p.scalar, TensorExpr[expand_derivatives(f) for f in p.factors])
end

function expand_derivatives(s::TSum)
    tsum(TensorExpr[expand_derivatives(t) for t in s.terms])
end

function expand_derivatives(d::TDeriv)
    arg = expand_derivatives(d.arg)
    _apply_deriv(d.index, arg)
end

# ─── Leibniz dispatch ────────────────────────────────────────────────

# Derivative of a single tensor: irreducible
_apply_deriv(idx::TIndex, t::Tensor) = TDeriv(idx, t)

# Derivative of another derivative: keep nesting
_apply_deriv(idx::TIndex, d::TDeriv) = TDeriv(idx, d)

# Derivative of a constant scalar: zero
_apply_deriv(::TIndex, ::TScalar) = ZERO

# Derivative of a sum: linearity
function _apply_deriv(idx::TIndex, s::TSum)
    tsum(TensorExpr[_apply_deriv(idx, t) for t in s.terms])
end

# Derivative of a product: Leibniz rule
function _apply_deriv(idx::TIndex, p::TProduct)
    factors = p.factors

    # If the product has no tensor factors (pure scalar), derivative is zero
    all(f -> f isa TScalar, factors) && return ZERO

    # If single factor, just wrap it
    if length(factors) == 1
        inner = _apply_deriv(idx, factors[1])
        return tproduct(p.scalar, TensorExpr[inner])
    end

    # Leibniz rule: ∂(f₁ f₂ ⋯ fₖ) = Σᵢ f₁ ⋯ (∂fᵢ) ⋯ fₖ
    terms = TensorExpr[]
    for i in eachindex(factors)
        new_factors = TensorExpr[]
        for (j, fj) in enumerate(factors)
            if j == i
                push!(new_factors, _apply_deriv(idx, fj))
            else
                push!(new_factors, fj)
            end
        end
        push!(terms, tproduct(p.scalar, new_factors))
    end
    tsum(terms)
end
