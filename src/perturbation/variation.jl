#= Variational derivatives.

Functional derivative δL/δφ for a Lagrangian density L that depends on
a field φ and its derivatives.

The algorithm:
1. Expand all derivatives via Leibniz rule
2. Identify terms containing δφ (the variation of φ)
3. Use IBP to move all derivatives off δφ
4. Read off the coefficient of δφ = the Euler-Lagrange equation
=#

"""
    variational_derivative(lagrangian, field; deriv_field=nothing) -> TensorExpr

Compute the variational derivative δL/δφ.

For a Lagrangian L(φ, ∂φ, ∂∂φ), this returns the Euler-Lagrange equation:
  δL/δφ = ∂L/∂φ - ∂_a(∂L/∂(∂_a φ)) + ∂_a∂_b(∂L/∂(∂_a∂_b φ)) - ...

The `field` argument is the symbol name of the field to vary with respect to.
"""
function variational_derivative(lagrangian::TensorExpr, field::Symbol)
    # Step 1: Expand all derivatives
    expanded = expand_derivatives(lagrangian)
    expanded = expand_products(expanded)

    # Step 2: For each term in the sum, apply IBP to move derivatives off `field`
    if expanded isa TSum
        terms = TensorExpr[]
        for term in expanded.terms
            ibp_result = _ibp_all(term, field)
            push!(terms, ibp_result)
        end
        result = tsum(terms)
    else
        result = _ibp_all(expanded, field)
    end

    # Step 3: Collect terms
    collect_terms(result)
end

"""
Apply IBP repeatedly until no derivatives act on `field`.
"""
function _ibp_all(expr::TensorExpr, field::Symbol; maxiter::Int=10)
    current = expr
    for _ in 1:maxiter
        if current isa TProduct
            next = ibp_product(current, field)
            next == current && return current
            current = expand_products(next)
            current = expand_derivatives(current)
        else
            return current
        end
    end
    current
end

"""
    euler_lagrange(lagrangian, fields::Vector{Symbol}) -> Vector{TensorExpr}

Compute the Euler-Lagrange equations for multiple fields.
Returns a vector of equations (one per field), each set to zero.
"""
function euler_lagrange(lagrangian::TensorExpr, fields::Vector{Symbol})
    [variational_derivative(lagrangian, f) for f in fields]
end
