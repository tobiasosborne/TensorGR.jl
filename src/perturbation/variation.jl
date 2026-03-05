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

"""
    metric_variation(expr::TensorExpr, metric::Symbol,
                     idx_a::TIndex, idx_b::TIndex) -> TensorExpr

Compute the variation of an expression with respect to the inverse metric g^{ab}.

Key identities:
- δ(g^{ab})/δ(g^{cd}) = (1/2)(δ^a_c δ^b_d + δ^a_d δ^b_c)
- δ(g_{ab})/δ(g^{cd}) = -(1/2)(g_{ac} g_{bd} + g_{ad} g_{bc})
"""
function metric_variation(expr::TensorExpr, metric::Symbol,
                          idx_c::TIndex, idx_d::TIndex)
    _metric_var_walk(expr, metric, idx_c, idx_d)
end

function _metric_var_walk(t::Tensor, metric::Symbol, c::TIndex, d::TIndex)
    if t.name == metric && length(t.indices) == 2
        a, b = t.indices
        if a.position == Up && b.position == Up
            # δ(g^{ab})/δ(g^{cd}) = (1/2)(δ^a_c δ^b_d + δ^a_d δ^b_c)
            return (1 // 2) * (Tensor(:δ, [a, c]) * Tensor(:δ, [b, d]) +
                               Tensor(:δ, [a, d]) * Tensor(:δ, [b, c]))
        elseif a.position == Down && b.position == Down
            # δ(g_{ab})/δ(g^{cd}) = -(1/2)(g_{ac} g_{bd} + g_{ad} g_{bc})
            return (-1 // 2) * (Tensor(metric, [a, c]) * Tensor(metric, [b, d]) +
                                Tensor(metric, [a, d]) * Tensor(metric, [b, c]))
        end
    end
    TScalar(0 // 1)
end

function _metric_var_walk(s::TScalar, ::Symbol, ::TIndex, ::TIndex)
    TScalar(0 // 1)
end

function _metric_var_walk(s::TSum, metric::Symbol, c::TIndex, d::TIndex)
    tsum(TensorExpr[_metric_var_walk(t, metric, c, d) for t in s.terms])
end

function _metric_var_walk(p::TProduct, metric::Symbol, c::TIndex, d::TIndex)
    # Leibniz rule on products
    factors = p.factors
    terms = TensorExpr[]
    for i in eachindex(factors)
        var_i = _metric_var_walk(factors[i], metric, c, d)
        var_i == TScalar(0 // 1) && continue
        new_factors = TensorExpr[]
        for (j, fj) in enumerate(factors)
            if j == i
                push!(new_factors, var_i)
            else
                push!(new_factors, fj)
            end
        end
        push!(terms, tproduct(p.scalar, new_factors))
    end
    tsum(terms)
end

function _metric_var_walk(d::TDeriv, metric::Symbol, c::TIndex, dd::TIndex)
    # Chain rule through derivatives
    TDeriv(d.index, _metric_var_walk(d.arg, metric, c, dd))
end

"""
    var_lagrangian(lagrangian::TensorExpr, metric::Symbol;
                   idx_a::TIndex=down(:a), idx_b::TIndex=down(:b)) -> TensorExpr

Vary a Lagrangian density with respect to the metric.
Returns (1/√-g) δ(√-g L)/δg^{ab}.
"""
function var_lagrangian(lagrangian::TensorExpr, metric::Symbol;
                        idx_a::TIndex=down(:a), idx_b::TIndex=down(:b))
    metric_variation(lagrangian, metric, idx_a, idx_b)
end
