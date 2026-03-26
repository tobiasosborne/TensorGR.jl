#= Parametric derivatives: d/dt, d/dτ, etc.

TParamDeriv represents derivatives with respect to scalar parameters
(time, proper time, etc.) independent of manifold coordinates.

Key properties:
  - Index-free: carries no tensor indices
  - Leibniz rule: d/dt(A*B) = dA/dt*B + A*dB/dt
  - Linearity: d/dt(A+B) = dA/dt + dB/dt
  - Zero on constants: d/dt(c) = 0 for constant c
  - Flattening: d/ds(d/dt(x)) = d²/dsdt(x) with sorted params
  - Commutes with partial derivatives: d/dt(∂_a X) = ∂_a(d/dt X)
=#

# ── Smart constructor ───────────────────────────────────────────────────

"""
    param_deriv(param, arg) -> TensorExpr
    param_deriv(params::Vector{Symbol}, arg) -> TensorExpr

Smart constructor for parametric derivatives. Flattens nested TParamDeriv,
sorts parameter lists into canonical order, and propagates zeros.
"""
function param_deriv(param::Symbol, arg::TensorExpr)
    param_deriv(Symbol[param], arg)
end

function param_deriv(params::Vector{Symbol}, arg::TensorExpr)
    isempty(params) && return arg
    # Flatten nested TParamDeriv
    if arg isa TParamDeriv
        return param_deriv(vcat(params, arg.params), arg.arg)
    end
    # Zero propagation
    if arg == TScalar(0 // 1)
        return TScalar(0 // 1)
    end
    TParamDeriv(sort(params), arg)
end

# ── Expansion (Leibniz, linearity, constants) ───────────────────────────

"""
    expand_param_deriv(expr; registry=current_registry()) -> TensorExpr

Expand all TParamDeriv nodes using the Leibniz rule, linearity, and
zero-on-constants. Peels off one parameter at a time from the right.
"""
function expand_param_deriv(expr::TensorExpr;
                            registry::TensorRegistry=current_registry())
    walk(expr) do node
        node isa TParamDeriv ? _expand_pd(node, registry) : node
    end
end

"""Expand a single TParamDeriv node by peeling off the last parameter."""
function _expand_pd(d::TParamDeriv, reg::TensorRegistry)
    isempty(d.params) && return d.arg

    # Peel off the last parameter
    last_p = d.params[end]
    outer_params = d.params[1:end-1]

    inner_result = _expand_pd_single(last_p, d.arg, reg)

    # Apply remaining params (if any)
    if isempty(outer_params)
        return inner_result
    else
        return param_deriv(outer_params, inner_result)
    end
end

"""Expand d/dp(arg) for a single parameter p."""
function _expand_pd_single(p::Symbol, arg::TensorExpr, reg::TensorRegistry)
    # Scalars: zero on rational constants
    if arg isa TScalar
        if arg.val isa Rational || arg.val isa Integer
            return TScalar(0 // 1)
        end
        # Symbolic scalar: d/dp(s) — keep as TParamDeriv
        return param_deriv(p, arg)
    end

    # Parameter self-derivative: d/dt(t) = 1
    if arg isa Tensor && is_parameter(reg, arg.name) && arg.name == p
        return TScalar(1 // 1)
    end

    # Tensor: keep as derivative (unless it doesn't depend on p)
    if arg isa Tensor
        return param_deriv(p, arg)
    end

    # Sum: linearity
    if arg isa TSum
        return tsum(TensorExpr[_expand_pd_single(p, t, reg) for t in arg.terms])
    end

    # Product: Leibniz rule
    if arg isa TProduct
        return _leibniz_param(p, arg, reg)
    end

    # TDeriv: commute (param deriv commutes with partial/covd)
    if arg isa TDeriv
        inner = _expand_pd_single(p, arg.arg, reg)
        return TDeriv(arg.index, inner, arg.covd)
    end

    # Nested TParamDeriv: flatten handled by smart constructor
    if arg isa TParamDeriv
        return param_deriv(vcat(Symbol[p], arg.params), arg.arg)
    end

    # Fallback: keep as is
    param_deriv(p, arg)
end

"""Apply Leibniz rule: d/dp(s * f1 * f2 * ...) = s * Σ_i (f1 * ... * d/dp(fi) * ... * fn)."""
function _leibniz_param(p::Symbol, prod::TProduct, reg::TensorRegistry)
    n = length(prod.factors)
    terms = TensorExpr[]

    for i in 1:n
        # Differentiate the i-th factor, keep all others
        df_i = _expand_pd_single(p, prod.factors[i], reg)
        # Skip zero terms
        if df_i == TScalar(0 // 1)
            continue
        end
        new_factors = TensorExpr[j == i ? df_i : prod.factors[j] for j in 1:n]
        push!(terms, tproduct(prod.scalar, new_factors))
    end

    isempty(terms) && return TScalar(0 // 1)
    tsum(terms)
end
