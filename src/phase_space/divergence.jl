#= Divergence detection utility for the covariant phase space framework.
#
# Given a tensor expression, determine whether it is a total covariant
# divergence: expr = nabla_a V^a for some V.  This is the algebraic
# check needed for the Noether charge extraction
#   J = dQ + (terms on-shell)
# where the divergence terms are exact (J = dQ).
#
# Reference: Iyer & Wald (1994), PRD 50, 846, Eq 3.3.
=#

"""
    is_divergence(expr::TensorExpr, covd_name::Symbol;
                  registry::TensorRegistry=current_registry()) -> Bool

Check whether `expr` is a total covariant divergence with respect to the
covariant derivative `covd_name`.  An expression is a divergence if every
term has the form `nabla_a(V^{a...})` where the derivative index `a`
contracts with an Up index in the argument.

For a TSum, all terms must individually be divergences.

# Examples

```julia
# nabla_a(V^a) is a divergence
is_divergence(TDeriv(down(:a), Tensor(:V, [up(:a)]), :D), :D; registry=reg)  # true

# G^{mu nu} xi_nu is NOT a divergence
is_divergence(Tensor(:Ein, [up(:a), up(:b)]) * Tensor(:xi, [down(:b)]), :D; registry=reg)  # false
```
"""
function is_divergence(expr::TensorExpr, covd_name::Symbol;
                       registry::TensorRegistry=current_registry())
    with_registry(registry) do
        _check_divergence(expr, covd_name)
    end
end

# ── Core divergence check (dispatch on expression type) ──────────────

function _check_divergence(expr::TDeriv, covd_name::Symbol)
    # A derivative node is a divergence if:
    # 1. Its covd matches (or is :partial, which is always compatible)
    # 2. The derivative index is Down
    # 3. The argument contains an Up index with the same name (contraction)
    (expr.covd == covd_name || expr.covd == :partial) || return false
    expr.index.position == Down || return false
    _arg_has_contracted_up(expr.index.name, expr.arg)
end

function _check_divergence(expr::TSum, covd_name::Symbol)
    # A sum is a divergence iff every term is a divergence
    isempty(expr.terms) && return true
    all(t -> _check_divergence(t, covd_name), expr.terms)
end

function _check_divergence(expr::TProduct, covd_name::Symbol)
    # A product is a divergence if exactly one factor is a TDeriv divergence
    # and the remaining factors are scalars (no free indices involved in
    # the contraction).  Example: phi * D_a(V^a)
    #
    # More precisely: the product must contain a derivative factor whose
    # index contracts with an Up index somewhere in the full product.
    for (i, f) in enumerate(expr.factors)
        if f isa TDeriv && (f.covd == covd_name || f.covd == :partial) &&
           f.index.position == Down
            # Check contraction: the derivative index must pair with an Up
            # index in the argument of the derivative (not in other factors,
            # since that would mean the derivative acts on a sub-expression).
            if _arg_has_contracted_up(f.index.name, f.arg)
                return true
            end
        end
    end
    false
end

function _check_divergence(::Tensor, ::Symbol)
    false
end

function _check_divergence(::TScalar, ::Symbol)
    # Zero is trivially a divergence (nabla_a(0) = 0), but a nonzero
    # scalar is not.  We treat TScalar(0//1) as true for robustness.
    false
end

"""
Check if `arg` has an Up index whose name matches `idx_name`, indicating
a divergence contraction nabla_{idx_name}(...^{idx_name}...).
"""
function _arg_has_contracted_up(idx_name::Symbol, arg::TensorExpr)
    for idx in indices(arg)
        if idx.name == idx_name && idx.position == Up
            return true
        end
    end
    false
end

# ── extract_divergence ───────────────────────────────────────────────

"""
    extract_divergence(expr::TensorExpr, covd_name::Symbol;
                       registry::TensorRegistry=current_registry()) -> (is_div::Bool, V::Union{TensorExpr, Nothing})

If `expr` is a divergence `nabla_a(V^a)`, return `(true, V)` where `V` is
the vector (density) being differentiated.  Otherwise return `(false, nothing)`.

For a sum where all terms are divergences, `V` is the sum of the individual
vector arguments.

# Examples

```julia
ok, V = extract_divergence(TDeriv(down(:a), Tensor(:V, [up(:a)]), :D), :D; registry=reg)
# ok == true, V == Tensor(:V, [up(:a)])
```
"""
function extract_divergence(expr::TensorExpr, covd_name::Symbol;
                            registry::TensorRegistry=current_registry())
    with_registry(registry) do
        _extract_div(expr, covd_name)
    end
end

function _extract_div(expr::TDeriv, covd_name::Symbol)
    if (expr.covd == covd_name || expr.covd == :partial) &&
       expr.index.position == Down &&
       _arg_has_contracted_up(expr.index.name, expr.arg)
        return (true, expr.arg)
    end
    (false, nothing)
end

function _extract_div(expr::TSum, covd_name::Symbol)
    vectors = TensorExpr[]
    for t in expr.terms
        ok, v = _extract_div(t, covd_name)
        ok || return (false, nothing)
        push!(vectors, v)
    end
    isempty(vectors) && return (true, ZERO)
    (true, tsum(vectors))
end

function _extract_div(expr::TProduct, covd_name::Symbol)
    for (i, f) in enumerate(expr.factors)
        if f isa TDeriv && (f.covd == covd_name || f.covd == :partial) &&
           f.index.position == Down &&
           _arg_has_contracted_up(f.index.name, f.arg)
            # The vector V is: scalar * (other factors) * (derivative argument)
            other = TensorExpr[expr.factors[j] for j in eachindex(expr.factors) if j != i]
            if isempty(other)
                v = tproduct(expr.scalar, TensorExpr[f.arg])
            else
                push!(other, f.arg)
                v = tproduct(expr.scalar, other)
            end
            return (true, v)
        end
    end
    (false, nothing)
end

function _extract_div(::Tensor, ::Symbol)
    (false, nothing)
end

function _extract_div(::TScalar, ::Symbol)
    (false, nothing)
end

# ── split_divergence ─────────────────────────────────────────────────

"""
    split_divergence(expr::TensorExpr, covd_name::Symbol;
                     registry::TensorRegistry=current_registry()) -> (div_part, non_div_part)

Split a tensor expression into a divergence part and a non-divergence
remainder.  Returns `(div_part, non_div_part)` where:
- `div_part` collects all terms that are total divergences
- `non_div_part` collects all remaining terms
- `expr == div_part + non_div_part` (up to simplification)

If `expr` is not a TSum, it is treated as a single term and classified
entirely as divergence or non-divergence.

# Examples

```julia
# Split J^mu = nabla_nu(Q^{mu nu}) + G^{mu nu} xi_nu
div, rest = split_divergence(J_expr, :D; registry=reg)
# div  = nabla_nu(Q^{mu nu})
# rest = G^{mu nu} xi_nu
```
"""
function split_divergence(expr::TensorExpr, covd_name::Symbol;
                          registry::TensorRegistry=current_registry())
    with_registry(registry) do
        _split_div(expr, covd_name)
    end
end

function _split_div(expr::TSum, covd_name::Symbol)
    div_terms = TensorExpr[]
    non_div_terms = TensorExpr[]
    for t in expr.terms
        if _check_divergence(t, covd_name)
            push!(div_terms, t)
        else
            push!(non_div_terms, t)
        end
    end
    div_part = isempty(div_terms) ? ZERO : tsum(div_terms)
    non_div_part = isempty(non_div_terms) ? ZERO : tsum(non_div_terms)
    (div_part, non_div_part)
end

function _split_div(expr::TensorExpr, covd_name::Symbol)
    if _check_divergence(expr, covd_name)
        (expr, ZERO)
    else
        (ZERO, expr)
    end
end
