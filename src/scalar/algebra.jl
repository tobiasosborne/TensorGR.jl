#= Scalar algebra engine for Expr trees and Rational{Int}.

Provides polynomial/rational arithmetic operations that work on Julia Expr
trees, enabling symbolic manipulation of scalar coefficients in tensor
expressions.
=#

"""
    scalar_expand(ex)

Distribute `*` over `+` in a scalar expression tree.
"""
scalar_expand(ex::Number) = ex
scalar_expand(ex::Symbol) = ex

function scalar_expand(ex::Expr)
    if ex.head == :call
        op = ex.args[1]
        args = [scalar_expand(a) for a in ex.args[2:end]]

        if op == :*
            return _scalar_expand_product(args)
        elseif op == :+
            return _scalar_make_sum(args)
        elseif op == :-
            if length(args) == 1
                inner = args[1]
                if _is_sum(inner)
                    # -(a + b) → (-a) + (-b)
                    terms = _sum_terms(inner)
                    return _scalar_make_sum([_scalar_negate(t) for t in terms])
                end
                return _scalar_negate(inner)
            else
                return scalar_expand(Expr(:call, :+, args[1], _scalar_negate(args[2])))
            end
        else
            return Expr(:call, op, args...)
        end
    elseif ex.head == :(//)
        return Rational{Int}(ex.args[1], ex.args[2])
    end
    ex
end

function _scalar_expand_product(factors)
    # Find a sum factor and distribute
    for (i, f) in enumerate(factors)
        if _is_sum(f)
            terms = _sum_terms(f)
            others = [factors[j] for j in eachindex(factors) if j != i]
            expanded = [_scalar_expand_product(vcat(others, [t])) for t in terms]
            return _scalar_make_sum(expanded)
        end
    end
    # No sums found, just build product
    _scalar_make_product(factors)
end

function _is_sum(ex)
    ex isa Expr && ex.head == :call && length(ex.args) >= 3 && ex.args[1] == :+
end

function _sum_terms(ex::Expr)
    ex.args[2:end]
end

function _scalar_negate(ex::Number)
    -ex
end
function _scalar_negate(ex::Symbol)
    Expr(:call, :*, -1, ex)
end
function _scalar_negate(ex::Expr)
    if ex.head == :call && ex.args[1] == :* && ex.args[2] isa Number
        Expr(:call, :*, -ex.args[2], ex.args[3:end]...)
    else
        Expr(:call, :*, -1, ex)
    end
end

function _scalar_make_sum(terms)
    flat = Any[]
    for t in terms
        if _is_sum(t)
            append!(flat, _sum_terms(t))
        elseif t == 0
            continue
        else
            push!(flat, t)
        end
    end
    isempty(flat) && return 0
    length(flat) == 1 && return flat[1]
    Expr(:call, :+, flat...)
end

function _scalar_make_product(factors)
    flat = Any[]
    coeff = 1 // 1
    for f in factors
        if f isa Number
            coeff *= f
        elseif f isa Expr && f.head == :call && f.args[1] == :* && f.args[2] isa Number
            coeff *= f.args[2]
            append!(flat, f.args[3:end])
        else
            push!(flat, f)
        end
    end
    coeff == 0 && return 0
    if isempty(flat)
        return _simplify_number(coeff)
    end
    if coeff == 1
        length(flat) == 1 && return flat[1]
        return Expr(:call, :*, flat...)
    end
    Expr(:call, :*, _simplify_number(coeff), flat...)
end

_simplify_number(x::Rational{Int}) = isinteger(x) ? Int(x) : x
_simplify_number(x) = x

"""
    scalar_collect(ex, var::Symbol)

Collect powers of `var` in a sum expression.
Returns a Dict mapping power => coefficient.
"""
function scalar_collect(ex, var::Symbol)
    terms = _is_sum(scalar_expand(ex)) ? _sum_terms(scalar_expand(ex)) : [scalar_expand(ex)]
    result = Dict{Int, Any}()
    for term in terms
        power, coeff = _extract_power(term, var)
        if haskey(result, power)
            result[power] = _scalar_make_sum([result[power], coeff])
        else
            result[power] = coeff
        end
    end
    result
end

function _extract_power(term, var::Symbol)
    if term == var
        return (1, 1)
    elseif term isa Expr && term.head == :call
        op = term.args[1]
        if op == :^  && term.args[2] == var && term.args[3] isa Integer
            return (term.args[3], 1)
        elseif op == :*
            total_power = 0
            remaining = Any[]
            for a in term.args[2:end]
                p, c = _extract_power(a, var)
                total_power += p
                c != 1 && push!(remaining, c)
            end
            coeff = isempty(remaining) ? 1 : _scalar_make_product(remaining)
            return (total_power, coeff)
        end
    end
    return (0, term)
end

"""
    scalar_subst(ex, rules::Dict{Symbol, Any})

Substitute symbols in an expression tree.
"""
scalar_subst(ex::Number, ::Dict) = ex
function scalar_subst(ex::Symbol, rules::Dict)
    haskey(rules, ex) ? rules[ex] : ex
end
function scalar_subst(ex::Expr, rules::Dict)
    if ex.head == :call
        new_args = [ex.args[1]; [scalar_subst(a, rules) for a in ex.args[2:end]]]
        return Expr(:call, new_args...)
    elseif ex.head == :(//)
        return Expr(ex.head, [scalar_subst(a, rules) for a in ex.args]...)
    end
    ex
end

"""
    scalar_cancel(ex)

Cancel common factors in a ratio `a / b` or simplify numeric rationals.
"""
scalar_cancel(ex::Number) = ex
scalar_cancel(ex::Symbol) = ex
function scalar_cancel(ex::Expr)
    if ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3
        num = scalar_cancel(ex.args[2])
        den = scalar_cancel(ex.args[3])
        if num isa Number && den isa Number
            return num // den
        end
        return Expr(:call, :/, num, den)
    end
    ex
end
