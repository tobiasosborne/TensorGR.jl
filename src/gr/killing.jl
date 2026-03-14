#= Killing vector fields.

A Killing vector ξ satisfies ∇_{(a} ξ_{b)} = 0, equivalently
Lie_xi g_{ab} = 0.

The Killing equation is automatically registered as rewrite rules when a
Killing vector is defined via `define_killing!`. The rules enforce that the
symmetrized covariant derivative of the Killing field vanishes:

    ∇_a ξ_b + ∇_b ξ_a = 0   =>   ∇_a ξ_b = -∇_b ξ_a

This is encoded as pattern-matching rules on TDeriv expressions acting on the
lowered Killing vector (ξ_b with a down index).
=#

"""
    define_killing!(reg, name; manifold, metric, covd=nothing) -> TensorProperties

Define a Killing vector field and automatically register the Killing equation
as rewrite rules. Registers the vector with an `:is_killing` flag and records
the associated metric.

The Killing equation ∇_{(a} ξ_{b)} = 0 is registered as a rule that rewrites
`∇_a ξ_b + ∇_b ξ_a` to zero. Specifically, we register:

1. A rule that detects the symmetric part of `∇ξ` in a TSum and eliminates it.

If `covd` is provided, the rules apply to that specific covariant derivative.
Otherwise, rules apply to `:partial` (untagged) derivatives.
"""
function define_killing!(reg::TensorRegistry, name::Symbol;
                         manifold::Symbol, metric::Symbol,
                         covd::Union{Symbol,Nothing}=nothing)
    # Register the vector field
    if !has_tensor(reg, name)
        register_tensor!(reg, TensorProperties(
            name=name, manifold=manifold, rank=(1, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_killing => true,
                                     :metric => metric,
                                     :killing_covd => covd)))
    end

    # Register Killing equation rules: ∇_a ξ_b + ∇_b ξ_a = 0
    # This means: ∇_a ξ_b is antisymmetric, i.e. ∇_a ξ_b = -∇_b ξ_a
    #
    # We register two pattern-based rules using make_rule:
    #
    # Rule 1: ∇_{a_} ξ_{b_} + ∇_{b_} ξ_{a_} => 0
    #   This catches the full Killing equation in a TSum.
    #
    # Rule 2: A functional rule that detects ∇_a ξ_b where ξ has a down
    #   (lowered) index, and canonicalizes it so that the derivative index
    #   is "smaller" than the field index (antisymmetry).
    _covd_sym = covd === nothing ? :partial : covd
    _field_name = name

    # Rule: the symmetrized covariant derivative vanishes.
    # Pattern: ∇_{a_} ξ_{b_} (with pattern indices) => will be paired via TSum.
    # We use a functional rule on TSum to detect and cancel symmetric pairs.
    register_rule!(reg, RewriteRule(
        function(expr)
            # Match a TSum containing ∇_a ξ_b + ∇_b ξ_a (or scalar multiples)
            expr isa TSum || return false
            _has_killing_symmetric_pair(expr, _field_name, _covd_sym)
        end,
        function(expr)
            _cancel_killing_symmetric(expr, _field_name, _covd_sym)
        end
    ))

    get_tensor(reg, name)
end

"""
    _is_covd_of_killing(expr, field_name, covd_sym) -> Union{Nothing, Tuple{TIndex, TIndex, Rational}}

Check if `expr` is of the form `scalar * ∇_a ξ_b` where ξ is the Killing field.
Returns `(deriv_index, field_index, scalar)` if matched, `nothing` otherwise.
"""
function _is_covd_of_killing(expr::TensorExpr, field_name::Symbol, covd_sym::Symbol)
    scalar = 1 // 1
    core = expr
    if expr isa TProduct
        scalar = expr.scalar
        length(expr.factors) == 1 || return nothing
        core = expr.factors[1]
    end
    core isa TDeriv || return nothing
    core.covd == covd_sym || return nothing
    inner = core.arg
    inner isa Tensor || return nothing
    inner.name == field_name || return nothing
    length(inner.indices) == 1 || return nothing
    inner.indices[1].position == Down || return nothing
    (core.index, inner.indices[1], scalar)
end

"""
    _has_killing_symmetric_pair(s::TSum, field_name, covd_sym) -> Bool

Check if a TSum contains a symmetric pair ∇_a ξ_b + ∇_b ξ_a for a Killing field.
"""
function _has_killing_symmetric_pair(s::TSum, field_name::Symbol, covd_sym::Symbol)
    n = length(s.terms)
    for i in 1:n
        mi = _is_covd_of_killing(s.terms[i], field_name, covd_sym)
        mi === nothing && continue
        (di, fi, si) = mi
        for j in (i+1):n
            mj = _is_covd_of_killing(s.terms[j], field_name, covd_sym)
            mj === nothing && continue
            (dj, fj, sj) = mj
            # Check for ∇_a ξ_b + ∇_b ξ_a pattern (same scalar coefficient)
            if di == fj && dj == fi && si == sj
                return true
            end
        end
    end
    false
end

"""
    _cancel_killing_symmetric(s::TSum, field_name, covd_sym) -> TensorExpr

Cancel symmetric pairs ∇_a ξ_b + ∇_b ξ_a in a TSum.
"""
function _cancel_killing_symmetric(s::TSum, field_name::Symbol, covd_sym::Symbol)
    terms = collect(s.terms)
    cancelled = Set{Int}()

    for i in 1:length(terms)
        i in cancelled && continue
        mi = _is_covd_of_killing(terms[i], field_name, covd_sym)
        mi === nothing && continue
        (di, fi, si) = mi
        for j in (i+1):length(terms)
            j in cancelled && continue
            mj = _is_covd_of_killing(terms[j], field_name, covd_sym)
            mj === nothing && continue
            (dj, fj, sj) = mj
            if di == fj && dj == fi && si == sj
                push!(cancelled, i)
                push!(cancelled, j)
                break
            end
        end
    end

    remaining = TensorExpr[terms[i] for i in 1:length(terms) if !(i in cancelled)]
    isempty(remaining) ? ZERO : tsum(remaining)
end

"""
    check_killing(reg, field, expr) -> TensorExpr

Simplify an expression using the Killing equation rules registered for `field`.
This calls `simplify` with the given registry, which applies all registered rules
including the Killing equation.

# Example
```julia
reg = TensorRegistry()
# ... setup manifold, metric, covd ...
define_killing!(reg, :xi; manifold=:M4, metric=:g, covd=:D)
expr = TDeriv(down(:a), Tensor(:xi, [down(:b)]), :D) +
       TDeriv(down(:b), Tensor(:xi, [down(:a)]), :D)
result = check_killing(reg, :xi, expr)  # => 0
```
"""
function check_killing(reg::TensorRegistry, field::Symbol, expr::TensorExpr)
    has_tensor(reg, field) || error("Tensor $field not registered")
    props = get_tensor(reg, field)
    get(props.options, :is_killing, false) ||
        error("$field is not a Killing vector field")
    simplify(expr; registry=reg)
end
