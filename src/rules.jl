#= Pattern-matching rewrite rules.

Provides the Julia equivalent of xAct's UpValues/DownValues system:
- RewriteRule: pattern → replacement with optional condition
- apply_rules: single-pass rule application
- apply_rules_fixpoint: iterate to fixed point
- AutomaticRules on TensorRegistry: rules triggered by tensor name

Pattern indices: index names ending with `_` are pattern variables that match
any index. E.g., `Tensor(:T, [down(:a_), down(:b_)])` matches `T_{cd}` and
binds `:a_ => down(:c), :b_ => down(:d)`, which are substituted in the replacement.
=#

"""
    RewriteRule(pattern, replacement, condition)

A rewrite rule that transforms expressions matching `pattern` into `replacement`.
The optional `condition` function takes the matched expression and returns a Bool.

Pattern matching supports three modes:
- **Structural**: pattern compared via `==` (exact match)
- **Pattern indices**: index names ending in `_` are variables that match any index
- **Functional**: pattern is a `Function(expr) -> Bool` for custom matching

# Pattern Index Example
```julia
rule = RewriteRule(
    Tensor(:Ric, [down(:a_), down(:b_)]),
    TScalar(:Λ) * Tensor(:g, [down(:a_), down(:b_)])
)
apply_rules(Tensor(:Ric, [down(:c), down(:d)]), [rule])
# => Λ * g_{cd}
```
"""
struct RewriteRule
    pattern::Any          # TensorExpr or Function(expr) -> Bool
    replacement::Any      # TensorExpr or Function(expr) -> TensorExpr
    condition::Function   # (expr) -> Bool, default always true
    _has_patterns::Bool   # cached: does pattern contain pattern variables?
end

function RewriteRule(pat, rep, cond)
    has_pat = pat isa TensorExpr ? _has_pattern_variables(pat) : false
    RewriteRule(pat, rep, cond, has_pat)
end

RewriteRule(pat, rep) = RewriteRule(pat, rep, _ -> true)

# ─── Pattern variable support ────────────────────────────────────────

"""
    is_pattern_variable(idx::TIndex) -> Bool

Check if an index is a pattern variable (name ends with `_`).
Pattern variables match any index during rule application.
"""
is_pattern_variable(idx::TIndex) = endswith(string(idx.name), "_")

"""Bindings from pattern variables to matched indices."""
const PatternBindings = Dict{Symbol, TIndex}

"""Check if an expression contains any pattern variables."""
_has_pattern_variables(t::Tensor) = any(is_pattern_variable, t.indices)
_has_pattern_variables(::TScalar) = false
function _has_pattern_variables(p::TProduct)
    any(_has_pattern_variables, p.factors)
end
function _has_pattern_variables(s::TSum)
    any(_has_pattern_variables, s.terms)
end
function _has_pattern_variables(d::TDeriv)
    is_pattern_variable(d.index) || _has_pattern_variables(d.arg)
end

"""
    _unify(pattern, expr) -> Union{PatternBindings, Nothing}

Attempt to unify a pattern expression with a concrete expression.
Returns a binding map if successful, `nothing` otherwise.
Pattern variables (index names ending in `_`) match any index.
"""
function _unify(pattern::Tensor, expr::Tensor)
    pattern.name == expr.name || return nothing
    length(pattern.indices) == length(expr.indices) || return nothing
    bindings = PatternBindings()
    for (p_idx, e_idx) in zip(pattern.indices, expr.indices)
        if is_pattern_variable(p_idx)
            key = p_idx.name
            if haskey(bindings, key)
                bindings[key] == e_idx || return nothing  # consistency check
            else
                bindings[key] = e_idx
            end
        else
            p_idx == e_idx || return nothing
        end
    end
    bindings
end

function _unify(pattern::TScalar, expr::TScalar)
    pattern == expr ? PatternBindings() : nothing
end

function _unify(pattern::TProduct, expr::TProduct)
    pattern.scalar == expr.scalar || return nothing
    length(pattern.factors) == length(expr.factors) || return nothing
    bindings = PatternBindings()
    for (pf, ef) in zip(pattern.factors, expr.factors)
        b = _unify(pf, ef)
        b === nothing && return nothing
        _merge_bindings!(bindings, b) || return nothing
    end
    bindings
end

function _unify(pattern::TSum, expr::TSum)
    length(pattern.terms) == length(expr.terms) || return nothing
    bindings = PatternBindings()
    for (pt, et) in zip(pattern.terms, expr.terms)
        b = _unify(pt, et)
        b === nothing && return nothing
        _merge_bindings!(bindings, b) || return nothing
    end
    bindings
end

function _unify(pattern::TDeriv, expr::TDeriv)
    pattern.covd == expr.covd || return nothing
    bindings = PatternBindings()
    # Unify derivative index
    if is_pattern_variable(pattern.index)
        key = pattern.index.name
        if haskey(bindings, key)
            bindings[key] == expr.index || return nothing
        else
            bindings[key] = expr.index
        end
    else
        pattern.index == expr.index || return nothing
    end
    # Unify argument
    b_arg = _unify(pattern.arg, expr.arg)
    b_arg === nothing && return nothing
    _merge_bindings!(bindings, b_arg) || return nothing
    bindings
end

# Fallback: different types never unify
_unify(::TensorExpr, ::TensorExpr) = nothing

"""Merge new bindings into existing, returning false on conflict."""
function _merge_bindings!(into::PatternBindings, from::PatternBindings)
    for (k, v) in from
        if haskey(into, k)
            into[k] == v || return false
        else
            into[k] = v
        end
    end
    true
end

"""
    _substitute_bindings(expr, bindings) -> TensorExpr

Replace pattern variables in `expr` with their bound values.
"""
function _substitute_bindings(t::Tensor, bindings::PatternBindings)
    new_indices = map(t.indices) do idx
        if is_pattern_variable(idx)
            get(bindings, idx.name) do
                error("Unbound pattern variable: $(idx.name)")
            end
        else
            idx
        end
    end
    Tensor(t.name, new_indices)
end

_substitute_bindings(s::TScalar, ::PatternBindings) = s

function _substitute_bindings(p::TProduct, bindings::PatternBindings)
    tproduct(p.scalar, TensorExpr[_substitute_bindings(f, bindings) for f in p.factors])
end

function _substitute_bindings(s::TSum, bindings::PatternBindings)
    tsum(TensorExpr[_substitute_bindings(t, bindings) for t in s.terms])
end

function _substitute_bindings(d::TDeriv, bindings::PatternBindings)
    new_idx = if is_pattern_variable(d.index)
        get(bindings, d.index.name) do
            error("Unbound pattern variable: $(d.index.name)")
        end
    else
        d.index
    end
    TDeriv(new_idx, _substitute_bindings(d.arg, bindings), d.covd)
end

# ─── Rule matching and application ───────────────────────────────────

function _rule_matches(rule::RewriteRule, expr::TensorExpr)
    if rule.pattern isa Function
        return rule.pattern(expr) && rule.condition(expr)
    elseif rule._has_patterns
        bindings = _unify(rule.pattern, expr)
        return bindings !== nothing && rule.condition(expr)
    else
        return expr == rule.pattern && rule.condition(expr)
    end
end

function _rule_apply(rule::RewriteRule, expr::TensorExpr)
    if rule.replacement isa Function
        return rule.replacement(expr)
    elseif rule._has_patterns
        bindings = _unify(rule.pattern, expr)
        return _substitute_bindings(rule.replacement, bindings)
    else
        return rule.replacement
    end
end

"""
    apply_rules(expr, rules::Vector{RewriteRule}) -> TensorExpr

Apply rewrite rules in a single bottom-up pass. Each sub-expression is checked
against all rules; the first matching rule is applied.
"""
function apply_rules(expr::TensorExpr, rules::Vector{RewriteRule})
    _apply_rules_walk(expr, rules)
end

function _apply_rules_walk(expr::Tensor, rules)
    for r in rules
        _rule_matches(r, expr) && return _rule_apply(r, expr)
    end
    expr
end

function _apply_rules_walk(expr::TScalar, rules)
    for r in rules
        _rule_matches(r, expr) && return _rule_apply(r, expr)
    end
    expr
end

function _apply_rules_walk(expr::TProduct, rules)
    new_factors = TensorExpr[_apply_rules_walk(f, rules) for f in expr.factors]
    rebuilt = TProduct(expr.scalar, new_factors)
    for r in rules
        _rule_matches(r, rebuilt) && return _rule_apply(r, rebuilt)
    end
    rebuilt
end

function _apply_rules_walk(expr::TSum, rules)
    new_terms = TensorExpr[_apply_rules_walk(t, rules) for t in expr.terms]
    rebuilt = TSum(new_terms)
    for r in rules
        _rule_matches(r, rebuilt) && return _rule_apply(r, rebuilt)
    end
    rebuilt
end

function _apply_rules_walk(expr::TDeriv, rules)
    new_arg = _apply_rules_walk(expr.arg, rules)
    rebuilt = TDeriv(expr.index, new_arg, expr.covd)
    for r in rules
        _rule_matches(r, rebuilt) && return _rule_apply(r, rebuilt)
    end
    rebuilt
end

"""
    apply_rules_fixpoint(expr, rules; maxiter=100) -> TensorExpr

Apply rules repeatedly until the expression stops changing or `maxiter` is reached.
"""
function apply_rules_fixpoint(expr::TensorExpr, rules::Vector{RewriteRule}; maxiter::Int=100)
    current = expr
    for _ in 1:maxiter
        next = apply_rules(current, rules)
        next == current && return current
        current = next
    end
    current
end

"""
    make_rule(lhs, rhs; use_symmetries=false, registry=nothing) -> Vector{RewriteRule}

Create rewrite rules. If use_symmetries=true and the LHS is a tensor with known
symmetries, auto-generate rule variants for all symmetry-related index permutations.
"""
function make_rule(lhs::TensorExpr, rhs::TensorExpr;
                   use_symmetries::Bool=false,
                   registry::Union{TensorRegistry, Nothing}=nothing)
    rules = RewriteRule[RewriteRule(lhs, rhs)]

    if use_symmetries && lhs isa Tensor
        reg = registry !== nothing ? registry : current_registry()
        if has_tensor(reg, lhs.name)
            props = get_tensor(reg, lhs.name)
            nslots = length(lhs.indices)
            if !isempty(props.symmetries)
                gens = symmetry_generators(props.symmetries, nslots)
                # Generate all permutations reachable from generators
                perms = _generate_orbit(gens, nslots)
                for p in perms
                    new_indices = [lhs.indices[p.data[i]] for i in 1:nslots]
                    new_lhs = Tensor(lhs.name, new_indices)
                    new_lhs == lhs && continue
                    # Determine sign from the permutation
                    sign_idx = nslots + 2
                    sgn = p.data[sign_idx - 1] == sign_idx - 1 ? 1 : -1
                    if sgn == 1
                        push!(rules, RewriteRule(new_lhs, rhs))
                    else
                        push!(rules, RewriteRule(new_lhs, tproduct(-1 // 1, TensorExpr[rhs])))
                    end
                end
            end
        end
    end
    rules
end

function _generate_orbit(gens::Vector{Perm}, nslots::Int)
    n = nslots + 2
    identity = Perm(collect(Int32, 1:n))
    seen = Set{Vector{Int32}}()
    push!(seen, identity.data)
    queue = [identity]
    result = Perm[]

    while !isempty(queue)
        current = popfirst!(queue)
        for g in gens
            # Compose: new[i] = g[current[i]]
            # Compose permutations: result[i] = g(current(i))
            composed_data = Int32[g.data[current.data[i]] for i in 1:n]
            if composed_data ∉ seen
                push!(seen, composed_data)
                composed = Perm(composed_data)
                push!(result, composed)
                push!(queue, composed)
            end
        end
    end
    result
end

"""
    folded_rule(rules1::Vector{RewriteRule}, rules2::Vector{RewriteRule}) -> Function

Create a function that applies rules1 first, then rules2 to an expression.
"""
function folded_rule(rules1::Vector{RewriteRule}, rules2::Vector{RewriteRule})
    function(expr::TensorExpr)
        r = apply_rules(expr, rules1)
        apply_rules(r, rules2)
    end
end

"""
    @rule pattern => replacement

Create a RewriteRule from a pair. Both sides must be TensorExpr values.
"""
macro rule(ex)
    if ex.head == :call && ex.args[1] == :(=>)
        pat = ex.args[2]
        rep = ex.args[3]
        return esc(:(RewriteRule($pat, $rep)))
    elseif ex.head == :when || (ex.head == :call && length(ex.args) >= 4)
        error("@rule with `when` clause: use RewriteRule(pat, rep, cond) directly")
    else
        error("@rule expects `pattern => replacement`")
    end
end
