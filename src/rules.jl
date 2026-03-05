#= Pattern-matching rewrite rules.

Provides the Julia equivalent of xAct's UpValues/DownValues system:
- RewriteRule: pattern → replacement with optional condition
- apply_rules: single-pass rule application
- apply_rules_fixpoint: iterate to fixed point
- AutomaticRules on TensorRegistry: rules triggered by tensor name
=#

"""
    RewriteRule(pattern, replacement, condition)

A rewrite rule that transforms expressions matching `pattern` into `replacement`.
The optional `condition` function takes the matched expression and returns a Bool.

Pattern matching is structural: the pattern is compared to sub-expressions via `==`.
For more flexible matching, supply a custom `match` function as the pattern.
"""
struct RewriteRule
    pattern::Any          # TensorExpr or Function(expr) -> Bool
    replacement::Any      # TensorExpr or Function(expr) -> TensorExpr
    condition::Function   # (expr) -> Bool, default always true
end

RewriteRule(pat, rep) = RewriteRule(pat, rep, _ -> true)

function _rule_matches(rule::RewriteRule, expr::TensorExpr)
    if rule.pattern isa Function
        return rule.pattern(expr) && rule.condition(expr)
    else
        return expr == rule.pattern && rule.condition(expr)
    end
end

function _rule_apply(rule::RewriteRule, expr::TensorExpr)
    if rule.replacement isa Function
        return rule.replacement(expr)
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
    rebuilt = TDeriv(expr.index, new_arg)
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
