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
    @rule pattern => replacement

Create a RewriteRule from a pair. Both sides must be TensorExpr values.

    @rule pattern => replacement when condition

Create a conditional RewriteRule.
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
