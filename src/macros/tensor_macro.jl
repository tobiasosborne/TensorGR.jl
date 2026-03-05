#= @tensor macro: compile-time translation from LaTeX-like notation to typed AST.

Syntax:
    @tensor R[-a, -b, -c, -d]           → Tensor(:R, [down(:a), ...])
    @tensor g[a, b] * R[-a, -b]         → g * R  (with operator overloads)
    @tensor (1//2) * R[-a, -b]          → (1//2) * R
    @tensor ∂[-a](T[b, -c])             → TDeriv(down(:a), T)
    @tensor R[-a, -b] + g[-a, -b]       → R + g
    @tensor -R[-a, -b]                  → -R
    @tensor Φ                           → Tensor(:Φ, TIndex[])

The macro transforms Julia Expr nodes into calls to TensorGR constructors.
All algebraic operations use the Base.* / Base.+ overloads, which handle
flattening, scalar absorption, and dummy-clash resolution.
=#

"""
    @tensor expr

Construct a TensorExpr from LaTeX-like notation.
"""
macro tensor(expr)
    esc(_translate(expr))
end

# ─── Recursive translator ────────────────────────────────────────────

function _translate(ex)
    if ex isa Symbol
        # Bare symbol: scalar-like tensor with no indices (e.g., Φ, RicciScalar)
        return :(Tensor($(QuoteNode(ex)), TIndex[]))
    elseif ex isa Number
        return :(TScalar($ex))
    elseif ex isa Expr
        return _translate_expr(ex)
    else
        error("@tensor: unexpected expression $ex")
    end
end

function _translate_expr(ex::Expr)
    if ex.head == :ref
        # T[a, -b, c] → Tensor(:T, [up(:a), down(:b), up(:c)])
        return _translate_ref(ex)

    elseif ex.head == :call
        return _translate_call(ex)

    elseif ex.head == ://
        # Literal rational: 1//2
        return ex

    else
        error("@tensor: unsupported expression head :$(ex.head) in $ex")
    end
end

function _translate_ref(ex::Expr)
    name = ex.args[1]
    name isa Symbol || error("@tensor: tensor name must be a symbol, got $name")
    idx_exprs = ex.args[2:end]
    indices = [_translate_index(i) for i in idx_exprs]
    :(Tensor($(QuoteNode(name)), TIndex[$(indices...)]))
end

function _translate_index(ex)
    if ex isa Symbol
        # Bare symbol: up index
        return :(up($(QuoteNode(ex))))
    elseif ex isa Expr && ex.head == :call && ex.args[1] == :(-) && length(ex.args) == 2
        # Negated symbol: -a → down(:a)
        inner = ex.args[2]
        inner isa Symbol || error("@tensor: expected symbol after -, got $inner")
        return :(down($(QuoteNode(inner))))
    else
        error("@tensor: invalid index expression $ex")
    end
end

function _translate_call(ex::Expr)
    op = ex.args[1]

    # Derivative: ∂[-a](body) — op is Expr(:ref, :∂, ...)
    if op isa Expr && op.head == :ref && op.args[1] == :∂
        return _translate_deriv(ex)

    elseif op == :(*) && length(ex.args) >= 3
        # a * b * c — left-fold multiplication
        result = _translate(ex.args[2])
        for i in 3:length(ex.args)
            rhs = _translate(ex.args[i])
            result = :($result * $rhs)
        end
        return result

    elseif op == :(+) && length(ex.args) >= 3
        result = _translate(ex.args[2])
        for i in 3:length(ex.args)
            rhs = _translate(ex.args[i])
            result = :($result + $rhs)
        end
        return result

    elseif op == :(-) && length(ex.args) == 2
        # Unary minus
        return :(-$(_translate(ex.args[2])))

    elseif op == :(-) && length(ex.args) == 3
        # Binary minus
        return :($(_translate(ex.args[2])) - $(_translate(ex.args[3])))

    elseif op == :(//)
        # Rational literal inside call: 1//2
        return ex

    else
        # Could be a function call on tensors; try translating arguments
        error("@tensor: unsupported call $op in $ex")
    end
end

function _translate_deriv(ex::Expr)
    # Expected form: ∂[-a](body)  which Julia parses as:
    #   Expr(:call, :∂, Expr(:call, :-, :a), body)
    # or ∂(idx_expr)(body) which Julia parses differently.
    #
    # Actually Julia parses `∂[-a](T[b])` as:
    #   Expr(:call, Expr(:ref, :∂, Expr(:call, :-, :a)), Expr(:ref, :T, :b))
    # i.e., (∂[-a]) applied to (T[b])
    #
    # But in our macro, `ex` is the full call. Let's handle both patterns.

    if ex.args[1] isa Expr && ex.args[1].head == :ref
        # ∂[-a](body): ex.args[1] = ∂[-a], ex.args[2] = body
        deriv_ref = ex.args[1]
        deriv_ref.args[1] == :∂ || error("@tensor: expected ∂, got $(deriv_ref.args[1])")
        idx = _translate_index(deriv_ref.args[2])
        body = _translate(ex.args[2])
        return :(TDeriv($idx, $body))
    else
        error("@tensor: unsupported derivative syntax in $ex")
    end
end
