module TensorGRSymEngineExt

using TensorGR
using SymEngine

"""
    to_symengine(s::TScalar) -> SymEngine.Basic

Convert a TScalar to a SymEngine symbolic expression.
"""
function TensorGR.to_symengine(s::TScalar)
    val = s.val
    if val isa Number
        return SymEngine.Basic(val)
    elseif val isa Symbol
        return SymEngine.symbols(String(val))
    elseif val isa Expr
        return _expr_to_symengine(val)
    end
    error("Cannot convert TScalar($val) to SymEngine")
end

function _expr_to_symengine(ex::Expr)
    if ex.head == :call
        op = ex.args[1]
        args = [_expr_to_symengine(a) for a in ex.args[2:end]]
        if op == :+
            return sum(args)
        elseif op == :-
            return length(args) == 1 ? -args[1] : args[1] - args[2]
        elseif op == :*
            return prod(args)
        elseif op == :/
            return args[1] / args[2]
        elseif op == :^
            return args[1] ^ args[2]
        end
    end
    error("Cannot convert expression: $ex")
end

_expr_to_symengine(x::Number) = SymEngine.Basic(x)
_expr_to_symengine(x::Symbol) = SymEngine.symbols(String(x))

"""
    from_symengine(expr::SymEngine.Basic) -> TScalar

Convert a SymEngine expression to a TScalar.
"""
function TensorGR.from_symengine(expr)
    TScalar(Meta.parse(string(expr)))
end

end # module
