module TensorGRSymbolicsExt

using TensorGR
using Symbolics

"""
    to_symbolics(s::TScalar) -> Symbolics.Num

Convert a TScalar to a Symbolics.jl symbolic number.
"""
function TensorGR.to_symbolics(s::TScalar)
    val = s.val
    if val isa Number
        return Symbolics.Num(val)
    elseif val isa Symbol
        return first(Symbolics.@variables $val)
    elseif val isa Expr
        return _expr_to_symbolics(val)
    end
    error("Cannot convert TScalar($val) to Symbolics")
end

function _expr_to_symbolics(ex::Expr)
    if ex.head == :call
        op = ex.args[1]
        args = [_expr_to_symbolics(a) for a in ex.args[2:end]]
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

_expr_to_symbolics(x::Number) = Symbolics.Num(x)
function _expr_to_symbolics(x::Symbol)
    first(Symbolics.@variables $x)
end

"""
    from_symbolics(num::Symbolics.Num) -> TScalar

Convert a Symbolics.jl number to a TScalar.
"""
function TensorGR.from_symbolics(num)
    TScalar(Symbolics.toexpr(num))
end

end # module
