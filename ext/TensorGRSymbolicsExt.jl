module TensorGRSymbolicsExt

using TensorGR
using Symbolics

# ─── to_symbolics / from_symbolics ───────────────────────────────

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
        # Normalize: Symbolics.toexpr can produce (*)(a,b) where op is the function itself
        op_sym = op isa Symbol ? op : op isa Function ? Symbol(op) : nameof(op)
        args = [_expr_to_symbolics(a) for a in ex.args[2:end]]
        if op_sym == :+ || op == +
            return sum(args)
        elseif op_sym == :- || op == -
            return length(args) == 1 ? -args[1] : args[1] - args[2]
        elseif op_sym == :* || op == *
            return prod(args)
        elseif op_sym == :/ || op == /
            return args[1] / args[2]
        elseif op_sym == :^ || op == ^
            return args[1] ^ args[2]
        elseif op_sym == :// || op == //
            return Rational{Int}(args[1], args[2])
        end
        # Try calling the function directly on Symbolics args
        if op isa Function
            return op(args...)
        end
    elseif ex.head == :(//)
        return _expr_to_symbolics(ex.args[1]) // _expr_to_symbolics(ex.args[2])
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

# ─── CAS-1: simplify hooks ───────────────────────────────────────

function TensorGR._simplify_scalar_val(ex::Expr)
    try
        sym = _expr_to_symbolics(ex)
        simplified = Symbolics.simplify(sym)
        return Symbolics.toexpr(simplified)
    catch
        return ex
    end
end

function TensorGR._try_simplify_entry(ex::Expr)
    try
        sym = _expr_to_symbolics(ex)
        simplified = Symbolics.simplify(sym)
        return Symbolics.toexpr(simplified)
    catch
        return ex
    end
end

# ─── CAS-2: Symbolics.Num dispatch for sym arithmetic ────────────

TensorGR._sym_mul(a::Symbolics.Num, b::Symbolics.Num) = a * b
TensorGR._sym_mul(a::Symbolics.Num, b::Number) = a * b
TensorGR._sym_mul(a::Number, b::Symbolics.Num) = a * b
TensorGR._sym_mul(a::Symbolics.Num, b) = a * _to_num(b)
TensorGR._sym_mul(a, b::Symbolics.Num) = _to_num(a) * b

TensorGR._sym_add(a::Symbolics.Num, b::Symbolics.Num) = a + b
TensorGR._sym_add(a::Symbolics.Num, b::Number) = a + b
TensorGR._sym_add(a::Number, b::Symbolics.Num) = a + b
TensorGR._sym_add(a::Symbolics.Num, b) = a + _to_num(b)
TensorGR._sym_add(a, b::Symbolics.Num) = _to_num(a) + b

TensorGR._sym_sub(a::Symbolics.Num, b::Symbolics.Num) = a - b
TensorGR._sym_sub(a::Symbolics.Num, b::Number) = a - b
TensorGR._sym_sub(a::Number, b::Symbolics.Num) = a - b
TensorGR._sym_sub(a::Symbolics.Num, b) = a - _to_num(b)
TensorGR._sym_sub(a, b::Symbolics.Num) = _to_num(a) - b

TensorGR._sym_neg(a::Symbolics.Num) = -a

TensorGR._sym_div(a::Symbolics.Num, b::Symbolics.Num) = a / b
TensorGR._sym_div(a::Symbolics.Num, b::Number) = a / b
TensorGR._sym_div(a::Number, b::Symbolics.Num) = a / b
TensorGR._sym_div(a::Symbolics.Num, b) = a / _to_num(b)
TensorGR._sym_div(a, b::Symbolics.Num) = _to_num(a) / b

_to_num(x::Symbolics.Num) = x
_to_num(x::Number) = Symbolics.Num(x)
function _to_num(x::Expr)
    _expr_to_symbolics(x)
end
_to_num(x::Symbol) = first(Symbolics.@variables $x)

"""
    sym_eval(expr::Symbolics.Num, vars::Dict) -> Number

Evaluate a Symbolics expression by substituting variable values.
"""
function TensorGR.sym_eval(expr::Symbolics.Num, vars::Dict)
    sym_vars = Dict{Symbolics.Num, Any}()
    for (k, v) in vars
        sym_k = first(Symbolics.@variables $k)
        sym_vars[sym_k] = v
    end
    result = Symbolics.substitute(expr, sym_vars)
    # Extract the underlying numeric value
    Float64(Symbolics.value(result))
end

# ─── CAS-2: symbolic_quadratic_form ──────────────────────────────

function TensorGR.symbolic_quadratic_form(entries::AbstractDict, fields::Vector{Symbol};
                                           variables::Vector{Symbol}=Symbol[])
    # Create Symbolics variables
    sym_vars = Dict{Symbol, Symbolics.Num}()
    for v in variables
        sym_vars[v] = first(Symbolics.@variables $v)
    end

    n = length(fields)
    fidx = Dict(f => i for (i, f) in enumerate(fields))
    M = Matrix{Any}(undef, n, n)
    fill!(M, Symbolics.Num(0))

    for ((f1, f2), val) in entries
        i, j = fidx[f1], fidx[f2]
        sym_val = val isa Symbolics.Num ? val : _to_num(val)
        M[i, j] = sym_val
        M[j, i] = sym_val
    end
    QuadraticForm(fields, M)
end

# ─── CAS-3: Symbolic Fourier transform ───────────────────────────

function TensorGR.to_fourier_symbolic(expr::TensorExpr;
                                       omega::Symbolics.Num,
                                       k_vars::Vector{Symbolics.Num}=Symbolics.Num[])
    _fourier_symbolic(expr, omega, k_vars)
end

function _fourier_symbolic(t::Tensor, ::Symbolics.Num, ::Vector{Symbolics.Num})
    t
end

function _fourier_symbolic(s::TScalar, ::Symbolics.Num, ::Vector{Symbolics.Num})
    s
end

function _fourier_symbolic(s::TSum, omega::Symbolics.Num, k_vars::Vector{Symbolics.Num})
    TSum(TensorExpr[_fourier_symbolic(t, omega, k_vars) for t in s.terms])
end

function _fourier_symbolic(p::TProduct, omega::Symbolics.Num, k_vars::Vector{Symbolics.Num})
    TProduct(p.scalar, TensorExpr[_fourier_symbolic(f, omega, k_vars) for f in p.factors])
end

function _fourier_symbolic(d::TDeriv, omega::Symbolics.Num, k_vars::Vector{Symbolics.Num})
    inner = _fourier_symbolic(d.arg, omega, k_vars)

    # Check if temporal derivative (component 0)
    s = string(d.index.name)
    if startswith(s, "_") && length(s) > 1
        comp = tryparse(Int, s[2:end])
        if comp !== nothing && comp == 0
            # Temporal: ∂_0 → ω factor
            return TProduct(1 // 1, TensorExpr[TScalar(omega), inner])
        elseif comp !== nothing && comp >= 1 && comp <= length(k_vars)
            # Spatial: ∂_i → k_i factor
            return TProduct(1 // 1, TensorExpr[TScalar(k_vars[comp]), inner])
        end
    end

    # If not a component index, use standard momentum tensor
    k = Tensor(:k, [d.index])
    TProduct(1 // 1, TensorExpr[k, inner])
end

end # module
