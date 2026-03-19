#= Quadratic action analysis.

A quadratic Lagrangian L(Φ) = Φᵢ Mᵢⱼ Φⱼ defines a kinetic matrix M whose
inverse is the propagator.  This module extracts M from a TensorExpr
Lagrangian and provides tools for symbolic matrix operations.

The approach is direct numerical evaluation: given the Lagrangian as a
function of field components and momenta, evaluate M at random momentum
points to determine the propagator.  Symbolic expressions are verified
by checking agreement at many points.

QuadraticForm stores the matrix as a Julia Matrix of Julia functions
(or expressions) of the momentum variables.
=#

"""
    QuadraticForm

A quadratic form L = Φᵢ Mᵢⱼ(k) Φⱼ with momentum-dependent matrix.
The matrix entries are callable: `M[i,j](k², p², β, ...)`.
"""
struct QuadraticForm
    fields::Vector{Symbol}
    matrix::Matrix{Any}  # entries are numbers or Expr trees
end

Base.size(q::QuadraticForm) = size(q.matrix)

function Base.show(io::IO, q::QuadraticForm)
    n = length(q.fields)
    println(io, "QuadraticForm with fields: ", q.fields)
    for i in 1:n, j in i:n
        println(io, "  M[$(q.fields[i]),$(q.fields[j])] = ", q.matrix[i, j])
    end
end

"""
    quadratic_form(entries::Dict{Tuple{Symbol,Symbol}, Any}, fields::Vector{Symbol})

Construct a QuadraticForm from a dictionary mapping field pairs to expressions.
Automatically symmetrises: M[i,j] = M[j,i] for a Lagrangian.
"""
function quadratic_form(entries::AbstractDict, fields::Vector{Symbol})
    n = length(fields)
    fidx = Dict(f => i for (i, f) in enumerate(fields))
    M = fill(0, n, n) |> x -> convert(Matrix{Any}, x)

    for ((f1, f2), val) in entries
        i, j = fidx[f1], fidx[f2]
        M[i, j] = val
        M[j, i] = val  # symmetric
    end
    QuadraticForm(fields, M)
end

# ─── Symbolic 2×2 and 3×3 matrix operations ─────────────────────────

"""
    sym_det(M::Matrix) -> Any

Symbolic determinant via cofactor expansion along the first row.
Optimized fast paths for 1×1 through 3×3; general recursion for larger.
"""
function sym_det(M::Matrix)
    n = size(M, 1)
    n == size(M, 2) || error("sym_det requires square matrix")

    if n == 1
        return M[1, 1]
    elseif n == 2
        return _sym_sub(_sym_mul(M[1,1], M[2,2]),
                        _sym_mul(M[1,2], M[2,1]))
    elseif n == 3
        # Sarrus' rule
        a, b, c = M[1,1], M[1,2], M[1,3]
        d, e, f = M[2,1], M[2,2], M[2,3]
        g, h, k = M[3,1], M[3,2], M[3,3]
        return _sym_add(
            _sym_sub(_sym_mul(a, _sym_sub(_sym_mul(e,k), _sym_mul(f,h))),
                     _sym_mul(b, _sym_sub(_sym_mul(d,k), _sym_mul(f,g)))),
            _sym_mul(c, _sym_sub(_sym_mul(d,h), _sym_mul(e,g))))
    else
        # Cofactor expansion along first row
        result = 0
        for j in 1:n
            minor = _sym_minor(M, 1, j)
            cofactor = iseven(j + 1) ? minor : _sym_neg(minor)
            result = _sym_add(result, _sym_mul(M[1, j], cofactor))
        end
        return result
    end
end

"""Return determinant of the (n-1)×(n-1) submatrix with row i and column j removed."""
function _sym_minor(M::Matrix, i::Int, j::Int)
    n = size(M, 1)
    rows = [r for r in 1:n if r != i]
    cols = [c for c in 1:n if c != j]
    sym_det(M[rows, cols])
end

"""
    sym_inv(M::Matrix) -> Matrix

Symbolic inverse via adjugate/determinant. Optimized for 1×1 and 2×2;
general cofactor method for larger matrices.
"""
function sym_inv(M::Matrix)
    n = size(M, 1)
    det = sym_det(M)

    if n == 1
        return reshape([_sym_div(1, det)], 1, 1)
    elseif n == 2
        inv_det = _sym_div(1, det)
        return [_sym_mul(M[2,2], inv_det)   _sym_mul(_sym_neg(M[1,2]), inv_det);
                _sym_mul(_sym_neg(M[2,1]), inv_det)  _sym_mul(M[1,1], inv_det)]
    else
        # General adjugate: adj[i,j] = (-1)^(i+j) * minor(j,i)  (note transpose)
        adj = Matrix{Any}(undef, n, n)
        for i in 1:n, j in 1:n
            minor = _sym_minor(M, j, i)
            adj[i, j] = iseven(i + j) ? minor : _sym_neg(minor)
        end
        return [_sym_div(adj[i,j], det) for i in 1:n, j in 1:n]
    end
end

"""
    propagator(qf::QuadraticForm) -> QuadraticForm

Invert the quadratic form matrix to obtain the propagator.
"""
function propagator(qf::QuadraticForm)
    inv_M = sym_inv(qf.matrix)
    QuadraticForm(qf.fields, inv_M)
end

"""
    determinant(qf::QuadraticForm) -> Any

Compute the determinant of the quadratic form matrix.
"""
determinant(qf::QuadraticForm) = sym_det(qf.matrix)

# ─── Symbolic arithmetic helpers ─────────────────────────────────────
# Operate on numbers (exact Rational) or Expr trees.

_sym_mul(a::Number, b::Number) = a * b
_sym_mul(a, b) = a == 0 || b == 0 ? 0 : a == 1 ? b : b == 1 ? a : :($a * $b)

_sym_add(a::Number, b::Number) = a + b
function _sym_add(a, b)
    a == 0 && return b
    b == 0 && return a
    # Cancellation: x + (-x) = 0
    if b isa Expr && b.head == :call && b.args[1] == :- && length(b.args) == 2 && b.args[2] == a
        return 0
    end
    if a isa Expr && a.head == :call && a.args[1] == :- && length(a.args) == 2 && a.args[2] == b
        return 0
    end
    :($a + $b)
end

_sym_sub(a::Number, b::Number) = a - b
_sym_sub(a, b) = b == 0 ? a : a == 0 ? _sym_neg(b) : :($a - $b)

_sym_neg(a::Number) = -a
_sym_neg(a) = :(-$a)

_sym_div(a::Number, b::Number) = a // b
_sym_div(a, b) = a == 0 ? 0 : :($a / $b)

"""
    sym_eval(expr, vars::Dict{Symbol, <:Number}) -> Number

Evaluate a symbolic expression by substituting variable values.
"""
function sym_eval(expr::Number, ::Dict)
    Float64(expr)
end

function sym_eval(expr::Symbol, vars::Dict)
    haskey(vars, expr) || error("Undefined variable: $expr")
    Float64(vars[expr])
end

function sym_eval(expr::Expr, vars::Dict)
    if expr.head == :call
        op = expr.args[1]
        args = [sym_eval(a, vars) for a in expr.args[2:end]]
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
        elseif op == ://
            return args[1] / args[2]
        else
            error("Unknown operation: $op")
        end
    elseif expr.head == :(//)
        # Alternative representation of rational: Expr(://, 3, 4) = 3//4
        return sym_eval(expr.args[1], vars) / sym_eval(expr.args[2], vars)
    else
        error("Cannot evaluate expression: $expr")
    end
end
