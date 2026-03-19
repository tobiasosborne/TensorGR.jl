# ── Scalar spherical harmonics Y_{lm} ─────────────────────────────────

"""
    ScalarHarmonic <: TensorExpr

Represents a scalar spherical harmonic Y_{lm} with quantum numbers l, m.
Satisfies: l >= 0, |m| <= l.
Eigenvalue of angular Laplacian on S^2: Delta_{S^2} Y_{lm} = -l(l+1) Y_{lm}.
Conjugation: Y*_{lm} = (-1)^m Y_{l,-m}.
"""
struct ScalarHarmonic <: TensorExpr
    l::Int
    m::Int
    function ScalarHarmonic(l::Int, m::Int)
        l >= 0 || throw(ArgumentError("l must be non-negative, got l=$l"))
        abs(m) <= l || throw(ArgumentError("|m| must be <= l, got l=$l, m=$m"))
        new(l, m)
    end
end

Base.:(==)(a::ScalarHarmonic, b::ScalarHarmonic) = a.l == b.l && a.m == b.m
Base.hash(a::ScalarHarmonic, h::UInt) = hash(a.m, hash(a.l, hash(:ScalarHarmonic, h)))

# ── AST integration ──────────────────────────────────────────────────

indices(::ScalarHarmonic) = TIndex[]
children(::ScalarHarmonic) = TensorExpr[]

function walk(f, expr::ScalarHarmonic)
    f(expr)
end

derivative_order(::ScalarHarmonic) = 0
is_constant(::ScalarHarmonic) = true
is_sorted_covds(::ScalarHarmonic) = true

rename_dummy(expr::ScalarHarmonic, ::Symbol, ::Symbol) = expr
rename_dummies(expr::ScalarHarmonic, ::Dict{Symbol,Symbol}) = expr
_replace_index_name(expr::ScalarHarmonic, ::Symbol, ::Symbol) = expr

# ── Escape hatch ─────────────────────────────────────────────────────

function to_expr(y::ScalarHarmonic)
    Expr(:call, :ScalarHarmonic, y.l, y.m)
end

function is_well_formed(::ScalarHarmonic)
    true
end

function _validate_walk(::ScalarHarmonic, ::TensorRegistry, ::Vector{String})
end

# ── Display ──────────────────────────────────────────────────────────

function Base.show(io::IO, y::ScalarHarmonic)
    print(io, "Y_{", y.l, ",", y.m, "}")
end

function to_latex(y::ScalarHarmonic)
    "Y_{$(y.l),$(y.m)}"
end

function to_unicode(y::ScalarHarmonic)
    l_str = _unicode_sub(string(y.l))
    m_str = _unicode_sub(string(abs(y.m)))
    m_prefix = y.m < 0 ? "-" : ""
    "Y" * l_str * "," * m_prefix * m_str
end

# ── Dagger (complex conjugation) ────────────────────────────────────

function dagger(y::ScalarHarmonic)
    conjugate(y)
end

# ── Operations ───────────────────────────────────────────────────────

"""
    conjugate(Y::ScalarHarmonic)

Complex conjugation: Y*_{lm} = (-1)^m Y_{l,-m}.
Returns a ScalarHarmonic if m == 0, otherwise a TProduct.
"""
function conjugate(y::ScalarHarmonic)
    conj_y = ScalarHarmonic(y.l, -y.m)
    if y.m == 0
        return conj_y
    end
    sign = iseven(y.m) ? 1//1 : -1//1
    TProduct(sign, TensorExpr[conj_y])
end

"""
    angular_laplacian(Y::ScalarHarmonic)

Angular Laplacian eigenvalue: Delta_{S^2} Y_{lm} = -l(l+1) Y_{lm}.
Returns a TProduct with scalar coefficient -l(l+1).
"""
function angular_laplacian(y::ScalarHarmonic)
    eigenvalue = Rational{Int}(-y.l * (y.l + 1))
    TProduct(eigenvalue, TensorExpr[y])
end

"""
    inner_product(Y1::ScalarHarmonic, Y2::ScalarHarmonic)

Orthogonality relation: <Y_{l1,m1}, Y_{l2,m2}> = delta_{l1,l2} delta_{m1,m2}.
Returns TScalar(1) if quantum numbers match, TScalar(0) otherwise.
"""
function inner_product(y1::ScalarHarmonic, y2::ScalarHarmonic)
    (y1.l == y2.l && y1.m == y2.m) ? TScalar(1) : TScalar(0)
end
