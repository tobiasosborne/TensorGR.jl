# -- Angular Laplacian operator on S^2 ----------------------------------------

"""
    LaplacianS2 <: TensorExpr

Angular Laplacian operator on S^2 applied to a tensor expression.
Represents Delta_Omega(arg) before simplification.  When the argument is a
recognised harmonic (ScalarHarmonic, EvenVectorHarmonic, OddVectorHarmonic),
the constructor `laplacian_S2` immediately reduces to the eigenvalue form;
otherwise the unevaluated wrapper is returned.
"""
struct LaplacianS2 <: TensorExpr
    arg::TensorExpr
end

Base.:(==)(a::LaplacianS2, b::LaplacianS2) = a.arg == b.arg
Base.hash(a::LaplacianS2, h::UInt) = hash(a.arg, hash(:LaplacianS2, h))

# -- Eigenvalue helpers -------------------------------------------------------

# Scalar: Delta_Omega Y_{lm} = -l(l+1) Y_{lm}
_laplacian_eigenvalue(y::ScalarHarmonic) = Rational{Int}(-y.l * (y.l + 1))

# Vector (even/odd): Delta_Omega V^a_{lm} = -(l(l+1) - 1) V^a_{lm}
_laplacian_eigenvalue(y::_VectorHarmonic) = Rational{Int}(-(y.l * (y.l + 1) - 1))

"""
    simplify_laplacian(L::LaplacianS2) -> TensorExpr

Resolve a LaplacianS2 to its eigenvalue form when the argument is a harmonic.
Returns the LaplacianS2 unchanged for non-harmonic arguments.
"""
function simplify_laplacian(L::LaplacianS2)
    arg = L.arg
    if arg isa ScalarHarmonic || arg isa _VectorHarmonic
        ev = _laplacian_eigenvalue(arg)
        return ev == 0 ? TScalar(0//1) : TProduct(ev, TensorExpr[arg])
    end
    L
end

"""
    laplacian_S2(expr::TensorExpr) -> TensorExpr

Angular Laplacian on S^2.  For harmonic arguments the eigenvalue relation is
applied immediately; for general expressions a `LaplacianS2` wrapper is returned.
"""
function laplacian_S2(expr::TensorExpr)
    if expr isa ScalarHarmonic || expr isa _VectorHarmonic
        ev = _laplacian_eigenvalue(expr)
        return ev == 0 ? TScalar(0//1) : TProduct(ev, TensorExpr[expr])
    end
    LaplacianS2(expr)
end

# -- AST integration ----------------------------------------------------------

indices(L::LaplacianS2) = indices(L.arg)
children(L::LaplacianS2) = TensorExpr[L.arg]

function walk(f, L::LaplacianS2)
    new_arg = walk(f, L.arg)
    f(LaplacianS2(new_arg))
end

derivative_order(L::LaplacianS2) = derivative_order(L.arg) + 2
is_constant(L::LaplacianS2) = is_constant(L.arg)
is_sorted_covds(::LaplacianS2) = true

function rename_dummy(L::LaplacianS2, old::Symbol, new::Symbol)
    LaplacianS2(rename_dummy(L.arg, old, new))
end

function rename_dummies(L::LaplacianS2, mapping::Dict{Symbol,Symbol})
    LaplacianS2(rename_dummies(L.arg, mapping))
end

function _replace_index_name(L::LaplacianS2, old::Symbol, new::Symbol)
    LaplacianS2(_replace_index_name(L.arg, old, new))
end

# -- Escape hatch -------------------------------------------------------------

function to_expr(L::LaplacianS2)
    Expr(:call, :LaplacianS2, to_expr(L.arg))
end

is_well_formed(L::LaplacianS2) = is_well_formed(L.arg)

function _validate_walk(L::LaplacianS2, reg::TensorRegistry, errs::Vector{String})
    _validate_walk(L.arg, reg, errs)
end

# -- Display ------------------------------------------------------------------

function Base.show(io::IO, L::LaplacianS2)
    print(io, "LaplacianS2(", L.arg, ")")
end

function to_latex(L::LaplacianS2)
    "\\Delta_{\\Omega}\\left(" * to_latex(L.arg) * "\\right)"
end

function to_unicode(L::LaplacianS2)
    "Delta_Omega(" * to_unicode(L.arg) * ")"
end
