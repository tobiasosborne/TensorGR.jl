#= CAS integration hooks for scalar simplification.

When Symbolics.jl is loaded, these dispatch to Symbolics.simplify.
Without Symbolics, they are no-ops or use the built-in Expr tree algebra.

Pattern: base defines `simplify_scalar` which calls `_simplify_scalar_val`.
The extension adds methods on `_simplify_scalar_val` for Expr types.
=#

"""
    _simplify_scalar_val(x)

Internal hook for scalar simplification. Default: identity.
Extended by Symbolics extension for Expr/Symbol types.
"""
_simplify_scalar_val(x) = x

"""
    simplify_scalar(s::TScalar) -> TScalar

Simplify a scalar expression. Dispatches to Symbolics.simplify when available,
otherwise returns the input unchanged.
"""
function simplify_scalar(s::TScalar)
    new_val = _simplify_scalar_val(s.val)
    new_val === s.val ? s : TScalar(new_val)
end

"""
    simplify_quadratic_form(qf::QuadraticForm) -> QuadraticForm

Simplify all entries in a QuadraticForm matrix using the CAS backend.
"""
function simplify_quadratic_form(qf::QuadraticForm)
    n = length(qf.fields)
    M = copy(qf.matrix)
    for i in 1:n, j in 1:n
        M[i, j] = _try_simplify_entry(M[i, j])
    end
    QuadraticForm(qf.fields, M)
end

"""
    _try_simplify_entry(x)

Attempt to simplify a matrix entry. Extended by Symbolics extension for Expr types.
"""
_try_simplify_entry(x) = x
_try_simplify_entry(x::Number) = x

"""
    symbolic_quadratic_form(entries, fields; variables)

Construct a QuadraticForm with Symbolics.Num entries. Requires Symbolics.jl.
"""
function symbolic_quadratic_form end

"""
    to_fourier_symbolic(expr; momentum_vars)

Convert derivatives to symbolic momentum variables. Requires Symbolics.jl.
"""
function to_fourier_symbolic end
