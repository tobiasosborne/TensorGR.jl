# ── Vector spherical harmonics Y^a_{lm} and X^a_{lm} ─────────────────────

"""
    EvenVectorHarmonic <: TensorExpr

Even-parity (polar/gradient) vector spherical harmonic Y^a_{lm} = D^a Y_{lm}.
Carries quantum numbers (l, m) and a single angular index on S².
Valid for l >= 1 (gradient of Y_{0,0} = const vanishes).
Eigenvalue of angular Laplacian on S²: D_a Y^a_{lm} = -l(l+1) Y_{lm}.
L² norm: ||Y^a_{lm}||² = l(l+1).
Conjugation: (Y^a_{lm})* = (-1)^m Y^a_{l,-m}.
"""
struct EvenVectorHarmonic <: TensorExpr
    l::Int
    m::Int
    index::TIndex
    function EvenVectorHarmonic(l::Int, m::Int, idx::TIndex)
        l >= 1 || throw(ArgumentError("l must be >= 1 for vector harmonics, got l=$l"))
        abs(m) <= l || throw(ArgumentError("|m| must be <= l, got l=$l, m=$m"))
        new(l, m, idx)
    end
end

"""
    OddVectorHarmonic <: TensorExpr

Odd-parity (axial) vector harmonic X^a_{lm} = epsilon^{ab} D_b Y_{lm}.
Carries quantum numbers (l, m) and a single angular index on S².
Valid for l >= 1.
Curl eigenvalue: epsilon^{ab} D_b X_a = l(l+1) Y_{lm}.
L² norm: ||X^a_{lm}||² = l(l+1).
Conjugation: (X^a_{lm})* = (-1)^m X^a_{l,-m}.
"""
struct OddVectorHarmonic <: TensorExpr
    l::Int
    m::Int
    index::TIndex
    function OddVectorHarmonic(l::Int, m::Int, idx::TIndex)
        l >= 1 || throw(ArgumentError("l must be >= 1 for vector harmonics, got l=$l"))
        abs(m) <= l || throw(ArgumentError("|m| must be <= l, got l=$l, m=$m"))
        new(l, m, idx)
    end
end

const _VectorHarmonic = Union{EvenVectorHarmonic, OddVectorHarmonic}

# ── Equality and hashing ──────────────────────────────────────────────

Base.:(==)(a::EvenVectorHarmonic, b::EvenVectorHarmonic) =
    a.l == b.l && a.m == b.m && a.index == b.index
Base.hash(a::EvenVectorHarmonic, h::UInt) =
    hash(a.index, hash(a.m, hash(a.l, hash(:EvenVectorHarmonic, h))))

Base.:(==)(a::OddVectorHarmonic, b::OddVectorHarmonic) =
    a.l == b.l && a.m == b.m && a.index == b.index
Base.hash(a::OddVectorHarmonic, h::UInt) =
    hash(a.index, hash(a.m, hash(a.l, hash(:OddVectorHarmonic, h))))

# ── AST integration ──────────────────────────────────────────────────

indices(y::_VectorHarmonic) = TIndex[y.index]
children(::_VectorHarmonic) = TensorExpr[]

function walk(f, expr::_VectorHarmonic)
    f(expr)
end

derivative_order(::_VectorHarmonic) = 0
is_constant(::_VectorHarmonic) = true
is_sorted_covds(::_VectorHarmonic) = true

function rename_dummy(expr::EvenVectorHarmonic, old::Symbol, new::Symbol)
    idx = expr.index
    new_idx = idx.name == old ? TIndex(new, idx.position, idx.vbundle) : idx
    EvenVectorHarmonic(expr.l, expr.m, new_idx)
end

function rename_dummy(expr::OddVectorHarmonic, old::Symbol, new::Symbol)
    idx = expr.index
    new_idx = idx.name == old ? TIndex(new, idx.position, idx.vbundle) : idx
    OddVectorHarmonic(expr.l, expr.m, new_idx)
end

function rename_dummies(expr::EvenVectorHarmonic, mapping::Dict{Symbol,Symbol})
    new_name = get(mapping, expr.index.name, expr.index.name)
    new_idx = new_name == expr.index.name ? expr.index :
        TIndex(new_name, expr.index.position, expr.index.vbundle)
    EvenVectorHarmonic(expr.l, expr.m, new_idx)
end

function rename_dummies(expr::OddVectorHarmonic, mapping::Dict{Symbol,Symbol})
    new_name = get(mapping, expr.index.name, expr.index.name)
    new_idx = new_name == expr.index.name ? expr.index :
        TIndex(new_name, expr.index.position, expr.index.vbundle)
    OddVectorHarmonic(expr.l, expr.m, new_idx)
end

function _replace_index_name(expr::EvenVectorHarmonic, old::Symbol, new::Symbol)
    idx = expr.index
    new_idx = idx.name == old ? TIndex(new, idx.position, idx.vbundle) : idx
    EvenVectorHarmonic(expr.l, expr.m, new_idx)
end

function _replace_index_name(expr::OddVectorHarmonic, old::Symbol, new::Symbol)
    idx = expr.index
    new_idx = idx.name == old ? TIndex(new, idx.position, idx.vbundle) : idx
    OddVectorHarmonic(expr.l, expr.m, new_idx)
end

# ── Escape hatch ─────────────────────────────────────────────────────

function to_expr(y::EvenVectorHarmonic)
    Expr(:call, :EvenVectorHarmonic, y.l, y.m, to_expr(y.index))
end

function to_expr(y::OddVectorHarmonic)
    Expr(:call, :OddVectorHarmonic, y.l, y.m, to_expr(y.index))
end

is_well_formed(::_VectorHarmonic) = true

function _validate_walk(::_VectorHarmonic, ::TensorRegistry, ::Vector{String})
end

# ── Display ──────────────────────────────────────────────────────────

function Base.show(io::IO, y::EvenVectorHarmonic)
    print(io, "Y_{", y.l, ",", y.m, "}[", y.index, "]")
end

function Base.show(io::IO, y::OddVectorHarmonic)
    print(io, "X_{", y.l, ",", y.m, "}[", y.index, "]")
end

function to_latex(y::EvenVectorHarmonic)
    idx_str = string(y.index.name)
    if y.index.position == Up
        "Y^{$idx_str}_{$(y.l),$(y.m)}"
    else
        "Y_{$idx_str\\,$(y.l),$(y.m)}"
    end
end

function to_latex(y::OddVectorHarmonic)
    idx_str = string(y.index.name)
    if y.index.position == Up
        "X^{$idx_str}_{$(y.l),$(y.m)}"
    else
        "X_{$idx_str\\,$(y.l),$(y.m)}"
    end
end

function to_unicode(y::EvenVectorHarmonic)
    l_str = _unicode_sub(string(y.l))
    m_str = _unicode_sub(string(abs(y.m)))
    m_prefix = y.m < 0 ? "-" : ""
    "Y" * to_unicode(y.index) * l_str * "," * m_prefix * m_str
end

function to_unicode(y::OddVectorHarmonic)
    l_str = _unicode_sub(string(y.l))
    m_str = _unicode_sub(string(abs(y.m)))
    m_prefix = y.m < 0 ? "-" : ""
    "X" * to_unicode(y.index) * l_str * "," * m_prefix * m_str
end

# ── Dagger (complex conjugation) ────────────────────────────────────

dagger(y::_VectorHarmonic) = conjugate(y)

# ── Operations ───────────────────────────────────────────────────────

"""
    conjugate(Y::EvenVectorHarmonic)

Complex conjugation: (Y^a_{lm})* = (-1)^m Y^a_{l,-m}.
Returns an EvenVectorHarmonic if m == 0, otherwise a TProduct.
"""
function conjugate(y::EvenVectorHarmonic)
    conj_y = EvenVectorHarmonic(y.l, -y.m, y.index)
    y.m == 0 && return conj_y
    sign = iseven(y.m) ? 1//1 : -1//1
    TProduct(sign, TensorExpr[conj_y])
end

"""
    conjugate(Y::OddVectorHarmonic)

Complex conjugation: (X^a_{lm})* = (-1)^m X^a_{l,-m}.
Returns an OddVectorHarmonic if m == 0, otherwise a TProduct.
"""
function conjugate(y::OddVectorHarmonic)
    conj_y = OddVectorHarmonic(y.l, -y.m, y.index)
    y.m == 0 && return conj_y
    sign = iseven(y.m) ? 1//1 : -1//1
    TProduct(sign, TensorExpr[conj_y])
end

"""
    divergence_eigenvalue(Y::EvenVectorHarmonic)

Divergence eigenvalue: D_a Y^a_{lm} = -l(l+1) Y_{lm},
since Y^a = D^a Y and D_a D^a Y = Delta_{S^2} Y = -l(l+1) Y.
Returns -l*(l+1) as an Int.
"""
divergence_eigenvalue(y::EvenVectorHarmonic) = -y.l * (y.l + 1)

"""
    curl_eigenvalue(Y::OddVectorHarmonic)

Curl eigenvalue: epsilon^{ab} D_b X_a = l(l+1) Y_{lm}.
Returns l*(l+1) as an Int.
"""
curl_eigenvalue(y::OddVectorHarmonic) = y.l * (y.l + 1)

"""
    norm_squared(Y::Union{EvenVectorHarmonic, OddVectorHarmonic})

L² norm on S²: ||Y^a_{lm}||² = ||X^a_{lm}||² = l(l+1).
"""
norm_squared(y::_VectorHarmonic) = y.l * (y.l + 1)
