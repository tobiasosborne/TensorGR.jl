# -- Even- and odd-parity tensor spherical harmonics on S^2 ------------------
#
# Martel & Poisson, Phys. Rev. D 71, 104003 (2005), arXiv:gr-qc/0502028.
#
# Y^{ab}_{lm} = Omega^{ab} Y_{lm}                              (Eq 2.14, metric-type)
# Z^{ab}_{lm} = (D^a D^b + 1/2 l(l+1) Omega^{ab}) Y_{lm}      (Eq 2.16, even STF)
# X^{ab}_{lm} = -1/2 (eps^a_c D^b + eps^b_c D^a) D^c Y_{lm}   (Eq 2.17, odd STF)

"""
    EvenTensorHarmonicY <: TensorExpr

Metric-type even-parity tensor harmonic Y^{ab}_{lm} = Omega^{ab} Y_{lm}.
Two S^2 indices, l >= 0.  Trace: Omega_{ab} Y^{ab} = 2 Y_{lm}.
L^2 norm: ||Y^{ab}_{lm}||^2 = 2  (MP Eq 2.18).
"""
struct EvenTensorHarmonicY <: TensorExpr
    l::Int
    m::Int
    index1::TIndex
    index2::TIndex
    function EvenTensorHarmonicY(l::Int, m::Int, idx1::TIndex, idx2::TIndex)
        l >= 0 || throw(ArgumentError("l must be >= 0 for Y^{ab}_{lm}, got l=$l"))
        abs(m) <= l || throw(ArgumentError("|m| must be <= l, got l=$l, m=$m"))
        new(l, m, idx1, idx2)
    end
end

"""
    EvenTensorHarmonicZ <: TensorExpr

Symmetric trace-free even-parity tensor harmonic
Z^{ab}_{lm} = (D^a D^b + 1/2 l(l+1) Omega^{ab}) Y_{lm}.
Two S^2 indices, l >= 2 (l=0,1 vanish identically).
Trace-free: Omega_{ab} Z^{ab} = 0.
L^2 norm: ||Z^{ab}_{lm}||^2 = 1/2 (l-1)l(l+1)(l+2)  (MP Eq 2.19).
"""
struct EvenTensorHarmonicZ <: TensorExpr
    l::Int
    m::Int
    index1::TIndex
    index2::TIndex
    function EvenTensorHarmonicZ(l::Int, m::Int, idx1::TIndex, idx2::TIndex)
        l >= 2 || throw(ArgumentError("l must be >= 2 for Z^{ab}_{lm}, got l=$l"))
        abs(m) <= l || throw(ArgumentError("|m| must be <= l, got l=$l, m=$m"))
        new(l, m, idx1, idx2)
    end
end

"""
    OddTensorHarmonic <: TensorExpr

Symmetric trace-free odd-parity tensor harmonic
X^{ab}_{lm} = -1/2 (eps^a_c D^b + eps^b_c D^a) D^c Y_{lm}.
Two S^2 indices, l >= 2 (l=0,1 vanish identically).
Trace-free: Omega_{ab} X^{ab} = 0.
L^2 norm: ||X^{ab}_{lm}||^2 = 1/2 (l-1)l(l+1)(l+2)  (MP Eq 2.20).
"""
struct OddTensorHarmonic <: TensorExpr
    l::Int
    m::Int
    index1::TIndex
    index2::TIndex
    function OddTensorHarmonic(l::Int, m::Int, idx1::TIndex, idx2::TIndex)
        l >= 2 || throw(ArgumentError("l must be >= 2 for X^{ab}_{lm}, got l=$l"))
        abs(m) <= l || throw(ArgumentError("|m| must be <= l, got l=$l, m=$m"))
        new(l, m, idx1, idx2)
    end
end

const _TensorHarmonic = Union{EvenTensorHarmonicY, EvenTensorHarmonicZ, OddTensorHarmonic}

# -- Equality and hashing ----------------------------------------------------

Base.:(==)(a::EvenTensorHarmonicY, b::EvenTensorHarmonicY) =
    a.l == b.l && a.m == b.m && a.index1 == b.index1 && a.index2 == b.index2
Base.hash(a::EvenTensorHarmonicY, h::UInt) =
    hash(a.index2, hash(a.index1, hash(a.m, hash(a.l, hash(:EvenTensorHarmonicY, h)))))

Base.:(==)(a::EvenTensorHarmonicZ, b::EvenTensorHarmonicZ) =
    a.l == b.l && a.m == b.m && a.index1 == b.index1 && a.index2 == b.index2
Base.hash(a::EvenTensorHarmonicZ, h::UInt) =
    hash(a.index2, hash(a.index1, hash(a.m, hash(a.l, hash(:EvenTensorHarmonicZ, h)))))

Base.:(==)(a::OddTensorHarmonic, b::OddTensorHarmonic) =
    a.l == b.l && a.m == b.m && a.index1 == b.index1 && a.index2 == b.index2
Base.hash(a::OddTensorHarmonic, h::UInt) =
    hash(a.index2, hash(a.index1, hash(a.m, hash(a.l, hash(:OddTensorHarmonic, h)))))

# -- AST integration ---------------------------------------------------------

indices(t::_TensorHarmonic) = TIndex[t.index1, t.index2]
children(::_TensorHarmonic) = TensorExpr[]

walk(f, expr::_TensorHarmonic) = f(expr)

derivative_order(::_TensorHarmonic) = 0
is_constant(::_TensorHarmonic) = true
is_sorted_covds(::_TensorHarmonic) = true

function _replace_indices(::Type{EvenTensorHarmonicY}, t, idx1, idx2)
    EvenTensorHarmonicY(t.l, t.m, idx1, idx2)
end
function _replace_indices(::Type{EvenTensorHarmonicZ}, t, idx1, idx2)
    EvenTensorHarmonicZ(t.l, t.m, idx1, idx2)
end
function _replace_indices(::Type{OddTensorHarmonic}, t, idx1, idx2)
    OddTensorHarmonic(t.l, t.m, idx1, idx2)
end

for T in (:EvenTensorHarmonicY, :EvenTensorHarmonicZ, :OddTensorHarmonic)
    @eval function rename_dummy(expr::$T, old::Symbol, new::Symbol)
        i1 = expr.index1
        i2 = expr.index2
        n1 = i1.name == old ? TIndex(new, i1.position, i1.vbundle) : i1
        n2 = i2.name == old ? TIndex(new, i2.position, i2.vbundle) : i2
        _replace_indices($T, expr, n1, n2)
    end

    @eval function rename_dummies(expr::$T, mapping::Dict{Symbol,Symbol})
        n1 = get(mapping, expr.index1.name, expr.index1.name)
        n2 = get(mapping, expr.index2.name, expr.index2.name)
        i1 = n1 == expr.index1.name ? expr.index1 :
            TIndex(n1, expr.index1.position, expr.index1.vbundle)
        i2 = n2 == expr.index2.name ? expr.index2 :
            TIndex(n2, expr.index2.position, expr.index2.vbundle)
        _replace_indices($T, expr, i1, i2)
    end

    @eval function _replace_index_name(expr::$T, old::Symbol, new::Symbol)
        i1 = expr.index1
        i2 = expr.index2
        n1 = i1.name == old ? TIndex(new, i1.position, i1.vbundle) : i1
        n2 = i2.name == old ? TIndex(new, i2.position, i2.vbundle) : i2
        _replace_indices($T, expr, n1, n2)
    end
end

# -- Escape hatch ------------------------------------------------------------

function to_expr(t::EvenTensorHarmonicY)
    Expr(:call, :EvenTensorHarmonicY, t.l, t.m, to_expr(t.index1), to_expr(t.index2))
end
function to_expr(t::EvenTensorHarmonicZ)
    Expr(:call, :EvenTensorHarmonicZ, t.l, t.m, to_expr(t.index1), to_expr(t.index2))
end
function to_expr(t::OddTensorHarmonic)
    Expr(:call, :OddTensorHarmonic, t.l, t.m, to_expr(t.index1), to_expr(t.index2))
end

is_well_formed(::_TensorHarmonic) = true

_validate_walk(::_TensorHarmonic, ::TensorRegistry, ::Vector{String}) = nothing

# -- Display ------------------------------------------------------------------

function Base.show(io::IO, t::EvenTensorHarmonicY)
    print(io, "Y_{", t.l, ",", t.m, "}[", t.index1, ",", t.index2, "]")
end
function Base.show(io::IO, t::EvenTensorHarmonicZ)
    print(io, "Z_{", t.l, ",", t.m, "}[", t.index1, ",", t.index2, "]")
end
function Base.show(io::IO, t::OddTensorHarmonic)
    print(io, "X_{", t.l, ",", t.m, "}[", t.index1, ",", t.index2, "]")
end

function _tensor_harmonic_latex(prefix::String, t::_TensorHarmonic)
    i1 = string(t.index1.name)
    i2 = string(t.index2.name)
    if t.index1.position == Up && t.index2.position == Up
        "$prefix^{$i1 $i2}_{$(t.l),$(t.m)}"
    elseif t.index1.position == Down && t.index2.position == Down
        "$(prefix)_{$i1 $i2\\,$(t.l),$(t.m)}"
    else
        up_str = t.index1.position == Up ? i1 : i2
        dn_str = t.index1.position == Down ? i1 : i2
        "$prefix^{$up_str}_{$dn_str\\,$(t.l),$(t.m)}"
    end
end

to_latex(t::EvenTensorHarmonicY) = _tensor_harmonic_latex("Y", t)
to_latex(t::EvenTensorHarmonicZ) = _tensor_harmonic_latex("Z", t)
to_latex(t::OddTensorHarmonic)   = _tensor_harmonic_latex("X", t)

function _tensor_harmonic_unicode(prefix::String, t::_TensorHarmonic)
    l_str = _unicode_sub(string(t.l))
    m_str = _unicode_sub(string(abs(t.m)))
    m_prefix = t.m < 0 ? "-" : ""
    prefix * to_unicode(t.index1) * to_unicode(t.index2) * l_str * "," * m_prefix * m_str
end

to_unicode(t::EvenTensorHarmonicY) = _tensor_harmonic_unicode("Y", t)
to_unicode(t::EvenTensorHarmonicZ) = _tensor_harmonic_unicode("Z", t)
to_unicode(t::OddTensorHarmonic)   = _tensor_harmonic_unicode("X", t)

# -- Dagger (complex conjugation) --------------------------------------------

dagger(t::_TensorHarmonic) = conjugate(t)

# -- Operations ---------------------------------------------------------------

"""
    conjugate(t::EvenTensorHarmonicY)

Complex conjugation: (Y^{ab}_{lm})* = (-1)^m Y^{ab}_{l,-m}.
"""
function conjugate(t::EvenTensorHarmonicY)
    conj_t = EvenTensorHarmonicY(t.l, -t.m, t.index1, t.index2)
    t.m == 0 && return conj_t
    sign = iseven(t.m) ? 1//1 : -1//1
    TProduct(sign, TensorExpr[conj_t])
end

"""
    conjugate(t::EvenTensorHarmonicZ)

Complex conjugation: (Z^{ab}_{lm})* = (-1)^m Z^{ab}_{l,-m}.
"""
function conjugate(t::EvenTensorHarmonicZ)
    conj_t = EvenTensorHarmonicZ(t.l, -t.m, t.index1, t.index2)
    t.m == 0 && return conj_t
    sign = iseven(t.m) ? 1//1 : -1//1
    TProduct(sign, TensorExpr[conj_t])
end

"""
    conjugate(t::OddTensorHarmonic)

Complex conjugation: (X^{ab}_{lm})* = (-1)^m X^{ab}_{l,-m}.
"""
function conjugate(t::OddTensorHarmonic)
    conj_t = OddTensorHarmonic(t.l, -t.m, t.index1, t.index2)
    t.m == 0 && return conj_t
    sign = iseven(t.m) ? 1//1 : -1//1
    TProduct(sign, TensorExpr[conj_t])
end

"""
    trace(t::EvenTensorHarmonicY)

Trace: Omega_{ab} Y^{ab}_{lm} = 2 Y_{lm}  (MP Eq 2.14).
"""
function trace(t::EvenTensorHarmonicY)
    TProduct(2//1, TensorExpr[ScalarHarmonic(t.l, t.m)])
end

"""
    trace(::EvenTensorHarmonicZ)

Trace: Omega_{ab} Z^{ab}_{lm} = 0  (STF, MP Eq 2.21).
"""
trace(::EvenTensorHarmonicZ) = TScalar(0//1)

"""
    trace(::OddTensorHarmonic)

Trace: Omega_{ab} X^{ab}_{lm} = 0  (STF, MP Eq 2.21).
"""
trace(::OddTensorHarmonic) = TScalar(0//1)

"""
    norm_squared(::EvenTensorHarmonicY)

L^2 norm: ||Y^{ab}_{lm}||^2 = 2  (MP Eq 2.18).
"""
norm_squared(::EvenTensorHarmonicY) = 2

"""
    norm_squared(t::EvenTensorHarmonicZ)

L^2 norm: ||Z^{ab}_{lm}||^2 = 1/2 (l-1)l(l+1)(l+2)  (MP Eq 2.19).
"""
norm_squared(t::EvenTensorHarmonicZ) = (t.l - 1) * t.l * (t.l + 1) * (t.l + 2) // 2

"""
    norm_squared(t::OddTensorHarmonic)

L^2 norm: ||X^{ab}_{lm}||^2 = 1/2 (l-1)l(l+1)(l+2)  (MP Eq 2.20).
"""
norm_squared(t::OddTensorHarmonic) = (t.l - 1) * t.l * (t.l + 1) * (t.l + 2) // 2

"""
    inner_product(t1::T, t2::T) where T <: _TensorHarmonic

Orthogonality: <T_{lm}, T_{l'm'}> = N delta_{ll'} delta_{mm'}
where N = norm_squared.  Returns TScalar(N) or TScalar(0).
Cross-type inner products vanish (MP Eqs 2.18-2.20).
"""
function inner_product(t1::T, t2::T) where T <: _TensorHarmonic
    (t1.l == t2.l && t1.m == t2.m) ? TScalar(norm_squared(t1)) : TScalar(0)
end

# Cross-type orthogonality (MP Eq 2.20 and by parity)
inner_product(::EvenTensorHarmonicY, ::EvenTensorHarmonicZ) = TScalar(0)
inner_product(::EvenTensorHarmonicZ, ::EvenTensorHarmonicY) = TScalar(0)
inner_product(::EvenTensorHarmonicY, ::OddTensorHarmonic)   = TScalar(0)
inner_product(::OddTensorHarmonic,   ::EvenTensorHarmonicY) = TScalar(0)
inner_product(::EvenTensorHarmonicZ, ::OddTensorHarmonic)   = TScalar(0)
inner_product(::OddTensorHarmonic,   ::EvenTensorHarmonicZ) = TScalar(0)
