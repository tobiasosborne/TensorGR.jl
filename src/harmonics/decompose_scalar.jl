# -- Scalar field harmonic decomposition -----------------------------------------
# Ground truth: Martel & Poisson (2005) Sec II.B, Eq 2.4
# f(t,r,theta,phi) = sum_{l,m} f_{lm}(t,r) Y_{lm}(theta,phi)

"""
    ScalarMode

Represents a single (l,m) mode of a scalar field decomposition.
Contains the radial coefficient name and the angular harmonic.

Fields:
- `l::Int` -- angular momentum quantum number (l >= 0)
- `m::Int` -- magnetic quantum number (|m| <= l)
- `coeff::Symbol` -- name of the radial coefficient function f_{lm}
- `harmonic::ScalarHarmonic` -- the spherical harmonic Y_{lm}
"""
struct ScalarMode
    l::Int
    m::Int
    coeff::Symbol
    harmonic::ScalarHarmonic
end

Base.:(==)(a::ScalarMode, b::ScalarMode) =
    a.l == b.l && a.m == b.m && a.coeff == b.coeff && a.harmonic == b.harmonic
Base.hash(a::ScalarMode, h::UInt) =
    hash(a.harmonic, hash(a.coeff, hash(a.m, hash(a.l, hash(:ScalarMode, h)))))

function Base.show(io::IO, mode::ScalarMode)
    print(io, mode.coeff, " * Y_{", mode.l, ",", mode.m, "}")
end

"""
    HarmonicDecomposition

Result of decomposing a scalar field into spherical harmonic modes.

Fields:
- `field::Symbol` -- original field name
- `modes::Vector{ScalarMode}` -- the individual (l,m) modes
- `lmax::Int` -- maximum l in the expansion
"""
struct HarmonicDecomposition
    field::Symbol
    modes::Vector{ScalarMode}
    lmax::Int
end

Base.:(==)(a::HarmonicDecomposition, b::HarmonicDecomposition) =
    a.field == b.field && a.modes == b.modes && a.lmax == b.lmax
Base.hash(a::HarmonicDecomposition, h::UInt) =
    hash(a.lmax, hash(a.modes, hash(a.field, hash(:HarmonicDecomposition, h))))

function Base.show(io::IO, decomp::HarmonicDecomposition)
    print(io, decomp.field, " = sum_{l=0}^{", decomp.lmax, "} sum_{m=-l}^{l} ",
          decomp.field, "_{lm} Y_{lm}  [", mode_count(decomp), " modes]")
end

"""
    mode_count(decomp::HarmonicDecomposition) -> Int

Total number of modes in the decomposition: (lmax+1)^2.
"""
mode_count(decomp::HarmonicDecomposition) = length(decomp.modes)

"""
    get_mode(decomp::HarmonicDecomposition, l::Int, m::Int) -> ScalarMode

Retrieve the specific (l,m) mode from the decomposition.
Throws `ArgumentError` if the mode is not present.
"""
function get_mode(decomp::HarmonicDecomposition, l::Int, m::Int)
    for mode in decomp.modes
        if mode.l == l && mode.m == m
            return mode
        end
    end
    throw(ArgumentError("Mode (l=$l, m=$m) not found in decomposition (lmax=$(decomp.lmax))"))
end

"""
    decompose_scalar(field_name::Symbol, lmax::Int; coeff_prefix=field_name) -> HarmonicDecomposition

Decompose a named scalar field into spherical harmonic modes up to lmax.
Each mode has a coefficient tensor f_{lm} (abstract, represents radial function)
and a ScalarHarmonic Y_{lm}.

The coefficient names are formed as `\$(coeff_prefix)_\$(l)_\$(m)`.
Negative m values produce e.g. `:Phi_2_neg1` for (l=2, m=-1).

Ground truth: Martel & Poisson (2005) Sec II.B, Eq 2.4.
"""
function decompose_scalar(field_name::Symbol, lmax::Int; coeff_prefix=field_name)
    lmax >= 0 || throw(ArgumentError("lmax must be non-negative, got lmax=$lmax"))
    modes = ScalarMode[]
    for l in 0:lmax
        for m in -l:l
            m_str = m < 0 ? "neg$(abs(m))" : string(m)
            coeff_name = Symbol(coeff_prefix, "_", l, "_", m_str)
            push!(modes, ScalarMode(l, m, coeff_name, ScalarHarmonic(l, m)))
        end
    end
    HarmonicDecomposition(field_name, modes, lmax)
end

"""
    to_expr(decomp::HarmonicDecomposition; registry=current_registry()) -> TensorExpr

Convert a HarmonicDecomposition to a TensorExpr:
    Sigma_{l,m} f_{lm} * Y_{lm}

Returns a TSum of TProduct terms, each being TScalar(:coeff_name) * Y_{lm}.
For a single mode (lmax=0), returns the single TProduct directly.
"""
function to_expr(decomp::HarmonicDecomposition)
    terms = TensorExpr[]
    for mode in decomp.modes
        term = TProduct(1//1, TensorExpr[TScalar(mode.coeff), mode.harmonic])
        push!(terms, term)
    end
    if length(terms) == 1
        return terms[1]
    end
    TSum(terms)
end
