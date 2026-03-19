# -- Vector field harmonic decomposition -------------------------------------------
# Ground truth: Martel & Poisson (2005) Sec III, Eqs 3.1-3.2
#
# A vector field on S^2 x M^2 decomposes as:
#   v_a = sum_{l,m} [ v^{lm}(x^b) r_a Y_{lm}
#                    + v^{lm}_even(x^b) Y^A_{lm}
#                    + v^{lm}_odd(x^b)  X^A_{lm} ]
#
# Three sectors per (l,m) mode:
#   :radial -- scalar coefficient times Y_{lm} (on M^2, contracted with r_a)
#   :even   -- scalar coefficient times Y^A_{lm} (even vector harmonic on S^2)
#   :odd    -- scalar coefficient times X^A_{lm} (odd vector harmonic on S^2)
#
# For l=0: only the radial sector exists (no vector harmonics at l=0).
# For l>=1: all three sectors are present.

"""
    VectorMode

Represents a single (l,m) mode of a vector field harmonic decomposition.
Contains coefficient names for the radial, even, and odd sectors.

Fields:
- `l::Int` -- angular momentum quantum number (l >= 0)
- `m::Int` -- magnetic quantum number (|m| <= l)
- `coeff_radial::Symbol` -- coefficient for the radial sector (Y_{lm})
- `coeff_even::Union{Symbol,Nothing}` -- coefficient for even sector (Y^A_{lm}), Nothing for l=0
- `coeff_odd::Union{Symbol,Nothing}` -- coefficient for odd sector (X^A_{lm}), Nothing for l=0
"""
struct VectorMode
    l::Int
    m::Int
    coeff_radial::Symbol
    coeff_even::Union{Symbol,Nothing}
    coeff_odd::Union{Symbol,Nothing}
end

Base.:(==)(a::VectorMode, b::VectorMode) =
    a.l == b.l && a.m == b.m &&
    a.coeff_radial == b.coeff_radial &&
    a.coeff_even == b.coeff_even &&
    a.coeff_odd == b.coeff_odd
Base.hash(a::VectorMode, h::UInt) =
    hash(a.coeff_odd, hash(a.coeff_even, hash(a.coeff_radial,
        hash(a.m, hash(a.l, hash(:VectorMode, h))))))

function Base.show(io::IO, mode::VectorMode)
    print(io, mode.coeff_radial, "*Y_{", mode.l, ",", mode.m, "}")
    if mode.coeff_even !== nothing
        print(io, " + ", mode.coeff_even, "*Y^A_{", mode.l, ",", mode.m, "}")
    end
    if mode.coeff_odd !== nothing
        print(io, " + ", mode.coeff_odd, "*X^A_{", mode.l, ",", mode.m, "}")
    end
end

"""
    VectorHarmonicDecomposition

Result of decomposing a vector field into spherical harmonic modes.

Fields:
- `field::Symbol` -- original field name
- `modes::Vector{VectorMode}` -- the individual (l,m) modes
- `lmax::Int` -- maximum l in the expansion
"""
struct VectorHarmonicDecomposition
    field::Symbol
    modes::Vector{VectorMode}
    lmax::Int
end

Base.:(==)(a::VectorHarmonicDecomposition, b::VectorHarmonicDecomposition) =
    a.field == b.field && a.modes == b.modes && a.lmax == b.lmax
Base.hash(a::VectorHarmonicDecomposition, h::UInt) =
    hash(a.lmax, hash(a.modes, hash(a.field, hash(:VectorHarmonicDecomposition, h))))

function Base.show(io::IO, decomp::VectorHarmonicDecomposition)
    print(io, decomp.field, "_a = sum_{l=0}^{", decomp.lmax,
          "} sum_{m=-l}^{l} [radial + even + odd]  [",
          mode_count(decomp), " modes]")
end

"""
    mode_count(decomp::VectorHarmonicDecomposition) -> Int

Total number of modes in the decomposition: (lmax+1)^2.
"""
mode_count(decomp::VectorHarmonicDecomposition) = length(decomp.modes)

"""
    get_mode(decomp::VectorHarmonicDecomposition, l::Int, m::Int) -> VectorMode

Retrieve the specific (l,m) mode from the decomposition.
Throws `ArgumentError` if the mode is not present.
"""
function get_mode(decomp::VectorHarmonicDecomposition, l::Int, m::Int)
    for mode in decomp.modes
        if mode.l == l && mode.m == m
            return mode
        end
    end
    throw(ArgumentError("Mode (l=$l, m=$m) not found in decomposition (lmax=$(decomp.lmax))"))
end

"""
    decompose_vector(field_name::Symbol, lmax::Int;
                     coeff_prefix=field_name, angular_index::TIndex=up(:A, :S2))
        -> VectorHarmonicDecomposition

Decompose a named vector field into spherical harmonic modes up to lmax.
Each (l,m) mode has three sectors (Martel & Poisson 2005, Sec III):
  - radial: coefficient * Y_{lm}  (scalar harmonic, l >= 0)
  - even:   coefficient * Y^A_{lm} (even vector harmonic, l >= 1)
  - odd:    coefficient * X^A_{lm} (odd vector harmonic, l >= 1)

For l=0, only the radial sector exists (vector harmonics vanish).

Coefficient names: `\$(prefix)_rad_l_m`, `\$(prefix)_even_l_m`, `\$(prefix)_odd_l_m`.
Negative m values produce e.g. `:v_rad_2_neg1`.

Ground truth: Martel & Poisson (2005) Sec III, Eqs 3.1-3.2.
"""
function decompose_vector(field_name::Symbol, lmax::Int;
                          coeff_prefix=field_name,
                          angular_index::TIndex=up(:A, :S2))
    lmax >= 0 || throw(ArgumentError("lmax must be non-negative, got lmax=$lmax"))
    modes = VectorMode[]
    for l in 0:lmax
        for m in -l:l
            m_str = m < 0 ? "neg$(abs(m))" : string(m)
            coeff_rad = Symbol(coeff_prefix, "_rad_", l, "_", m_str)
            if l >= 1
                coeff_even = Symbol(coeff_prefix, "_even_", l, "_", m_str)
                coeff_odd = Symbol(coeff_prefix, "_odd_", l, "_", m_str)
            else
                coeff_even = nothing
                coeff_odd = nothing
            end
            push!(modes, VectorMode(l, m, coeff_rad, coeff_even, coeff_odd))
        end
    end
    VectorHarmonicDecomposition(field_name, modes, lmax)
end

"""
    to_expr(decomp::VectorHarmonicDecomposition;
            angular_index::TIndex=up(:A, :S2)) -> TensorExpr

Convert a VectorHarmonicDecomposition to a TensorExpr.

Each (l,m) mode contributes:
  - radial:  TScalar(coeff_rad) * Y_{lm}
  - even:    TScalar(coeff_even) * Y^A_{lm}  (l >= 1)
  - odd:     TScalar(coeff_odd)  * X^A_{lm}  (l >= 1)

Returns a TSum of all terms. For a single term, returns the TProduct directly.
"""
function to_expr(decomp::VectorHarmonicDecomposition;
                 angular_index::TIndex=up(:A, :S2))
    terms = TensorExpr[]
    for mode in decomp.modes
        # Radial sector: coeff * Y_{lm}
        push!(terms, TProduct(1//1, TensorExpr[
            TScalar(mode.coeff_radial), ScalarHarmonic(mode.l, mode.m)]))
        # Even sector: coeff * Y^A_{lm} (l >= 1 only)
        if mode.coeff_even !== nothing
            push!(terms, TProduct(1//1, TensorExpr[
                TScalar(mode.coeff_even),
                EvenVectorHarmonic(mode.l, mode.m, angular_index)]))
        end
        # Odd sector: coeff * X^A_{lm} (l >= 1 only)
        if mode.coeff_odd !== nothing
            push!(terms, TProduct(1//1, TensorExpr[
                TScalar(mode.coeff_odd),
                OddVectorHarmonic(mode.l, mode.m, angular_index)]))
        end
    end
    if length(terms) == 1
        return terms[1]
    end
    TSum(terms)
end
