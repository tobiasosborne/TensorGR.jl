# -- Symmetric tensor harmonic decomposition -----------------------------------
# Ground truth: Martel & Poisson (2005) Sec III.A, Eqs 3.1-3.3
#
# Decompose a symmetric rank-2 tensor h_{ab} on M^2 x S^2 into harmonic modes.
# For each (l,m) mode the decomposition has:
#
#   Even parity (7 coefficients):
#     h_{ab}^{lm} Y_{lm}        -- M^2-block:  3 coeffs (tt, tr, rr) x Y
#     j_a^{lm}   Y^A_{lm}       -- mixed block: 2 coeffs (t, r) x Y^A
#     K^{lm}     Y^{AB}_{lm}    -- S^2-block:   metric-trace part x Omega_{AB} Y
#     G^{lm}     Z^{AB}_{lm}    -- S^2-block:   trace-free part x Z^{AB}
#
#   Odd parity (3 coefficients):
#     h_a^{lm}   X^A_{lm}       -- mixed block: 2 coeffs (t, r) x X^A
#     h2^{lm}    X^{AB}_{lm}    -- S^2-block:   1 coeff x X^{AB}
#
# Total: 10 independent coefficient functions per (l,m).
# Note: Y^A, Z^{AB}, X^A, X^{AB} require l >= 1 or l >= 2 respectively,
#       so low multipoles have fewer components.

"""
    Parity

Parity label for tensor harmonic modes.
"""
@enum Parity EVEN ODD

"""
    TensorMode

Represents a single (l,m) mode of the symmetric tensor harmonic decomposition.
Contains coefficient names for each sector and harmonic objects.

Fields:
- `l::Int` -- angular momentum quantum number (l >= 0)
- `m::Int` -- magnetic quantum number (|m| <= l)
- `parity::Parity` -- EVEN or ODD
- `coeffs::Dict{Symbol, Symbol}` -- sector label => coefficient name
"""
struct TensorMode
    l::Int
    m::Int
    parity::Parity
    coeffs::Dict{Symbol, Symbol}
end

Base.:(==)(a::TensorMode, b::TensorMode) =
    a.l == b.l && a.m == b.m && a.parity == b.parity && a.coeffs == b.coeffs
Base.hash(a::TensorMode, h::UInt) =
    hash(a.coeffs, hash(a.parity, hash(a.m, hash(a.l, hash(:TensorMode, h)))))

function Base.show(io::IO, mode::TensorMode)
    p = mode.parity == EVEN ? "even" : "odd"
    nsec = length(mode.coeffs)
    print(io, "(l=", mode.l, ", m=", mode.m, ", ", p, ", ", nsec, " sectors)")
end

"""
    TensorHarmonicDecomposition

Result of decomposing a symmetric rank-2 tensor field into tensor spherical harmonics.

Fields:
- `field::Symbol` -- original field name
- `modes::Vector{TensorMode}` -- the individual (l,m) modes (even then odd)
- `lmax::Int` -- maximum l in the expansion
"""
struct TensorHarmonicDecomposition
    field::Symbol
    modes::Vector{TensorMode}
    lmax::Int
end

Base.:(==)(a::TensorHarmonicDecomposition, b::TensorHarmonicDecomposition) =
    a.field == b.field && a.modes == b.modes && a.lmax == b.lmax
Base.hash(a::TensorHarmonicDecomposition, h::UInt) =
    hash(a.lmax, hash(a.modes, hash(a.field, hash(:TensorHarmonicDecomposition, h))))

function Base.show(io::IO, decomp::TensorHarmonicDecomposition)
    n_even = count(m -> m.parity == EVEN, decomp.modes)
    n_odd = count(m -> m.parity == ODD, decomp.modes)
    print(io, decomp.field, " = sum_{l=0}^{", decomp.lmax,
          "} [", n_even, " even + ", n_odd, " odd modes]")
end

"""
    mode_count(decomp::TensorHarmonicDecomposition) -> Int

Total number of modes (even + odd) in the decomposition.
"""
mode_count(decomp::TensorHarmonicDecomposition) = length(decomp.modes)

"""
    get_mode(decomp::TensorHarmonicDecomposition, l::Int, m::Int, parity::Parity) -> TensorMode

Retrieve the specific (l, m, parity) mode from the decomposition.
Throws `ArgumentError` if the mode is not present.
"""
function get_mode(decomp::TensorHarmonicDecomposition, l::Int, m::Int, parity::Parity)
    for mode in decomp.modes
        if mode.l == l && mode.m == m && mode.parity == parity
            return mode
        end
    end
    p_str = parity == EVEN ? "even" : "odd"
    throw(ArgumentError("Mode (l=$l, m=$m, $p_str) not found in decomposition (lmax=$(decomp.lmax))"))
end

# ── Coefficient naming helper ────────────────────────────────────────────────

function _m_str(m::Int)
    m < 0 ? "neg$(abs(m))" : string(m)
end

# ── Even-parity coefficient names for (l,m) ────────────────────────────────

function _even_coeffs(prefix::Symbol, l::Int, m::Int)
    ms = _m_str(m)
    coeffs = Dict{Symbol, Symbol}()
    # M^2-block: h_{tt}, h_{tr}, h_{rr} (scalar harmonics, l >= 0)
    coeffs[:tt] = Symbol(prefix, "_tt_", l, "_", ms)
    coeffs[:tr] = Symbol(prefix, "_tr_", l, "_", ms)
    coeffs[:rr] = Symbol(prefix, "_rr_", l, "_", ms)
    # Mixed block: j_t, j_r (vector harmonics, l >= 1)
    if l >= 1
        coeffs[:jt] = Symbol(prefix, "_jt_", l, "_", ms)
        coeffs[:jr] = Symbol(prefix, "_jr_", l, "_", ms)
    end
    # S^2-block metric part: K (metric-type tensor harmonic, l >= 0)
    coeffs[:K] = Symbol(prefix, "_K_", l, "_", ms)
    # S^2-block STF part: G (trace-free tensor harmonic, l >= 2)
    if l >= 2
        coeffs[:G] = Symbol(prefix, "_G_", l, "_", ms)
    end
    coeffs
end

# ── Odd-parity coefficient names for (l,m) ─────────────────────────────────

function _odd_coeffs(prefix::Symbol, l::Int, m::Int)
    ms = _m_str(m)
    coeffs = Dict{Symbol, Symbol}()
    # Mixed block: h_t, h_r (odd vector harmonics, l >= 1)
    if l >= 1
        coeffs[:ht] = Symbol(prefix, "_ht_", l, "_", ms)
        coeffs[:hr] = Symbol(prefix, "_hr_", l, "_", ms)
    end
    # S^2-block: h2 (odd tensor harmonic, l >= 2)
    if l >= 2
        coeffs[:h2] = Symbol(prefix, "_h2_", l, "_", ms)
    end
    coeffs
end

"""
    decompose_symmetric_tensor(field_name::Symbol, lmax::Int;
                               coeff_prefix=field_name) -> TensorHarmonicDecomposition

Decompose a named symmetric rank-2 tensor field into tensor spherical harmonic modes
up to lmax, following the Regge-Wheeler / Martel-Poisson decomposition.

Ground truth: Martel & Poisson (2005) Sec III.A, Eqs 3.1-3.3.

Each mode contains coefficient names for the relevant sectors:
- Even parity: :tt, :tr, :rr (l >= 0), :jt, :jr (l >= 1), :K (l >= 0), :G (l >= 2)
- Odd parity: :ht, :hr (l >= 1), :h2 (l >= 2)

Coefficient names are formed as `\$(prefix)_\$(sector)_\$(l)_\$(m)`.
Negative m values produce e.g. `:h_tt_2_neg1` for (l=2, m=-1).
"""
function decompose_symmetric_tensor(field_name::Symbol, lmax::Int;
                                    coeff_prefix=field_name)
    lmax >= 0 || throw(ArgumentError("lmax must be non-negative, got lmax=$lmax"))
    modes = TensorMode[]
    for l in 0:lmax
        for m in -l:l
            # Even-parity mode
            ec = _even_coeffs(coeff_prefix, l, m)
            push!(modes, TensorMode(l, m, EVEN, ec))
            # Odd-parity mode (but l=0 odd has no sectors at all)
            oc = _odd_coeffs(coeff_prefix, l, m)
            if !isempty(oc)
                push!(modes, TensorMode(l, m, ODD, oc))
            end
        end
    end
    TensorHarmonicDecomposition(field_name, modes, lmax)
end

"""
    to_expr(decomp::TensorHarmonicDecomposition;
            angular_idx1::TIndex=down(:A), angular_idx2::TIndex=down(:B)) -> TensorExpr

Convert a TensorHarmonicDecomposition to a TensorExpr (TSum of TProduct terms).

Each sector of each mode produces one term:
- :tt, :tr, :rr, :K sectors -> coeff * Y_{lm}  (or coeff * Y^{AB}_{lm} for :K)
- :jt, :jr sectors -> coeff * Y^A_{lm}
- :G sector -> coeff * Z^{AB}_{lm}
- :ht, :hr sectors -> coeff * X^A_{lm}
- :h2 sector -> coeff * X^{AB}_{lm}

The angular indices on vector/tensor harmonics use `angular_idx1`, `angular_idx2`.
For a single term, returns TProduct directly instead of a 1-element TSum.
"""
function to_expr(decomp::TensorHarmonicDecomposition;
                 angular_idx1::TIndex=down(:A), angular_idx2::TIndex=down(:B))
    terms = TensorExpr[]
    for mode in decomp.modes
        l, m = mode.l, mode.m
        for (sector, coeff_name) in sort(collect(mode.coeffs), by=first)
            harmonic = _sector_harmonic(sector, l, m, mode.parity,
                                        angular_idx1, angular_idx2)
            term = TProduct(1//1, TensorExpr[TScalar(coeff_name), harmonic])
            push!(terms, term)
        end
    end
    if length(terms) == 1
        return terms[1]
    end
    TSum(terms)
end

function _sector_harmonic(sector::Symbol, l::Int, m::Int, parity::Parity,
                          idx1::TIndex, idx2::TIndex)
    if parity == EVEN
        if sector in (:tt, :tr, :rr)
            return ScalarHarmonic(l, m)
        elseif sector in (:jt, :jr)
            return EvenVectorHarmonic(l, m, idx1)
        elseif sector == :K
            return EvenTensorHarmonicY(l, m, idx1, idx2)
        elseif sector == :G
            return EvenTensorHarmonicZ(l, m, idx1, idx2)
        end
    else  # ODD
        if sector in (:ht, :hr)
            return OddVectorHarmonic(l, m, idx1)
        elseif sector == :h2
            return OddTensorHarmonic(l, m, idx1, idx2)
        end
    end
    error("Unknown sector $sector for parity $parity")
end
