# -- Mode coupling coefficients from angular integrals of tensor harmonics ------
#
# At second order in BH perturbation theory, products of first-order modes couple:
#   h_{ab}^{(1)} h_{cd}^{(1)} -> sum_{l,m} C^{l1 m1, l2 m2}_{l m} h^{(2)}_{lm}
#
# The coupling coefficients are angular integrals of products of three harmonics:
#   C = integral Y*_{lm} . Y_{l1,m1} . Y_{l2,m2} dOmega
#
# These reduce to products of Wigner 3j symbols and normalization factors from
# the harmonic definitions (scalar, vector, tensor).
#
# Reference: Brizuela, Martin-Garcia & Mena Marugan, PRD 80, 024021 (2009),
#            arXiv:0903.1134, Appendix A;
#            Gleiser, Nicasio, Price & Pullin, Phys. Rep. 325, 41 (2000), App B;
#            Edmonds, "Angular Momentum in Quantum Mechanics" (1957), Eq 4.6.3.

# ---- Selection rules ---------------------------------------------------------

"""
    coupling_selection_rule(l, l1, l2) -> Bool

Check if the angular momenta (l, l1, l2) satisfy the selection rules for a
nonzero mode coupling coefficient:

- Triangle inequality: |l1 - l2| <= l <= l1 + l2
- Parity: l1 + l2 + l must be even (from the parity integral on S^2)

These are necessary conditions; the coefficient may still vanish for particular
m values if m != m1 + m2.

Ground truth: Edmonds (1957), Sec 3.7; Brizuela et al. (2009), Appendix A.
"""
function coupling_selection_rule(l::Int, l1::Int, l2::Int)
    l < abs(l1 - l2) && return false
    l > l1 + l2 && return false
    isodd(l1 + l2 + l) && return false
    true
end

# ---- Mode coupling coefficient (single entry) -------------------------------

"""
    mode_coupling_coefficient(l::Int, m::Int, l1::Int, m1::Int, l2::Int, m2::Int;
                              type1::Symbol=:scalar, type2::Symbol=:scalar,
                              type_out::Symbol=:scalar) -> Float64

Compute the mode coupling coefficient for the product of two harmonic modes
(l1,m1) and (l2,m2) projecting onto the output mode (l,m).

The coefficient is the angular integral:

    C^{l1 m1, l2 m2}_{l m} = integral H1_{l1,m1} H2_{l2,m2} Y*_{l,m} dOmega

where H1, H2 are harmonics of the specified types, and the output projection
is always onto a scalar harmonic Y*_{l,m}.

# Harmonic types

- `:scalar` -- scalar harmonic Y_{lm}
- `:even_vector` -- even-parity vector harmonic Y^A_{lm} = D^A Y_{lm}
- `:odd_vector` -- odd-parity vector harmonic X^A_{lm} = eps^A_B D^B Y_{lm}
- `:even_tensor_Y` -- metric-type tensor harmonic Y^{AB}_{lm} = Omega^{AB} Y_{lm}
- `:even_tensor_Z` -- STF even tensor harmonic Z^{AB}_{lm}
- `:odd_tensor` -- STF odd tensor harmonic X^{AB}_{lm}

For scalar-scalar coupling, this reduces to the Gaunt integral.

# Selection rules

Returns 0 if any of:
- m != m1 + m2  (azimuthal conservation)
- |l1 - l2| > l or l > l1 + l2  (triangle inequality)
- l1 + l2 + l is odd  (parity)
- Cross-parity tensor types (e.g., even_vector x odd_vector)

# Ground truth

- Scalar-scalar: Gaunt integral, Edmonds (1957) Eq 4.6.3
- Vector-vector: Brizuela et al. (2009) Eq B.2
- Tensor-tensor: Gleiser et al. (2000) Eq B.12
"""
function mode_coupling_coefficient(l::Int, m::Int, l1::Int, m1::Int, l2::Int, m2::Int;
                                   type1::Symbol=:scalar, type2::Symbol=:scalar,
                                   type_out::Symbol=:scalar)
    # Validate quantum numbers
    l >= 0 || return 0.0
    l1 >= 0 || return 0.0
    l2 >= 0 || return 0.0
    abs(m) > l && return 0.0
    abs(m1) > l1 && return 0.0
    abs(m2) > l2 && return 0.0

    # The output projection is always onto Y*_{l,m}, so m = m1 + m2
    m != m1 + m2 && return 0.0

    # Triangle inequality and parity
    !coupling_selection_rule(l, l1, l2) && return 0.0

    # Dispatch on harmonic types
    if type_out == :scalar
        return _coupling_dispatch(l, m, l1, m1, l2, m2, type1, type2)
    else
        # For non-scalar output projection, we would need additional infrastructure.
        # Currently only scalar projection (expansion in Y_{lm}) is supported.
        error("Only :scalar output projection is currently supported, got type_out=$type_out")
    end
end

"""
    _coupling_dispatch(l, m, l1, m1, l2, m2, type1, type2) -> Float64

Internal dispatch for mode coupling coefficient based on harmonic types.
"""
function _coupling_dispatch(l::Int, m::Int, l1::Int, m1::Int, l2::Int, m2::Int,
                            type1::Symbol, type2::Symbol)
    # Scalar-scalar: Gaunt integral
    if type1 == :scalar && type2 == :scalar
        return gaunt_integral(l1, m1, l2, m2, l, m)
    end

    # Even vector - even vector: vector Gaunt
    if type1 == :even_vector && type2 == :even_vector
        return vector_gaunt(l1, m1, l2, m2, l, m)
    end

    # Odd vector - odd vector: same coupling as even-even by parity symmetry on S^2
    if type1 == :odd_vector && type2 == :odd_vector
        return vector_gaunt(l1, m1, l2, m2, l, m)
    end

    # Cross-parity vector pairs vanish
    if (type1 == :even_vector && type2 == :odd_vector) ||
       (type1 == :odd_vector && type2 == :even_vector)
        return 0.0
    end

    # Tensor-tensor: dispatch to tensor_gaunt with appropriate type symbols
    ttype1 = _to_tensor_gaunt_type(type1)
    ttype2 = _to_tensor_gaunt_type(type2)
    if ttype1 !== nothing && ttype2 !== nothing
        return tensor_gaunt(l1, m1, l2, m2, l, m, ttype1, ttype2)
    end

    # Scalar-vector (no index contraction): vanishes by parity
    if (type1 == :scalar && _is_vector_type(type2)) ||
       (_is_vector_type(type1) && type2 == :scalar)
        return 0.0
    end

    # Scalar-tensor (no contraction): vanishes by rank mismatch
    if (type1 == :scalar && _is_tensor_type(type2)) ||
       (_is_tensor_type(type1) && type2 == :scalar)
        return 0.0
    end

    # Vector-tensor (no full contraction possible): vanishes
    if (_is_vector_type(type1) && _is_tensor_type(type2)) ||
       (_is_tensor_type(type1) && _is_vector_type(type2))
        return 0.0
    end

    error("Unsupported harmonic type combination: type1=$type1, type2=$type2")
end

"""Map user-facing type symbol to tensor_gaunt internal type symbol."""
function _to_tensor_gaunt_type(t::Symbol)
    t == :even_tensor_Y && return :Y
    t == :even_tensor_Z && return :Z
    t == :odd_tensor && return :X
    nothing
end

_is_vector_type(t::Symbol) = t == :even_vector || t == :odd_vector
_is_tensor_type(t::Symbol) = t == :even_tensor_Y || t == :even_tensor_Z || t == :odd_tensor

# ---- Mode coupling table ----------------------------------------------------

"""
    ModeCouplingTable

Stores precomputed mode coupling coefficients C^{l1 m1, l2 m2}_{l m} up to
a given l_max.

The table is indexed by a NamedTuple key:
    (l=l, m=m, l1=l1, m1=m1, l2=l2, m2=m2) => Float64

For tensor harmonic couplings, the types are stored in the key as well.

Fields:
- `l_max::Int` -- maximum angular momentum computed
- `entries::Dict` -- (l,m,l1,m1,l2,m2,type1,type2) => coefficient
- `types::Vector{Tuple{Symbol,Symbol}}` -- harmonic type pairs included
"""
struct ModeCouplingTable
    l_max::Int
    entries::Dict{NTuple{8,Any}, Float64}
    types::Vector{Tuple{Symbol,Symbol}}

    function ModeCouplingTable(; types::Vector{Tuple{Symbol,Symbol}} = [(:scalar, :scalar)])
        new(0, Dict{NTuple{8,Any}, Float64}(), types)
    end
end

# Allow mutation of l_max via a mutable wrapper
mutable struct _MutableModeCouplingTable
    l_max::Int
    entries::Dict{NTuple{8,Any}, Float64}
    types::Vector{Tuple{Symbol,Symbol}}
end

"""
    ModeCouplingTable(; types=[(:scalar,:scalar)]) -> ModeCouplingTable

Create an empty mode coupling table. Use `compute_coupling_table!` to fill it.

# Example

```julia
table = ModeCouplingTable(types=[(:scalar,:scalar), (:even_vector,:even_vector)])
compute_coupling_table!(table, 4)
```
"""
function ModeCouplingTable(l_max::Int; types::Vector{Tuple{Symbol,Symbol}} = [(:scalar, :scalar)])
    t = _MutableModeCouplingTable(0, Dict{NTuple{8,Any}, Float64}(), types)
    compute_coupling_table!(t, l_max)
    # Return immutable snapshot conceptually (we use the mutable version for table ops)
    t
end

"""
    compute_coupling_table!(table, l_max::Int)

Fill the mode coupling table with all nonzero coefficients for
|l1| <= l_max, |l2| <= l_max, and the resulting |l| up to l1+l2 (capped at 2*l_max).

For each type pair registered in `table.types`, iterates over all valid
(l1, m1, l2, m2) combinations, checks selection rules, and stores nonzero
coupling coefficients.

# Arguments
- `table` -- a `_MutableModeCouplingTable` or `ModeCouplingTable`-like object
- `l_max::Int` -- maximum angular momentum for input modes

# Performance
The total number of entries scales as O(l_max^5) for each type pair (from
the five free parameters l1,m1,l2,m2,l subject to m=m1+m2 and triangle rule).
For l_max=10 this is a few thousand entries per type pair.
"""
function compute_coupling_table!(table::_MutableModeCouplingTable, l_max::Int)
    l_max >= 0 || throw(ArgumentError("l_max must be non-negative, got $l_max"))

    for (type1, type2) in table.types
        # Minimum l for each type
        l1_min = _min_l_for_type(type1)
        l2_min = _min_l_for_type(type2)

        for l1 in l1_min:l_max
            for l2 in l2_min:l_max
                for l in abs(l1 - l2):min(l1 + l2, 2 * l_max)
                    # Parity check
                    isodd(l1 + l2 + l) && continue

                    for m1 in -l1:l1
                        for m2 in -l2:l2
                            m = m1 + m2
                            abs(m) > l && continue

                            val = mode_coupling_coefficient(l, m, l1, m1, l2, m2;
                                                            type1=type1, type2=type2)
                            if abs(val) > 1e-15
                                key = (l, m, l1, m1, l2, m2, type1, type2)
                                table.entries[key] = val
                            end
                        end
                    end
                end
            end
        end
    end

    table.l_max = l_max
    table
end

"""Minimum l for a given harmonic type."""
function _min_l_for_type(t::Symbol)
    t == :scalar && return 0
    t == :even_vector && return 1
    t == :odd_vector && return 1
    t == :even_tensor_Y && return 0
    t == :even_tensor_Z && return 2
    t == :odd_tensor && return 2
    error("Unknown harmonic type: $t")
end

"""
    Base.getindex(table::_MutableModeCouplingTable, l, m, l1, m1, l2, m2;
                  type1=:scalar, type2=:scalar) -> Float64

Look up a coupling coefficient from the table. Returns 0.0 if not found.
"""
function coupling_coefficient(table::_MutableModeCouplingTable,
                              l::Int, m::Int, l1::Int, m1::Int, l2::Int, m2::Int;
                              type1::Symbol=:scalar, type2::Symbol=:scalar)
    get(table.entries, (l, m, l1, m1, l2, m2, type1, type2), 0.0)
end

"""
    coupling_count(table::_MutableModeCouplingTable) -> Int

Return the number of nonzero entries in the coupling table.
"""
coupling_count(table::_MutableModeCouplingTable) = length(table.entries)

# ---- Special cases and identities -------------------------------------------

"""
    _orthonormality_coefficient(l::Int) -> Float64

The coupling coefficient C^{0,0}_{l,0,l,0} = integral Y_{l,0} Y*_{l,0} Y_{0,0} dOmega,
which equals (-1)^l * sqrt((2l+1)/(4pi)) by orthonormality.

This is the Edmonds (1957) Eq 4.6.3 identity:
    integral Y_{lm} Y*_{lm} dOmega = 1
projected onto Y_{0,0} = 1/sqrt(4pi):
    C = integral Y_{l,0} Y_{l,0} Y*_{0,0} dOmega = 1/sqrt(4pi)

(Note: for general m, C^{0,0}_{l,m,l,m} = 1/sqrt(4pi) for all m.)

More precisely: gaunt_integral(l, m, l, m, 0, 0) via the 3j symbols.
"""
function _orthonormality_coefficient(l::Int, m::Int)
    # Direct evaluation: integral Y_{lm} * Y*_{lm} * Y_{00} dOmega
    # Y_{0,0} = 1/sqrt(4pi), so integral = <Y_{lm}|Y_{lm}> * Y_{0,0} conjugated
    # = 1/sqrt(4pi) by orthonormality
    gaunt_integral(l, m, 0, 0, l, m)
end
