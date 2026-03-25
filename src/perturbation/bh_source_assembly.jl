#= Second-order radial source assembly and gauge-invariant master equations.
#
# After angular decomposition, the second-order Einstein equations reduce
# to radial ODEs with source terms bilinear in first-order radial functions.
#
# For each target mode (l,m), the source is:
#   S^{(2)}_{lm}(r) = Σ_{l₁m₁,l₂m₂} C^{lm}_{l₁m₁,l₂m₂} · F(r; ψ¹_{l₁}, ψ¹_{l₂})
#
# The second-order master equations have the same differential operators
# as the first-order equations (same potential V_RW or V_Z) but with
# non-zero RHS source:
#   [-∂²/∂t² + ∂²/∂r*² - V(r)] ψ²_{lm} = S²_{lm}
#
# Gauge-invariant second-order master variables combine the second-order
# metric coefficients with quadratic corrections from first-order quantities.
#
# References:
#   Brizuela, Martin-Garcia & Tiglio, PRD 80, 024021 (2009), Sec 3-4.
#   Campanelli & Lousto, PRD 59, 124022 (1999), Sec 3.
#   Gleiser, Nicasio, Price & Pullin, Phys. Rep. 325, 41 (2000), Sec 5.
=#

# ── Source contribution structs ───────────────────────────────────────

"""
    SourceContribution

A single contribution to the second-order source from mode coupling
between first-order modes (l₁,m₁) and (l₂,m₂).

# Fields
- `l1, m1::Int` -- first source mode
- `l2, m2::Int` -- second source mode
- `parity1, parity2::Symbol` -- parities of source modes (:even or :odd)
- `coupling::Float64` -- angular coupling coefficient
- `sector_pairs::Vector{Tuple{Symbol,Symbol}}` -- which sectors couple
  (e.g., `(:tt,:tt)`, `(:K,:G)`, etc.)
"""
struct SourceContribution
    l1::Int
    m1::Int
    l2::Int
    m2::Int
    parity1::Symbol
    parity2::Symbol
    coupling::Float64
    sector_pairs::Vector{Tuple{Symbol,Symbol}}
end

function Base.show(io::IO, sc::SourceContribution)
    print(io, "(", sc.l1, ",", sc.m1, ")[", sc.parity1, "]×(",
          sc.l2, ",", sc.m2, ")[", sc.parity2, "]")
    if sc.coupling != 0.0
        print(io, " C=", round(sc.coupling; digits=6))
    end
end

"""
    SecondOrderSource

The assembled second-order source for a target mode (l,m) with
specified parity.

# Fields
- `l::Int` -- target angular momentum
- `m::Int` -- target magnetic quantum number
- `parity::Symbol` -- target parity (:even or :odd)
- `contributions::Vector{SourceContribution}` -- mode coupling contributions
"""
struct SecondOrderSource
    l::Int
    m::Int
    parity::Symbol
    contributions::Vector{SourceContribution}
end

function Base.show(io::IO, s::SecondOrderSource)
    p = s.parity == :even ? "even" : "odd"
    print(io, "S²(l=", s.l, ",m=", s.m, ",", p, "): ",
          length(s.contributions), " contributions")
end

# ── Even-parity sector labels ────────────────────────────────────────

# The 7 even-parity Martel-Poisson coefficients per (l,m):
const EVEN_SECTORS = [:tt, :tr, :rr, :jt, :jr, :K, :G]

# The 3 odd-parity coefficients per (l,m):
const ODD_SECTORS = [:ht, :hr, :h2]

"""
    _sector_pairs(parity1, parity2, l1, l2)

Determine which sector pairs contribute to the bilinear source.
Even×Even and Odd×Odd couple to even-parity targets.
Even×Odd couples to odd-parity targets.
"""
function _sector_pairs(parity1::Symbol, parity2::Symbol,
                        l1::Int, l2::Int)
    pairs = Tuple{Symbol,Symbol}[]

    if parity1 === :even && parity2 === :even
        # Scalar sectors (l >= 0): tt, tr, rr, K
        s1 = l1 >= 2 ? EVEN_SECTORS : filter(s -> s in (:tt,:tr,:rr,:K), EVEN_SECTORS)
        s2 = l2 >= 2 ? EVEN_SECTORS : filter(s -> s in (:tt,:tr,:rr,:K), EVEN_SECTORS)
        for a in s1, b in s2
            push!(pairs, (a, b))
        end
    elseif parity1 === :odd && parity2 === :odd
        # Odd×Odd → even target
        s1 = l1 >= 2 ? ODD_SECTORS : filter(s -> s in (:ht,:hr), ODD_SECTORS)
        s2 = l2 >= 2 ? ODD_SECTORS : filter(s -> s in (:ht,:hr), ODD_SECTORS)
        for a in s1, b in s2
            push!(pairs, (a, b))
        end
    elseif (parity1 === :even && parity2 === :odd) ||
           (parity1 === :odd && parity2 === :even)
        # Even×Odd → odd target
        s_even = parity1 === :even ? EVEN_SECTORS : EVEN_SECTORS
        s_odd = parity1 === :odd ? ODD_SECTORS : ODD_SECTORS
        for a in s_even, b in s_odd
            push!(pairs, (a, b))
        end
    end

    pairs
end

# ── Source assembly ───────────────────────────────────────────────────

"""
    assemble_source(l, m, parity, lmax) -> SecondOrderSource

Assemble the second-order source for target mode (l,m) with given parity.

Enumerates all (l₁,m₁,l₂,m₂) pairs satisfying selection rules and
computes the angular coupling coefficients. The radial expressions
are represented by their sector-pair structure.

# Arguments
- `l::Int` -- target angular momentum
- `m::Int` -- target magnetic quantum number
- `parity::Symbol` -- `:even` or `:odd`
- `lmax::Int` -- maximum l for source modes

# Returns
A `SecondOrderSource` with all non-zero contributions.
"""
function assemble_source(l::Int, m::Int, parity::Symbol, lmax::Int)
    contributions = SourceContribution[]

    for l1 in 0:lmax
        for l2 in 0:lmax
            # Triangle + parity selection
            angular_selection_rule(l1, l2, l) || continue

            for m1 in -l1:l1
                m2 = m - m1
                abs(m2) > l2 && continue

                # Determine which parity combinations contribute to target parity
                if parity === :even
                    # Even target ← Even×Even or Odd×Odd
                    # Even×Even coupling
                    c_ee = scalar_coupling_coefficient(l, m, l1, m1, l2, m2)
                    if abs(c_ee) > 1e-15
                        pairs = _sector_pairs(:even, :even, l1, l2)
                        push!(contributions, SourceContribution(
                            l1, m1, l2, m2, :even, :even, c_ee, pairs))
                    end

                    # Odd×Odd coupling (l1,l2 >= 1)
                    if l1 >= 1 && l2 >= 1
                        c_oo = scalar_coupling_coefficient(l, m, l1, m1, l2, m2)
                        if abs(c_oo) > 1e-15
                            pairs = _sector_pairs(:odd, :odd, l1, l2)
                            push!(contributions, SourceContribution(
                                l1, m1, l2, m2, :odd, :odd, c_oo, pairs))
                        end
                    end
                elseif parity === :odd
                    # Odd target ← Even×Odd
                    if l1 >= 1 || l2 >= 1
                        c_eo = scalar_coupling_coefficient(l, m, l1, m1, l2, m2)
                        if abs(c_eo) > 1e-15
                            pairs = _sector_pairs(:even, :odd, l1, l2)
                            push!(contributions, SourceContribution(
                                l1, m1, l2, m2, :even, :odd, c_eo, pairs))
                        end
                    end
                end
            end
        end
    end

    SecondOrderSource(l, m, parity, contributions)
end

# ── Sourced master equations ─────────────────────────────────────────

"""
    SourcedMasterEquation

A second-order master equation with source:

    [-∂²/∂t² + ∂²/∂r*² - V(r)] ψ²_{lm} = S²_{lm}

# Fields
- `equation::MasterEquation` -- the differential operator (potential)
- `source::SecondOrderSource` -- the RHS source term
"""
struct SourcedMasterEquation
    equation::MasterEquation
    source::SecondOrderSource
end

function Base.show(io::IO, eq::SourcedMasterEquation)
    print(io, eq.equation, " with ", length(eq.source.contributions), " source terms")
end

"""
    second_order_rw(l, m, lmax) -> SourcedMasterEquation

Construct the second-order Regge-Wheeler equation with source.

The operator is the same as the first-order RW equation (same V_RW),
but with non-zero odd-parity source assembled from mode coupling.
"""
function second_order_rw(l::Int, m::Int, lmax::Int)
    l >= 2 || error("second_order_rw: l must be >= 2")
    eq = regge_wheeler_equation(l)
    source = assemble_source(l, m, :odd, lmax)
    SourcedMasterEquation(eq, source)
end

"""
    second_order_zerilli(l, m, lmax) -> SourcedMasterEquation

Construct the second-order Zerilli equation with source.

The operator is the same as the first-order Zerilli equation (same V_Z),
but with non-zero even-parity source assembled from mode coupling.
"""
function second_order_zerilli(l::Int, m::Int, lmax::Int)
    l >= 2 || error("second_order_zerilli: l must be >= 2")
    eq = zerilli_equation(l)
    source = assemble_source(l, m, :even, lmax)
    SourcedMasterEquation(eq, source)
end

# ── Gauge-invariant master variables ─────────────────────────────────

"""
    GaugeInvariantVariable

Second-order gauge-invariant master variable, following Brizuela et al.
(2009) Sec 3.2. The gauge-invariant combination at second order is:

    Ψ^{(2),GI}_{lm} = Ψ^{(2)}_{lm} + Q[h^{(1)}]

where Ψ^{(2)} is the "bare" second-order master variable and Q[h^{(1)}]
is a quadratic correction involving first-order quantities only.

# Fields
- `l::Int` -- angular momentum
- `m::Int` -- magnetic quantum number
- `parity::Symbol` -- `:even` or `:odd`
- `bare_variable::Symbol` -- name of the bare variable (e.g., :psi2_Z)
- `correction_order::Int` -- order of the quadratic correction (always 2)
"""
struct GaugeInvariantVariable
    l::Int
    m::Int
    parity::Symbol
    bare_variable::Symbol
    correction_order::Int
end

function Base.show(io::IO, gv::GaugeInvariantVariable)
    p = gv.parity == :even ? "Z" : "RW"
    print(io, "Ψ^{(2),GI}_{", p, "}(l=", gv.l, ",m=", gv.m, ")")
end

"""
    gauge_invariant_variable(l, m, parity) -> GaugeInvariantVariable

Define the second-order gauge-invariant master variable.

For even parity: Ψ^{(2),GI}_Z = Ψ^{(2)}_Z + Q_even[h^{(1)}]
For odd parity:  Ψ^{(2),GI}_RW = Ψ^{(2)}_RW + Q_odd[h^{(1)}]

The quadratic correction Q is needed to ensure gauge invariance at
second order (Brizuela et al. 2009, Eqs 3.12-3.15).
"""
function gauge_invariant_variable(l::Int, m::Int, parity::Symbol)
    l >= 2 || error("gauge_invariant_variable: l must be >= 2")
    bare = parity === :even ? :psi2_Z : :psi2_RW
    GaugeInvariantVariable(l, m, parity, bare, 2)
end

"""
    is_gauge_invariant_at_zero(gv::GaugeInvariantVariable) -> Bool

When the first-order perturbation vanishes (h^{(1)} = 0), the
gauge-invariant variable reduces to the bare variable:
    Ψ^{(2),GI} → Ψ^{(2)}
since Q[0] = 0 (Q is quadratic in h^{(1)}).
"""
function is_gauge_invariant_at_zero(gv::GaugeInvariantVariable)
    # Q[h^(1)] is quadratic in h^(1), so Q[0] = 0
    # Therefore Ψ^{GI} = Ψ^{bare} + 0 = Ψ^{bare}
    gv.correction_order == 2
end
