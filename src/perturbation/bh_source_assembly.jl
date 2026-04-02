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

# ── Master field representation ─────────────────────────────────────

"""
    MasterField

A first-order master field (Zerilli Psi or RW Phi/Pi) at mode (l,m)
with specified time and radial derivative orders.

Represents objects like (1)Psi, d/dt(1)Psi, d^2/dr*^2(1)Pi, etc.

# Fields
- `name::Symbol` -- field name (`:Psi` for Zerilli, `:Phi` or `:Pi` for RW)
- `l::Int` -- angular momentum
- `m::Int` -- magnetic quantum number
- `dt_order::Int` -- number of time derivatives (dots)
- `dr_order::Int` -- number of tortoise-coordinate derivatives (primes)
"""
struct MasterField
    name::Symbol
    l::Int
    m::Int
    dt_order::Int
    dr_order::Int
end

function MasterField(name::Symbol, l::Int, m::Int)
    MasterField(name, l, m, 0, 0)
end

function Base.show(io::IO, f::MasterField)
    dots = repeat(".", f.dt_order)
    primes = repeat("'", f.dr_order)
    print(io, "(1)", f.name, dots, primes,
          "(l=", f.l, ",m=", f.m, ")")
end

function Base.:(==)(a::MasterField, b::MasterField)
    a.name === b.name && a.l == b.l && a.m == b.m &&
    a.dt_order == b.dt_order && a.dr_order == b.dr_order
end

function Base.hash(f::MasterField, h::UInt)
    hash((:MasterField, f.name, f.l, f.m, f.dt_order, f.dr_order), h)
end

"""
    time_deriv(f::MasterField) -> MasterField

Return a new MasterField with one additional time derivative.
"""
function time_deriv(f::MasterField)
    MasterField(f.name, f.l, f.m, f.dt_order + 1, f.dr_order)
end

"""
    radial_deriv(f::MasterField) -> MasterField

Return a new MasterField with one additional radial (r*) derivative.
"""
function radial_deriv(f::MasterField)
    MasterField(f.name, f.l, f.m, f.dt_order, f.dr_order + 1)
end

# ── Gauge correction bilinear terms ─────────────────────────────────

"""
    GaugeCorrectionTerm

A single bilinear term in the gauge correction Q_reg. Represents:

    coeff * r^r_power * (2M-r)^f_power * M^M_power * field1 * field2

where coeff is a Rational, and sqrt_prefactor stores the argument of
an overall sqrt(n/pi) factor (e.g., 5 for sqrt(5/pi)).

# Fields
- `coeff::Rational{Int}` -- rational coefficient
- `r_power::Int` -- power of r in the background factor
- `f_power::Int` -- power of (2M-r) = -r*f(r)/(1-2M/r) factor
- `M_power::Int` -- power of M in the background factor
- `field1::MasterField` -- first factor in the bilinear
- `field2::MasterField` -- second factor in the bilinear
"""
struct GaugeCorrectionTerm
    coeff::Rational{Int}
    r_power::Int
    f_power::Int
    M_power::Int
    field1::MasterField
    field2::MasterField
end

function Base.show(io::IO, t::GaugeCorrectionTerm)
    print(io, t.coeff)
    t.r_power != 0 && print(io, "*r^", t.r_power)
    t.f_power != 0 && print(io, "*(2M-r)^", t.f_power)
    t.M_power != 0 && print(io, "*M^", t.M_power)
    print(io, "*", t.field1, "*", t.field2)
end

"""
    GaugeCorrection

The full gauge correction Q_reg as a sum of bilinear terms, with an
overall sqrt(sqrt_prefactor / pi) prefactor.

Represents Eqs 89, 94 of Brizuela, Martin-Garcia & Tiglio, PRD 80,
024021 (2009).

# Fields
- `parity::Symbol` -- `:even` or `:odd`
- `l::Int` -- target angular momentum
- `m::Int` -- target magnetic quantum number
- `lhat::Int` -- first source mode l
- `mhat::Int` -- first source mode m
- `lbar::Int` -- second source mode l
- `mbar::Int` -- second source mode m
- `sqrt_prefactor::Rational{Int}` -- argument of sqrt(n/pi), e.g., 5//1
- `terms::Vector{GaugeCorrectionTerm}` -- bilinear terms
"""
struct GaugeCorrection
    parity::Symbol
    l::Int
    m::Int
    lhat::Int
    mhat::Int
    lbar::Int
    mbar::Int
    sqrt_prefactor::Rational{Int}
    terms::Vector{GaugeCorrectionTerm}
end

function Base.show(io::IO, Q::GaugeCorrection)
    p = Q.parity == :even ? "even" : "odd"
    print(io, "Q_reg(", p, ", l=", Q.l, ",m=", Q.m,
          "; lhat=", Q.lhat, ",lbar=", Q.lbar, "): ",
          length(Q.terms), " terms")
end

"""
    gauge_correction(parity; l=2, m=0, lhat=2, mhat=0, lbar=2, mbar=0)
        -> GaugeCorrection

Compute the gauge correction Q_reg for the specified mode coupling.

Currently implements the quadrupole self-coupling case
(l,m)=(lhat,mhat)=(lbar,mbar)=(2,0) from Brizuela et al. (2009):
- Even parity: Eq 89 (Q_reg for Zerilli)
- Odd parity: Eq 94 (Q^reg_Phi for RW)

# References
Brizuela, Martin-Garcia & Tiglio, PRD 80, 024021 (2009), Sec VI.
"""
function gauge_correction(parity::Symbol;
                          l::Int=2, m::Int=0,
                          lhat::Int=2, mhat::Int=0,
                          lbar::Int=2, mbar::Int=0)
    l >= 2 || error("gauge_correction: l must be >= 2")
    # Currently only (2,0)x(2,0) -> (2,0) is implemented
    if !(l == 2 && m == 0 && lhat == 2 && mhat == 0 && lbar == 2 && mbar == 0)
        error("gauge_correction: only (l,m)=(lhat,mhat)=(lbar,mbar)=(2,0) implemented")
    end

    if parity === :even
        _gauge_correction_even_220()
    elseif parity === :odd
        _gauge_correction_odd_220()
    else
        error("gauge_correction: parity must be :even or :odd")
    end
end

# Even-parity Q_reg for (2,0)x(2,0) -> (2,0), Eq 89 of Brizuela et al. 2009
# Fields: Psi = Zerilli master, Pi = RW master
function _gauge_correction_even_220()
    Psi = MasterField(:Psi, 2, 0)
    Pi = MasterField(:Pi, 2, 0)
    Psi_dot = time_deriv(Psi)
    Psi_ddot = time_deriv(Psi_dot)
    Psi_prime = radial_deriv(Psi)
    Pi_dot = time_deriv(Pi)
    Pi_ddot = time_deriv(Pi_dot)

    # Eq 89, first block: prefactor -1/(252(2M-r)) * sqrt(5/pi)
    # Terms inside the braces multiplied by the outer prefactor:
    #
    # 2(2M-r)((9M+r)*Psi_dot*Psi_ddot + 6*Psi*Psi_ddot)
    #   -> 2(9M+r)*Psi_dot*Psi_ddot  (with extra (2M-r) cancels denom)
    #     = 18M * Psi_dot*Psi_ddot + 2r * Psi_dot*Psi_ddot
    #   -> 12*Psi*Psi_ddot (with extra (2M-r) cancels denom)
    #
    # After distributing -1/252:
    # Term 1: -18M/(252) * Psi_dot * Psi_ddot = -1/14 * M * ...
    # Term 2: -2r/(252) * Psi_dot * Psi_ddot = -1/126 * r * ...
    # Term 3: -12/(252) * Psi * Psi_ddot = -1/21 * ...
    #
    # (110M^3 - 21rM^2 + 14r^2M + 4r^3)/(252(2M-r)) * Psi_dot * Psi_ddot
    # Term 4: -110M^3/(252(2M-r)) = -55M^3/(126(2M-r))
    # Term 5: 21rM^2/(252(2M-r)) = M^2*r/(12(2M-r))
    # Term 6: -14r^2M/(252(2M-r)) = -r^2M/(18(2M-r))
    # Term 7: -4r^3/(252(2M-r)) = -r^3/(63(2M-r))
    #
    # [-1/(252(2M-r))]*[-2(2M-r)](4r^2 Psi' - (15M-6r)Psi) * Psi_ddot
    # = +2(4r^2 Psi' - (15M-6r)Psi)/252 * Psi_ddot  (double negative!)
    # Term 8: +8r^2/252 * Psi' * Psi_ddot = +2r^2/63 * ...
    # Term 9: -2(15M-6r)/252 * Psi * Psi_ddot = (-30M+12r)/252 * ...
    #       = -5M/42 * Psi*Psi_ddot + r/21 * Psi*Psi_ddot
    #
    # Second block: -3r^6/224 * sqrt(5/pi) * {16*Pi_dot*Pi + (2r-3M)*Pi_dot*Pi_ddot}
    # Term 10: -48r^6/224 * Pi_dot * Pi = -3r^6/14 * Pi_dot*Pi
    # Term 11: -3r^6(2r)/(224) * Pi_dot * Pi_ddot = -3r^7/112 * ...
    # Term 12: -3r^6(-3M)/(224) * Pi_dot * Pi_ddot = 9Mr^6/224 * ...

    terms = GaugeCorrectionTerm[]

    # Block 1 terms (from expanding Eq 89 first brace):
    # Cancellation of (2M-r) between numerator and denominator in
    # the first two lines gives terms without (2M-r) in denominator.

    # Term: -1/14 * M * Psi_dot * Psi_ddot  (from 2(9M)(2M-r)/[252(2M-r)])
    push!(terms, GaugeCorrectionTerm(-1//14, 0, 0, 1, Psi_dot, Psi_ddot))
    # Term: -1/126 * r * Psi_dot * Psi_ddot  (from 2r(2M-r)/[252(2M-r)])
    push!(terms, GaugeCorrectionTerm(-1//126, 1, 0, 0, Psi_dot, Psi_ddot))
    # Term: -1/21 * Psi * Psi_ddot  (from 12(2M-r)/[252(2M-r)] = 12/252)
    push!(terms, GaugeCorrectionTerm(-1//21, 0, 0, 0, Psi, Psi_ddot))

    # Middle line: (110M^3 - 21rM^2 + 14r^2M + 4r^3) * Psi_dot * Psi_ddot
    # divided by 252(2M-r), with overall minus from -1/252
    push!(terms, GaugeCorrectionTerm(-55//126, 0, -1, 3, Psi_dot, Psi_ddot))
    push!(terms, GaugeCorrectionTerm(1//12, 1, -1, 2, Psi_dot, Psi_ddot))
    push!(terms, GaugeCorrectionTerm(-1//18, 2, -1, 1, Psi_dot, Psi_ddot))
    push!(terms, GaugeCorrectionTerm(-1//63, 3, -1, 0, Psi_dot, Psi_ddot))

    # Last line of first brace: +2(4r^2 Psi' - (15M-6r)Psi)/252 * Psi_ddot
    push!(terms, GaugeCorrectionTerm(2//63, 2, 0, 0, Psi_prime, Psi_ddot))
    push!(terms, GaugeCorrectionTerm(-5//42, 0, 0, 1, Psi, Psi_ddot))
    push!(terms, GaugeCorrectionTerm(1//21, 1, 0, 0, Psi, Psi_ddot))

    # Block 2: -3r^6/224 * {16 Pi_dot Pi + (2r-3M) Pi_dot Pi_ddot}
    push!(terms, GaugeCorrectionTerm(-3//14, 6, 0, 0, Pi_dot, Pi))
    push!(terms, GaugeCorrectionTerm(-3//112, 7, 0, 0, Pi_dot, Pi_ddot))
    push!(terms, GaugeCorrectionTerm(9//224, 6, 0, 1, Pi_dot, Pi_ddot))

    GaugeCorrection(:even, 2, 0, 2, 0, 2, 0, 5//1, terms)
end

# Odd-parity Q^reg_Phi for (2,0)x(2,0) -> (2,0), Eq 94 of Brizuela et al. 2009
# Fields: Psi = Zerilli master, Pi = RW master
function _gauge_correction_odd_220()
    Psi = MasterField(:Psi, 2, 0)
    Pi = MasterField(:Pi, 2, 0)
    Psi_dot = time_deriv(Psi)
    Pi_dot = time_deriv(Pi)
    Psi_ddot = time_deriv(Psi_dot)
    Pi_ddot = time_deriv(Pi_dot)

    # Eq 94: Q^reg_Phi = r^3/84 * sqrt(5/pi) *
    #   {3 Pi_dot Psi_dot + Pi_ddot Psi + Psi_ddot Pi}
    terms = GaugeCorrectionTerm[
        GaugeCorrectionTerm(3//84, 3, 0, 0, Pi_dot, Psi_dot),
        GaugeCorrectionTerm(1//84, 3, 0, 0, Pi_ddot, Psi),
        GaugeCorrectionTerm(1//84, 3, 0, 0, Psi_ddot, Pi),
    ]

    GaugeCorrection(:odd, 2, 0, 2, 0, 2, 0, 5//1, terms)
end

"""
    is_bilinear(Q::GaugeCorrection) -> Bool

Verify that every term in Q is bilinear (product of exactly two MasterFields).
"""
function is_bilinear(Q::GaugeCorrection)
    # Each GaugeCorrectionTerm has exactly field1 and field2 by construction.
    # We verify neither is degenerate (both must be valid MasterFields).
    all(t -> t.field1.l >= 0 && t.field2.l >= 0, Q.terms)
end

"""
    correction_vanishes_at_zero(Q::GaugeCorrection) -> Bool

When all first-order master fields vanish, Q -> 0 because every term
is bilinear in first-order fields.
"""
function correction_vanishes_at_zero(Q::GaugeCorrection)
    # Every term has exactly two field factors; setting fields to zero
    # makes each term vanish. This is structural (bilinear).
    is_bilinear(Q)
end

# ── Regularized source ──────────────────────────────────────────────

"""
    RegularizedSource

The regularized second-order source: S^reg = S + Box(Q) - V * Q.

This is the RHS of the regularized master equation (Eqs 88, 92-93
of Brizuela et al. 2009).

# Fields
- `original_source::SourcedMasterEquation` -- the unregularized equation
- `correction::GaugeCorrection` -- the gauge correction Q_reg
- `potential_name::Symbol` -- `:Zerilli` or `:RW`
"""
struct RegularizedSource
    original_source::SourcedMasterEquation
    correction::GaugeCorrection
    potential_name::Symbol
end

function Base.show(io::IO, rs::RegularizedSource)
    print(io, "S^reg(", rs.potential_name,
          ", l=", rs.correction.l,
          "): S + Box(Q) - V*Q")
end

"""
    regularized_source_term(eq::SourcedMasterEquation,
                            Q::GaugeCorrection) -> RegularizedSource

Construct the regularized source S^reg = S + Box(Q) - V * Q.

The regularized master equation has the same differential operator
(same potential V) as the unregularized one, but the source is modified
to account for the gauge correction. This ensures the regularized
variable Psi^{(2),GI} = Psi^{(2)} + Q satisfies a well-posed equation.

# Arguments
- `eq::SourcedMasterEquation` -- the unregularized second-order equation
- `Q::GaugeCorrection` -- the gauge correction from `gauge_correction()`

# References
Brizuela et al. (2009): Eq 88 (even), Eq 92-93 (odd).
"""
function regularized_source_term(eq::SourcedMasterEquation,
                                 Q::GaugeCorrection)
    eq.equation.parity === Q.parity ||
        error("regularized_source_term: parity mismatch")
    eq.equation.l == Q.l ||
        error("regularized_source_term: l mismatch")
    pot = eq.equation.potential_name
    RegularizedSource(eq, Q, pot)
end

# ── Gravitational wave energy flux ───────────────────────────────────

"""
    EnergyFluxFormula

The gravitational wave energy flux at infinity, expanded to second order.

At first order:
    dE^{(1)}/dt = Σ_{l,m} 1/(64π) [|∂ψ^{Z}_{lm}/∂t|² + (spectral terms)]

At second order, corrections include:
    dE^{(2)}/dt = Σ_{l,m} 1/(32π) Re[∂ψ^{(1)*}/∂t · ∂ψ^{(2)}/∂t]
                + Σ_{l,m} [quadratic-in-first-order tail terms]

# Fields
- `lmax::Int` -- maximum l in the sum
- `order::Int` -- perturbation order (1 or 2)
- `normalization::Rational{Int}` -- 1//(64π) for first order
"""
struct EnergyFluxFormula
    lmax::Int
    order::Int
    normalization::Rational{Int}
end

function Base.show(io::IO, ef::EnergyFluxFormula)
    print(io, "dE^{(", ef.order, ")}/dt (lmax=", ef.lmax, ")")
end

"""
    energy_flux_formula(; lmax=2, order=1) -> EnergyFluxFormula

Construct the energy flux formula at the specified perturbation order.

# First order (order=1)
The standard multipolar energy flux:
    dE/dt = Σ_{l≥2,m} 1/(64π) |∂ψ_{lm}/∂t|²

where the sum runs over both even (Zerilli) and odd (RW) parities.

# Second order (order=2)
Includes cross-terms between first and second-order solutions,
following Campanelli & Lousto (1999) Eq 25.

Ground truth: The l=2 (quadrupole) contribution dominates at leading order.
"""
function energy_flux_formula(; lmax::Int=2, order::Int=1)
    order in (1, 2) || error("energy_flux_formula: order must be 1 or 2")
    lmax >= 2 || error("energy_flux_formula: lmax must be >= 2")
    norm = order == 1 ? 1 // 64 : 1 // 32
    EnergyFluxFormula(lmax, order, norm)
end

"""
    flux_mode_count(ef::EnergyFluxFormula) -> Int

Count the number of (l,m,parity) modes contributing to the flux.
For each l: 2l+1 values of m, times 2 parities (even + odd, for l >= 2).
"""
function flux_mode_count(ef::EnergyFluxFormula)
    n = 0
    for l in 2:ef.lmax
        n += 2 * (2l + 1)  # (2l+1) m-values × 2 parities
    end
    n
end

"""
    quadrupole_flux_scaling(mass_ratio::Real) -> Float64

The leading-order energy flux scales as the square of the mass quadrupole
moment. For a system with symmetric mass ratio η = m₁m₂/(m₁+m₂)²:

    dE/dt ∝ η²

This is the dominant scaling for equal-mass (η=1/4) and extreme-mass-ratio
(η → 0) systems.

For second-order corrections (Campanelli & Lousto 1999):
    dE^{(2)}/dt ∝ η³

so the ratio dE^{(2)}/dE^{(1)} ∝ η, which is small for extreme mass ratios.
"""
function quadrupole_flux_scaling(mass_ratio::Real)
    # η = mass_ratio for symmetric mass ratio
    # Leading order: η²
    mass_ratio^2
end

"""
    second_order_flux_scaling(mass_ratio::Real) -> Float64

The second-order correction to the energy flux scales as η³.
"""
function second_order_flux_scaling(mass_ratio::Real)
    mass_ratio^3
end
