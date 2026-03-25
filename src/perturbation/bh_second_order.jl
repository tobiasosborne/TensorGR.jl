#= Second-order perturbation source terms for black hole perturbation theory.
#
# On a vacuum (Ricci-flat) background like Schwarzschild:
#   δ²G_{ab} = S_{ab}[h¹, h¹]
#
# where S is the effective source term quadratic in the first-order
# perturbation h¹. This source drives the second-order perturbation
# equation δ¹G_{ab}[h²] = -S_{ab}[h¹, h¹].
#
# The source decomposes into tensor spherical harmonic modes:
#   S^{(2)}_{lm}(r) = Σ_{l₁m₁ l₂m₂} C^{lm}_{l₁m₁,l₂m₂} · f(r)
#
# where the coupling coefficients C involve angular integrals (Gaunt,
# vector Gaunt, tensor Gaunt) and f(r) involves products of first-order
# radial functions.
#
# References:
#   Brizuela, Martin-Garcia & Tiglio, PRD 80, 024021 (2009), Sec 3.
#   Gleiser, Nicasio, Price & Pullin, Phys. Rep. 325, 41 (2000).
#   Martel & Poisson, PRD 71, 104003 (2005).
=#

"""
    second_order_einstein_source(mp, a, b; registry) -> TensorExpr

Compute the second-order Einstein tensor source term δ²G_{ab} on a
vacuum (Ricci-flat) background.

    δ²G_{ab} = δ²R_{ab} - (1/2)[g_{ab}·δ²R + h_{ab}·δ¹R]

On vacuum background (R₀=0), the term δ²g_{ab}·R₀ vanishes.

This is the effective source for the second-order perturbation equation.
It is bilinear in the first-order perturbation h.

# Arguments
- `mp::MetricPerturbation` -- the perturbation setup (must use `curved=true`)
- `a, b::TIndex` -- Down Tangent indices

# Returns
An unsimplified `TensorExpr`.
"""
function second_order_einstein_source(mp::MetricPerturbation,
                                       a::TIndex, b::TIndex;
                                       registry::TensorRegistry=current_registry())
    a.position === Down || error("second_order_einstein_source: a must be Down")
    b.position === Down || error("second_order_einstein_source: b must be Down")

    # δ²R_{ab}
    d2Ric = δricci(mp, a, b, 2)

    # δ²R (scalar)
    d2R = δricci_scalar(mp, 2)

    # δ¹R (first-order scalar curvature perturbation)
    d1R = δricci_scalar(mp, 1)

    # Background metric
    g_ab = Tensor(mp.metric, [a, b])

    # First-order perturbation
    h_ab = Tensor(mp.perturbation, [a, b])

    # δ²G_{ab} = δ²R_{ab} - (1/2) g_{ab} δ²R - (1/2) h_{ab} δ¹R
    term1 = d2Ric
    term2 = tproduct(-1 // 2, TensorExpr[g_ab, d2R])
    term3 = tproduct(-1 // 2, TensorExpr[h_ab, d1R])

    TSum(TensorExpr[term1, term2, term3])
end

"""
    source_is_bilinear(mp; registry) -> Bool

Verify that the second-order source is bilinear in the first-order
perturbation h. This ensures S^{(2)}[0, 0] = 0 and that the source
is exactly quadratic in h (as required by the perturbation expansion).

Checks that every term in δ²G_{ab} contains exactly 2 powers of h
(counting h and its derivatives ∂h as one power each).
"""
function source_is_bilinear(mp::MetricPerturbation;
                             registry::TensorRegistry=current_registry())
    source = second_order_einstein_source(mp, down(:a), down(:b); registry=registry)
    _check_bilinear(source, mp.perturbation)
end

function _check_bilinear(expr::TSum, h_name::Symbol)
    all(t -> _count_h_powers(t, h_name) == 2, expr.terms)
end

function _check_bilinear(expr::TProduct, h_name::Symbol)
    _count_h_powers(expr, h_name) == 2
end

function _check_bilinear(expr::TensorExpr, h_name::Symbol)
    _count_h_powers(expr, h_name) == 2
end

function _count_h_powers(t::Tensor, h_name::Symbol)
    t.name === h_name ? 1 : 0
end

function _count_h_powers(p::TProduct, h_name::Symbol)
    sum(_count_h_powers(f, h_name) for f in p.factors)
end

function _count_h_powers(s::TSum, h_name::Symbol)
    isempty(s.terms) ? 0 : _count_h_powers(s.terms[1], h_name)
end

function _count_h_powers(d::TDeriv, h_name::Symbol)
    _count_h_powers(d.arg, h_name)
end

function _count_h_powers(::TScalar, ::Symbol)
    0
end

function _count_h_powers(::TensorExpr, ::Symbol)
    0
end

# ── Mode coupling infrastructure ─────────────────────────────────────

"""
    SourceModeCoupling

Represents the coupling of two first-order modes (l₁,m₁) and (l₂,m₂)
into a second-order mode (l,m) with a specific angular coupling coefficient.

# Fields
- `l::Int` -- output angular momentum
- `m::Int` -- output magnetic quantum number
- `l1::Int, m1::Int` -- first source mode
- `l2::Int, m2::Int` -- second source mode
- `parity1::Symbol` -- parity of first source (:even or :odd)
- `parity2::Symbol` -- parity of second source (:even or :odd)
"""
struct SourceModeCoupling
    l::Int
    m::Int
    l1::Int
    m1::Int
    l2::Int
    m2::Int
    parity1::Symbol
    parity2::Symbol
end

function Base.show(io::IO, c::SourceModeCoupling)
    print(io, "(", c.l, ",", c.m, ") ← (",
          c.l1, ",", c.m1, ")[", c.parity1, "] × (",
          c.l2, ",", c.m2, ")[", c.parity2, "]")
end

"""
    source_coupling_modes(l, m, lmax) -> Vector{SourceModeCoupling}

Enumerate all (l₁,m₁,l₂,m₂) mode pairs that can couple into the
target mode (l,m) at second order, respecting selection rules:

1. m-conservation: m₁ + m₂ = m
2. Triangle inequality: |l₁ - l₂| ≤ l ≤ l₁ + l₂
3. Parity: l₁ + l₂ + l must be even (for scalar-type coupling)

For each allowed mode pair, both even×even and odd×odd parity
combinations are included (even×odd vanishes by parity).

# Arguments
- `l::Int` -- target angular momentum
- `m::Int` -- target magnetic quantum number
- `lmax::Int` -- maximum l to consider for source modes
"""
function source_coupling_modes(l::Int, m::Int, lmax::Int)
    couplings = SourceModeCoupling[]

    for l1 in 0:lmax
        for l2 in 0:lmax
            # Triangle inequality
            l < abs(l1 - l2) && continue
            l > l1 + l2 && continue
            # Parity
            isodd(l1 + l2 + l) && continue

            for m1 in -l1:l1
                m2 = m - m1  # m-conservation
                abs(m2) > l2 && continue

                # Even × Even coupling
                push!(couplings, SourceModeCoupling(l, m, l1, m1, l2, m2,
                                                     :even, :even))
                # Odd × Odd coupling (requires l1 >= 1, l2 >= 1)
                if l1 >= 1 && l2 >= 1
                    push!(couplings, SourceModeCoupling(l, m, l1, m1, l2, m2,
                                                         :odd, :odd))
                end
            end
        end
    end

    couplings
end

"""
    scalar_coupling_coefficient(l, m, l1, m1, l2, m2) -> Float64

Compute the scalar-scalar angular coupling coefficient for the
second-order source. This is the Gaunt integral:

    C^{lm}_{l₁m₁,l₂m₂} = ∫ Y_{l₁m₁} Y_{l₂m₂} Y*_{lm} dΩ

Used for the M²-block (tt, tr, rr) contributions where both first-order
modes contribute through scalar harmonics Y_{lm}.
"""
function scalar_coupling_coefficient(l::Int, m::Int,
                                      l1::Int, m1::Int,
                                      l2::Int, m2::Int)
    gaunt_integral(l1, m1, l2, m2, l, m)
end

"""
    vector_coupling_coefficient(l, m, l1, m1, l2, m2) -> Float64

Compute the vector-vector angular coupling coefficient:

    ∫ Y^A_{l₁m₁} Y_{A,l₂m₂} Y*_{lm} dΩ

Used for the mixed-block contributions where first-order modes
contribute through even vector harmonics.
"""
function vector_coupling_coefficient(l::Int, m::Int,
                                      l1::Int, m1::Int,
                                      l2::Int, m2::Int)
    vector_gaunt(l1, m1, l2, m2, l, m)
end

"""
    tensor_coupling_coefficient(l, m, l1, m1, l2, m2,
                                 type1::Symbol, type2::Symbol) -> Float64

Compute the tensor-tensor angular coupling coefficient:

    ∫ T1^{AB}_{l₁m₁} T2_{AB,l₂m₂} Y*_{lm} dΩ

where type1, type2 ∈ {:Y, :Z, :X} specify the tensor harmonic type.
"""
function tensor_coupling_coefficient(l::Int, m::Int,
                                      l1::Int, m1::Int,
                                      l2::Int, m2::Int,
                                      type1::Symbol, type2::Symbol)
    tensor_gaunt(l1, m1, l2, m2, l, m, type1, type2)
end

"""
    count_coupling_modes(l, m, lmax) -> Int

Count the number of mode coupling pairs for a given target (l,m).
Useful for estimating computational cost.
"""
function count_coupling_modes(l::Int, m::Int, lmax::Int)
    length(source_coupling_modes(l, m, lmax))
end

# ══════════════════════════════════════════════════════════════════════
# Regge-Wheeler and Zerilli master equations
# ══════════════════════════════════════════════════════════════════════

"""
    MasterEquation

Represents a wave equation of the form:

    [-∂²/∂t² + ∂²/∂r*² - V(r)] ψ_{lm} = S_{lm}

in tortoise coordinate r* = r + 2M ln(r/2M - 1).

# Fields
- `parity::Symbol` -- `:odd` (Regge-Wheeler) or `:even` (Zerilli)
- `l::Int` -- angular momentum quantum number
- `potential::Function` -- V(r, M, l) -> potential value
- `potential_name::Symbol` -- `:RW` or `:Zerilli`
"""
struct MasterEquation
    parity::Symbol
    l::Int
    potential::Function
    potential_name::Symbol
end

function Base.show(io::IO, eq::MasterEquation)
    print(io, eq.potential_name, " master equation (l=", eq.l,
          ", ", eq.parity, " parity)")
end

# ── Regge-Wheeler (odd parity) ────────────────────────────────────────

"""
    regge_wheeler_potential(r, M, l) -> Rational or symbolic

The Regge-Wheeler potential for odd-parity perturbations:

    V_RW(r) = (1 - 2M/r) [l(l+1)/r² - 6M/r³]

This is the effective potential in the wave equation for the
Cunningham-Price-Moncrief master variable.

# Arguments
- `r` -- radial coordinate (must be > 2M for exterior)
- `M` -- black hole mass
- `l::Int` -- angular momentum quantum number (l >= 2)

# Ground truth
Regge & Wheeler, Phys. Rev. 108, 1063 (1957).
"""
function regge_wheeler_potential(r, M, l::Int)
    f = 1 - 2M / r  # Schwarzschild factor
    f * (l * (l + 1) / r^2 - 6M / r^3)
end

"""
    regge_wheeler_equation(l; source=nothing) -> MasterEquation

Construct the Regge-Wheeler master equation for angular momentum `l`.

    [-∂²/∂t² + ∂²/∂r*² - V_RW(r)] ψ^{RW}_{lm} = S^{odd}_{lm}

At first order (source=nothing), the RHS is zero. At second order,
the source S^{odd} comes from mode-coupled products of first-order
perturbations.

# Example
```julia
eq = regge_wheeler_equation(2)
V = eq.potential(10.0, 1.0, 2)  # evaluate potential at r=10, M=1
```
"""
function regge_wheeler_equation(l::Int; source=nothing)
    l >= 2 || error("regge_wheeler_equation: l must be >= 2, got $l")
    MasterEquation(:odd, l, regge_wheeler_potential, :RW)
end

# ── Zerilli (even parity) ─────────────────────────────────────────────

"""
    zerilli_potential(r, M, l) -> Rational or symbolic

The Zerilli potential for even-parity perturbations:

    V_Z(r) = f(r) · [2λ²(λ+1)r³ + 6λ²Mr² + 18λM²r + 18M³]
                     / [r³(λr + 3M)²]

where f(r) = 1 - 2M/r and λ = (l-1)(l+2)/2.

# Ground truth
Zerilli, Phys. Rev. D 2, 2141 (1970).
Brizuela, Martin-Garcia & Tiglio, PRD 80, 024021 (2009), Eq 4.12.
"""
function zerilli_potential(r, M, l::Int)
    f = 1 - 2M / r
    lambda = (l - 1) * (l + 2) / 2
    numerator = 2lambda^2 * (lambda + 1) * r^3 +
                6lambda^2 * M * r^2 +
                18lambda * M^2 * r +
                18M^3
    denominator = r^3 * (lambda * r + 3M)^2
    f * numerator / denominator
end

"""
    zerilli_equation(l; source=nothing) -> MasterEquation

Construct the Zerilli master equation for angular momentum `l`.

    [-∂²/∂t² + ∂²/∂r*² - V_Z(r)] ψ^{Z}_{lm} = S^{even}_{lm}

# Example
```julia
eq = zerilli_equation(2)
V = eq.potential(10.0, 1.0, 2)  # evaluate potential at r=10, M=1
```
"""
function zerilli_equation(l::Int; source=nothing)
    l >= 2 || error("zerilli_equation: l must be >= 2, got $l")
    MasterEquation(:even, l, zerilli_potential, :Zerilli)
end

# ── Tortoise coordinate ───────────────────────────────────────────────

"""
    tortoise_coordinate(r, M) -> Float64

Compute the tortoise coordinate r* = r + 2M ln(r/(2M) - 1).

Only valid for r > 2M (exterior region).
"""
function tortoise_coordinate(r::Real, M::Real)
    r > 2M || error("tortoise_coordinate: r must be > 2M")
    r + 2M * log(r / (2M) - 1)
end

"""
    inverse_tortoise(rstar, M; tol=1e-12, maxiter=100) -> Float64

Invert the tortoise coordinate: given r*, find r such that
r* = r + 2M ln(r/(2M) - 1).

Uses Newton's method starting from r₀ = max(r*, 2M + 1).
"""
function inverse_tortoise(rstar::Real, M::Real; tol::Real=1e-12, maxiter::Int=100)
    # Initial guess
    r = max(rstar, 2M + 1.0)
    for _ in 1:maxiter
        f = r + 2M * log(r / (2M) - 1) - rstar
        df = r / (r - 2M)  # dr*/dr = 1/(1 - 2M/r) = r/(r-2M)
        dr = -f / df
        r += dr
        abs(dr) < tol && return r
    end
    error("inverse_tortoise: Newton's method did not converge")
end

# ── Potential evaluation utilities ────────────────────────────────────

"""
    evaluate_potential(eq::MasterEquation, r, M) -> Number

Evaluate the potential V(r) for a master equation at given r and M.
"""
function evaluate_potential(eq::MasterEquation, r, M)
    eq.potential(r, M, eq.l)
end

"""
    potential_at_horizon(eq::MasterEquation, M) -> Float64

The potential at the horizon r = 2M vanishes for both RW and Zerilli:
V(r=2M) = 0 (since f(2M) = 0).
"""
function potential_at_horizon(eq::MasterEquation, M)
    # f(2M) = 0, so V(2M) = 0 for both potentials
    0.0
end

"""
    potential_at_infinity(eq::MasterEquation) -> Float64

The potential at spatial infinity r → ∞ vanishes: V(r→∞) → 0.
Both RW and Zerilli potentials decay as 1/r² at large r.
"""
function potential_at_infinity(eq::MasterEquation)
    0.0
end
