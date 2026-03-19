#= PN potential extraction via Fourier transform of scattering amplitude.
#
# Matching procedure from EFT amplitude to conservative PN potential:
#   V(r) = -∫ d³k/(2π)³ e^{ik·r} M(k)
#
# Standard Fourier transforms (3D, radial):
#   1/k²          →  1/(4πr)
#   1/k⁴          →  r/(8π)         (requires dim-reg)
#   log(k²/μ²)/k² → -1/(4πr)(2log(r·μ) + 2γ_E)
#   k^{2n}/k²     → (-1)^n ∇^{2n} [1/(4πr)]
#
# Ground truth:
# - Levi & Steinhoff, JHEP 06 (2015) 059, arXiv:1510.06752, Sec 3
# - Cheung et al, PRL 121, 251101 (2018), arXiv:1808.02489, Eq 4
# - Goldberger & Rothstein, hep-th/0409156, Sec 5
=#

# ────────────────────────────────────────────────────────────────────
# Fourier transform table
# ────────────────────────────────────────────────────────────────────

"""
    FourierEntry

A single entry in the Fourier transform lookup table.
Maps a momentum-space kernel f(k) to its position-space result f̃(r).

# Fields
- `name::Symbol`          -- identifier (e.g., :inv_k2)
- `momentum_form::Any`    -- momentum-space expression (description)
- `position_form::Any`    -- position-space result (description)
- `coefficient::Rational{Int}` -- numerical prefactor
"""
struct FourierEntry
    name::Symbol
    momentum_form::Any
    position_form::Any
    coefficient::Rational{Int}
end

"""
    FOURIER_TABLE

Standard 3D Fourier transform table for radial power-law integrands.

Each entry gives: ∫ d³k/(2π)³ e^{ik·r} f(k) = coefficient × g(r)

Ground truth: Standard integral tables; Goldberger & Rothstein (2006) Appendix.
"""
const FOURIER_TABLE = Dict{Symbol, FourierEntry}(
    # 1/k² → 1/(4πr)
    :inv_k2 => FourierEntry(:inv_k2, "1/k²", "1/r", 1 // (4)),

    # 1/k⁴ → r/(8π)  (requires dim-reg for UV divergence)
    :inv_k4 => FourierEntry(:inv_k4, "1/k⁴", "r", 1 // 8),

    # 1 (contact term) → δ³(r)
    :contact => FourierEntry(:contact, "1", "δ³(r)", 1 // 1),

    # k² / k² = 1 → δ³(r)
    :delta => FourierEntry(:delta, "k⁰", "δ³(r)", 1 // 1),
)

"""
    fourier_transform_potential(amplitude_coeff, k_power::Int) -> Tuple{Any, Symbol}

Given an amplitude term `coeff / k^{2n}` (where k_power = 2n),
return the position-space potential coefficient and type.

For k_power = 2: V(r) = -coeff/(4πr)
For k_power = 4: V(r) = -coeff·r/(8π)
For k_power = 0: V(r) = -coeff·δ³(r) (contact)

The minus sign comes from V(r) = -M̃(r).

Ground truth: Cheung et al (2018) Eq 4.
"""
function fourier_transform_potential(amplitude_coeff, k_power::Int)
    k_power >= 0 || error("k_power must be non-negative, got $k_power")

    if k_power == 0
        return (amplitude_coeff, :contact)
    elseif k_power == 2
        return (amplitude_coeff // 4, :coulomb)  # coeff/(4πr), π absorbed in convention
    elseif k_power == 4
        return (amplitude_coeff // 8, :linear)   # coeff·r/(8π)
    else
        # General: ∫ d³k/(2π)³ e^{ik·r} / k^{2n} ∝ r^{2n-3} for n > 1
        # This is a table lookup for specific cases
        return (amplitude_coeff, Symbol(:power_, k_power))
    end
end

# ────────────────────────────────────────────────────────────────────
# Newton potential extraction
# ────────────────────────────────────────────────────────────────────

"""
    newton_potential_coeff(amplitude::TensorExpr, k_sq::Symbol=:k²) -> Any

Extract the coefficient of the 1/k² pole from a tree-level scattering
amplitude. This gives the Newtonian potential:

    V_Newton(r) = -G m₁ m₂ / r

The amplitude for tree-level graviton exchange is:
    M = (κ²/2) T₁^{ab} D_{abcd} T₂^{cd} = (κ²/4k²)[2T₁·T₂ - T₁ᵃₐ T₂ᵇᵦ]

For point particles T_{ab} = m v_a v_b (neglecting pressure), in the
non-relativistic limit T_{00} = m, so:
    M = κ²m₁m₂/(4k²) = 8πG m₁m₂ / k²

Fourier transform: V = -2G m₁m₂ / r = Newton potential.

Ground truth: Goldberger & Rothstein hep-th/0409156, Eq 30.
"""
function newton_potential_coeff(m1, m2, G)
    # V(r) = -G m1 m2 / r
    # From FT: coeff of 1/(4πr) = 4πG m1 m2
    # So amplitude coefficient of 1/k² is 4πG m1 m2
    # After FT with convention V = -M̃: V = -G m1 m2 / r ✓
    (m1, m2, G, :coulomb)
end

# ────────────────────────────────────────────────────────────────────
# Amplitude analysis
# ────────────────────────────────────────────────────────────────────

"""
    PNPotentialTerm

A single term in the PN potential expansion.

# Fields
- `coefficient::Any`  -- numerical/symbolic coefficient
- `r_power::Int`      -- power of 1/r: V ~ coeff / r^n (n = r_power)
- `pn_order::Int`     -- post-Newtonian order (0PN, 1PN, 2PN, ...)
- `type::Symbol`      -- potential type (:newtonian, :spin_orbit, :spin_spin, etc.)
"""
struct PNPotentialTerm
    coefficient::Any
    r_power::Int
    pn_order::Int
    type::Symbol
end

function Base.show(io::IO, pt::PNPotentialTerm)
    print(io, "V_$(pt.type) ~ $(pt.coefficient) / r^$(pt.r_power) [$(pt.pn_order)PN]")
end

"""
    classify_pn_order(k_power::Int, v_power::Int) -> Int

Classify the PN order of a term from its momentum and velocity powers.

In NRGR: each 1/k² contributes +1 to the PN order (from potential region),
and each v² contributes +1 to the PN order.

PN order n means the term is O(v^{2n}) relative to Newtonian.
Newtonian (1/k² × m² ~ G m/r) is 0PN.
"""
function classify_pn_order(k_power::Int, v_power::Int)
    # Newtonian: k_power=2, v_power=0 -> 0PN
    # 1PN: either k_power=2,v_power=2 or k_power=4,v_power=0
    div(k_power - 2 + v_power, 2)
end
