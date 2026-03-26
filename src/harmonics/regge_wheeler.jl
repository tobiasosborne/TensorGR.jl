#= Regge-Wheeler and Zerilli master equations derived from Schwarzschild
#  background via harmonic decomposition + RW gauge.
#
# Odd parity (Regge-Wheeler equation):
#   [-∂²/∂t² + ∂²/∂r*² - V_RW(r)] Ψ_RW = 0
#   V_RW(r) = f(r)[l(l+1)/r² - 6M/r³]
#   Ψ_RW = (r/(l-1)(l+2)) × f(r) × [∂_r(h₁/r) - (1/r)h₀]
#
# Even parity (Zerilli equation):
#   [-∂²/∂t² + ∂²/∂r*² - V_Z(r)] Ψ_Z = 0
#   V_Z(r) = f(r)·[2n²(n+1)r³ + 6n²Mr² + 18nM²r + 18M³]/(r³(nr+3M)²)
#   where n = (l-1)(l+2)/2
#   Ψ_Z = (r/(n+1))[K + f/(nr+3M)(rH₁ - r²∂K/∂r*)]
#
# Isospectrality via Darboux/SUSY:
#   V_RW = W² + dW/dr*,  V_Z = W² - dW/dr*
#   W(r) = f(r)·[n(n+1)r² + 3nMr + 6M²]/(r²(nr+3M))
#
# References:
#   Regge & Wheeler (1957), Eq 11.
#   Zerilli (1970), Eq 11.
#   Chandrasekhar, Mathematical Theory of Black Holes (1983), Ch 4.
#   Martel & Poisson (2005), Eqs 4.7, 4.9, 4.20, 4.22.
=#

"""
    RWMasterEquation

A first-order Regge-Wheeler or Zerilli master equation in Schrödinger form:
    [-∂²/∂t² + ∂²/∂r*² - V(r)] Ψ = 0

# Fields
- `parity::Symbol` — `:odd` or `:even`
- `l::Int` — angular momentum quantum number (l ≥ 2)
- `potential` — callable V(r, M) -> Number
- `label::Symbol` — name label (`:RW` or `:Zerilli`)
"""
struct RWMasterEquation
    parity::Symbol
    l::Int
    potential::Function
    label::Symbol
end

function Base.show(io::IO, eq::RWMasterEquation)
    print(io, "$(eq.label) master equation ($(eq.parity), l=$(eq.l))")
end

"""
    derive_rw_equation(l::Int) -> RWMasterEquation

Derive the Regge-Wheeler equation (odd-parity master) for angular momentum l.

The potential is:
    V_RW(r) = (1 - 2M/r)[l(l+1)/r² - 6M/r³]

Ground truth: Regge & Wheeler (1957) Eq 11; Martel & Poisson (2005) Eq 4.9.
"""
function derive_rw_equation(l::Int)
    l >= 2 || error("Regge-Wheeler equation requires l ≥ 2")
    V_RW = schwarzschild_rw_potential(l)
    RWMasterEquation(:odd, l, V_RW, :RW)
end

"""
    derive_zerilli_equation(l::Int) -> RWMasterEquation

Derive the Zerilli equation (even-parity master) for angular momentum l.

The potential is:
    V_Z(r) = f(r)·[2n²(n+1)r³ + 6n²Mr² + 18nM²r + 18M³]/(r³(nr+3M)²)
    where n = (l-1)(l+2)/2

Ground truth: Zerilli (1970) Eq 11; Martel & Poisson (2005) Eq 4.22.
"""
function derive_zerilli_equation(l::Int)
    l >= 2 || error("Zerilli equation requires l ≥ 2")
    V_Z = schwarzschild_zerilli_potential(l)
    RWMasterEquation(:even, l, V_Z, :Zerilli)
end

"""
    evaluate_rw_potential(eq::RWMasterEquation, r, M) -> Number

Evaluate the master equation potential at radius r with mass M.
"""
evaluate_rw_potential(eq::RWMasterEquation, r, M) = eq.potential(r, M)

"""
    rw_potential_at_horizon(eq::RWMasterEquation, M) -> Float64

Potential at r = 2M (horizon): always zero since f(2M) = 0.
"""
rw_potential_at_horizon(eq::RWMasterEquation, M) =
    Float64(evaluate_rw_potential(eq, 2M, M))

"""
    rw_potential_at_infinity(eq::RWMasterEquation) -> Float64

Potential as r → ∞: approaches zero (Coulomb tail).
"""
rw_potential_at_infinity(eq::RWMasterEquation) = 0.0
