# -- Regge-Wheeler and Zerilli master equations ----------------------------------
#
# Martel & Poisson, Phys. Rev. D 71, 104003 (2005), Secs IV.C and V.C.
#
# The odd-parity (Regge-Wheeler) and even-parity (Zerilli) master functions
# each satisfy a one-dimensional wave equation on the Schwarzschild background:
#
#   d^2 Psi / dr*^2 + [omega^2 - V(r)] Psi = S
#
# where r* = r + 2M ln(r/(2M) - 1) is the tortoise coordinate.
#
# Odd parity (MP Eqs 5.14-5.15):
#   V_RW  = f [l(l+1)/r^2 - 6M/r^3]
#   f     = 1 - 2M/r
#
# Even parity (MP Eqs 4.25-4.26):
#   V_Z   = f [2 lambda^2 (lambda+1) r^3 + 6 lambda^2 M r^2
#              + 18 lambda M^2 r + 18 M^3] / [r^3 (lambda r + 3M)^2]
#   lambda = (l-1)(l+2)/2
#
# The Cunningham-Price-Moncrief function (MP Eq 5.13):
#   Psi_odd = (2r / ((l-1)(l+2))) eps^{ab} (nabla_a htilde_b - (2/r) r_a htilde_b)
#
# The Zerilli-Moncrief function (MP Eq 4.23):
#   Psi_even = (2r / (l(l+1))) [Ktilde + (2/Lambda)(r^a r^b htilde_{ab} - r r^a nabla_a Ktilde)]
#   Lambda   = (l-1)(l+2) + 6M/r

"""
    MasterEquation

Regge-Wheeler or Zerilli master equation for a Schwarzschild perturbation mode.

The equation is:  d^2 Psi / dr*^2 + [omega^2 - V(r)] Psi = S

Fields:
- `potential::Function`  -- V(r): effective potential, callable as `potential(r)`
- `source::Function`     -- S(r): source term, callable as `source(r)` (zero for vacuum)
- `parity::Parity`       -- EVEN (Zerilli) or ODD (Regge-Wheeler)
- `l::Int`               -- angular momentum quantum number
- `mass::Symbol`         -- symbolic mass parameter
"""
struct MasterEquation
    potential::Function
    source::Function
    parity::Parity
    l::Int
    mass::Symbol
end

function Base.show(io::IO, me::MasterEquation)
    name = me.parity == ODD ? "Regge-Wheeler" : "Zerilli"
    print(io, "MasterEquation(", name, ", l=", me.l, ", M=", me.mass, ")")
end

Base.:(==)(a::MasterEquation, b::MasterEquation) =
    a.parity == b.parity && a.l == b.l && a.mass == b.mass

# ── Regge-Wheeler potential (odd parity) ─────────────────────────────────────

"""
    regge_wheeler_potential(M, l, r)

Regge-Wheeler effective potential for odd-parity perturbations of Schwarzschild.

    V_RW(r) = (1 - 2M/r) [l(l+1)/r^2 - 6M/r^3]

Ground truth: Martel & Poisson (2005), Eq 5.15.

# Arguments
- `M` -- black hole mass (numeric)
- `l::Int` -- angular momentum quantum number (l >= 2)
- `r` -- radial coordinate (numeric, r > 2M)
"""
function regge_wheeler_potential(M, l::Int, r)
    l >= 2 || throw(ArgumentError("Regge-Wheeler potential requires l >= 2, got l=$l"))
    f = 1 - 2M / r
    f * (l * (l + 1) / r^2 - 6M / r^3)
end

# ── Zerilli potential (even parity) ──────────────────────────────────────────

"""
    zerilli_potential(M, l, r)

Zerilli effective potential for even-parity perturbations of Schwarzschild.

    V_Z(r) = (1 - 2M/r) [2 lambda^2 (lambda+1) r^3 + 6 lambda^2 M r^2
              + 18 lambda M^2 r + 18 M^3] / [r^3 (lambda r + 3M)^2]

where lambda = (l-1)(l+2)/2.

Ground truth: Martel & Poisson (2005), Eq 4.26.

# Arguments
- `M` -- black hole mass (numeric)
- `l::Int` -- angular momentum quantum number (l >= 2)
- `r` -- radial coordinate (numeric, r > 2M)
"""
function zerilli_potential(M, l::Int, r)
    l >= 2 || throw(ArgumentError("Zerilli potential requires l >= 2, got l=$l"))
    lambda = (l - 1) * (l + 2) // 2
    f = 1 - 2M / r
    Lambda_r = lambda * r + 3M
    numerator = 2 * lambda^2 * (lambda + 1) * r^3 +
                6 * lambda^2 * M * r^2 +
                18 * lambda * M^2 * r +
                18 * M^3
    f * numerator / (r^3 * Lambda_r^2)
end

# ── Master equation constructor ──────────────────────────────────────────────

"""
    master_equation(sp::SchwarzschildPerturbation, parity::Parity) -> MasterEquation
    master_equation(sp::SchwarzschildPerturbation) -> MasterEquation

Construct the master equation for a Schwarzschild perturbation mode.

If `parity` is not specified, uses `sp.parity`.

For `ODD` parity (Regge-Wheeler): potential is `regge_wheeler_potential`.
For `EVEN` parity (Zerilli): potential is `zerilli_potential`.

The source term is zero (vacuum perturbation at first order).

Requires l >= 2 (radiative modes). For l < 2, throws an `ArgumentError`.

Ground truth: Martel & Poisson (2005), Eqs 4.25 (Zerilli), 5.14 (RW).
"""
function master_equation(sp::SchwarzschildPerturbation, parity::Parity)
    sp.l >= 2 || throw(ArgumentError(
        "Master equations require l >= 2 (radiative modes), got l=$(sp.l)"))
    M_sym = sp.mass
    l = sp.l
    zero_source = _ -> 0
    if parity == ODD
        V = r -> regge_wheeler_potential(1, l, r)
        MasterEquation(V, zero_source, ODD, l, M_sym)
    else
        V = r -> zerilli_potential(1, l, r)
        MasterEquation(V, zero_source, EVEN, l, M_sym)
    end
end

master_equation(sp::SchwarzschildPerturbation) = master_equation(sp, sp.parity)
