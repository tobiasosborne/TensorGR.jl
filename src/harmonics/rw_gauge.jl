#= Regge-Wheeler gauge for BH perturbation theory.
#
# The Regge-Wheeler gauge eliminates gauge degrees of freedom in the
# harmonic decomposition of metric perturbations on Schwarzschild.
#
# Odd parity (3 gauge freedoms removed):
#   h_{tr}^{odd} = 0,  h_{rr}^{odd} = 0,  h_t^{odd} = 0
#   Leaves: h_0(t,r), h_1(t,r)  (2 functions)
#
# Even parity (4 gauge freedoms removed):
#   h_t^{even,a} = 0,  G(t,r) = 0
#   Leaves: H_0(t,r), H_1(t,r), H_2(t,r), K(t,r)  (4 functions)
#
# References:
#   Regge & Wheeler, Phys. Rev. 108, 1063 (1957), Eq 6.
#   Martel & Poisson, Phys. Rev. D 71, 104003 (2005), Sec III.C.
=#

"""
    RWGaugeChoice

Describes which perturbation components are set to zero in the Regge-Wheeler gauge.

# Fields
- `parity::Symbol` — `:odd` or `:even`
- `vanishing::Vector{Symbol}` — tensor component symbols set to zero
- `remaining::Vector{Symbol}` — surviving DOF symbols
"""
struct RWGaugeChoice
    parity::Symbol
    vanishing::Vector{Symbol}
    remaining::Vector{Symbol}
end

"""
    rw_gauge_odd(; prefix=:h) -> RWGaugeChoice

Odd-parity Regge-Wheeler gauge: sets h₂(even-type mixing)=0,
leaving h₀(t,r) and h₁(t,r) as the two odd-parity DOFs.

Convention (Martel & Poisson):
- h^{odd}_{AB} ∝ ε_{ab} Y_{lm;b}  → coefficients h_0, h_1
- Gauge removes: h_2 (the piece ∝ Y_{lm})
"""
function rw_gauge_odd(; prefix::Symbol=:h)
    vanishing = [Symbol(prefix, :_2_odd)]  # h_2^odd = 0
    remaining = [Symbol(prefix, :_0_odd), Symbol(prefix, :_1_odd)]
    RWGaugeChoice(:odd, vanishing, remaining)
end

"""
    rw_gauge_even(; prefix=:h) -> RWGaugeChoice

Even-parity Regge-Wheeler gauge: sets h_0^even = h_1^even = G = 0,
leaving H_0, H_1, H_2, K as the four even-parity DOFs.

Convention (Martel & Poisson Eq 3.4):
- Even sector: h_{AB}^{even} has H_0, H_1, H_2; h_A^{even} has h_0, h_1; scalar piece G, K
- Gauge removes: h_0^even, h_1^even, G
"""
function rw_gauge_even(; prefix::Symbol=:h)
    vanishing = [Symbol(prefix, :_0_even), Symbol(prefix, :_1_even),
                 Symbol(prefix, :_G)]
    remaining = [Symbol(:H_0), Symbol(:H_1), Symbol(:H_2), Symbol(:K)]
    RWGaugeChoice(:even, vanishing, remaining)
end

"""
    apply_rw_gauge!(reg, gauge::RWGaugeChoice)

Register vanishing rules for the RW gauge components.
"""
function apply_rw_gauge!(reg::TensorRegistry, gauge::RWGaugeChoice)
    for v in gauge.vanishing
        if has_tensor(reg, v)
            set_vanishing!(reg, v)
        end
    end
end

"""
    rw_dof_count(gauge::RWGaugeChoice) -> Int

Number of remaining physical DOFs after gauge fixing.
"""
rw_dof_count(gauge::RWGaugeChoice) = length(gauge.remaining)

"""
    rw_gauge_full() -> Tuple{RWGaugeChoice, RWGaugeChoice}

Both odd and even parity RW gauges.
"""
rw_gauge_full() = (rw_gauge_odd(), rw_gauge_even())
