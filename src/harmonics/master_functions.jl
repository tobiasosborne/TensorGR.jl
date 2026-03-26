#= Master function extraction from metric perturbation.
#
# Given the harmonic decomposition of a metric perturbation h_{ab} on
# Schwarzschild in RW gauge, extract the Regge-Wheeler and Zerilli
# master functions.
#
# Odd-parity (Regge-Wheeler master function):
#   Ψ_RW(t,r) = r/[(l-1)(l+2)] × f(r) × [∂_r(h₁/r) - (1/r)h₀]
#   where h₀, h₁ are odd-parity RW gauge coefficients.
#
# Even-parity (Zerilli master function):
#   Ψ_Z(t,r) = r/(n+1) × [K + f/(nr+3M)(rH₁ - r²∂K/∂r*)]
#   where n = (l-1)(l+2)/2, and H₁, K are even-parity RW gauge coefficients.
#
# References:
#   Martel & Poisson (2005), Eqs 4.7, 4.20.
#   Nagar & Rezzolla, Class. Quantum Grav. 22, R167 (2005), Eqs 2.7, 2.11.
=#

"""
    MasterFunctionSpec

Specification for a Regge-Wheeler or Zerilli master function.

# Fields
- `parity::Symbol` — `:odd` or `:even`
- `l::Int` — angular momentum
- `n::Int` — shorthand parameter: (l-1)(l+2)/2
- `input_fields::Vector{Symbol}` — names of the RW gauge DOFs needed
- `label::Symbol` — `:Psi_RW` or `:Psi_Z`
"""
struct MasterFunctionSpec
    parity::Symbol
    l::Int
    n::Int
    input_fields::Vector{Symbol}
    label::Symbol
end

function Base.show(io::IO, mf::MasterFunctionSpec)
    print(io, "$(mf.label) ($(mf.parity), l=$(mf.l), n=$(mf.n))")
end

"""
    rw_master_function(l::Int) -> MasterFunctionSpec

Specification for the Regge-Wheeler master function (odd parity).

    Ψ_RW = r/[(l-1)(l+2)] × f(r) × [∂_r(h₁/r) - (1/r)h₀]

Ground truth: Martel & Poisson (2005) Eq 4.7.
"""
function rw_master_function(l::Int)
    l >= 2 || error("RW master function requires l ≥ 2")
    n = (l - 1) * (l + 2) ÷ 2
    MasterFunctionSpec(:odd, l, n, [:h_0_odd, :h_1_odd], :Psi_RW)
end

"""
    zerilli_master_function(l::Int) -> MasterFunctionSpec

Specification for the Zerilli master function (even parity).

    Ψ_Z = r/(n+1) × [K + f/(nr+3M)(rH₁ - r²∂K/∂r*)]

where n = (l-1)(l+2)/2.

Ground truth: Martel & Poisson (2005) Eq 4.20.
"""
function zerilli_master_function(l::Int)
    l >= 2 || error("Zerilli master function requires l ≥ 2")
    n = (l - 1) * (l + 2) ÷ 2
    MasterFunctionSpec(:even, l, n, [:H_0, :H_1, :H_2, :K], :Psi_Z)
end

"""
    extract_master_functions(l::Int) -> Tuple{MasterFunctionSpec, MasterFunctionSpec}

Return specifications for both master functions (Ψ_RW, Ψ_Z) at angular momentum l.
"""
function extract_master_functions(l::Int)
    (rw_master_function(l), zerilli_master_function(l))
end
