# GHP (Geroch-Held-Penrose) spin/boost weight system.
#
# Every GHP quantity has a weight pair (p,q) under the rescaling
#   l -> lambda*l, n -> lambda^{-1}*n, m -> e^{i*theta}*m
# A quantity eta has type {p,q} if:
#   eta -> lambda^{(p+q)/2} e^{i*(p-q)*theta/2} eta
#
# Reference: Geroch, Held, Penrose, J. Math. Phys. 14, 874 (1973).

"""
    GHPWeight(p::Int, q::Int)

GHP weight (spin/boost type) of an NP quantity.
- `p`: related to spin weight s = (p-q)/2
- `q`: related to boost weight b = (p+q)/2
"""
struct GHPWeight
    p::Int
    q::Int
end

Base.:(+)(a::GHPWeight, b::GHPWeight) = GHPWeight(a.p + b.p, a.q + b.q)
Base.:(-)(a::GHPWeight, b::GHPWeight) = GHPWeight(a.p - b.p, a.q - b.q)
Base.:(-)(a::GHPWeight) = GHPWeight(-a.p, -a.q)
Base.:(==)(a::GHPWeight, b::GHPWeight) = a.p == b.p && a.q == b.q
Base.show(io::IO, w::GHPWeight) = print(io, "{$(w.p), $(w.q)}")

"""Spin weight s = (p-q)/2."""
spin_weight(w::GHPWeight) = (w.p - w.q) ÷ 2

"""Boost weight b = (p+q)/2."""
boost_weight(w::GHPWeight) = (w.p + w.q) ÷ 2

# ── Weight tables ────────────────────────────────────────────────────────

"""GHP weights of the 5 Weyl scalars Psi_0 ... Psi_4."""
const WEYL_SCALAR_WEIGHTS = Dict{Int, GHPWeight}(
    0 => GHPWeight(4, 0),
    1 => GHPWeight(2, 0),
    2 => GHPWeight(0, 0),
    3 => GHPWeight(-2, 0),
    4 => GHPWeight(-4, 0),
)

"""GHP weights of the NP Ricci scalars Phi_{ij}."""
const RICCI_SCALAR_WEIGHTS = Dict{Tuple{Int,Int}, GHPWeight}(
    (0, 0) => GHPWeight(2, 2),
    (0, 1) => GHPWeight(2, 0),
    (0, 2) => GHPWeight(2, -2),
    (1, 0) => GHPWeight(0, 2),
    (1, 1) => GHPWeight(0, 0),
    (1, 2) => GHPWeight(0, -2),
    (2, 0) => GHPWeight(-2, 2),
    (2, 1) => GHPWeight(-2, 0),
    (2, 2) => GHPWeight(-2, -2),
)

"""GHP weights of the 12 NP spin coefficients."""
const SPIN_COEFF_WEIGHTS = Dict{Symbol, GHPWeight}(
    :kappa     => GHPWeight(3, 1),
    :sigma_np  => GHPWeight(3, -1),
    :rho_np    => GHPWeight(1, 1),
    :tau_np    => GHPWeight(1, -1),
    :nu_np     => GHPWeight(-3, -1),
    :lambda_np => GHPWeight(-3, 1),
    :mu_np     => GHPWeight(-1, -1),
    :pi_np     => GHPWeight(-1, 1),
    # epsilon, gamma, alpha, beta are NOT proper GHP quantities
    # (they transform inhomogeneously under the GHP group)
    :epsilon_np => GHPWeight(1, 1),   # formal weight
    :gamma_np   => GHPWeight(-1, -1), # formal weight
    :alpha_np   => GHPWeight(-1, 1),  # formal weight
    :beta_np    => GHPWeight(1, -1),  # formal weight
)

"""
    ghp_weight(name::Symbol) -> GHPWeight

Return the GHP weight of a named NP quantity (spin coefficient or Weyl/Ricci scalar).

For Weyl scalars, use e.g., `ghp_weight(:Psi_0)`.
For Ricci scalars, use e.g., `ghp_weight(:Phi_01)`.
For spin coefficients, use the standard names (`:kappa`, `:rho_np`, etc.).
"""
function ghp_weight(name::Symbol)
    # Check spin coefficients
    haskey(SPIN_COEFF_WEIGHTS, name) && return SPIN_COEFF_WEIGHTS[name]

    # Check Weyl scalars: :Psi_0, :Psi_1, ..., :Psi_4
    s = string(name)
    if startswith(s, "Psi_") && length(s) == 5
        n = parse(Int, s[5:5])
        haskey(WEYL_SCALAR_WEIGHTS, n) && return WEYL_SCALAR_WEIGHTS[n]
    end

    # Check Ricci scalars: :Phi_00, :Phi_01, ..., :Phi_22
    if startswith(s, "Phi_") && length(s) == 6
        i = parse(Int, s[5:5])
        j = parse(Int, s[6:6])
        haskey(RICCI_SCALAR_WEIGHTS, (i, j)) && return RICCI_SCALAR_WEIGHTS[(i, j)]
    end

    # Lambda has weight {0,0}
    name == :Lambda && return GHPWeight(0, 0)

    error("Unknown GHP quantity: $name")
end

"""
    is_proper_ghp(name::Symbol) -> Bool

Return true if `name` is a proper GHP quantity (transforms homogeneously
under the GHP group). The four spin coefficients epsilon, gamma, alpha, beta
are NOT proper GHP quantities.
"""
function is_proper_ghp(name::Symbol)
    name in (:epsilon_np, :gamma_np, :alpha_np, :beta_np) && return false
    haskey(SPIN_COEFF_WEIGHTS, name) && return true
    # Weyl and Ricci scalars are all proper
    s = string(name)
    (startswith(s, "Psi_") || startswith(s, "Phi_") || name == :Lambda) && return true
    false
end
