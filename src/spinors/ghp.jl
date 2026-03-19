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

# ── GHP derivative operators ─────────────────────────────────────────────
#
# The 4 GHP covariant derivatives are gauge-covariant versions of the
# NP directional derivatives:
#   thorn  (þ)  = D     - p*epsilon - q*epsilon_bar    -> shifts weight by {1, 1}
#   thorn' (þ') = Delta - p*gamma   - q*gamma_bar      -> shifts weight by {-1,-1}
#   edth   (ð)  = delta - p*beta    - q*alpha_bar       -> shifts weight by {1, -1}
#   edth'  (ð') = deltabar - p*alpha - q*beta_bar       -> shifts weight by {-1, 1}
#
# Reference: GHP (1973), Eqs 2.9-2.12.

"""
    GHPDerivative

Represents one of the 4 GHP covariant derivative operators.

Fields:
- `name`: `:thorn`, `:thorn_prime`, `:edth`, `:edth_prime`
- `tetrad_vec`: the NP directional derivative direction
- `weight_shift`: the GHPWeight shift when acting on a {p,q} quantity
- `conn1`, `conn2`: the two spin coefficients subtracted (connection terms)
"""
struct GHPDerivative
    name::Symbol
    tetrad_vec::Symbol
    weight_shift::GHPWeight
    conn1::Symbol   # multiplied by p
    conn2::Symbol   # multiplied by q (conjugate of conn1)
end

"""Table of the 4 GHP derivative operators."""
const GHP_DERIVATIVES = Dict{Symbol, GHPDerivative}(
    :thorn       => GHPDerivative(:thorn,       :np_l,    GHPWeight(1, 1),   :epsilon_np, :epsilon_np),
    :thorn_prime => GHPDerivative(:thorn_prime,  :np_n,    GHPWeight(-1, -1), :gamma_np,   :gamma_np),
    :edth        => GHPDerivative(:edth,         :np_m,    GHPWeight(1, -1),  :beta_np,    :alpha_np),
    :edth_prime  => GHPDerivative(:edth_prime,   :np_mbar, GHPWeight(-1, 1),  :alpha_np,   :beta_np),
)

"""
    ghp_derivative(op::Symbol, expr::TensorExpr, weight::GHPWeight;
                   covd_name::Symbol=:D,
                   registry::TensorRegistry=current_registry()) -> TensorExpr

Apply GHP derivative operator `op` to `expr` with known GHP weight.

The GHP derivative is:
  þ(eta) = D(eta) - p*epsilon*eta - q*epsilon_bar*eta

where eta has weight {p,q}.

Returns a TensorExpr representing the result (which has shifted weight).

# Arguments
- `op`: one of `:thorn`, `:thorn_prime`, `:edth`, `:edth_prime`
- `expr`: the expression to differentiate
- `weight`: the GHP weight {p,q} of `expr`
- `covd_name`: name of the registered covariant derivative
"""
function ghp_derivative(op::Symbol, expr::TensorExpr, weight::GHPWeight;
                        covd_name::Symbol=:D,
                        registry::TensorRegistry=current_registry())
    haskey(GHP_DERIVATIVES, op) || error("Unknown GHP operator: $op. " *
        "Valid: thorn, thorn_prime, edth, edth_prime")

    ghpd = GHP_DERIVATIVES[op]
    p, q = weight.p, weight.q

    # Term 1: directional derivative v^a nabla_a(expr)
    dir_deriv = np_directional_derivative(ghpd.tetrad_vec, expr; covd_name=covd_name)

    # Term 2: -p * conn1 * expr
    sc1 = spin_coefficient(ghpd.conn1; covd_name=covd_name, registry=registry)

    # Term 3: -q * conn2_bar * expr  (conn2 is already the "bar" coefficient)
    sc2 = spin_coefficient(ghpd.conn2; covd_name=covd_name, registry=registry)

    terms = TensorExpr[dir_deriv]

    if p != 0
        push!(terms, tproduct(Rational{Int}(-p), TensorExpr[sc1, expr]))
    end

    if q != 0
        push!(terms, tproduct(Rational{Int}(-q), TensorExpr[sc2, expr]))
    end

    length(terms) == 1 ? terms[1] : tsum(terms)
end

"""
    ghp_weight_shift(op::Symbol) -> GHPWeight

Return the weight shift produced by GHP operator `op`.
- thorn:       {+1, +1}
- thorn_prime: {-1, -1}
- edth:        {+1, -1}
- edth_prime:  {-1, +1}
"""
function ghp_weight_shift(op::Symbol)
    haskey(GHP_DERIVATIVES, op) || error("Unknown GHP operator: $op")
    GHP_DERIVATIVES[op].weight_shift
end
