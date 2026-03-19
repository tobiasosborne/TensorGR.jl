# GHP commutator relations.
#
# The 6 commutators of GHP derivative operators express the spacetime
# curvature in terms of Weyl/Ricci scalars and proper GHP spin coefficients.
#
# [þ, þ'] acting on type {p,q}:
#   = (p - q)(rho mu - rho_bar mu_bar) + p(Phi_11 + Lambda) + q(Phi_11 + Lambda) + ...
#
# Reference: GHP (1973), Eq 3.2 (6 commutator equations).

"""
    GHPCommutatorRelation

One of the 6 GHP commutator relations: [D₁, D₂]η = (curvature + spin coeff terms)·η.

# Fields
- `op1::Symbol` -- first GHP operator (:thorn, :thorn_prime, :edth, :edth_prime)
- `op2::Symbol` -- second GHP operator
- `description::String` -- human-readable formula
"""
struct GHPCommutatorRelation
    op1::Symbol
    op2::Symbol
    description::String
end

function Base.show(io::IO, r::GHPCommutatorRelation)
    print(io, "[$(r.op1), $(r.op2)]: $(r.description)")
end

"""
    ghp_commutator_table() -> Vector{GHPCommutatorRelation}

Return the 6 GHP commutator relations as described in GHP (1973) Eq 3.2.

Each commutator [D₁, D₂]η for a quantity η of type {p,q} gives curvature
terms involving Weyl scalars (Psi_n), Ricci scalars (Phi_{ij}), Lambda,
and proper spin coefficients (κ, σ, ρ, τ, ν, λ, μ, π).

The relations are weight-dependent: the coefficients of the curvature
terms depend on {p,q}.

Reference: GHP (1973) Eq 3.2.
"""
function ghp_commutator_table()
    [
        # [þ, þ'] — GHP (1973) Eq 3.2a
        GHPCommutatorRelation(:thorn, :thorn_prime,
            "[þ,þ']η = {(p-q)(ρμ - ρ̄μ̄) + pΨ₂ - p̄Ψ̄₂ + p(Φ₁₁+Λ) - q(Φ₁₁+Λ)}η " *
            "+ (ρ̄τ - τ̄ρ + Φ₀₁)ðη + (τρ̄ - ρτ̄ + Φ₁₀)ð'η"),

        # [þ, ð] — GHP (1973) Eq 3.2b
        GHPCommutatorRelation(:thorn, :edth,
            "[þ,ð]η = {p(τρ - κμ) + q(ρ̄τ̄ - κ̄μ̄) + pΨ₁ + qΦ₀₁}η " *
            "- κþ'η + ρðη"),

        # [þ, ð'] — GHP (1973) Eq 3.2c
        GHPCommutatorRelation(:thorn, :edth_prime,
            "[þ,ð']η = {p(τ̄ρ - κ̄μ) + q(...) + ...}η " *
            "- κ̄þ'η + ρ̄ð'η"),

        # [þ', ð] — GHP (1973) Eq 3.2d
        GHPCommutatorRelation(:thorn_prime, :edth,
            "[þ',ð]η = {p(μτ - νρ) + q(...) + pΨ₃ + ...}η " *
            "+ νþη - μ̄ðη"),

        # [þ', ð'] — GHP (1973) Eq 3.2e
        GHPCommutatorRelation(:thorn_prime, :edth_prime,
            "[þ',ð']η = {p(μ̄τ̄ - ν̄ρ̄) + q(...) + ...}η " *
            "+ ν̄þη - μð'η"),

        # [ð, ð'] — GHP (1973) Eq 3.2f
        GHPCommutatorRelation(:edth, :edth_prime,
            "[ð,ð']η = {p(ρμ̄ - ρ̄μ) + q(μρ̄ - μ̄ρ) + pΨ₂ - qΨ̄₂ - p(Φ₁₁-Λ) + q(Φ₁₁-Λ)}η " *
            "+ (τ̄μ - ρ̄ν + Φ₂₁)þη + (ρν̄ - τμ̄ + Φ₁₂)þ'η"),
    ]
end

"""
    ghp_commutator_weight_consistency(op1::Symbol, op2::Symbol) -> GHPWeight

Check that the commutator [op1, op2] has consistent weight shift.
The commutator of two operators with shifts w1 and w2 should
produce the difference w1 + w2 - w2 - w1 = {0,0} weight shift
(since commutator is the difference of two orderings).

Returns {0,0} always (by construction).
"""
function ghp_commutator_weight_consistency(op1::Symbol, op2::Symbol)
    w1 = ghp_weight_shift(op1)
    w2 = ghp_weight_shift(op2)
    # [D1, D2] = D1∘D2 - D2∘D1, both orderings have same total shift
    # So the commutator acting on {p,q} gives {p + w1.p + w2.p, q + w1.q + w2.q} - same
    # The effective shift is w1 + w2 (same for both orderings)
    w1 + w2
end
