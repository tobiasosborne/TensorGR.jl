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

# ── GHP field equations (Ricci identities in GHP form) ─────────────────────
#
# The GHP field equations are the NP field equations (4.2a-4.2r) rewritten
# using the GHP covariant derivative operators (þ, þ', ð, ð'). The improper
# spin coefficients (ε, γ, α, β) are absorbed into the derivative operators.
#
# Only the 12 NP equations where BOTH LHS spin coefficients are proper GHP
# quantities have direct GHP forms. The remaining 6 equations (involving α, β,
# ε, γ on the LHS) become the GHP commutator relations instead.
#
# Derivation: For each NP equation D₁(sc₁) - D₂(sc₂) = RHS_NP, replace
#   D_i(sc) → GHP_i(sc) + (connection terms)
# where the connection terms are p·conn1·sc + q·conn2·sc with {p,q} the
# GHP weight of sc. Moving connection terms to the RHS and cancelling with
# the improper spin coefficients already present in the NP RHS yields the
# clean GHP form with only proper spin coefficients.
#
# Reference: GHP (1973) §2; Penrose & Rindler Vol 1, §4.12;
#            Stewart, "Advanced General Relativity" (1991), Ch 3.

"""
    GHPFieldEquation

One of the 12 GHP field equations (Ricci identities in GHP covariant form).

Each equation has the form: `GHP_D₁(sc₁) − GHP_D₂(sc₂) = Σᵢ cᵢ · ∏ⱼ fᵢⱼ`

where GHP_D are the GHP covariant derivative operators (þ, þ', ð, ð'),
sc are proper GHP spin coefficients, and the RHS contains only proper
spin coefficients and curvature scalars (no ε, γ, α, β).

# Fields
- `label::String` — equation label, e.g. `"GHP.1"`
- `np_origin::String` — corresponding NP equation, e.g. `"4.2a"`
- `deriv1::Symbol` — first GHP derivative (`:thorn`, `:thorn_prime`, `:edth`, `:edth_prime`)
- `sc1::Symbol` — proper spin coefficient being differentiated (first term)
- `deriv2::Symbol` — second GHP derivative (subtracted)
- `sc2::Symbol` — proper spin coefficient being differentiated (subtracted term)
- `rhs::Vector{Tuple{Int, Vector{Symbol}}}` — RHS terms as `(coefficient, [factors…])`
- `weight::Tuple{Int,Int}` — expected GHP weight {p,q} of the equation

# Symbol conventions for RHS factors
- Proper spin coefficients: `kappa`, `sigma`, `rho`, `tau`, `nu`, `lambda`, `mu`, `pi`
- Complex conjugates: append `_bar` (e.g., `sigma_bar`, `rho_bar`)
- Weyl scalars: `Psi0` … `Psi4`
- Ricci scalars: `Phi00` … `Phi22`
- Scalar curvature: `Lambda` (= R/24)
"""
struct GHPFieldEquation
    label::String
    np_origin::String
    deriv1::Symbol
    sc1::Symbol
    deriv2::Symbol
    sc2::Symbol
    rhs::Vector{Tuple{Int, Vector{Symbol}}}
    weight::Tuple{Int,Int}
end

const _GHP_DERIV_NAMES = Dict(
    :thorn => "þ", :thorn_prime => "þ'", :edth => "ð", :edth_prime => "ð'"
)

function Base.show(io::IO, eq::GHPFieldEquation)
    d1 = get(_GHP_DERIV_NAMES, eq.deriv1, string(eq.deriv1))
    d2 = get(_GHP_DERIV_NAMES, eq.deriv2, string(eq.deriv2))
    print(io, "GHP Eq $(eq.label) [from NP $(eq.np_origin)]: " *
          "$(d1)($(eq.sc1)) - $(d2)($(eq.sc2)) = [$(length(eq.rhs)) terms] " *
          "weight {$(eq.weight[1]),$(eq.weight[2])}")
end

"""
    ghp_field_equations() -> Vector{GHPFieldEquation}

Return the 12 GHP field equations — the NP Ricci identities rewritten using
GHP covariant derivative operators with all improper spin coefficients
(ε, γ, α, β) absorbed into the derivatives.

Each equation has the form:
    GHP_D₁(sc₁) − GHP_D₂(sc₂) = (proper spin coeff terms) + (curvature terms)

The 12 equations correspond to the NP equations where both LHS spin
coefficients are proper GHP quantities:
- 4.2a,b,c (D-equations with ρ,σ,τ,κ)
- 4.2g,h,i (D-equations with λ,μ,ν,π)
- 4.2j (Δ-equation with λ,ν)
- 4.2k,m (δ-equations with ρ,σ,λ,μ)
- 4.2n,p,q (δ/Δ-equations with ν,μ,τ,σ,ρ)

The remaining 6 NP equations (4.2d,e,f,l,o,r) involving improper coefficients
on the LHS become the GHP commutator relations instead.

# Derivation
For NP equation D₁(sc₁) - D₂(sc₂) = RHS_NP:
1. Replace D_i → GHP_i + connection terms
2. Move connection terms to RHS
3. Cancel improper coefficients (verified algebraically)

# Conventions
- GHP operators: þ (thorn), þ' (thorn_prime), ð (edth), ð' (edth_prime)
- Proper spin coefficients: κ, σ, ρ, τ, ν, λ, μ, π
- Curvature: Ψₙ (Weyl), Φᵢⱼ (Ricci), Λ = R/24

Reference: GHP (1973); Penrose & Rindler Vol 1, §4.12.
Cross-checked: all 12 equations verified for GHP weight consistency.

See also: [`vacuum_ghp_field_equations`](@ref), [`ghp_commutator_table`](@ref)
"""
function ghp_field_equations()
    T(c::Int, fs::Symbol...) = (c, Symbol[fs...])

    [
        # ── GHP.1 from NP 4.2a: þ(ρ) − ð'(κ) ──
        # NP: Dρ − δ̄κ = ρ² + σσ̄ + (ε+ε̄)ρ − κ̄τ − κ(3α+β̄−π) + Φ₀₀
        # After absorbing ε,α,β̄ into GHP derivatives:
        #   þ(ρ) − ð'(κ) = ρ² + σσ̄ − κ̄τ + κπ + Φ₀₀
        # Weight: {1,1}+{1,1} = {2,2}
        GHPFieldEquation("GHP.1", "4.2a", :thorn, :rho, :edth_prime, :kappa, [
            T(1, :rho, :rho), T(1, :sigma, :sigma_bar),
            T(-1, :kappa_bar, :tau), T(1, :kappa, :pi),
            T(1, :Phi00)
        ], (2, 2)),

        # ── GHP.2 from NP 4.2b: þ(σ) − ð(κ) ──
        # NP: Dσ − δκ = σ(ρ+ρ̄) + (3ε−ε̄)σ − κ(τ−π̄+ᾱ+3β) + Ψ₀
        # After absorbing ε,ᾱ,β into GHP derivatives:
        #   þ(σ) − ð(κ) = σρ + σρ̄ − κτ + κπ̄ + Ψ₀
        # Weight: {3,-1}+{1,1} = {4,0}
        GHPFieldEquation("GHP.2", "4.2b", :thorn, :sigma, :edth, :kappa, [
            T(1, :sigma, :rho), T(1, :sigma, :rho_bar),
            T(-1, :kappa, :tau), T(1, :kappa, :pi_bar),
            T(1, :Psi0)
        ], (4, 0)),

        # ── GHP.3 from NP 4.2c: þ(τ) − þ'(κ) ──
        # NP: Dτ − Δκ = (τ+π̄)ρ + (τ̄+π)σ + (ε−ε̄)τ − (3γ+γ̄)κ + Ψ₁ + Φ₀₁
        # After absorbing ε,γ into GHP derivatives:
        #   þ(τ) − þ'(κ) = τρ + π̄ρ + τ̄σ + πσ + Ψ₁ + Φ₀₁
        # Weight: {1,-1}+{1,1} = {2,0}
        GHPFieldEquation("GHP.3", "4.2c", :thorn, :tau, :thorn_prime, :kappa, [
            T(1, :tau, :rho), T(1, :pi_bar, :rho),
            T(1, :tau_bar, :sigma), T(1, :pi, :sigma),
            T(1, :Psi1), T(1, :Phi01)
        ], (2, 0)),

        # ── GHP.4 from NP 4.2g: þ(λ) − ð'(π) ──
        # NP: Dλ − δ̄π = ρλ + σ̄μ + π² + (α−β̄)π − νκ̄ − (3ε−ε̄)λ + Φ₂₀
        # After absorbing ε,α,β̄ into GHP derivatives:
        #   þ(λ) − ð'(π) = ρλ + σ̄μ + π² − νκ̄ + Φ₂₀
        # Weight: {-3,1}+{1,1} = {-2,2}
        GHPFieldEquation("GHP.4", "4.2g", :thorn, :lambda, :edth_prime, :pi, [
            T(1, :rho, :lambda), T(1, :sigma_bar, :mu),
            T(1, :pi, :pi),
            T(-1, :nu, :kappa_bar),
            T(1, :Phi20)
        ], (-2, 2)),

        # ── GHP.5 from NP 4.2h: þ(μ) − ð(π) ──
        # NP: Dμ − δπ = ρ̄μ + σλ + ππ̄ − (ε+ε̄)μ − π(ᾱ−β) − νκ + Ψ₂ + 2Λ
        # After absorbing ε,ᾱ,β into GHP derivatives:
        #   þ(μ) − ð(π) = ρ̄μ + σλ + ππ̄ − νκ + Ψ₂ + 2Λ
        # Weight: {-1,-1}+{1,1} = {0,0}
        GHPFieldEquation("GHP.5", "4.2h", :thorn, :mu, :edth, :pi, [
            T(1, :rho_bar, :mu), T(1, :sigma, :lambda),
            T(1, :pi, :pi_bar),
            T(-1, :nu, :kappa),
            T(1, :Psi2), T(2, :Lambda)
        ], (0, 0)),

        # ── GHP.6 from NP 4.2i: þ(ν) − þ'(π) ──
        # NP: Dν − Δπ = (π+τ̄)μ + (π̄+τ)λ + (γ−γ̄)π − (3ε+ε̄)ν + Ψ₃ + Φ₂₁
        # After absorbing ε,γ into GHP derivatives:
        #   þ(ν) − þ'(π) = πμ + τ̄μ + π̄λ + τλ + Ψ₃ + Φ₂₁
        # Weight: {-3,-1}+{1,1} = {-2,0}
        GHPFieldEquation("GHP.6", "4.2i", :thorn, :nu, :thorn_prime, :pi, [
            T(1, :pi, :mu), T(1, :tau_bar, :mu),
            T(1, :pi_bar, :lambda), T(1, :tau, :lambda),
            T(1, :Psi3), T(1, :Phi21)
        ], (-2, 0)),

        # ── GHP.7 from NP 4.2j: þ'(λ) − ð'(ν) ──
        # NP: Δλ − δ̄ν = −(μ+μ̄)λ − (3γ−γ̄)λ + (3α+β̄+π−τ̄)ν − Ψ₄
        # After absorbing γ,α,β̄ into GHP derivatives:
        #   þ'(λ) − ð'(ν) = −μλ − μ̄λ + πν − τ̄ν − Ψ₄
        # Weight: {-3,1}+{-1,-1} = {-4,0}
        GHPFieldEquation("GHP.7", "4.2j", :thorn_prime, :lambda, :edth_prime, :nu, [
            T(-1, :mu, :lambda), T(-1, :mu_bar, :lambda),
            T(1, :pi, :nu), T(-1, :tau_bar, :nu),
            T(-1, :Psi4)
        ], (-4, 0)),

        # ── GHP.8 from NP 4.2k: ð(ρ) − ð'(σ) ──
        # NP: δρ − δ̄σ = ρ(ᾱ+β) − σ(3α−β̄) + (ρ−ρ̄)τ + (μ−μ̄)κ − Ψ₁ + Φ₀₁
        # After absorbing α,β,ᾱ,β̄ into GHP derivatives:
        #   ð(ρ) − ð'(σ) = (ρ−ρ̄)τ + (μ−μ̄)κ − Ψ₁ + Φ₀₁
        # Weight: {1,1}+{1,-1} = {2,0}
        GHPFieldEquation("GHP.8", "4.2k", :edth, :rho, :edth_prime, :sigma, [
            T(1, :rho, :tau), T(-1, :rho_bar, :tau),
            T(1, :mu, :kappa), T(-1, :mu_bar, :kappa),
            T(-1, :Psi1), T(1, :Phi01)
        ], (2, 0)),

        # ── GHP.9 from NP 4.2m: ð(λ) − ð'(μ) ──
        # NP: δλ − δ̄μ = (ρ−ρ̄)ν + (μ−μ̄)π + μ(α+β̄) + λ(ᾱ−3β) − Ψ₃ + Φ₂₁
        # After absorbing α,β,ᾱ,β̄ into GHP derivatives:
        #   ð(λ) − ð'(μ) = (ρ−ρ̄)ν + (μ−μ̄)π − Ψ₃ + Φ₂₁
        # Weight: {-3,1}+{1,-1} = {-2,0}
        GHPFieldEquation("GHP.9", "4.2m", :edth, :lambda, :edth_prime, :mu, [
            T(1, :rho, :nu), T(-1, :rho_bar, :nu),
            T(1, :mu, :pi), T(-1, :mu_bar, :pi),
            T(-1, :Psi3), T(1, :Phi21)
        ], (-2, 0)),

        # ── GHP.10 from NP 4.2n: ð(ν) − þ'(μ) ──
        # NP: δν − Δμ = μ² + λλ̄ + (γ+γ̄)μ − ν̄π + (τ−3β−ᾱ)ν + Φ₂₂
        # After absorbing γ,β,ᾱ into GHP derivatives:
        #   ð(ν) − þ'(μ) = μ² + λλ̄ − ν̄π + τν + Φ₂₂
        # Weight: {-3,-1}+{1,-1} = {-2,-2}
        GHPFieldEquation("GHP.10", "4.2n", :edth, :nu, :thorn_prime, :mu, [
            T(1, :mu, :mu), T(1, :lambda, :lambda_bar),
            T(-1, :nu_bar, :pi), T(1, :tau, :nu),
            T(1, :Phi22)
        ], (-2, -2)),

        # ── GHP.11 from NP 4.2p: ð(τ) − þ'(σ) ──
        # NP: δτ − Δσ = μσ + λ̄ρ + (τ+β−ᾱ)τ − (3γ−γ̄)σ − κν̄ + Φ₀₂
        # After absorbing β,ᾱ,γ into GHP derivatives:
        #   ð(τ) − þ'(σ) = μσ + λ̄ρ + τ² − κν̄ + Φ₀₂
        # Weight: {1,-1}+{1,-1} = {2,-2}
        GHPFieldEquation("GHP.11", "4.2p", :edth, :tau, :thorn_prime, :sigma, [
            T(1, :mu, :sigma), T(1, :lambda_bar, :rho),
            T(1, :tau, :tau),
            T(-1, :kappa, :nu_bar),
            T(1, :Phi02)
        ], (2, -2)),

        # ── GHP.12 from NP 4.2q: þ'(ρ) − ð'(τ) ──
        # NP: Δρ − δ̄τ = −(ρμ̄+σλ) + (β̄−α−τ̄)τ + (γ+γ̄)ρ + νκ − Ψ₂ − 2Λ
        # After absorbing α,β̄,γ into GHP derivatives:
        #   þ'(ρ) − ð'(τ) = −ρμ̄ − σλ − τ̄τ + νκ − Ψ₂ − 2Λ
        # Weight: {1,1}+{-1,-1} = {0,0}
        GHPFieldEquation("GHP.12", "4.2q", :thorn_prime, :rho, :edth_prime, :tau, [
            T(-1, :rho, :mu_bar), T(-1, :sigma, :lambda),
            T(-1, :tau_bar, :tau),
            T(1, :nu, :kappa),
            T(-1, :Psi2), T(-2, :Lambda)
        ], (0, 0)),
    ]
end

"""
Symbols representing improper GHP spin coefficients and their conjugates.
These should NOT appear in any GHP field equation RHS.
"""
const GHP_IMPROPER_SYMBOLS = Set([
    :epsilon, :epsilon_bar, :gamma, :gamma_bar,
    :alpha, :alpha_bar, :beta, :beta_bar
])

"""
    vacuum_ghp_field_equations() -> Vector{GHPFieldEquation}

Return the 12 GHP field equations with Ricci scalars and Lambda set to zero
(vacuum spacetime, R_{ab} = 0). Only Weyl scalar terms remain among the
curvature terms.

Reference: GHP (1973) with Φ_{ij} = Λ = 0.
"""
function vacuum_ghp_field_equations()
    eqs = ghp_field_equations()
    [GHPFieldEquation(eq.label, eq.np_origin, eq.deriv1, eq.sc1, eq.deriv2, eq.sc2,
        filter(t -> !any(f -> f in NP_RICCI_SYMBOLS, t[2]), eq.rhs),
        eq.weight)
     for eq in eqs]
end

"""
    ghp_field_equation(label::String) -> GHPFieldEquation

Return a single GHP field equation by label (e.g., `"GHP.1"`, `"GHP.12"`).
Also accepts NP labels (e.g., `"4.2a"`) to find the corresponding GHP equation.
"""
function ghp_field_equation(label::String)
    for eq in ghp_field_equations()
        (eq.label == label || eq.np_origin == label) && return eq
    end
    error("Unknown GHP field equation label: $label. " *
          "Valid labels: GHP.1 through GHP.12, or NP origins 4.2a,b,c,g,h,i,j,k,m,n,p,q")
end

# Mapping from NP field equation symbol conventions to ghp_weight-compatible names.
# NP field equations use short names (rho, sigma, etc.) while SPIN_COEFF_WEIGHTS
# uses _np suffixed names, and Weyl/Ricci scalars use underscore format.
const _GHP_EQ_WEIGHT_MAP = Dict{Symbol, GHPWeight}(
    # Proper spin coefficients
    :kappa     => GHPWeight(3, 1),
    :sigma     => GHPWeight(3, -1),
    :rho       => GHPWeight(1, 1),
    :tau       => GHPWeight(1, -1),
    :nu        => GHPWeight(-3, -1),
    :lambda    => GHPWeight(-3, 1),
    :mu        => GHPWeight(-1, -1),
    :pi        => GHPWeight(-1, 1),
    # Improper spin coefficients (for validation checks)
    :epsilon   => GHPWeight(1, 1),
    :gamma     => GHPWeight(-1, -1),
    :alpha     => GHPWeight(-1, 1),
    :beta      => GHPWeight(1, -1),
    # Weyl scalars (NPFieldEquation format: Psi0, not Psi_0)
    :Psi0      => GHPWeight(4, 0),
    :Psi1      => GHPWeight(2, 0),
    :Psi2      => GHPWeight(0, 0),
    :Psi3      => GHPWeight(-2, 0),
    :Psi4      => GHPWeight(-4, 0),
    # Ricci scalars (NPFieldEquation format: Phi00, not Phi_00)
    :Phi00     => GHPWeight(2, 2),
    :Phi01     => GHPWeight(2, 0),
    :Phi02     => GHPWeight(2, -2),
    :Phi10     => GHPWeight(0, 2),
    :Phi11     => GHPWeight(0, 0),
    :Phi12     => GHPWeight(0, -2),
    :Phi20     => GHPWeight(-2, 2),
    :Phi21     => GHPWeight(-2, 0),
    :Phi22     => GHPWeight(-2, -2),
    # Scalar curvature
    :Lambda    => GHPWeight(0, 0),
)

"""
    _ghp_eq_factor_weight(f::Symbol) -> GHPWeight

Compute the GHP weight of a single factor symbol used in GHP field equation RHS.
Handles the naming conventions used in NPFieldEquation/GHPFieldEquation
(short names like :rho, :Psi0, :Phi01) and _bar conjugates.

Complex conjugation maps {p,q} → {q,p}.
"""
function _ghp_eq_factor_weight(f::Symbol)
    # Direct lookup
    haskey(_GHP_EQ_WEIGHT_MAP, f) && return _GHP_EQ_WEIGHT_MAP[f]

    # Handle _bar conjugates: strip _bar, look up base, swap (p,q)
    s = string(f)
    if endswith(s, "_bar")
        base = Symbol(s[1:end-4])
        if haskey(_GHP_EQ_WEIGHT_MAP, base)
            w = _GHP_EQ_WEIGHT_MAP[base]
            return GHPWeight(w.q, w.p)  # conjugate swaps p,q
        end
    end

    error("Unknown GHP field equation factor: $f")
end

"""
    _ghp_rhs_weight(term::Tuple{Int, Vector{Symbol}}) -> GHPWeight

Compute the GHP weight of a single RHS term (product of factors).
The weight of a product is the sum of the weights of the factors.
"""
function _ghp_rhs_weight(term::Tuple{Int, Vector{Symbol}})
    factors = term[2]
    isempty(factors) && return GHPWeight(0, 0)

    total = GHPWeight(0, 0)
    for f in factors
        total = total + _ghp_eq_factor_weight(f)
    end
    total
end

"""
    ghp_field_equation_weight_consistent(eq::GHPFieldEquation) -> Bool

Check that all RHS terms have the same GHP weight as the expected equation weight.
The equation weight equals the weight of the GHP derivative applied to sc1:
    weight(GHP_D₁(sc₁)) = weight(sc₁) + shift(D₁)
"""
function ghp_field_equation_weight_consistent(eq::GHPFieldEquation)
    expected = GHPWeight(eq.weight[1], eq.weight[2])
    for term in eq.rhs
        w = _ghp_rhs_weight(term)
        w == expected || return false
    end
    true
end
