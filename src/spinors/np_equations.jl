# NP commutator relations and directional derivatives.
#
# The four directional derivatives are:
#   D = l^a nabla_a,  Delta = n^a nabla_a
#   delta = m^a nabla_a,  deltabar = mbar^a nabla_a
#
# The NP commutator equations (NP 1962, Eq 4.1):
#   [D, Delta] = (gamma + gamma_bar)D + (epsilon + epsilon_bar)Delta
#                - (tau_bar + pi)delta - (tau + pi_bar)deltabar
#   ... (3 more)
#
# Reference: Newman & Penrose, J. Math. Phys. 3, 566 (1962), Eq 4.1.

"""
    np_directional_derivative(tetrad_vec::Symbol, expr::TensorExpr;
                              covd_name::Symbol=:D) -> TensorExpr

Apply the NP directional derivative `v^a nabla_a` to `expr`.

Tetrad vector must be one of `:np_l` (D), `:np_n` (Delta),
`:np_m` (delta), `:np_mbar` (deltabar).
"""
function np_directional_derivative(tetrad_vec::Symbol, expr::TensorExpr;
                                   covd_name::Symbol=:D)
    all_idxs = indices(expr)
    used = Set{Symbol}(idx.name for idx in all_idxs)
    a = fresh_index(used; vbundle=:Tangent)

    deriv = TDeriv(TIndex(a, Down, :Tangent), expr, covd_name)
    v = Tensor(tetrad_vec, [TIndex(a, Up, :Tangent)])
    tproduct(1 // 1, TensorExpr[v, deriv])
end

"""
    NPCommutatorRelation

Stores one NP commutator relation: `[D1, D2] f` equals a linear combination
of directional derivatives of f with spin-coefficient-valued coefficients.

Fields:
- `d1`, `d2`: tetrad vector names for the two directional derivatives
- `coefficients`: Dict mapping tetrad vector name to its coefficient
  (a TensorExpr built from spin coefficients and their conjugates)
"""
struct NPCommutatorRelation
    d1::Symbol
    d2::Symbol
    coefficients::Dict{Symbol, Rational{Int}}
end

"""
    np_commutator_table() -> Vector{NamedTuple}

Return the 4 NP commutator relations as named tuples describing
which spin coefficients appear as coefficients of each directional
derivative in the commutator.

Format: `(d1, d2, D_coeff, Delta_coeff, delta_coeff, deltabar_coeff)`
where each `*_coeff` is a vector of `(spin_coeff_name, sign)` pairs.

Reference: NP (1962) Eq 4.1a-4.1d.
"""
function np_commutator_table()
    # [D, Delta] = (gamma + gamma_bar)D + (epsilon + epsilon_bar)Delta
    #              - (tau_bar + pi)delta - (tau + pi_bar)deltabar
    comm1 = (
        d1 = :np_l, d2 = :np_n,
        D_coeffs = [(:gamma_np, 1), (:gamma_np, 1)],     # gamma + gamma_bar
        Delta_coeffs = [(:epsilon_np, 1), (:epsilon_np, 1)], # epsilon + epsilon_bar
        delta_coeffs = [(:tau_np, -1), (:pi_np, -1)],     # -(tau_bar + pi)
        deltabar_coeffs = [(:tau_np, -1), (:pi_np, -1)],  # -(tau + pi_bar)
    )

    # [D, delta] = (alpha_bar + beta - pi_bar)D + kappa Delta
    #              - sigma delta - (rho + epsilon - epsilon_bar)deltabar  (... simplified)
    comm2 = (
        d1 = :np_l, d2 = :np_m,
        D_coeffs = [(:alpha_np, 1), (:beta_np, 1), (:pi_np, -1)],
        Delta_coeffs = [(:kappa, 1)],
        delta_coeffs = [(:sigma_np, -1)],
        deltabar_coeffs = [(:rho_np, -1), (:epsilon_np, -1), (:epsilon_np, 1)],
    )

    # [Delta, delta] = -nu_bar D + (tau - alpha_bar - beta)Delta
    #                  + (mu - gamma + gamma_bar)delta + lambda_bar deltabar
    comm3 = (
        d1 = :np_n, d2 = :np_m,
        D_coeffs = [(:nu_np, -1)],
        Delta_coeffs = [(:tau_np, 1), (:alpha_np, -1), (:beta_np, -1)],
        delta_coeffs = [(:mu_np, 1), (:gamma_np, -1), (:gamma_np, 1)],
        deltabar_coeffs = [(:lambda_np, 1)],
    )

    # [delta, deltabar] = (mu_bar - mu)D + (rho_bar - rho)Delta
    #                     + (alpha - beta_bar)delta - (alpha_bar - beta)deltabar  (... simplified)
    comm4 = (
        d1 = :np_m, d2 = :np_mbar,
        D_coeffs = [(:mu_np, 1), (:mu_np, -1)],
        Delta_coeffs = [(:rho_np, 1), (:rho_np, -1)],
        delta_coeffs = [(:alpha_np, 1), (:beta_np, -1)],
        deltabar_coeffs = [(:alpha_np, -1), (:beta_np, 1)],
    )

    [comm1, comm2, comm3, comm4]
end

# ── NP field equations (Ricci identities) ──────────────────────────────────
#
# The 18 NP field equations arise from projecting the Ricci identity
#   (∇_a ∇_b − ∇_b ∇_a) V_c = R_{abcd} V^d
# onto the null tetrad {l, n, m, m̄}. Each equation relates directional
# derivatives of spin coefficients to quadratic spin-coefficient terms
# and curvature scalars (Weyl Ψ_n, Ricci Φ_{ij}, Lambda=R/24).
#
# Reference: Newman & Penrose, J. Math. Phys. 3, 566 (1962), Eqs 4.2a–4.2r.

"""
    NPFieldEquation

One of the 18 Newman-Penrose field equations (Ricci identities projected
onto the null tetrad).

Each equation has the form:  `D₁(sc₁) − D₂(sc₂) = Σᵢ cᵢ · ∏ⱼ fᵢⱼ`

# Fields
- `label::String` — equation label from NP (1962), e.g. `"4.2a"`
- `deriv1::Symbol` — first directional derivative (`:D`, `:Delta`, `:delta`, `:deltabar`)
- `sc1::Symbol` — spin coefficient being differentiated (first term)
- `deriv2::Symbol` — second directional derivative (subtracted)
- `sc2::Symbol` — spin coefficient being differentiated (subtracted term)
- `rhs::Vector{Tuple{Int, Vector{Symbol}}}` — RHS terms as `(coefficient, [factors…])`

# Symbol conventions for RHS factors
- Spin coefficients: `kappa`, `sigma`, `rho`, `tau`, `nu`, `lambda`, `mu`, `pi`,
  `epsilon`, `gamma`, `alpha`, `beta`
- Complex conjugates: append `_bar` (e.g., `sigma_bar`, `rho_bar`)
- Weyl scalars: `Psi0` … `Psi4`
- Ricci scalars: `Phi00` … `Phi22`
- Scalar curvature: `Lambda` (= R/24)
"""
struct NPFieldEquation
    label::String
    deriv1::Symbol
    sc1::Symbol
    deriv2::Symbol
    sc2::Symbol
    rhs::Vector{Tuple{Int, Vector{Symbol}}}
end

const _NP_DERIV_NAMES = Dict(:D => "D", :Delta => "Δ", :delta => "δ", :deltabar => "δ̄")

function Base.show(io::IO, eq::NPFieldEquation)
    d1 = get(_NP_DERIV_NAMES, eq.deriv1, string(eq.deriv1))
    d2 = get(_NP_DERIV_NAMES, eq.deriv2, string(eq.deriv2))
    print(io, "NP Eq $(eq.label): $(d1)($(eq.sc1)) − $(d2)($(eq.sc2)) = [$(length(eq.rhs)) terms]")
end

"""
    np_field_equations() -> Vector{NPFieldEquation}

Return all 18 Newman-Penrose field equations (NP 1962, Eqs 4.2a–4.2r).

These are the Ricci identities projected onto the null tetrad, relating
directional derivatives of spin coefficients to quadratic spin-coefficient
products and curvature scalars. Each equation has the form:

    D₁(sc₁) − D₂(sc₂) = (quadratic spin coeff terms) + (curvature terms)

# Conventions
- D = l^a∇_a, Δ = n^a∇_a, δ = m^a∇_a, δ̄ = m̄^a∇_a
- Sign convention: l_a n^a = −1, m_a m̄^a = +1
- Curvature: Ψ_n (Weyl), Φ_{ij} (Ricci), Λ = R/24

See also: [`vacuum_np_field_equations`](@ref), [`np_commutator_table`](@ref)
"""
function np_field_equations()
    T(c::Int, fs::Symbol...) = (c, Symbol[fs...])

    [
        # ── D equations (4.2a–4.2i): first derivative is D = l^a∇_a ──

        # (4.2a) Dρ − δ̄κ = ρ² + σσ̄ + (ε+ε̄)ρ − κ̄τ − κ(3α+β̄−π) + Φ₀₀
        NPFieldEquation("4.2a", :D, :rho, :deltabar, :kappa, [
            T(1, :rho, :rho), T(1, :sigma, :sigma_bar),
            T(1, :epsilon, :rho), T(1, :epsilon_bar, :rho),
            T(-1, :kappa_bar, :tau),
            T(-3, :kappa, :alpha), T(-1, :kappa, :beta_bar), T(1, :kappa, :pi),
            T(1, :Phi00)
        ]),

        # (4.2b) Dσ − δκ = σ(ρ+ρ̄) + (3ε−ε̄)σ − κ(τ−π̄+ᾱ+3β) + Ψ₀
        NPFieldEquation("4.2b", :D, :sigma, :delta, :kappa, [
            T(1, :sigma, :rho), T(1, :sigma, :rho_bar),
            T(3, :epsilon, :sigma), T(-1, :epsilon_bar, :sigma),
            T(-1, :kappa, :tau), T(1, :kappa, :pi_bar),
            T(-1, :kappa, :alpha_bar), T(-3, :kappa, :beta),
            T(1, :Psi0)
        ]),

        # (4.2c) Dτ − Δκ = (τ+π̄)ρ + (τ̄+π)σ + (ε−ε̄)τ − (3γ+γ̄)κ + Ψ₁ + Φ₀₁
        NPFieldEquation("4.2c", :D, :tau, :Delta, :kappa, [
            T(1, :tau, :rho), T(1, :pi_bar, :rho),
            T(1, :tau_bar, :sigma), T(1, :pi, :sigma),
            T(1, :epsilon, :tau), T(-1, :epsilon_bar, :tau),
            T(-3, :gamma, :kappa), T(-1, :gamma_bar, :kappa),
            T(1, :Psi1), T(1, :Phi01)
        ]),

        # (4.2d) Dα − δ̄ε = (ρ+ε̄−2ε)α + βσ̄ − β̄ε − κλ − κ̄γ + (ε+ρ)π + Φ₁₀
        NPFieldEquation("4.2d", :D, :alpha, :deltabar, :epsilon, [
            T(1, :rho, :alpha), T(1, :epsilon_bar, :alpha), T(-2, :epsilon, :alpha),
            T(1, :beta, :sigma_bar),
            T(-1, :beta_bar, :epsilon),
            T(-1, :kappa, :lambda),
            T(-1, :kappa_bar, :gamma),
            T(1, :epsilon, :pi), T(1, :rho, :pi),
            T(1, :Phi10)
        ]),

        # (4.2e) Dβ − δε = (α+π)σ + (ρ̄−ε̄)β − (μ+γ)κ − (ᾱ−π̄)ε + Ψ₁
        NPFieldEquation("4.2e", :D, :beta, :delta, :epsilon, [
            T(1, :alpha, :sigma), T(1, :pi, :sigma),
            T(1, :rho_bar, :beta), T(-1, :epsilon_bar, :beta),
            T(-1, :mu, :kappa), T(-1, :gamma, :kappa),
            T(-1, :alpha_bar, :epsilon), T(1, :pi_bar, :epsilon),
            T(1, :Psi1)
        ]),

        # (4.2f) Dγ − Δε = (τ+π̄)α + (τ̄+π)β − (ε+ε̄)γ − (γ+γ̄)ε + τπ − νκ + Ψ₂ + Φ₁₁ − Λ
        NPFieldEquation("4.2f", :D, :gamma, :Delta, :epsilon, [
            T(1, :tau, :alpha), T(1, :pi_bar, :alpha),
            T(1, :tau_bar, :beta), T(1, :pi, :beta),
            T(-1, :epsilon, :gamma), T(-1, :epsilon_bar, :gamma),
            T(-1, :gamma, :epsilon), T(-1, :gamma_bar, :epsilon),
            T(1, :tau, :pi), T(-1, :nu, :kappa),
            T(1, :Psi2), T(1, :Phi11), T(-1, :Lambda)
        ]),

        # (4.2g) Dλ − δ̄π = ρλ + σ̄μ + π² + (α−β̄)π − νκ̄ − (3ε−ε̄)λ + Φ₂₀
        NPFieldEquation("4.2g", :D, :lambda, :deltabar, :pi, [
            T(1, :rho, :lambda), T(1, :sigma_bar, :mu),
            T(1, :pi, :pi),
            T(1, :alpha, :pi), T(-1, :beta_bar, :pi),
            T(-1, :nu, :kappa_bar),
            T(-3, :epsilon, :lambda), T(1, :epsilon_bar, :lambda),
            T(1, :Phi20)
        ]),

        # (4.2h) Dμ − δπ = ρ̄μ + σλ + ππ̄ − (ε+ε̄)μ − π(ᾱ−β) − νκ + Ψ₂ + 2Λ
        NPFieldEquation("4.2h", :D, :mu, :delta, :pi, [
            T(1, :rho_bar, :mu), T(1, :sigma, :lambda),
            T(1, :pi, :pi_bar),
            T(-1, :epsilon, :mu), T(-1, :epsilon_bar, :mu),
            T(-1, :pi, :alpha_bar), T(1, :pi, :beta),
            T(-1, :nu, :kappa),
            T(1, :Psi2), T(2, :Lambda)
        ]),

        # (4.2i) Dν − Δπ = (π+τ̄)μ + (π̄+τ)λ + (γ−γ̄)π − (3ε+ε̄)ν + Ψ₃ + Φ₂₁
        NPFieldEquation("4.2i", :D, :nu, :Delta, :pi, [
            T(1, :pi, :mu), T(1, :tau_bar, :mu),
            T(1, :pi_bar, :lambda), T(1, :tau, :lambda),
            T(1, :gamma, :pi), T(-1, :gamma_bar, :pi),
            T(-3, :epsilon, :nu), T(-1, :epsilon_bar, :nu),
            T(1, :Psi3), T(1, :Phi21)
        ]),

        # ── Δ and δ equations (4.2j–4.2r) ──

        # (4.2j) Δλ − δ̄ν = −(μ+μ̄)λ − (3γ−γ̄)λ + (3α+β̄+π−τ̄)ν − Ψ₄
        NPFieldEquation("4.2j", :Delta, :lambda, :deltabar, :nu, [
            T(-1, :mu, :lambda), T(-1, :mu_bar, :lambda),
            T(-3, :gamma, :lambda), T(1, :gamma_bar, :lambda),
            T(3, :alpha, :nu), T(1, :beta_bar, :nu),
            T(1, :pi, :nu), T(-1, :tau_bar, :nu),
            T(-1, :Psi4)
        ]),

        # (4.2k) δρ − δ̄σ = ρ(ᾱ+β) − σ(3α−β̄) + (ρ−ρ̄)τ + (μ−μ̄)κ − Ψ₁ + Φ₀₁
        NPFieldEquation("4.2k", :delta, :rho, :deltabar, :sigma, [
            T(1, :rho, :alpha_bar), T(1, :rho, :beta),
            T(-3, :sigma, :alpha), T(1, :sigma, :beta_bar),
            T(1, :rho, :tau), T(-1, :rho_bar, :tau),
            T(1, :mu, :kappa), T(-1, :mu_bar, :kappa),
            T(-1, :Psi1), T(1, :Phi01)
        ]),

        # (4.2l) δα − δ̄β = μρ − λσ + αᾱ + ββ̄ − 2αβ + γ(ρ−ρ̄) + ε(μ−μ̄) − Ψ₂ + Φ₁₁ + Λ
        NPFieldEquation("4.2l", :delta, :alpha, :deltabar, :beta, [
            T(1, :mu, :rho), T(-1, :lambda, :sigma),
            T(1, :alpha, :alpha_bar), T(1, :beta, :beta_bar),
            T(-2, :alpha, :beta),
            T(1, :gamma, :rho), T(-1, :gamma, :rho_bar),
            T(1, :epsilon, :mu), T(-1, :epsilon, :mu_bar),
            T(-1, :Psi2), T(1, :Phi11), T(1, :Lambda)
        ]),

        # (4.2m) δλ − δ̄μ = (ρ−ρ̄)ν + (μ−μ̄)π + μ(α+β̄) + λ(ᾱ−3β) − Ψ₃ + Φ₂₁
        NPFieldEquation("4.2m", :delta, :lambda, :deltabar, :mu, [
            T(1, :rho, :nu), T(-1, :rho_bar, :nu),
            T(1, :mu, :pi), T(-1, :mu_bar, :pi),
            T(1, :mu, :alpha), T(1, :mu, :beta_bar),
            T(1, :lambda, :alpha_bar), T(-3, :lambda, :beta),
            T(-1, :Psi3), T(1, :Phi21)
        ]),

        # (4.2n) δν − Δμ = μ² + λλ̄ + (γ+γ̄)μ − ν̄π + (τ−3β−ᾱ)ν + Φ₂₂
        NPFieldEquation("4.2n", :delta, :nu, :Delta, :mu, [
            T(1, :mu, :mu), T(1, :lambda, :lambda_bar),
            T(1, :gamma, :mu), T(1, :gamma_bar, :mu),
            T(-1, :nu_bar, :pi),
            T(1, :tau, :nu), T(-3, :beta, :nu), T(-1, :alpha_bar, :nu),
            T(1, :Phi22)
        ]),

        # (4.2o) δγ − Δβ = (τ−ᾱ−β)γ + μτ − σν − εν̄ − β(γ−γ̄−μ) + αλ̄ + Φ₁₂
        NPFieldEquation("4.2o", :delta, :gamma, :Delta, :beta, [
            T(1, :tau, :gamma), T(-1, :alpha_bar, :gamma), T(-2, :beta, :gamma),
            T(1, :mu, :tau), T(-1, :sigma, :nu), T(-1, :epsilon, :nu_bar),
            T(1, :beta, :gamma_bar), T(1, :beta, :mu),
            T(1, :alpha, :lambda_bar),
            T(1, :Phi12)
        ]),

        # (4.2p) δτ − Δσ = μσ + λ̄ρ + (τ+β−ᾱ)τ − (3γ−γ̄)σ − κν̄ + Φ₀₂
        NPFieldEquation("4.2p", :delta, :tau, :Delta, :sigma, [
            T(1, :mu, :sigma), T(1, :lambda_bar, :rho),
            T(1, :tau, :tau), T(1, :beta, :tau), T(-1, :alpha_bar, :tau),
            T(-3, :gamma, :sigma), T(1, :gamma_bar, :sigma),
            T(-1, :kappa, :nu_bar),
            T(1, :Phi02)
        ]),

        # (4.2q) Δρ − δ̄τ = −(ρμ̄+σλ) + (β̄−α−τ̄)τ + (γ+γ̄)ρ + νκ − Ψ₂ − 2Λ
        NPFieldEquation("4.2q", :Delta, :rho, :deltabar, :tau, [
            T(-1, :rho, :mu_bar), T(-1, :sigma, :lambda),
            T(1, :beta_bar, :tau), T(-1, :alpha, :tau), T(-1, :tau_bar, :tau),
            T(1, :gamma, :rho), T(1, :gamma_bar, :rho),
            T(1, :nu, :kappa),
            T(-1, :Psi2), T(-2, :Lambda)
        ]),

        # (4.2r) Δα − δ̄γ = (ρ+ε)ν − (τ+β)λ + (γ̄−μ)α + (β̄−τ̄)γ − Ψ₃
        NPFieldEquation("4.2r", :Delta, :alpha, :deltabar, :gamma, [
            T(1, :rho, :nu), T(1, :epsilon, :nu),
            T(-1, :tau, :lambda), T(-1, :beta, :lambda),
            T(1, :gamma_bar, :alpha), T(-1, :mu, :alpha),
            T(1, :beta_bar, :gamma), T(-1, :tau_bar, :gamma),
            T(-1, :Psi3)
        ]),
    ]
end

"""
Symbols representing NP Ricci curvature scalars and the scalar curvature Lambda.
In vacuum (R_{ab} = 0, R = 0), all of these vanish.
"""
const NP_RICCI_SYMBOLS = Set([:Phi00, :Phi01, :Phi02, :Phi10, :Phi11, :Phi12,
                               :Phi20, :Phi21, :Phi22, :Lambda])

"""
Symbols representing NP Weyl curvature scalars.
"""
const NP_WEYL_SYMBOLS = Set([:Psi0, :Psi1, :Psi2, :Psi3, :Psi4])

"""
    vacuum_np_field_equations() -> Vector{NPFieldEquation}

Return the 18 NP field equations with Ricci scalars and Lambda set to zero
(vacuum spacetime, R_{ab} = 0). Only Weyl scalar terms remain among the
curvature terms.

Reference: NP (1962) Eqs 4.2a–4.2r with Φ_{ij} = Λ = 0.
"""
function vacuum_np_field_equations()
    eqs = np_field_equations()
    [NPFieldEquation(eq.label, eq.deriv1, eq.sc1, eq.deriv2, eq.sc2,
        filter(t -> !any(f -> f in NP_RICCI_SYMBOLS, t[2]), eq.rhs))
     for eq in eqs]
end

"""
    np_field_equation(label::String) -> NPFieldEquation

Return a single NP field equation by label (e.g., `"4.2a"`, `"4.2r"`).
"""
function np_field_equation(label::String)
    for eq in np_field_equations()
        eq.label == label && return eq
    end
    error("Unknown NP field equation label: $label. " *
          "Valid labels: 4.2a through 4.2r")
end
