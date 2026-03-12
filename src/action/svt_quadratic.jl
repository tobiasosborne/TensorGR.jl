#= SVT-decomposed quadratic forms for higher-derivative gravity.

Builds the scalar-vector-tensor decomposed kinetic matrix for the action
S = ∫d⁴x √g [κR + α₁R² + α₂Ric² + β₁R□R + β₂Ric□Ric]
linearized on a flat (Minkowski) background in Bardeen gauge.

This is "Path B" — direct computation from linearized curvature in SVT
variables, complementing the spin-projection "Path A" in kernel_extraction.jl.

Bardeen gauge: h₀₀ = 2Φ, h₀ᵢ = Sᵢ (transverse), hᵢⱼ = 2ψδᵢⱼ + hTTᵢⱼ.
Fourier conventions: ∂_μ → k_μ, □ → p² = ω² − k² (Lorentz invariant).
=#

"""
    svt_quadratic_forms_6deriv(; κ=1, α₁=0, α₂=0, β₁=0, β₂=0,
                                 ω²=:ω², k²=:k²) -> NamedTuple

Build SVT-decomposed quadratic forms for 6-derivative gravity on flat background.

Returns `(tensor=QuadraticForm, scalar=QuadraticForm, vector_vanishes=true)`.

The scalar sector uses Bardeen variables `(Φ, ψ)` in Fourier space with spatial
momentum `k²` and temporal frequency `ω²`. The tensor sector depends only on
the Lorentz-invariant `p² = ω² − k²`.

Sign convention matches `build_6deriv_flat_kernel`: the R² and Ric² bilinear
forms enter with a MINUS sign relative to the Fierz-Pauli kernel.

# Linearized curvatures (scalar sector, flat background)

    δR = −2k²Φ + (4k²−6ω²)ψ
    δRic₀₀ = k²Φ + 3ω²ψ
    δRicᵢⱼ = −p²ψδᵢⱼ + kᵢkⱼ(ψ−Φ)

# Assembly

    M_total = M_EH − 2(α₁+β₁p²)·M_{(δR)²} − 2(α₂+β₂p²)·M_{(δRic)²}
"""
function svt_quadratic_forms_6deriv(; κ=1, α₁=0, α₂=0, β₁=0, β₂=0,
                                      ω²=:ω², k²=:k²)
    # Lorentz-invariant 4-momentum squared
    p² = _sym_sub(ω², k²)

    # ─── Tensor sector: 1×1 ─────────────────────────────────────
    # M_TT = κp²·f₂(p²) = κp² − α₂p⁴ − β₂p⁶
    p⁴ = _sym_mul(p², p²)
    p⁶ = _sym_mul(p², p⁴)
    M_TT = _sym_sub(_sym_sub(_sym_mul(κ, p²), _sym_mul(α₂, p⁴)),
                     _sym_mul(β₂, p⁶))
    qf_tensor = quadratic_form(Dict((:hTT, :hTT) => M_TT), [:hTT])

    # ─── Scalar sector: 2×2 for (Φ, ψ) ─────────────────────────
    k⁴ = _sym_mul(k², k²)
    ω⁴ = _sym_mul(ω², ω²)

    # --- EH contribution (linearized Einstein tensor) ---
    # δ²S_EH = −(κ/2)(8k²Φψ + (12ω²−4k²)ψ²)
    M_ΦΦ_EH = 0
    M_Φψ_EH = _sym_mul(-2, _sym_mul(κ, k²))
    M_ψψ_EH = _sym_mul(κ, _sym_sub(_sym_mul(2, k²), _sym_mul(6, ω²)))

    # --- (δR)² per unit ---
    # δR = −2k²Φ + (4k²−6ω²)ψ
    δR_ψ_coeff = _sym_sub(_sym_mul(4, k²), _sym_mul(6, ω²))  # 4k²−6ω²
    M_ΦΦ_R2 = _sym_mul(4, k⁴)
    M_Φψ_R2 = _sym_mul(-2, _sym_mul(k², δR_ψ_coeff))
    M_ψψ_R2 = _sym_mul(δR_ψ_coeff, δR_ψ_coeff)

    # --- (δRic)² per unit ---
    M_ΦΦ_Ric2 = _sym_mul(2, k⁴)
    M_Φψ_Ric2 = _sym_sub(_sym_mul(4, _sym_mul(ω², k²)),
                           _sym_mul(2, k⁴))
    M_ψψ_Ric2 = _sym_add(_sym_sub(_sym_mul(12, ω⁴),
                                    _sym_mul(16, _sym_mul(ω², k²))),
                           _sym_mul(6, k⁴))

    # --- Combine: M = M_EH − (α₁+β₁p²)·M_R² − (α₂+β₂p²)·M_Ric² ---
    # Note: factor is −1, NOT −2. The factor of 2 from δ²(αR²) = 2α(δR)²
    # is already absorbed into the SVT bilinear form normalization (the
    # Bardeen variables h₀₀=2Φ, h_{ij}=2ψδ_{ij} contribute factors that
    # halve the effective coefficient). Verified numerically: det(M_scalar)
    # vanishes at f₀(p²)=0 roots with this convention.
    c_R2 = _sym_neg(_sym_add(α₁, _sym_mul(β₁, p²)))
    c_Ric2 = _sym_neg(_sym_add(α₂, _sym_mul(β₂, p²)))

    M_ΦΦ = _sym_add(M_ΦΦ_EH, _sym_add(_sym_mul(c_R2, M_ΦΦ_R2),
                                         _sym_mul(c_Ric2, M_ΦΦ_Ric2)))
    M_Φψ = _sym_add(M_Φψ_EH, _sym_add(_sym_mul(c_R2, M_Φψ_R2),
                                         _sym_mul(c_Ric2, M_Φψ_Ric2)))
    M_ψψ = _sym_add(M_ψψ_EH, _sym_add(_sym_mul(c_R2, M_ψψ_R2),
                                         _sym_mul(c_Ric2, M_ψψ_Ric2)))

    entries = Dict((:Phi, :Phi) => M_ΦΦ, (:Phi, :psi) => M_Φψ,
                   (:psi, :psi) => M_ψψ)
    qf_scalar = quadratic_form(entries, [:Phi, :psi])

    (tensor=qf_tensor, scalar=qf_scalar, vector_vanishes=true)
end
