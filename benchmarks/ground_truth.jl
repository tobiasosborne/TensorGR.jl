if !@isdefined(_GROUND_TRUTH_LOADED)

const _GROUND_TRUTH_LOADED = true

# Ground truth data from published papers.
# Sources are cited inline. "Measured" values are pinned from verified TensorGR runs
# and cross-checked against the paper where possible.

# ── xPert (arXiv:0807.0824) ─────────────────────────────────────────────────
# Fig. 3: term counts after canonicalization of δⁿ[Riemann]
# n=10 stated explicitly (line 750): "contains 44544 terms"
# n=1..3 measured from TensorGR (compact representation, not fully expanded)
const XPERT_RIEMANN_TERMS = Dict(
    1 => 2,
    2 => 4,
    3 => 6,
    10 => 44544,
)

# Term counts for other xPert quantities (measured)
const XPERT_CHRISTOFFEL1_TERMS = 1     # δ¹Γ is a single TProduct
const XPERT_RICCI1_RAW_TERMS = 2       # δ¹Ric before background rules
const XPERT_RICCI1_SIMPLIFIED_TERMS = 2 # δ¹Ric after flat bg simplification
const XPERT_RICCI_SCALAR1_TERMS = 2    # δ¹R raw
const XPERT_EINSTEIN2_TERMS = 5        # δ²G_{ab} raw

# ── xTras (arXiv:1308.3493) ─────────────────────────────────────────────────
# Section 6.1: exactly 5 independent contractions of h_{ab} ∂_c ∂_d h_{ef}
const XTRAS_SPIN2_CONTRACTIONS = 5

# all_contractions([h,h]) with no free indices: 2 independent scalars (measured)
const XTRAS_HH_CONTRACTIONS = 2

# VarD(h^{ab}□h_{ab}, h) term count (measured)
const XTRAS_VARD_HBOXH_TERMS = 2

# ── Conformal gravity (arXiv:1310.0819) ──────────────────────────────────────
# Weyl → Riemann expansion: 3 top-level terms (measured)
const WEYL_EXPANSION_TERMS = 3

# Weyl → Riemann decomposition: R_{abcd} = C_{abcd} + (Ricci terms) + (scalar terms)
const WEYL_DECOMPOSITION_TERMS = 3

# ── Schwarzschild (arXiv:0903.1134) ──────────────────────────────────────────
const SCHWARZ_D1RIC_SIMPLIFIED_TERMS = 18  # δ¹Ric on vacuum bg, simplified
const SCHWARZ_D2RIC_RAW_TERMS = 8          # δ²Ric on vacuum bg, raw

# ── Chern-Simons (arXiv:1012.3144) ──────────────────────────────────────────
const CS_PONTRYAGIN_FACTORS = 3   # ε·R·R = 3 tensor factors
const CS_EULER_TERMS = 3          # Riem² - 4Ric² + R²
const CS_ISAACSON_TERMS = 8       # ⟨δ²Ric⟩ bilinear in h

# ── de Sitter (arXiv:1403.3335) ─────────────────────────────────────────────
const DS_D1RIEM_RAW_TERMS = 6     # δ¹Riem on dS, raw
const DS_D1RIEM_SIMPLIFIED_TERMS = 6   # δ¹Riem on dS, simplified (contract_curvature in pipeline)

# ── Galileon (arXiv:0901.1314) ──────────────────────────────────────────────
const GALILEON_L2_FACTORS = 3     # g^{ab} ∂_a π ∂_b π
const GALILEON_L4_TERMS = 2       # (□π)² - (∇∇π)²
const GALILEON_L4_EOM_DERIV_ORDER = 4  # structural (before commutation)
const GALILEON_L4_EOM_TERMS = 17  # VarD(L₄, π) simplified (improved collect_terms)

# ── PSALTer (arXiv:2406.09500) ──────────────────────────────────────────────
const PSALTER_THETA_TERMS = 2     # θ_{μν} = η_{μν} - k_μ k_ν/k²
const PSALTER_OMEGA_TERMS = 1     # ω_{μν} = k_μ k_ν/k²
const PSALTER_P2_TERMS = 2
const PSALTER_P1_TERMS = 1
const PSALTER_P0S_TERMS = 1
const PSALTER_P0W_TERMS = 1

# ── xPand / FLRW (arXiv:1302.6174) ──────────────────────────────────────────
const XPAND_SPLIT_HAB_TERMS = 16          # split_all_spacetime(h_{ab}) terms
const XPAND_SVT_SUBSTITUTED_TERMS = 16    # after apply_svt
const XPAND_SECTOR_NAMES = Set([:scalar, :vector, :tensor])
const XPAND_E2E_SECTOR_NAMES = Set([:pure_scalar, :scalar, :vector, :tensor])
const XPAND_DRIC_SPLIT_TERMS = 320        # δ¹Ric split into 3+1

# ── EFTofPNG (arXiv:1705.06309) ─────────────────────────────────────────────
const EFTPNG_DRICCI_TERMS = Dict(1 => 2, 2 => 4, 3 => 6)

end # if !@isdefined
