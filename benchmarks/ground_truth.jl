if !@isdefined(_GROUND_TRUTH_LOADED)

const _GROUND_TRUTH_LOADED = true

# Ground truth data from published papers.
# Sources are cited inline. Values marked "measured" will be pinned after
# the first successful TensorGR run and cross-checked against the paper.

# ── xPert (arXiv:0807.0824) ─────────────────────────────────────────────────
# Fig. 3: term counts after canonicalization of δⁿ[Riemann]
# n=10 stated explicitly (line 750): "contains 44544 terms"
# n=1..9 read from log-scale plot — will be pinned after first run.
const XPERT_RIEMANN_TERMS = Dict(
    10 => 44544,
)

# ── xTras (arXiv:1308.3493) ─────────────────────────────────────────────────
# Section 6.1: exactly 5 independent contractions of h_{ab} ∂_c ∂_d h_{ef}
const XTRAS_SPIN2_CONTRACTIONS = 5

# Gauge invariance constraints (Section 6.1, line ~520):
# C₃ = -C₁ - C₂,  C₄ = -C₂/2,  C₅ = C₂/2
const XTRAS_GAUGE_CONSTRAINTS = Dict(
    :C3 => (:C1 => -1, :C2 => -1),
    :C4 => (:C2 => -1//2,),
    :C5 => (:C2 => 1//2,),
)

# ── Conformal gravity (arXiv:1310.0819) ──────────────────────────────────────
# Eq. 12: C_{abcd}C^{abcd} = 2 R_{ab}R^{ab} - (2/3) R² + Gauss-Bonnet
const WEYL_SQ_RICCI_SQ_COEFF = 2//1
const WEYL_SQ_SCALAR_SQ_COEFF = -2//3

end # if !@isdefined
