# HANDOFF: Session 21 — SVT Path B Analysis (TGR-pr04)

## Status: ANALYSIS COMPLETE, IMPLEMENTATION NOT STARTED

- **All tests still pass**: 5640 (include path) / 6042 (Pkg.test)
- **No code changes this session** — pure research/analysis
- **Beads DB broken**: Dolt migration issue, use `.beads/issues.jsonl` for issue tracking

## Critical Path

```
TGR-pr04 (SVT QuadraticForms) → TGR-tztc (Cross-check A vs B) → TGR-j6r9 (Tests) → TGR-af4a (Example)
```

**TGR-pr04 is the critical unblock.** All dependencies are closed.

## What Was Done This Session

Comprehensive derivation of the SVT scalar sector quadratic form entries for 6-derivative gravity on flat background. Key results below.

## Derivation Results (Ready to Implement)

### Fourier Conventions (matching TensorGR code)

The code's `to_fourier` replaces `∂_μ → k_μ` (formal substitution). The `build_*_momentum_kernel` functions use:
- `□ → k²` where `k² = k^μk_μ` (Lorentzian invariant, can be ±)
- `δRic_{μν} = (1/2)(k^ρk_μ h_{νρ} + k^ρk_ν h_{μρ} - k²h_{μν} - k_μk_ν h)`
- FP kernel: `L_FP = (1/2)k² h_{ab}h^{ab} - k_bk_c h^{ab}h^c_a + k_ak_b h^{ab}h - (1/2)k²h²`

### CRITICAL SIGN CONVENTION

The code's `build_6deriv_flat_kernel` combines kernels as:
```
K_total = κ·K_FP − 2(α₁+β₁k²)·K_R² − 2(α₂+β₂k²)·K_Ric²
```

The **MINUS** signs mean `δ²S = κ·L_FP − 2α₁(δR)² − 2α₂(δRic)²`, NOT plus. This is because the second variation of the EH action `δ²(κR)` already contains the full Fierz-Pauli structure, and the R²/Ric² terms enter with opposite sign in the linearized equations. The SVT matrix entries must use the SAME sign convention.

### Linearized Curvature in SVT (Bardeen Gauge, Flat Background)

Bardeen gauge: `h_{00} = 2Φ`, `h_{0i} = S_i` (transverse), `h_{ij} = 2ψδ_{ij} + hTT_{ij}`.

**Linearized Ricci scalar** (scalar sector only, verified by trace check):
```
δR = -2k²Φ + (4k²-6ω²)ψ
```
where `ω² = k₀²` (temporal), `k² = |k_spatial|²`, `p² = ω²-k²` (Lorentz invariant).

**Linearized Ricci tensor components** (scalar sector):
```
δRic_{00} = k²Φ + 3ω²ψ
δRic_{0i} = -2ωk_iψ
δRic_{ij} = -p²ψδ_{ij} + k_ik_j(ψ-Φ)
```

**Verification**: `η^{μν}δRic_{μν} = -δRic_{00} + δRic_{ii} = δR` ✓

### Bilinear Forms (Scalar Sector)

All expressed as `S = X^T M X` where `X = (Φ, ψ)^T`, so `S = M_{ΦΦ}Φ² + 2M_{Φψ}Φψ + M_{ψψ}ψ²`.

#### From EH (κR) — via linearized Einstein tensor

```
h^{μν}G^(1)_{μν} = 8k²Φψ + (12ω²-4k²)ψ²
δ²S_EH[scalar] = -(κ/2) × above
```

```
M_ΦΦ = 0
M_Φψ = -2κk²
M_ψψ = κ(2k² - 6ω²)
```

Check: `det(M_EH) = -4κ²k⁴` → zeros only at `k=0` (no scalar mode in GR) ✓

#### From (δR)² — R² and R□R contributions

```
(δR)² = 4k⁴Φ² - 4k²(4k²-6ω²)Φψ + (4k²-6ω²)²ψ²
```

Matrix entries (per unit of (δR)²):
```
M_ΦΦ = 4k⁴
M_Φψ = -2k²(4k²-6ω²)     [half the cross-term coefficient]
M_ψψ = (4k²-6ω²)²
```

#### From (δRic)² — Ric² and Ric□Ric contributions

Full contraction `δRic_{μν}δRic^{μν}` for scalar sector:
```
(δRic)² = 2k⁴Φ² + (8ω²k²-4k⁴)Φψ + (12ω⁴-16ω²k²+6k⁴)ψ²
```

Matrix entries (per unit of (δRic)²):
```
M_ΦΦ = 2k⁴
M_Φψ = (4ω²k²-2k⁴)       [half the cross-term coefficient]
M_ψψ = 12ω⁴-16ω²k²+6k⁴
```

Verified at ω=0: `(δRic)² = 2k⁴Φ² - 4k⁴Φψ + 6k⁴ψ²` (independently computed from components) ✓

#### R□R and Ric□Ric

On flat: `δ²(R□R) = 2p²(δR)²` and `δ²(Ric□Ric) = 2p²(δRic)²`.

So β₁ and β₂ contributions are just the R² and Ric² contributions multiplied by `p² = ω²-k²`.

### Total Scalar Matrix (WITH sign convention from code)

Using the code's minus-sign convention (`δ²S = κ·FP - 2α·(δcurv)²`):

```julia
M_ΦΦ = -(8α₁+4α₂)k⁴ - (8β₁+4β₂)p²k⁴

M_Φψ = -2κk² - [-4α₁k²(4k²-6ω²) + α₂(8ω²k²-4k⁴)]
                - p²[-4β₁k²(4k²-6ω²) + β₂(8ω²k²-4k⁴)]

M_ψψ = κ(2k²-6ω²) - [2α₁(4k²-6ω²)² + 2α₂(12ω⁴-16ω²k²+6k⁴)]
                     - p²[2β₁(4k²-6ω²)² + 2β₂(12ω⁴-16ω²k²+6k⁴)]
```

**WARNING**: The signs above assume the code's convention. If the cross-check fails, try flipping the minus signs on the 4-deriv and 6-deriv contributions to plus signs. The EH contribution signs are definitely correct.

### Tensor Sector

From Path A directly (no ambiguity):
```
M_TT = κk²·f₂(k²) = κk² - α₂k⁴ - β₂k⁶
```

where `k² = p²` (Lorentz invariant 4-momentum squared).

### Vector Sector

Should vanish identically (gauge invariance). Assert `M_V = 0`.

## Implementation Plan

### Step 1: New file `src/action/svt_quadratic.jl` (~100 lines)

```julia
"""
    svt_quadratic_forms_6deriv(; κ, α₁=0, α₂=0, β₁=0, β₂=0,
                                 ω²=:ω², k²=:k²) -> NamedTuple

Build SVT-decomposed QuadraticForms for 6-derivative gravity on flat background.
Returns `(tensor=QuadraticForm, scalar=QuadraticForm, vector_vanishes=true)`.

Uses Symbolics.jl variables for CAS if available, otherwise Expr trees.
"""
function svt_quadratic_forms_6deriv(; κ, α₁=0, α₂=0, β₁=0, β₂=0,
                                      ω²=:ω², k²=:k²)
    p² = _sym_sub(ω², k²)  # p² = ω² - k²

    # Tensor sector: 1×1
    # M_TT = κk² - α₂k⁴ - β₂k⁶ (from Path A, k² here is 4-momentum p²)
    M_TT = _sym_sub(_sym_sub(_sym_mul(κ, p²), _sym_mul(α₂, _sym_mul(p², p²))),
                     _sym_mul(β₂, _sym_mul(p², _sym_mul(p², p²))))
    qf_tensor = quadratic_form(Dict((:hTT,:hTT) => M_TT), [:hTT])

    # Scalar sector: 2×2 matrix M for (Φ, ψ)
    # [use formulas from handoff, with Symbolics arithmetic]
    M_ΦΦ = ...  # build from δR², δRic² entries
    M_Φψ = ...
    M_ψψ = ...
    entries = Dict((:Phi,:Phi) => M_ΦΦ, (:Phi,:psi) => M_Φψ, (:psi,:psi) => M_ψψ)
    qf_scalar = quadratic_form(entries, [:Phi, :psi])

    (tensor=qf_tensor, scalar=qf_scalar, vector_vanishes=true)
end
```

### Step 2: Add to `src/TensorGR.jl`

```julia
include("action/svt_quadratic.jl")  # after kernel_extraction.jl
export svt_quadratic_forms_6deriv
```

### Step 3: Tests in `test/test_6deriv_spectrum.jl` (~80 lines)

```julia
@testset "SVT Path B: quadratic forms" begin
    # 1. Tensor sector: propagator poles match Path A
    # Evaluate at random (κ, α₂, β₂) values
    # Check f₂(p²) = 0 ↔ M_TT(p²) = 0

    # 2. Scalar sector: determinant zeros match Path A
    # det(M_scalar) should have zeros matching f₀(p²) = 0
    # Evaluate at random (κ, α₁, α₂, β₁, β₂, ω, k) points

    # 3. Vector sector vanishes

    # 4. Special limits:
    #    - GR (α=β=0): det = -4κ²k⁴ (no scalar mode)
    #    - Stelle (β=0): correct Stelle masses

    # 5. GR scalar matrix: M_ΦΦ=0, M_Φψ=-2κk², M_ψψ=κ(2k²-6ω²)
end
```

### Step 4: Cross-check (TGR-tztc)

Once the SVT forms are verified, the cross-check compares:
- Tensor poles from `M_TT = 0` vs zeros of `f₂(p²)` from `flat_6deriv_spin_projections`
- Scalar poles from `det(M_scalar) = 0` vs zeros of `f₀(p²)`
- At 100+ random parameter points with `rtol=1e-10`

## Key Reference Files

| File | Purpose |
|------|---------|
| `src/action/kernel_extraction.jl:296-372` | `build_FP/R2/Ric2_momentum_kernel` — sign conventions |
| `src/action/kernel_extraction.jl:600+` | `build_6deriv_flat_kernel` — the MINUS sign combination |
| `src/action/quadratic_action.jl` | `QuadraticForm`, `sym_det`, `sym_inv`, `_sym_*` helpers |
| `examples/08_postquantum_gravity.jl` | Complete SVT pipeline template (4th-order case) |
| `test/test_6deriv_spectrum.jl` | Existing Path A tests (add Path B tests here) |
| `ext/TensorGRSymbolicsExt.jl` | Symbolics.jl dispatch for `_sym_*` operations |

## Debugging Tips

1. **If signs don't match**: The EH scalar matrix is definitely `[[0, -2κk²], [-2κk², κ(2k²-6ω²)]]`. If the higher-derivative corrections have wrong signs, flip the `-` to `+` on the `(8α₁+4α₂)` etc. terms.

2. **Normalization**: The overall normalization of M_TT vs O₂ doesn't matter for pole locations. Only the RATIO of polynomial coefficients matters.

3. **Lorentzian vs Euclidean**: The code uses `k²` as a Lorentz invariant (`p² = ω²-|k|²`). For the SVT matrix, entries depend on BOTH `ω²` and `k²` separately. But the POLES (propagator singularities) depend only on `p² = ω²-k²`.

4. **Symbolics.jl**: Use `@variables ω² k² κ α₁ α₂ β₁ β₂` for full symbolic computation. The `_sym_*` functions dispatch on `Symbolics.Num` when the extension is loaded.

5. **Quick numerical test**: For EH only (α=β=0), scalar det should be `-4κ²k⁴`. For Stelle (α₂=-1, κ=1, β=0), spin-2 mass at `p²=κ/α₂=-1` and scalar mass at `p²=-κ/(6α₁+2α₂)`.

## Beads Status

Beads DB stuck in Dolt migration. Commands fail with:
```
Error: database "beads" not found on Dolt server at 127.0.0.1:13359
```
Use `.beads/issues.jsonl` for issue data. To fix: `rm -rf .beads/dolt && bd init` or downgrade bd.

## Open Issues (from .beads/issues.jsonl)

### P2 — Active Pipeline
| ID | Title | Status | Blocked By |
|----|-------|--------|------------|
| TGR-pr04 | Step 2.2: SVT QuadraticForms + propagators (flat) | open | — (READY) |
| TGR-tztc | Step 2.3: Cross-check Path A vs Path B (flat) | open | TGR-pr04 |
| TGR-j6r9 | Step 5: Tests + benchmark for symbolic spectrum | open | TGR-tztc |
| TGR-af4a | Step 6: Example script + results + module integration | open | TGR-j6r9 |

### P2 — Infrastructure
| ID | Title | Status |
|----|-------|--------|
| TGR-byb | L1: BinaryBuilder for xperm.c | open |
| TGR-erv | L3: Pkg registration | open |

### P3/P4 — Future
| ID | Title | Priority |
|----|-------|----------|
| TGR-1kw | G4: Submanifolds/boundaries | P3 |
| TGR-61p | Geodesic equation ODE integration | P3 |
| TGR-dhp | TOV equation solver | P3 |
| TGR-293h | Symmetry-reduced metric ansatz | P4 |
| TGR-38d | H6: Invar database | P4 |
