# HANDOFF: 6-Deriv dS Spectrum — Session 12

## Completed This Session

- **Research only** — no code changes, all findings are in this document
- Computed Bueno-Cano (a,b,c,e) parameters numerically for ALL 10 invariants
- Identified convention errors in existing test `bc_R2`, `bc_RicSq`, `bc_R3` functions
- Verified perturbation engine works on dS: δ²R gives 123 terms (15.5s), δR gives 22 terms
- Read Bueno-Cano 1607.06463 paper (Eqs. 6, 11, 13-14, 17-19)

## Key Discovery: BC Parameter Convention

The numerical extraction method (parametric Riemann + finite differences) gives **raw** parameters.
The test file uses a **scaled** convention: `a_test = 4×a_raw, b_test = 4×b_raw, c_test = 4×c_raw, e_test = e_raw`.

### Method
```julia
# Build parametric Riemann R̃(Λ_BC, α) with projector k (rank χ < D)
# Evaluate L(R̃) at α=±ε, extract ∂L/∂α and ∂²L/∂α² via finite differences
# Fit e from: ∂L/∂α|₀ = e·χ(χ-1)  [NOTE: no factor 2, despite paper's Eq 13]
# Fit a,b,c from: ∂²L/∂α²|₀ = 4χ(χ-1)(a + bχ(χ-1) + c(χ-1))
```

### Verified Results (Λ_BC = 0.1, D=4)

**Raw extraction** (e_raw, a_raw, b_raw, c_raw):

| Invariant | e_raw | a_raw | b_raw | c_raw |
|-----------|-------|-------|-------|-------|
| EH (κ=1) | 1.000 | 0 | 0 | 0 |
| R² | 2.400 | ~0 | 0.500 | ~0 |
| Ric² | 0.600 | ~0 | ~0 | 0.500 |
| Riem² | 0.400 | 1.000 | ~0 | ~0 |
| R³ | 4.320 | ~0 | 1.800 | ~0 |
| R·Ric² | 1.080 | ~0 | 0.300 | 0.600 |
| Ric³ | 0.270 | ~0 | ~0 | 0.450 |
| R·Riem² | 0.720 | 1.200 | 0.200 | ~0 |
| Ric·Riem² | 0.180 | 0.319 | ~0 | 0.162 |
| Riem³ | 0.120 | 0.486 | ~0 | ~0 |

NOTE: I5 and I6 have small spurious b,c from underdetermined system (only χ=2,3 usable for D=4).
Rerun with analytical formulas for better precision.

**Test convention** (×4 for a,b,c; ×1 for e):

| Invariant | e | a | b | c |
|-----------|---|---|---|---|
| EH (κ=1) | κ | 0 | 0 | 0 |
| R² (per α₁) | 24α₁Λ_BC | 0 | 2α₁ | 0 |
| Ric² (per α₂) | 6α₂Λ_BC | 0 | 0 | 2α₂ |
| Riem² (per γ) | 4γΛ_BC | 4γ | 0 | 0 |

Converting e to TGR's Λ (Λ_TGR = 3Λ_BC):
- EH: e = κ
- R²: e = 24α₁(Λ_TGR/3) = 8α₁Λ_TGR (NOT 8α₁Λ_TGR/3 as in current test)
- Ric²: e = 6α₂(Λ_TGR/3) = 2α₂Λ_TGR (NOT 2α₂Λ_TGR/3 as in current test)

**Cubics normalized** (e/Λ_BC², a,b,c /Λ_BC in test convention):

| Invariant | e/Λ² | a/Λ | b/Λ | c/Λ |
|-----------|------|-----|-----|-----|
| R³ | 432 | 0 | 72 | 0 |
| R·Ric² | 108 | 0 | 16 | 8* |
| Ric³ | 27 | 0 | 0 | 18* |
| R·Riem² | 72 | ~38 | ~3 | ~0 |
| Ric·Riem² | 18 | ~13 | ~0 | ~6 |
| Riem³ | 12 | ~19 | ~0 | ~0 |

*Values marked ~ are approximate due to underdetermined fitting for I4-I6.
I1-I3 are exact (polynomial parametric formulas). I4-I6 need analytical verification.

## Bugs Found in Test File

### test/test_6deriv_spectrum.jl

1. **Line 349**: `bc_R2(α₁, Λ) = (..., e=8α₁*Λ/3)` should be `e=8α₁*Λ` (factor 3 error)
2. **Line 352**: `bc_RicSq(α₂, Λ) = (..., e=2α₂*Λ/3)` should be `e=2α₂*Λ` (factor 3 error)
3. **Line 355**: `bc_R3(γ₁, Λ) = (..., b=24γ₁*Λ/3, e=48γ₁*(Λ/3)^2)` should be `b=24γ₁*Λ, e=48γ₁*Λ^2/3`
   Wait — need to recheck this. Raw b=1.8 at Λ_BC=0.1, so b/Λ_BC=18 per unit coupling.
   Test convention b = 4×1.8 = 7.2. At Λ_BC=0.1: b_test = 7.2. So b_test/Λ_BC = 72.
   Current test: b=24γ₁*Λ_TGR/3 = 24γ₁*Λ_BC = 24*0.1 = 2.4 for γ₁=1. But we get 7.2. Off by 3.
   Correct: b = 72γ₁*Λ_BC = 24γ₁*Λ_TGR

**Root cause**: The test file consistently uses Λ_TGR/3 where it should use Λ_TGR (or equivalently Λ_BC where it should use 3Λ_BC). The e parameters for R² and Ric² are divided by an extra factor of 3.

### Impact on existing tests
- Tests at lines 396-445 only check `isfinite()` for Λ≠0, so they still pass with wrong e values
- Tests at lines 447-470 check flat limit (Λ=0) where e corrections vanish, so they pass
- The bug only affects dS-specific predictions (mass corrections at Λ≠0)

## Correct BC Parameter Functions (TGR Λ convention)

```julia
bc_EH(κ, Λ) = (a=0.0, b=0.0, c=0.0, e=κ)
bc_R2(α₁, Λ) = (a=0.0, b=2α₁, c=0.0, e=8α₁*Λ)     # was e=8α₁*Λ/3
bc_RicSq(α₂, Λ) = (a=0.0, b=0.0, c=2α₂, e=2α₂*Λ)   # was e=2α₂*Λ/3
bc_R3(γ, Λ) = (a=0.0, b=24γ*Λ, c=0.0, e=144γ*Λ^2)   # NEEDS ANALYTICAL VERIFICATION
```

## What To Do Next (TGR-mphe)

### Step 1: Fix test BC parameter functions
Fix bc_R2, bc_RicSq, bc_R3. Update the tests at lines 487-491 that hardcode the wrong values.

### Step 2: Derive I4-I6 BC parameters analytically
The I5 and I6 numerical extraction is underdetermined (only 2 usable χ values for D=4).
Options:
a) Use larger ε and higher-precision finite differences
b) Derive analytically using the block-diagonal Riemann structure:
   - kk-block: R = (Λ+α)·antisym, dimension χ
   - ⊥⊥-block: R = Λ·antisym, dimension D-χ
   - cross-block: R_{iajb} = Λ·δ_{ij}δ_{ab}
c) Use D>4 (temporarily) to get more χ values, then specialize to D=4

### Step 3: Build dS momentum-space kernels
The perturbation engine works on dS (δ²R = 123 terms in 15.5s). Two approaches:
a) **Perturbation engine**: compute δ²(each term) → Fourier not applicable on curved space
b) **Bueno-Cano**: use corrected (a,b,c,e) + mass formulas (Eqs 17-19) directly

Approach (b) is much simpler and sufficient for the spectrum. Approach (a) is a cross-check.

### Step 4: Implement full dS spectrum
```julia
# Bueno-Cano mass formulas (D=4)
κ_eff_inv(a, e, Λ_BC) = 4e - 8Λ_BC * a
m2_g(a, c, e, Λ_BC) = (-e + 2Λ_BC * a) / (2a + c)
m2_s(a, b, c, e, Λ_BC) = (2e - 4Λ_BC*(a + 4b + c)) / (2a + 4c + 12b)
```

### Step 5: Box terms (R□R, Ric□Ric) on dS
These are NOT covered by standard Bueno-Cano (which handles L(R_{μνρσ}) only, no ∇R).
On MSS, □R̄ = 0, so box terms contribute through:
- δ²(R□R) = 2(δR)(□δR) + 4Λ·□(δ²R) + ...
- Effect: replace α₁ → α₁ - β₁m² in the mass equations (implicit)
- The dS form factors become: f₂(m²) = 1 + (α₂-β₂m²)/κ·(...) etc.

## Key Files

| File | Role |
|------|------|
| `test/test_6deriv_spectrum.jl` | BC param functions to fix (lines 349,352,355,487-491) |
| `src/action/kernel_extraction.jl` | Existing flat kernel builders + spin projection |
| `examples/11_6deriv_gravity_dS.jl` | Cubic invariant builders + dS perturbation engine |
| `benchmarks/papers/1607.06463_Bueno_Cano_ECG_2016.pdf` | Reference: Eqs 11,13-14,17-19 |

## Beads Issues
- **TGR-mphe**: In progress — BC parameters computed, test bugs found, implementation pending
- **TGR-7tcs**: Blocked by mphe — cubic contributions need corrected BC params
- **TGR-ug98**: Blocked by mphe+7tcs — full dS spectrum
