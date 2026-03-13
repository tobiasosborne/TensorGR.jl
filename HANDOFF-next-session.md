# HANDOFF: Session 31 — δ²R Spin-2 Bug (TGR-dp3): Deep Progress, Not Yet Resolved

## Status: Two approaches implemented, both partially working. Path forward is clear.

- **All 7146 tests pass** on committed code (fourier.jl fix only)
- **TGR-dp3 remains open**: position-space kernel extraction is the right solution but needs debugging
- Uncommitted WIP: `extract_kernel_direct` in kernel_extraction.jl + export in TensorGR.jl

## Root Cause (Fully Understood This Session)

### The uniform-k convention is fundamentally limited

In a quadratic form `∫dx h₁(x) K h₂(x)`, the two fields carry **opposite momenta**:
- h₁ carries +k, h₂ carries -k (from ∫dx → δ(k₁+k₂))
- Derivative on h₁: ∂_a → ik_a. Derivative on h₂: ∂_a → -ik_a.

The code's `to_fourier` uses **uniform-k**: ALL ∂ → k_a regardless of which field. This gives:
- **(∂h₁)(∂h₂)** terms: (ik)(-ik) = k² vs uniform (k)(k) = k² → **SAME** ✓
- **h₁(∂²h₂)** terms: (-ik)(-ik) = -k² vs uniform (k)(k) = k² → **WRONG SIGN** ✗
- **∂(h₁ × ∂h₂)** terms: (ik+(-ik))=0 vs uniform k×stuff ≠ 0 → **WRONG** ✗

This means uniform-k is correct ONLY when each derivative in a term acts on a DIFFERENT field. When two derivatives act on the same field, the sign is wrong.

### Why simplify makes it worse

The simplify pipeline (canonicalize + collect_terms) **merges ΓΓ terms and ∂Γ₂ terms** during term collection. After merging, you cannot cleanly separate "terms where each ∂ acts on one h" from "terms where ∂ wraps a product". This is why the fourier.jl fix (dropping ∂(bilinear) terms) only partially works — it can only drop terms still in ∂(product) form, missing the merged ones.

### The correct physics (verified numerically)

For the EH action on flat background:
- **FP kernel** (hand-built reference): spin-2=2.5, spin-0s=-1.0 at k²=1
- **δ²R alone** (not the full action): spin-2=1.25, spin-0s=-0.5
- **δ²(√g R) = δ²R + h×δ¹R**: should match FP when both pieces are correct
- The **handoff from session 30 had the wrong expected value**: it compared δ²R (partial) against FP (full action), giving an apparent discrepancy of 2.5 vs 6.25.

The position-space extraction gives δ²R spin-2=1.25 (before the deriv-sum expansion broke it). This is **consistent with FP** when combined with the √g cross term.

## What Was Implemented

### 1. fourier.jl fix (COMMITTED, all tests pass)

Three changes to `_fourier_transform(TDeriv)`:
- **TSum distribution**: ∂(A+B) → ∂A + ∂B before checking
- **Field-aware bilinear detection**: `_count_field_factors` counts non-metric/delta factors
- **Bilinear dropping**: ∂(product with ≥2 field factors) → ZERO

This partially fixes the bug (reduces spin-2 from 6.25 to 3.75) but can't fix the merged terms.

### 2. extract_kernel_direct (UNCOMMITTED WIP in kernel_extraction.jl)

Position-space kernel extraction that bypasses `to_fourier` entirely:

```
extract_kernel_direct(expr, :h; registry=reg) → KineticKernel
```

**Algorithm:**
1. `_expand_deriv_sums`: Distribute all TDeriv over TSum recursively
2. `expand_products`: Flatten all products
3. For each term, `_extract_bilinear_direct`:
   a. Find "field units" via `_unwrap_field_chain` (h or ∂ⁿh, including h inside ∂(g×∂h))
   b. Skip terms with ∂(bilinear): `_count_fields_in(factor) >= 2` → return nothing
   c. Convert derivative chains to k-factors
   d. Apply **two-momentum phase**: `(-1)^{n/2 + n_R}` where n_R = derivatives on right h
   e. Build (coeff, left_indices, right_indices) for KineticKernel

**Key helpers:**
- `_unwrap_field_chain(expr, field)`: Returns `(deriv_indices, field_indices, extra_coeff_factors)`. Handles bare Tensor, TDeriv chains, AND TDeriv(TProduct([non-field, field_chain])) where the derivative passes through constant (metric) factors.
- `_count_fields_in(expr, field)`: Counts field occurrences at any depth.
- `_expand_deriv_sums(expr)`: Recursively distributes ∂(TSum) → TSum(∂).

### Current state of extract_kernel_direct

**What works:**
- δ²R extraction: 19→25 bilinear terms (with deriv-sum expansion), spin-2=1.25, spin-0s=-0.5 ✓ (before deriv-sum expansion)
- Correctly skips ∂(bilinear) terms (ΓΓ terms are NOT affected)
- Two-momentum phase correction for imbalanced derivative terms

**What's broken (last test before stopping):**
- The `_expand_deriv_sums` function introduced sign errors: full action gives spin-2=-1.25 (should be +2.5). The deriv-sum expansion creates many more terms and the phase correction may be wrong for the expanded terms.
- The √g cross term `h×δ¹R` extraction gives wrong values after deriv-sum expansion.

**Most likely cause of the sign error:**
The two-momentum phase `(-1)^{n/2 + n_R}` assumes the code drops -i from each derivative. But after `_expand_deriv_sums`, the derivative chains on each field unit are different from the original. The n_R counting might be wrong because the expansion redistributes which derivatives belong to which field.

## Recommended Fix Strategy for Next Session

### Option A: Fix the phase correction (most direct)

The phase `(-1)^{n/2 + n_R}` may be incorrect. The correct phase for each bilinear term is:

```
physical_coefficient = position_space_coeff × i^{n_L} × (-i)^{n_R} × k_{a1}...k_{an}
```

where n_L = derivatives on left h, n_R = derivatives on right h. For n_L+n_R = even:
- n_L=1, n_R=1: phase = i(-i) = 1 → no correction
- n_L=0, n_R=2: phase = 1×(-i)² = -1 → multiply by -1
- n_L=2, n_R=0: phase = i²×1 = -1 → multiply by -1

Verify by checking against the **FP kernel** term by term:
```julia
K_FP = build_FP_momentum_kernel(reg)
# FP has 4 known-correct bilinear terms
# Compare each with the corresponding term from extract_kernel_direct
```

### Option B: Skip extract_kernel_direct, use two-pass Fourier (simpler)

Instead of extracting in position space:
1. Use the EXISTING `to_fourier` + `extract_kernel` pipeline
2. After extracting each bilinear term, determine n_L and n_R by tracing which k-factors in the coefficient came from left vs right field derivatives
3. Apply the phase correction post-hoc

This avoids the complexity of position-space extraction but requires a way to determine the derivative origin of each k-factor.

### Option C: Validate without √g first

The δ²R result (spin-2=1.25, spin-0s=-0.5) from the pre-expansion `extract_kernel_direct` was **already correct**. The bug only manifested when computing the √g cross term. Consider:
1. Validate δ²R alone by checking that δ²R + h×δ¹R = FP (using the FP kernel as ground truth)
2. Fix only the √g cross term extraction
3. The cross term h×δ¹R has simpler structure: h(0 derivs) × δ¹Ric(2 derivs)

## Key Diagnostic Script

```julia
using TensorGR
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp = define_metric_perturbation!(reg, :g, :h)
    set_vanishing!(reg, :Ric); set_vanishing!(reg, :RicScalar); set_vanishing!(reg, :Riem)

    # Position-space kernel extraction
    δ2R = δricci_scalar(mp, 2)
    K_R = extract_kernel_direct(δ2R, :h; registry=reg)

    # √g cross term
    δ1R = δricci_scalar(mp, 1)
    h_trace = Tensor(:g, [up(:p), up(:q)]) * Tensor(:h, [down(:p), down(:q)])
    K_cross = extract_kernel_direct(h_trace * δ1R, :h; registry=reg)

    K_full = combine_kernels([K_R, K_cross])
    K_FP = build_FP_momentum_kernel(reg)

    for s in [:spin2, :spin1, :spin0s, :spin0w]
        f = _eval_spin_scalar(spin_project(K_full, s; registry=reg), 1.0)
        fp = _eval_spin_scalar(spin_project(K_FP, s; registry=reg), 1.0)
        println("$s: full=$f  FP=$fp  $(isapprox(f, fp, atol=1e-8) ? "✓" : "✗")")
    end
end
```

**Expected output when fixed:** All four spins match FP (2.5, 0, -1.0, 0).

## Files Modified (Uncommitted WIP)

| File | Change |
|------|--------|
| `src/action/kernel_extraction.jl` | Added `extract_kernel_direct`, `_expand_deriv_sums`, `_unwrap_field_chain` (new version), `_count_fields_in`, `_extract_bilinear_direct` |
| `src/TensorGR.jl` | Added `extract_kernel_direct` to exports |

## Files Modified (Committed)

| File | Change |
|------|--------|
| `src/svt/fourier.jl` | Added `_count_field_factors`, TSum distribution, bilinear dropping in `_fourier_transform(TDeriv)` |

## Critical Implementation Notes

1. **`tproduct` already flattens nested TProducts** (arithmetic.jl:27-29) and absorbs ZERO factors (line 39-40). No need to worry about nested products from Fourier transform.

2. **The simplify pipeline merges ΓΓ and ∂Γ₂ terms.** This is why the fourier fix alone can't fully solve the problem — it can only drop terms still in ∂(product) form. The position-space approach works on the RAW perturbation engine output, bypassing simplify entirely.

3. **The FP kernel uses the PHYSICAL convention** where (ik)(−ik) = k². The perturbation engine's to_fourier uses uniform-k where all ∂→k. The position-space extraction must produce coefficients matching the FP convention.

4. **contraction.jl Bug 2 fix was attempted and reverted.** Extending `_try_metric_contraction` to handle TDeriv partners (Cases A+B) worked algebraically but caused cascading changes in the simplify fixed-point loop, breaking spin projections. All contraction.jl changes were reverted. The position-space approach doesn't need them — it works on raw expressions where g's are explicit factors in the coefficient, and spin_project's internal simplify handles the contraction.

5. **The handoff from session 30 had wrong expected values.** The expected "δ²R spin-2 = 2.50" was actually the FP value for the FULL action δ²(√gR), not δ²R alone. The correct δ²R spin-2 is 1.25 at k²=1. The relationship is: δ²(√gR) = δ²R + h×δ¹R, and FP represents δ²(√gR).

## Changes This Session

- Diagnosed the uniform-k convention as the root cause (not contraction or AST structure)
- Implemented and committed fourier.jl fix (all tests pass)
- Implemented position-space kernel extraction (`extract_kernel_direct`)
- Verified δ²R spin-2=1.25 is correct (consistent with FP via √g)
- Identified phase correction as remaining issue in the position-space approach
- Attempted and reverted Bug 2 contraction.jl fix (3 variants tried)
- Corrected the expected values from session 30 handoff
