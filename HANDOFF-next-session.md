# HANDOFF: 6-Deriv dS Spectrum — Session 16

## Status: ALL GREEN (tests unchanged)

- **5851 tests pass**, 0 errors, 0 broken, 0 failed
- **13 benchmarks pass** (271 benchmark tests)
- All code pushed to remote (after this commit)

## Completed This Session (16)

- **Rewrote `examples/15_perturbation_spectrum_crosscheck.jl`** to attempt full BC parameter cross-check via perturbation engine:
  - Added `sqrt_g_correction()` helper for metric determinant expansion
  - Included cosmological constant `-2Λ` for on-shell gauge invariance
  - Full pipeline: `expand_perturbation → commute_covds → to_fourier → extract_kernel → spin_project`
  - Pipeline runs end-to-end without errors (16 bilinear kernel terms for EH)

- **Identified normalization mismatch**: The perturbation engine path produces spin projections that don't match the direct FP momentum kernel builder:
  - spin-2: factor of **-0.5** relative to `build_FP_momentum_kernel`
  - spin-0s: factor of **2.5** relative to FP
  - spin-1: **non-zero** (should be 0 for gauge-invariant EH action)
  - spin-0w: 0 ✓

## Root Cause Analysis (for next session)

The mismatch between the perturbation engine's `δ²R` path and the direct FP kernel builder has NOT been resolved. Likely causes, in order of probability:

### 1. Missing `½h·δR` normalization (MOST LIKELY)
The FP Lagrangian is `L_FP = δ²(√g·R)/√g₀ = δ²R + ½h·δR + (⅛h² − ¼hh)·(R₀−2Λ)`. At Λ→0, `L_FP = δ²R + ½h·δR`. The `½h·δR` term mixes trace and TT sectors. If the perturbation engine's δ²R already includes some of these cross-terms (from the Cauchy product), they may be double-counted.

**Diagnosis**: Compute `δ²R` alone (without √g correction), Fourier transform, spin project. Compare spin-2 against FP. If spin-2 is ½ of FP, then the `½h·δR` term accounts for the other half. If spin-2 is -½ of FP (as observed), there's a sign issue.

### 2. Fourier convention sign issue
`to_fourier` replaces `∇_a → k_a` (no imaginary unit, no sign). The FP builder uses the same convention. BUT: the covariant derivative `∇g` on MSS includes Christoffel terms. When `commute_covds` sorts derivatives, the commutator `[∇_a, ∇_b]h = Riem·h` produces extra terms. These extra terms may have wrong signs relative to the ∂→k convention.

**Diagnosis**: Try the non-covariant perturbation mode (remove `covariant_output=true`) and use partial derivatives. The `to_fourier` default handles `∂ → k` cleanly.

### 3. The `δ²R` definition vs Taylor convention
The memory says "`δricci_scalar(mp, n)` returns the ε^n COEFFICIENT". This means `R(ε) = R₀ + ε·δR + ε²·δ²R + ...`. The second variation of the action is `ε²·∫√g₀·Q` where `Q = δ²R + ½h·δR + ...`. If instead `δ²R` = `R''(0)` (without the 1/2! factor), then `Q = ½·δ²R + ½h·δR + ...`.

**Diagnosis**: Check a simple case: for g = η + εh with h = const·η (pure trace), compute δ²R analytically and compare with `expand_perturbation(R, mp, 2)`.

### 4. Integration by parts (IBP) needed
The quadratic Lagrangian may contain total derivatives (terms like ∂_μ(h·∂_νh)). These contribute to the bilinear kernel but integrate to zero. Before spin projection, IBP should be applied to put everything in standard form (all derivatives acting symmetrically on the two h's). Without IBP, the kernel has extra terms that pollute the spin projections.

**Diagnosis**: Check if the kernel has terms with all momenta on one side (asymmetric). The FP builder has balanced momentum structure by construction.

## Recommended Next Session Strategy

1. **Quick diagnostic** (~5 min): Compute δ²R on flat (no MSS), add `set_vanishing!` for Ric and RicScalar, then Fourier + spin project. This avoids MSS complications and tests the core pipeline.

2. **Non-covariant path** (~5 min): Try `define_metric_perturbation!(reg, :g, :h; curved=true)` (no `covariant_output=true`), use `to_fourier` with default convention. The non-covariant expressions have partial derivatives only.

3. **IBP experiment** (~10 min): Before `extract_kernel`, apply `ibp_product` to symmetrize derivatives. Compare spin projections before/after.

4. **Normalization test** (~5 min): Check if `expand_perturbation(R, mp, 2)` returns the Taylor coefficient or the second derivative. Use a pure-trace perturbation h = λ·g where δ²R is analytically known.

## Key Files

| File | Role |
|------|------|
| `examples/15_perturbation_spectrum_crosscheck.jl` | Cross-check script (UPDATED this session) |
| `src/action/kernel_extraction.jl` | KineticKernel, spin_project, BC params |
| `src/svt/fourier.jl` | `to_fourier`: replaces ∂/∇ → k (no sign, no i) |
| `src/perturbation/expand.jl` | `expand_perturbation` (Cauchy product engine) |
| `src/algebra/ibp.jl` | `ibp_product` (integration by parts) |
| `benchmarks/bench_13_spectrum.jl` | Existing spin projection tests (FP, R², Ric²) |

## Critical Gotchas (accumulated)

- `spin_project` returns **Tr(K·P^J)**, NOT f_J. Divide by {5,3,1,1}
- On flat without MSS rules, spin projection leaves uncontracted `Ric` tensors → use MSS or `set_vanishing!`
- `to_fourier` uses ∂→k (no -i, no sign) — same convention as `build_FP_momentum_kernel`
- The cosmological constant -2Λ is needed in L₀ for gauge invariance (spin-1=0) on MSS
- `δricci_scalar(mp, n)` returns ε^n COEFFICIENT (Cauchy product)
- √g expansion: Q = δ²L + ½h·δL + (⅛h² − ¼hh)·L₀ (see derivation in script header)

## Open Beads Issues (6-deriv related)

| Issue | Status | Description |
|-------|--------|-------------|
| TGR-c6su | OPEN P2 | Step 2.1: SVT decomposition of δ²S (flat) |
| TGR-pr04 | OPEN P2 | Step 2.2: SVT QuadraticForms + propagators |
| TGR-tztc | OPEN P2 | Step 2.3: Cross-check Path A vs Path B |
| TGR-j6r9 | OPEN P2 | Step 5: Tests + benchmark (blocked by tztc) |
| TGR-af4a | OPEN P2 | Step 6: Example script + module integration |
