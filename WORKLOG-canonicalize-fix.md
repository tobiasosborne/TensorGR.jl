# WORKLOG — Principled Canonicalize + Derivative Distribution Fix

## Date: 2026-03-17

## Summary of All Changes

### Committed:
1. `11f8ff8` — canonical_perm_ext + inner sum collection + _apply_position_fixes
2. `f51146f` — flatten_metric_derivs + distribute_derivs improvements
3. `5c49566` — kernel extraction bug identification
4. `612a60a` — kernel metric contraction in coefficients

### Uncommitted (current working tree):
5. `δricci_flat` / `δricci_scalar_flat` — flat-space linearized Ricci in canonical form
6. `_merge_kernel_terms` (disabled) — kernel term merging attempt (L/R boundary issues)
7. Various exports

## Root Causes Found

### 1. canonical_perm non-idempotency — FIXED ✅
`canonical_perm`'s PERM^{-1} conversion scrambles dummy pairings.
Fix: `xperm_canonical_perm_ext` wrapper passes names directly to C function.

### 2. All-free canonicalization → same-position pairs — FIXED ✅
Proper Up/Down pairs now classified as dummies with metricQ=1.

### 3. Trapped inner sums — FIXED ✅
`collect_inner_sums` + `_merge_identical_terms` in simplify pipeline.

### 4. Kernel g-factors in coefficients — FIXED ✅
`_contract_kernel_metrics` contracts metrics with h indices post-extraction.
`contract_metrics` + `contract_momenta` on remaining coefficient factors.

### 5. Perturbation δR_{ab} non-manifest symmetry — ROOT CAUSE OF spin1 ✅
The Christoffel-based formula ∂_c(δΓ^c_{ab}) - ∂_b(δΓ^c_{ac}) is NOT manifestly
symmetric. After inner sum merging, the coefficient ratios become asymmetric,
causing the kernel extraction to produce non-canceling spin1 contributions.

Fix: `δricci_flat` constructs the canonical 4-term form directly:
  δR_{ab} = (1/2)*g^{cd}*(∂_a∂_c h_{bd} + ∂_b∂_c h_{ad} - ∂_c∂_d h_{ab} - ∂_a∂_b h_{cd})

With this form: spin1=0 ✓, spin2=2.5 ✓

### 6. δricci_scalar_flat coefficients — REMAINING BUG
spin0s=0.5 (should be -1.0), spin0w=-1.5 (should be 0.0).
The `δricci_scalar_flat` formula likely has wrong coefficients.
The correct flat-space formula is: δR = ∂^a∂^b h_{ab} - □h
Current implementation uses a 3-term g^{ab}g^{cd} form that may have errors.

## Test Results

| Test | Result | Status |
|------|--------|--------|
| spin2 (with δricci_flat) | 2.5 = FP | ✅ |
| spin1 (with δricci_flat) | 0.0 = FP | ✅ |
| spin0s (with δricci_flat) | 0.5 ≠ -1.0 | ❌ needs δricci_scalar_flat fix |
| spin0w (with δricci_flat) | -1.5 ≠ 0.0 | ❌ needs δricci_scalar_flat fix |
| spin2 (with δricci from perturbation) | 2.5 = FP | ✅ |
| spin1 (with δricci from perturbation) | 0.75 ≠ 0.0 | ⚠️ non-manifest symmetry |
| R³ terms | 229 (was 362) | ✅ improved |
| Idempotency | Stable | ✅ |

## Next Steps for Continuation

### Immediate: Fix δricci_scalar_flat
The scalar formula should be: δR = g^{ab}*δR_{ab} where δR_{ab} is the flat form.
Instead of constructing separately, TRACE the δricci_flat result:
```julia
d1R = simplify(Tensor(:g, [up(:a), up(:b)]) * δricci_flat(mp, down(:a), down(:b)), registry=reg)
```
This avoids coefficient errors from a separate formula.

### Then: Integration
- Test all 3 kernels (FP, R², Ric²) with δricci_flat
- Run targeted test suite (kernel extraction tests)
- Verify bench_12 with increased maxiter

### Architecture notes
- `flatten_metric_derivs` EXISTS but doesn't help for the spin1 issue because
  inner sum merging corrupts coefficient ratios before flattening can fix them
- The `_merge_kernel_terms` function was attempted but corrupts L/R index
  boundaries during _normalize_dummies. Reverted.
- The proper approach (matching xAct's `IndexCoefficient` + `CollectTensors`)
  would need L/R-aware dummy normalization. This is a significant infrastructure
  addition, not attempted yet.

## Key Insight from xAct Research
xAct does NOT have built-in kernel extraction or spin projections.
Users extract kernels manually via `IndexCoefficient` + `CollectTensors`.
The key enabler is `UseMetricOnVBundle->All` in `ToCanonical`, which allows
the Butler-Portugal algorithm to merge equivalent terms via metric-aware
dummy relabeling. TensorGR now has this via `canonical_perm_ext`.

## Files Modified
- `src/xperm/wrapper.jl` — `xperm_canonical_perm_ext`
- `src/algebra/canonicalize.jl` — proper dummy routing, _apply_position_fixes
- `src/algebra/simplify.jl` — collect_inner_sums, distribute_derivs, flatten_metric_derivs
- `src/action/kernel_extraction.jl` — _contract_kernel_metrics, δricci_flat, δricci_scalar_flat
- `src/TensorGR.jl` — exports
