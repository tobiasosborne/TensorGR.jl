# WORKLOG — Principled Canonicalize + Derivative Distribution Fix

## Date: 2026-03-17

## Checkpoint: 11f8ff8

Committed: canonical_perm_ext + inner sum collection + _apply_position_fixes.

## Results After Checkpoint

### spin1 Investigation
- `flatten_metric_derivs` implemented: applies Leibniz + ∂g=0 to ∂(g*∂h) → g*∂²h
- Handles TDeriv wrapping TSum (distributes derivative first)
- Handles scalar extraction (∂(c*X) → c*∂X)
- Produces 3 flattened terms from the 2-term perturbation d1R_ab

### But spin1 remains 0.75
The flattened d1R_ab has different algebraic structure from the manual construction:

Flattened (from perturbation, gives spin1=0.75):
```
1. g^{cd} * ∂_c∂_a(h_{b,d})      — ∂_a∂^d h_{bd} term (coefficient 1)
2. (-1/2) * g^{cd} * ∂_c∂_d(h_ab) — -□h_{ab}/2 term
3. -(1/2) * g^{cd} * ∂_b∂_c(h_ad) — -∂_b∂^d h_{ad}/2 term
```

Manual (gives spin1=0):
```
1. (-1/2) * g^{b,c} * ∂_c∂_a(h_{d,d}) — -∂_a∂_b h/2 (trace term)
2. g^{cd} * ∂_c∂_b(h_{a,d})            — ∂_b∂^d h_{ad} term (coefficient 1)
3. (-1/2) * g^{cd} * ∂_c∂_d(h_{a,b})   — -□h_{ab}/2 term
```

Both are algebraically equivalent (same tensor δR_{ab}) but have different coefficient
partitioning between the ∂_a∂^c h_{bc} and ∂_b∂^c h_{ac} terms. The kernel extraction
is sensitive to this partitioning.

### Root Cause of spin1 residual
The perturbation engine produces δR_{ab} in a specific algebraic form that DIFFERS from
the canonical textbook form. Both are correct expressions for the same tensor, but the
kernel extraction assumes a specific structure (each h factor with clearly separated
derivative count) that matches the canonical form but not the perturbation form.

### Fix Options (not yet implemented)
1. **Fix perturbation engine**: Make δricci produce the canonical 4-term form directly
2. **Fix kernel extraction**: Handle arbitrary algebraically-equivalent forms of δR_{ab}
3. **Add symmetrization**: Symmetrize d1R_ab in (a,b) before extraction
4. **Manual override**: Allow users to provide δR in canonical form (workaround, not fix)

## Full Summary of Changes

### Committed (11f8ff8):
- `xperm_canonical_perm_ext` wrapper
- Proper dummy canonicalization
- `collect_inner_sums` + `_merge_identical_terms`
- `_apply_position_fixes` for TProduct/TSum

### Uncommitted (working tree):
- `flatten_metric_derivs` function (Leibniz + ∂g=0)
- `distribute_derivs_over_sums` improvements (expand_products inside, scalar recursion)
- Export of `flatten_metric_derivs`

### Test Results

| Test | Before All Changes | After Committed | With flatten | Status |
|------|-------------------|-----------------|--------------|--------|
| Idempotency | Period-2 | Stable | Stable | ✅ |
| spin2 | Wrong | 2.5 ✓ | 2.5 ✓ | ✅ |
| spin0s | Wrong | -1.0 ✓ | -1.0 ✓ | ✅ |
| spin0w | 1.5 | 0.0 ✓ | 0.0 ✓ | ✅ |
| spin1 | 3.0 | 0.75 | 0.75 | ⚠️ |
| spin1 (manual δR) | — | — | 0.0 ✓ | ✅ (proves extraction works) |
| R³ terms | 362 | 229 | — | ✅ improved |

## Physics Ground Truth
- spin1 = 0, spin0w = 0 for ALL kernels (diffeomorphism invariance)
- ALL 4 constraints satisfied with manual flat-space δR construction
- 3/4 constraints satisfied with perturbation-generated δR
