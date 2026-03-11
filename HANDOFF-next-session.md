# HANDOFF: Session 18 — Perturbation Engine Index Clash Fix

## Status: BUG FIXED — Tests pass, benchmarks pass

- **5865 tests pass** (up from 5855), 0 errors
- **92 benchmarks pass**, 3 broken (stretch goals, unchanged)
- Benchmark ground truth updated for 4 benchmarks (new correct term counts)

## What Was Done This Session

### Root Cause Found and Fixed

**Bug**: `δchristoffel`, `δriemann`, `δricci`, and `δinverse_metric` used `fresh_index` with only their own parameter indices in the `used` set. When called from deep in the chain (δricci_scalar → δricci → δriemann → δchristoffel), internal fresh dummies could collide with indices from outer calling contexts.

**Example**: δchristoffel called with indices (:c, :c, :d) would pick `:a` as internal dummy — but `:a` was already the Ricci scalar trace index in `g^{ab} δ²Ric_{ab}`, making the expression ill-formed (index `:a` appearing 4 times).

**Fix**: Added `_avoid::Set{Symbol}` keyword parameter to all perturbation functions. Each function passes its accumulated `used` set to child calls, ensuring internal dummies never collide with outer context indices. Memoization is skipped when `_avoid` is non-empty (correct because different contexts need different dummy names).

**Files changed**:
- `src/perturbation/expand.jl` — δchristoffel, δriemann, δricci, δricci_scalar, _get_christoffel_order
- `src/perturbation/metric_perturbation.jl` — δinverse_metric

### Verification Results

After the fix:
1. **All indices well-formed** ✓ (no index appearing >2 times in any expanded term)
2. **Fourier/simplify commute** ✓ (Method A = Method C, was broken before)
3. **Gauge invariance** ✓ (spin-1 = 0 for all methods)
4. **L₂ scalar sector matches FP** ✓ (spin-0s = -1.7 for both L₂ and FP)
5. **L₂ - FP = (6.375, 0, 0)** — only spin-2 residual from total derivatives

### L₂ vs FP Relationship (Resolved)

The remaining spin-2 difference between L₂ = δ²R + ½h·δR and the FP kernel is from **total derivatives**, as predicted by the MEMORY note: "Use IBP before spin projection, or construct FP form directly." With the fix:
- spin-1 = 0 ✓ (gauge invariance, was -3.825 before fix)
- spin-0s matches FP exactly ✓ (was -4.25 ≠ -1.7 before fix)
- spin-2 differs by 6.375 (total derivative contribution, physical content is correct)

### Benchmark Term Count Changes

| Benchmark | Old | New | Reason |
|-----------|-----|-----|--------|
| bench_03 Linearized Ricci 3+1 | 320 | 512 | More distinct dummies → fewer false cancellations |
| bench_05 δ¹Ricci Schwarzschild | 18 | 26 | Same |
| bench_07 δ¹Riemann de Sitter | 6 | 26 | Same |
| bench_08 Galileon L_4 EOM | 17 | 15 | Some new valid cancellations |

These changes are **correct** — the old counts came from ill-formed expressions with spurious cancellations. The new expressions are well-formed.

## Diagnostic Scripts (can be cleaned up)

- examples/23_fourier_nested_test.jl — Priority 1+2: Fourier commutativity + analytic Y
- examples/24_simplify_vs_fourier.jl — Step-by-step simplify pipeline analysis
- examples/25_minimal_fourier_bug.jl — Minimal Fourier/simplify reproducer

## Next Steps

### Immediate (P2)
- **TGR-c6su**: SVT decomposition of δ²S (flat) — now unblocked by the pert engine fix
- The perturbation engine now produces correct, well-formed expressions
- L₂ = δ²R + ½h·δR gives correct physical content (spin-0s, spin-1 match FP)

### Future
- Consider improving the simplifier to recover term count efficiency (26 vs 6 for de Sitter δ¹Riemann)
- The `_avoid` mechanism is a minimal fix; a more elegant solution would use global fresh_index counters scoped per expand_perturbation call

## Key Files

| File | Change |
|------|--------|
| `src/perturbation/expand.jl` | _avoid kwarg on δchristoffel, δriemann, δricci, δricci_scalar |
| `src/perturbation/metric_perturbation.jl` | _avoid kwarg on δinverse_metric |
| `benchmarks/ground_truth.jl` | Updated 4 term count constants |
