# HANDOFF — 2026-03-16 Regression Investigation & Partial Fix

## DO NOT DELETE THIS FILE. Read it completely before working.

## Executive Summary

A deep investigation found TWO independent regressions causing the bench_12 R³ term count to inflate from 324 to 685. Regression 1 (canonicalize.jl conjugation) has been reverted. Regression 2 (perturbation `_avoid` sets) is kept because removing it breaks kernel extraction. The March 15 chaos wave (~20 parallel agents, 15k lines) has been quarantined on `march15-preserve`. Current state: R³=409, 337,351 pass, 19 fail, 0 error. The 19 failures need fixing before anything else.

---

## What Happened (Chronological)

### March 10: Regression 1 — canonicalize.jl (commit 1faf32f)
An agent added generator conjugation (`perm * gen * perm_inv`) to `_canonicalize_product` and changed reconstruction from `cperm_inv.data[slot]` to `cperm.data[slot]`. The stated goal was fixing dummy index pairing for spin projection. The actual effect: canonicalization became ~2x weaker (324→505 terms for R³ on dS). **This has been REVERTED** — unconjugated generators + `cperm_inv` reconstruction restored.

### March 11: Regression 2 — perturbation expand.jl (commit 858d73e)
An agent added `_avoid::Set{Symbol}` parameter threading through the entire perturbation expansion chain (`δinverse_metric` → `δchristoffel` → `δriemann` → `δricci` → `δricci_scalar`). When `_avoid` is non-empty, the memo cache is bypassed and each sub-expression uses fresh dummy names. Effect: 303→409 terms for R³. **This is KEPT** because removing it causes index scoping bugs in kernel extraction (indices appearing 3+ times after flattening nested `TDeriv(TProduct(...))` structures).

### March 15: The Chaos Wave
~20 parallel worktree agents committed 15,239 lines across 62 files in a 3-hour window (16:14-18:57). Multiple agents edited the same core pipeline files from isolated worktrees, merged together via conflict resolution. Key commits that compounded regression 1:
- `355d298` — extended canonicalize.jl with vbundle sort key (505→685)
- `79c617b` / `7012156` — added enforce_tracefree/enforce_divfree to simplify pipeline
- `9aed69d` — mega-squash "Add 9 new subsystems"

**All March 15+ work preserved on `march15-preserve` branch.** Master reset to `9c65895` (March 14 EOD).

### March 16: This Investigation
- Git bisect confirmed 1faf32f as first bad commit
- Verified 858d73e as secondary regression source via worktree testing
- Reverted canonicalize.jl conjugation (restores strong term merging)
- Fixed `contract_metrics` to handle same-position metric traces (`g^{a,a}` → dim)
- Fixed `extract_kernel_direct` to deconflict duplicate dummy names between field halves
- Fixed `spin_project` to call `fix_dummy_positions` on coefficients and post-simplify results
- Verified (δR)² spin projections give correct physics values (0, 3, 0, 0)

---

## Current State of the Code

### Modified Files (relative to March 14 HEAD)

**`src/algebra/canonicalize.jl`** — Reverted conjugation:
- Removed `perm * gen * perm_inv` conjugation of xperm generators (lines ~205-250)
- Removed pre-loop `perm = Perm(perm_data)` and `perm_inv_data = perm_inverse(perm)`
- Restored `cperm_inv = perm_inverse(cperm)` in reconstruction
- Changed `cname = Int(cperm.data[slot])` back to `cname = Int(cperm_inv.data[slot])`
- Kept ALL vbundle-aware changes (Dict{Tuple{Symbol,Symbol}}, vbundle sort key)

**`src/algebra/contraction.jl`** — Same-position metric/delta trace:
- Line ~24: Removed `position != position` requirement for delta self-contraction
- Line ~38: Removed `position != position` requirement for metric self-contraction
- Now `g^{a,a}` and `δ^{a,a}` correctly contract to dim (was silently passed through)

**`src/action/kernel_extraction.jl`** — Three fixes:
1. `extract_kernel_direct`: calls `fix_dummy_positions` on input, then `_deconflict_field_halves` to rename dummies in the second field-containing factor
2. `_deconflict_field_halves`: new function that identifies two field-containing factors in a TProduct and calls `ensure_no_dummy_clash` between them
3. `spin_project`: calls `fix_dummy_positions(bt.coeff)` before projector construction, and `fix_dummy_positions(next)` after each simplify iteration

### Test Results
```
337,351 passed, 19 failed, 0 errored, 0 broken
```

### Failing Tests (19 total)

| Test | File:Line | Count | Issue |
|------|-----------|-------|-------|
| δ²S term counts | test_6deriv_spectrum.jl:816,854 | 2 | `length(δ2R.terms) == 22` evaluates to `9 == 22` (better simplification, assertion needs updating) |
| h^{ab}δ¹G = FP kernel | test_kernel_extraction.jl:129 | 2 | spin1=3.0 (expected 0), spin0w=1.5 (expected 0) — gauge sectors nonzero |
| (δRic)² spin projections | test_kernel_extraction.jl:165-166 | 2 | spin0s and spin1 values wrong |
| 4-derivative flat spectrum | test_kernel_extraction.jl:207 | 13 | Multiple parameter combinations give wrong spin projection values |

### Root Cause of Remaining 19 Failures

The unconjugated canonicalization produces **same-position dummy pairs** (e.g., both Up or both Down for the same index name). This is mathematically valid for term merging but breaks three downstream mechanisms:

1. **`_analyze_indices`** (in indices.jl) — only detects Up/Down pairs as dummies. Same-position pairs are invisible, so `ensure_no_dummy_clash` misses them.

2. **`contract_momenta`** (in kernel_extraction.jl) — requires opposite positions to contract `k_a k^a → k²`. Same-position `k_a k_a` passes through uncontracted.

3. **`_try_metric_contraction`** in products (contraction.jl) — the product-level metric contraction also checks `position != position` for dummy detection. Same-position dummies in products don't trigger contraction.

The `contract_metrics` fix for self-traced metrics was necessary but not sufficient. The same-position issue permeates the contraction pipeline.

### Architecture: Why This Is Hard

The canonicalization uses **all-free mode** — it treats every index as free for xperm, getting maximum symmetry exploitation. The trade-off: xperm can swap index names between Up and Down slots, producing same-position dummy pairs. These are valid canonical representatives (two terms that differ only by dummy position are mathematically identical and WILL be merged by `collect_terms`). But downstream code that needs to CONTRACT indices requires proper Up/Down pairing.

Two approaches to fix:
1. **Make all downstream consumers same-position-aware** — modify `_analyze_indices`, `contract_momenta`, `_try_metric_contraction`, etc. to treat same-name same-position pairs as dummies. This is the thorough fix.
2. **Apply `fix_dummy_positions` at the API boundary** — call it in `simplify`'s output, or in `extract_kernel_direct`, or wherever downstream code needs valid pairs. The challenge: adding it to the simplify inner loop broke convergence (tried and reverted).

Approach 1 is cleaner. Approach 2 is faster but fragile.

---

## Git Recovery Information

| Tag/Branch | Commit | Description |
|------------|--------|-------------|
| `march15-preserve` | `ebfae36` | Full HEAD before reset — ALL March 15+ work (15k lines, 62 files) |
| `pre-regression-baseline` | `09be750` | Last state where R³ = 324 with all tests passing |
| `regression1-canonicalize` | `1faf32f` | The canonicalization conjugation commit |
| `regression2-avoid` | `858d73e` | The perturbation `_avoid` commit |
| `march15-chaos-start` | `9c65895` | Last commit before March 15 chaos (current master base) |

---

## Beads Issues

| ID | Priority | Title | Status |
|----|----------|-------|--------|
| TGR-9ay | P0 | Fix kernel extraction index scoping for unconjugated canonicalization | in_progress |
| TGR-3et | P1 | Update test assertions for improved canonicalization term counts | open (blocked by 9ay) |
| TGR-e04 | P2 | Investigate removing `_avoid` to recover 303-term R³ simplification | open (blocked by 9ay) |
| TGR-t28 | P2 | Incrementally merge March 15 subsystems from march15-preserve | open (blocked by 9ay) |

---

## How to Test

```bash
# R³ benchmark (currently 409, target ≤324):
julia --project -e '
using TensorGR
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    maximally_symmetric_background!(reg, :M4; metric=:g, cosmological_constant=:Λ)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp = define_metric_perturbation!(reg, :g, :h; curved=true)
    R1 = Tensor(:RicScalar, TIndex[])
    expr = R1 * R1 * R1
    raw = expand_perturbation(expr, mp, 2)
    s = simplify(raw; registry=reg, maxiter=100)
    n = s isa TSum ? length(s.terms) : 1
    println("R^3 terms: $n")
end'

# Quick spin projection sanity check ((δR)² should give 0, 3, 0, 0):
julia --project -e '
using TensorGR
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp = define_metric_perturbation!(reg, :g, :h)
    set_vanishing!(reg, :Ric); set_vanishing!(reg, :RicScalar); set_vanishing!(reg, :Riem)
    kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)
    d1R = simplify(δricci_scalar(mp, 1); registry=reg)
    K = extract_kernel_direct(d1R * d1R, :h; registry=reg)
    for s in [:spin2, :spin0s, :spin1, :spin0w]
        v = _eval_spin_scalar(spin_project(K, s; kw...), 1.0)
        println("  $s = $v")
    end
end'

# Full test suite (target: 0 fail, 0 error):
julia --project -e 'using Pkg; Pkg.test()'
```

---

## Rules for Future Agents

1. **DO NOT re-introduce generator conjugation in canonicalize.jl.** The unconjugated approach gives 2x better simplification. The downstream issues must be fixed downstream.
2. **DO NOT batch-merge from march15-preserve.** Merge subsystems one at a time with full test suite validation.
3. **DO NOT use parallel worktree agents on core pipeline files.** The March 15 chaos resulted from this.
4. **DO NOT modify canonicalize.jl without running R³ benchmark.** Ground truth is ≤409 terms.
5. **DO NOT trust subagent outputs without independent verification.** An agent fabricated 31 test failures during this investigation.
6. **DO NOT add `fix_dummy_positions` to the inner simplify loop.** This was tried and broke convergence. Fix consumers instead.

---

## Key Source Files

| File | Role | Modified? |
|------|------|-----------|
| `src/algebra/canonicalize.jl` | xperm canonicalization (THE core fix) | YES |
| `src/algebra/contraction.jl` | Metric/delta contraction | YES (same-position trace) |
| `src/action/kernel_extraction.jl` | Bilinear kernel extraction + spin projection | YES (deconflict + fix_dummy) |
| `src/algebra/simplify.jl` | Simplify pipeline orchestrator | no |
| `src/perturbation/expand.jl` | Perturbation engine (has _avoid) | no |
| `src/xperm/wrapper.jl` | xperm.c FFI | no |
| `docs/xperm_algorithm.md` | xperm convention documentation | no |
| `src/algebra/canonicalize.jl` → `fix_dummy_positions` | Repair same-position dummy pairs | no (utility, already existed) |
