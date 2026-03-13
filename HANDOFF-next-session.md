# HANDOFF: Session 28 — Corrected Diagnosis of TGR-76k Crosscheck Failure

## Status: TGR-wm8 closed, TGR-dp3 rediagnosed, 1 new issue needed

- **All 7267 tests pass**: full suite unchanged
- **TGR-wm8 CLOSED**: √g coefficient reverted 1/4→1/8 in example 15
- **TGR-dp3 MISDIAGNOSED**: the bug is NOT in collect_terms/canonicalize

## What Was Done This Session

### 1. Attempted TGR-dp3 fix (canonicalizer mixed positions)

Implemented `normalize_field_positions` in `src/algebra/canonicalize.jl` — lowers
all Up indices on a specified field by inserting inverse metric connectors. Function
works correctly and is exported. However, **it doesn't fix the crosscheck failure**
because the root cause is elsewhere.

### 2. Fixed TGR-wm8 (√g coefficient)

Changed `examples/15_perturbation_spectrum_crosscheck.jl` line 49: `1//4` → `1//8`.
The standard √g expansion to second order has (1/8)(tr h)², not (1/4).

### 3. Deep diagnosis: identified real root cause

**The bug is NOT in collect_terms or canonicalize.** Verified by:
- `spin_project` correctly handles shared dummies between h factors (confirmed with FP kernel: exact 2.5k², 0, -k², 0)
- `_standardize_h_indices` properly lowers mixed-position indices
- `normalize_field_positions` correctly lowers all h to Down position

**The actual bug: the perturbation engine produces wrong δ²R on flat background.**

## Root Cause: Wrong δ²R Coefficients

### Evidence

```
δ²R alone (flat):     spin-2=6.25,  spin-0s=0.5    (at k²=1)
√g correction (flat): spin-2=0.0,   spin-0s=-1.5
Sum:                  spin-2=6.25,  spin-0s=-1.0
FP kernel:            spin-2=2.5,   spin-0s=-1.0   ← correct
```

- **spin-0s is correct**: 0.5 + (-1.5) = -1.0 ✓
- **spin-2 is 2.5× too large**: 6.25 instead of 2.5

The √g correction contributes 0 to spin-2 (it only has trace-type structures).
So δ²R alone should give spin-2 = 2.5, but gives 6.25.

### Specific coefficient error

The Fourier-transformed δ²R has 8 terms. The h^{ab}h_{ab}k² term (A-type)
has coefficient **5/4**, but it should contribute to give total spin-2 = 2.5.

The spin-2 projection of the A-type structure gives: Tr(P²) × coeff × k² = 5 × coeff × k².
With coeff=5/4: 5 × 5/4 = 25/4 = 6.25. All other terms (B,C,D-type) project to 0
for spin-2. So the A-type coefficient is the sole source of error.

Expected: 5 × coeff = 2.5 → coeff = 1/2. Got: 5/4.

### Where to investigate

The excess 3/4 in the A-type coefficient comes from δ²R itself. Possible causes:

1. **δricci_scalar convention**: Does `δricci_scalar(mp, 2)` return d²R/dε² or
   (1/2)d²R/dε²? The crosscheck formula may assume one convention while the
   engine uses another.

2. **Inverse metric perturbation**: δ²R involves δ(g^{-1}) terms. If the
   expansion of g^{ab} to second order has wrong coefficients, it would
   propagate into δ²R.

3. **Term cancellation failure**: The 22-term position-space δ²R may contain
   terms that should cancel but don't due to a simplification issue (not in
   collect_terms, but perhaps in contract_curvature or expand_products).

## Recommended Next Session: Deep Research Plan

### Phase 1: Isolate the δ²R discrepancy (use parallel subagents)

**Subagent A**: Read `src/perturbation/expand.jl` and `src/perturbation/linearize.jl`.
Document the exact convention: does `expand_perturbation(R, mp, n)` return
dⁿR/dεⁿ or (1/n!)dⁿR/dεⁿ? Check the docstrings and the base cases (n=0,1).

**Subagent B**: Read `src/perturbation/metric_perturbation.jl`. Check how
δ(g^{ab}) is computed at order 2. The standard formula is:
δ²(g^{ab}) = 2h^{ac}h^b_c (convention-dependent). Verify the coefficient.

**Subagent C**: Compute δ²R by hand for a minimal case. Take h_{ab} = εδ_{a0}δ_{b0}
on flat 4D background. Compute R(ε) analytically to second order. Compare with
what `expand_perturbation` gives for this specific perturbation.

**Subagent D**: Check if the crosscheck formula in example 15 has the right
overall factor. The formula Q = δ²L + √g_correction should be
Q = (1/2)(d²/dε²)(√g L)|_{ε=0} if the action is S = (1/2)∫h K h.
Verify whether example 15 accounts for this 1/2 correctly.

### Phase 2: Fix and verify

Once the convention mismatch is identified:
1. Fix either the perturbation engine or the example 15 formula
2. Re-run example 15 → FP check should pass
3. Run full test suite
4. Close TGR-dp3 (with corrected description) and TGR-76k

## Key Files

| File | Role |
|------|------|
| `src/perturbation/expand.jl` | `expand_perturbation`, `_expand_pert` — convention lives here |
| `src/perturbation/linearize.jl` | `δricci_scalar` — delegates to expand_perturbation |
| `src/perturbation/metric_perturbation.jl` | `MetricPerturbation`, inverse metric expansion |
| `src/algebra/canonicalize.jl:434` | `normalize_field_positions` — NEW, works, may be useful later |
| `src/action/kernel_extraction.jl:302` | `build_FP_momentum_kernel` — CORRECT reference |
| `examples/15_perturbation_spectrum_crosscheck.jl` | crosscheck pipeline — may need factor fix |

## Changes Made This Session

| File | Change |
|------|--------|
| `src/algebra/canonicalize.jl` | Added `normalize_field_positions` (metric-insertion approach) |
| `src/TensorGR.jl` | Exported `normalize_field_positions` |
| `examples/15_perturbation_spectrum_crosscheck.jl` | Fixed √g coeff 1/4→1/8, added normalize call |

## Key Diagnostic Commands (copy-paste ready)

```julia
# Quick reproducer: compare δ²R spin-2 with FP
using TensorGR
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    @define_tensor k on=M4 rank=(0,1)
    mp = define_metric_perturbation!(reg, :g, :h; curved=false)
    set_vanishing!(reg, :Ric)

    δ2R = simplify(δricci_scalar(mp, 2); registry=reg, maxiter=200)
    Qf = to_fourier(δ2R)
    Qf = simplify(Qf; registry=reg)
    Qf = fix_dummy_positions(Qf)
    K = extract_kernel(Qf, :h; registry=reg)
    s2 = _eval_spin_scalar(spin_project(K, :spin2; registry=reg), 1.0)
    println("δ²R spin-2 = $s2 (should be 2.5 if convention matches FP)")

    K_FP = build_FP_momentum_kernel(reg)
    s2_FP = _eval_spin_scalar(spin_project(K_FP, :spin2; registry=reg), 1.0)
    println("FP spin-2  = $s2_FP (reference)")
end
```
