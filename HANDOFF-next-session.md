# HANDOFF: Session 31 — Fourier ∂(bilinear) fix applied, partial spin-2 fix

## Status: Fourier pipeline partially fixed, spin-2 needs further work

- **All 7146 tests pass**: fourier.jl change only, no contraction.jl changes
- **TGR-dp3 remains open**: spin-2 coefficient still incorrect

## What Was Done

### Fix Applied: `src/svt/fourier.jl`

Two changes to `_fourier_transform(TDeriv)`:

1. **TSum distribution**: When a TDeriv wraps a TSum, distribute the derivative
   over the sum (`∂(A+B) → ∂A + ∂B`) before processing. This exposes individual
   product terms for the bilinear check.

2. **Bilinear product dropping**: When a TDeriv wraps a TProduct that is bilinear
   in field tensors (≥2 non-metric/delta factors), return ZERO. This correctly
   handles the physics: in quadratic forms under ∫d⁴x, derivatives of bilinear
   field products vanish because the two fields carry opposite momenta (k and -k).

A helper `_count_field_factors(p::TProduct)` counts factors that are "field-like"
(TDeriv or non-metric/non-delta Tensor), excluding background metrics.

### Results

Spin projections for δ²R at k²=1:

| Component | Before Fix | After Fix | Expected (δ²R) |
|-----------|-----------|-----------|-----------------|
| spin-2    | 6.25      | 3.75      | 2.50            |
| spin-1    | 0.0       | 0.0       | 0.0             |
| spin-0s   | 0.5       | 0.0       | ???             |
| spin-0w   | 0.0       | 0.0       | 0.0             |

spin-1 and spin-0w are correct. spin-2 improved but not resolved.
spin-0s changed from 0.5 to 0.0 — needs verification against known results.

### What Was Attempted But Reverted

1. **Bug 2 fix (contraction.jl)**: Extended `_try_metric_contraction` to handle
   TDeriv factors (Case A: raise/lower derivative index; Case B: push metric
   inside derivative). This correctly contracted metrics paired with TDeriv
   factors in the ΓΓ terms. However, it changed the expression structure during
   simplify, cascading into different term counts and spin projections. All
   contraction.jl changes were reverted.

2. **Factor recursion in `contract_metrics(TProduct)`**: Adding recursion into
   TDeriv factors broke the simplify convergence, causing spin-0s to become 0.0
   (wrong) and spin-2 to become 3.75 (same as no fix).

## Root Cause Analysis (Refined)

The original handoff identified two bugs. The fourier.jl fix addresses Bug 1.
Bug 2 remains unresolved.

### Bug 1 (FIXED): `to_fourier` mishandles `∂(bilinear product)`

`_fourier_transform(TDeriv)` now:
- Distributes derivatives over sums before processing
- Drops derivatives wrapping bilinear field products (≥2 field factors)

This correctly handles the physics of quadratic forms under ∫d⁴x.

### Bug 2 (UNRESOLVED): Uncontracted metrics paired with TDeriv factors

In the ΓΓ terms of δ²Ric, products like `g^{af} × ∂_c(h_{ef}) × g^{eg} × ∂_d(h_{gb})`
have metrics paired with TDeriv factors. `_try_metric_contraction` only checks
`fj isa Tensor` partners, skipping TDeriv. The metrics remain uncontracted through
the Fourier pipeline.

**Why the fix was hard**: Extending `_try_metric_contraction` to handle TDeriv
partners (pushing metrics inside derivatives) works algebraically but changes the
expression structure during simplify, causing cascading differences in canonicalization
and term collection. This led to wrong spin projections.

**Proposed path forward**:
- Option A: Add a dedicated `contract_metrics_deep(expr)` function that runs ONCE
  as a preprocessing step before `to_fourier`, rather than modifying the general
  `contract_metrics` used in the simplify loop. This avoids convergence issues.
- Option B: Fix the kernel extraction / spin projection pipeline to handle
  uncontracted `g` tensors correctly (treating them as η^{ab} on flat background).
- Option C: Add metric contraction to TDeriv partners only for partial derivatives
  (where ∂g=0 is guaranteed), with careful handling to preserve simplify convergence.

## Key Diagnostic Script

```julia
using TensorGR
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp = define_metric_perturbation!(reg, :g, :h)
    set_vanishing!(reg, :Ric)
    set_vanishing!(reg, :RicScalar)
    set_vanishing!(reg, :Riem)

    δ2R = simplify(δricci_scalar(mp, 2); registry=reg)
    f = to_fourier(δ2R); f = simplify(f; registry=reg); f = fix_dummy_positions(f)
    K = extract_kernel(f, :h; registry=reg)
    println("spin-2  = $(_eval_spin_scalar(spin_project(K, :spin2; registry=reg), 1.0))")
    println("spin-0s = $(_eval_spin_scalar(spin_project(K, :spin0s; registry=reg), 1.0))")
    println("spin-1  = $(_eval_spin_scalar(spin_project(K, :spin1; registry=reg), 1.0))")
    println("spin-0w = $(_eval_spin_scalar(spin_project(K, :spin0w; registry=reg), 1.0))")
end
```

## Files Modified

| File | Change |
|------|--------|
| `src/svt/fourier.jl` | Added `_count_field_factors`, TSum distribution, bilinear dropping |

## Changes This Session

- Added `_count_field_factors` helper to distinguish field tensors from background metrics
- Added TSum distribution in `_fourier_transform(TDeriv)` (∂(A+B) → ∂A + ∂B)
- Added bilinear product dropping (∂(h₁h₂) → 0 in quadratic forms)
- Attempted and reverted Bug 2 fix to `contract_metrics` (TDeriv partner handling)
- Attempted and reverted factor recursion in `contract_metrics(TProduct)`
