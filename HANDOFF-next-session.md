# HANDOFF: Session 30 — Root Cause Found: to_fourier + contract_metrics bugs

## Status: Root cause identified, no code changes, ready for fix

- **All 7267 tests pass**: no changes to source
- **TGR-dp3 remains open**: root cause fully diagnosed

## Root Cause (Two Interacting Bugs)

### Bug 1: `to_fourier` mishandles `∂(product)` for quadratic forms

`_fourier_transform(d::TDeriv)` in `src/svt/fourier.jl:55-76` replaces:
```
∂_a(arg) → k_a × fourier(arg)
```

When `arg` is a **product** of two h-fields (like `h^{cd} × ∂_b h_{ad}` from δ²Christoffel),
this treats the derivative as acting on the whole product as a single entity. But in a
quadratic action `S₂ = ∫dx h K h`, the derivative distributes via Leibniz over both h-fields,
and each h carries **different momentum** (k and -k). The correct physics gives:

- `∫dx ∂_c(h₁ × ∂_b h₂)` = 0 (total derivative, boundary term vanishes)
- Code gives: `k_c × k_b × h₁ × h₂` (nonzero — **wrong**)

This only affects **second-order** perturbations where `∂[Γ₂]` terms wrap products.
First-order terms work correctly because all derivatives act on a single h-field.

### Bug 2: `contract_metrics` can't contract metrics with TDeriv partners

`_try_metric_contraction` in `src/algebra/contraction.jl:124-183` only checks
`fj isa Tensor` partners, skipping TDeriv factors. So `g^{ab} × ∂_b(h_{cd})` where
the dummy `b` appears in the metric AND the derivative index remains uncontracted.

After simplify, 18 of 20 δ²Ric terms still have `g^{_d1,_d2} × g^{_d3,_d4}` factors
because the metric's dummies only appear in TDeriv factors (derivative indices or
indices inside derivatives).

**Note**: Inside each TDeriv's arg, the g's DO share dummies with bare Tensor h-factors,
so `contract_metrics(TDeriv)` should recurse and contract them. Need to verify why
this doesn't happen — possible that the simplify loop structure or TDeriv wrapping
prevents the contraction from reaching the inner products.

## Quantitative Evidence

```
δ²Ric after simplify: 20 terms
  18 "expanded" terms: (∂h)(∂h) products with uncontracted g's  → spin-2 = 1.25
   2 "unexpanded" terms: ∂(h × ∂h) with outer ∂ not distributed → spin-2 = 2.50
  Total η^{ab}δ²Ric spin-2 = 3.75 (should be 0)

Cross term -h^{ab}δ¹Ric_{ab} spin-2 = 2.50 (CORRECT, = FP)
Full δ²R spin-2 = 6.25 (should be 2.50)

Naive Leibniz fix (expand_derivatives before to_fourier):
  η^{ab}×(unexpanded with Leibniz) spin-2 = 10.0 (WORSE — uniform-k convention
  double-counts total-derivative contributions instead of canceling them)
```

## The Physics

In a quadratic form `S₂ = ∫dx Q(h, ∂h, ∂²h)`:
- Each h(x) decomposes as `∫dk h(k) e^{ikx}`
- The two h-fields carry momenta k₁ and k₂ = -k₁ (from ∫dx → δ(k₁+k₂))
- A derivative ∂_a acting on h(k) gives `ik_a`, acting on h(-k) gives `-ik_a`
- Total derivatives `∂_c J^c` give `i(k₁+k₂)_c J^c = 0` — they vanish!
- The "uniform k" convention (all ∂ → k) is only correct when each ∂ acts on ONE h-factor

The code's uniform-k convention works for `(∂h)(∂h)` products (first order Γ₁² terms)
but fails for `∂(h×∂h)` products (second order ∂[Γ₂] terms).

## Recommended Fix Strategy

### Option A: Fix at the source (perturbation engine) — RECOMMENDED

Modify `δriemann` in `src/perturbation/expand.jl:177-261` to expand the Leibniz rule
on the ∂[Γ₂] terms immediately, rather than wrapping them in TDeriv:

```julia
# Instead of:
push!(terms, TDeriv(c, δnΓ_adb, _rcovd))

# Do:
wrapped = TDeriv(c, δnΓ_adb, _rcovd)
expanded = expand_derivatives(wrapped)
push!(terms, expanded)  # or push each term of the expanded TSum
```

But this introduces `∂(g)` terms from the uncontracted g's in δ²Γ. Need to either:
1. Simplify δ²Γ first (contract g's), then wrap in TDeriv, then expand Leibniz
2. Or add a rule that `∂(metric) = 0` on flat background

### Option B: Fix to_fourier for quadratic forms

Add a Leibniz-aware Fourier transform that handles TDeriv(TProduct):

```julia
function _fourier_transform(d::TDeriv, conv, cn)
    if d.arg isa TProduct
        # Apply Fourier-space Leibniz: sum over which factor gets the k
        # BUT: need to handle the momentum sign correctly for quadratic forms
        # (k for "right" h, -k for "left" h)
    end
    ...
end
```

This is more complex because it requires knowing which h is "left" vs "right" in
the quadratic form — information that to_fourier doesn't currently have.

### Option C: Integration by parts before Fourier transform

Add an IBP step that moves all derivatives to one side of the quadratic form before
Fourier transforming. This would:
1. Turn `∂_c(h × ∂_b h)` into `(∂_c h)(∂_b h) + h(∂_c ∂_b h)` [Leibniz]
2. IBP the `h(∂_c ∂_b h)` term: → `-(∂_c h)(∂_b h)` + boundary [IBP]
3. Total: `(∂_c h)(∂_b h) - (∂_c h)(∂_b h)` = 0 (total derivative cancels)

After IBP, all remaining terms have `(∂h)(∂h)` structure where to_fourier works correctly.

### Option D: Drop total-derivative terms before Fourier

Detect terms of the form `∂_c(...)` where c is a contracted dummy and remove them,
since they're total divergences that vanish under ∫dx.

**This is the simplest fix** and directly targets the 2 problematic terms.

## Files to Modify

| File | What to change |
|------|---------------|
| `src/svt/fourier.jl:55-76` | Fix `_fourier_transform(TDeriv)` for product args |
| `src/perturbation/expand.jl:196-209` | Optionally expand Leibniz at source in δriemann |
| `src/algebra/contraction.jl:124-183` | Optionally extend metric contraction to TDeriv partners |

## Key Diagnostic Script

```julia
using TensorGR
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=TensorGR.Symmetric(1,2)
    @define_tensor k on=M4 rank=(0,1)
    mp = define_metric_perturbation!(reg, :g, :h; curved=false)
    set_vanishing!(reg, :Ric)

    δ2Ric = simplify(δricci(mp, down(:a), down(:b), 2); registry=reg, maxiter=200)
    # Should have 20 terms: 18 expanded + 2 unexpanded (∂(product))

    # Cross term (CORRECT reference):
    δ1Ric = simplify(δricci(mp, down(:c), down(:d), 1); registry=reg, maxiter=200)
    cross = simplify(tproduct(-1//1, TensorExpr[Tensor(:h, [up(:c), up(:d)]), δ1Ric]); registry=reg)
    Cf = to_fourier(cross); Cf = simplify(Cf; registry=reg); Cf = fix_dummy_positions(Cf)
    KC = extract_kernel(Cf, :h; registry=reg)
    println("Cross spin-2 = $(_eval_spin_scalar(spin_project(KC, :spin2; registry=reg), 1.0))")
    # Should print 2.5

    # η^{ab}δ²Ric (BUGGY):
    traced = simplify(Tensor(:g, [up(:a), up(:b)]) * δ2Ric; registry=reg, maxiter=200)
    Tf = to_fourier(traced); Tf = simplify(Tf; registry=reg); Tf = fix_dummy_positions(Tf)
    KT = extract_kernel(Tf, :h; registry=reg)
    println("η^{ab}δ²Ric spin-2 = $(_eval_spin_scalar(spin_project(KT, :spin2; registry=reg), 1.0))")
    # Prints 3.75, should be 0.0
end
```

## Changes Made This Session

None. Pure diagnosis session.
