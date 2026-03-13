# HANDOFF: Session 29 — Deep Diagnosis of δ²R A-type Coefficient Bug

## Status: Bug localized to η^{ab}δ²Ric_{ab}, no code changes

- **All 7267 tests pass**: no changes to source
- **TGR-dp3 remains open**: root cause further narrowed

## What Was Done This Session

### 1. Four-agent parallel research on perturbation conventions

- **Subagent A**: Confirmed perturbation engine uses **Taylor coefficient convention** (no factorial divisors, no binomial coefficients). `_expand_pert(TProduct)` uses `all_compositions` without multinomial weights. Internally consistent.
- **Subagent B**: Verified `δinverse_metric` at order 2 gives `h^{ac}h_c^b` (correct for Taylor convention). Recursion works for order ≤ 2.
- **Subagent C**: Analyzed the mathematical structure of δ²R and confirmed the A-type excess.
- **Subagent D**: Verified example 15 formula Q = δ²R + √g_correction is structurally correct. FP kernel is standard Fierz-Pauli.

### 2. Decomposition of δ²R into Leibniz contributions

On flat background (curved=false, Ric vanishing), at k²=1:

```
δ²R = η^{ab}δ²Ric_{ab} + (-h^{ab})δ¹Ric_{ab}

Contribution breakdown (spin-2 projections at k²=1):
  (-h^{ab})δ¹Ric_{ab}  spin-2 = 2.5   ← CORRECT (matches FP alone!)
  η^{ab}δ²Ric_{ab}     spin-2 = 3.75  ← WRONG (should be 0)
  Total δ²R             spin-2 = 6.25  ← WRONG (should be 2.5)

Further decomposition of η^{ab}δ²Ric_{ab}:
  ∂[Γ₂] terms           spin-2 = 2.5   (A-coeff = 1/2)
  Γ₁×Γ₁ terms           spin-2 = 1.25  (A-coeff = 1/4)
  Total                  spin-2 = 3.75  (A-coeff = 3/4, should be 0)
```

### 3. Full kernel comparison (δ²R vs FP)

```
Type     δ²R     FP      Ratio    Correct δ²R*
A (k²hh) 5/4    1/2     5/2      1/2
B (kkhh) -5/2   -1      5/2      -1
C (kkhtr) 3/2    1      3/2      1/2
D (k²tr²) -1/4  -1/2    1/2     0

* Correct δ²R = L_FP - (1/2)h·δR (on flat, √g correction)
  On TT modes: correct δ²R = L_FP (since δR=0 on TT)
```

### 4. TT-mode analysis proves engine bug

On transverse-traceless (TT) perturbations:
- δR = 0 (linearized Ricci scalar vanishes on TT)
- √g correction = 0 (involves traces)
- Therefore: Q = δ²R on TT = L_FP on TT = (1/2)k²|h|²
- But code gives: δ²R on TT = (5/4)k²|h|² (WRONG, 5/2× too large)

## Root Cause Analysis

The bug is in the **second-order Ricci tensor** δ²Ric_{ab}. When traced with η^{ab}, it gives nonzero spin-2 projection (3.75) when mathematically it must give zero (because all spin-2 content of δ²R comes from the cross term).

### Manual calculation vs code for η^{ab}δ²Ric (TT modes)

Manual calculation of A-type coefficients on TT:
- Step 1: η^{ab}∂_c[Γ^c_{ba}]₂ → A-type = 0 (all terms involve ∂^α h_{fα}=0)
- Step 2: -η^{ab}∂_b[Γ^c_{ca}]₂ → A-type = 1 (two contributions of 1/2 each)
- Step 3: Γ₁²  → A-type = 1/4 (three terms: -1/4, +1/4, +1/4)
- Manual total: 5/4

Code gives: 3/4 for η^{ab}δ²Ric A-type.
Manual gives: 5/4.
Correct answer: 0.

Both disagree! This suggests:
1. The code's simplification pipeline loses some terms (giving 3/4 instead of 5/4)
2. There should be additional cancellations not captured (bringing 5/4 down to 0)

### Likely suspects

**Hypothesis A: The simplification pipeline fails to cancel terms in δ²Ric.**
The 20-term δ²Ric_{ab} may have terms that should cancel but don't, due to:
- `canonicalize` not detecting equivalent terms with different dummy arrangements
- `contract_metrics` missing contractions in nested derivative products
- `expand_products` not fully distributing derivatives through products

**Hypothesis B: The Leibniz expansion of ∂_c(h^{cf} × ∂h) in δ²Christoffel is incorrect.**
When δriemann wraps δ²Christoffel in a TDeriv, the outer derivative must distribute through the product. If `expand_derivatives` doesn't fully distribute, terms are missing.

**Hypothesis C: Index clash or dummy collision in δ²Riemann.**
The `fresh_index` / `ensure_no_dummy_clash` logic might be incorrect for the deeply nested expressions in δ²Riemann, leading to terms being dropped or miscombined.

## Recommended Next Session: Focused Bug Hunt

### Phase 1: Check expand_derivatives on δ²Christoffel (30 min)

```julia
# Compute ∂_c[Γ₂] and check if expand_derivatives distributes correctly
δ2Γ = δchristoffel(mp, up(:c), down(:b), down(:a), 2)
wrapped = TDeriv(down(:c), δ2Γ, :partial)
expanded = expand_derivatives(wrapped)
# Count terms: should have (∂h)(∂h) AND h(∂²h) terms
# If only h(∂²h), the Leibniz distribution is buggy
```

### Phase 2: Numeric verification (30 min)

Compute R(ε) = R[η + ε h] using the COMPONENT engine (CTensor) for a specific TT polarization, extract ε² coefficient, compare with the abstract engine's δ²R.

```julia
# Use symbolic_ricci_scalar with g = η + ε*h for specific h_{12}=h_{21}=ε
# Extract coefficient of ε² → this is [R]₂ numerically
# Compare with abstract δ²R evaluated on same h
```

### Phase 3: Fix and verify

Once the specific bug is found:
1. Fix the engine (likely in expand_derivatives, canonicalize, or contract_metrics)
2. Re-run the diagnostic: δ²R spin-2 should give 2.5
3. Run example 15: FP check should pass
4. Run full test suite
5. Close TGR-dp3

## Key Files

| File | Role |
|------|------|
| `src/perturbation/expand.jl:60-157` | `δchristoffel` — builds [Γ]₂ as product h^{cf}×(∂h bracket) |
| `src/perturbation/expand.jl:177-261` | `δriemann` — wraps [Γ]₂ in TDeriv, adds Γ₁² |
| `src/perturbation/expand.jl:278-298` | `δricci` — traces Riemann |
| `src/perturbation/expand.jl:312-359` | `δricci_scalar` — Leibniz over (g^{ab}, Ric) |
| `src/algebra/simplify.jl` | `simplify` pipeline (expand_products, canonicalize, etc.) |
| `src/algebra/derivatives.jl` | `expand_derivatives` — distributes ∂ through products |
| `src/action/kernel_extraction.jl:302-308` | `build_FP_momentum_kernel` — reference (VERIFIED CORRECT) |
| `examples/15_perturbation_spectrum_crosscheck.jl` | crosscheck pipeline |

## Key Diagnostic Commands

```julia
# Quick reproducer: decompose δ²R
using TensorGR
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    @define_tensor k on=M4 rank=(0,1)
    mp = define_metric_perturbation!(reg, :g, :h; curved=false)
    set_vanishing!(reg, :Ric)

    # Cross term alone = correct FP!
    δ1Ric = simplify(δricci(mp, down(:a), down(:b), 1); registry=reg, maxiter=200)
    cross = tproduct(-1//1, TensorExpr[Tensor(:h, [up(:c), up(:d)]),
            simplify(δricci(mp, down(:c), down(:d), 1); registry=reg, maxiter=200)])
    cross = simplify(cross; registry=reg, maxiter=200)
    Cf = to_fourier(cross)
    Cf = simplify(Cf; registry=reg)
    Cf = fix_dummy_positions(Cf)
    KC = extract_kernel(Cf, :h; registry=reg)
    println("Cross term spin-2 = $(_eval_spin_scalar(spin_project(KC, :spin2; registry=reg), 1.0))")
    # Should print 2.5 (CORRECT)

    # η^{ab}δ²Ric is the buggy part
    δ2Ric = simplify(δricci(mp, down(:a), down(:b), 2); registry=reg, maxiter=200)
    traced = Tensor(:g, [up(:a), up(:b)]) * δ2Ric
    traced = simplify(traced; registry=reg, maxiter=200)
    Tf = to_fourier(traced)
    Tf = simplify(Tf; registry=reg)
    Tf = fix_dummy_positions(Tf)
    KT = extract_kernel(Tf, :h; registry=reg)
    println("η^{ab}δ²Ric spin-2 = $(_eval_spin_scalar(spin_project(KT, :spin2; registry=reg), 1.0))")
    # Prints 3.75, should be 0.0
end
```

## Changes Made This Session

None. Pure research/diagnosis session.
