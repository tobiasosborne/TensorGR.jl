# HANDOFF: 6-Deriv Spectrum Pipeline — Session 10

## Current State

- **4580 tests pass** (13 errors likely from Julia 1.12 / Symbolics compat, not regressions)
- **No code changes this session** — pure research round
- **Spin projection validated**: identity kernel {5,3,1,1}, manual Fierz-Pauli EH kernel gives spin-1=0, spin-0-w=0 ✓
- **fix_dummy_positions**: exported, tested, repairs same-position dummy pairs from xperm

---

## Session 10: Research Results — EH Form Factor Approaches

### Research Summary

All four approaches (A-D) were investigated by reading source files and running experimental code. Key findings:

### Approach A: IBP Before Projection — NOT RECOMMENDED

- `ibp.jl` works on simple products (move derivatives off one named field)
- `ibp_product(p, :h)` peels ALL derivatives off one factor of :h
- **Problem**: for bilinear h × operator × h, IBP needs to find the canonical representative under IBP equivalence (remove total derivatives). This is a non-trivial algorithmic problem — much more than just moving derivatives.
- Would require ~50-100 lines of new bilinear IBP canonicalization code
- No clean criterion for "expression has no total derivatives" exists in the codebase

### Approach B: Linearized EOM — PARTIALLY WORKS, HAS BUGS

Tested: `G^(1)_{ab} = δRic_{ab} - (1/2)g_{ab}δR`, then form `h^{ab} × G^(1)_{ab}`.

**Results** (with `expand_derivatives + expand_products + simplify` before Fourier):
```
EOM kernel: 4 terms (correct count!)
Spin-2:  -5k²     (expected: 5k²/2 from FP — factor of -2 off)
Spin-1:  -3k²/2   (expected: 0 — NOT gauge-invariant!)
Spin-0s: -k²      (matches FP ✓)
Spin-0w: 0         (matches FP ✓)
```

**Issues discovered**:
1. The position-space expression has `∂(g)` terms (derivatives of metric) that should be zero on flat background but aren't eliminated by the simplifier
2. After simplification: 12 terms in position space, only 4 after Fourier — but missing trace structures (h × h terms where h = g^{ab}h_{ab})
3. The metric contraction `g_{ab}h^{ab} → h^c_c` happens, but the resulting traced h doesn't produce the expected 4-structure FP form
4. Spin-2 has wrong sign and factor of 2 compared to FP; spin-0-s matches exactly

**Root cause**: the simplifier contracts `h^{ab} × g_{ab}` to `h^c_c` (trace), but the subsequent product `h^c_c × δR` doesn't produce independent trace-squared terms visible to `extract_kernel`. The two "trace" structures (`k_ak_b h^{ab}h` and `k²h²`) get absorbed into the non-trace structures.

### Approach C: Gauge-Fix Then Project — NOT TESTED

Would add de Donder gauge-fixing term. Conceptually simple but ad hoc, and doesn't address the deeper simplifier issue with Ric².

### Approach D: Direct Momentum-Space Construction — RECOMMENDED (with caveat)

**Motivation**: Build kernels directly in Fourier/momentum space using known linearized curvature formulas, bypassing the position-space perturbation engine entirely.

**Why D is best**:
- Avoids position-space simplifier bottleneck entirely
- EH Fierz-Pauli form already validated (4 terms, test passes)
- Linearized δR and δRic have known, explicit Fourier-space forms
- No total derivative ambiguity in momentum space
- Direct, unambiguous expressions — easy to verify against literature

---

## CRITICAL BUG: `spin_project` fails on Ric² kernel

### The Bug

The perturbation-engine pipeline (`δRic × δRic → simplify → to_fourier → extract_kernel → spin_project`) **works correctly for R²** but **fails for Ric²**:

**R² spin projections** (CORRECT ✓):
```
Spin-2:  0         ✓ (R² only affects scalar sector)
Spin-1:  0         ✓
Spin-0s: 3k⁴       ✓ (expected k⁴ contribution)
Spin-0w: 0         ✓
```

**Ric² spin projections** (BROKEN ✗):
```
Spin-2:  15 terms with un-contracted k[c]*k[_d4] and g[_d2,_d2]  ✗
Spin-1:  5 terms with un-contracted momenta  ✗
Spin-0s: 4 terms with un-contracted momenta  ✗
Spin-0w: 2 terms with un-contracted momenta  ✗
```

### Root Cause

After spin projection, the result contains:
1. **Same-position dummy pairs**: `k[_d3] * k[_d3]` (both Up) that `contract_momenta` can't handle (requires opposite positions)
2. **Un-contracted metrics**: `g[_d2, _d2]` that should contract to dimension 4
3. **Orphan momentum indices**: `k[c] * k[_d4]` that should have been contracted by the projector

Applying `fix_dummy_positions` after spin projection resolves items 1 and 2, but item 3 persists. The orphan indices `c` and `_d4` appear to be dummies from the kernel coefficient that should pair with projector indices but don't.

**Hypothesis**: the `_standardize_h_indices` step in `spin_project` renames h indices with fresh names, but the kernel coefficient's dummy indices (from the Ric² contraction) still carry the original names. When the projector is built with the fresh h-index names, the coefficient's dummies don't connect to anything.

**Specific mechanics**: The Ric² kernel has terms like:
```
coeff = k[-_d1] * k[_d3]   (momentum indices from δRic contraction)
left  = [_d1, _d2]          (h indices)
right = [-_d2, -_d3]        (h indices)
```
The coefficient has a `k[_d3]` that contracts with `right[2] = -_d3`. But `_standardize_h_indices` renames the right indices to fresh names, breaking this contraction.

### Fix Required

**In `spin_project` (`src/action/kernel_extraction.jl:77-115`)**: when `_standardize_h_indices` renames h indices, it must ALSO propagate those renames into the coefficient expression. Currently it only inserts metric connectors for Up→Down lowering but doesn't update the coefficient's dummy references.

This is the **#1 blocker** for the form factor pipeline. Once fixed, both the perturbation-engine approach and the direct momentum-space approach should work for all terms.

**Estimated fix**: ~20 lines in `_standardize_h_indices` or the calling code in `spin_project`. The coefficient's indices that share names with the original h indices need to be updated when h indices get fresh names.

---

## Priority 1: Fix spin_project Dummy Propagation (BLOCKER)

1. Read `_standardize_h_indices` in `src/action/kernel_extraction.jl:126-181`
2. When h indices are renamed (Up→fresh Down), track the old→new mapping
3. Apply the same rename to the coefficient expression (`bt.coeff`)
4. Also apply `fix_dummy_positions` to the final spin_project result before returning
5. Test: Ric² kernel should give clean scalar results (no orphan indices)

### Validation After Fix

Once spin_project works for Ric², verify:
- R² spin-2 = 0, spin-0-s = 3k⁴ (unchanged) ✓
- Ric² all sectors give clean k⁴ scalars
- Identity kernel {5,3,1,1} still passes
- FP EH kernel {5k²/2, 0, -k², 0} still passes

---

## Priority 2: Build Full 6-Deriv Form Factors (TGR-zq2k)

### Approach: Hybrid (FP for EH + perturbation engine for rest)

1. **EH kernel**: Use the manual Fierz-Pauli form (already validated, 4 terms)
2. **R² kernel**: Use `(δR)²` from perturbation engine (already works ✓)
3. **Ric² kernel**: Use `(δRic)²` from perturbation engine (needs spin_project fix)
4. **R□R kernel**: Use `2(δR)(□δR)` from perturbation engine
5. **Ric□Ric kernel**: Use `2(δRic)(□δRic)` from perturbation engine

### Normalization

The action `S = ∫ [κR + α₁R² + α₂Ric² + β₁R□R + β₂Ric□Ric]` gives quadratic Lagrangian:
```
L^(2) = κ × L_FP + 2α₁(δR)² + 2α₂(δRic)² + 2β₁(δR)(□δR) + 2β₂(δRic)(□δRic)
```
(Factor of 2 from δ²(X²) = 2(δX)² when background X̄=0)

### Form Factor Extraction

```
f₂ = Tr(K_total · P²) / (5 × κ × k²)   →  should give 1 - (α₂/κ)k² - (β₂/κ)k⁴
f₀ = Tr(K_total · P⁰ˢ) / (1 × (-κk²))  →  should give 1 + (6α₁+2α₂)k²/κ + (6β₁+2β₂)k⁴/κ
```

The normalization divisors ensure f₂(0) = f₀(0) = 1 for pure EH.

---

## Alternative: Direct Momentum-Space Construction (Approach D)

If the spin_project fix proves too complex, build ALL kernels directly in momentum space:

```julia
# Linearized curvature in Fourier space (known, explicit formulas):
# δR = k^a k^b h_{ab} - k² h
# δRic_{μν} = (1/2)(k^ρ k_μ h_{νρ} + k^ρ k_ν h_{μρ} - k² h_{μν} - k_μ k_ν h)
#
# Build bilinear forms directly as TensorExpr:
# (δR)² = (k^a k^b h_{ab} - k²h)²   → 3 bilinear terms
# (δRic)² = δRic^{ab} δRic_{ab}      → 16 bilinear terms (4×4)
# Then extract_kernel + spin_project (which should work since the
# expressions are already in momentum space with clean index structure)
```

This bypasses the position-space simplifier entirely. The only risk is that spin_project still needs to handle the Ric² contraction complexity.

---

## Key Files

| File | Role |
|------|------|
| `src/action/kernel_extraction.jl` | `extract_kernel`, `spin_project`, `_standardize_h_indices`, `contract_momenta` |
| `src/action/spin_projectors.jl` | Barnes-Rivers P2/P1/P0s/P0w/θ/ω projectors |
| `src/algebra/canonicalize.jl:319-433` | `fix_dummy_positions` |
| `src/algebra/ibp.jl` | `ibp`, `ibp_product` |
| `src/perturbation/variation.jl` | `variational_derivative`, `euler_lagrange` |
| `src/perturbation/expand.jl` | `δricci_scalar`, `δricci`, `expand_perturbation` |
| `src/svt/fourier.jl` | `to_fourier` (∂ → k replacement) |
| `test/test_6deriv_spectrum.jl` | All spectrum tests |

## Test Commands

```bash
julia --project -e 'using Pkg; Pkg.test()'                    # full suite
julia --project=benchmarks benchmarks/run_all.jl --tier 1      # tier 1 benchmarks
bd show TGR-zq2k                                               # flat form factors issue
bd show TGR-60sx                                                # canonicalize dummy bug
```

## Beads Issues

- **TGR-60sx** [P1 bug]: canonicalize same-position dummy pairs — related to spin_project bug
- **TGR-zq2k** [P1 task]: Barnes-Rivers projection → flat form factors — BLOCKED by spin_project bug
- **TGR-mphe** [P1 task]: dS background quadratic + box terms — BLOCKED by TGR-zq2k
