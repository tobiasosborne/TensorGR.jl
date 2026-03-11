# HANDOFF: Perturbation Engine ↔ FP Cross-Check — Session 17

## Status: ALL GREEN (tests unchanged, diagnostic-only session)

- **5851 tests pass**, 0 errors, 0 broken, 0 failed
- No src/ changes this session — only diagnostic example scripts (16-22)
- All prior code pushed to remote

## What Was Done This Session

### Confirmed: `spin_project` swap symmetry is CORRECT
- Individual bilinear terms produce identical spin projections regardless of which h is "left" vs "right"
- Tested via: (a) swapping h factors in original expression before `extract_kernel`, (b) manually constructing `KineticKernel` with swapped left/right
- The Barnes-Rivers projectors P^J are correctly symmetric: P^J_{μν,ρσ} = P^J_{ρσ,μν}
- **Previous session's "swap bug" was a test construction error** — example 21's "swapped" expression was mathematically different, not just reordered

### Confirmed: Pipeline works end-to-end for known-correct inputs
- Position-space FP Lagrangian → `to_fourier` → `extract_kernel` → `spin_project` gives exact match with `build_FP_momentum_kernel`
- spin-2=4.25, spin-1=0, spin-0s=-1.7, spin-0w=0 (at k²=1.7) — all correct

### Identified: `expand_derivatives` corrupts δR
- Engine δR (after `expand_derivatives` + Fourier) = **(3/2)×** the correct analytic result
- Analytic: δR = k^a k^b h_{ab} - k² h
- Engine with expand_derivatives: (3/2)(k^a k^b h_{ab} - k²h)
- **Root cause**: `expand_derivatives` applies Leibniz rule to g·∂h, producing spurious ∂g terms. On flat background ∂g=0, but expand_derivatives treats g as a variable
- **DO NOT use `expand_derivatives` on perturbation engine output** — it introduces wrong terms

### Identified: L₂ = δ²R + ½h·δR does NOT match FP
- L₂ - FP = 28 position-space terms = 13 Fourier kernel terms
- These 13 terms have **nonzero** spin projections: spin-2=-6.375, spin-1=-3.825, spin-0s=-2.55
- This means L₂ - FP is NOT purely total derivatives — there's a genuine content mismatch

### Key numerical results (at k²=1.7)

| Quantity | spin-2 | spin-1 | spin-0s | spin-0w |
|----------|--------|--------|---------|---------|
| FP (reference) | 4.25 | 0.0 | -1.7 | 0.0 |
| δ²R (engine) | -2.125 | -3.825 | -1.7 | 0.0 |
| ½h·δR | 0.0 | 0.0 | -2.55 | 0.0 |
| L₂ = δ²R + ½h·δR | -2.125 | -3.825 | -4.25 | 0.0 |
| X = -h^{ab}δRic_{ab} (analytic) | 4.25 | 0.0 | 0.85 | 0.0 |
| Y = δ²R - X (engine residual) | -6.375 | -3.825 | -2.55 | 0.0 |

### Decomposition insight
δ²R = X + Y where:
- X = -h^{ab}δRic_{ab} = known-correct analytic expression (3 terms in Fourier)
- Y = η^{ab}δ²Ric_{ab} = Christoffel-squared contributions from the perturbation engine

**X matches FP for spin-2** (4.25 = 4.25). The problem is entirely in Y.

## ROOT CAUSE (narrowed down)

The bug is in the **perturbation engine's computation of δ²R**, specifically the Y = η^{ab}δ²Ric_{ab} piece (Christoffel-squared terms). The Y piece should project to:
- spin-2: 0 (it doesn't contribute to spin-2 in the analytic result)
- spin-1: 0 (gauge invariance)

But the engine Y gives spin-2 = -6.375, spin-1 = -3.825.

### Most likely sub-causes (investigate in order)

1. **`to_fourier` mishandles nested TDeriv(∂, TSum(...))**
   - δ²R has 4 terms (of 22) with nested structure: `h × ∂(g·∂h + g·∂h - g·∂h)` and `g × ∂(g·g·h·∂h + ...)`
   - `to_fourier` processes TDeriv by: (a) recursively transform inner, (b) multiply by k
   - On flat background this SHOULD be correct (∂g=0 so ∂ passes through g)
   - But need to verify the recursive transform handles TDeriv(TSum) correctly
   - **Test**: expand_derivatives on δ²R BEFORE Fourier, compare. If different, to_fourier is losing information from nested TDeriv
   - **CAUTION**: expand_derivatives introduces spurious ∂g terms. Need to add a `set_constant!(reg, :g)` or manually filter ∂g=0 after expanding

2. **Perturbation engine δ²Ric has wrong Christoffel-squared terms**
   - The Cauchy product in `δricci_scalar` computes: [g^{ab}·Ric_{ab}]₂ = η^{ab}[Ric]₂ + [g^{-1}]₁·[Ric]₁
   - [Ric]₂ involves δΓ·δΓ (Palatini identity at second order)
   - These δΓ·δΓ terms have complex index structure and might have coefficient errors
   - **Test**: Build analytic Y = η^{ab}(δΓ^c_{ad}δΓ^d_{bc} - δΓ^c_{ab}δΓ^d_{cd}) in Fourier space and compare with engine Y

3. **Convention mismatch in `δricci_scalar`**
   - The MEMORY says δricci_scalar returns ε^n COEFFICIENT
   - But verify: is [R]₂ = ½R''(0) or R''(0)?
   - If it's R''(0) (without 1/2!), then the formula should be L₂ = ½δ²R + ½hδR, not δ²R + ½hδR
   - **Test**: pure-trace perturbation h=λη: R(η+ελη) = R(η(1+ελ)) = (1+ελ)^{-2}×0 = 0 on flat. So this test is trivial on flat. Try on a curved background instead.

## Recommended Next Session Strategy

### Priority 1: Verify to_fourier on nested TDeriv (~10 min)
```julia
# Take a single nested-TDeriv term from δ²R
term = δ2R.terms[1]  # -h × ∂(g·∂h + g·∂h - g·∂h)
# Fourier transform with and without manually expanding ∂ through the sum
result_nested = to_fourier(term)
# Manually: expand outer ∂ by Leibniz, drop ∂g terms, THEN Fourier
term_expanded = expand_derivatives(term)  # gives ∂g terms
# Filter: remove any factor containing ∂g (since ∂g=0 on flat)
# Then Fourier and compare
```

### Priority 2: Build analytic Y and compare (~15 min)
Build Y = η^{ab}δ²Ric_{ab} analytically in Fourier space using:
```
δΓ^c_{ab} = ½η^{cd}(k_a h_{bd} + k_b h_{ad} - k_d h_{ab})
δ²Ric_{ab} = δΓ^c_{ad}δΓ^d_{bc} - δΓ^c_{ab}δΓ^d_{cd}
Y = η^{ab}δ²Ric_{ab}
```
This is messy (4×4 = 16 cross-terms from the two δΓ products, minus another 16, contracted with η^{ab}) but mechanical. Compare spin projections of analytic Y with engine Y.

### Priority 3: If to_fourier is the culprit, fix it (~20 min)
If nested TDeriv handling is wrong, fix `_fourier_transform(::TDeriv)` to expand ∂(TSum) = Σ∂(term) before multiplying by k. Or add a pre-processing step that flattens nested TDeriv before Fourier.

### Priority 4: If perturbation engine is the culprit, fix δ²Ric (~30 min)
If the analytic Y differs from the engine Y, the bug is in `expand.jl`'s computation of second-order Ricci. Check the Cauchy product structure and δΓ·δΓ assembly.

## Key Files

| File | Role |
|------|------|
| `src/perturbation/expand.jl` | `δricci_scalar`, `δricci`, `δchristoffel` — likely bug location |
| `src/svt/fourier.jl` | `to_fourier`, `_fourier_transform` — nested TDeriv handling |
| `src/action/kernel_extraction.jl` | KineticKernel, spin_project (CONFIRMED CORRECT) |
| `src/action/spin_projectors.jl` | Barnes-Rivers projectors (CONFIRMED CORRECT) |
| `src/algebra/derivatives.jl` | `expand_derivatives` — DO NOT USE on pert engine output |
| `examples/16_diagnostic_flat_pipeline.jl` | Diagnostic: derivative distribution check |
| `examples/17_pipeline_isolation_test.jl` | Diagnostic: FP position-space through pipeline (PASSES) |
| `examples/18_expand_derivs_fix.jl` | Diagnostic: expand_derivatives makes things worse |
| `examples/19_kernel_debug.jl` | Diagnostic: all Fourier terms are bilinear |
| `examples/20_kernel_comparison.jl` | Diagnostic: pert engine vs analytic X decomposition |
| `examples/21_total_deriv_test.jl` | Diagnostic: swap test (FLAWED test construction) |
| `examples/22_swap_symmetry_test.jl` | Diagnostic: proper swap test (PASSES) |

## Critical Gotchas (accumulated)

- `spin_project` swap symmetry is FINE — individual terms project identically under left↔right swap
- `expand_derivatives` on pert engine output produces spurious ∂g terms (3/2× factor on δR) — DO NOT USE
- `to_fourier` nested TDeriv: `∂(TSum)` → `k × fourier(TSum)` — correct in principle but needs verification
- `spin_project` returns **Tr(K·P^J)**, NOT f_J. Divide by {5,3,1,1}
- `δricci_scalar(mp, n)` returns ε^n COEFFICIENT (Cauchy product)
- The analytic X = -h^{ab}δRic_{ab} correctly gives spin-2=4.25 matching FP — the problem is ONLY in Y = δΓ·δΓ terms
- L₂ - FP has 13 Fourier kernel terms with nonzero spin content — NOT total derivatives

## Open Beads Issues (6-deriv related)

| Issue | Status | Description |
|-------|--------|-------------|
| TGR-c6su | OPEN P2 | Step 2.1: SVT decomposition of δ²S (flat) |
| TGR-pr04 | OPEN P2 | Step 2.2: SVT QuadraticForms + propagators |
| TGR-tztc | OPEN P2 | Step 2.3: Cross-check Path A vs Path B |
| TGR-j6r9 | OPEN P2 | Step 5: Tests + benchmark (blocked by tztc) |
| TGR-af4a | OPEN P2 | Step 6: Example script + module integration |

## Diagnostic Example Files to Clean Up

Examples 16-22 are all diagnostic scripts from sessions 16-17. They can be deleted once the cross-check is resolved. None are referenced by tests or benchmarks.
