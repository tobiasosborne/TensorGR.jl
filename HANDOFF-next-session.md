# HANDOFF: 6-Deriv Spectrum Pipeline — Session 4

## What Was Done (Sessions 1-3)

### Session 1
- **TGR-ncdr** ✅ — Kernel extraction (`extract_kernel`, `spin_project`, `contract_momenta`)
- **TGR-0i4m** ✅ — `sym_inv` 3×3

### Session 2
- **TGR-ud97** ✅ — Numerical Lichnerowicz verification
- **TGR-w7jq** ✅ — δ²S term counts on flat (8/4/4/16/18)
- Diagnosed 3 simplifier gaps

### Session 3
- **TGR-mr8p** ✅ — Changed default metric `:η` → `:g` in Barnes-Rivers projectors + `spin_project`
- **TGR-5wit** ✅ — Added `_simplify_scalars` step in `_simplify_one_pass` (CAS hook wiring)
- **TGR-uy04** ✅ — Removed `Symbol` type on `k_sq` params; `_sym_div` for `1/k_sq`; `Num` dispatch
- Fixed `TScalar.==` to use `isequal` (avoids symbolic boolean from `Num == Num`)
- Added `contract_momenta` loop inside `spin_project` (simplify ↔ contract_momenta iteration)
- **Verified**: hand-built EH Lichnerowicz kernel → spin-2=-5k²/2, spin-1=0, spin-0s=k², spin-0w=0
- All 4585 tests pass

---

## Dependency Graph

```
READY NOW:
  TGR-7m26  [P1] Fourier transform + kernel extraction (flat)  ← NEXT
  TGR-mphe  [P1] dS background quadratic + box terms
  TGR-c6su  [P2] SVT decomposition (flat, Path B)

BLOCKED:
  TGR-7m26 ──→ TGR-zq2k (flat form factors f₂, f₀)
  TGR-zq2k + TGR-pr04 ──→ TGR-tztc (cross-check Path A vs B)
  TGR-mphe ──→ TGR-7tcs (dS cubics) ──→ TGR-ug98 (full dS spectrum)
  TGR-zq2k + TGR-tztc + TGR-ug98 ──→ TGR-j6r9 (tests) ──→ TGR-af4a (example)
```

## Recommended Session 4 Strategy

**Priority 1 — Fourier transform (the missing piece):**
1. TGR-7m26: Implement `fourier_transform(expr)` that replaces `TDeriv(idx, arg, :partial)` → `Tensor(:k, [idx]) * arg`, then run `contract_momenta` to get `k_a k^a → k²`. Need to handle nested derivatives correctly (chain of ∂ → product of k's).
2. Apply to all 5 δ²S expressions, extract kernels, verify term counts.

**Priority 2 — Form factors:**
3. TGR-zq2k: `spin_project` all 5 kernels with `@variables k²` → verify f₂(z), f₀(z) match ground truth. With the simplifier fixes from session 3, this should now work end-to-end.

**Priority 3 — dS spectrum (independent track):**
4. TGR-mphe → TGR-7tcs → TGR-ug98

## What Works Now (verified in session 3)

The full spin projection pipeline works on momentum-space expressions:
```julia
using TensorGR; using Symbolics: @variables
@variables k²
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    # ... build momentum-space bilinear ...
    K = extract_kernel(expr, :h; registry=reg)
    result = spin_project(K, :spin2; registry=reg, k_sq=k²)
    # → clean scalar like (5//2) * k²
end
```

**What's missing:** converting position-space δ²S (with ∂ derivatives) to momentum-space (with k vectors). This is TGR-7m26.

## Ground Truth

```
f₂(z) = 1 − (α₂/κ)z − (β₂/κ)z²     [spin-2, z = k²]
f₀(z) = 1 + (6α₁+2α₂)z/κ + (6β₁+2β₂)z²/κ  [spin-0]
spin-1 projection = 0 identically
```

## Key Files

| File | Role |
|------|------|
| `src/action/kernel_extraction.jl` | extract_kernel, spin_project (now with contract_momenta loop), contract_momenta |
| `src/action/spin_projectors.jl` | Barnes-Rivers P², P¹, P⁰ˢ, P⁰ʷ (metric=:g, k_sq accepts Num) |
| `src/algebra/simplify.jl` | _simplify_one_pass (includes _simplify_scalars step) |
| `src/types.jl` | TScalar.== uses isequal (Num-safe) |
| `ext/TensorGRSymbolicsExt.jl` | _simplify_scalar_val for Expr and Num |
| `test/test_6deriv_spectrum.jl` | Tests: kernel, term counts, projector completeness |

## Quick Start

```bash
bd ready                              # see unblocked work
bd show TGR-7m26                      # Fourier transform (next task)
bd show TGR-zq2k                      # flat form factors (after 7m26)
julia --project -e 'using Pkg; Pkg.test()'
bd sync && git push
```
