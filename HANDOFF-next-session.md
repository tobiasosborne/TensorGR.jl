# HANDOFF: Session 19 — 6-Derivative Flat Kernel & Spin Projections

## Status: COMPLETE — Tests pass, benchmarks unchanged

- **6042 tests pass** (up from 5865), 0 errors
- **268 benchmarks pass**, 6 pre-existing failures (bench_12 dS simplified terms), 3 broken (stretch goals)
- New: `build_6deriv_flat_kernel`, `flat_6deriv_spin_projections`, `scale_kernel`, `combine_kernels`

## What Was Done This Session

### 6-Derivative Flat Kernel Construction (TGR-c6su)

Built the combined kinetic kernel for 6-derivative gravity on flat background:

```
K = κ·K_FP − 2(α₁ + β₁k²)·K_R² − 2(α₂ + β₂k²)·K_Ric²
```

**Key sign determination**: All curvature-squared terms enter with −2× coefficient relative to their coupling constants. The box terms (R□R, Ric□Ric) give −2βk² (NOT +2βk²), confirmed numerically against Buoninfante form factors.

### New API Functions

- `scale_kernel(K, factor)` — scale bilinear coefficients by TensorExpr or Rational
- `combine_kernels(kernels)` — concatenate terms from multiple kernels
- `build_6deriv_flat_kernel(reg; κ, α₁, α₂, β₁, β₂)` — combined 6-deriv kernel
- `flat_6deriv_spin_projections(reg; κ, α₁, α₂, β₁, β₂)` — spin-projected form factors

### Verification

Spin projections verified against Buoninfante et al. (2012.11829) Eq. 2.13:
- f₂(z) = 1 − (α₂/κ)z − (β₂/κ)z² ✓ (20 random parameter sets × 3 k² values)
- f₀(z) = 1 + (6α₁+2α₂)z/κ + (6β₁+2β₂)z²/κ ✓
- spin-1 = 0 ✓ (gauge invariance)
- spin-0w = 0 ✓ (gauge invariance)
- GR limit (κ=1, all α=β=0): matches pure Fierz-Pauli ✓

### Files Changed

| File | Change |
|------|--------|
| `src/action/kernel_extraction.jl` | Fixed box term signs (−2β, not +2β); added scale_kernel, combine_kernels, build_6deriv_flat_kernel, flat_6deriv_spin_projections |
| `src/TensorGR.jl` | Added exports for new functions |
| `test/test_6deriv_spectrum.jl` | Added 177 new tests: Buoninfante form factors (random params), GR limit, scale/combine utilities |

## Next Steps

### Immediate
- Close TGR-c6su
- Consider writing `examples/26_6deriv_flat_spectrum.jl` demonstrating the full pipeline

### Future (P3+)
- Improve simplifier to recover term count efficiency for dS benchmarks (bench_12)
- The `_avoid` mechanism is a minimal fix; a more elegant solution would use global fresh_index counters
