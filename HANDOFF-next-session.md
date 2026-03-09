# HANDOFF: 6-Deriv Spectrum Pipeline — Session 3

## What Was Done (Session 2)

### TGR-ud97 ✅ — Barnes-Rivers Spin Projection (Step 0.2)
**Verified numerically** against the Lichnerowicz kernel for pure EH:
- spin-2: coefficient k² ✓
- spin-1: 0 ✓ (diffeomorphism invariance)
- spin-0s: -k²/2 ✓
- spin-0w: -k²/2 ✓
- Transfer operators: 0 ✓
- Reconstruction residual: ~10⁻¹⁵

**Key finding**: The symbolic pipeline (δ²R → Fourier → extract_kernel → spin_project) works end-to-end, but the simplifier **does not fully reduce** the spin-projected expressions to scalar form. The issue is that the simplifier cannot evaluate:
- Flat metric traces: g^a_a = d (dimension)
- Momentum contractions: already handled by `contract_momenta`
- Mixed k²/1/k² cancellation: partially handled by `_simplify_k_sq_pairs!`

The spin-projected results have 58/36/33/12 terms instead of collapsing to single k² expressions. **Numerical evaluation confirms correctness.**

### TGR-w7jq ✅ — Build δ²S for 6-deriv Action on Flat
All 5 expressions computed with pinned term counts:

| Expression | Terms | Time |
|---|---|---|
| δ²R (EH) | 8 | ~13s |
| (δR)² (R²) | 4 | ~0.2s |
| (δRic)² (Ric²) | 4 | ~0.06s |
| δ²(R□R) | 16 | ~1.8s |
| δ²(Ric□Ric) | 18 | ~0.1s |

**Note**: These counts use `set_vanishing!` for Ric, RicScalar, Riem (flat background). The simplifier emits "did not converge" warnings but results are stable.

### Tests added to `test/test_6deriv_spectrum.jl`:
- `extract_kernel basic` (5 assertions)
- `extract_kernel from TSum` (1 assertion)
- `contract_momenta` (3 assertions)
- `δ²S term counts on flat background` (3 term-count assertions: 8, 4, 4)
- `spin projection: numerical Lichnerowicz verification` (structural assertion + documentation)

### All tests pass. Not yet committed.

---

## What's Ready Now (Wave 2b)

Run `bd ready` to see unblocked issues. After closing TGR-ud97 and TGR-w7jq, these are now unblocked:

### TGR-7m26 — Step 1.2: Fourier transform + kernel extraction (flat)
**Status**: Unblocked (was waiting on TGR-w7jq).

**What to do**:
1. For each of the 5 δ²S expressions, apply `to_fourier` + `simplify`
2. Extract kernel via `extract_kernel(fourier_expr, :h)`
3. Pin Fourier term counts

```julia
# After building all 5 δ²S (reuse TGR-w7jq code):
δ2R_f = simplify(to_fourier(δ2R); registry=reg, maxiter=40)
K_R = extract_kernel(δ2R_f, :h; registry=reg)
# ... repeat for all 5
```

### TGR-c6su — Step 2.1: 3+1 SVT decomposition of δ²S (flat)
**Status**: Unblocked (was waiting on TGR-w7jq).

**What to do**: Follow `examples/08_postquantum_gravity.jl` pattern with `define_foliation!`, `split_all_spacetime`, SVT substitution.

### TGR-mphe — Step 3.1: dS background — quadratic + box terms
**Status**: Unblocked (was waiting on TGR-ud97).

**What to do**: Set up dS background with `maximally_symmetric_background!` and `curved=true`, compute δ²S for quadratic terms.

---

## Critical Path for Flat Spectrum

```
TGR-7m26 (Fourier+kernel) ──→ TGR-zq2k (BR projection → form factors)
                                   ↓
                              VERIFY f₂, f₀ against ground truth
```

The key bottleneck is the **symbolic simplification limitation**: `spin_project` produces expressions that don't fully reduce. Two options:

### Option A: Numerical evaluation (recommended for validation)
Evaluate the spin-projected expressions at specific parameter values and verify f₂(z) = 1 − (α₂/κ)z − (β₂/κ)z² etc. This was already proven to work in TGR-ud97.

### Option B: Add momentum-space trace rules
Teach the simplifier that g^a_a = d in momentum space (or use `set_dimension!` if available). This would allow full symbolic reduction. More work but gives symbolic form factors.

---

## Parallelism Strategy for Session 3

**Wave 2b** (can be parallel):
- Agent A: TGR-7m26 — Fourier + kernel extraction for all 5 terms
- Agent B: TGR-c6su — SVT decomposition (independent path)
- Agent C: TGR-mphe — dS background quadratic terms

**Wave 3** (after 2b):
- TGR-zq2k: spin_project all 5 kernels → flat form factors f₂(k²), f₀(k²)
- TGR-pr04: SVT QuadraticForms
- TGR-7tcs: dS cubic contributions

---

## Key Files

| File | Role |
|------|------|
| `src/action/kernel_extraction.jl` | extract_kernel, spin_project, contract_momenta |
| `src/action/spin_projectors.jl` | Barnes-Rivers P², P¹, P⁰ˢ, P⁰ʷ, T^sw, T^ws |
| `test/test_6deriv_spectrum.jl` | MODIFIED: kernel + term count tests added |
| `HANDOFF-6deriv-spectrum.md` | Master handoff (full pipeline spec) |
| `examples/13_6deriv_particle_spectrum.jl` | Ground truth: numerical form factors |

## Ground Truth (flat spectrum)

```
f₂(z) = 1 − (α₂/κ)z − (β₂/κ)z²     [spin-2, z = k²]
f₀(z) = 1 + (6α₁+2α₂)z/κ + (6β₁+2β₂)z²/κ  [spin-0]
spin-1 projection = 0 identically
```

## Quick Start

```bash
bd ready
bd blocked

# Verify tests pass
julia --project /tmp/test_kernel.jl
julia --project /tmp/test_termcounts.jl

# Session end protocol
bd sync && git push
```
