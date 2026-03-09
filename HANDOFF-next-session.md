# HANDOFF: 6-Deriv Spectrum Pipeline — Session 5

## What Was Done (Sessions 1-4)

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

### Session 4
- **TGR-7m26** ✅ — Fourier transform + kernel extraction for all 5 flat δ²S terms
  - Pipeline: δ²S → to_fourier(∂→k) → simplify → extract_kernel → KineticKernel
  - EH (κR): 12 bilinear kernel terms, k² momentum degree
  - R² (α₁R²): 23 terms, k⁴ degree
  - Ric² (α₂RμνRμν): 19 terms, k⁴ degree
  - R□R (β₁R□R): 26 terms, k⁶ degree (built as 2(δR)(□δR) on flat)
  - Ric□Ric (β₂Rμν□Rμν): 19 terms, k⁶ degree (built as 2(δRic)(□δRic) on flat)
- All 4927 tests pass

---

## What's Next — Priority Order

### Priority 1: TGR-zq2k — Barnes-Rivers projection → flat form factors [P1, READY]

**This is the main deliverable of Path A on flat background.**

Use `spin_project` to contract each kernel with Barnes-Rivers projectors.
Expected results (Buoninfante 2012.11829 Eq.2.13):
- Spin-2: O₂(k²) = κ·k²·f₂(k²) where f₂(z) = 1 − (α₂/κ)z − (β₂/κ)z²
- Spin-0: O₀(k²) = −2κ·k²·f₀(k²) where f₀(z) = 1 + (6α₁+2α₂)z/κ + (6β₁+2β₂)z²/κ
- Spin-1: O₁ = 0 identically (diffeomorphism invariance)

**Key approach**: Use `@variables k²` from Symbolics.jl so CAS handles k²/k² cancellations.
The spin_project pipeline (session 3 verified) handles contract_momenta internally.

### Priority 2: TGR-mphe — dS background quadratic + box terms [P1, READY]

Independent track. Compute δ²S on maximally symmetric background (curved=true).
Uses `maximally_symmetric_background!` with `:Λ` cosmological constant.

### Priority 3: TGR-c6su — SVT decomposition of δ²S flat [P2, READY]

Path B (3+1 SVT) for cross-check against Path A form factors.

---

## Dependency Graph (updated)

```
COMPLETED:
  ✅ TGR-ncdr (0.1 kernel) → ✅ TGR-ud97 (0.2 spin proj)
  ✅ TGR-w7jq (1.1 δ²S flat) → ✅ TGR-7m26 (1.2 Fourier+kernel)

READY NOW:
  TGR-zq2k  [P1] Step 1.3: BR flat form factors  ← CRITICAL PATH (next!)
  TGR-mphe  [P1] Step 3.1: dS quad+box terms
  TGR-c6su  [P2] Step 2.1: SVT decompose (Path B)

BLOCKED:
  TGR-zq2k ──→ TGR-tztc (cross-check A vs B)  [needs TGR-pr04 too]
  TGR-c6su ──→ TGR-pr04 (SVT QF) ──→ TGR-tztc
  TGR-mphe ──→ TGR-7tcs (dS cubics) ──→ TGR-ug98 (full dS spectrum)
  TGR-zq2k + TGR-tztc + TGR-ug98 ──→ TGR-j6r9 (tests) ──→ TGR-af4a (example)
```

## How to Build Kernels for TGR-zq2k

The Step 1.2 tests show the exact recipe. Here's the combined kernel construction:

```julia
using TensorGR
# using Symbolics: @variables  # needed for k_sq=k² in spin_project

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp = define_metric_perturbation!(reg, :g, :h)
    set_vanishing!(reg, :Ric)
    set_vanishing!(reg, :RicScalar)
    set_vanishing!(reg, :Riem)

    # 1. EH: κR → δ²R
    δ2R = simplify(δricci_scalar(mp, 2); registry=reg)
    K_EH = extract_kernel(simplify(to_fourier(δ2R); registry=reg), :h; registry=reg)

    # 2. R²: α₁R² → (δR)²
    δ1R = δricci_scalar(mp, 1)
    δR_sq = simplify(δ1R * δ1R; registry=reg)
    K_R2 = extract_kernel(simplify(to_fourier(δR_sq); registry=reg), :h; registry=reg)

    # 3. Ric²: α₂RμνRμν → (δRic)²
    δRic1 = δricci(mp, down(:a), down(:b), 1)
    δRic2 = δricci(mp, down(:c), down(:d), 1)
    δRic_sq = simplify(
        δRic1 * δRic2 * Tensor(:g, [up(:a), up(:c)]) * Tensor(:g, [up(:b), up(:d)]);
        registry=reg)
    K_Ric2 = extract_kernel(simplify(to_fourier(δRic_sq); registry=reg), :h; registry=reg)

    # 4. R□R: β₁R□R → 2(δR)(□δR) on flat
    δ1R_2 = δricci_scalar(mp, 1)
    box_δR = Tensor(:g, [up(:e), up(:f)]) *
             TDeriv(down(:e), TDeriv(down(:f), δ1R_2))
    δ2_RboxR = simplify(TScalar(2) * δ1R * box_δR; registry=reg)
    K_RboxR = extract_kernel(simplify(to_fourier(δ2_RboxR); registry=reg), :h; registry=reg)

    # 5. Ric□Ric: β₂Rμν□Rμν → 2(δRic)(□δRic) on flat
    δRic_left = δricci(mp, down(:p), down(:q), 1)
    δRic_ij = δricci(mp, down(:i), down(:j), 1)
    δRic_up = δRic_ij * Tensor(:g, [up(:p), up(:i)]) * Tensor(:g, [up(:q), up(:j)])
    box_δRic = Tensor(:g, [up(:e), up(:f)]) *
               TDeriv(down(:e), TDeriv(down(:f), δRic_up))
    δ2_RicBoxRic = simplify(TScalar(2) * δRic_left * box_δRic; registry=reg)
    K_RicBoxRic = extract_kernel(simplify(to_fourier(δ2_RicBoxRic); registry=reg), :h; registry=reg)

    # Now spin_project each with @variables k² and combine with couplings
    # result_spin2 = κ * spin_project(K_EH, :spin2; k_sq=k²) + ...
end
```

## Known Issues

- `simplify` emits "did not converge after 20 iterations" on Fourier-space expressions.
  This is cosmetic — the simplifier oscillates between equivalent forms due to
  canonicalization + metric contraction fighting. Results are correct.
  The `g^a_a → dim` trace rule EXISTS in contraction.jl:173-180.
- Default maxiter=20 in simplify. Use `maxiter=40` if needed.

## Key Files

| File | Role |
|------|------|
| `src/action/kernel_extraction.jl` | extract_kernel, spin_project, contract_momenta |
| `src/action/spin_projectors.jl` | Barnes-Rivers P², P¹, P⁰ˢ, P⁰ʷ |
| `src/svt/fourier.jl` | to_fourier (∂_a → k_a) |
| `src/algebra/simplify.jl` | simplify pipeline (maxiter=20 default) |
| `src/algebra/contraction.jl` | metric contraction + g^a_a → dim trace rule |
| `ext/TensorGRSymbolicsExt.jl` | CAS dispatch for Symbolics.Num |
| `test/test_6deriv_spectrum.jl` | All spectrum tests (1321 tests in this file) |
| `examples/13_6deriv_particle_spectrum.jl` | Numerical ground truth |

## Quick Start

```bash
bd ready                              # see unblocked work
bd show TGR-zq2k                      # flat form factors (next task)
julia --project -e 'using Pkg; Pkg.test()'  # 4927 tests, ~4.5min
bd sync && git push
```
