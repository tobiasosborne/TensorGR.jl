# HANDOFF: 6-Deriv dS Spectrum — Session 13

## Completed This Session

- **Fixed BC parameter convention bugs**: factor-of-3 error on `e` for R², Ric², R³ in test + example
- **Added all 6 cubic invariant BC parameters** (I1-I6): `bc_R3`, `bc_RRicSq`, `bc_Ric3`, `bc_RRiem2`, `bc_RicRiem2`, `bc_Riem3`
- **Built `BuenoCanoParams` struct** + `dS_spectrum_6deriv()` API in `src/action/kernel_extraction.jl`
- **Exported** all BC functions + `dS_spectrum_6deriv` from `src/TensorGR.jl`
- **Updated example** `examples/13_6deriv_particle_spectrum.jl` with full 9-coupling dS spectrum
- **657 new passing tests** (2244 total in test_6deriv_spectrum.jl), zero regressions
- 26 pre-existing errors unchanged (simplify import issue in test_solve.jl, test_product_manifold.jl)

## What Works Now

```julia
using TensorGR

# Full dS spectrum with all 9 algebraic couplings
s = dS_spectrum_6deriv(κ=1.0, α₁=-0.1, α₂=0.3,
                        γ₁=0.01, γ₂=-0.005, γ₃=0.003,
                        γ₄=0.002, γ₅=-0.001, γ₆=0.001, Λ=0.1)
s.κ_eff_inv    # effective Newton constant (Eq.17)
s.m2_graviton  # massive spin-2 mass (Eq.18)
s.m2_scalar    # spin-0 mass (Eq.19)
s.flat_f2      # flat form factor coefficients (c₁, c₂)
s.flat_f0      # flat form factor coefficients (c₁, c₂)

# Individual BC parameters
p = bc_EH(1.0, 0.1) + bc_R2(-0.1, 0.1) + bc_RicSq(0.3, 0.1) + bc_R3(0.01, 0.1)
# p.a, p.b, p.c, p.e
```

## What To Do Next

### 1. Fix pre-existing 26 test errors
- `simplify` not defined in `Main` in test_solve.jl and test_product_manifold.jl
- `_eval_spin_scalar` errors in momentum-space kernel tests (R² and Ric²)
- These are import/scoping issues in runtests.jl, not logic bugs

### 2. Analytical verification of I4-I6 cubic BC parameters
- I1-I3 (R³, R·Ric², Ric³) are exact polynomial formulas
- I4-I6 (R·Riem², Ric·Riem², Riem³) use numerical extraction (underdetermined at D=4)
- Options: use D>4 temporarily, or derive analytically from block-diagonal Riemann

### 3. Box terms (β₁R□R, β₂Ric□Ric) on dS
- Currently these only contribute to flat form factors
- On MSS, □R̄ = 0, so effect is through α → α − βm² replacement
- Need to implement the momentum-dependent mass shift

### 4. Close beads issues
- TGR-mphe (Step 3.1): mostly done — BC params computed, API built
- TGR-7tcs (Step 3.2): cubic γᵢ contributions implemented
- TGR-ug98 (Step 3.3): full dS spectrum API is working

## Key Files Changed

| File | Change |
|------|--------|
| `src/action/kernel_extraction.jl` | Added `BuenoCanoParams`, 9 `bc_*` functions, `dS_spectrum_6deriv` |
| `src/TensorGR.jl` | Added exports for BC params and spectrum API |
| `test/test_6deriv_spectrum.jl` | Fixed BC convention bugs, added cubic+full spectrum+API tests |
| `examples/13_6deriv_particle_spectrum.jl` | Full 9-coupling dS spectrum with cubics |

## BC Parameter Convention (RESOLVED)

TGR uses Λ where R̄_μν = Λg_μν (D=4). Bueno-Cano uses Λ_BC = Λ/3.

Correct formulas (TGR convention):
- `bc_R2(α₁, Λ)`: e = 8α₁Λ (was 8α₁Λ/3)
- `bc_RicSq(α₂, Λ)`: e = 2α₂Λ (was 2α₂Λ/3)
- `bc_R3(γ, Λ)`: b = 24γΛ, e = 48γΛ² (was b=8γΛ, e=48γΛ²/9)
