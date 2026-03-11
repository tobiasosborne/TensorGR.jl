# HANDOFF: 6-Deriv dS Spectrum — Session 14

## Status: ALL GREEN

- **5851 tests pass**, 0 errors, 0 broken, 0 failed — just verified, DO NOT re-run tests
- **13 benchmarks pass** (152 + 119 = 271 benchmark tests) — bench_13 is new
- All code pushed to remote, beads synced

## Completed This Session (14)

- **Fixed 26 test errors** (two bugs):
  1. `simplify` name collision: `Symbolics.simplify` shadowed `TensorGR.simplify` after `test_cas_integration.jl` loads Symbolics. Fix: added `import TensorGR: simplify` in `test/runtests.jl` line 58.
  2. `_eval_ksq_val` function-ref Expr: Symbolics produces `Expr(:call, ^, ...)` with actual function `^` instead of symbol `:^`. Fix: added `op === *`, `op === ^`, etc. alongside existing `op === :*` checks in `src/action/kernel_extraction.jl:401-409`.
- **Created bench_13_spectrum.jl** (119 tests, 3 tiers): dS API + spin projection + perturbation pipeline
- **Closed beads**: TGR-60sx, TGR-mphe, TGR-7tcs, TGR-ug98

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

# Barnes-Rivers spin projection (flat)
K_FP = build_FP_momentum_kernel(reg)
r2 = spin_project(K_FP, :spin2; registry=reg)  # returns Tr(K·P²)

# Direct momentum-space kernels
K_R2 = build_R2_momentum_kernel(reg)     # (δR)²
K_Ric2 = build_Ric2_momentum_kernel(reg) # (δRic)²
```

## What To Do Next

### Priority 1: Analytical verification of I4-I6 cubic BC parameters

The BC parameters for I4-I6 (involving Riemann) were numerically extracted. They need analytical verification.

**Current hardcoded formulas** (in `src/action/kernel_extraction.jl:468-475`):
- `bc_RRiem2(γ, Λ)`:  a=4γΛ, b=(8/3)γΛ, c=0, e=8γΛ²
- `bc_RicRiem2(γ, Λ)`: a=(4/3)γΛ, b=0, c=(2/3)γΛ, e=2γΛ²
- `bc_Riem3(γ, Λ)`:   a=2γΛ, b=0, c=0, e=(4/3)γΛ²

**How to verify**: Use the Bueno-Cano Eqs. (13)-(14) from 1607.06463. The BC parameters come from expanding the Lagrangian L on the "fiducial Riemann" R̃_{abcd}(Λ_BC, α):
- R̃_{abcd} = Λ_BC(g_{ac}g_{bd} − g_{ad}g_{bc}) + α R̂_{abcd}
- a = (1/2) ∂²L/∂α² |_{α=0},  e = ∂L/∂α |_{α=0}
- b and c come from tracing P^{μρνσ} = ∂L/∂R_{μρνσ}

On MSS with Λ_BC = Λ/3 (D=4):
- R̄_{abcd} = (Λ/3)(g_{ac}g_{bd} − g_{ad}g_{bc})
- Kretschner = R̄_{abcd}R̄^{abcd} = 8Λ²/3
- Ricci² = R̄_{ab}R̄^{ab} = 4Λ²
- R̄² = 16Λ²

For each cubic I_i, compute ∂I_i/∂R_{μρνσ} and ∂²I_i/∂R_{μρνσ}∂R_{αβγδ} on MSS, then trace to get (a,b,c,e). This is a pen-and-paper calculation — verify against the hardcoded values.

**Alternative approach**: Use TGR's perturbation engine to compute δ²(I_i) on MSS and compare the resulting quadratic expression structure. This is expensive (~66s for all 6 cubics with `parallel=true`) but fully automated.

### Priority 2: Box terms (β₁R□R, β₂Ric□Ric) on dS

Currently β₁, β₂ only contribute to flat form factors:
```julia
flat_f2 = (-α₂/κ, -β₂/κ)   # f₂(z) = 1 + c₁z + c₂z²
flat_f0 = ((6α₁+2α₂)/κ, (6β₁+2β₂)/κ)
```

On dS, the Bueno-Cano framework only handles algebraic (non-derivative) curvature theories. Box terms introduce 6th-order field equations that go beyond BC's parametrization. The physical effect is:
- On MSS, □R̄ = 0 and □R̄_{μν} = 0, so box terms don't directly shift (a,b,c,e)
- Their effect enters through **momentum-dependent mass shifts**: at the pole z = m², replace α → α_eff(m²) = α − β·m²
- This means the dS masses from Eqs.17-19 get implicit β corrections when you solve for the actual pole locations

**Implementation idea**: Add a `β_correction` option to `dS_spectrum_6deriv` that iteratively solves for the self-consistent mass poles: m² = m²(α_eff(m²)).

### Priority 3: Path B (SVT decomposition) cross-check

Open beads chain: TGR-c6su → TGR-pr04 → TGR-tztc → TGR-j6r9 → TGR-af4a

This provides independent validation of the flat spectrum via 3+1 SVT decomposition. Large body of work. Steps:
1. `split_all_spacetime` on each δ²(term) using `define_foliation!`
2. SVT substitution (Bardeen gauge)
3. Build QuadraticForm per sector with Symbolics.jl
4. Compute propagators, compare poles to Path A form factor zeros

### Priority 4: Example script polish

`examples/13_6deriv_particle_spectrum.jl` already works but duplicates the BC parameter functions locally instead of using the exported API. Could be simplified to just call `dS_spectrum_6deriv()`.

## Key Files

| File | Role |
|------|------|
| `src/action/kernel_extraction.jl` | KineticKernel, spin_project, BC params, dS_spectrum_6deriv |
| `src/action/spin_projectors.jl` | Barnes-Rivers P², P¹, P⁰ˢ, P⁰ʷ, θ, ω |
| `test/test_6deriv_spectrum.jl` | 2245 tests: BC params, form factors, spin projection, dS spectrum |
| `benchmarks/bench_13_spectrum.jl` | 119 tests: API + projection + perturbation pipeline |
| `benchmarks/bench_12_6deriv_dS.jl` | δ²(I_i) term count ground truth for 6 cubic invariants |
| `examples/13_6deriv_particle_spectrum.jl` | End-to-end example script |

## Critical Gotchas

- `spin_project` returns **Tr(K·P^J)**, NOT the form factor. Divide by dim(sector): {5, 3, 1, 1} for {spin-2, spin-1, spin-0-s, spin-0-w}
- `_eval_ksq_val` now handles both symbol ops (`:^`) and function-ref ops (`^`) — don't regress this
- `simplify` is ambiguous in Main after Symbolics loads — the `import TensorGR: simplify` in runtests.jl:58 fixes it
- BC convention: TGR's Λ means R̄_{μν} = Λg_{μν}. Bueno-Cano's Λ_BC = Λ/3
- `δricci_scalar(mp, n)` returns the ε^n COEFFICIENT, not (1/n!)d^n/dε^n
- Position-space perturbation path (δ²R → Fourier → kernel → spin_project) may leave uncontracted `RicScalar` tensors — use direct momentum-space kernel builders (`build_FP_momentum_kernel`, etc.) for reliable spin projection

## Open Beads Issues (6-deriv related)

| Issue | Status | Description |
|-------|--------|-------------|
| TGR-c6su | OPEN P2 | Step 2.1: SVT decomposition of δ²S (flat) |
| TGR-pr04 | OPEN P2 | Step 2.2: SVT QuadraticForms + propagators |
| TGR-tztc | OPEN P2 | Step 2.3: Cross-check Path A vs Path B |
| TGR-j6r9 | OPEN P2 | Step 5: Tests + benchmark (blocked by tztc) |
| TGR-af4a | OPEN P2 | Step 6: Example script + module integration |
