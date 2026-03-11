# HANDOFF: 6-Deriv dS Spectrum — Session 15

## Status: ALL GREEN

- **5851 tests pass**, 0 errors, 0 broken, 0 failed
- **13 benchmarks pass** (152 + 119 = 271 benchmark tests)
- All code pushed to remote

## Completed This Session (15)

- **Fixed all 6 cubic BC parameters** — `bc_R3`, `bc_RRicSq`, `bc_Ric3`, `bc_RRiem2`, `bc_RicRiem2`, `bc_Riem3` all had incorrect coefficients. The b/c params were 4x too large and e was 2x too large, caused by a convention conversion error when transcribing from the numerical Bueno-Cano procedure (Λ_BC vs Λ_TGR).
- **Added `@assert` for all 6 invariants** in `examples/14_cubic_bc_params.jl` (previously only I1-I2 had assertions)
- **Corrected values** (verified against parametric Riemann numerical procedure):
  - I1 (R³): b=6γΛ, e=24γΛ² (was b=24γΛ, e=48γΛ²)
  - I2 (R·Ric²): b=γΛ, c=2γΛ, e=6γΛ² (was b=4γΛ, e=12γΛ²)
  - I3 (Ric³): c=3γΛ/2, e=3γΛ²/2 (was c=6γΛ, e=3γΛ²)
  - I4 (R·Riem²): a=4γΛ, b=2γΛ/3, e=4γΛ² (was b=8γΛ/3, e=8γΛ²)
  - I5 (Ric·Riem²): a=γΛ, c=2γΛ/3, e=γΛ² (was a=4γΛ/3, e=2γΛ²)
  - I6 (Riem³): a=2γΛ, e=2γΛ²/3 (was e=4γΛ²/3)

## What To Do Next

### Priority 1: Independent cross-check via TGR perturbation engine

The BC parameters are now verified against ONE method (parametric Riemann numerical procedure). They need a second independent verification. The best approach is the TGR perturbation engine:

1. Set up 4D manifold with MSS background via `maximally_symmetric_background!`
2. For each cubic invariant I_i, construct the expression and compute `expand_perturbation(I_i, mp, 2)`
3. Simplify on the MSS background
4. Fourier transform, extract kernel, spin project
5. Compare spin-sector form factors against BC mass formulas (Eqs. 17-19)

**Infrastructure exists**: `bench_12` already computes `expand_perturbation + simplify` for all 6 invariants on MSS and pins term counts. The missing piece is step 4-5 (Fourier → kernel → spin_project → compare to BC predictions). This would close the verification loop.

**Cost**: ~66s for all 6 cubics with `parallel=true` (from bench_12 timings).

### Priority 2: Box terms (β₁R□R, β₂Ric□Ric) on dS

On dS, box terms don't shift (a,b,c,e) directly (since □R̄ = 0), but enter through momentum-dependent mass shifts: α → α_eff(m²) = α − β·m². Add a `β_correction` option to `dS_spectrum_6deriv` that iteratively solves for self-consistent mass poles.

### Priority 3: Path B (SVT decomposition) cross-check

Open beads chain: TGR-c6su → TGR-pr04 → TGR-tztc → TGR-j6r9 → TGR-af4a

Independent validation of flat spectrum via 3+1 SVT decomposition. Large body of work.

### Priority 4: Example 13 polish

`examples/13_6deriv_particle_spectrum.jl` still has local BC parameter functions — could use the exported API instead.

## Key Files

| File | Role |
|------|------|
| `src/action/kernel_extraction.jl` | KineticKernel, spin_project, BC params, dS_spectrum_6deriv |
| `src/action/spin_projectors.jl` | Barnes-Rivers P², P¹, P⁰ˢ, P⁰ʷ, θ, ω |
| `test/test_6deriv_spectrum.jl` | 2245 tests: BC params, form factors, spin projection, dS spectrum |
| `benchmarks/bench_12_6deriv_dS.jl` | δ²(I_i) term count ground truth for 6 cubic invariants |
| `benchmarks/bench_13_spectrum.jl` | 119 tests: API + projection + perturbation pipeline |
| `examples/13_6deriv_particle_spectrum.jl` | End-to-end example script |
| `examples/14_cubic_bc_params.jl` | Numerical BC param derivation with @assert for all 6 invariants |

## Critical Gotchas

- `spin_project` returns **Tr(K·P^J)**, NOT the form factor. Divide by dim(sector): {5, 3, 1, 1} for {spin-2, spin-1, spin-0-s, spin-0-w}
- `_eval_ksq_val` handles both symbol ops (`:^`) and function-ref ops (`^`) — don't regress this
- `simplify` is ambiguous in Main after Symbolics loads — the `import TensorGR: simplify` in runtests.jl:58 fixes it
- BC convention: TGR's Λ means R̄_{μν} = Λg_{μν}. Bueno-Cano's Λ_BC = Λ/3
- `δricci_scalar(mp, n)` returns the ε^n COEFFICIENT, not (1/n!)d^n/dε^n
- Position-space perturbation path may leave uncontracted `RicScalar` tensors — use direct momentum-space kernel builders for reliable spin projection

## Open Beads Issues (6-deriv related)

| Issue | Status | Description |
|-------|--------|-------------|
| TGR-c6su | OPEN P2 | Step 2.1: SVT decomposition of δ²S (flat) |
| TGR-pr04 | OPEN P2 | Step 2.2: SVT QuadraticForms + propagators |
| TGR-tztc | OPEN P2 | Step 2.3: Cross-check Path A vs Path B |
| TGR-j6r9 | OPEN P2 | Step 5: Tests + benchmark (blocked by tztc) |
| TGR-af4a | OPEN P2 | Step 6: Example script + module integration |
