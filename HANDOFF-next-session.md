# HANDOFF: 6-Deriv Spectrum Pipeline — Session 11

## Completed This Session

- **Flat form factors DONE** (TGR-zq2k): 6-deriv flat-space particle spectrum fully verified
- **Approach D implemented**: Direct momentum-space kernel construction bypasses `_standardize_h_indices` bug
- 3 new exported functions: `build_FP_momentum_kernel`, `build_R2_momentum_kernel`, `build_Ric2_momentum_kernel`
- `_eval_spin_scalar`: numerical evaluator for spin projection results
- **300 tests pass** for form factors (100 random points × 3 checks each)
- Verified: Stelle mass formulas, residue sum rule, gauge invariance (spin-1=0, spin-0-w=0)
- 4662 tests pass total (26 pre-existing Julia 1.12 errors, 0 regressions)

## Key Results

Individual kernel spin projections (verified numerically):
- FP: {5k²/2, 0, -k², 0}
- R²: {0, 0, 3k⁴, 0}
- Ric²: {5k⁴/4, 0, k⁴, 0}

Combined form factors (K = κ K_FP + 2(α₁-β₁k²)K_R² + 2(α₂-β₂k²)K_Ric²):
- f₂ = 1 + (α₂/κ)k² - (β₂/κ)k⁴  [spin-2]
- f₀ = 1 - (6α₁+2α₂)/κ k² + (6β₁+2β₂)/κ k⁴  [spin-0]
- Sign convention note: differs from Buoninfante by sign on k² terms (metric signature convention), but physics is identical (mass formulas match)

## Still Open

### TGR-60sx [P1 bug]: _standardize_h_indices dummy propagation
- Root cause understood: when lowering Up h indices, metric connectors with original names clash with right-side Down indices
- Workaround in place (Approach D), but the bug still blocks the perturbation-engine pipeline
- Fix: ~30 lines in `_standardize_h_indices` — rename right-side Down indices that clash with connector names

### TGR-mphe [P1 task]: dS background quadratic + box terms
- Next major milestone: compute δ²(term) for all 14 action terms on maximally symmetric background
- Setup: `maximally_symmetric_background!(reg, :M4; metric=:g, cosmological_constant=:Λ)`
- Background: R̄=4Λ, R̄_{μν}=Λg_{μν}, R̄_{abcd}=(Λ/3)(g_{ac}g_{bd}-g_{ad}g_{bc})
- The 6 cubic invariant δ² computations are already in `examples/11_6deriv_gravity_dS.jl` (66s parallel)
- Approach: use direct momentum-space for quadratic/box terms, perturbation engine for cubics
- Can potentially use Bueno-Cano formalism (1607.06463 Eqs.17-19) for dS spectrum

### TGR-c6su [P2 task]: 3+1 SVT decomposition of δ²S (flat)
- Alternative path (Path B) to form factors via SVT decomposition
- Would cross-validate Path A results

## Key Files

| File | Role |
|------|------|
| `src/action/kernel_extraction.jl` | Kernel builders, spin_project, contract_momenta, _eval_spin_scalar |
| `src/action/spin_projectors.jl` | Barnes-Rivers P2/P1/P0s/P0w/θ/ω projectors |
| `test/test_6deriv_spectrum.jl` | All spectrum tests (form factors, Stelle, residues) |
| `examples/11_6deriv_gravity_dS.jl` | 6 cubic invariants on dS |
| `examples/13_6deriv_particle_spectrum.jl` | Numerical reference |

## Beads Issues to Update

- **TGR-zq2k**: CLOSE — flat form factors verified (Approach D)
- **TGR-60sx**: Keep open — underlying bug not fixed, just worked around
- **TGR-mphe**: Ready to start — dS background terms
