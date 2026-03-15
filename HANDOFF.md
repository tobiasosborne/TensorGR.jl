# Session Handoff — 2026-03-15 (Session 2)

## Summary

Continuation session closing **22 more issues** (126 → 148). All pushed to master.
Test count: 360,500 → 361,065 (+565 new tests, zero regressions, zero failures).
Total issues in database: 353 (2 new bug issues). 49 ready to work.

## What was built this session

### Feynman Diagram Extensions
- `graviton_propagator(reg)` — harmonic (de Donder) gauge propagator D^{abcd}(k)
- `propagator_numerator` — P^{abcd} = ½(η^{ac}η^{bd} + η^{ad}η^{bc} - η^{ab}η^{cd})
- `matter_graviton_vertex(n, reg)` — n-graviton coupling for point particles (n=1,2)
- `scalar_matter_vertex(n, reg)` — n-graviton coupling for minimally-coupled scalar
- Files: `src/feynman/{propagator,matter_vertices}.jl`

### Spinor Extensions
- `SpinIndex` convenience wrapper — ergonomic `spinor(:A)`, `spinor_dot(:A)` constructors
- `spinor_dummy`, `spinor_dot_dummy`, `spinor_pair` — matched pair generation
- `define_soldering_form!` — Infeld-van der Waerden symbol σ^a_{AA'}
- `to_spinor_indices` / `to_tensor_indices` — bidirectional conversion
- Completeness rule: σ^a_{AA'} σ_a^{BB'} → δ^B_A δ^{B'}_A'
- Metric reconstruction: σ^a σ^b ε^{AB} ε^{A'B'} → g^{ab}
- Files: `src/spinors/{spin_index,soldering_form}.jl`

### Tetrad System
- `TetradProperties` struct + `define_tetrad!(reg, :e; manifold, metric)`
- Registry extended with `tetrads::Dict{Symbol, Any}` field
- Completeness rules: e^I_a E^a_J → δ^I_J, E^a_I e^I_b → δ^a_b
- Metricity rules: e^I_a e^J_b η_{IJ} → g_{ab} (both explicit and post-contraction)
- Files: `src/tetrads/tetrad.jl`, `src/registry.jl` (modified)

### PPN Framework
- `PPNParameters` struct — 10 standard PPN parameters (γ, β, ξ, α₁₋₃, ζ₁₋₄)
- `ppn_gr()` — GR values, `is_gr()` — check
- `define_ppn_potentials!(reg)` — U, U_{ij}, Φ_W, Φ₁₋₄, A, V_i, W_i
- `ppn_metric_ansatz(params, reg; order)` — Will (2018) Eqs 4.1–4.3
- Files: `src/ppn/metric_ansatz.jl`

### Scalar-Tensor Extensions
- `EFTDarkEnergy` — Bellini-Sawicki α parametrization (α_M, α_K, α_B, α_T, α_H)
- `eft_from_horndeski`, `eft_from_beyond_horndeski` — compute from theory
- `eft_stability`, `eft_observables`, `gw170817_constraint` — physics checks
- `degeneracy_conditions(DHOSTTheory)` — 3 algebraic conditions C₁,C₂,C₃
- `is_degenerate`, `dhost_class`, `horndeski_as_dhost`, `reduce_to_horndeski`
- `MultiHorndeskiTheory` — N scalar fields with field-space metric G_{IJ}
- `kinetic_matrix`, `multi_horndeski_L2/L3/L4`, `to_single_field`
- Files: `src/scalar_tensor/{eft_de,dhost_degeneracy,multi_horndeski}.jl`

### DDI Pipeline
- `simplify_with_ddis(expr; dim=4)` — convenience: registers + simplifies
- `has_ddi_rules(reg; dim, order)` — idempotent registration check
- Files: `src/algebra/ddi_rules.jl` (extended)

### Curvature Invariants
- `canonicalize_rinv(rinv, reg)` — xperm Butler-Portugal for RInv
- `rinv_symmetry_group(degree)` — Riemann symmetry generators on 4k+2 points
- `are_equivalent(r1, r2, reg)` — canonical comparison
- `DualRInv` type — invariants with Levi-Civita (left/right/double dual)
- `left_dual`, `right_dual`, `double_dual`, `pontryagin_rinv`
- Files: `src/invariants/{rinv.jl (extended), dual_rinv.jl}`

### Mode Coupling
- `mode_coupling_coefficient(l,m,l1,m1,l2,m2; types)` — angular integral coefficients
- `coupling_selection_rule(l,l1,l2)` — triangle + parity check
- `ModeCouplingTable`, `compute_coupling_table!` — precomputed tables
- Files: `src/perturbation/mode_coupling.jl`

### xIdeal Validation
- pp-wave (Brinkmann coords, H=x²-y²) = Petrov Type N
- Only Ψ₄ ≠ 0, I = J = 0, Kretschmann = 0 (VSI spacetime)
- **xIdeal epic auto-closed** (all validation issues complete)
- Files: `test/test_petrov_ppwave.jl`

### Research Designs Completed (8)
| Design | Issue | Key reference |
|--------|-------|---------------|
| Bianchi cosmology | TGR-34t.1 | Ellis & MacCallum (1969) |
| Multi-field PSALTer | TGR-4zw.1 | Lin et al (2019) |
| Bimetric gravity | TGR-wq0.1 | Hassan & Rosen (2012) |
| Metric-affine gravity | TGR-swh.1 | Hehl et al (1995) |
| Hamiltonian analysis | TGR-vdm.1 | Henneaux & Teitelboim (1992) |
| Index-free notation | TGR-xmm.1 | Wald (1984), Nutma (2014) |
| Clifford algebra | TGR-dai.1 | Peskin & Schroeder (1995) |

## Known Issues (bug issues created)

- **TGR-55f**: DHOST degeneracy symbolic evaluation — `degeneracy_conditions()` returns symbolic expressions that don't reduce to zero for Horndeski-as-DHOST. Needs CAS integration or numeric path. 9 `@test_broken`.
- **TGR-4aw**: RInv canonicalization conjugation — `_xperm_to_contraction` conjugation formula has edge cases where canonical contraction doesn't match expected form. 5 `@test_broken`.

## Cumulative state (Sessions 1+2)

- **148 issues closed** (69 → 148)
- **361,065 tests** pass, zero failures
- **49 issues ready** to work
- **203 open**, 154 blocked

## What's ready next

### Newly unblocked by this session:
- TGR-d42: Tensor contraction engine for Feynman diagrams (vzx + 9vn done)
- TGR-1rf: @spinor_manifold macro (ozc done)
- TGR-7me: Weyl spinor Ψ_{ABCD} (ozc done)
- TGR-2d4.2: Tetrad tensor registration (96h done)
- TGR-0o2: Spin coefficients as Ricci rotation coefficients (96h done)
- TGR-jt5: Cartan structure equations in tetrad basis (96h done)
- TGR-443.1.4: Syzygy detection via DDIs (443.1.2 done)
- TGR-443.1.5: Bidirectional RInv ↔ TensorExpr (443.1.2 done)
- TGR-68g: Second-order source terms (4p8 done)
- TGR-2vm: DHOST reduces to Horndeski (2wf done)
- TGR-bgl.4: PPN potential equations (bgl.2 done, needs bgl.13)

### High-impact ready issues:
- TGR-lej: Abstract tetrad indices in AST
- TGR-bgl.13: PPN-to-component bridge
- TGR-bgl.10: PPN gauge conditions
- TGR-bgl.11: PPN Poisson solver
- TGR-u19: BH-pert2 radial source assembly
- TGR-xlu.6: Lanczos-Lovelock validation
- TGR-4zw.2: Vector field spin projectors

## Key patterns for next agent

1. **Worktree isolation**: Implementation uses `isolation: worktree`. Merge conflicts are always in `test/runtests.jl` and `src/TensorGR.jl` (include/export lines) — resolve by keeping HEAD + adding unique lines from other side.

2. **Registry change**: `TensorRegistry` now has a `tetrads` field (added this session). Constructor takes one more `Dict{Symbol,Any}()` argument.

3. **Beads daemon**: Auto-pushes via `bd: backup` commits.

4. **Ground truth**: Every test MUST cite actual equation numbers from local PDFs in `reference/ground_truth/`. Previous agents hallucinated equation numbers.

5. **Test pattern**: New features go in `src/<subsystem>/file.jl`, tests in `test/test_file.jl`, include in `src/TensorGR.jl`, test include at end of `test/runtests.jl`.

6. **Registry pattern**: New tensor types register via `register_tensor!`, new VBundles via `define_vbundle!`, metrics via `metric_cache`/`delta_cache`, tetrads via `tetrads` dict.

7. **Subagent auditing**: Always verify subagent work — common bugs: wrong field names on structs (e.g. `rule.lhs` vs `rule.pattern`), missing exports, tuple destructuring issues, wrong xperm API calling conventions.
