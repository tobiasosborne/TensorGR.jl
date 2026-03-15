# Session Handoff — 2026-03-15

## Summary

Marathon session closing **126 issues** (69 → 126, +57). All pushed to master.
Test count: 337,302 → 360,500+ (+23,200 new tests, zero regressions).
Total issues in database: 351 (21 new from research rounds). 51 ready to work.

## What was built (by subsystem)

### Harmonics (Epics 1-3 COMPLETE)
- `ScalarHarmonic` type with eigenvalues, conjugation, orthogonality
- `wigner3j`, `clebsch_gordan`, `harmonic_product` (Gaunt coefficients)
- `EvenVectorHarmonic`, `OddVectorHarmonic` with norms, divergence/curl eigenvalues
- `EvenTensorHarmonicY`, `EvenTensorHarmonicZ`, `OddTensorHarmonic` with traces, norms
- `LaplacianS2` operator with eigenvalue reduction
- `inner_product` / `vector_inner_product` / `tensor_inner_product` — full orthogonality
- `decompose_scalar`, `decompose_vector`, `decompose_symmetric_tensor` — field decomposition
- `angular_integral` — scalar/vector/tensor Gaunt integrals
- Files: `src/harmonics/{scalar_harmonics,clebsch_gordan,vector_harmonics,tensor_harmonics,orthogonality,laplacian,decompose_scalar,decompose_vector,decompose_tensor,angular_integrals}.jl`

### Constraints & CovD Sorting
- `set_tracefree!` + `enforce_tracefree` in simplify pipeline
- `set_divfree!` + `enforce_divfree` in simplify pipeline
- `sort_covds_to_box` — detect g^{ab}∇_a∇_b → □
- `sort_covds_to_div` — detect ∇_a V^a divergence patterns
- `symmetrize_covds` — ∇_a∇_b → ½(∇_a∇_b + ∇_b∇_a)
- Files: `src/algebra/contraction.jl` (extended), `src/gr/sort_covds.jl` (extended)

### AllContractions (Epic COMPLETE)
- `all_contractions(expr, metric; filter=true)` — enumerate (2n-1)!! contractions
- `filter_independent_contractions` — symmetry-aware dedup via canonicalization
- `contraction_ansatz([:Ric,:Ric], :g)` — most general scalar ansatz
- Files: `src/algebra/all_contractions.jl`

### Algebra-Valued Forms (Epic COMPLETE)
- `AlgValuedForm` type with algebra/degree tracking
- `alg_wedge`, `alg_exterior_d`, `connection_1form`, `curvature_2form`
- `gauge_covd` — D_A ω = dω + [A∧ω]
- `field_strength` (= curvature_2form), `yang_mills_eom`, `instanton_density`
- `chern_simons_form` — CS = Tr(A∧dA + ⅔A∧A∧A), `bianchi_identity`
- Files: `src/exterior/algebra_forms.jl`

### xIdeal — Exact Solution Identification
- `weyl_scalars(C, l, n, m, mbar)` — Newman-Penrose Ψ₀...Ψ₄
- `null_tetrad_from_metric`, `validate_null_tetrad`
- `petrov_invariants(Weyl, g)` — I, J from NP scalars
- `is_algebraically_special(I, J)`, `weyl_contraction_invariants`
- `petrov_classify(Weyl, g)` → PetrovType enum (I/II/III/D/N/O)
- `segre_classify(Ric, ginv)` → SegreType (eigenvalue structure)
- `check_energy_conditions(Ric, R, g, ginv)` → WEC/NEC/SEC/DEC
- Validation: Schwarzschild = Type D + Segre {(1111)} + K=48M²/r⁶
- Validation: FRW = Type O (conformally flat) + Segre {1,(111)}
- Files: `src/xideal/{weyl_scalars,petrov_invariants,petrov_classify,segre,energy_conditions}.jl`

### Covariant Phase Space (Epic COMPLETE — Wald formalism)
- `LagrangianDensity`, `EOMResult`, `eom_extract` — L → EOM
- `SymplecticPotential`, `symplectic_potential`, `theta_eh` — δL = EOM·δφ + dΘ
- `SymplecticCurrent`, `symplectic_current` — ω = δ₁Θ - δ₂Θ
- `NoetherCurrent`, `noether_current`, `noether_current_eh` — J = Θ(£_ξ) - ξ·L
- `is_divergence`, `extract_divergence`, `split_divergence` — divergence detection
- `NoetherCharge`, `noether_charge_eh` — Q^{ab} = ∇^a ξ^b - ∇^b ξ^a (Komar)
- `hamiltonian_variation`, `wald_entropy_integrand` — δH = ∫(δQ - ξ·Θ), S = 2πQ
- Validation: EH Komar integral, BH entropy S = A/4
- Files: `src/phase_space/{eom,symplectic,divergence,noether,potential,first_law}.jl`

### Scalar-Tensor Theory (xIST Epic COMPLETE)
- `HorndeskiTheory`, `define_horndeski!` — container + registration
- `horndeski_L2` through `horndeski_L5` — the 4 Lagrangians
- `horndeski_metric_eom`, `horndeski_scalar_eom` — field equations
- `BelliniSawickiAlphas`, `compute_alphas` — α_M, α_K, α_B, α_T, α_H
- `ScalarTensorQuadraticAction`, `quadratic_action_horndeski` — perturbations on FRW
- `tensor_sound_speed`, `scalar_sound_speed`, `stability_conditions`
- `BeyondHorndeskiTheory`, `beyond_horndeski_L4/L5`, `alpha_H`
- `DHOSTTheory`, `dhost_L1` through `dhost_L5` — DHOST class I
- Validation: f(R) as Horndeski subcase (α_T = 0 from GW170817)
- Files: `src/scalar_tensor/{horndeski,horndeski_eom,alpha_params,quadratic_action_st,beyond_horndeski,dhost}.jl`

### Spinors
- `define_spinor_bundles!` — SL2C/SL2C_dot VBundles with conjugation
- `spin_up/down`, `spin_dot_up/down`, `is_dotted`, `conjugate_index`
- `fresh_spinor_index`, `spinor_dummy_pairs`, `normalize_spinor_dummies`
- `define_spin_metric!` — ε_{AB} antisymmetric metric
- See-saw contraction sign tracking in `contract_metrics`
- xperm canonicalization extended for vbundle boundaries
- Spinor display: dotted notation (A' / \dot{A})
- Files: `src/spinors/{spinor_bundles,spinor_indices,spin_metric}.jl`, `src/show.jl` (extended), `src/algebra/canonicalize.jl` (extended), `src/algebra/contraction.jl` (extended)

### Tetrads
- `define_frame_bundle!` — :Lorentz VBundle for frame indices
- `frame_up/down`, `is_frame_index`, `fresh_frame_index`
- Frame metric η_{IJ}, delta, cache management
- Files: `src/tetrads/frame_bundle.jl`

### BH Perturbations
- `SchwarzschildPerturbation`, `decompose_schwarzschild` — RW/Zerilli decomposition
- `regge_wheeler_gauge`, `zerilli_gauge` — gauge fixing
- `MasterEquation`, `regge_wheeler_potential`, `zerilli_potential` — master equations
- `ModeCouplingTable`, `compute_coupling_table`, `coupling_coefficient`
- Files: `src/perturbation/{schwarzschild_decompose,master_equations,mode_coupling}.jl`

### EFT / Feynman Diagrams
- `TensorVertex`, `TensorPropagator`, `FeynmanDiagram`, `DiagramAmplitude`
- `vertex_from_perturbation`, `build_diagram`, `tree_exchange_diagram`, `contract_diagram`
- `gauge_fixing_condition/action`, `fp_operator`, `ghost_propagator/vertex`
- `graviton_3vertex`, `graviton_4vertex`, `graviton_vertex_n`
- Files: `src/feynman/{types,gauge_fixing,vertices}.jl`

### DDI (Dimensionally-Dependent Identities)
- `generalized_delta(up, down)` — antisymmetrized Kronecker product, vanishes for p>dim
- `generate_ddi_rules(dim; order)` — rank-2 DDI rules (Gauss-Bonnet in d=4)
- `gauss_bonnet_ddi()` — explicit GB identity
- `generate_riemann_ddi(dim, order)` — Riemann DDIs (cubic in d=4 from Fulling Table 1)
- `register_ddi_rules!(reg; dim)` — add to registry
- Files: `src/algebra/{generalized_delta,ddi_rules}.jl`

### Invar (Curvature Invariants)
- `RInv` type — contraction permutation representation
- Orbit-based canonicalization (not xperm — preserves contraction structure)
- `to_tensor_expr` / `from_tensor_expr` — bidirectional conversion
- Correctly distinguishes R², Ric², Kretschner at degree 2
- Files: `src/invariants/rinv.jl`

### Matter
- `PerfectFluid` + `stress_energy`, `trace_stress_energy`, `conservation_equation`
- Files: `src/matter/perfect_fluid.jl`

## Ground Truth Verification

### Equation citation corrections
All test files corrected against actual local PDFs (99 substitutions across 12 files):
- Martel-Poisson: "Eq 2.x" → "Eq 3.x" (harmonics are in Sec III, not II)
- Nakahara: "Eq 11.x" → "Eq 10.x" (gauge theory is Ch 10, not 11)
- DLMF: "34.3.7-9" → "34.3.8-10", "34.3.11-12" → "34.3.16-17"
- Nutma: "Sec 5.1.2" → "Sec 5.1.1", "Eq 60" → "cf. Sec 5.1.4"
- Fulling: "Sec 5" → "Sec 3"

### Reference library
`reference/ground_truth/` contains local copies of all cited papers:
- martel_poisson_2005.pdf, nutma_2014_xtras.pdf, fulling_1992.pdf
- iyer_wald_1994.pdf, kobayashi_2019_horndeski.pdf, eguchi_gilkey_hanson_1980.pdf
- wald_1984.djvu, nakahara_2003.pdf + full textbook library
- nist_dlmf_34.3.md (public domain snapshot)
- All copyrighted material `.gitignore`d

## Research Designs Completed (→ beads issue chains)

| Epic | Issues created | Key reference |
|------|---------------|---------------|
| Petrov classification | 8 (TGR-1t9.2-1t9.9) | Stephani et al. (2003) |
| Covariant phase space | 9 (TGR-s50.2-s50.10) | Iyer & Wald (1994) |
| Horndeski theory | 6 (TGR-ble.2-ble.7) | Kobayashi (2019) |
| Spinor indices | 7 (TGR-oun etc.) | Penrose & Rindler Vol 1 |
| BH pert2 mode coupling | 7 (TGR-5i0 etc.) | Brizuela et al. (2006) |
| EFT Feynman diagrams | 7 (TGR-ec6 etc.) | Goldberger & Rothstein (2006) |
| DDI generation | 5 (TGR-xlu.2-xlu.6) | Fulling et al. (1992) |
| PPN framework | 7 (TGR-bgl.9-bgl.15) | Will (2014) |
| Tetrad/frame basis | 7 (TGR-wgy etc.) | Chandrasekhar (1983) |
| Invar canonical forms | 7 (TGR-443.1.1-443.1.7) | Zakhary & McIntosh (1997) |

## What's ready next (51 issues)

### High-impact implementation issues ready now:
- TGR-1t9.9: pp-wave = Petrov Type N validation (completes xIdeal validation chain)
- TGR-vzx: Graviton propagator in harmonic gauge (unblocks Feynman contraction engine)
- TGR-9vn: Matter-graviton coupling vertices
- TGR-d42: Tensor contraction engine for Feynman diagrams (blocked by vzx+9vn+qks — qks done)
- TGR-ozc: Soldering form σ^a_{AA'} (spinor-tensor bridge)
- TGR-96h: Tetrad struct + define_tetrad! API
- TGR-lej: Abstract tetrad indices in AST
- TGR-443.1.2: RInv canonicalization via xperm symmetry group
- TGR-443.1.3: DualRInv with Levi-Civita
- TGR-xlu.5: DDI pipeline integration into simplify
- TGR-2wf: DHOST degeneracy conditions
- TGR-bgl.2: PPN metric ansatz
- TGR-bgl.13: PPN-to-component bridge
- TGR-u19: BH-pert2 radial source assembly
- TGR-tcj: Multi-scalar Horndeski
- TGR-5f2: EFT of dark energy alpha parametrization

### Remaining research issues (8):
- TGR-34t.1: Bianchi cosmology decomposition design
- TGR-4zw.1: Multi-field PSALTer design
- TGR-vdm.1: Dirac-Bergmann constraint analysis design
- TGR-wq0.1: Bimetric framework design
- TGR-swh.1: Metric-affine gravity design
- TGR-xmm.1: Index-free notation design
- TGR-dai.1: Clifford algebra design
- TGR-0ny: SpinIndex type (may be superseded by VBundle approach)

## Key patterns for next agent

1. **Worktree isolation**: All implementation uses `isolation: worktree`. Merge conflicts are always in `test/runtests.jl` and `src/TensorGR.jl` (include/export lines) — resolve by keeping both sides.

2. **Beads daemon**: Auto-pushes via `bd: backup` commits. Must `git pull --rebase` before pushing.

3. **Ground truth**: Every test MUST cite actual equation numbers from local PDFs in `reference/ground_truth/`. Previous agents hallucinated equation numbers.

4. **Test pattern**: New features go in `src/<subsystem>/file.jl`, tests in `test/test_file.jl`, include in `src/TensorGR.jl`, test include at end of `test/runtests.jl`.

5. **Registry pattern**: New tensor types register via `register_tensor!`, new VBundles via `define_vbundle!`, metrics via `metric_cache`/`delta_cache`.
