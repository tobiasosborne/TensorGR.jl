# HANDOFF — 2026-03-25 (Session 12)

## DO NOT DELETE THIS FILE. Read it completely before working.

## TOBIAS'S RULES — FOLLOW TO THE LETTER

1. **SKEPTICISM**: All subagent work, handoffs — verify everything twice.
2. **DEEP BUGS**: Deep, complex, interlocked. Do not underestimate.
3. **NO BANDAIDS**: Best-practices full solutions only.
4. **WORKFLOW**: 3 subagents before any core code change (xAct research + 2 solutions).
5. **REVIEW**: Rigorous reviewer agent after every core change. No exceptions.
6. **GROUND TRUTH**: Physics is ground truth, not pinned numbers. Tests may be suspect.
7. **TESTING**: Targeted only, or full suite in background.
8. **REPEAT RULES**: Repeat occasionally to maintain focus.
9. **DO NOT UNDERESTIMATE**: This is deeply nontrivial.
10. **NO PARALLEL AGENTS**: Julia precompilation cache conflicts. Run agents sequentially only.

**Corollary**: Review xAct source (at `reference/xAct/`) BEFORE changing core modules.
**Max 2-3 subagents at a time (sequential).** Checkpoint regularly.
**USE MAX THINKING (opus) for all subagents.** Medium effort missed a bimetric sign bug (session 8) AND a generator conjugation bug (session 10).
**WSL2 MEMORY**: Never enumerate large combinatorial sets in memory. Use streaming/chunked processing.
**NO PARALLEL JULIA**: Never run two Julia processes simultaneously on WSL2 — cache conflicts and OOM.

---

## Current State

- **513 of 529 issues closed** (24 closed this session, 6 epics completed)
- **Full test suite: 375,174 tests, ALL PASS** (1 known @test_skip in test_euler_density.jl:480)
- All pushed to `master`, no uncommitted work
- `bd stats` for live counts, `bd ready` for available work
- Only 6 open issues remain (infrastructure + external deps + research)

---

## What Was Done This Session (24 issues closed, 6 epics completed)

### Epic 1: Tetrad/xCoba (TGR-2d4) — COMPLETE (5 issues closed)
- **`src/tetrads/ricci_rotation.jl`** (~165 LOC)
  - `define_ricci_rotation!`, `ricci_rotation_expr`: γ^I_{JK} from anholonomy
  - Formula: γ^I_{JK} = ½(c^I_{JK} + η^{IM}η_{KN}c^N_{MJ} - η^{IM}η_{JN}c^N_{KM})
  - Reviewer caught critical sign bug in Term 3 (c^N_{MK} → c^N_{KM}), fixed before merge
- **`src/tetrads/to_frame.jl`** (~120 LOC)
  - `to_frame(expr, tetrad)`: projects coordinate → frame basis via tetrad insertion
  - `from_frame(expr, tetrad)`: inverse projection
- **`src/tetrads/change_frame.jl`** (~140 LOC)
  - `change_frame(expr, from, to)`: Lorentz transformation between tetrad choices
  - `define_frame_transformation!`: registers Λ^I_J transition matrix
- **`src/tetrads/frame_curvature.jl`** (~185 LOC)
  - `frame_riemann_expr`: R^I_{JKL} = e^I_a e^b_J e^c_K e^d_L R^a_{bcd}
  - `frame_riemann_structure_expr`: Cartan structure equation form
  - `frame_ricci_expr`, `frame_ricci_scalar_expr`
- 153 tests

### Epic 2: Fermion Fields (TGR-2jh) — COMPLETE (4 issues closed)
- **`src/fermions/dirac.jl`** (~200 LOC)
  - `define_fermion!(reg, :psi; type=:dirac)`: registers ψ and ψ_bar with Grassmann parity
  - `dirac_bar`, `scalar_bilinear`, `vector_bilinear`, `axial_bilinear`, `pseudo_bilinear`
  - `dirac_kinetic_expr`: L = iψ̄γ^a∂_aψ - mψ̄ψ
  - `dirac_equation_expr`: (iγ^a∂_a - m)ψ = 0
  - Supports Dirac, Majorana, Weyl types
- **`src/fermions/spin_connection.jl`** (~215 LOC)
  - `define_spin_connection!`: registers ω_a^{IJ} (antisymmetric in I,J)
  - `spin_connection_expr`: ω_a^{IJ} = e^K_a η^{JN} γ^I_{NK} from Ricci rotation
  - `dirac_covd_expr`: ∇_aψ = ∂_aψ + (1/4)ω_a^{IJ}γ_Iγ_Jψ
  - `dirac_bar_covd_expr`: ∇_aψ̄ = ∂_aψ̄ - (1/4)ω_a^{IJ}ψ̄γ_Iγ_J (minus sign!)
- **`src/fermions/stress_energy.jl`** (~165 LOC)
  - Belinfante T^{ab} = (i/4)[ψ̄γ^a∇^bψ + ψ̄γ^b∇^aψ - (∇^aψ̄)γ^bψ - (∇^bψ̄)γ^aψ]
  - On-shell trace: T^a_a = mψ̄ψ
- 148 tests

### Epic 3: Gauge/BRST (TGR-655) — COMPLETE (4 issues closed)
- **`src/gauge/brst.jl`** (~300 LOC)
  - `define_gauge_group!`: registers VBundle, structure constants f^I_{JK}, gauge field A^I_a, ghost c^I (gh=+1), anti-ghost c̄^I (gh=-1), NL field B^I
  - BRST rules: s(A) = D_ac, s(c) = -(1/2)fcc, s(c̄) = B, s(B) = 0
  - `ghost_number(expr)`, `filter_by_ghost_number(sum, n)`
  - Nilpotency: s²(c̄) = s(B) = 0 verified
- 75 tests

### Epic 4: Index-Free Notation (TGR-xmm) — COMPLETE (3 issues closed)
- **`src/algebra/index_free.jl`** (~300 LOC)
  - `IndexFree` struct: contraction topology without explicit index names
  - `to_index_free(expr)`: extracts tensor names, slot vbundles, contraction pairs
  - `from_index_free(ifree)`: reconstructs indexed form with fresh dummies
  - `index_free_structure`, `same_tensor_structure`
- 69 tests

### Epic 5: BH-Pert2 — ALL 7 ISSUES COMPLETE
- **`src/perturbation/bh_second_order.jl`** (~500 LOC)
  - `second_order_einstein_source(mp, a, b)`: δ²G_{ab} on vacuum background
  - `source_is_bilinear(mp)`: verifies every term is quadratic in h
  - `source_coupling_modes(l, m, lmax)`: enumerates all (l₁,m₁,l₂,m₂) pairs
  - `scalar/vector/tensor_coupling_coefficient`: angular coupling via Gaunt integrals
  - `regge_wheeler_potential`, `zerilli_potential`: known closed-form potentials
  - `MasterEquation`, `regge_wheeler_equation(l)`, `zerilli_equation(l)`
  - `tortoise_coordinate`, `inverse_tortoise`: r* coordinate system
- **`src/perturbation/bh_source_assembly.jl`** (~350 LOC)
  - `SecondOrderSource`, `SourceContribution`: structured source representation
  - `assemble_source(l, m, parity, lmax)`: full mode coupling assembly
  - `SourcedMasterEquation`, `second_order_rw/zerilli(l, m, lmax)`
  - `GaugeInvariantVariable`: second-order gauge-invariant master variable
  - `EnergyFluxFormula`, `quadrupole_flux_scaling`, `second_order_flux_scaling`
- 1,575 tests (including Brizuela et al. validation to 1e-13)

### Symmetry Ansatz (TGR-293h)
- **`src/gr/symmetry_reduce.jl`** (~200 LOC)
  - `symmetry_reduce(SphericalSymmetry)`: Schwarzschild-type, 2 free functions of r
  - `symmetry_reduce(StaticSymmetry)`: time-independent, 7 free functions
  - `symmetry_reduce(HomogeneousIsotropy)`: FLRW, 1 free function, k=0,±1
  - `symmetry_reduce(AxialSymmetry)`: Lewis-Papapetrou, 5 free functions
  - `MetricAnsatzResult`, `independent_components`, `constrained_components`
- 48 tests

---

## Key Decisions / Lessons

### Carried from previous sessions
- **FullySymmetric(n)** takes slot numbers as varargs: `FullySymmetric(1,2,3,4)` NOT `FullySymmetric(4)`
- **make_rule** RETURNS rules but does NOT register them
- **symmetrize** takes `Vector{Symbol}` not `Vector{TIndex}`
- **xperm convention for canonical_perm_ext**: Renato notation. Generators SLOT-SPACE for right-coset.
- **No parallel agents/Julia**: cache conflicts + OOM on WSL2.
- **AntiSymmetric fields**: `.i` and `.j`, NOT `.slot1`/`.slot2`
- **Beads issues.jsonl is source of truth**: `.beads/issues.jsonl` in git.
- **RInv BFS orbit canonicalization** is too slow for degree >= 4
- **WSL2 memory**: Never store millions of items in memory

### New this session

- **Reviewer agents catch real bugs**: The Ricci rotation reviewer found a critical index order bug in Term 3 (c^N_{MK} vs c^N_{KM} — antisymmetric indices, flips sign). Always run reviewer before commit.
- **Zerilli potential formula**: The issue description had an extra 1/r² factor. The correct formula (Chandrasekhar 1983, Brizuela 2009) has NO 1/r² prefactor: V_Z = f · [...] / [r³(λr+3M)²]. Both RW and Zerilli approach l(l+1)/r² at large r (isospectral).
- **Cross-vbundle cancellation limitation**: The simplifier cannot verify algebraic identities that require recognizing dummy relabeling through tetrad insertions (e.g., R^I_{JKL} + R^I_{JLK} = 0 via Riemann antisymmetry projected through e). Tests should check structural properties instead.
- **Weyl fields don't auto-register conjugate**: Only `:dirac` and `:majorana` types register the `_bar` conjugate. Weyl types (`:weyl_left`, `:weyl_right`) skip conjugate registration to avoid dangling references.
- **TScalar(:im) is the imaginary unit convention**: Used in gamma5() and Dirac bilinears, matches existing codebase pattern.
- **Grassmann parity in options Dict**: `is_grassmann` is stored in `options[:is_grassmann]`, NOT as a hot-path boolean on TensorProperties (design doc recommends adding it later, but current implementation works fine via Dict lookup).
- **set_vanishing! + simplify cannot fully reduce**: `∂(0)` terms don't automatically simplify to zero. Use structural verification (e.g., `source_is_bilinear`) instead of trying to simplify to TScalar(0).
- **Gaunt integral C^{0,0}_{2,0,2,0} = 1/(2√π)**: Not sqrt(5/(4π))·(3j)² as one might naively write — the prefactor is sqrt(25/(4π)) = 5/(2√π) because it includes both (2l+1) factors.

---

## Ready Queue

```bash
bd ready    # see available work
bd stats    # project health
```

**Only 6 open issues remain. All are infrastructure, external dependencies, or research:**

**P2 (Infrastructure):**
- TGR-byb: BinaryBuilder for xperm.c — needs BinaryBuilder.jl recipe, CI setup
- TGR-erv: Pkg registration — needs General registry PR, UUID, version bump

**P2 (Research):**
- TensorGR.jl-6e8: Collect all papers using xAct (full corpus download)
  - Spec'd out: ~1,355 papers, ~3-4 GB PDFs, playwright-cli + INSPIRE/ADS APIs

**P3 (External Dependencies):**
- TGR-61p: Geodesic equation ODE integration — needs DifferentialEquations.jl weak dep
- TGR-dhp: TOV equation solver — needs DifferentialEquations.jl weak dep
- TGR-1kw: Submanifolds/boundaries — labeled RESEARCH, underspecified

**In Progress (pre-existing from prior session):**
- TGR-0tm (P1 bug): bench_12 regression — cubic dS invariants produce ~2x expected term counts

**No more epics remain open.** All 6 completed epics:
- Tetrad/xCoba (TGR-2d4): 6/6 children closed
- Fermion Fields (TGR-2jh): 5/5 children closed
- Gauge/BRST (TGR-655): 7/7 children closed
- Index-Free Notation (TGR-xmm): 4/4 children closed
- BH-Pert2: all 7 issues closed (no explicit epic issue)
- Invar Epic 1 (TGR-443): all children closed (session 11)

---

## New Source Files Added This Session

```
src/tetrads/ricci_rotation.jl      # Ricci rotation coefficients γ^I_{JK}
src/tetrads/to_frame.jl            # to_frame / from_frame (tetrad projection)
src/tetrads/change_frame.jl        # change_frame (Lorentz transformation)
src/tetrads/frame_curvature.jl     # Frame Riemann, Ricci, scalar
src/fermions/dirac.jl              # Dirac field, conjugate, bilinears, kinetic term
src/fermions/spin_connection.jl    # Spin connection ω_a^{IJ}, Dirac covariant derivative
src/fermions/stress_energy.jl      # Belinfante stress-energy for Dirac field
src/gauge/brst.jl                  # BRST differential, ghost number, gauge group
src/algebra/index_free.jl          # IndexFree type, to/from conversion
src/perturbation/bh_second_order.jl    # δ²G, RW/Zerilli potentials, mode coupling
src/perturbation/bh_source_assembly.jl # Source assembly, master equations, energy flux
src/gr/symmetry_reduce.jl          # Symmetry-reduced metric ansatz generation
```

New test files: 12 (matching the source files above).

---

## Physics Ground Truth

### Carried forward
- K_FP: spin2=2.5k², spin0s=-k², spin1=0, spin0w=0
- K_R²: spin2=0, spin0s=3k⁴, spin1=0, spin0w=0
- K_Ric²: spin2=1.25k⁴, spin0s=k⁴, spin1=0, spin0w=0
- Spin-1 and spin-0w MUST be zero for ALL kernels (diffeomorphism invariance)
- PPN scalar-tensor: gamma=(omega+1)/(omega+2), beta=1+Psi*omega'/(4(2omega+3)(omega+2)^2)
- NP Schwarzschild: Ψ₂ = -M/r³, ρ=+1/r (our convention)
- Riemann d=4: 20 independent components
- GR: 2 propagating DOF; Proca: 3 propagating DOF
- Degree-2 invariants: 4 canonical, 3 independent (Fulling 1992)
- Degree-3 invariants: 13 canonical, 8 independent
- Degree-4 invariants: 57 canonical, 26 independent, 31 Bianchi relations

### New this session
- Ricci rotation: γ_{IJK} = -γ_{JIK} (antisymmetric in first two when lowered)
- RW potential: V_RW = (1-2M/r)[l(l+1)/r² - 6M/r³], positive for r > 2M, l >= 2
- Zerilli potential: V_Z = f·[2λ²(λ+1)r³ + 6λ²Mr² + 18λM²r + 18M³]/[r³(λr+3M)²], NO 1/r² prefactor
- Both V_RW and V_Z → l(l+1)/r² at large r (isospectral at leading order)
- Both vanish at horizon f(2M) = 0
- Gaunt coupling C^{0,0}_{2,0,2,0} = 1/(2√π) ≈ 0.28209 (verified to 1e-13)
- Quadrupole self-coupling 2×2: sources l = 0, 2, 4 (not 1, 3, 5)
- Energy flux scaling: dE¹/dt ∝ η², dE²/dt ∝ η³, ratio ∝ η
- Dirac stress-energy trace: T^a_a = mψ̄ψ (massive), T^a_a = 0 (massless/conformal)
- BRST: s² = 0 on all fields (requires Jacobi identity for ghost self-coupling)
- Ghost numbers: A=0, c=+1, c̄=-1, B=0; s increases ghost number by 1

## Quick Commands

```bash
bd ready                    # see available work (6 issues)
bd stats                    # project health (513/529 closed)
bd export -o .beads/issues.jsonl  # save issues to git-tracked file
bd backup export-git        # push backup to beads-backup branch
julia --project -e 'using Pkg; Pkg.test()'  # full test suite (~375k tests)
git log --oneline -10       # recent commits
```
