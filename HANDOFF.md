# HANDOFF — 2026-03-21 (Session 9)

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

**Corollary**: Review xAct source (at `reference/xAct/`) BEFORE changing core modules.
**Max 2-3 subagents at a time.** Checkpoint regularly.
**USE MAX THINKING (opus) for all subagents.** Medium effort missed a bimetric sign bug last session.
**WSL2 MEMORY**: Never enumerate large combinatorial sets in memory. Use streaming/chunked processing.

---

## Current State

- **298 of 355 issues closed** (37 closed this session + 3 new issues created)
- **Full test suite: 370,224+ tests, ALL PASS** (verified this session)
- All pushed to `master`, no uncommitted work
- `bd stats` for live counts, `bd ready` for available work

---

## What Was Done This Session (37 issues closed, 8 epics completed)

### Invar Pipeline (Epic 2, 3, 6 — ALL CLOSED)
- **riemann_simplify**: Top-level 6-level orchestrator (21 tests)
- **Invar database**: Complete infrastructure + data for degrees 2-7
  - Degree 2: 4 canonical, 3 independent, 1 Bianchi relation (134 tests)
  - Degree 3: 13 canonical, 8 independent, 5 Bianchi relations (695 tests)
  - Degree 4: 57 canonical, 26 independent, ALL 31 Bianchi relations computed via numerical SVD
  - Degrees 5-7: Counts verified (75/409/2247 independent) from Garcia-Parrado & Martin-Garcia 2007
  - Dual invariants: degrees 2-5, Pontryagin density independent
  - Differential invariants: orders 4 and 6 (10 entries total)
  - **inv_simplify**: Database-driven fast-path lookup with fallback
  - **xAct parser**: Mathematica Invar.m parser, cross-check confirms ALL degrees 2-7 match
  - **Generation script**: Memory-safe streaming enumeration with --verify-orbits mode
  - **Fast canonicalization**: xperm TensorExpr round-trip (~17% speedup for degree 4)
- Validation: Gauss-Bonnet, degree-2 independence, degree-3 independence (8 invariants), Weyl completeness

### Invar Pipeline (Epics 4, 5 — IN PROGRESS)
- **TInvar design doc**: Proposes TRInv partial involution + xperm canonical_perm_ext
- **SymManipulator design doc**: Proposes SymH type hierarchy, 5-phase plan
- **SymH type implemented**: MonotermSym, MultitermSym, riemann_symh(), n_independent_components (68 tests)

### Spatial Spinors (SU(2) / Loop Quantum Gravity)
- **define_space_spinors!**: SU(2) VBundle, eps_space, tau soldering form (69 tests)
- **Sen connection**: Gamma_sen, F_sen curvature, metricity/compatibility rules (58 tests)
- **Ashtekar-Barbero variables**: A^i_a connection, E^a_i densitized triad, F^i_{ab} curvature, Gauss constraint (67 tests)
- **Space spinors design doc**: Full design including Ashtekar variables

### Hamiltonian Analysis (Epic CLOSED)
- **classify_constraints**: First-class vs second-class with DOF formula (30 tests)
- **DOF counting**: DOFSummary, dof_count, dof_summary pipeline (54 tests)
- **GR validation**: 2 DOF verified via full pipeline (24 tests)
- **Proca validation**: 3 DOF, Maxwell comparison, Stückelberg (23 tests)

### Other Features
- **invariant_lagrangian**: Most general Lagrangian at orders 0/2/4 with Gauss-Bonnet DDI (47 tests)
- **Metric-affine validation**: Levi-Civita limit (32 tests), Einstein-Cartan (38 tests)
- **Fermion design doc**: Recommends is_grassmann registry flag + Grassmann-aware sort
- **GradedTensor**: register_grassmann_field!, grassmann_parity, grassmann_sign (24 tests)
- **Invar database design doc**: Recommends lazy-loaded Julia source files (Option E)

### Design Documents Created
- `docs/design/invar_database_design.md` — Database format comparison (DuckDB vs Julia source)
- `docs/design/tinvar_design.md` — Tensorial invariant canonicalization
- `docs/design/symmanipulator_design.md` — SymH type hierarchy (966 lines)
- `docs/design/space_spinors_design.md` — SU(2) spatial spinors + Ashtekar
- `docs/design/fermion_design.md` — Grassmann algebra + fermion field types

---

## Key Decisions / Lessons

### Carried from previous sessions
- **FullySymmetric(n)** takes slot numbers as varargs: `FullySymmetric(1,2,3,4)` NOT `FullySymmetric(4)`
- **make_rule** RETURNS rules but does NOT register them
- **symmetrize** takes `Vector{Symbol}` not `Vector{TIndex}`

### New this session
- **RInv BFS orbit canonicalization** is too slow for degree ≥ 4 (~0.1s per involution). The xperm TensorExpr round-trip (to_tensor_expr → canonicalize → from_tensor_expr) provides modest speedup but doesn't solve the fundamental conjugation problem. For degree 4, full enumeration (2M involutions) takes ~55 hours via BFS. Solved by computing Bianchi relations directly on known canonical forms via numerical SVD instead.
- **xAct doesn't solve conjugation either**: It uses standard left-action canonicalization via ToCanonical[]. The conjugation problem σ→g·σ·g⁻¹ is fundamentally different from xperm's left-action g·σ.
- **WSL2 memory**: Never store millions of items in memory. Use streaming enumeration (generate → process → discard).
- **Integralis uses DuckDB** for integral storage, but invariant relations are frozen math — static Julia source files are simpler and have zero dependencies.
- **Garcia-Parrado & Martin-Garcia 2007 Table 1** is the ground truth for canonical form counts. MaxIndex in Invar.m counts NON-PRODUCT forms only.
- **Degree-4 Bianchi relations**: All 31 computed via numerical SVD at d=8 with random Bianchi-satisfying Riemann tensors. Rank verified = 26. All coefficients are clean rationals.

---

## Ready Queue

```bash
bd ready    # see available work
bd stats    # project health
```

**P2 (implementation):**
- TGR-4zb.3: SymManipulator: SymH canonicalization
- TGR-5lp.2: TInvar: tensorial Riemann monomial canonicalization
- TGR-u19: BH-Pert2: radial source assembly (check deps exist first!)

**P3 (implementation):**
- TGR-2jh.3: Dirac field with kinetic term
- TGR-bm6.1: Schwarzschild 2+2 decomposition

**Epics still open:**
- Invar Epic 4 (TInvar) — design doc ready
- Invar Epic 5 (SymManipulator) — SymH type done, canonicalization next
- FullSimplification (xTras)
- Tetrad/xCoba
- Fermion Fields — GradedTensor done, Dirac field next
- Harmonics Epic 4 (RW/Zerilli)
- Index-Free Notation

---

## Physics Ground Truth

- K_FP: spin2=2.5k², spin0s=-k², spin1=0, spin0w=0
- K_R²: spin2=0, spin0s=3k⁴, spin1=0, spin0w=0
- K_Ric²: spin2=1.25k⁴, spin0s=k⁴, spin1=0, spin0w=0
- Spin-1 and spin-0w MUST be zero for ALL kernels (diffeomorphism invariance)
- PPN scalar-tensor: gamma=(omega+1)/(omega+2), beta=1+Psi*omega'/(4(2omega+3)(omega+2)^2)
- NP Schwarzschild: Ψ₂ = -M/r³, ρ=+1/r (our convention)
- EIH 1PN: L_EIH coefficients from Goldberger-Rothstein Eq 40
- Riemann d=4: 20 independent components (verified via SymH n_independent_components)
- GR: 2 propagating DOF (verified via full Hamiltonian pipeline)
- Proca: 3 propagating DOF (0 first-class, 2 second-class)
- Degree-4 invariants: 57 canonical, 26 independent, 31 Bianchi relations (all computed)

## Quick Commands

```bash
bd ready                    # see available work
bd stats                    # project health
julia --project -e 'using Pkg; Pkg.test()'  # full test suite
git log --oneline -10       # recent commits
```
