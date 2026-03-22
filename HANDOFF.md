# HANDOFF — 2026-03-22 (Session 10)

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

---

## Current State

- **~306 of 355 issues closed** (8 closed this session: 4 new code + 4 already-done)
- **Full test suite: 370,465 tests, ALL PASS** (verified this session)
- All pushed to `master`, no uncommitted work
- `bd stats` for live counts, `bd ready` for available work

---

## What Was Done This Session (8 issues closed)

### TInvar Pipeline (Invar Epic 4 — 3 issues closed)
- **TGR-5lp.2: TRInv struct + canonicalization** (~300 LOC in `src/invariants/trinv.jl`)
  - `TRInv` type: partial involution for tensorial Riemann monomials (free indices = fixed points)
  - Canonicalization via `xperm_canonical_perm_ext` with proper free/dummy separation
  - Right-coset P·S algorithm with slot-space generators (Renato convention)
  - Inter-factor exchange via bubble-sort generators (adjacent transpositions)
  - to/from TensorExpr, to/from RInv conversions
  - **Critical bug caught during review**: generator conjugation was WRONG for `canonical_perm_ext` direct calls. The existing `_canonicalize_product` conjugates because it calls `xperm_canonical_perm` (which internally inverts the perm). Direct `_ext` calls need unconjugated slot-space generators for right-coset computation. Medium thinking would have missed this.
  - 45 tests
- **TGR-5lp.3: First Bianchi cyclic reduction**
  - `bianchi_cyclic_trinv`: applies R_{a[bcd]}=0 to TRInv by conjugating contraction by cyclic permutation π·σ·π⁻¹
  - `bianchi_relations_trinv`: generates all Bianchi linear relations among canonical TRInvs
  - 12 tests
- **TGR-5lp.4: Second Bianchi differential reduction**
  - `apply_bianchi2_tensorial`: applies ∇_{[a}R_{bc]de}=0 to TDeriv-wrapped Riemann factors
  - `has_diff_riemann`, `diff_riemann_factor_indices` for detecting differential monomials
  - Works at TensorExpr level (TDeriv nodes), not TRInv level
  - 18 tests

### SymManipulator (Invar Epic 5 — 2 issues closed)
- **TGR-4zb.2**: Already implemented in session 9 (closed as done)
- **TGR-4zb.3: SymH canonicalization** (~150 LOC added to `src/invariants/symh.jl`)
  - `canonicalize_symh`: monoterm via xperm + multi-term reduction stub
  - `symmetrize_symh`: Young projector P = (1/|G|) Σ s·σ(expr)
  - `verify_symh`: checks monoterm symmetries hold for a tensor
  - 6 tests (74 total SymH tests)

### Already-Done Issues (3 closed)
- **TGR-adi**: Space spinors (already implemented session 9)
- **TGR-3up**: Sen connection (already implemented session 9)
- **TGR-x9t**: Ashtekar-Barbero variables (already implemented session 9)

---

## Key Decisions / Lessons

### Carried from previous sessions
- **FullySymmetric(n)** takes slot numbers as varargs: `FullySymmetric(1,2,3,4)` NOT `FullySymmetric(4)`
- **make_rule** RETURNS rules but does NOT register them
- **symmetrize** takes `Vector{Symbol}` not `Vector{TIndex}`

### New this session
- **xperm convention for canonical_perm_ext**: Perm is in Renato notation (slot→name). Generators must be SLOT-SPACE (not conjugated) for right-coset P·S computation. The existing `_canonicalize_product` uses a DIFFERENT code path (canonical_perm wrapper which inverts internally), so its conjugation pattern does NOT apply to direct `_ext` calls. This is a deep convention trap.
- **No parallel agents**: Julia precompilation cache conflicts on WSL2 cause failures when multiple agents run Julia simultaneously. Always run agents sequentially.
- **Multi-term symmetries (Bianchi) can't be verified at TensorExpr level**: The simplify pipeline doesn't know about multi-term identities. Verification requires component computation or the Invar pipeline.
- **TensorProperties constructor**: Uses keyword args: `TensorProperties(name=:T, manifold=:M4, rank=(0,2), symmetries=[...])`. NOT positional args.
- **Many session-9 issues were not properly closed in beads**: Space spinors, Sen connection, Ashtekar variables, and SymH type were all implemented but left open.

### Previous sessions (still relevant)
- **RInv BFS orbit canonicalization** is too slow for degree ≥ 4
- **WSL2 memory**: Never store millions of items in memory
- **Garcia-Parrado & Martin-Garcia 2007 Table 1** is the ground truth for canonical form counts
- **Degree-4 Bianchi relations**: All 31 computed via numerical SVD

---

## Ready Queue

```bash
bd ready    # see available work
bd stats    # project health
```

**P2 (implementation):**
- TGR-5lp.5: TInvar: tensorial DDI reduction (dimension-dependent, architecturally heavier)
- TGR-4zb.4: SymManipulator: SymH arithmetic
- TGR-u19: BH-Pert2: radial source assembly (check deps exist first!)
- TGR-ah2: Full Invar parity: exhaustive canonical forms degrees 4-7

**P3 (implementation):**
- TGR-2jh.1: Research: fermion field types
- TGR-bm6.1: Schwarzschild 2+2 decomposition

**Epics still open:**
- Invar Epic 4 (TInvar) — TRInv + Bianchi done, DDI next
- Invar Epic 5 (SymManipulator) — SymH type + canonicalization done, arithmetic next
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
