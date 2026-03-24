# HANDOFF — 2026-03-24 (Session 11)

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

- **489 of 528 issues closed** (69 closed this session: 2 new code + 66 audit + 1 epic)
- **Full test suite: 373,486 tests, ALL PASS** (1 known @test_skip in test_euler_density.jl:480)
- All pushed to `master`, no uncommitted work
- `bd stats` for live counts, `bd ready` for available work
- Beads recovered from git history and pushed to `beads-backup` branch

---

## What Was Done This Session (69 issues closed)

### Beads Recovery & Audit
- **Beads database recovered**: Old Dolt database was deleted by commit `fae1a9c`. Restored 350 issues from git history backup (commit `f61b529`) + 178 from current `issues.jsonl`. Total: 528 issues.
- **Beads sync fixed**: `bd init --from-jsonl` with type-fixed JSONL (int→bool for crystallizes/ephemeral/etc, int→string for comment IDs).
- **Backup pushed**: `bd backup export-git` to `beads-backup` branch on origin.
- **Full audit of 98 open issues**: Systematically verified each against codebase + git history. Closed 66 stale issues that were already implemented but never marked done in beads.

### New Code: RInv Independent Basis Enumeration (TGR-443.1.6)
- **`src/invariants/enumerate.jl`** (~230 LOC)
  - `enumerate_independent_rinvs(degree; level)`: returns `(canonical, independent, relations)` named tuple
  - Database-backed: uses `degree2/3/4_canonical_rinvs()` + Level 2 Bianchi relations
  - `enumerate_live_canonical_rinvs(degree)`: algorithmic cross-validation path
    - Generates all (4k-1)!! perfect matchings of 4k Riemann slots
    - Canonicalizes each via xperm, deduplicates
    - Filters vanishing invariants (antisymmetric slot pairings)
  - Ground truth verified: degree 2 (4→3), degree 3 (13→8), degree 4 (57→26)
  - 2,570 tests
- **Closed TGR-443 (Invar Epic 1)**: all children now complete

### New Code: Anholonomy Coefficients (TGR-2d4.4)
- **`src/tetrads/anholonomy.jl`** (~160 LOC)
  - `define_anholonomy!(reg, tetrad_name)`: registers c^I_{JK} with `AntiSymmetric(2,3)` on `:Lorentz` VBundle
  - `anholonomy_expr(tetrad, I, J, K)`: builds TDeriv expression c^I_{JK} = e^I_a (e^b_J ∂_b e^a_K - e^b_K ∂_b e^a_J)
  - Proper Tangent/Lorentz index separation via `fresh_index()`
  - `has_anholonomy()`, `get_anholonomy_name()` for lookup
  - 37 tests: registration, error handling, expression structure, antisymmetry verification, abstract tensor usage
- **TGR-2d4.4 closed**, Tetrad epic (TGR-2d4) has 4 remaining children

---

## Key Decisions / Lessons

### Carried from previous sessions
- **FullySymmetric(n)** takes slot numbers as varargs: `FullySymmetric(1,2,3,4)` NOT `FullySymmetric(4)`
- **make_rule** RETURNS rules but does NOT register them
- **symmetrize** takes `Vector{Symbol}` not `Vector{TIndex}`
- **xperm convention for canonical_perm_ext**: Renato notation. Generators SLOT-SPACE for right-coset.
- **No parallel agents/Julia**: cache conflicts + OOM on WSL2.
- **AntiSymmetric fields**: `.i` and `.j`, NOT `.slot1`/`.slot2`

### New this session
- **Beads issues.jsonl is source of truth**: Lives at `.beads/issues.jsonl` in git. Use `bd init --from-jsonl` to bootstrap. Use `bd export -o .beads/issues.jsonl` to save. Use `bd backup export-git` to push to `beads-backup` branch.
- **Beads JSONL schema changes between versions**: Fields `crystallizes`, `ephemeral`, `is_template`, `no_history`, `pinned` must be bool not int. Field `waiters` must be `[]` not `""`. Comment `id` must be string not int.
- **RInv Level 2 relations don't contain ALL canonical forms**: Independent forms that don't appear in any Bianchi relation (like R², Ric²) are missing from relations. Must use named accessor functions (`degree2_canonical_rinvs()` etc.) as source of truth for canonical forms.
- **Perfect matching vanishing filter**: Not all vanishing pairings pair adjacent antisymmetric slots directly — must also filter the zero canonical form (`all(==(0), canon.contraction)`) after canonicalization.

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

**P2 (Tetrad/xCoba — 4 remaining):**
- TGR-2d4.5: Ricci rotation coefficients (depends on anholonomy, just completed)
- TGR-2d4.6: ToBasis for tetrad frames
- TGR-2d4.7: ChangeBasis between frame choices
- TGR-2d4.8: Curvature in tetrad frame

**P2 (BH-Pert2 — 7 issues, all genuinely open):**
- TGR-68g: Second-order source terms from first-order products
- TGR-2yl: Second-order Regge-Wheeler equation with source
- TGR-31k: Second-order Zerilli equation with source
- TGR-22h: Second-order gauge-invariant master equations
- TGR-u19: Radial source term assembly
- TGR-2y0: Brizuela et al validation
- TGR-2gv: Second-order GW energy flux

**P2 (Infrastructure):**
- TGR-byb: BinaryBuilder for xperm.c
- TGR-erv: Pkg registration

**P3 (Fermions — 3 remaining):**
- TGR-2jh.3: Dirac field with kinetic term
- TGR-2jh.4: Covariant derivative on spinors (spin connection)
- TGR-2jh.5: Stress-energy for Dirac field

**P3 (BRST — 3 remaining):**
- TGR-655.5: BRST differential
- TGR-655.6: Ghost number grading
- TGR-655.7: BRST nilpotency validation

**P3 (Index-Free — 2 remaining):**
- TGR-xmm.3: ToIndexFree conversion
- TGR-xmm.4: FromIndexFree conversion

**Epics still open:**
- Tetrad/xCoba (4 of 6 children remain)
- Fermion Fields (3 of 5 children remain)
- Gauge/BRST (3 of 7 children remain)
- Index-Free Notation (2 of 4 children remain)
- BH-Pert2 (all 7 genuinely open)

**Research/Infrastructure:**
- TensorGR.jl-6e8: Collect all papers using xAct (full corpus download)
  - Spec'd out with 7 search sources, download pipeline, dedup logic
  - Scale estimate: ~1,355 unique papers on INSPIRE (union of all xAct subpackage fulltext searches), ~1,500-2,000 total including Scholar/ADS/theses
  - Core paper citation counts: xPerm (215), xPert (246), xTras (280)
  - Estimated ~3-4 GB of PDFs
  - Uses playwright-cli + INSPIRE/ADS/Semantic Scholar APIs + ArXiv bulk download

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
- Degree-2 invariants: 4 canonical, 3 independent (Fulling 1992)
- Degree-3 invariants: 13 canonical, 8 independent (Fulling 1992)
- Degree-4 invariants: 57 canonical, 26 independent, 31 Bianchi relations
- Anholonomy: c^I_{JK} = -c^I_{KJ} (antisymmetric, from Lie bracket)
- Coordinate basis: c^I_{JK} = 0 (holonomic)

## Quick Commands

```bash
bd ready                    # see available work
bd stats                    # project health
bd export -o .beads/issues.jsonl  # save issues to git-tracked file
bd backup export-git        # push backup to beads-backup branch
julia --project -e 'using Pkg; Pkg.test()'  # full test suite
git log --oneline -10       # recent commits
```
