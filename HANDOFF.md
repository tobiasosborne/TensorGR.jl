# HANDOFF — 2026-03-19 Session

## DO NOT DELETE THIS FILE. Read it completely before working.

## TOBIAS'S RULES — FOLLOW TO THE LETTER

These rules were given explicitly by Tobias. They override any other guidance.

**Rule 1:** Previous work and handoff is never to be completely trusted. Generations of agents have committed heresies.

**Rule 2:** Goal is to fix the current bugs. Nothing else is currently in scope.

**Rule 3:** These bugs are DEEP and COMPLEX. Do NOT underestimate this task.

**Rule 4:** Use smart subagents to get a full overview of all the project. There is deep interlock between the current bugs, probably at the module contract level as well as line-by-line level.

**Rule 5:** Liberally clean-room suspect modules to INDEPENDENT smart subagents.

**Rule 6:** No more than 2-3 subagents at a time.

**Rule 7:** DO NOT UNDERESTIMATE THIS TASK. Robust solutions, NO BANDAIDS. Prefer to work longer for a more solid solution, even at the expense of downstream work.

**Rule 8:** Workflow for code changes: always spawn two independent subagents to propose a codebase change. Review and take best solution. THEN spawn a rigorous reviewer to IMMEDIATELY review the work. Do not deviate.

**Rule 9:** Repeat rules when asked.

**Corollary of Rule 1:** Tests may be suspect. Only trustworthy source is physics ground truth. Numbers of terms are not necessarily physics ground truth.

**Additional:**
- BEFORE making any changes, review xAct source (stored locally) to understand how the problem is solved there. Changing contraction.jl without this has led to "pain and misery."
- Tests take ages — run selectively or in background only.
- Document regularly / checkpoint in case of premature termination.
- Move slowly and carefully via cleanroom testing and auditing of subagent proposals.

**The Extended 9 Rules (from memory/feedback_session_rules.md):**
1. **SKEPTICISM**: All subagent work, handoffs — verify everything twice.
2. **DEEP BUGS**: Deep, complex, interlocked. Do not underestimate.
3. **NO BANDAIDS**: Best-practices full solutions only.
4. **WORKFLOW**: 3 subagents before any core code change (xAct research + 2 solutions).
5. **REVIEW**: Rigorous reviewer agent after every core change. No exceptions.
6. **GROUND TRUTH**: Physics is ground truth, not pinned numbers.
7. **TESTING**: Targeted only, or full suite in background.
8. **REPEAT RULES**: Repeat occasionally to maintain focus.
9. **DO NOT UNDERESTIMATE**: This is deeply nontrivial.

---

## Current State (2026-03-19)

- **355,558 tests passing, 0 failed, 0 errored, 0 broken**
- **131 of 357 issues closed** (81 closed this session, up from 50)
- **226 open**, 45 ready to work, 181 blocked
- All pushed to `master` on remote
- Full test suite last verified: 7m11s, clean

---

## What Was Done This Session (2026-03-19)

### All march15-preserve Subsystems Ported

TGR-t28 (the meta-merge issue) is **CLOSED**. All 10 subsystems ported:

| Subsystem | Directory | Issues | Tests | Source Lines |
|-----------|-----------|:---:|:---:|:---:|
| Spinors (display, see-saw, canon, soldering, macro) | `src/spinors/` | 5 | 165 | 576 |
| Frame bundle (tetrads) | `src/tetrads/` | 1 | 49 | 100 |
| xIdeal (Petrov/Segre/energy conditions) | `src/xideal/` | 10 | 228 | 873 |
| Scalar-tensor (Horndeski/DHOST/EFT-DE) | `src/scalar_tensor/` | 12 | ~400 | 2,995 |
| Harmonics (scalar/vector/tensor) | `src/harmonics/` | 8 | ~16,000 | 1,821 |
| Phase space (Noether/symplectic/Wald) | `src/phase_space/` | 10 | 134 | 1,458 |
| DDI (dimensionally dependent identities) | `src/algebra/ddi_rules.jl` | 4 | 92 | 789 |
| RInv/DualRInv (Invar canonical forms) | `src/invariants/` | 6 | 324 | 965 |
| Feynman (vertices/propagators/diagrams) | `src/feynman/` | 6 | 59 | 1,556 |
| PPN (metric ansatz) | `src/ppn/` | 4 | 133 | 611 |

### Bug Fixes (14 broken -> 0 broken)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| TGR-55f (9 broken) | DHOST degeneracy: `horndeski_as_dhost` didn't encode a1=-a2 constraint | Store G4_X on a1/a2 via `options[:dhost_coeff_expr]`, add `_sym_add` x+(-x)=0 cancellation, fix `_is_vanishing` to check `.vanishing` field |
| TGR-4aw (5 broken) | RInv xperm canonicalization: xperm solves LEFT-ACTION, RInv needs CONJUGATION | Delegate `canonicalize_rinv` to BFS orbit enumeration (`canonicalize(::RInv)`) |

### Key Technical Findings

1. **xperm already works for spinor indices** — symmetry generators operate on slot numbers within a single tensor, so no `canonicalize.jl` changes needed
2. **xperm CANNOT solve RInv conjugation canonicalization** — fundamentally different group-theoretic problem (left-action vs conjugation orbit). BFS is correct for degree <= 6
3. **`_sym_add` x+(-x)=0 cancellation** — reviewer-approved, safe, only fires on unary-minus Expr nodes
4. **`_is_vanishing` was dead code** — old rule-scanning path never matched because `set_vanishing!` creates Function-pattern rules, not Tensor-pattern rules
5. **Include ordering matters** — phase space has circular struct refs, scalar-tensor has forward refs. Document the correct orders in TensorGR.jl

---

## What Remains

### Ready Issues by Category

#### Self-contained new functionality (low risk)
- **TGR-bgl.13** [P2] xPPN: PPN-to-component bridge
- **TGR-d42** [P2] EFTofPNG: tensor contraction engine for Feynman diagrams
- Various research/design tasks

#### Pipeline-touching issues (REQUIRE 3-AGENT PROTOCOL)
- **TGR-e04** [P2] Investigate removing `_avoid` to recover 303-term R³ simplification
  - `_avoid::Set{Symbol}` in perturbation engine prevents dummy collisions but breaks memoization
  - Kernel extraction depends on `_avoid` — removing it breaks 19 numerical tests
  - This is the MOST DANGEROUS remaining issue
- **TGR-avk** [P2] Constraints: automatic trace-free enforcement in simplify pipeline
- **TGR-ulo.2** [P2] SortCovDsToDiv enhancement (CLAUDE.md says stub is intentional)
- **TGR-xlu.5** [P2] DDI simplification pass in simplify pipeline

### The march15-preserve Branch — Exhausted for Safe Ports

Remaining diffs require **struct-level changes** to core types:
- `VBundleProperties`: `options::Dict` -> `conjugate_bundle::Union{Nothing,Symbol}`
- `TensorProperties`: add `tracefree_pairs`, `divfree_indices` fields
- `TensorRegistry`: add `tetrads::Dict` field

These are needed for the full tetrad engine but require the 3-agent protocol.

---

## Physics Ground Truth (ONLY trustworthy reference)

- K_FP: spin2=2.5k², spin0s=-k², spin1=0, spin0w=0
- K_R²: spin2=0, spin0s=3k⁴, spin1=0, spin0w=0
- K_Ric²: spin2=1.25k⁴, spin0s=k⁴, spin1=0, spin0w=0
- spin-1 and spin-0w MUST be zero for ALL kernels (diffeomorphism invariance)
- On MSS: R₀ = 4Λ, Ric₀_{ab} = Λg_{ab}, Riem₀_{abcd} = (Λ/3)(g_{ac}g_{bd} - g_{ad}g_{bc})
- Horndeski DHOST: a1 = G4_X, a2 = -G4_X, a3=a4=a5=0 (Langlois & Noui 2016)
- RInv canonicalization is a CONJUGATION problem, not a left-action problem

## Quick Commands

```bash
bd ready                    # see available work
bd stats                    # project health (131 closed / 357 total)
julia --project -e 'using Pkg; Pkg.test()'  # full test suite (~7min)
git log --oneline -10       # recent commits
```

## Session Close Protocol

```
[ ] git status
[ ] git add <files>
[ ] git commit -m "..."
[ ] git push
[ ] bd close <completed issues>
```
