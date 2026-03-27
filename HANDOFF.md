# HANDOFF — 2026-03-27 (Session 17: Issue cleanup, Yang-Mills, RW/Zerilli, full green)

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
**NO PARALLEL JULIA**: NEVER run two Julia processes simultaneously within the same project on WSL2 — cache conflicts and OOM. Cross-project parallel is OK. Check `ps aux | grep julia` before ANY julia command. DO NOT use `while` loops polling for julia processes — they spawn extra shells.

---

## Current State

- **526 of 529 issues closed** (3 remaining open)
- **Full test suite: 375,695 tests, ALL PASS** (1 known Broken in test_euler_density.jl:480)
- **Benchmarks: Tier 1-3 ALL PASS** (445 pass, 3 broken stretch goals)
- All changes pushed to `master` (commit d09ba45)
- `bd stats` for live counts

**IMPORTANT**: 177 issues were missing from Dolt DB (stale JSONL vs Dolt desync). Imported via `bd import` this session. Total issues now 529 (was showing 352 before import).

---

## 3 Remaining Open Issues

| ID | P | Title | Notes |
|----|---|-------|-------|
| `TGR-byb` | P2 | BinaryBuilder for xperm.c | Yggdrasil recipe for cross-platform binaries. Blocks Pkg registration. |
| `TGR-erv` | P2 | Pkg registration | Submit to Julia General registry. Requires BinaryBuilder or deps/build.jl. Tobias wants to think about it. |
| `TensorGR.jl-6e8` | P2 | Collect xAct papers corpus | Research task: download all papers using xAct. Old prefix (pre-rename). |

---

## What Was Done This Session

### 1. Bulk issue triage: 40+ implemented-but-unclosed issues closed

Found and batch-closed issues implemented in prior sessions but never `bd close`d:
- **10 epics cleared**: Index-Free, BRST (partial), BH-Pert2, Fermion Fields, Tetrad, Hamiltonian, Metric-Affine, Bimetric, Invar, xPPN
- **3 deferred "stretch goals" found already implemented**: Syzygy detection (simplify_levels.jl), RInv conversion (to/from_tensor_expr), Tetrad indices (VBundle :Lorentz)
- **Submanifolds/boundaries** (TGR-1kw): already fully implemented with 111 tests
- **Symmetry-reduced ansatz** (TGR-293h): implemented in commit 79d3a28

### 2. Database repair: 177 issues imported from JSONL to Dolt

Beads Dolt DB had 352 issues but JSONL had 529. Imported the missing 177 (172 closed + 5 open) via `bd import`. Database now complete.

### 3. TGR-655.3 + TGR-655.4: Yang-Mills field strength & equations

**New file**: `src/gauge/yang_mills.jl` (~220 lines)

Indexed tensor Yang-Mills (complements AlgValuedForm versions in exterior/algebra_forms.jl):
- `yang_mills_field_strength(ggp, I, a, b)` → F^I_{ab}
- `gauge_covariant_deriv(ggp, expr, I, a)` → D_a X^I
- `yang_mills_bianchi(ggp, I, a, b, c)` → D_{[a} F^I_{bc]}
- `yang_mills_lagrangian(ggp)` → −(1/4) F^I_{ab} F_I^{ab}
- `yang_mills_field_equations(ggp, I, b)` → D_a F^{Ia}_b

**Tests**: 13 tests in `test/test_yang_mills.jl`

### 4. TGR-bm6.1–6: Regge-Wheeler / Zerilli master equations

**New files** (4, ~420 lines total):
- `src/harmonics/schwarzschild.jl`: Schwarzschild 2+2 (M2 × S2) background
- `src/harmonics/rw_gauge.jl`: RW gauge DOF counting
- `src/harmonics/regge_wheeler.jl`: RW/Zerilli master equations with potentials
- `src/harmonics/master_functions.jl`: Ψ_RW and Ψ_Z extraction specs

Isospectrality verified via cross-check with bh_second_order.jl, sign-change test, large-r centrifugal limit.

**Tests**: 62 tests in `test/test_rw_zerilli.jl`

### 5. Missing test coverage filled

- **12 tests** for order-independent rule matching (TGR-88e, pending since session 14)
- **11 tests** for symbolic manifold dimensions (session 15 gap)

### 6. Benchmark ground truth updated

Updated 2 pinned term counts (26→14) per Rule 6: improved canonicalization produces fewer terms with correct physics. All 445 benchmarks green (Tier 1-3).

---

## Key Decisions / Lessons

### Carried from previous sessions
- All decisions from session 16 HANDOFF still apply
- Cross-project parallel Julia is OK (different --project paths)

### New this session (session 17)
- **Beads JSONL ↔ Dolt desync**: The JSONL (git-tracked) and Dolt DB (live) can diverge. Always check both. Use `bd import` to repair.
- **Old prefix issues**: `TensorGR.jl-6e8` uses old prefix, invisible to `bd` until imported.
- **Schwarzschild 2+2 uses separate vbundles**: `:Tangent_M2` and `:Tangent_S2`. Warning about overwriting `:Tangent` is cosmetic.
- **Superpotential formula dropped**: Chandrasekhar's W(r) Darboux relation is convention-dependent. Used direct algebraic verification instead.
- **Yang-Mills indexed form**: Parallel API to exterior calculus forms. Both coexist.
- **Nested `using` in Julia 1.12**: `using TensorGR:` inside nested `@testset` blocks causes "syntax: using expression not at top level". Move all imports to the outermost `@testset` block.
- **Pkg registration**: Tobias is considering but not ready. Main blocker: xperm.c cross-platform (BinaryBuilder or deps/build.jl). License concern: xperm.c is GPL, package is Apache-2.0.

---

## ⚠ Core Changes To Monitor

**This session** (commits 0b0c7bc through dc857ee):

**Yang-Mills** (TGR-655.3, TGR-655.4):
- Location: `src/gauge/yang_mills.jl` (new)
- Risk: Low — purely additive
- Revert: Delete file, remove include + exports from TensorGR.jl

**RW/Zerilli** (TGR-bm6.1 through bm6.6):
- Location: `src/harmonics/{schwarzschild,rw_gauge,regge_wheeler,master_functions}.jl` (all new)
- Risk: Low — purely additive
- Revert: Delete files, remove includes + exports from TensorGR.jl

**Benchmark ground truth** (bench_05, bench_07):
- Location: `benchmarks/ground_truth.jl`
- Change: SCHWARZ_D1RIC_SIMPLIFIED_TERMS 26→14, DS_D1RIEM_SIMPLIFIED_TERMS 26→14
- Risk: None — physics unchanged, simplifier produces fewer terms

**Test additions** (test_rules.jl, test_registry.jl):
- 23 new tests for order-independent matching + symbolic dims
- Risk: None — purely additive

---

## What Was Done Later This Session (REPL + Chaos Monkey + Wald)

### REPL tensor mode (src/repl/tensor_mode.jl)
- Press `\` to enter `tensor>` prompt, type LaTeX, get Unicode output
- Commands: simplify, canon, contract, expand, latex, indices, terms
- Name resolution: R(4)→Riem, R(2)→Ric, R(0)→RicScalar, G(2)→Ein, C(4)→Weyl
- Registry context: `init_repl_mode!(reg)` binds to a registry
- MIME dispatch: `text/plain` → Unicode, `text/latex` → LaTeX (Jupyter/Pluto)
- Tests: 36 tests in test_repl_mode.jl

### Chaos Monkey epic (TGR-kq0y) — 7 subtasks, all closed
- 121 tests in test_chaos_monkey.jl, zero crashes
- Random bytes, typo fuzzing, clipboard dumps, code injection, Unicode, stress test
- Parameterized: `chaos_monkey(n=500, seed=42)` — deterministic, CI-ready

### Wald textbook verification
- 37 tests in test_wald_textbook.jl (Ch 3,4,6,7,10, App C)
- Tutorial: docs/src/wald_verification.md (what simplifies automatically vs needs Level 2)

## TODO Next Session

1. **`deps/build.jl`** for cross-platform xperm.c compilation (enables Pkg registration)
2. **xAct papers corpus** (TensorGR.jl-6e8) — if desired
3. **GPL/Apache-2.0 license review** — xperm.c is GPL, rest is Apache-2.0
4. **REPL tab-completion** of tensor names from registry
5. **Einstein trace rule** and **Weyl trace-free rule** — would make G^a_a=-R and g^{ac}C_{abcd}=0 work in simplify

## Quick Commands

```bash
bd stats                    # project health (529 total, 3 open)
bd list --status=open       # remaining open issues
julia --project -e 'using Pkg; Pkg.test()'  # full test suite (~375k tests)
julia -t4 --project=benchmarks benchmarks/run_all.jl --tier 3  # all benchmarks
git log --oneline -15       # recent commits
```
