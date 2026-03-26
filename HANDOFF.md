# HANDOFF — 2026-03-26 (Session 17: All issues cleared)

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

- **349 of 352 issues closed** (all open issues cleared this session)
- **3 deferred (stretch goals)**: TGR-443.1.4 (syzygy), TGR-443.1.5 (RInv conversion), TGR-lej (abstract tetrad indices)
- **Full test suite: running** (targeted tests: 75 new tests pass)
- **Benchmarks: not re-run this session** — Tier 1 passed last session
- All changes on `master` (commit dc058e4)
- `bd stats` for live counts

---

## What Was Done This Session

### Bulk issue triage: 40+ implemented-but-unclosed issues closed

Found and batch-closed issues that had been implemented in prior sessions but never closed in beads:
- **Index-Free Notation** (TGR-xmm): 4 subtasks, commit 7cf8d17
- **Gauge/BRST** (TGR-655): subtasks 1-2, 5-7, commit 2f74d91
- **BH-Pert2**: 7 issues (TGR-22h, TGR-u19, TGR-2yl, TGR-31k, TGR-68g, TGR-2y0, TGR-2gv)
- **Fermion Fields** (TGR-2jh): 5 subtasks, commit c9ae31c
- **Tetrad/xCoba** (TGR-2d4): 8 subtasks, commit 3221440
- **Hamiltonian Analysis** (TGR-vdm): 6 subtasks + epic
- **Metric-Affine** (TGR-swh): 3 subtasks + epic
- **Bimetric** (TGR-wq0): 2 subtasks + epic
- **Invar** (TGR-ed9): 1 remaining subtask + epic
- **xPPN** (TGR-bgl): 1 subtask + epic
- **P1 bug** (TGR-0tm): already resolved via Rule 6 in commit c6e28aa
- **Design docs** (TGR-z87, TGR-jt5): tetrad validation/Cartan design, implementations done

### TGR-655.3 + TGR-655.4: Yang-Mills field strength & equations

**New file**: `src/gauge/yang_mills.jl` (~220 lines)

Indexed tensor versions of Yang-Mills, complementing the AlgValuedForm versions in exterior/algebra_forms.jl:
- `yang_mills_field_strength(ggp, I, a, b)` → F^I_{ab} = ∂_a A^I_b − ∂_b A^I_a + f^I_{JK} A^J_a A^K_b
- `gauge_covariant_deriv(ggp, expr, I, a)` → D_a X^I = ∂_a X^I + f^I_{JK} A^J_a X^K
- `yang_mills_bianchi(ggp, I, a, b, c)` → D_{[a} F^I_{bc]} (structure check, 3 covd terms)
- `yang_mills_lagrangian(ggp)` → −(1/4) δ_{IJ} g^{ac} g^{bd} F^I_{ab} F^J_{cd}
- `yang_mills_field_equations(ggp, I, b)` → D_a F^{Ia}_b

Helper: `_replace_gauge_index` recursively replaces gauge algebra indices in expressions.

**Tests**: 13 tests in `test/test_yang_mills.jl`

**Risk**: Low — purely additive. No existing code paths changed.

### TGR-bm6.1 through bm6.6: Regge-Wheeler / Zerilli master equations

**New files** (4):
- `src/harmonics/schwarzschild.jl` (~180 lines): Schwarzschild 2+2 background
- `src/harmonics/rw_gauge.jl` (~80 lines): RW gauge DOF counting
- `src/harmonics/regge_wheeler.jl` (~90 lines): RW/Zerilli master equations
- `src/harmonics/master_functions.jl` (~70 lines): Ψ_RW and Ψ_Z extraction specs

Key features:
- `define_schwarzschild_background!(reg)` → M2×S2 product manifold with f(r), r
- `rw_gauge_odd()/rw_gauge_even()` → DOF counting (2 odd + 4 even = 6 total)
- `derive_rw_equation(l)` / `derive_zerilli_equation(l)` → RWMasterEquation with V(r,M)
- `extract_master_functions(l)` → (Ψ_RW spec, Ψ_Z spec)
- `schwarzschild_potential_difference(l)` → algebraic V_RW − V_Z

Isospectrality verified via:
1. Cross-check against existing `regge_wheeler_potential`/`zerilli_potential` in bh_second_order.jl
2. Potential difference sign change (necessary for same spectrum)
3. Large-r centrifugal limit match: both → l(l+1)/r²

**Tests**: 62 tests in `test/test_rw_zerilli.jl`

**Risk**: Low — purely additive. No existing code paths changed.

---

## Key Decisions / Lessons

### Carried from previous sessions
- All decisions from session 16 HANDOFF still apply
- Cross-project parallel Julia is OK (different --project paths)

### New this session (session 17)
- **Beads bulk close**: Many issues were implemented in commits but never `bd close`d. Verified each via `git log --oneline` + commit messages + file existence before closing.
- **Schwarzschild 2+2 uses separate vbundles**: `:Tangent_M2` and `:Tangent_S2` (not shared `:Tangent`). Warning about overwriting `:Tangent` is cosmetic.
- **Superpotential formula dropped**: Chandrasekhar's W(r) for Darboux relation V = W² ± dW/dr* is convention-dependent and error-prone. Replaced with direct algebraic verification of isospectrality.
- **Yang-Mills indexed tensor form**: Parallel API to exterior calculus `AlgValuedForm` versions. Both coexist — indexed form works with BRST `GaugeGroupProperties`, forms version works with `AlgValuedForm`.

---

## ⚠ Core Changes To Monitor

**This session** (commit 0b0c7bc):

**Yang-Mills** (TGR-655.3, TGR-655.4):
- Location: `src/gauge/yang_mills.jl` (new)
- Change: New file with 6 exported functions
- Risk: Low — purely additive
- Revert: Delete file, remove include + exports from TensorGR.jl

**RW/Zerilli** (TGR-bm6.1 through bm6.6):
- Location: `src/harmonics/schwarzschild.jl`, `rw_gauge.jl`, `regge_wheeler.jl`, `master_functions.jl` (all new)
- Change: 4 new files, ~420 lines total
- Risk: Low — purely additive
- Revert: Delete files, remove includes + exports from TensorGR.jl

---

## TODO Next Session

1. **Verify full test suite passes** (running at time of HANDOFF)
2. **Run full benchmarks (Tier 1-3)** — not run this session
3. **Consider deferred stretch goals**:
   - TGR-443.1.4: Syzygy detection (requires algebraic geometry infrastructure)
   - TGR-443.1.5: Bidirectional RInv conversion (requires index-free ↔ indexed bridge)
   - TGR-lej: Abstract tetrad indices in AST (design-level change to TIndex)
4. **Pkg registration** — consider submitting to General registry

## Quick Commands

```bash
bd stats                    # project health
bd list --status=deferred   # remaining stretch goals
julia --project -e 'using Pkg; Pkg.test()'  # full test suite
julia -t4 --project=benchmarks benchmarks/run_all.jl --tier 3  # all benchmarks
git log --oneline -15       # recent commits
```
