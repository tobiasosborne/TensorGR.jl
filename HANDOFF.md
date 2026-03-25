# HANDOFF — 2026-03-25 (Session 14)

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
**NO PARALLEL JULIA**: NEVER run two Julia processes simultaneously on WSL2 — cache conflicts and OOM. Check `ps aux | grep julia` before ANY julia command. This includes benchmarks while tests run in background. DO NOT use `while` loops polling for julia processes — they spawn extra shells.

---

## Current State

- **530 of 568 issues closed** (3 closed this session)
- **Full test suite: 375,396 tests, ALL PASS** (1 known @test_skip in test_euler_density.jl:480)
- **Benchmarks: 445 pass, 0 fail, 3 broken (stretch goals)** — run last session, not re-run this session
- All pushed to `master`, no uncommitted work
- `bd stats` for live counts, `bd ready` for available work

---

## What Was Done This Session (3 issues closed)

### 1. TensorGR.jl-2on (P1 Bug): Parallel/serial inconsistency in simplify pipeline

The `canonicalize_terms=false` optimization from session 13 (commit 1b6502d) was only applied to the serial `collect_terms` path. The parallel `_collect_terms_parallel` still always re-canonicalized.

**Fix**: Added `canonicalize_terms::Bool=true` kwarg to `_collect_terms_parallel`, matching the `collect_terms` API. Pipeline passes `false` to both paths.

- Location: `src/algebra/simplify.jl`, lines 330-360 and 504
- 3 proposers + 2 reviewers (all opus, all PASS)
- 375,351 tests + 445 benchmarks pass

### 2. TensorGR.jl-05y (P2 Feature): evaluate_components — abstract-to-component evaluation

**New file**: `src/components/evaluate.jl` (~350 lines)

`evaluate_components(expr, chart, values)` fully evaluates abstract TensorExpr trees to numeric CTensor arrays. Handles:
- Free index expansion (Cartesian product over chart dimension)
- Dummy index summation (Einstein convention) at expression level
- Per-term internal contractions in TSum (each term's dummies summed independently)
- Partial derivative evaluation via `deriv_fn` callback or pre-computed values
- Deterministic output axis ordering via `indices()` traversal order

Also adds `prepare_values(chart, metric_data)` convenience helper.

**Key design decisions**:
- Hybrid expression-level summation reusing existing `_replace_index` + `_evaluate_component`
- Dummy names computed from ORIGINAL expression before free index replacement (otherwise `:_1` duplicates are falsely detected as dummies)
- Uses `indices(expr)` for axis ordering, NOT `free_indices(expr)` (Dict iteration is non-deterministic in Julia 1.12)
- Value lookup is position-agnostic: `T^{ab}` and `T_{ab}` look up same key. Users should `contract_metrics` at abstract level first.
- Reviewer 1 caught TSum per-term dummy bug → fixed with `_eval_term_with_local_dummies`

- Location: `src/components/evaluate.jl` (new), `src/TensorGR.jl` (+2 lines), `test/test_evaluate_components.jl` (new, 47 tests)
- 3 proposers + 2 reviewers (Reviewer 1 FAIL→fixed→PASS, Reviewer 2 PASS)
- 375,398 tests pass (47 new)
- **Unblocks**: TensorGR.jl-77q (CCovD) and TensorGR.jl-irx (chart transitions)

### 3. TensorGR.jl-88e (P2 Bug): Order-dependent TProduct/TSum rule unification

`_unify(::TProduct, ::TProduct)` used positional zip — pattern `A*B` would not match expression `B*A`. `make_rule` did not canonicalize the pattern.

**Fix**: Grouped backtracking matcher:
1. Group factors by `(type, name, rank)` key
2. Single-element groups: direct match (fast path, O(1))
3. Multi-element groups: backtracking permutation search within group
4. `_merge_bindings!` ensures consistency across groups

Same fix applied to `_unify(::TSum, ::TSum)`.

**Critical edge case**: Shared pattern variables across same-name factors (e.g., `T_{a_,b_} * T_{b_,c_}` matching `T_{z,y} * T_{y,x}`). Proposer 3 proved that simple sorting fails here but backtracking finds the valid alignment.

- Location: `src/rules.jl`, lines 105-210 (replaced ~10 lines with ~100 lines)
- 3 proposers + 1 reviewer (PASS). Reviewer 2 interrupted by session end — **should be run next session**
- 375,396 tests pass
- **Note**: No new dedicated test file was added for order-independent matching. The fix was verified with an ad-hoc REPL test (10 tests pass). A proper test should be added.

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
- **Pinned term counts are NOT ground truth**: physics correctness is what matters
- **Code review agents can be WRONG about physics**: verify against textbooks

### New this session
- **`free_indices()` uses Dict iteration → non-deterministic in Julia 1.12**: Use `indices(expr)` for deterministic ordering, then filter to free indices by name set.
- **Dummy detection after free-index replacement is WRONG**: After replacing free indices `:a` → `:_1`, `:b` → `:_1`, the duplicate `:_1` entries are falsely detected as dummy pairs. Always compute dummies from the ORIGINAL expression.
- **Sort-based rule matching fails with shared pattern variables**: Pattern `T_{a_,b_} * T_{b_,c_}` vs `T_{z,y} * T_{y,x}` — sorting aligns factors such that `b_` gets conflicting bindings. Backtracking within same-name groups is the correct approach.
- **`while` loops polling for Julia processes**: These can spawn extra shell processes and cause confusion. Avoid them — just run tests in foreground or use background tasks with notifications.

---

## ⚠ Core Changes To Monitor

**Commit bf27dab** (`canonicalize_terms` in `_collect_terms_parallel`):
- Location: `src/algebra/simplify.jl`, lines 330-360
- Change: Added `canonicalize_terms::Bool=true` kwarg, pipeline passes `false`
- Risk: Low — matches serial path behavior

**Commit 293c59c** (`evaluate_components`):
- Location: `src/components/evaluate.jl` (new file)
- Change: Full abstract-to-component evaluation pipeline
- Risk: Low — purely additive, no existing code modified

**Commit 3bb1ad3** (order-independent rule matching):
- Location: `src/rules.jl`, lines 105-210
- Change: Grouped backtracking in `_unify(::TProduct/TSum)`
- Risk: Medium — core pattern matching infrastructure
- **Only 1 of 2 reviewers completed** — run Reviewer 2 next session
- Revert: Restore the simple `zip`-based `_unify` if rule matching regresses

---

## TODO Next Session

1. **Run Reviewer 2 for TensorGR.jl-88e** (rule matching fix) — was interrupted
2. **Add dedicated tests for order-independent rule matching** — currently only verified via REPL
3. **Run benchmarks** — not run this session after the rule fix
4. Continue with ready queue (`bd ready`)

## Ready Queue

```bash
bd ready    # see available work
bd stats    # project health
```

**28 ready issues** after closing 3 this session. Key items:
- TensorGR.jl-77q (CCovD) — NOW UNBLOCKED by evaluate_components
- TensorGR.jl-irx (chart transitions) — NOW UNBLOCKED
- TensorGR.jl-lj1 (global registry sync) — P2 bug, blocks 1 other
- TensorGR.jl-7cb (parametric derivatives) — P2 feature
- 8 P2 test coverage issues
- 3 P2 infrastructure (BinaryBuilder, Pkg registration, papers corpus)

---

## Physics Ground Truth

### Carried forward
- K_FP: spin2=2.5k², spin0s=-k², spin1=0, spin0w=0
- K_R²: spin2=0, spin0s=3k⁴, spin1=0, spin0w=0
- K_Ric²: spin2=1.25k⁴, spin0s=k⁴, spin1=0, spin0w=0
- Spin-1 and spin-0w MUST be zero for ALL kernels (diffeomorphism invariance)
- Contracted Bianchi: ∇^a G_{ab} = 0, ∇^a R_{ab} = (1/2)∇_b R (Wald eq 3.2.17)
- Wald entropy: S = -2π ∫_H (∂L/∂R_{abcd}) ε_{ab}ε_{cd} → A/4G for EH (Iyer-Wald 1994)

## Quick Commands

```bash
bd ready                    # see available work
bd stats                    # project health
bd blocked                  # see blocked issues
julia --project -e 'using Pkg; Pkg.test()'  # full test suite (~375k tests)
julia -t4 --project=benchmarks benchmarks/run_all.jl --tier 3  # all benchmarks
git log --oneline -15       # recent commits
```
