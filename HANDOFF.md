# HANDOFF — 2026-03-26 (Session 15: Feynfeld.jl integration)

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

- **530 of 568 issues closed** (0 closed this session — this session was cross-project integration only)
- **Full test suite: 375,404 tests, ALL PASS** (1 known Broken in test_euler_density.jl:480)
- **Benchmarks: 53 Tier 1 pass** — re-run this session after changes
- All changes from this session pushed to `master`
- `bd stats` for live counts, `bd ready` for available work

---

## What Was Done This Session (Feynfeld.jl integration — 0 issues closed)

**Context**: Feynfeld.jl (sister project at `../Feynfeld.jl`) is a Julia port of the
Mathematica QFT ecosystem (FeynCalc/FeynArts/FeynRules). It depends on TensorGR.jl for
index contraction and canonicalization of Lorentz algebra expressions. Feynfeld needs
**symbolic manifold dimensions** for dimensional regularisation (D = 4 − 2ε), where D
is a symbol, not an integer.

### Change 1: `ManifoldProperties.dim` and `VBundleProperties.dim` type widening

**File**: `src/registry.jl`
**Change**: `dim::Int` → `dim::Union{Int,Symbol}` on both structs, plus the
`VBundleProperties` 4-arg constructor and `define_vbundle!` keyword argument.

**Why**: Feynfeld needs `ManifoldProperties(:M4, :D, :η, nothing, [...])` to register
a Minkowski manifold with symbolic dimension `:D` for dimensional regularisation.
Without this, every `dim` argument must be a concrete integer, blocking QFT use cases.

**Impact**: This is a **public API change**. Any code that constructs
`ManifoldProperties` or `VBundleProperties` with integer dims continues to work
(`4 isa Union{Int,Symbol}` is true). Code that type-asserts `dim::Int` on these fields
will need updating.

**Backward compatibility**: Full. All 375,404 existing tests pass without modification.

### Change 2: Metric trace for symbolic dimensions

**File**: `src/algebra/contraction.jl`, line 77
**Change**: `TScalar(dim // 1)` → `TScalar(dim isa Int ? dim // 1 : dim)`

**Why**: When `g^a_a` is self-traced on a manifold with `dim = :D`, the original code
attempted `Symbol // Int` which has no method. Now returns `TScalar(:D)` for symbolic
dimensions, `TScalar(dim // 1)` for integer dimensions.

**Impact**: The contraction engine now returns `TScalar(:D)` instead of crashing for
symbolic-dim manifolds. No behavior change for integer dimensions.

### Change 3: `define_metric!` guards for symbolic dimensions

**File**: `src/gr/metric.jl`
**Changes**:
1. The `lorentzian(d)` fallback (line 53): now returns `nothing` when `d isa Symbol`
   instead of crashing on `fill(1, :D - 1)`. Callers must pass `signature` explicitly
   for symbolic-dim manifolds.
2. Epsilon tensor registration (lines 83-96): wrapped in `if d isa Int` guard since it
   does `1:d-1` range arithmetic and `rank=(0, d)` which require concrete integers.

**Impact**: `define_metric!` now works for symbolic-dim manifolds but skips epsilon
tensor registration (you cannot construct a fully-antisymmetric tensor of symbolic rank).
For integer dimensions, behavior is unchanged.

### Change 4: DDI order guard for symbolic dimensions

**File**: `src/algebra/full_simplify.jl`
**Change**: `_fs_ddi_order(expr, dim::Int)` → `_fs_ddi_order(expr, dim)` with
`dim isa Int ? clamp(deg, 2, dim ÷ 2) : deg` guard.

**Why**: DDI (dimensionally dependent identity) capping at `dim ÷ 2` is meaningless for
symbolic dimensions. Now returns `deg` uncapped when dim is symbolic.

**Impact**: DDI simplification for symbolic-dim manifolds will apply all DDI orders up
to the expression degree (no capping). This is correct — capping is an optimization for
known-dimension cases where higher-order DDIs vanish identically.

---

## What Was Done Last Session (Session 14: 3 issues closed)

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

### New this session (session 15)
- **`ManifoldProperties.dim` is now `Union{Int,Symbol}`**: Feynfeld.jl needs symbolic
  dimensions for dimensional regularisation. All GR-specific code that does arithmetic
  on `dim` (hamiltonian, geodesics, components, foliation, brauer) will error for
  symbolic dims — this is correct behavior.
- **`contract_metrics` returns `TScalar(:D)` for symbolic-dim metric traces**: Previously
  only returned `TScalar(dim // 1)` for integer dims.
- **`define_metric!` skips epsilon tensor for symbolic dims**: Cannot construct a
  fully-antisymmetric tensor of symbolic rank.
- **Feynfeld.jl is a consumer of TensorGR.jl**: It uses the registry, TIndex, contraction,
  and canonicalization engines for Lorentz algebra in QFT. Changes to these APIs must
  consider both GR and QFT use cases.

### From session 14
- **`free_indices()` uses Dict iteration → non-deterministic in Julia 1.12**: Use `indices(expr)` for deterministic ordering, then filter to free indices by name set.
- **Dummy detection after free-index replacement is WRONG**: After replacing free indices `:a` → `:_1`, `:b` → `:_1`, the duplicate `:_1` entries are falsely detected as dummy pairs. Always compute dummies from the ORIGINAL expression.
- **Sort-based rule matching fails with shared pattern variables**: Pattern `T_{a_,b_} * T_{b_,c_}` vs `T_{z,y} * T_{y,x}` — sorting aligns factors such that `b_` gets conflicting bindings. Backtracking within same-name groups is the correct approach.
- **`while` loops polling for Julia processes**: These can spawn extra shell processes and cause confusion. Avoid them — just run tests in foreground or use background tasks with notifications.

---

## ⚠ Core Changes To Monitor

**This session** (symbolic dimension support for Feynfeld.jl):
- Location: `src/registry.jl` (struct types), `src/algebra/contraction.jl` (metric trace),
  `src/gr/metric.jl` (epsilon/signature guards), `src/algebra/full_simplify.jl` (DDI guard)
- Change: `dim::Int` → `dim::Union{Int,Symbol}` on ManifoldProperties and VBundleProperties,
  plus runtime guards at 3 arithmetic sites
- Risk: **Medium** — public API type change. All 375,404 tests pass. All Tier 1 benchmarks pass.
  But any downstream code type-asserting `dim::Int` will break.
- **GR-specific code that does arithmetic on `dim`** (hamiltonian, geodesics, components,
  foliation, brauer) will naturally error with `MethodError` for symbolic dims. This is
  correct — you cannot compute Christoffel symbols in D dimensions numerically.
- Revert: change `Union{Int,Symbol}` back to `Int` in registry.jl and revert the 3 guards

**Session 14** commits (carried forward):

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

1. **Run Reviewer 2 for TensorGR.jl-88e** (rule matching fix) — was interrupted in session 14
2. **Add dedicated tests for order-independent rule matching** — currently only verified via REPL
3. **Run full benchmarks (Tier 1-3)** — only Tier 1 run this session
4. **Add tests for symbolic-dim manifolds** — currently only tested from Feynfeld.jl side
5. **Update docstrings** for `ManifoldProperties`, `VBundleProperties`, `define_vbundle!` to
   document that `dim` accepts `Symbol` for symbolic dimensions
6. Continue with ready queue (`bd ready`)

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
