# HANDOFF — 2026-03-26 (Session 16: Thread safety, API cleanup, parametric derivatives)

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

- **533 of 569 issues closed** (3 closed this session)
- **Full test suite: 375,443 tests, ALL PASS** (1 known Broken in test_euler_density.jl:480)
- **Benchmarks: not re-run this session** — Tier 1 passed last session
- All changes pushed to `master` (commit dc95a27)
- `bd stats` for live counts, `bd ready` for available work

---

## What Was Done This Session (3 issues closed)

### 1. TensorGR.jl-lj1 (P2 Bug): Thread-safe TensorRegistry

**Problem**: `_GLOBAL_REGISTRY` is a module-level mutable `TensorRegistry` shared across tasks with no lock protection. Concurrent tasks calling `register_tensor!` etc. can corrupt Dict internals.

**Fix**: Added `lock::ReentrantLock` field to `TensorRegistry` struct. Wrapped all 25 mutation sites across 17 files with `@lock reg.lock begin ... end`. Uses `ReentrantLock` because compound operations (e.g., `define_metric!` → `register_tensor!` → `register_rule!`) nest up to 3 levels deep.

**Files modified** (17):
- `src/registry.jl` (struct + 9 core mutators)
- `src/gr/metric.jl` (define_metric!, set_flat!, freeze_metric!, unfreeze_metric!, set_conformal_to!)
- `src/foliation/foliation.jl`, `src/gr/mapping.jl`, `src/gr/hypersurface.jl`, `src/gr/product_manifold.jl`, `src/gr/matter.jl`, `src/gauge/brst.jl`, `src/fermions/stress_energy.jl` (direct reg.foliations/mappings writes)
- `src/tetrads/frame_bundle.jl`, `src/spinors/spin_metric.jl`, `src/spinors/space_spinors.jl`, `src/spinors/ashtekar_variables.jl` (direct metric_cache/delta_cache writes)
- `src/spinors/soldering_form.jl`, `src/scalar/functions.jl`, `src/scalar_tensor/dhost_degeneracy.jl`, `src/bimetric/potential.jl` (direct tp.options or reg.rules writes)

**Design decisions**:
- Lock on the struct (not module-level) — protects ANY shared registry, not just global
- Reads are NOT locked — safe because all writes are serialized and Julia Dict reads of completed writes are consistent
- ~20 compound functions (define_covd!, define_curvature_tensors!, etc.) are NOT individually locked — they're always called from within already-locked functions, and primitives lock individually
- Fixed `push!(reg.rules, rule)` bypass in `register_sqrt_rules!` to use `register_rule!`

**Risk**: Low. No behavioral change for single-threaded code. Uncontended ReentrantLock is ~20ns overhead. 375,443 tests pass.
**Revert**: Remove `lock::ReentrantLock` from struct, update constructor, remove all `@lock` wrappers.
**Unblocks**: TensorGR.jl-304 (registry passing pattern standardization)

### 2. TensorGR.jl-6sb (P2 API): @manifold vs define_metric! overlap

**Problem**: `@manifold` only registered manifold + metric + delta (no curvature, CovD, Bianchi). `define_metric!` did full setup but couldn't be called after `@manifold` without errors. Users had no clear one-stop solution.

**Fix**:
1. `@manifold` now calls `define_metric!` internally (full setup: metric, delta, epsilon, curvature tensors, CovD, Bianchi rules)
2. `define_curvature_tensors!` made idempotent with `has_tensor` guards on all 6 tensors (Riem, Ric, RicScalar, Ein, Weyl, Sch)

**Files modified**:
- `src/macros/definitions.jl` (simplified @manifold body)
- `src/gr/curvature.jl` (added has_tensor guards)

**Backward compatibility**: Full. Tests that did `@manifold` + `define_curvature_tensors!` still work (second call is a no-op). Tests that did `register_manifold!` + `define_metric!` unaffected.

**Risk**: Low. Purely additive behavior change. 375,443 tests pass.

### 3. TensorGR.jl-7cb (P2 Feature): Parametric derivatives (TParamDeriv)

**New AST node**: `TParamDeriv(params::Vector{Symbol}, arg::TensorExpr)`

Represents d/dp₁ d/dp₂ ⋯ d/dpₙ applied to a tensor expression. Parameters are scalar symbols (time `t`, proper time `τ`) independent of manifold coordinates.

**Key properties** (following xAct ParamD semantics):
- **Index-free**: carries no tensor indices (unlike TDeriv)
- **Auto-flatten**: `d/ds(d/dt(x))` → `TParamDeriv([:s,:t], x)` with sorted params
- **Leibniz**: `d/dt(A*B) = dA/dt*B + A*dB/dt` (peels off one param at a time)
- **Linearity**: distributes over TSum
- **Zero on constants**: `d/dt(c) = 0` for rational constants
- **Self-derivative**: `d/dt(t) = 1` for registered parameters
- **Commutes with ∂**: `d/dt(∂_a X) = ∂_a(d/dt X)`

**Files created**:
- `src/algebra/param_deriv.jl` (~120 lines): `param_deriv` smart constructor, `expand_param_deriv`
- `test/test_param_deriv.jl` (45 tests)

**Files modified**:
- `src/types.jl` (TParamDeriv struct + ==, hash)
- `src/registry.jl` (define_parameter!, is_parameter)
- `src/ast/walk.jl` (children, walk, dagger, derivative_order, is_constant)
- `src/ast/indices.jl` (indices — returns arg's indices, no own indices)
- `src/show.jl` (Base.show, to_latex, to_unicode)
- `src/TensorGR.jl` (include + exports)
- `test/runtests.jl` (include test file)

**Risk**: Low — purely additive. New AST node, no changes to existing expression handling.

---

## What Was Done Last Session (Session 15: Feynfeld.jl integration — 0 issues closed)

Symbolic manifold dimensions (`dim::Union{Int,Symbol}`) for Feynfeld.jl dimensional regularisation.
4 changes: registry struct types, metric trace guard, define_metric! epsilon/signature guards, DDI order guard.
See previous HANDOFF for full details.

---

## Key Decisions / Lessons

### Carried from previous sessions
- **FullySymmetric(n)** takes slot numbers as varargs: `FullySymmetric(1,2,3,4)` NOT `FullySymmetric(4)`
- **make_rule** RETURNS rules but does NOT register them
- **symmetrize** takes `Vector{Symbol}` not `Vector{TIndex}`
- **xperm convention for canonical_perm_ext**: Renato notation. Generators SLOT-SPACE for right-coset.
- **No parallel agents/Julia**: cache conflicts + OOM on WSL2 (cross-project parallel is OK).
- **AntiSymmetric fields**: `.i` and `.j`, NOT `.slot1`/`.slot2`
- **Beads issues.jsonl is source of truth**: `.beads/issues.jsonl` in git.
- **Pinned term counts are NOT ground truth**: physics correctness is what matters
- **Code review agents can be WRONG about physics**: verify against textbooks
- **`ManifoldProperties.dim` is now `Union{Int,Symbol}`**: for Feynfeld.jl dimensional regularisation
- **Feynfeld.jl is a consumer of TensorGR.jl**: Changes to APIs must consider both GR and QFT use cases

### New this session (session 16)
- **`TensorRegistry` now has a `lock::ReentrantLock` field**: All mutating operations are locked. Reads are lock-free. Compound operations nest via reentrancy (up to 3 levels: e.g., `define_metric!` → `register_tensor!` → lock).
- **`@manifold` now does full setup**: Calls `define_metric!` internally, giving curvature tensors, CovD, and Bianchi rules. No need for separate `define_curvature_tensors!` call (though it still works — idempotent).
- **`define_curvature_tensors!` is idempotent**: `has_tensor` guards on all 6 curvature tensors. Safe to call multiple times.
- **`TParamDeriv` is the new AST node for parametric derivatives**: Index-free, auto-flattening, sorted params. Follows xAct `ParamD` design. Parameters registered via `define_parameter!`.
- **Cross-project parallel Julia is OK**: The no-parallel-Julia rule applies only within the same project (shared precompile cache). Different `--project` paths are safe.

---

## ⚠ Core Changes To Monitor

**This session** (commit dc95a27):

**Registry lock** (TensorGR.jl-lj1):
- Location: `src/registry.jl` (struct + 9 functions) + 16 other files
- Change: Added `lock::ReentrantLock` to `TensorRegistry`, `@lock` wrappers on all mutations
- Risk: Low — no behavioral change for single-threaded code
- Revert: Remove `lock` field, update constructor, remove all `@lock` wrappers

**@manifold full setup** (TensorGR.jl-6sb):
- Location: `src/macros/definitions.jl`, `src/gr/curvature.jl`
- Change: `@manifold` calls `define_metric!`; `define_curvature_tensors!` idempotent
- Risk: Low — strictly more functionality, backward-compatible
- Revert: Restore old `@manifold` body with manual `register_tensor!` calls

**TParamDeriv** (TensorGR.jl-7cb):
- Location: `src/types.jl`, `src/algebra/param_deriv.jl` (new), `src/ast/*`, `src/show.jl`, `src/registry.jl`
- Change: New AST node type + parameter infrastructure
- Risk: Low — purely additive, no existing code paths changed
- Revert: Remove TParamDeriv from types.jl, delete param_deriv.jl, revert walk/indices/show additions

**Carried from session 15** (symbolic dimensions):
- `dim::Union{Int,Symbol}` on ManifoldProperties/VBundleProperties + 3 arithmetic guards
- Risk: Medium — public API type change

**Carried from session 14** (rule matching):
- Grouped backtracking in `_unify(::TProduct/TSum)` — `src/rules.jl` lines 105-210
- Risk: Medium — core pattern matching. **Reviewer 2 still not run.**

---

## TODO Next Session

1. **Run Reviewer 2 for TensorGR.jl-88e** (rule matching fix) — still pending from session 14
2. **Add dedicated tests for order-independent rule matching** — only verified via REPL
3. **Run full benchmarks (Tier 1-3)** — not run this session
4. **Add tests for symbolic-dim manifolds** — still only tested from Feynfeld.jl side
5. Continue with ready queue (`bd ready`) — 25 issues ready

## Ready Queue

```bash
bd ready    # see available work
bd stats    # project health
```

**25 ready issues** (down from 27). Key items:
- TensorGR.jl-77q (CCovD) — P2, unblocked by evaluate_components
- TensorGR.jl-304 (registry passing pattern) — P3, NOW UNBLOCKED by lj1 fix
- 8 P2 test coverage issues
- TensorGR.jl-irx (chart transitions) — P3, unblocked
- 2 P2 infrastructure (BinaryBuilder, Pkg registration)

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
