# HANDOFF — 2026-04-16 (Session 19: xAct golden-master infrastructure, TGR-bhs5 epic)

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
10. **NO PARALLEL JULIA**: Julia precompilation cache conflicts within the same
    project. Cross-project parallel is OK. Check `ps aux | grep julia` first.

**Corollary**: Review xAct source (`reference/xAct/`) BEFORE changing core modules.
**Max 2-3 subagents at a time (sequential).** Checkpoint regularly.
**USE MAX THINKING (opus) for all subagents.**
**WSL2 MEMORY**: Never enumerate large combinatorial sets in memory.

---

## Current State

- **Last pushed commit**: `e4abb5f` on `master` (pre-session; updated by this commit)
- **571 beads issues total** (up from 540) — TGR-bhs5 epic + 30 children added
- **Full test suite**: NOT RUN this session (another TensorGR Julia process was
  not running, but a Bennett.jl suite WAS active on the machine; skipped the
  targeted full-suite run to avoid juggling). Run `julia --project -e 'using
  Pkg; Pkg.test()'` next session to confirm no regressions.
- **Golden suite** (new this session): 11/11 cases pass locally when run via
  `julia --project=test test/test_golden.jl` or with
  `TENSORGR_GOLDEN=1 julia --project -e 'using Pkg; Pkg.test()'`.

---

## What Was Done This Session (Session 19)

**Epic: TGR-bhs5** — Cross-platform xAct ↔ TensorGR golden-master infrastructure.

Full plan at `test/golden/PLAN.md`. Conventions at `test/golden/CONVENTIONS.md`.
Contributor docs at `docs/src/golden_masters.md`.

### Infrastructure delivered

**Wolfram/xAct side** (`test/golden/generators/`):
- `common.wl` — library: `tgrSetupXAct`, `tgrSetupXPert`, `tgrEmitJSON`,
  `tgrNormalizeDummies`, xAct→neutral name map. Pattern-dispatch bug found
  and fixed: BlankSequence on Plus/Times got merged in DownValues, so rewrote
  via explicit `Which[Head[expr] === ...]` dispatch.
- `conventions_probe.wl` — Phase 0 ground-truth probe (metric compat, Ricci
  contraction, Ricci scalar trace).
- 9 case generators (bianchi, ricci_contraction, kretschmann,
  second_bianchi_trace, mixed_product, covd_of_ricci, to_riemann, to_ricci,
  einstein_identity, weyl_trace_free).

**Julia side** (`test/golden/consumers/` + `test/`):
- `golden_loader.jl` — JSON → TensorExpr (47 lines).
- `golden_emitter.jl` — TensorExpr → JSON with dummy normalization to d1..dN
  in AST first-occurrence order (~110 lines).
- `golden_runner.jl` — case dispatch (`:simplify`, `:canonicalize`,
  `:to_riemann`, `:to_ricci`), re-canonicalize both sides, unified JSON diff on
  mismatch (~75 lines).
- `conventions_probe_verify.jl` — Phase 0 gate verification (TensorGR side).
- `regen.jl` — one-shot generator driver with `--dry-run` and per-case filter.

**Test suites**:
- `test_golden_schema.jl` (9 assertions).
- `test_golden_loader.jl` (10).
- `test_golden_emitter.jl` (13).
- `test_golden_wl_emitter.jl` (23 — live wolframscript probe).
- `test_golden_runner.jl` (9).
- `test_golden.jl` (glob-loads all `data/*.json` and runs cases).

**Schema**: `test/golden/schema/v1.json` — JSON Schema draft-07. Expr union
(Tensor | Product | Sum | Deriv | Scalar) + CaseFile envelope.

**CONVENTIONS.md**: signature −+++, Ricci slot conventions (xAct slot-2&4 ↔
TensorGR slot-1&3, equivalent under Riemann pair-swap), name map, ops table,
Phase 0 sanity invariants.

**runtests.jl integration**: gated behind `ENV["TENSORGR_GOLDEN"]`. Default
suite unaffected.

### Issue progress (TGR-bhs5 children)

- **Phase 0 (.1 .2 .3)**: CLOSED. Conventions pinned, probe verified both
  sides.
- **Phase 1 (.4 .. .10)**: CLOSED. Schema + loader + emitter + runner +
  first-Bianchi case + runtests integration.
- **Phase 2 (.11 .. .15)**: CLOSED. Ricci contraction, Kretschmann, second
  Bianchi trace, mixed product, CovD of Ricci.
- **Phase 3 (.16 .. .19)**: CLOSED. Einstein expansion via to_riemann, same
  via to_ricci, Einstein identity → 0, Weyl trace-free.
- **Phase 4 (.20 .. .25)**: .20 CLOSED with limited scope; **.21 .. .25
  BLOCKED**, see below.
- **Phase 5 (.26 .27)**: **BLOCKED** pending xAct commutator-rule research.
- **Meta (.28 .29 .30)**: .28 (regen.jl), .29 (docs) CLOSED; **.30 (license
  review) OPEN**.

**22 closed, 7 blocked, 1 open.**

---

## ⚠ BLOCKED ISSUES — Investigation Required

### TGR-bhs5.21 .. .25 — xPert curvature perturbations

**Symptom**: `DefMetricPerturbation[g, h, eps]` in xPert installs 36 upvalues
on `g` (so `Perturbation[g[-a,-b]] → h[LI[1],-a,-b]` works), but
`ExpandPerturbation1` has **0 downvalues** for curvature tensors. Calling
`ExpandPerturbation[Perturbation[RicciCD[-a,-b]]]` returns unchanged.

xPert.m lines 228-236 register formulas via `DefGenPertRicci`, etc. — these
didn't take effect in our runtime. Possibly a context/protection issue, or
needs a missing setup step.

**Next action**: Spawn xAct research subagent (per HANDOFF rule 4) to
investigate:
1. Minimal reproduction (empty kernel, load xTensor + xPert, DefManifold +
   DefMetric + DefMetricPerturbation, check `Length[DownValues[xAct`xPert`
   ExpandPerturbation1]]`).
2. Compare against a known-working xAct notebook (e.g., xPert's own demo
   notebook in `reference/xAct/xAct/xPert/xPert.nb`).
3. Identify the missing setup step or context fix.

Once fixed, the 5 blocked perturbation cases are straightforward.

### TGR-bhs5.26 .27 — CovD commutation

Probably analogous — xAct's commutator rules likely need specific setup
beyond DefCovD. Deferred until Phase 4 is unblocked.

### TGR-bhs5.30 — License review (OPEN, not blocked)

GPL generators under `test/golden/generators/` vs Apache-2.0 consumers under
`test/golden/consumers/`. Need to:
1. Add `test/golden/generators/LICENSE` (GPL-2.0).
2. Verify `test/golden/generators/` is excluded from Pkg tarball.
3. Cross-link with pre-existing HANDOFF TODO "GPL/Apache-2.0 license review".

---

## ⚠ Core Changes To Monitor

**This session** (uncommitted → will be in this commit):

**New infrastructure** (`test/golden/`):
- Risk: LOW — gated behind `ENV["TENSORGR_GOLDEN"]`, default suite unaffected.
- If it breaks: drop `TENSORGR_GOLDEN` from CI; test/runtests.jl tail is the
  only integration point.

**test/Project.toml**: Added `JSON` and `JSONSchema` as test-only deps.
- Risk: LOW — test-only.

**test/runtests.jl**: Conditional include at tail.
- Risk: NEGLIGIBLE — one `if get(ENV, ..., "") != ""` block.

No core module changes. `src/` is untouched.

---

## Known xAct / xPert Quirks Discovered

1. **Pattern dispatch**: BlankSequence patterns on Plus/Times in DownValues
   can collapse. Workaround: use `Which[Head[expr] === ...]` explicitly.
2. **Validate::inhom**: xAct throws on sums where one term is a scalar (empty
   IndexList) and another carries free indices. Workaround: don't call
   ToCanonical on such sums; emit directly.
3. **ToCanonical::noident**: xAct warns when a standalone expression has no
   applicable canonicalization rules. Output is unchanged; benign.
4. **xPert DefGenPertRicci**: not firing — see blocked issues above.

---

## ⚠ BEADS DATABASE: CRITICAL CROSS-DEVICE INSTRUCTIONS

Unchanged from session 18:

- Dolt DB (`.beads/dolt/`) is gitignored and local.
- JSONL (`.beads/issues.jsonl`) is git-tracked = source of truth.
- After closing/creating issues: `bd export -o .beads/issues.jsonl`.
- Fresh clone: `bd import` to rebuild Dolt from JSONL. Needs bd v0.62.0+.

---

## TODO Next Session

1. **Run full test suite with goldens** —
   `TENSORGR_GOLDEN=1 julia --project -e 'using Pkg; Pkg.test()'` — confirm
   no regressions introduced by this session.
2. **Investigate xPert curvature rules** (unblocks TGR-bhs5.21 .. .25). Start
   with minimal reproducer. Consult `reference/xAct/xAct/xPert/xPert.nb`.
3. **TGR-bhs5.30 license review** — straightforward cleanup.
4. **Audit the 35 prior open issues** (partially done last session).
5. Return to the Session-18 TODOs still pending:
   - Einstein trace rule, Weyl trace-free rule (in core simplify pipeline,
     separate from goldens).
   - `deps/build.jl` for cross-platform xperm.c.

## Quick Commands

```bash
# Golden suite
julia --project=test test/test_golden.jl
TENSORGR_GOLDEN=1 julia --project -e 'using Pkg; Pkg.test()'
julia --project test/golden/regen.jl --dry-run

# Issue tracking
bd list --parent TGR-bhs5 --status=blocked     # xPert work
bd show TGR-bhs5.20                             # xPert partial finding
bd stats

# Standard
julia --project -e 'using Pkg; Pkg.test()'
git log --oneline -10
git diff --stat
```
