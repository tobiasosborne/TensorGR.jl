# HANDOFF — 2026-09-29 (Session 20: architecture review, canonicalization reboot research, C prototypes)

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

- **master** is pushed through the session-20 review/research commits (see `git log`); **no `src/` changes
  this session** — all work is in `reviews/` plus two local prototype branches.
- **Tobias is considering a total reboot**, starting with replacing `deps/xperm.c` by a new canonicalization
  core. Session 20 produced the research, a speed-of-light bound, and two blind C prototypes.
- **⚠ Beads DB out of sync**: after the bd v0.62→v1.0.0 upgrade the local Dolt DB has 537 issues, but
  `.beads/issues.jsonl` (git) has 571 incl. the TGR-bhs5 epic (`bd show TGR-bhs5` → not found).
  Run `bd doctor` then `bd import` before touching issues. **No beads issues were filed this session**
  for the bugs below — file them after the resync.
- **Full test suite**: still NOT RUN since session 18. Golden suite: last known 11/11 (session 19).
- `deps/libxperm.so` was built by hand this session (gitignored); `deps/build.jl` cannot build it (below).

---

## What Was Done This Session (Session 20) — read `reviews/07`–`10`

**1. Architecture review vs Julia best practices** — `reviews/07_julia_best_practices_review.md`,
runtime probes in `reviews/07_probes/` (`julia --project reviews/07_probes/probe_correctness.jl`).
**Verified silent correctness bugs in the current core (P0, unfiled):**
- `canonicalize` treats covariant derivatives as commuting (`src/algebra/canonicalize.jl:272`, no
  `covd == :partial` check): `simplify(V^e(∇_b∇_a S_cd − ∇_a∇_b S_cd))` returns **0**. Affects `∇∇h` on
  curved backgrounds → MSS covariant output, 6-deriv dS spectrum, bench_12 must be re-verified.
- `canonicalize` permutes index names but pins Up/Down to slots (`canonicalize.jl:313`): `S^b_a → S^a_b`
  (free-index positions change); `A^b_a + A_a^b` does not simplify to 0. Root cause: all indices passed
  to xperm as free (no dummy double-coset step) — `fix_dummy_positions` is a band-aid for this.
- `to_latex`/`to_unicode` print every derivative as ∂ (`show.jl:189,306`).
- No free-index consistency check: `V^a + V_a` accepted.
- `deps/build.jl` can never succeed (soft-scope + top-level `return`); CI ran once (Mar 2026, failed).
- `canonicalize(trinv; registry=…)` at `invariants/trinv.jl:958,982` is a guaranteed MethodError (JET).
- `simplify` on `TParamDeriv` / harmonic node types → MethodError; objectid-keyed global caches leak.
- Architecture: 54k lines, 1,083 exports, 18 TensorExpr subtypes with hand-written traversals (83 passes
  on TDeriv, 9 on TParamDeriv), ambient task-local registry, `Rational{Int}` overflow, `Expr` as CAS.
  CLAUDE.md is stale (claims 12k lines / 71 files).

**2. Canonicalization literature survey** — `reviews/08_canonicalization_literature_survey.md` (+ parallel
study `reviews/08b_parallel_racing_pareto.md`). Key points: graph individualization–refinement (IR)
beats Butler–Portugal on identical-factor products (SeQuant 2511.09943); graph backtracking
(Jefferson–Waldecker–Wilson, Vole) unifies both; multi-term (Bianchi) is a separate linear-algebra layer;
**`leanprover/hex-graph-iso` has verified Lean 4 implementations of pinned nauty 2.9.3 configurations**
(verified to exist) — the cheap route to Lean-checked canonicity. Competitors: Alakazam.jl (Aug 2026),
Symbolica/graphica, GraphCombinations.jl.

**3. Speed-of-light bound** — `reviews/09_canonicalization_perf_bound.md` (+ rendered `.html`, model
`reviews/09_probes/bound.py`). Generic 8-core desktop + RTX 4060-class GPU: CPU is compute-bound
(10–200 ns/term), GPU only pays if terms stay on device (PCIe floor 10–22 ns/term). Current TensorGR:
438 µs/term for Riem³ (≈500–1000× above floor). Known model flaw: search term assumes S_k on identical
factors (wrong for chains) — revise §4.

**4. Two blind C prototypes (IR canonicalizer, SeQuant-style graph)** — `reviews/10_blind_prototype_comparison.md`.
- Branches (**local only, not pushed**): `proto/canon-ir-baseline` (Sonnet, `302bd12`) and
  `proto/canon-ir-baseline-opus` (Opus, `a5351dc`); code in `proto/canonir/`, write-ups in
  `proto/canonir/RESULTS.md`. Worktrees still exist under `.claude/worktrees/agent-*` (locked).
- Both pass their own suites (1.0 M / 0.32 M checks, plain + ASan/UBSan) and a **blind cross-check**
  (`reviews/10_probes/xcheck.c`, ~2.33 M checks, 0 disagreements, sensitivity verified).
  Reproduce: `bash reviews/10_probes/build_and_run.sh`.
- Opus impl is 1.2–1.35× faster on random terms, 1.9–3.3× on symmetric products → **recommended base**.
- **vs xperm.c called correctly** (`reviews/10_probes/hard.c`, full double-coset mode):
  $(R_{abcd}R^{abcd})^3$ = 6 identical Riemanns: xperm **5.48 s/call** vs Opus **28.6 µs** (≈190,000×).
  xperm convention (found empirically): `PERM[name] = slot`, dummies given as slots of (up, down) —
  TensorGR never passed dummies, so this was never exercised.

---

## Previous Session (Session 19)

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

**Session-20 items (reboot track — decide direction with Tobias first):**
1. **Beads resync** (`bd doctor`, `bd import`), then file P0 issues for the verified core bugs in
   `reviews/07` §1 (∇ commutation, free-index positions, ∂ display) and the broken `deps/build.jl`.
2. **Decide**: patch the current core vs reboot on the C prototype. If rebooting: take the Opus prototype
   as base, port Sonnet's `tests/mutate.sh` + high-volume random tests, keep `xcheck.c` as a permanent
   differential test; next features = spinor ε metric signs, multiple vbundles, derivative slots
   (∂ commuting, ∇ not), certificate output compatible with `hex-graph-iso`, Julia `ccall` binding,
   then xAct golden-master comparison.
3. Hostile review of the chosen prototype before relying on it (HANDOFF rule 5).
4. Decide whether to push the two `proto/*` branches; clean up `.claude/worktrees/agent-*`.
5. Re-verify curved-background results (dS 6-deriv spectrum, bench_12) independently of `canonicalize`.

**Carried over from session 19:**
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
# Session 20: correctness probes, bound model, prototype cross-check
julia --project reviews/07_probes/probe_correctness.jl   # needs deps/libxperm.so:
#   gcc -shared -fPIC -O2 -o deps/libxperm.so deps/xperm.c   (build.jl is broken)
python3 reviews/09_probes/bound.py
bash reviews/10_probes/build_and_run.sh                  # needs local proto/* branches
git log --oneline proto/canon-ir-baseline-opus -3

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
