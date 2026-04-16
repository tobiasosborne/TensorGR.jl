# Cross-Platform Golden Master Infrastructure (xAct ↔ TensorGR)

**Epic**: `TGR-gm` — Cross-platform xAct↔TensorGR golden master infrastructure
**Status**: Planned, not started
**Authored**: 2026-04-16

## Purpose

Use xAct (Mathematica, via wolframscript) as an independent ground-truth oracle
for TensorGR.jl's tensor algebra. A golden master is a triple
`(input expression, operation, expected output)` serialized in a neutral JSON
schema. **Both engines are observers of the same ground truth.** Committed
`data/*.json` files are authoritative; CI runs only the Julia consumer — no
Wolfram dependency at test time.

## Hard rules

These apply to EVERY issue below. No exceptions.

1. **Red-green TDD — strict**: For every implementation issue, the failing
   test is committed FIRST in a separate commit titled `red: <issue>` that
   demonstrably fails (`julia --project -e 'using Pkg; Pkg.test()'` reports
   the new failure). The green commit is titled `green: <issue>` and
   **only** contains the minimum code to pass; any refactor goes in a
   third `refactor: <issue>` commit. Three commits minimum per issue.
2. **Physics is ground truth, not pinned numbers** (HANDOFF rule 6). A
   golden that disagrees with physical identity takes precedence over
   any committed `.json` — investigate the JSON, do not silence the
   test.
3. **Subagent workflow** (HANDOFF rule 4) applies to any issue that
   crosses into canonicalization semantics (emitter, loader, runner,
   convention probe). xAct research agent + 2 competing solution
   proposals + reviewer agent. Pure plumbing issues (schema file,
   Makefile) may skip, with justification in the issue notes.
4. **No parallel Julia** (HANDOFF rule 10). wolframscript runs are fine in
   parallel with Julia since they target a different kernel, but multiple
   Julia processes on the same project still fight the precompile cache.
5. **Convention drift kills the suite**. Convention decisions (signs,
   signatures, name map) are pinned in `CONVENTIONS.md` and verified by
   a probe case BEFORE any other golden is generated.

## Schema (v1, draft)

See `test/golden/schema/v1.json` (to be written in Phase 1).

Typed AST, recursive union:

```
Expr     = Tensor | Product | Sum | Deriv | Scalar
Tensor   = {type:"tensor", name:str, indices:[TIndex]}
Product  = {type:"product", coef:{num:int,den:int}, factors:[Expr]}
Sum      = {type:"sum", terms:[Expr]}
Deriv    = {type:"deriv", covd:str|"partial", index:TIndex, arg:Expr}
Scalar   = {type:"scalar", value:{rational:{num,den}} | {symbol:str}}
TIndex   = {name:str, pos:"up"|"down", vbundle:str}
```

Case file:

```
{
  schema_version: 1,
  conventions: {signature:"-+++", riemann_sign:"...", name_map:{...}},
  manifolds: {...}, tensors: {...},
  cases: [
    {
      name: "first_bianchi_unreduced",
      input: <Expr>,
      op: "canonicalize" | "to_riemann" | "delta_ricci" | ...,
      op_args: {...},
      expected: <Expr>,
      generator: "generators/bianchi.wl",
      notes: "3-term monoterm canonical form"
    }
  ]
}
```

Dummy normalization: after canonicalization, both emitters rewrite dummies
left-to-right as `d1, d2, ...`. Free indices keep their names. Byte-equal JSON
comparison becomes meaningful.

## Layout

```
test/golden/
  PLAN.md                     # this file
  CONVENTIONS.md              # sign/signature/name map — authoritative
  schema/v1.json              # JSON Schema
  generators/                 # xAct / wolframscript — NOT run in CI
    common.wl                 # setup, name map, JSON emitter, dummy norm
    conventions_probe.wl      # pinned reference ahead of any other generator
    bianchi.wl
    contractions.wl
    curvature_basis.wl
    perturbation.wl
    covd_commute.wl
    LICENSE                   # GPL (xAct derivative)
  data/                       # committed JSON ground truth
    conventions_probe.json
    bianchi.json
    ...
  consumers/
    golden_loader.jl          # JSON → TensorExpr
    golden_emitter.jl         # TensorExpr → JSON
    golden_runner.jl          # dispatch op, re-canonicalize, compare
  test_golden.jl              # integrated into runtests.jl, ENV-gated
  regen.jl                    # runs wolframscript across generators
```

## Phases and issues

### Phase 0 — Conventions (blocks everything else)

- **TGR-gm.1**: Write `CONVENTIONS.md` documenting signature, Riemann sign,
  CovD orientation, name map. Physics ground truth only.
- **TGR-gm.2**: Write `conventions_probe.wl`. Dumps xAct results for:
  `CD[-c][g[-a,-b]]`, `RiemannCD[-a,-c,-b,c] - RicciCD[-a,-b]`,
  `g^{ab} RicciCD[-a,-b] - RicciScalarCD[]`. Output: xAct InputForm strings.
- **TGR-gm.3**: Manual comparison against TensorGR's equivalents. Ground-truth
  gate. Close only when all three match on both sides.

### Phase 1 — Schema and minimal pipeline

- **TGR-gm.4**: Write `schema/v1.json` (JSON Schema). RED: test that valid
  sample validates, malformed sample rejects. GREEN: schema.
- **TGR-gm.5**: `golden_loader.jl`. RED: load canned JSON, assert structural
  equality with hand-built TensorExpr (5 small fixtures). GREEN: implement
  minimal recursive loader.
- **TGR-gm.6**: `golden_emitter.jl` (TensorExpr → JSON, with dummy
  normalization). RED: round-trip test (expr → JSON → expr == canonicalized
  original). GREEN: implement.
- **TGR-gm.7**: `common.wl` `EmitJSON` function. RED: hand-authored fixture
  JSON committed first; wolframscript run must produce byte-equal output.
  GREEN: implement.
- **TGR-gm.8**: `golden_runner.jl`. RED: hand-built case runs and passes;
  synthetic wrong-expected case runs and fails with readable diff. GREEN:
  implement dispatch + re-canonicalize + compare.
- **TGR-gm.9**: First real case — first Bianchi three-term monoterm survival
  (`R_{a[bcd]}` → 3-term sum). RED: commit JSON, runner fails because
  emitter/loader have trivial mapping bug or similar. GREEN: end-to-end pass.
- **TGR-gm.10**: Integrate `test_golden.jl` into `runtests.jl`, gated behind
  `ENV["TENSORGR_GOLDEN"]`. RED: test file exists, env gate works (skipped
  by default, runs when set). GREEN: wire into runtests.

### Phase 2 — Canonicalization and contractions (~5 cases)

- **TGR-gm.11**: Ricci contraction identity `R^c_{acb} = Ric_{ab}`.
- **TGR-gm.12**: Kretschmann canonical form `R_{abcd} R^{abcd}`.
- **TGR-gm.13**: Second Bianchi trace identity `∇_a R^a_{bcd} = ∇_b R_{cd} - ∇_c R_{bd}` (or analogue).
- **TGR-gm.14**: Product canonicalization with mixed symmetries
  `R_{abcd} R^{cd}_{ef}` canonical form.
- **TGR-gm.15**: Derivative canonical form `∇_c R_{ab}`.

### Phase 3 — Curvature basis conversions (~4 cases)

- **TGR-gm.16**: `to_riemann`: `Ric_{ab}` → `g^{cd} R_{cabd}` or analogue.
- **TGR-gm.17**: `to_ricci`: expression in Weyl + trace reduces back to Ricci form.
- **TGR-gm.18**: Einstein identity `G_{ab} = R_{ab} - (1/2) g_{ab} R`.
- **TGR-gm.19**: Weyl trace-free `g^{ac} C_{abcd} = 0`.

### Phase 4 — Perturbation theory (highest value)

- **TGR-gm.20**: Set up xPert in `common.wl` (`DefMetricPerturbation`).
- **TGR-gm.21**: δ¹R_{ab} golden — cross-check `delta_ricci` output.
- **TGR-gm.22**: δ¹R golden.
- **TGR-gm.23**: δ²R_{ab} golden (second-order, exercises `expand_perturbation`).
- **TGR-gm.24**: δ¹G_{ab} golden (Einstein tensor perturbation).
- **TGR-gm.25**: Gauge transformation `δh_{ab} → δh_{ab} + 2∇_{(a}ξ_{b)}`.

### Phase 5 — CovD commutation (stretch)

- **TGR-gm.26**: `[∇_a, ∇_b] T^c = -R^c_{dab} T^d`.
- **TGR-gm.27**: `[∇_a, ∇_b] T_{cd}` = Riemann on each slot.

### Meta

- **TGR-gm.28**: `regen.jl` — runs all `generators/*.wl` and refreshes `data/*.json`.
- **TGR-gm.29**: Docs snippet in `docs/` covering local regeneration workflow.
- **TGR-gm.30**: License review — GPL `generators/` vs Apache-2.0 `consumers/`
  separation; ensure `test/golden/generators/` is NOT included in the Pkg
  tarball. Ties into the existing HANDOFF TODO "GPL/Apache-2.0 license review".

## Risks

- **BP canonical-form tie-breaking** between xAct and our xperm wrapper on
  edge cases. Mitigation: committed JSON reflects whatever canonical form the
  round-trip converges to; the test asserts *idempotent convergence*, not raw
  byte equality to xAct's first-pass output.
- **Convention drift**. Mitigation: Phase 0 probe + `CONVENTIONS.md`.
- **xAct is GPL**. Generated JSON is data, not linked code, so no
  contamination of Apache-2.0 consumers. Generators live under
  `test/golden/generators/LICENSE` (GPL) and are excluded from the package
  tarball.
- **Schema evolution**. Once JSON is committed, schema changes require
  regeneration. Keep `schema_version` bumped on any breaking change.

## Rollout discipline

- Each phase checkpoints into master before the next starts.
- Every implementation issue produces three commits: `red:`, `green:`,
  optionally `refactor:`.
- After each issue, full test suite runs in background (HANDOFF rule 7).
- Phase 0 is a hard gate — no other phase starts until the conventions
  probe is green on both sides.
