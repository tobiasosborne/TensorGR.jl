# HANDOFF — 2026-03-18 Session Recovery

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

---

## Current Bug Status

### The bugs being fixed (Rule 2 scope)

1. **bench_12 regression**: R³ simplified term count inflated (was 324, currently ~229-362 depending on pipeline state). Root cause: `_avoid` set in perturbation engine producing 35 unique dummy names instead of 9, preventing `_normalize_dummies` from merging same-position pairs.

2. **Spin projection failures**: Kernel extraction gives spin1≠0, spin0w≠0 for Fierz-Pauli kernel (physics requires both = 0 for gauge invariance). Root cause: perturbation engine's δR_{ab} is not manifestly symmetric; inner sum merging corrupts coefficient ratios.

### What the crashed session achieved (uncommitted, in working tree)

Three files modified but NOT committed:

**`src/algebra/canonicalize.jl`** (+85 lines):
- Moved `_sort_partial_chains` here from simplify.jl
- Excluded derivative indices from xperm domain (only tensor indices enter xperm). This prevents xperm from swapping names between derivative and tensor slots.
- Removed partial-derivative symmetry generators from xperm (sorting handled by `_sort_partial_chains` instead)

**`src/algebra/simplify.jl`** (-35 lines):
- Removed `_sort_partial_chains` definition (moved to canonicalize.jl)

**`src/action/kernel_extraction.jl`** (+283 lines):
- `_lower_h_indices_to_down`: Lower all Up h-indices to Down via metric connectors
- `_contract_via_tagged_tensors`: Replace h factors with synthetic `_KL_field`/`_KR_field` tensors, run contract_metrics on full expression, then re-extract bilinear structure
- `_safe_surviving_name!`: Prevent index name collisions during metric contraction in bilinear terms

### Earlier committed work (from 2026-03-17 sessions)

- `11f8ff8`: `canonical_perm_ext` + inner sum collection + `_apply_position_fixes`
- `f51146f`: `flatten_metric_derivs` + `distribute_derivs` improvements
- `5c49566`: Kernel extraction bug identification
- `612a60a`: Kernel metric contraction in coefficients
- `f3444ed`: `δricci_flat`, kernel metric contraction, worklog

### Key findings from xAct research

- xAct does NOT have built-in kernel extraction or spin projections
- Users extract kernels manually via `IndexCoefficient` + `CollectTensors`
- Key enabler: `UseMetricOnVBundle->All` in `ToCanonical` allows Butler-Portugal to merge via metric-aware dummy relabeling
- TensorGR now has this via `canonical_perm_ext`

### Test results as of last run (WORKLOG-canonicalize-fix.md)

| Test | Result | Status |
|------|--------|--------|
| spin2 (with δricci_flat) | 2.5 = FP | PASS |
| spin1 (with δricci_flat) | 0.0 = FP | PASS |
| spin0s (with δricci_flat) | 0.5 ≠ -1.0 | FAIL |
| spin0w (with δricci_flat) | -1.5 ≠ 0.0 | FAIL |
| R³ terms | 229 (was 362) | improved |

### Physics ground truth (the ONLY trustworthy reference)

- K_FP: spin2=2.5k², spin0s=-k², spin1=0, spin0w=0
- K_R²: spin2=0, spin0s=3k⁴, spin1=0, spin0w=0
- K_Ric²: spin2=1.25k⁴, spin0s=k⁴, spin1=0, spin0w=0
- spin-1 and spin-0w MUST be zero for ALL kernels (diffeomorphism invariance)
- On MSS: R₀ = 4Λ, Ric₀_{ab} = Λg_{ab}, Riem₀_{abcd} = (Λ/3)(g_{ac}g_{bd} - g_{ad}g_{bc})

### Immediate next step identified by crashed session

Fix `δricci_scalar_flat`: instead of a separate 3-term formula, TRACE the δricci_flat result:
```julia
d1R = simplify(Tensor(:g, [up(:a), up(:b)]) * δricci_flat(mp, down(:a), down(:b)); registry=reg)
```

---

## Other handoff files (read with Rule 1 skepticism)

- `HANDOFF-canonicalize-investigation.md` — 2026-03-17 root cause analysis
- `HANDOFF-6deriv-crosscheck.md` — Covariant pipeline / √g perturbation approaches
- `HANDOFF-6deriv-spectrum.md` — Full spectrum mission specs
- `HANDOFF-next-session.md` — TOV solver + remaining issues (OUT OF SCOPE per Rule 2)
- `WORKLOG-canonicalize-fix.md` — Detailed change log from 2026-03-17 session

## Key source files

| File | Role |
|------|------|
| `src/algebra/canonicalize.jl` | xperm canonicalization (modified, uncommitted) |
| `src/algebra/simplify.jl` | Simplify pipeline (modified, uncommitted) |
| `src/action/kernel_extraction.jl` | Kernel extraction + spin projection (modified, uncommitted) |
| `src/ast/indices.jl` | `_analyze_indices` — same-position pair recognition (committed) |
| `src/perturbation/expand.jl` | Perturbation engine with `_avoid` set |
| `src/xperm/wrapper.jl` | xperm.c FFI + `canonical_perm_ext` |
| `reference/xAct/` | Local xAct source for research |
