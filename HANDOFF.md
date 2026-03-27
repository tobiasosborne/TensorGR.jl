# HANDOFF — 2026-03-27 (Session 18: REPL UX fixes, Bianchi reduction, parser brackets)

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

- **540 issues total** (495 closed, 35 open) — Dolt DB fully synced with JSONL
- **Full test suite: NOT YET RUN this session** — changes need testing before push
- **Benchmarks: Tier 1-3 ALL PASS** (as of prior session)
- Last pushed commit: `94a6fcb` on `master`
- `bd stats` for live counts

---

## ⚠ CRITICAL: UNCOMMITTED CHANGES — TEST BEFORE PUSH

All changes below are **uncommitted**. The next agent MUST:

1. **Kill any running Julia** (`ps aux | grep julia` — Tobias's REPL may still be running)
2. **Run the full test suite**: `julia --project -e 'using Pkg; Pkg.test()'`
3. **Run benchmarks tier 1-3**: `julia -t4 --project=benchmarks benchmarks/run_all.jl --tier 3`
4. **If all pass**: commit with message below, push to master, close issues
5. **If tests fail**: fix the failures, DO NOT push broken code

### Suggested commit message

```
Add REPL UX fixes + Bianchi identity reduction (TGR-2ai)

REPL tensor mode (src/repl/tensor_mode.jl):
- Function-call syntax: simplify(expr), contract(%), level2(%)
- Variable assignment: expr = R_{abcd}, simplify varname
- Up-arrow history recall via _record_history in on_done callback
- New commands: level2, simplify_level2, to_riemann, to_ricci

LaTeX parser (src/parser/latex_parser.jl):
- Antisymmetrization brackets: R_{a[bcd]} → antisymmetrize(R, [:b,:c,:d])
- Symmetrization parentheses: T_{(ab)} → symmetrize(T, [:a,:b])
- New _IndexGroupResult struct for bracket position tracking

Bianchi identity (src/invariants/simplify_levels.jl):
- simplify_level2 now applies first Bianchi via _bianchi_reduce_direct
- Greedy rewrite: R_{abcd} = -R_{acdb} - R_{adbc}, keep if term count drops
- R_{a[bcd]} correctly simplifies to zero with level2 command
- Fixed wrong comment in bianchi.jl (xperm does NOT handle multi-term Bianchi)

Tests: ~20 new tests (parser brackets, REPL function-call + variables)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

### Issues to close after push

```bash
bd close TGR-2ai.1  # LaTeX parser brackets
bd close TGR-2ai.2  # Variable assignment
bd close TGR-2ai.3  # Function-call syntax
bd close TGR-2ai.4  # Up-arrow history
```

---

## What Was Done This Session (Session 18)

### 1. REPL Mode Epic (TGR-2ai): 4 P1 bugs fixed + 6 P2/P3 features filed

**Epic**: TGR-2ai — REPL Mode Improvements (10 children)

**Bugs fixed (P1):**

**TGR-2ai.1 — LaTeX parser antisymmetrization/symmetrization brackets**
- Files: `src/parser/latex_parser.jl`
- Added `[` `]` tokens to tokenizer (`:lbracket`, `:rbracket`)
- New `_IndexGroupResult` struct tracks bracket positions within index groups
- `_parse_index_group` detects `[...]` (antisymmetrize) and `(...)` (symmetrize) inside `_{...}` brace groups
- `_parse_tensor` applies `antisymmetrize()`/`symmetrize()` on bracketed index positions
- Expression-level `(...)` still works for grouping (separate code path in `_parse_atom`)
- Tests: 7 new tests in `test/test_latex_parser.jl`

**TGR-2ai.2 — Variable assignment in REPL**
- File: `src/repl/tensor_mode.jl`
- Added `TensorREPL._variables` dict (`Dict{String, Any}`)
- `_process_tensor_input` detects `name = expr` pattern via regex before command loop
- `_process_tensor_rhs` handles RHS as command, variable ref, or LaTeX
- Bare variable names recall stored values
- Commands accept variable names as arguments: `simplify myvar`
- Tests: 5 new tests in `test/test_repl_mode.jl`

**TGR-2ai.3 — Function-call syntax**
- File: `src/repl/tensor_mode.jl`
- Command loop now matches `cmd(arg)` in addition to `cmd arg`
- Extracted `_apply_command` helper shared by space and paren syntax
- Tests: 3 new tests in `test/test_repl_mode.jl`

**TGR-2ai.4 — Up-arrow history recall**
- File: `src/repl/tensor_mode.jl`
- Added `_record_history(prompt, line)` that pushes to `hp.history`/`hp.modes`
- Called from `on_done` callback before processing input
- Uses `:tensor` mode tag for mode-filtered history navigation

**New REPL commands added:**
- `level2` / `simplify_level2` — Bianchi-aware simplification
- `to_riemann` — curvature basis conversion
- `to_ricci` — curvature basis conversion

**Features filed (not implemented):**
- TGR-2ai.5 (P2): Tab completion of tensor names
- TGR-2ai.6 (P2): Numbered output history (%1, %2, %N)
- TGR-2ai.7 (P2): Additional commands (covd, perturbation, substitute, define)
- TGR-2ai.8 (P2): vars/info workspace introspection
- TGR-2ai.9 (P3): Pipe/chain syntax
- TGR-2ai.10 (P3): Derivative shorthands

### 2. First Bianchi identity: simplify_level2 now works

**Problem**: `simplify_level2(R_{a[bcd]})` returned 3 terms instead of 0. The comment in `bianchi.jl:27` claimed "algebraic Bianchi is already captured by RiemannSymmetry in xperm" — this was **wrong**. xperm only handles monoterm symmetries. The Bianchi identity `R_{abcd} + R_{acdb} + R_{adbc} = 0` is multi-term.

**Full workflow followed** (Rules 4+5):
1. **xAct research agent**: Confirmed xAct handles this via Invar pre-stored rule database (scalar invariants only). For free-index expressions, xAct users register explicit `MakeRule` rewrite rules. xperm never sees the Bianchi identity.
2. **Solution A (TRInv-based)**: Convert TensorExpr → TRInv, use existing `bianchi_relations_trinv` + Gaussian elimination, convert back. Complex: ~150 lines, free index name reconstruction required.
3. **Solution B (direct rewrite)**: Greedy TensorExpr-level rewrite. For each Riemann term, try `R_{abcd} = -R_{acdb} - R_{adbc}`, re-simplify, keep only if term count decreases. ~70 lines, no conversion layer.
4. **Chose Solution B**: simpler, lower-risk, preserves free index names naturally.
5. **Reviewer agent**: PASS on all 7 checklist items (math correctness, termination, edge cases).

**Changes**:
- `src/invariants/simplify_levels.jl`: Replaced `simplify_level2` body with `simplify` + `_bianchi_reduce_direct`. Added `_bianchi_reduce_direct`, `_extract_riem_indices`, `_rebuild_with_riem` helpers.
- `src/gr/bianchi.jl`: Fixed comment (line 27) — multi-term Bianchi handled by `_bianchi_reduce_direct`, NOT by xperm.
- Algorithm: greedy, monotone (only reduces term count), guaranteed termination.

**Risk**: LOW — `_bianchi_reduce_direct` is only called from `simplify_level2` (opt-in), never from the default `simplify` pipeline. Existing tests/benchmarks unaffected.

---

## ⚠ Core Changes To Monitor

**This session** (uncommitted):

**Bianchi reduction** (simplify_level2):
- Location: `src/invariants/simplify_levels.jl` (lines ~186-290)
- Risk: LOW — opt-in via `simplify_level2`, not in default `simplify` pipeline
- Revert: restore old `simplify_level2` body (just `simplify_level1 + simplify`)

**LaTeX parser brackets**:
- Location: `src/parser/latex_parser.jl`
- Risk: LOW — additive (new token types, new struct, `(` in index groups)
- Key invariant: expression-level `(...)` grouping unchanged (tested)
- Revert: restore old `_parse_index_group` (returns `Vector{TIndex}`), old `_parse_tensor`

**REPL mode**:
- Location: `src/repl/tensor_mode.jl`
- Risk: LOW — additive features, no changes to existing command behavior
- Revert: restore old `_process_tensor_input`, remove `_variables` dict

---

## Open Issues (prior + new)

| ID | P | Title | Notes |
|----|---|-------|-------|
| `TGR-byb` | P2 | BinaryBuilder for xperm.c | Yggdrasil recipe. Blocks Pkg registration. |
| `TGR-erv` | P2 | Pkg registration | Requires BinaryBuilder or deps/build.jl. |
| `TensorGR.jl-6e8` | P2 | Collect xAct papers corpus | Research task. |
| `TGR-2ai` | P1 | REPL Mode Improvements (epic) | 4 P1 bugs done, 6 P2/P3 features open |
| `TGR-2ai.5` | P2 | Tab completion | Not implemented |
| `TGR-2ai.6` | P2 | Numbered output history | Not implemented |
| `TGR-2ai.7` | P2 | Additional commands | Not implemented |
| `TGR-2ai.8` | P2 | vars/info commands | Not implemented |
| `TGR-2ai.9` | P3 | Pipe/chain syntax | Not implemented |
| `TGR-2ai.10` | P3 | Derivative shorthands | Not implemented |

---

## Key Decisions / Lessons

### Carried from previous sessions
- All decisions from session 17 HANDOFF still apply
- Cross-project parallel Julia is OK (different --project paths)

### New this session (session 18)
- **xperm does NOT handle multi-term symmetries**: The first Bianchi identity `R_{a[bcd]}=0` is multi-term. xperm/Butler-Portugal only handles monoterm (permutation) symmetries. This was incorrectly claimed in `bianchi.jl:27` (now fixed).
- **Bianchi reduction strategy**: Direct greedy rewrite (Solution B) chosen over TRInv Gaussian elimination (Solution A). B is simpler, preserves free index names, lower regression risk.
- **Parser bracket context**: `(` inside `_{...}` means symmetrization; `(` at expression level means grouping. These are separate code paths (`_parse_index_group` vs `_parse_atom`).
- **REPL history**: Julia's `REPLHistoryProvider` has parallel `history::Vector{String}` and `modes::Vector{Symbol}` vectors. Must push to both with matching `:tensor` mode tag.
- **AbstractString vs String**: `strip()` returns `SubString`, Dict{String,...} keys need `String()` conversion. Functions accepting user input should use `AbstractString` parameter types.

---

## ⚠ BEADS DATABASE: CRITICAL CROSS-DEVICE INSTRUCTIONS

The beads Dolt DB (`.beads/dolt/`) is **gitignored and local** to each machine.
The JSONL (`.beads/issues.jsonl`) is **git-tracked** and is the source of truth.

**On EVERY new device or fresh clone:**
1. Update bd: `go install github.com/steveyegge/beads/cmd/bd@latest && cp ~/go/bin/bd ~/.local/bin/bd`
2. Verify version: `bd --version` must be **v0.62.0+** (older versions lack `bd import`)
3. Import JSONL: `bd import` (imports `.beads/issues.jsonl` into Dolt)
4. Verify: `bd stats` should show ~540 total issues

**After closing/creating issues:** `bd export -o .beads/issues.jsonl` to flush Dolt → JSONL, then git commit the JSONL.

**ROOT CAUSE of repeated desyncs**: bd v0.58.0 lacked `bd import`. Sessions claimed to import but the command didn't exist. Fixed by upgrading to v0.62.0.

---

## TODO Next Session

1. **RUN TESTS** — full suite + benchmarks (changes not yet tested!)
2. **Audit remaining ~35 open issues** for stale ones (audit agent was started but not finished)
3. **REPL tab-completion** (TGR-2ai.5) — most impactful remaining REPL feature
4. **Einstein trace rule** and **Weyl trace-free rule** — G^a_a=-R and g^{ac}C_{abcd}=0
5. **`deps/build.jl`** for cross-platform xperm.c compilation
6. **GPL/Apache-2.0 license review** — xperm.c is GPL, rest is Apache-2.0

## Quick Commands

```bash
bd stats                    # project health
bd list --status=open       # remaining open issues
bd list --parent TGR-2ai    # REPL epic status
julia --project -e 'using Pkg; Pkg.test()'  # full test suite
julia -t4 --project=benchmarks benchmarks/run_all.jl --tier 3  # all benchmarks
git log --oneline -15       # recent commits
git diff --stat             # uncommitted changes summary
```
