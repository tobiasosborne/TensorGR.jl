# Session Handoff — 2026-03-16 (Session 3)

## Summary

Session 3: restored beads (351 issues), closed 4 issues (+1 auto-closed epic), filed 1 P1 bug. **The P1 bug (TGR-0tm) is the top priority for the next session.**

- Test count: 361,065 → 362,051 (+986 new tests, zero regressions, zero failures)
- 14 `@test_broken` (unchanged, all known: 9 DHOST, 5 RInv)
- 152 closed issues total (148 → 152)
- 1 new P1 bug: **TGR-0tm** — bench_12 canonicalize regression

## What was done this session

### Issues closed
1. **TGR-xlu.6** — Lanczos-Lovelock identity validation (5 tests in test_ddi_rules.jl)
2. **TGR-4zw.2** — Vector field spin projectors: `vector_spin1_projector`, `vector_spin0_projector` (8 tests in test_vector_spin_projectors.jl)
3. **TGR-bgl.3** — PPN conservation/frame checks: `is_fully_conservative`, `is_preferred_frame_free`, `is_preferred_location_free`, `is_semi_conservative`, named constructors `PPNParameters(:GR/:BransDicke/:Nordtvedt/:Rosen)` (added to test_ppn_metric.jl)
4. **TGR-xlu** (epic) — auto-closed when TGR-xlu.6 completed

### Beads restored
The Dolt database was missing on session start. Restored from the git-committed `.beads/backup/` files (extracted from commit `d8c413c`) via `bd backup restore`. 351 issues recovered.

---

## CRITICAL: TGR-0tm — bench_12 Canonicalize Regression

### What is broken

The 6-derivative gravity benchmark (bench_12) has a term-count regression. All 6 cubic curvature invariants on de Sitter produce roughly **2x the expected simplified term counts**:

| Invariant | Expected | Actual | Ratio |
|-----------|----------|--------|-------|
| R³ | 324 | 685 | 2.11 |
| R·Ric² | 1042 | 2384 | 2.29 |
| Ric³ | 1144 | 2398 | 2.10 |
| R·Riem² | 1344 | 2568 | 1.91 |
| Ric·Riem² | 1202 | 2463 | 2.05 |
| Riem³ | 1488 | 2416 | 1.62 |

The **unit tests all pass** (362,051). This is a canonicalization quality regression, not a correctness bug — the simplifier produces valid but insufficiently simplified expressions.

### Root cause: commit history

The ground truth was correct at commit **`058f035`** ("Fix deep simplifier bugs"). Four subsequent commits modified `src/algebra/canonicalize.jl`:

#### Commit 1: `1faf32f` — "Fix invalid dummy index pairs in canonicalization"
**This is the primary regression commit.** It changed two things:
1. **Added generator conjugation**: Before passing generators to xperm, it conjugates them via `perm ∘ gen_slot ∘ perm⁻¹`
2. **Switched reconstruction**: Changed from `cperm_inv.data[slot]` to `cperm.data[slot]`

The stated motivation was fixing "corrupt Up/Down dummy pairing in multi-tensor products" that blocked Barnes-Rivers spin projection (TGR-zq2k). The commit also **relaxed term-count tests from `==` to `>=`** in test_6deriv_spectrum.jl, masking the regression.

R³ went from **324 → 505 terms** at this commit.

#### Commit 2: `59e4e81` — "Add fix_dummy_positions"
Added `fix_dummy_positions()` utility that repairs same-position dummy pairs by flipping one occurrence. This is an independent post-processing function — it does NOT modify the canonicalize pipeline itself. **This commit is fine.**

#### Commit 3: `25e9fa3` — "normalize_field_positions"
Added `normalize_field_positions()` for lowering field indices. Independent utility. **This commit is fine.**

#### Commit 4: `355d298` — "Extend xperm canonicalization for spinor index symmetries"
Changed the name-assignment sort key from `by = i -> all_indices[i].name` to `by = i -> (string(all_indices[i].vbundle), all_indices[i].name)`. This groups names by VBundle, preventing cross-vbundle name swaps. Also made dummy classification vbundle-aware. **The vbundle changes are correct and necessary for spinors**, but the sort-key change alters the `perm_data` values, which interacts with the conjugation from commit 1. R³ went from **505 → 685 terms** (further degradation).

### The xperm convention question

The core mathematical question is: **what does `canonical_perm` in xperm.c expect?**

Per `docs/xperm_algorithm.md` (Section 2):
- `PERM` input: slot→slot permutation (xAct convention)
- `GS` generators: slot-to-slot permutations
- Inside, `canonical_perm` converts to Renato's notation: `PERM1 = PERM⁻¹` (slot→name)
- Output `CPERM = PERM2⁻¹` converted back to xAct convention

**However**, the Julia code constructs `perm_data[slot] = slot_to_name[slot]`, which is a **slot→name** mapping (Renato notation), NOT a slot→slot map (xAct convention). This means the code is passing the WRONG type of permutation to `canonical_perm`, which expects slot→slot.

The baseline code (058f035) passed this slot→name map with unconjugated slot-space generators and used `cperm_inv` for reconstruction. This **happened to work** and produced strong canonicalization (324 terms for R³).

Commit 1faf32f added conjugation `perm ∘ gen ∘ perm⁻¹` to "fix" the convention mismatch, and switched to `cperm` for reconstruction. The mathematical intent was to transform slot-space generators into name-space generators to compensate for passing a slot→name map. But this produces **weaker canonicalization** (505→685 terms) because the conjugated generators and direct cperm usage lead to xperm exploring a different (smaller effective) orbit in its internal coset_rep algorithm.

### What needs to happen

**The next agent must properly understand the xperm algorithm and fix the canonicalization.** There are essentially three approaches:

1. **Revert conjugation + verify spin projection still works.** The baseline code at 058f035 produced correct bench_12 results. Commit 59e4e81 added `fix_dummy_positions` as a post-processing utility that can handle the same-position dummy pairs that motivated the conjugation in the first place. So: revert the conjugation (use unconjugated generators + `cperm_inv`), keep the vbundle-aware sort from 355d298, and test whether all 362k+ tests AND bench_12 all pass together. If spin projection dummy pairs are still broken, `fix_dummy_positions` should handle them.

2. **Fix the convention properly.** Instead of passing a slot→name map to `canonical_perm` (which expects slot→slot), construct a proper slot→slot permutation. This means the "name assignment" step needs rethinking. The `perm_data` should encode a proper slot permutation that, when xperm internally inverts it, produces the desired slot→name mapping.

3. **Use `canonical_perm_ext` directly.** Skip the `canonical_perm` wrapper and call `canonical_perm_ext` with Renato's notation directly: pass the slot→name map as PERM (which is what the ext function expects), pass slot-space generators (unchanged), and pass name-based free/dummy lists. This avoids the double-inversion confusion.

### How to test the fix

```bash
# Quick smoke test (R³ should give 324):
julia --project -e '
using TensorGR
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    maximally_symmetric_background!(reg, :M4; metric=:g, cosmological_constant=:Λ)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp = define_metric_perturbation!(reg, :g, :h; curved=true)
    R1 = Tensor(:RicScalar, TIndex[])
    expr = R1 * R1 * R1
    raw = expand_perturbation(expr, mp, 2)
    s = simplify(raw; registry=reg, maxiter=100)
    n = s isa TSum ? length(s.terms) : 1
    println("R^3 terms: $n (expected: 324)")
end'

# Full test suite (must pass 362k+, 0 failures):
julia --project -e 'using Pkg; Pkg.test()'

# Full bench_12 (all 6 invariants must match pinned counts):
julia -t4 --project=benchmarks benchmarks/bench_12_6deriv_dS.jl
```

### Key files

| File | Role |
|------|------|
| `src/algebra/canonicalize.jl` | **THE file to fix** — `_canonicalize_product` function |
| `docs/xperm_algorithm.md` | Complete analysis of xperm.c conventions (603 lines, READ THIS FIRST) |
| `deps/xperm.c` | The C implementation (2400 lines) |
| `src/xperm/wrapper.jl` | Julia FFI wrapper for xperm |
| `src/xperm/permutations.jl` | Perm type definition |
| `benchmarks/bench_12_6deriv_dS.jl` | Ground truth term counts (lines 23-28) |
| `test/test_6deriv_spectrum.jl` | Spectrum tests (may have relaxed `>=` that should be `==`) |
| `benchmarks/ground_truth.jl` | Shared ground truth constants |

### Commit archaeology commands

```bash
# Baseline (working) canonicalize.jl:
git show 058f035:src/algebra/canonicalize.jl

# The regression commit:
git show 1faf32f:src/algebra/canonicalize.jl
git diff 058f035..1faf32f -- src/algebra/canonicalize.jl

# All 4 commits in order:
git log --oneline 058f035..HEAD -- src/algebra/canonicalize.jl

# Parent of regression (last known good):
git show 0806808:src/algebra/canonicalize.jl
```

---

## Other session state

### Beads
- 352 total issues, 152 closed, 48 ready
- Backup synced to git and pushed
- `bd ready` shows current work queue
- `bd show TGR-0tm` for the regression bug details

### What else is ready (after fixing TGR-0tm)
- TGR-443.1.5: RInv bidirectional conversion
- TGR-lej: Abstract tetrad indices in AST
- TGR-bgl.13: PPN-to-component bridge
- TGR-4zw.3: Antisymmetric rank-2 field spin projectors (unblocked by 4zw.2)
- TGR-bgl.4: PPN velocity-order expansion (unblocked by bgl.3)
- 43 more ready issues (run `bd ready -n 48`)

### Key patterns for next agent

1. **FIX TGR-0tm FIRST.** Everything else is secondary.
2. **Read `docs/xperm_algorithm.md` completely** before touching canonicalize.jl.
3. **Test with bench_12 R³ term count** as the primary regression test (must be 324).
4. **Test with full test suite** to ensure no other regressions.
5. **Beads**: `bd ready`, `bd show <id>`, `bd update <id> --claim`, `bd close <id>`
6. **Worktree isolation**: Implementation uses `isolation: worktree`. Merge conflicts in runtests.jl and TensorGR.jl.
7. **Ground truth**: Every test MUST cite actual equation numbers from references.
8. **Test pattern**: src/<subsystem>/file.jl + test/test_file.jl + include in TensorGR.jl + runtests.jl.
