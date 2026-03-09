# HANDOFF: 6-Deriv Spectrum Pipeline — Session 3

## What Was Done (Sessions 1-2)

### Session 1
- **TGR-ncdr** ✅ — Kernel extraction (`extract_kernel`, `spin_project`, `contract_momenta` in `src/action/kernel_extraction.jl`)
- **TGR-0i4m** ✅ — `sym_inv` 3×3 in `src/action/quadratic_action.jl`

### Session 2
- **TGR-ud97** ✅ — `spin_project` verified numerically against EH Lichnerowicz kernel (spin-2=k², spin-1=0, spin-0s=-k²/2, spin-0w=-k²/2, residual ~10⁻¹⁵)
- **TGR-w7jq** ✅ — All 5 δ²S expressions on flat: δ²R=8, (δR)²=4, (δRic)²=4, δ²(R□R)=16, δ²(Ric□Ric)=18 terms
- Tests added to `test/test_6deriv_spectrum.jl` (extract_kernel, contract_momenta, term counts)
- **Diagnosed simplifier gaps** blocking full symbolic reduction of spin-projected expressions
- Created 3 new issues for the root causes (see below)

---

## Simplifier Gaps (MUST FIX before TGR-zq2k)

Three root causes prevent `spin_project` from reducing to scalar form factors:

### TGR-mr8p [P1 bug] — η metric not registered
**The biggest blocker.** Barnes-Rivers projectors default to `metric=:η`, but η is never registered in the TensorRegistry. `contract_metrics` (src/algebra/contraction.jl:20) checks `has_tensor(reg, :η)` → FALSE, and **silently skips ALL η contractions**. No raising/lowering, no traces, nothing.

**Fix options:**
- A) `spin_project` should default to the manifold's registered metric (`:g`), not hardcoded `:η`
- B) Add warning/error in `contract_metrics` when it encounters an unregistered metric-like tensor

**Files:** `src/action/spin_projectors.jl` (defaults), `src/action/kernel_extraction.jl` (spin_project), `src/algebra/contraction.jl` (silent skip)

### TGR-5wit [P1 feature] — Scalar simplification hooks unused in pipeline
`simplify_scalar()` and `_simplify_scalar_val()` hooks exist (src/scalar/simplify_cas.jl) and the Symbolics.jl extension overrides them — but they're **never called during `simplify()`**.

TScalar values with Expr/Symbol (like `:(1/k²)`, `:k²`) are opaque cores in `_split_scalar` (line 158 in simplify.jl). Products of such TScalars accumulate as separate factors without cancellation.

**Fix:** Add a scalar factor simplification step in `_simplify_one_pass()` (after canonicalize, before collect_terms). Walk TProduct factors, find TScalar values, call `_simplify_scalar_val()` on their product.

**Files:** `src/algebra/simplify.jl` (_simplify_one_pass, _split_scalar), `src/scalar/simplify_cas.jl` (hooks)

### TGR-uy04 [P1 feature] — Use Symbolics.Num for k² (depends on TGR-5wit)
Barnes-Rivers uses `TScalar(:k²)` and `TScalar(:(1/k²))` — bare Julia Symbols/Exprs that even the fixed hooks can't simplify. Need `@variables k²` → `Symbolics.Num` values so the CAS extension handles `k² * (1/k²) → 1` automatically.

**Fix:** Update `omega_projector` to accept Num values (currently builds `:(1/$k_sq)` Expr which fails for Num). Update `contract_momenta` to produce `TScalar(k²::Num)`.

**Files:** `src/action/spin_projectors.jl` (omega_projector), `src/action/kernel_extraction.jl` (contract_momenta)

---

## Dependency Graph (Updated)

```
READY NOW (parallel):
  TGR-mr8p  [P1] Fix η metric registration           ← CRITICAL (simplifier bug)
  TGR-5wit  [P1] Scalar simplification in pipeline    ← CRITICAL (dead hooks)
  TGR-7m26  [P1] Fourier + kernel extraction (flat)
  TGR-mphe  [P1] dS background quadratic + box terms
  TGR-c6su  [P2] SVT decomposition (flat, Path B)

BLOCKED:
  TGR-5wit ──→ TGR-uy04 (CAS k² variables)
  TGR-mr8p + TGR-5wit + TGR-uy04 + TGR-7m26 ──→ TGR-zq2k (flat form factors)
  TGR-zq2k + TGR-pr04 ──→ TGR-tztc (cross-check Path A vs B)
  TGR-mphe ──→ TGR-7tcs (dS cubics) ──→ TGR-ug98 (full dS spectrum)
  TGR-zq2k + TGR-tztc + TGR-ug98 ──→ TGR-j6r9 (tests) ──→ TGR-af4a (example)
```

## Recommended Session 3 Strategy

**Priority 1 — Fix simplifier (must do first):**
1. TGR-mr8p: Fix η metric default → use registered metric `:g` in spin_project
2. TGR-5wit: Wire `_simplify_scalar_val` into `_simplify_one_pass`
3. TGR-uy04: Switch k² to Symbolics.Num in projectors + contract_momenta

**Priority 2 — Continue pipeline (parallel with simplifier fixes):**
4. TGR-7m26: Fourier + extract_kernel for all 5 δ²S
5. TGR-c6su: SVT decomposition (independent Path B)

**After simplifier fixes + kernels:**
6. TGR-zq2k: `spin_project` all 5 kernels → verify f₂(z), f₀(z) symbolically

---

## Pinned Term Counts

| Expression | Terms (flat, vanishing bg) |
|---|---|
| δ²R | 8 |
| (δR)² | 4 |
| (δRic)² | 4 |
| δ²(R□R) | 16 |
| δ²(Ric□Ric) | 18 |

## Ground Truth (flat spectrum)

```
f₂(z) = 1 − (α₂/κ)z − (β₂/κ)z²     [spin-2, z = k²]
f₀(z) = 1 + (6α₁+2α₂)z/κ + (6β₁+2β₂)z²/κ  [spin-0]
spin-1 projection = 0 identically
```

## Key Files

| File | Role |
|------|------|
| `src/action/kernel_extraction.jl` | extract_kernel, spin_project, contract_momenta |
| `src/action/spin_projectors.jl` | Barnes-Rivers P², P¹, P⁰ˢ, P⁰ʷ, T^sw, T^ws |
| `src/algebra/simplify.jl` | _simplify_one_pass (pipeline), _split_scalar |
| `src/algebra/contraction.jl` | contract_metrics (silent skip on unregistered metrics) |
| `src/scalar/simplify_cas.jl` | simplify_scalar, _simplify_scalar_val (unused hooks) |
| `ext/TensorGRSymbolicsExt.jl` | Symbolics.jl dispatch for scalar hooks |
| `test/test_6deriv_spectrum.jl` | Tests: kernel, term counts, projector completeness |
| `HANDOFF-6deriv-spectrum.md` | Master handoff (full pipeline spec, all issues) |
| `examples/13_6deriv_particle_spectrum.jl` | Ground truth: numerical form factors |

## Quick Start

```bash
bd ready                              # see unblocked work
bd blocked                            # see dependency chain
bd show TGR-mr8p                      # η metric bug details
bd show TGR-5wit                      # scalar hooks details

# Run existing tests
julia --project -e 'using Pkg; Pkg.test()'

# Session end protocol
bd sync && git push
```
