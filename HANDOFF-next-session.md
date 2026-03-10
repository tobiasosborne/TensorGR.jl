# HANDOFF: 6-Deriv Spectrum Pipeline — Session 6

## What Was Done (Sessions 1-5)

### Sessions 1-4
- **TGR-ncdr** ✅ — Kernel extraction (`extract_kernel`, `spin_project`, `contract_momenta`)
- **TGR-0i4m** ✅ — `sym_inv` 3×3
- **TGR-ud97** ✅ — Numerical Lichnerowicz verification
- **TGR-w7jq** ✅ — δ²S term counts on flat (8/4/4/16/18)
- **TGR-mr8p** ✅ — Changed default metric `:η` → `:g` in Barnes-Rivers projectors
- **TGR-5wit** ✅ — Added `_simplify_scalars` step in simplify pipeline (CAS hook)
- **TGR-uy04** ✅ — Symbolics.Num dispatch for k_sq params
- **TGR-7m26** ✅ — Fourier transform + kernel extraction for all 5 flat δ²S terms
- **Verified**: hand-built EH Lichnerowicz kernel → spin-2=-5k²/2, spin-1=0, spin-0s=k², spin-0w=0
- All 4927 tests pass

### Session 5 (this session)
- **Attempted TGR-zq2k** — Barnes-Rivers projection → flat form factors
- **DISCOVERED BLOCKER**: `spin_project` does NOT produce fully reduced scalars
- Root-caused to **invalid dummy index pairs** in the simplify pipeline

---

## CRITICAL BUG: Invalid Dummy Index Pairs

### Symptom
`spin_project(K_EH, :spin2)` returns a TSum with ~50 terms containing uncontracted indices,
metrics, and deltas — instead of a single scalar proportional to k².

### Root Cause
The Fourier-transformed perturbation expressions (`simplify(to_fourier(δ²R))`) have **invalid
dummy index pairs**: the same index name appears twice in the SAME position (both Up or both Down).

**Example**: Term `(1//2) * h[_d1, _d2] * h[-_d1, -_d5] * k[_d2] * k[-_d5]`
- Display convention: `k[a]` = Up, `k[-a]` = Down
- `_d2` appears as Up on both `h` and `k` → INVALID (needs one Up, one Down)
- `_d5` appears as Down on both `h` and `k` → INVALID

### Verification
```julia
# This test confirms the bug:
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp = define_metric_perturbation!(reg, :g, :h)
    set_vanishing!(reg, :Ric); set_vanishing!(reg, :RicScalar); set_vanishing!(reg, :Riem)

    δ2R = simplify(δricci_scalar(mp, 2); registry=reg)
    # δ2R already has 8 invalid index pairs (1 per term) BEFORE Fourier!
    fourier_raw = to_fourier(δ2R)
    # Still 8 invalid pairs (to_fourier preserves positions correctly)
    fourier_simplified = simplify(fourier_raw; registry=reg)
    # Still 8 invalid pairs; canonicalize makes it WORSE (8 → 26 after canonicalize)
end
```

### Impact
- `contract_metrics` requires opposite positions (`midx.position != tidx.position`) → can't contract
- `contract_momenta` requires opposite positions → can't contract k pairs
- `_try_delta_contraction` requires opposite positions → can't contract deltas
- Result: spin_project returns unreduced tensor expressions instead of scalars

### Where the Bug Lives
The invalid pairs originate in `simplify(δricci_scalar(mp, 2))` — the PERTURBATION result
already has them before any Fourier transform. Likely causes:

1. **`_normalize_dummies` in `src/algebra/simplify.jl:84-112`**: Uses `rename_dummy` which
   preserves positions. But the pairs coming from `_analyze_indices` might already be wrong.

2. **`_analyze_indices` in `src/ast/indices.jl:50+`**: Identifies dummy pairs. If it groups
   indices by name without checking for valid Up/Down pairing, it could label same-position
   pairs as "dummies" and then the rename preserves the error.

3. **`canonicalize` (xperm-based)**: The Butler-Portugal canonicalization permutes index slots.
   If the permutation reassigns index names without preserving position pairing, it creates
   invalid pairs. `canonicalize` calls `_normalize_dummies` after xperm which may reassign
   names incorrectly.

### Also Confirmed: `δ_a^b` contraction asymmetry
```julia
# This WORKS:
δ(up(:a), down(:b)) * k(down(:a)) * k(up(:b))  # → k[-b]*k[b] → can contract
# This FAILS:
δ(down(:a), up(:b)) * k(down(:a)) * k(up(:b))  # → unchanged! No contraction!
```
The `_try_delta_contraction` at `src/algebra/contraction.jl:195-239` only contracts when
`didx.position != tidx.position`. When both delta and partner have the same position, no match.
This is correct behavior — the BUG is that same-position pairs shouldn't exist in the first place.

### Possible Fixes (not yet implemented)

**Fix A (root cause)**: Fix `_normalize_dummies` / `canonicalize` to ensure all dummy pairs
have opposite positions. This is the correct fix but may affect xperm canonicalization behavior.
⚠️ CLAUDE.md warns: "Do NOT sort TSum terms or batch-rename in _normalize_dummies — breaks benchmark term counts"

**Fix B (workaround in spin_project)**: Add a `_fix_dummy_positions` function that flips one
position in each same-position pair before the contraction loop. Safe for flat background
(scalar expressions are Lorentz-invariant so position flips don't change the value).

**Fix C (relax contract_momenta)**: Remove the `position != position` check in
`contract_momenta` only (not metrics/deltas). Since k_a k^a = k² = k^a k_a on flat background
with standard metric, contracting by name alone is safe for momentum pairs.

---

## What's Next — Priority Order

### Priority 1: Fix the index pairing bug, then TGR-zq2k

**Step 1**: Investigate `canonicalize` → `_normalize_dummies` flow to find where positions
get corrupted. Check `_analyze_indices` to see if it properly requires opposite positions.
Look at `src/algebra/canonicalize.jl` for how xperm output maps back to index positions.

**Step 2**: Implement a fix (A, B, or C from above). Run full test suite to verify no regression.

**Step 3**: Complete TGR-zq2k using `spin_project`:
- Build combined K_total (5 kernels × coupling constants)
- Project onto spin-2, spin-1, spin-0s, spin-0w
- Verify f₂(k²) = 1 − (α₂/κ)k² − (β₂/κ)k⁴
- Verify f₀(k²) = 1 + (6α₁+2α₂)k²/κ + (6β₁+2β₂)k⁴/κ
- Verify spin-1 = 0 (gauge invariance)
- Evaluate at 100+ random parameter points

### Priority 2: TGR-mphe — dS background quadratic + box terms [P1, READY]
### Priority 3: TGR-c6su — SVT decomposition of δ²S flat [P2, READY]

---

## Dependency Graph (updated)

```
COMPLETED:
  ✅ TGR-ncdr (0.1 kernel) → ✅ TGR-ud97 (0.2 spin proj)
  ✅ TGR-w7jq (1.1 δ²S flat) → ✅ TGR-7m26 (1.2 Fourier+kernel)

BLOCKED BY BUG:
  TGR-zq2k  [P1] Step 1.3: BR flat form factors  ← NEEDS INDEX BUG FIX
  TGR-zq2k ──→ TGR-tztc (cross-check A vs B)
  TGR-zq2k ──→ TGR-j6r9 (tests+benchmark)

READY (independent of bug):
  TGR-mphe  [P1] Step 3.1: dS quad+box terms
  TGR-c6su  [P2] Step 2.1: SVT decompose (Path B)
```

## Key Files

| File | Role |
|------|------|
| `src/action/kernel_extraction.jl` | extract_kernel, spin_project, contract_momenta |
| `src/action/spin_projectors.jl` | Barnes-Rivers P², P¹, P⁰ˢ, P⁰ʷ |
| `src/svt/fourier.jl` | to_fourier (∂_a → k_a) |
| `src/algebra/simplify.jl:84-112` | `_normalize_dummies` — likely bug location |
| `src/algebra/canonicalize.jl` | xperm canonicalization — likely bug location |
| `src/ast/indices.jl:50+` | `_analyze_indices` — check how pairs are identified |
| `src/algebra/contraction.jl` | metric/delta/momentum contraction (requires opposite positions) |
| `test/test_6deriv_spectrum.jl` | All spectrum tests |

## Key Test Commands

```bash
bd ready                              # see unblocked work
bd show TGR-zq2k                      # flat form factors (blocked by bug)
julia --project -e 'using Pkg; Pkg.test()'  # full test suite
bd sync && git push
```

## Display Convention Reference
- `k[a]` = k(Up(:a)), `k[-a]` = k(Down(:a))
- `g[a, b]` = g(Up(:a), Up(:b)), `g[-a, -b]` = g(Down(:a), Down(:b))
- `δ[a, -b]` = δ(Up(:a), Down(:b)) = δ^a_b
