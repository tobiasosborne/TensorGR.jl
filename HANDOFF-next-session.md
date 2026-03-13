# HANDOFF: Session 27 — TGR-76k Deep Diagnosis, 2 New Bugs Filed

## Status: 10 closed, 20 open (7 ready, 11 blocked, 2 new bugs)

- **All tests pass**: full suite unchanged
- **No code modified** — this was a diagnosis-only session

## What Was Done This Session

Attempted TGR-76k (P1): validate dS crosscheck via perturbation engine pipeline.
Ran `examples/15_perturbation_spectrum_crosscheck.jl` — **Step 1 (EH) fails**:

```
Λ→0 limit (k²=1.7):
  Tr(K·P²)  = 10.625,  expected 4.25    ← 2.5x too large
  Tr(K·P¹)  = 0.0,     expected 0       ← OK
  Tr(K·P⁰ˢ) = -1.7,    expected -1.7    ← OK
  Tr(K·P⁰ʷ) = 0.0,     expected 0       ← OK
Gauge sectors at Λ=0.3:
  Tr(K·P¹)  = 0.45   (should be 0)      ← LEAKS
  Tr(K·P⁰ʷ) = 0.15   (should be 0)      ← LEAKS
```

### Root Cause Identified: Two Bugs

**Bug 1 — TGR-dp3 (P1)**: Canonicalizer fails to combine equivalent tensor
structures at mixed index positions in Fourier space. This is the **primary blocker**.

**Bug 2 — TGR-wm8 (P2)**: √g coefficient in example 15 is wrong (1/4 should be 1/8).
Secondary — even with correct coefficient, Bug 1 prevents validation.

## New Issues Created

| ID | P | Type | Title | Blocks |
|----|---|------|-------|--------|
| TGR-dp3 | P1 | bug | Canonicalizer fails to combine symmetric tensor traces at mixed index positions | TGR-76k |
| TGR-wm8 | P2 | bug | Revert √g coefficient from 1/4 back to 1/8 in example 15 | TGR-76k |

---

# DEEP RESEARCH: TGR-dp3 — Canonicalization Bug

## The Problem In Detail

After Fourier-transforming δ²R (second-order Ricci scalar variation), the simplifier
produces **8 terms** instead of the **4 terms** that make up the Fierz-Pauli kernel.

The pre-built FP kernel (`build_FP_momentum_kernel` in `src/action/kernel_extraction.jl:302`)
works **perfectly** with `spin_project` — gives exact (5/2)k², 0, -k², 0. So the
spin projectors and evaluation are correct. The bug is purely in the perturbation
engine → simplify path.

### Concrete Reproducer

```julia
using TensorGR
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    @define_tensor k on=M4 rank=(0,1)
    mp = define_metric_perturbation!(reg, :g, :h; curved=false)
    set_vanishing!(reg, :Ric)

    δ2R = simplify(δricci_scalar(mp, 2); registry=reg, maxiter=200)
    Qf = to_fourier(δ2R)
    Qf = simplify(Qf; registry=reg)
    Qf = fix_dummy_positions(Qf)
    # Qf has 8 terms — should have 4
    # Specifically, these 8 terms should combine into the 4 FP terms
end
```

### The 8 Terms (Actual) vs 4 Terms (Expected)

Actual Fourier δ²R output:
```
1. -h[_d1, _d2] * h[-_d1, -_d3] * k[-_d2] * k[_d3]
2. -h[-_d1, -_d2] * h[_d2, _d3] * k[_d1] * k[-_d3]
3. (5//4) * h[_d1, _d2] * h[-_d1, -_d2] * k[_d3] * k[-_d3]
4. (-1//4) * h[_d1, -_d1] * h[_d2, -_d2] * k[_d3] * k[-_d3]
5. (1//2) * h[-_d1, _d2] * h[_d3, -_d3] * k[_d1] * k[-_d2]
6. (-1//2) * h[_d1, _d2] * h[-_d2, -_d3] * k[-_d1] * k[_d3]
7. (1//2) * h[-_d1, _d1] * h[-_d2, _d3] * k[_d2] * k[-_d3]
8. (1//2) * h[_d1, _d2] * h[_d3, -_d3] * k[-_d1] * k[-_d2]
```

Expected FP kernel (from `build_FP_momentum_kernel`):
```
A. (1//2) * k² * h_{ab} h^{ab}              — trace h·h with k²
B. (-1) * k_b k_c * h^{ab} h^c_a            — h²-type contraction with 2 k's
C. (+1) * k_a k_b * h^{ab} * tr(h)          — h·tr(h) with 2 k's
D. (-1//2) * k² * tr(h)²                    — trace squared with k²
```

### Why Terms Don't Combine

The tensor structures are metrically equivalent but have **different index positions
on h**. For a symmetric tensor h, these are all the same:
- `h[up(a), down(b)]` = `h[down(b), up(a)]` (symmetry)
- `h[up(a), up(b)]` = `g^{bc} h[up(a), down(c)]` (metric equivalence)

But `collect_terms` in `src/algebra/simplify.jl:48` compares terms structurally
after `_normalize_dummies` — it does NOT apply metric contractions or use h's
symmetry to normalize index positions.

**Example**: Terms 1, 2, and 6 are all of the form `-(h²)^{μν} k_μ k_ν`
(matrix product h·h contracted with two momenta), but they have h indices at
different positions:
- Term 1: `h^{ab} h_{ac} k_b k^c` — both h's contract through first index
- Term 2: `h_{ab} h^{bc} k^a k_c` — h's contract through second/first index
- Term 6: `h^{ab} h_{bc} k_a k^c` — same structure, different slot positions

These should sum to (-1 + -1 + -1/2) × (structure B) = -5/2, but only -1
appears in the expected FP result. So actually, the perturbation engine is
generating CORRECT terms that are equivalent to FP, but the coefficients
appear distributed differently because they're not being combined.

Similarly, terms 3 and the trace-like term 13 (from term `h[_d1, -_d2]*h[-_d1, _d2]*k²`)
in the un-set_vanishing run would combine to give the correct 1/2 coefficient.

### Even δR (First Order) Shows The Bug

```
δR Fourier = h[-_d1, -_d2] * k[_d1] * k[_d2]
           + (-1//2) * h[_d1, -_d1] * k[_d2] * k[-_d2]
           + (-1//2) * h[-_d1, _d1] * k[_d2] * k[-_d2]
```

Should be 2 terms: `k^a k^b h_{ab} - k² tr(h)`. The two trace terms
`h[_d1, -_d1]` and `h[-_d1, _d1]` differ only in which slot is Up vs Down.
For symmetric h, these are identical but `collect_terms` doesn't recognize this.

## Where To Fix: Recommended Approach

### Option A: Normalize field index positions post-Fourier (RECOMMENDED)

Add a function `normalize_field_positions(expr, field; metric)` that:
1. Walks the expression tree
2. For each product term, finds all `Tensor(field, indices)` factors
3. Lowers all field indices to Down by inserting `g^{old_up, fresh_down}` connectors
4. Then calls `simplify` which will `contract_metrics` and `collect_terms`

Insert this step in the crosscheck pipeline between `to_fourier` and `extract_kernel`.

This is the **least invasive** fix — it doesn't change the core simplify pipeline.

**Location**: Add to `src/action/kernel_extraction.jl` (near `fix_dummy_positions`)
or `src/algebra/canonicalize.jl`.

**Sketch**:
```julia
function normalize_field_positions(expr::TSum, field::Symbol; metric::Symbol=:g)
    tsum([normalize_field_positions(t, field; metric) for t in expr.terms])
end

function normalize_field_positions(p::TProduct, field::Symbol; metric::Symbol=:g)
    new_factors = TensorExpr[]
    all_names = Set(idx.name for f in p.factors for idx in indices(f))
    for f in p.factors
        if f isa Tensor && f.name == field
            lowered_idxs = TIndex[]
            connectors = TensorExpr[]
            for idx in f.indices
                if idx.position == Up
                    fn = fresh_index(all_names)
                    push!(all_names, fn)
                    push!(lowered_idxs, TIndex(fn, Down, idx.vbundle))
                    push!(connectors, Tensor(metric, [idx, TIndex(fn, Up, idx.vbundle)]))
                else
                    push!(lowered_idxs, idx)
                end
            end
            push!(new_factors, Tensor(field, lowered_idxs))
            append!(new_factors, connectors)
        else
            push!(new_factors, f)
        end
    end
    tproduct(p.scalar, new_factors)
end
```

After lowering, `simplify` will contract the inserted metrics and `collect_terms`
can then combine terms with matching structure.

### Option B: Fix collect_terms to be metric-aware (MORE INVASIVE)

Make `_normalize_dummies` or `canonicalize` insert metric contractions to
standardize index positions before comparison. This is a deeper change to the
core engine and risks regressions. Not recommended for this fix.

### Option C: Fix in the perturbation engine (MOST INVASIVE)

Have `expand_perturbation` / `δricci_scalar` produce terms with h always at
a canonical index position. This would require changes to the Leibniz rule
expansion and is the hardest approach.

## Verification Plan

After fixing TGR-dp3:

1. **Unit test**: Fourier δ²R on flat background should give exactly 4 terms
   matching the FP kernel structure
2. **Spin projection**: flat δ²R should give (5/2)k², 0, -k², 0
3. **Then fix TGR-wm8**: revert √g coefficient from 1/4 to 1/8 in example 15
   line 49
4. **Run example 15**: both Step 1 (EH) and Step 2 (R³) should pass
5. **Run full test suite**: `julia --project -e 'using Pkg; Pkg.test()'`

## Key Files

| File | Role |
|------|------|
| `src/algebra/simplify.jl:48` | `collect_terms` — where terms fail to combine |
| `src/algebra/simplify.jl:84` | `_normalize_dummies` — renames dummies but doesn't normalize positions |
| `src/algebra/canonicalize.jl:44` | `canonicalize(TProduct)` — xperm-based, doesn't handle cross-tensor metric equivalence |
| `src/algebra/canonicalize.jl:348` | `fix_dummy_positions` — fixes same-position pairs but doesn't lower indices |
| `src/action/kernel_extraction.jl:77` | `spin_project` — works correctly |
| `src/action/kernel_extraction.jl:302` | `build_FP_momentum_kernel` — reference correct kernel |
| `src/action/spin_projectors.jl` | Barnes-Rivers projectors — all correct |
| `examples/15_perturbation_spectrum_crosscheck.jl:49` | √g coefficient (TGR-wm8: change 1//4 back to 1//8) |
| `src/svt/fourier.jl` | `to_fourier` — works correctly, inherits mixed positions from input |

## Ready Issues (Updated)

| ID | P | Type | Title | Status |
|----|---|------|-------|--------|
| **TGR-dp3** | **P1** | **bug** | **Canonicalizer mixed index positions** | **READY — fix first** |
| **TGR-wm8** | **P2** | **bug** | **√g coefficient revert** | **READY — fix second** |
| TGR-76k | P1 | task | Validate dS crosscheck | blocked by dp3 + wm8 |
| TGR-nzk | P2 | feature | integrate_geodesic() | ready |
| TGR-f0c | P3 | feature | Axial symmetry ansatz | ready |
| TGR-3gx | P3 | feature | Order-2 curvature syzygies | ready |
| TGR-0mg | P3 | feature | Gauss-Codazzi relations | ready |
| TGR-760 | P3 | feature | GHY boundary term | ready |
| TGR-vhp | P3 | feature | EOS types for matter | ready |

## Recommended Next Session Priority

1. **TGR-dp3** — Implement `normalize_field_positions` (Option A above), add to pipeline
2. **TGR-wm8** — One-line fix: `examples/15_perturbation_spectrum_crosscheck.jl` line 49: `1//4` → `1//8`
3. **TGR-76k** — Re-run example 15, verify all spin projections match
4. **TGR-nzk** — If time, implement integrate_geodesic()
