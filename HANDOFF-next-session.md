# HANDOFF: 6-Deriv Spectrum Pipeline — Session 9

## Current State

- **4800 tests pass**, all pushed (commit da63ca2)
- **Spin projection validated**: identity kernel {5,3,1,1}, manual Fierz-Pauli EH kernel gives spin-1=0, spin-0-w=0 ✓
- **fix_dummy_positions**: exported, tested, repairs same-position dummy pairs from xperm
- **Simplify convergence fixed**: two-phase dummy renaming in _normalize_dummies (commit e6f34de)
- δ²R on flat converges to 22 terms (was oscillating at 23)

---

## What Was Done (Sessions 7-8)

### Session 7: Convergence Fix
- Two-phase dummy renaming in `_normalize_dummies` (old→tmp, tmp→canonical) prevents name collisions during batch rename
- δ²R now converges to 22 terms in ≤4 passes

### Session 8: fix_dummy_positions + spin_project Index Standardization

**fix_dummy_positions** (`src/algebra/canonicalize.jl:319-433`):
Post-processing that repairs same-position dummy pairs (both-Up or both-Down) from the all-free xperm mode. Flips one occurrence to restore valid (Up,Down) pairing. Needed before Fourier transform and spin projection.

**spin_project index standardization** (`src/action/kernel_extraction.jl:76-185`):
Added `_standardize_h_indices` — lowers all h factor indices to Down position with fresh names before building Barnes-Rivers projectors. This prevents projector self-contraction when left/right h indices share names (e.g., h^{ab} h_{ad} with shared index :a would partially trace the projector).

**Validation** (all in `test/test_6deriv_spectrum.jl`):
- Identity kernel h_{ab}k²h^{ab} → {5k², 3k², k², k²} = sector dimensions × k² ✓
- Manual Fierz-Pauli EH kernel → {5k²/2, 0, -k², 0} — gauge invariance confirmed ✓
- fix_dummy_positions: validates repair of same-position pairs ✓

---

## Priority 1: RESEARCH — Best Approach to EH Form Factors

### The Problem

The perturbation engine computes `δ²R + ½h·δR` for the EH quadratic Lagrangian on flat vacuum. This is mathematically correct but contains **total derivative terms** that don't affect the action integral but DO produce non-zero spin-1 contributions when projected term-by-term in Fourier space.

Spin projection of the raw perturbation output gives:
- spin-2: -5k²/4 (wrong)
- spin-1: -9k²/4 (should be 0!)
- spin-0-s: -5k²/2 (wrong)
- spin-0-w: 0 ✓

But spin projection of the manual Fierz-Pauli form gives the correct result:
- spin-2: 5k²/2 ✓ (f₂ = k²/2)
- spin-1: 0 ✓
- spin-0-s: -k² ✓ (f₀s = -k²)
- spin-0-w: 0 ✓

### Research Task for Next Agent

**Before writing any code**, investigate and compare the following approaches. Read the relevant source files, think through edge cases, and recommend the best path.

#### Approach A: IBP Before Projection

Apply integration by parts to the quadratic Lagrangian to remove total derivatives, converting it to Fierz-Pauli form before Fourier transform + spin projection.

Questions to investigate:
- Does TensorGR's `ibp` / `ibp_product` work on bilinear h expressions?
- Can we write a dedicated `to_fierz_pauli(expr, field)` that IBPs until no total derivatives remain?
- What does IBP look like in Fourier space? (Hint: `k_a × (term)` → boundary at k→∞ = 0)
- Is there a clean criterion for "the expression has no total derivatives"?

Files to read: `src/algebra/ibp.jl`, `src/svt/fourier.jl`

#### Approach B: Build Kernel from Linearized Equations of Motion

Instead of computing the Lagrangian δ²S, compute the linearized field equations (linearized Einstein tensor G^(1)_{μν}) and build the kernel directly from the equations:

K_{μν,ρσ} h^{ρσ} = G^(1)_{μν}

The equations of motion have NO total derivative ambiguity. The kernel is the differential operator mapping h to G^(1).

Questions to investigate:
- Can we extract K_{μν,ρσ} from G^(1)_{μν}(h) by treating h as a "source" with free indices?
- Does `euler_lagrange` / `variational_derivative` already do this?
- How does this generalize to higher-derivative terms (R², Ric², R□R, Ric□Ric)?
- For 4th/6th derivative terms: the EOM is 4th/6th order in derivatives — does the Fourier transform handle this correctly?

Files to read: `src/perturbation/variation.jl`, `src/perturbation/linearize.jl`

#### Approach C: Gauge-Fix Then Project

Add a gauge-fixing term (de Donder gauge: `-(1/2)(∂_μ h^μν - ½∂^ν h)²`) to the quadratic Lagrangian. This makes the kinetic operator invertible (all 4 sectors non-degenerate) and removes the total derivative issue.

Questions to investigate:
- Does gauge fixing change the spin-2 and spin-0-s form factors? (It shouldn't — gauge fixing only affects spin-1 and spin-0-w)
- Can we extract f₂ and f₀s from the gauge-fixed operator and verify they match the gauge-invariant result?
- Is this simpler than IBP?

#### Approach D: Direct Momentum-Space Construction

Skip the position-space perturbation engine entirely. Build the momentum-space kernel from the known structure of each curvature invariant:

- EH: the Fierz-Pauli kernel (4 terms, already validated in Test 3)
- R²: the kernel is `(k_a k_b h^{ab} - k² h)² / something`
- Ric²: similarly from δRic in Fourier space

Questions: Can we compute δRic and δR directly in Fourier space without going through the perturbation engine?

### Important Context for Higher-Derivative Terms

The higher-derivative terms (R², Ric², R□R, Ric□Ric) are DIFFERENT from EH:
- `(δR)²` is already a product of first-order variations — NO total derivative issue
- Same for `(δRic)²`
- The box terms `2(δR)(□δR)` and `2(δRic)(□δRic)` are also products

So the total-derivative problem is **specific to the EH term**. The higher-derivative terms should "just work" with the existing pipeline. The research should confirm this.

### Deliverable

Write a brief recommendation (in this handoff file or a comment) with:
1. Which approach is best and why
2. Estimated complexity (how many lines of code, which files to change)
3. Any blockers or risks discovered
4. Then implement the chosen approach

---

## Priority 2: Complete 6-Deriv Form Factors (TGR-zq2k)

Once the EH kernel is working, combine all 5 kernels with coupling constants and verify:

```
f₂(z) = 1 − (α₂/κ)z − (β₂/κ)z²      (Buoninfante Eq. 2.13)
f₀(z) = 1 + (6α₁+2α₂)z/κ + (6β₁+2β₂)z²/κ
```

Test plan:
1. Build combined kernel: `κ·K_EH + α₁·K_R² + α₂·K_Ric² + β₁·K_R□R + β₂·K_Ric□Ric`
2. `spin_project(:spin2)` → extract coefficient of k² and k⁴ → verify matches f₂
3. `spin_project(:spin0s)` → extract coefficients → verify matches f₀
4. `spin_project(:spin1)` → must be exactly 0
5. `spin_project(:spin0w)` → must be exactly 0

### Convention Notes

- `spin_project` returns `Tr(K·P^J)`, NOT `f_J`. Divide by sector dimension to get f_J: {5,3,1,1} for d=4
- `δricci_scalar(mp, n)` returns the ε^n coefficient (Cauchy product), not (1/n!)d^n/dε^n
- `to_fourier` replaces ∂_a → k_a (no factor of i)
- k² is stored as `TScalar(:k²)`, 1/k² as `TScalar(:(1/k²))`

---

## Key Files

| File | Role |
|------|------|
| `src/action/kernel_extraction.jl` | `extract_kernel`, `spin_project`, `_standardize_h_indices`, `contract_momenta` |
| `src/action/spin_projectors.jl` | Barnes-Rivers P2/P1/P0s/P0w/θ/ω projectors |
| `src/algebra/canonicalize.jl:319-433` | `fix_dummy_positions` |
| `src/algebra/ibp.jl` | `ibp`, `ibp_product` (potential approach A) |
| `src/perturbation/variation.jl` | `variational_derivative`, `euler_lagrange` (potential approach B) |
| `src/perturbation/expand.jl` | `δricci_scalar`, `δricci`, `expand_perturbation` |
| `src/svt/fourier.jl` | `to_fourier` (∂ → k replacement) |
| `test/test_6deriv_spectrum.jl` | All spectrum tests (1194 pass) |

## Test Commands

```bash
julia --project -e 'using Pkg; Pkg.test()'                    # full suite (4800 pass)
julia --project=benchmarks benchmarks/run_all.jl --tier 1      # tier 1 benchmarks (53 pass)
bd show TGR-zq2k                                               # flat form factors issue
```
