# HANDOFF: 6-Deriv Spectrum Pipeline — Session 2

## What Was Done (Session 1)

### TGR-ncdr ✅ — Kernel Extraction (Step 0.1)
**File**: `src/action/kernel_extraction.jl` (new, 202 lines)

Implemented three functions:

1. **`extract_kernel(expr, field; registry)`** → `KineticKernel`
   - Decomposes a bilinear δ²S into per-term `(coefficient, h₁_indices, h₂_indices)`
   - Each TProduct term is split: the two `field`-named Tensor factors are separated from everything else
   - Handles arbitrary h index positions (mixed Up/Down across terms)

2. **`spin_project(K::KineticKernel, spin; dim, metric, k_name, k_sq, registry)`** → `TensorExpr`
   - Contracts kernel with Barnes-Rivers projectors **per-term** (key design decision)
   - Builds projector with each term's **actual h index labels**, ensuring correct contraction regardless of mixed positions
   - Uses `ensure_no_dummy_clash` to prevent index collisions between projector internals and coefficient
   - Calls `simplify(...; maxiter=40)` on the full sum

3. **`contract_momenta(expr; k_name, k_sq)`** → `TensorExpr`
   - Walks TProducts, finds `k_a k^a` dummy pairs → replaces with `TScalar(k²)`
   - Also cancels `TScalar(1/k²) × TScalar(k²)` → `TScalar(1)`

**Struct**:
```julia
struct KineticKernel
    field::Symbol
    terms::Vector{@NamedTuple{coeff::TensorExpr, left::Vector{TIndex}, right::Vector{TIndex}}}
end
```

### TGR-0i4m ✅ — sym_inv 3×3 (Step 4.1)
**File**: `src/action/quadratic_action.jl` (10 lines added)
- Cofactor/adjugate method, verified against `LinearAlgebra.inv`

### All 4568 tests pass. Committed and pushed.

---

## What's Ready Now (Wave 2)

Run `bd ready` — the two critical-path P1 issues:

### TGR-ud97 — Step 0.2: Barnes-Rivers spin projection
**Status**: ~80% done. `spin_project` already exists in `kernel_extraction.jl`. This issue originally called for a separate `spin_projection.jl` file, but the functionality is already implemented.

**Remaining work**:
1. **Test against known result**: Build the Lichnerowicz kernel (h_{ab} □ h^{ab} term from EH action), project onto spin-2, verify result ∝ k²
2. **Verify k²/k⁻² cancellation**: The Barnes-Rivers projectors use `TScalar(:(1/k²))` from the ω building block. After contraction + simplify + contract_momenta, these should cancel. If not, may need Symbolics.jl dispatch.
3. **Handle transfer operators**: `spin_project` currently supports `:spin2`, `:spin1`, `:spin0s`, `:spin0w`. May want to add `:transfer_sw` and `:transfer_ws`.
4. **Update issue description**: The spin_project function is in kernel_extraction.jl, not a separate file. Close TGR-ud97 after testing.

### TGR-w7jq — Step 1.1: Build δ²S for 6-deriv action on flat
**Status**: Not started. Independent of the kernel work.

**What to do**: Build all 5 δ²S expressions. See HANDOFF-6deriv-spectrum.md §Step 1.1 for exact code. Key setup:

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp = define_metric_perturbation!(reg, :g, :h)
    set_vanishing!(reg, :Ric)
    set_vanishing!(reg, :RicScalar)
    set_vanishing!(reg, :Riem)

    # 1. δ²R (~9.5s, expect 9 terms)
    δ2R = simplify(δricci_scalar(mp, 2); registry=reg)

    # 2. (δR)² (~0.2s, expect 9 terms)
    δ1R = δricci_scalar(mp, 1)
    δR_sq = simplify(δ1R * δ1R; registry=reg)

    # 3. (δRic)² (~0.1s, expect 4 terms)
    δRic1 = δricci(mp, down(:a), down(:b), 1)
    δRic2 = δricci(mp, down(:c), down(:d), 1)
    δRic_sq = simplify(δRic1 * δRic2 * Tensor(:g, [up(:a), up(:c)]) * Tensor(:g, [up(:b), up(:d)]); registry=reg)

    # 4. δ²(R□R) (~13s, expect 18 terms)
    R1 = Tensor(:RicScalar, TIndex[])
    R2 = Tensor(:RicScalar, TIndex[])
    box_R2 = box(R2, :g; registry=reg)
    δ2_RboxR = simplify(expand_perturbation(R1 * box_R2, mp, 2); registry=reg)

    # 5. δ²(Ric□Ric) (~13.5s, expect 21 terms)
    Ric_ab = Tensor(:Ric, [down(:c), down(:d)])
    box_Ric = box(Ric_ab, :g; registry=reg)
    gac = Tensor(:g, [up(:a), up(:c)])
    gbd = Tensor(:g, [up(:b), up(:d)])
    δ2_RicBoxRic = simplify(expand_perturbation(Ric_ab * box_Ric * gac * gbd, mp, 2); registry=reg)
end
```

Pin term counts. Total ~37s serial.

---

## Parallelism Strategy for This Session

**Wave 2a** (can be parallel):
- Agent A: TGR-ud97 — test spin_project, close issue
- Agent B: TGR-w7jq — build all 5 δ²S expressions, save term counts

**Wave 2b** (after both done):
- TGR-7m26: Fourier-transform each δ²S + extract_kernel → 5 KineticKernels
- TGR-c6su: SVT decomposition of δ²S (Path B, independent of kernel)

**Wave 3** (after 2b):
- TGR-zq2k: spin_project all 5 kernels → flat form factors f₂(k²), f₀(k²)
- TGR-pr04: SVT QuadraticForms

---

## Critical Findings from Session 1

### Index Position Issue
After `simplify(to_fourier(δ²R))`, the terms have **mixed h index positions**:
- Some terms: h(Up,Up) × h(Down,Down)
- Some terms: h(Up,Down) × h(Up,Down)
- Some terms have h with **self-traced indices** like h(Down(:c), Down(:c))

The `free_indices` count on individual terms is non-zero (typically 4) even though δ²S should be scalar. This is because same-position index pairs (e.g., two Downs) are NOT recognized as dummy pairs by TensorGR. The simplifier warns "did not converge after 20 iterations" — **the expressions are correct but not fully canonical**.

**Design response**: `spin_project` builds the projector **per-term** with the h factors' actual index labels, bypassing the need for a single canonical-position kernel tensor. This is robust to mixed positions.

### `simplify` Non-Convergence Warning
Emitted on several Fourier-transformed expressions. Use `maxiter=40` if needed. Results are numerically correct despite the warning.

### Vanishing Background Rules
Must explicitly set `set_vanishing!(reg, :Ric)`, `:RicScalar`, `:Riem` for flat background. Without this, terms with background Ric survive in δ²R.

---

## Key Files

| File | Role |
|------|------|
| `src/action/kernel_extraction.jl` | NEW: extract_kernel, spin_project, contract_momenta |
| `src/action/quadratic_action.jl` | MODIFIED: sym_inv 3×3 added |
| `src/action/spin_projectors.jl` | Barnes-Rivers P², P¹, P⁰ˢ, P⁰ʷ, T^sw, T^ws |
| `src/action/extract_quadratic.jl` | Old approach (loses tensor structure — reference only) |
| `src/TensorGR.jl` | MODIFIED: include + exports added |
| `HANDOFF-6deriv-spectrum.md` | Master handoff (full pipeline spec, all 14 issues) |
| `examples/13_6deriv_particle_spectrum.jl` | Ground truth: numerical form factors |
| `examples/11_6deriv_gravity_dS.jl` | Cubic invariant builders (build_I1…build_I6) |

---

## Ground Truth (flat spectrum)

```
f₂(z) = 1 − (α₂/κ)z − (β₂/κ)z²     [spin-2, z = k²]
f₀(z) = 1 + (6α₁+2α₂)z/κ + (6β₁+2β₂)z²/κ  [spin-0]
spin-1 projection = 0 identically
```

Stelle limit (β₁=β₂=0): m²_spin2 = κ/α₂, m²_spin0 = −κ/(6α₁+2α₂)

---

## Quick Start

```bash
# Check state
bd ready
bd blocked

# Run sanity check (should print 12 terms)
julia --project -e '
using TensorGR
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp = define_metric_perturbation!(reg, :g, :h)
    set_vanishing!(reg, :Ric); set_vanishing!(reg, :RicScalar); set_vanishing!(reg, :Riem)
    δ2R = simplify(δricci_scalar(mp, 2); registry=reg)
    δ2R_f = simplify(to_fourier(δ2R); registry=reg)
    K = extract_kernel(δ2R_f, :h; registry=reg)
    println("KineticKernel: ", length(K.terms), " terms")
'

# Session end protocol
bd sync && git push
```
