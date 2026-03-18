# HANDOFF: Full Symbolic 6-Derivative Gravity Particle Spectrum

## Mission

Compute the complete symbolic particle spectrum of six-derivative gravity in 4D on flat (Minkowski) and de Sitter backgrounds, using two independent methods (Path A: covariant Barnes-Rivers, Path B: 3+1 SVT) with cross-checks. The flat result validates against Buoninfante et al. 2012.11829 Eq.2.13. The dS result with all 14 parameters is **novel**.

## Issue Tracker

All work is tracked in beads. Run `bd ready` to see unblocked issues, `bd blocked` to see the dependency chain, `bd show TGR-XXXX` for details.

### Issue Map (14 issues)

```
READY NOW (can start in parallel):
  TGR-ncdr  [P1] Step 0.1: Kernel extraction for rank-2 fields     ← CRITICAL PATH
  TGR-w7jq  [P1] Step 1.1: Build δ²S for 6-deriv action on flat    ← independent
  TGR-0i4m  [P3] Step 4.1: Extend sym_inv to 3×3 matrices          ← independent

DEPENDENCY CHAIN:
  TGR-ncdr (0.1 kernel) ──→ TGR-ud97 (0.2 spin projection) ──→ TGR-zq2k (1.3 BR flat)
  TGR-w7jq (1.1 build δ²S) ──→ TGR-7m26 (1.2 Fourier+kernel) ──→ TGR-zq2k (1.3 BR flat)
  TGR-w7jq (1.1 build δ²S) ──→ TGR-c6su (2.1 SVT decompose) ──→ TGR-pr04 (2.2 SVT QF)
  TGR-zq2k (1.3 BR flat) + TGR-pr04 (2.2 SVT QF) ──→ TGR-tztc (2.3 cross-check)
  TGR-ncdr + TGR-ud97 ──→ TGR-mphe (3.1 dS quad) ──→ TGR-7tcs (3.2 dS cubics)
  TGR-mphe + TGR-7tcs ──→ TGR-ug98 (3.3 full dS spectrum, NOVEL)
  TGR-zq2k + TGR-tztc + TGR-ug98 ──→ TGR-j6r9 (5 tests+bench) ──→ TGR-af4a (6 example)
```

### Parallelism Strategy

**Wave 1** (3 parallel agents):
- Agent A: TGR-ncdr (Step 0.1) — kernel extraction
- Agent B: TGR-w7jq (Step 1.1) — build all δ²S expressions
- Agent C: TGR-0i4m (Step 4.1) — sym_inv 3×3

**Wave 2** (after 0.1 done):
- Agent D: TGR-ud97 (Step 0.2) — spin projection (needs 0.1)
- Agent E: TGR-7m26 (Step 1.2) — Fourier + kernel (needs 0.1 + 1.1)
- Agent F: TGR-c6su (Step 2.1) — SVT decomposition (needs 1.1)

**Wave 3** (after 0.2 + 1.2 done):
- Agent G: TGR-zq2k (Step 1.3) — BR projection flat (needs 0.2 + 1.2)
- Agent H: TGR-pr04 (Step 2.2) — SVT QuadraticForms (needs 2.1)
- Agent I: TGR-mphe (Step 3.1) — dS quad+box terms (needs 0.1 + 0.2)

**Wave 4**:
- Agent J: TGR-tztc (Step 2.3) — cross-check A vs B (needs 1.3 + 2.2)
- Agent K: TGR-7tcs (Step 3.2) — dS cubics (needs 3.1)

**Wave 5**:
- Agent L: TGR-ug98 (Step 3.3) — full dS spectrum (needs 3.1 + 3.2)

**Wave 6**:
- Agent M: TGR-j6r9 (Step 5) — tests + benchmark
- Agent N: TGR-af4a (Step 6) — example script + integration

---

## Confirmed Timings (single thread, from this session's measurements)

| Operation | Time | Terms |
|-----------|------|-------|
| δ²R (EH, flat) | 9.5s | 9 |
| (δR)² (R², flat) | 0.2s | 9 |
| (δRic)² (Ric², flat) | 0.1s | 4 |
| δ²(R□R) (flat) | 13s | 18 |
| δ²(Ric□Ric) (flat) | 13.5s | 21 |
| to_fourier(any) | <0.5s | — |
| simplify(Fourier result) | <0.4s | 13–72 |
| dS 6 cubics (parallel 8T) | 66s | varies |
| extract_quadratic_form(δ²R, [:h]) | 1.0s | BROKEN output |
| Barnes-Rivers P² construction | instant | 2 TSum terms |
| extract_kernel_direct(δR*□δR, :h) | ~5s | 36 terms | **FIXED 2026-03-18** |
| extract_kernel_direct(δRic*□δRic, :h) | ~5s | 36 terms | **FIXED 2026-03-18** |

**WARNING**: `simplify` emits "did not converge after 20 iterations" on some expressions. Results appear correct. Use `maxiter=40` in the example script if needed.

**NOTE (2026-03-18)**: TGR-9ay resolved. Box kernel extraction (R□R, Ric□Ric) now works via one-line fix in `_distribute_derivs_sums`. All gauge sectors verified (spin-1=0, spin-0w=0). See HANDOFF-canonicalize-investigation.md for details.

---

## Critical Source Files

### Files to CREATE

1. **`src/action/kernel_extraction.jl`** — NEW (Step 0.1)
2. **`src/action/spin_projection.jl`** — NEW (Step 0.2)
3. **`examples/14_6deriv_spectrum_symbolic.jl`** — NEW (Step 6)
4. **`benchmarks/bench_13_spectrum.jl`** — NEW (Step 5)
5. **`results/6deriv_spectrum_symbolic.jl`** — NEW (Step 6)

### Files to MODIFY

6. **`src/action/quadratic_action.jl`** — extend `sym_inv` to 3×3 (Step 4.1)
7. **`src/TensorGR.jl`** — add includes + exports (Step 6)
8. **`test/test_6deriv_spectrum.jl`** — add kernel/projection/crosscheck tests (Step 5)

### Files to READ (reference, do not modify)

9. `src/action/extract_quadratic.jl` — existing broken approach, reference for bilinear collection pattern
10. `src/action/spin_projectors.jl` — Barnes-Rivers projector implementations (theta, omega, P², P¹, P⁰ˢ, P⁰ʷ, T^sw, T^ws)
11. `src/svt/fourier.jl` — `to_fourier` implementation (∂_a → k_a)
12. `src/gr/box.jl` — `box(field, metric)` = g^{ab}∂_a∂_b(field)
13. `src/algebra/ibp.jl` — `ibp_product(expr, field)` for integration by parts
14. `examples/08_postquantum_gravity.jl` — template for Path B (4th-deriv, 3+1 SVT, manual QF)
15. `examples/11_6deriv_gravity_dS.jl` — cubic invariant builders (build_I1…build_I6) + δ² on dS
16. `examples/13_6deriv_particle_spectrum.jl` — numerical form factors (ground truth values)
17. `test/test_6deriv_spectrum.jl` — existing tests (form factors, residue sums, dS limits)

---

## Step 0.1: Kernel Extraction — DETAILED SPEC

### The Problem

After `expand_perturbation` + `simplify` + `to_fourier`, the expression δ²S is a TSum of TProduct terms. Each term is bilinear in h_{μν} with k-dependent coefficients. Example:

```
(-1//2) h^{_d1,_d2} h_{_d1,_d4} k^{_d2} k_{_d4}
```

We need to extract the 4-index kernel K_{μν,ρσ}(k) such that δ²S = h^{μν} K_{μν,ρσ}(k) h^{ρσ}.

### Algorithm

For each TProduct term in the expanded TSum:

```julia
function extract_kernel(expr::TensorExpr, field::Symbol; registry=current_registry())
    # 1. Ensure expr is expanded (TSum of TProducts)
    expanded = expand_products(expr)

    # 2. For each term, identify the two h factors
    # 3. Everything else is the kernel coefficient for that index pattern
    # 4. Accumulate with canonical index labeling

    # Return: KineticKernel struct with the kernel as a TensorExpr
    # with 4 free indices (μ,ν for left h, ρ,σ for right h)
end
```

**Key implementation detail**: Each TProduct term has factors. Walk the factors to find exactly 2 whose `.name == field`. Record their indices. The remaining factors (k tensors, η/g tensors, TScalar coefficients) form the kernel entry for that index pattern.

**Index routing**: The h factors have indices that may be Up or Down, and may be contracted with the coefficient factors. After Fourier transform, there are NO TDeriv nodes — all derivatives are already k tensors. So each term is a flat product of Tensor and TScalar nodes.

**Canonical labeling**: Rename the h indices to standard labels (e.g., :μ,:ν for left, :ρ,:σ for right) using `rename_dummy` or manual substitution. This ensures all terms accumulate correctly.

**Symmetrization**: The kernel must satisfy K_{μν,ρσ} = K_{νμ,ρσ} = K_{μν,σρ} = K_{ρσ,μν}. Enforce by averaging over permutations or checking.

### Also include: contract_momenta

```julia
function contract_momenta(expr::TensorExpr; k_name::Symbol=:k, k_sq::Symbol=:k²)
    # Walk TProduct factors, find pairs k_a k^a (same name, opposite position)
    # Replace pair with TScalar(k_sq)
    # Also handle: TScalar(:(1/k²)) * TScalar(:k²) → 1
end
```

This is needed because after contracting the kernel with Barnes-Rivers projectors, the result has many k_a k^a dummy pairs that should be k².

### Struct

```julia
struct KineticKernel
    field::Symbol
    kernel::TensorExpr       # The kernel K with 4 free indices
    left_indices::Tuple{TIndex, TIndex}   # (μ, ν) labels on left h
    right_indices::Tuple{TIndex, TIndex}  # (ρ, σ) labels on right h
end
```

### What the Fourier-transformed δ²R looks like (confirmed output)

```
(-1//2) h^_d1^_d2 h__d1__d4 k^_d2 k__d4
+(1//2) h^_d1^_d2 h__d1__d5 k^_d2 k__d5
+(1//2) h^_d1^_d4 h^_d1__d4 k__d2 k__d2
+(1//4) g^_d1^_d1 h^_d1_e h_c_c k^_d1 k_e
+(-1//2) h^_d1^_d2 h__d1__d2 k^_d2 k__d2
... (13 terms total)
```

Each term: scalar × h_{..} × h_{..} × (k and/or g factors).
The kernel coefficient for each term = scalar × k-factors × g-factors.

### Testing

Extract kernel from `h_{ab} □ h^{ab}` (= h_{ab} g^{cd} ∂_c ∂_d h^{ab}). After Fourier: h_{ab} g^{cd} k_c k_d h^{ab} = k² h_{ab} h^{ab}. The kernel should be K_{μν,ρσ} ∝ k² · (δ_μ^ρ δ_ν^σ + δ_μ^σ δ_ν^ρ)/2.

---

## Step 0.2: Spin Projection — DETAILED SPEC

### The Algorithm

```julia
function spin_project(kernel::KineticKernel, spin::Symbol;
                      dim::Int=4, metric::Symbol=:η,
                      k_name::Symbol=:k, k_sq::Symbol=:k²)
    μ, ν = kernel.left_indices
    ρ, σ = kernel.right_indices

    # Build the appropriate projector with matching indices
    P = if spin == :spin2
        spin2_projector(μ, ν, ρ, σ; dim, metric, k_name, k_sq)
    elseif spin == :spin1
        spin1_projector(μ, ν, ρ, σ; metric, k_name, k_sq)
    # ... etc
    end

    # Contract: O_J = P^J_{μν,ρσ} · K^{μν,ρσ}
    # This means: multiply P * kernel.kernel, then simplify
    # All 4 indices (μ,ν,ρ,σ) become dummies and contract
    product = P * kernel.kernel
    result = simplify(product; registry=current_registry())

    # Apply contract_momenta to reduce k_a k^a → k²
    result = contract_momenta(result; k_name, k_sq)

    return result  # Should be a scalar expression in k²
end
```

### Key subtlety: k² and 1/k² cancellation

The Barnes-Rivers projectors contain `TScalar(:(1 / k²))` from the ω_{μν} = k_μ k_ν / k² building block. After contraction, k_a k^a produces TScalar(:k²). The product TScalar(:(1/k²)) × TScalar(:k²) must simplify to 1.

**Options**:
1. Post-process: walk expression, find `:(1/k²) * k²` Expr patterns and cancel
2. Convert to Symbolics.Num: use `@variables k²` and let CAS cancel
3. Work with Rational exponents: represent k² as a symbolic variable from the start

Option 2 (Symbolics.jl) is safest — the extension already dispatches `_sym_mul`/`_sym_div` on `Symbolics.Num`.

### Expected results (ground truth)

For the flat 6-derivative action with kernel K_total:
- `spin_project(K_total, :spin2)` → expression proportional to `k² · (1 - (α₂/κ)k² - (β₂/κ)k⁴)`
- `spin_project(K_total, :spin0s)` → expression proportional to `k² · (1 + (6α₁+2α₂)k²/κ + (6β₁+2β₂)k⁴/κ)`
- `spin_project(K_total, :spin1)` → 0 (identically)

---

## Step 1.1: Build δ²S — DETAILED SPEC

### Flat background setup

```julia
reg = TensorRegistry()
mp = nothing
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp = define_metric_perturbation!(reg, :g, :h)
end
```

### The 5 terms to compute

```julia
with_registry(reg) do
    # 1. κR → δ²R
    δ2R = simplify(δricci_scalar(mp, 2); registry=reg)  # ~9.5s → 9 terms

    # 2. α₁R² → 2(δR)² on flat (R̄=0)
    δ1R = δricci_scalar(mp, 1)
    δR_sq = simplify(δ1R * δ1R; registry=reg)  # ~0.2s → 9 terms

    # 3. α₂Ric² → 2δRic·δRic on flat (R̄_{μν}=0)
    δRic1 = δricci(mp, down(:a), down(:b), 1)
    δRic2 = δricci(mp, down(:c), down(:d), 1)
    δRic_sq = simplify(δRic1 * δRic2 * Tensor(:g, [up(:a), up(:c)]) * Tensor(:g, [up(:b), up(:d)]); registry=reg)  # ~0.1s → 4 terms

    # 4. β₁R□R → δ²(R · g^{ab}∂_a∂_b R)
    R1 = Tensor(:RicScalar, TIndex[])
    R2 = Tensor(:RicScalar, TIndex[])
    box_R2 = box(R2, :g; registry=reg)  # g^{ab} ∂_a(∂_b(RicScalar))
    R_box_R = R1 * box_R2
    δ2_RboxR = simplify(expand_perturbation(R_box_R, mp, 2); registry=reg)  # ~13s → 18 terms

    # 5. β₂Ric□Ric → δ²(Ric_{cd} · g^{ab}∂_a∂_b Ric^{cd})
    Ric_ab = Tensor(:Ric, [down(:c), down(:d)])
    box_Ric = box(Ric_ab, :g; registry=reg)
    gac = Tensor(:g, [up(:a), up(:c)])
    gbd = Tensor(:g, [up(:b), up(:d)])
    Ric_box_Ric = Ric_ab * box_Ric * gac * gbd  # Note: box acts on Ric_{cd}, not Ric^{cd}
    δ2_RicBoxRic = simplify(expand_perturbation(Ric_box_Ric, mp, 2); registry=reg)  # ~13.5s → 21 terms
end
```

**IMPORTANT**: For term 5, `box(Ric_ab, :g)` creates `g^{ef} ∂_e(∂_f(Ric_{cd}))`, which is correct. The contraction with `g^{ac} g^{bd}` raises the Ric indices. Confirmed working — see timing results.

---

## Step 2.1: SVT Decomposition — DETAILED SPEC (Path B)

Follow `examples/08_postquantum_gravity.jl` pattern:

```julia
reg2 = TensorRegistry()
with_registry(reg2) do
    @manifold M4 dim=4 metric=g
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    fol = define_foliation!(reg2, :flat31; manifold=:M4)

    # For each δ²(term), apply foliation
    for expr in [δ2R, δR_sq, δRic_sq, δ2_RboxR, δ2_RicBoxRic]
        split = split_all_spacetime(expr, fol)
        substituted = apply_svt(split, :h, fol; gauge=:bardeen)
        sectors = collect_sectors(substituted)
        # sectors is Dict{Symbol, TensorExpr}
        # Keys: :scalar, :vector, :tensor
    end
end
```

Then for each sector, apply `to_fourier` (or `to_fourier_symbolic` with Symbolics.jl) and collect into matrix entries.

**Note**: Example 08 builds the quadratic form entries MANUALLY (lines 144-177) from knowledge of the Lichnerowicz operator in component form. For the 6-derivative terms, this manual approach is harder. The automated path through `split_all_spacetime` + `apply_svt` + `collect_sectors` should produce the raw expressions, but you still need to identify which terms are M_{ΦΦ}, M_{Φψ}, M_{ψψ} by inspecting which SVT fields appear.

---

## Step 3.1–3.3: de Sitter — DETAILED SPEC

### dS setup (differs from flat)

```julia
reg_dS = TensorRegistry()
mp_dS = nothing
with_registry(reg_dS) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg_dS, :M4, :g)
    maximally_symmetric_background!(reg_dS, :M4; metric=:g, cosmological_constant=:Λ)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp_dS = define_metric_perturbation!(reg_dS, :g, :h; curved=true)  # NOTE: curved=true
end
```

**Key difference**: `curved=true` means the perturbation engine accounts for non-zero background curvature. The background rules set R̄=4Λ, R̄_{μν}=Λg_{μν}, R̄_{abcd}=(Λ/3)(g_{ac}g_{bd}−g_{ad}g_{bc}).

### Cubic invariant builders

Reuse from `examples/11_6deriv_gravity_dS.jl`:
- `build_I1(reg)` through `build_I6(reg)` — already defined and working
- Run `expand_perturbation(Ii, mp_dS, 2)` + `simplify` for each
- These are the most expensive computations (~66s total with 8 threads)

### Bueno-Cano verification

After obtaining the full dS kernel and projecting, extract the Bueno-Cano parameters (a,b,c,e) and verify:
- κ_eff⁻¹ = 4e − 8Λ_BC·a (Eq.17)
- m²_g = (−e + 2Λ_BC·a)/(2a + c) (Eq.18)
- m²_s = (2e − 4Λ_BC·(a+4b+c))/(2a + 4c + 12b) (Eq.19)

where Λ_BC = Λ/3 (convention difference).

---

## Step 4.1: sym_inv 3×3 — DETAILED SPEC

Add to `src/action/quadratic_action.jl`, in the `sym_inv` function:

```julia
elseif n == 3
    det = sym_det(M)  # Already handles 3×3 via Sarrus
    adj = Matrix{Any}(undef, 3, 3)
    for i in 1:3, j in 1:3
        # Cofactor C_{ij} = (-1)^{i+j} * Minor_{ij}
        # Minor_{ij} = det of 2×2 submatrix excluding row j, col i (transpose!)
        rows = [r for r in 1:3 if r != j]
        cols = [c for c in 1:3 if c != i]
        minor = _sym_sub(_sym_mul(M[rows[1],cols[1]], M[rows[2],cols[2]]),
                         _sym_mul(M[rows[1],cols[2]], M[rows[2],cols[1]]))
        sign = iseven(i + j) ? 1 : -1
        adj[i, j] = sign == 1 ? minor : _sym_neg(minor)
    end
    return [_sym_div(adj[i,j], det) for i in 1:3, j in 1:3]
```

---

## Ground Truth Values (for verification)

### Flat-space form factors (Buoninfante 2012.11829 Eq.2.13)

```
f₂(z) = 1 − (α₂/κ)z − (β₂/κ)z²     [spin-2]
f₀(z) = 1 + (6α₁+2α₂)z/κ + (6β₁+2β₂)z²/κ  [spin-0]
```

### Residue sum rule

For 1/(z·f(z)) with deg(z·f(z)) = 3: sum of all residues = 0.

### Stelle limit (β₁=β₂=0)

```
m²_spin2 = κ/α₂
m²_spin0 = −κ/(6α₁+2α₂)
```

### dS Bueno-Cano parameters (quadratic terms only)

```
κR:      (a=0,  b=0,     c=0,    e=κ)
α₁R²:   (a=0,  b=2α₁,   c=0,    e=8α₁Λ/3)
α₂Ric²: (a=0,  b=0,     c=2α₂,  e=2α₂Λ/3)
```

---

## Code Style & Conventions

- Use `with_registry(reg) do ... end` blocks for all TensorGR operations
- Use `Rational{Int}` (e.g., `1//2`) for exact arithmetic, never Float64 in symbolic expressions
- Use `tproduct(scalar, factors)` and `tsum(terms)` smart constructors
- `TDeriv` has `.covd` field (default `:partial`) — propagate in all TDeriv constructions
- `_base_field_name(t::Tensor) = t.name` — this is how field identification works
- Exports go in `src/TensorGR.jl` grouped with the action section (after line 224)
- New includes go after line 79 (after `spin_projectors.jl`)
- Tests go in `@testset` blocks in `test/test_6deriv_spectrum.jl`
- Benchmarks follow the pattern in `benchmarks/bench_12_6deriv_dS.jl`

## Session Close Protocol

After completing work:
```bash
git status
git add <files>
bd close <completed-issues>
bd sync
git commit -m "..."
git push
```

## Quick Start

```bash
# Check what's ready
bd ready

# Start with the critical path
bd update TGR-ncdr --status=in_progress

# Run tests to verify nothing is broken
julia --project -e 'using Pkg; Pkg.test()'

# Time a quick sanity check
julia --project -e '
using TensorGR
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    mp = define_metric_perturbation!(reg, :g, :h)
    δ2R = simplify(δricci_scalar(mp, 2); registry=reg)
    δ2R_f = simplify(to_fourier(δ2R); registry=reg)
    println("δ²R Fourier: ", δ2R_f isa TSum ? length(δ2R_f.terms) : 1, " terms")
    println(to_unicode(δ2R_f))
end
'
```
