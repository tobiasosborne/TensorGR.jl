# HANDOFF: Session 32 — δ²R Bug RESOLVED (TGR-dp3 closed)

## Status: FIXED. All 7298 tests pass. Pushed to master.

## What Was Done

### Clean-room reimplementation (the key insight)

Two subagents were spawned with the full spec but NOT the buggy source code:
- **Agent A**: reimplemented `fourier.jl` (phase formula was wrong: `(-1)^{n_R}` instead of `(-1)^{n/2+n_R}`)
- **Agent B**: reimplemented `kernel_extraction_v2.jl` (correct phase, verified against FP ground truth)

Agent B's implementation was integrated into `kernel_extraction.jl` as the new `extract_kernel_direct`.

### Three interlocking root causes identified

1. **`simplify` corrupts bilinear structure**: merges ΓΓ+∂Γ terms into ∂(bilinear) nodes that can't be decomposed
2. **`to_fourier` uniform-k wrong for asymmetric derivatives**: h₁(∂²h₂) gets k² instead of -k²
3. **Wrong formula in HANDOFFs**: δ²R + (1/2)tr(h)δ¹R does NOT produce the FP kernel. The correct EH bilinear is `h^{ab}δ¹G_{ab} = h^{ab}(δ¹R_{ab} - (1/2)g_{ab}δ¹R)`

### The correct pipeline (verified against Buoninfante form factors)

```julia
# Pre-simplify each FACTOR individually
d1R_ab = simplify(δricci(mp, down(:a), down(:b), 1); registry=reg)
d1R    = simplify(δricci_scalar(mp, 1); registry=reg)
d1Ric_cd = simplify(δricci(mp, down(:c), down(:d), 1); registry=reg)

# EH kernel: linearized Einstein tensor contracted with h
K_EH = extract_kernel_direct(
    Tensor(:h, [up(:a), up(:b)]) * d1R_ab -
    (1//2) * Tensor(:g, [up(:p), up(:q)]) * Tensor(:h, [down(:p), down(:q)]) * d1R,
    :h; registry=reg)

# R² and Ric² kernels (from pre-simplified factors)
K_R2 = extract_kernel_direct(d1R * d1R, :h; registry=reg)
K_Ric2 = extract_kernel_direct(
    d1R_ab * d1Ric_cd * Tensor(:g, [up(:a), up(:c)]) * Tensor(:g, [up(:b), up(:d)]),
    :h; registry=reg)

# Full kernel: note the -2 sign convention!
K = combine_kernels([K_EH, scale_kernel(K_R2, -2α₁), scale_kernel(K_Ric2, -2α₂)])

# Spin project → matches Buoninfante 2012.11829 Eq.2.13
spin_project(K, :spin2; registry=reg)
```

### Key sign convention: -2

The full kinetic kernel for S = ∫√g(κR + α₁R² + α₂Ric²) on flat is:
```
K = κ·K_EH - 2α₁·K_{R²} - 2α₂·K_{Ric²}
```

The -2 is NOT a bug. It comes from the relationship between the action's second variation and the kinetic operator. Verified against all Buoninfante form factors.

### Two-momentum phase formula

For a bilinear term with n_L derivatives on h₁ and n_R on h₂:
```
phase = (-1)^{n/2 + n_R}   (n = n_L + n_R, must be even)
```

This corrects the uniform-k convention (which drops all i factors) by accounting for the fact that h₂ carries momentum -k in the integral ∫dx h₁(x) K(∂) h₂(x).

## Files Changed
- `src/action/kernel_extraction.jl` — clean-room `extract_kernel_direct` + helpers
- `test/test_kernel_extraction.jl` — 59 new tests (NEW)
- `test/runtests.jl` — added include

## Next Steps
- TGR-76k: dS crosscheck blocked — needs dS-adapted Barnes-Rivers projectors. Flat projectors on dS kernel give spin-2 = 2.5 - 5Λ (wrong). BC predicts 2.5 (constant). The eigenvalue spectrum of the Lichnerowicz operator on MSS differs from flat.
- Ready P2: TGR-nzk (geodesic integration in DiffEq extension)
- Ready P3: TGR-3gx (curvature syzygies), TGR-0mg (Gauss-Codazzi), TGR-760 (GHY boundary), TGR-vhp (EOS types), TGR-zkw (Killing rules), TGR-f0c (axial metric ansatz)
