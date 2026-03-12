# HANDOFF: Session 24 — TGR-af4a Complete, Full Backlog Created

## Status: TGR-af4a DONE, 28 issues in beads

- **All tests pass**: 6788 (Pkg.test)
- **Beads DB**: Reinitialized (was broken), 28 issues created with dependency chains
- **Committed**: examples/26 showcase + results/6deriv_spectrum_results.jl

## What Was Done This Session

### TGR-af4a: Example Script + Results + Module Integration (CLOSED)

1. Created `examples/26_6deriv_spectrum_showcase.jl` (~200 lines):
   - Path A: Barnes-Rivers spin projection with individual kernel traces
   - Path B: SVT quadratic forms with tensor/scalar sector verification
   - Path C: Bueno-Cano dS spectrum with BC parameter table
   - Cross-check: Path A ≡ Path B at 4 momentum values
   - All paths verified, example runs end-to-end

2. Created `results/6deriv_spectrum_results.jl` (~130 lines):
   - `SixDerivSpectrumResults` module with form factors, pole finders, Stelle limits
   - BC coefficient tables, dS physical spectrum calculator
   - SVT structure documentation, spin projection traces

3. Module integration was already done in prior sessions (confirmed).

### Full Backlog Assessment + Issue Creation

Spawned 6 parallel agents to assess all backlog items. Created 28 granular beads issues across 6 workstreams with chained dependencies.

## Ready Issues (no blockers)

```
bd ready
```

| ID | P | Type | Title |
|----|---|------|-------|
| TGR-l8h | P1 | bug | Fix √g coefficient bug in perturbation crosscheck |
| TGR-61p | P2 | task | Add DifferentialEquations.jl as weak dependency |
| TGR-4yo | P2 | feature | Define SymmetryAnsatz types for metric reduction |
| TGR-jpo | P2 | task | Analytic FP cross-check from BC params (Approach 3) |
| TGR-m3r | P3 | feature | Order 1-2 curvature invariant catalog |
| TGR-889 | P3 | feature | Generalize HypersurfaceProperties to SubmanifoldProperties |
| TGR-dhp | P3 | feature | Perfect fluid stress-energy tensor |

## Recommended Next Session Priority

1. **TGR-l8h** (30 min) — Fix `1//8` → `1//4` in examples/15 line 49, validate dS crosscheck
2. **TGR-4yo** (1-2 hrs) — SymmetryAnsatz types, then TGR-xqh spherical generator
3. **TGR-61p** (30 min) — DiffEq weak dep setup (unblocks geodesic + TOV chains)

## Workstream Dependency Graph

### dS Crosscheck (3 issues)
```
TGR-l8h [P1] Fix √g coefficient
  └→ TGR-76k [P1] Validate full dS pipeline
TGR-jpo [P2] Analytic FP cross-check (independent)
```

### Symmetry Ansatz (6 issues)
```
TGR-4yo [P2] Types
  ├→ TGR-xqh [P2] Spherical ──┐
  ├→ TGR-aqj [P2] FLRW ───────┼→ TGR-soo [P2] Tests
  ├→ TGR-f0c [P3] Axial ──────┘
  └→ TGR-zkw [P3] Killing rules
```

### Geodesic Integration (5 issues)
```
TGR-61p [P2] DiffEq weak dep
  └→ TGR-egq → TGR-nzk → TGR-ak3 → TGR-adq
```

### TOV Solver (5 issues)
```
TGR-dhp [P3] Matter tensor
  └→ TGR-vhp → TGR-r4i → TGR-3q9 (also needs TGR-61p) → TGR-8ea
```

### Submanifolds (5 issues)
```
TGR-889 [P3] SubmanifoldProperties
  ├→ TGR-760 GHY ───────┐
  ├→ TGR-0mg Gauss-Codazzi ┼→ TGR-ugs Tests
  └→ TGR-prb Israel junction
```

### Invar Database (4 issues)
```
TGR-m3r [P3] Order 1-2 catalog
  ├→ TGR-3gx Syzygies ──┐
  └→ TGR-141 Cubics ─────┼→ TGR-dcw Tests
```

## Key Files

| File | Purpose |
|------|---------|
| `examples/26_6deriv_spectrum_showcase.jl` | **NEW**: Full 3-path showcase |
| `results/6deriv_spectrum_results.jl` | **NEW**: Machine-readable results module |
| `examples/15_perturbation_spectrum_crosscheck.jl` | Has √g bug at line 49 |
| `src/gr/hypersurface.jl` | Starting point for submanifold generalization |
| `src/gr/killing.jl` | Starting point for Killing equation enforcement |
| `src/algebra/ansatz.jl` | Ansatz framework (used by symmetry ansatz) |
| `src/components/symbolic_metric.jl` | Symbolic metric pipeline |

## Infrastructure Notes

- Beads DB was reinitialized (`bd init --force --prefix TGR`). Old issue IDs (TGR-af4a etc.) are gone from the DB but referenced in git history.
- Infrastructure issues (BinaryBuilder TGR-byb, Pkg registration TGR-erv) deprioritized per user request.
- 3 `@test_skip` benchmarks remain (spherical harmonics, bitensors) — stretch goals.
