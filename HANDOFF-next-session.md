# HANDOFF: Session 25 — 6 Issues Closed, 12 Ready

## Status: 6 closed this session, 28 total issues, 12 ready

- **All tests pass**: 6788 (Pkg.test)
- **Beads DB**: 6 closed, 22 open (12 ready, 10 blocked)

## What Was Done This Session

### TGR-l8h [P1 bug] — Fix √g coefficient in perturbation crosscheck (CLOSED)
- Fixed `1//8` → `1//4` for (tr h)² coefficient in `sqrt_g_correction()` at `examples/15_perturbation_spectrum_crosscheck.jl:49`
- Updated docstring to match

### TGR-889 [P3 feature] — Generalize HypersurfaceProperties to SubmanifoldProperties (CLOSED)
- Refactored `src/gr/hypersurface.jl`: `SubmanifoldProperties` struct with `codimension`, `normal_names`, `extrinsic_names`, `signatures` vectors
- `const HypersurfaceProperties = SubmanifoldProperties` for backward compat
- Added `define_submanifold!` for arbitrary codimension, `define_hypersurface!` is now a wrapper
- Extended `induced_metric_expr` and `projector_expr` with codim-k overloads (multiple normals/signatures)
- Extracted `_register_normal!` helper (shared by both APIs)
- All existing tests pass unchanged

### TGR-4yo [P2 feature] — SymmetryAnsatz types (CLOSED, via worktree agent)
- Created `src/gr/symmetry_ansatz.jl` (128 lines): abstract `SymmetryAnsatz` + 4 concrete types
  - `SphericalSymmetry`, `AxialSymmetry`, `StaticSymmetry`, `HomogeneousIsotropy`
- Exported from TensorGR.jl

### TGR-61p [P2 task] — DifferentialEquations.jl weak dependency (CLOSED, via worktree agent)
- Added to `[weakdeps]`, `[extensions]`, `[compat]` in Project.toml
- Created `ext/TensorGRDiffEqExt.jl` stub module
- Added to test/Project.toml
- **Note**: Agent used wrong UUID (`caa7`), fixed to correct `fbaa` post-merge

### TGR-dhp [P3 feature] — Perfect fluid stress-energy tensor (CLOSED, via worktree agent)
- Created `src/gr/matter.jl` (222 lines)
- `PerfectFluidProperties` struct, `define_perfect_fluid!`, `perfect_fluid_expr`, `get_perfect_fluid`
- Registers T^{ab} (symmetric), ρ (scalar), p (scalar), u^a (vector) + normalization rule u_a u^a = -1

### TGR-m3r [P3 feature] — Curvature invariant catalog (CLOSED, via worktree agent)
- Created `src/gr/invariants.jl` (193 lines)
- `InvariantEntry` struct, `INVARIANT_CATALOG` with 5 entries: R, R², Ric², Kretschmann, Weyl²
- `curvature_invariant(name)`, `list_invariants(; order)` API

## Ready Issues (no blockers) — Recommended Priority

```
bd ready
```

| ID | P | Type | Title | Unblocks |
|----|---|------|-------|----------|
| TGR-76k | P1 | task | Validate dS crosscheck via √g perturbation (Approach 1) | — |
| TGR-egq | P2 | feature | GeodesicEquation type and setup function | TGR-nzk |
| TGR-aqj | P2 | feature | FLRW metric ansatz generator | TGR-soo |
| TGR-xqh | P2 | feature | Spherical symmetry metric ansatz generator | TGR-soo |
| TGR-jpo | P2 | task | Analytic FP cross-check from BC params (Approach 3) | — |
| TGR-3gx | P3 | feature | Order-2 curvature syzygies as rewrite rules | TGR-dcw |
| TGR-0mg | P3 | feature | Gauss-Codazzi relations as rewrite rules | TGR-ugs |
| TGR-760 | P3 | feature | GHY boundary term extraction | TGR-ugs |
| TGR-vhp | P3 | feature | Equation of state types for matter | TGR-r4i |
| TGR-zkw | P3 | feature | Auto-register Killing equation as rewrite rules | — |
| TGR-f0c | P3 | feature | Axial symmetry metric ansatz generator | TGR-soo |
| TGR-141 | P4 | feature | Order-3 cubic curvature invariants to catalog | TGR-dcw |

## Recommended Next Session Priority

1. **TGR-76k** (P1) — Run the fixed dS crosscheck (examples/15) end-to-end, validate FP kernel
2. **TGR-xqh + TGR-aqj** (P2) — Spherical + FLRW ansatz generators (unblock TGR-soo tests)
3. **TGR-egq** (P2) — GeodesicEquation type (unblocks geodesic chain)
4. **TGR-jpo** (P2) — Analytic FP cross-check from BC params

## Workstream Dependency Graph

### dS Crosscheck (2 remaining)
```
✓ TGR-l8h Fix √g coefficient
  └→ TGR-76k [P1] Validate full dS pipeline (READY)
TGR-jpo [P2] Analytic FP cross-check (READY, independent)
```

### Symmetry Ansatz (4 remaining)
```
✓ TGR-4yo Types
  ├→ TGR-xqh [P2] Spherical (READY) ──┐
  ├→ TGR-aqj [P2] FLRW (READY) ───────┼→ TGR-soo [P2] Tests
  ├→ TGR-f0c [P3] Axial (READY) ──────┘
  └→ TGR-zkw [P3] Killing rules (READY)
```

### Geodesic Integration (4 remaining)
```
✓ TGR-61p DiffEq weak dep
  └→ TGR-egq [P2] (READY) → TGR-nzk → TGR-ak3 → TGR-adq
```

### TOV Solver (4 remaining)
```
✓ TGR-dhp Matter tensor
  └→ TGR-vhp [P3] (READY) → TGR-r4i → TGR-3q9 (needs TGR-61p ✓) → TGR-8ea
```

### Submanifolds (4 remaining)
```
✓ TGR-889 SubmanifoldProperties
  ├→ TGR-760 [P3] GHY (READY) ────────┐
  ├→ TGR-0mg [P3] Gauss-Codazzi (READY) ┼→ TGR-ugs Tests
  └→ TGR-prb [P4] Israel junction
```

### Invar Database (3 remaining)
```
✓ TGR-m3r Order 1-2 catalog
  ├→ TGR-3gx [P3] Syzygies (READY) ──┐
  └→ TGR-141 [P4] Cubics (READY) ─────┼→ TGR-dcw Tests
```

## Key Files Changed This Session

| File | Change |
|------|--------|
| `examples/15_perturbation_spectrum_crosscheck.jl` | Fixed √g coefficient 1//8 → 1//4 |
| `src/gr/hypersurface.jl` | Refactored to SubmanifoldProperties (codim-k) |
| `src/gr/symmetry_ansatz.jl` | **NEW**: SymmetryAnsatz type hierarchy |
| `src/gr/matter.jl` | **NEW**: Perfect fluid stress-energy |
| `src/gr/invariants.jl` | **NEW**: Curvature invariant catalog |
| `ext/TensorGRDiffEqExt.jl` | **NEW**: DiffEq extension stub |
| `Project.toml` | Added DifferentialEquations weakdep |
| `src/TensorGR.jl` | Added includes + exports for all new modules |
