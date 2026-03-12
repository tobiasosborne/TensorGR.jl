# HANDOFF: Session 20 — Codebase Audit, Test Robustness, sym_det Generalization

## Status: CLEAN — All tests pass, 3 commits on master

- **5640 tests pass** via `include` path (no Symbolics), 0 errors
- **6042 tests pass** via `Pkg.test()` (with Symbolics), 0 errors
- Commits: `fbd55ad`, `086374f`, `c066fbc` (not yet pushed — remote needs auth)

## What Was Done This Session

### 1. Codebase Audit (6 parallel research agents)

Comprehensive audit of the entire project, findings below.

**Test gaps**: NONE. All 72 source files have corresponding test coverage. Zero @test_broken, zero TODO/FIXME in src/. 3 @test_skip in benchmarks (stretch goals: spherical harmonics, bitensors).

**Stubs status** (CLAUDE.md was outdated):
- `sort_covds_to_box` — FULLY IMPLEMENTED (detects ∂_a(∂^a(T)) → g^{ab}∂_a∂_b T)
- `lorentzian_contract` — FULLY IMPLEMENTED (applies metric sign flips in foliation)
- `sort_covds_to_div` — intentional no-op (divergence patterns already canonical in AST)
- `sym_det`/`sym_inv` — was limited to 3×3, now generalized (see below)
- `linearize` — order 1 only (higher orders throw error)
- `gauge_transformation` — orders 1-2 only (higher orders throw error)

**Documentation coverage**: 80.2% of 343 exports documented. 68 gaps:
- 9 `bc_*` functions had no docstrings → FIXED this session
- Product manifold API (17 functions) — no dedicated doc page
- Symbolic components API (11 functions) — no dedicated doc page
- Kernel extraction API (10 functions) — only partially in action.md
- Scalar algebra (5 functions) — no doc coverage
- See full report in agent output for priority-ordered list

**P2 pipeline status** (6-deriv gravity):
- Path A (covariant spin projection): COMPLETE, verified against Buoninfante
- Path B (SVT 3+1 foliation): NOT STARTED — this is the critical unblock
- Cross-check (Path A vs Path B): BLOCKED on Path B
- Handoff doc: `HANDOFF-6deriv-crosscheck.md` has 3 approaches for √g correction

### 2. Fix: Test Robustness (commit fbd55ad)

`test_cas_integration.jl` and `test_symbolic_components.jl` used `using Symbolics` unconditionally, causing an error when tests were run via `include("test/runtests.jl")` instead of `Pkg.test()`.

**Fix**: Guard the Symbolics-dependent includes in `runtests.jl` with:
```julia
if isdefined(Main, :Symbolics) || try @eval(using Symbolics); true catch; false end
    include("test_cas_integration.jl")
    import TensorGR: simplify
    include("test_symbolic_components.jl")
else
    @info "Skipping CAS/symbolic component tests: Symbolics.jl not available"
end
```

Also updated CLAUDE.md:
- Fixed outdated "Stubs (Not Yet Implemented)" → "Stubs / Stretch Goals"
- Corrected test count from "4,100+" to "~2,900"

### 3. Feature: Generalize sym_det/sym_inv (commit 086374f)

`sym_det` and `sym_inv` were limited to 3×3 matrices, throwing errors for larger. Generalized using recursive cofactor expansion:

- Fast paths for 1×1, 2×2, 3×3 retained (no performance change)
- New: general n×n via `_sym_minor` helper + cofactor expansion along first row
- Tested: 4×4 (det=85, inverse verified M·M⁻¹=I) and 5×5 (det=492, inverse verified)
- 17 new tests added to `test/test_quadratic_action.jl`

**Why it matters**: SVT sector analysis with >3 fields needs 4×4+ quadratic forms. The 3×3 limit was blocking Step 2.2 (SVT QuadraticForms).

### 4. Docs: bc_* Function Docstrings (commit c066fbc)

All 9 `bc_*` functions (`bc_EH`, `bc_R2`, `bc_RicSq`, `bc_R3`, `bc_RRicSq`, `bc_Ric3`, `bc_RRiem2`, `bc_RicRiem2`, `bc_Riem3`) now have proper docstrings with signatures, parameter descriptions, and which BC params (a,b,c,e) each term produces.

## Files Changed

| File | Change |
|------|--------|
| `CLAUDE.md` | Fix stubs section, correct test count |
| `test/runtests.jl` | Guard Symbolics-dependent test includes |
| `src/action/quadratic_action.jl` | Generalize sym_det/sym_inv to arbitrary n×n |
| `test/test_quadratic_action.jl` | Add 4×4 det + inverse tests (17 new) |
| `src/action/kernel_extraction.jl` | Add proper docstrings to 9 bc_* functions |

## Beads Status

Beads DB is stuck between SQLite and Dolt versions. `bd` commands fail with:
```
Error: failed to connect to dolt server: invalid database name "TensorGR.jl"
```
The `.beads/issues.jsonl` file is readable and has the full issue list. To fix:
- Either downgrade to `bd v0.49.6` (SQLite) or run `bd migrate --to-dolt` with a running Dolt server.
- The backup is at `.beads/beads.backup-pre-dolt-20260312-162256.db`

## Open Issues (from .beads/issues.jsonl)

### P2 — Active Pipeline
| ID | Title | Status | Blocked By |
|----|-------|--------|------------|
| TGR-pr04 | Step 2.2: SVT QuadraticForms + propagators (flat) | open | — |
| TGR-tztc | Step 2.3: Cross-check Path A vs Path B (flat) | open | TGR-pr04 |
| TGR-j6r9 | Step 5: Tests + benchmark for symbolic spectrum | open | TGR-tztc |
| TGR-af4a | Step 6: Example script + results + module integration | open | TGR-j6r9 |

### P2 — Infrastructure
| ID | Title | Status |
|----|-------|--------|
| TGR-byb | L1: BinaryBuilder for xperm.c | open |
| TGR-erv | L3: Pkg registration | open |

### P3/P4 — Future
| ID | Title | Priority |
|----|-------|----------|
| TGR-1kw | G4: Submanifolds/boundaries | P3 |
| TGR-61p | Geodesic equation ODE integration | P3 |
| TGR-dhp | TOV equation solver | P3 |
| TGR-293h | Symmetry-reduced metric ansatz | P4 |
| TGR-38d | H6: Invar database | P4 |

## Recommended Next Steps (Priority Order)

### 1. TGR-pr04: SVT QuadraticForms + Propagators (Path B)

**Most impactful** — unblocks 3 downstream issues. Estimated 150 lines + 20 tests.

The idea: decompose δ²S using 3+1 foliation (instead of covariant Barnes-Rivers projectors), extract SVT sector quadratic forms, compute propagators, and verify they match the covariant Path A results.

**Implementation plan**:
1. Build flat-background foliation: `define_foliation!(reg, :flat31; manifold=:M4, temporal=0, spatial=[1,2,3])`
2. Decompose perturbation: `foliate_and_decompose(δ²S, :h; foliation=fol)` → Dict of SVT sectors
3. For scalar sector: extract 3×3 QuadraticForm for (Φ, B, ψ) fields
4. Compute propagator via `sym_inv` (now supports 3×3+)
5. Verify scalar form factors match Path A's `f₀(z)`
6. Verify tensor sector matches Path A's `f₂(z)`

**Key files**:
- `src/foliation/` — complete, ready to use
- `src/svt/decompose.jl` — SVTFields, svt_substitute
- `src/action/quadratic_action.jl` — QuadraticForm, sym_inv (now generalized)
- `examples/08_postquantum_gravity.jl` — 4th-order analog template

### 2. Documentation Gaps

Add API doc pages for:
- Product manifolds (17 functions, extend `docs/src/api/gr.md`)
- Symbolic components (11 functions, new page `docs/src/api/symbolic_components.md`)
- Kernel extraction (10 functions, extend `docs/src/api/action.md`)

### 3. Fix Beads DB

Either:
- `bd migrate --to-dolt` (requires running Dolt server: `bd dolt start`)
- Or downgrade `bd` to v0.49.6

### 4. Push to Remote

The 3 commits are local only. Remote push needs GitHub auth (HTTPS remote at `https://github.com/tobiasosborne/TensorGR.jl.git`). Either:
- Configure SSH remote: `git remote set-url origin git@github.com:tobiasosborne/TensorGR.jl.git`
- Or use a GitHub token

## Key Caveats

- `commute_covds` may need `maxiter=200` for large expressions (300-1500 terms)
- Spin projection: `Tr(K·P^J)`, NOT `f_J` — divide by `dim(sector)` = {5,3,1,1}
- On MSS: R₀ = 4Λ, Ric₀ = Λg, Riem₀ = (Λ/3)(g⊗g − g⊗g)
- Bueno-Cano convention: Λ_BC = Λ/3 (their Λ vs TGR's Λ)
- `_expand_pert(tensor, mp, 0)` returns tensor itself (background), NOT zero
- `delta_riemann`/`delta_ricci`/`delta_ricci_scalar` return ZERO at order 0
