# HANDOFF: Close Remaining Issues

## Current State (2026-03-18)

- **337,311 tests passing, 0 failures, 53 benchmarks green**
- **TGR-9ay [P0] RESOLVED** — kernel extraction for 6-deriv box terms (R□R, Ric□Ric) now works
- **Canonicalize restored** to pre-March 17 all-free mode (March 17 proper-dummy-mode caused 9 regressions)
- Full documentation round complete (README, tutorial, 13 API pages, examples guide)
- Run `bd ready` to see available work; `bd stats` for overview

## Next: Chaos Branch Integration

Master is GREEN: 337,311 tests, 0 failures, 53 benchmarks. Start integrating march15-preserve subsystems ONE AT A TIME. See memory/project_chaos_integration.md for the full plan.

**Already merged:** c40367c (symmetrize_covds improvements)

**Next safe merges** (existing-file enhancements, no new directories):
- 255b4a5: sort_covds_to_box enhancements
- 622e03b: sort_covds_to_div
- 5300ce7: contraction_ansatz new method
- da7df98: all_contractions(expr, metric)

**Then:** ground truth tests, then new subsystem directories (spinors first).

**CRITICAL:** Follow the 9 rules below. ONE commit at a time. Full test suite after each. STOP if any test regresses.

## The 9 Rules (MANDATORY for all work on this project)

1. **SKEPTICISM**: All subagent work, all previous agent work, handoffs — treat with SKEPTICISM. Verify everything twice.
2. **DEEP BUGS**: The bugs are DEEP, COMPLEX, and INTERLOCKED. Do not underestimate.
3. **NO BANDAIDS**: No bandaid solutions. Prefer best-practices full solutions at expense of downstream work.
4. **3-AGENT WORKFLOW**: Always spawn 3 subagents before any code change:
   - Subagent 1: Research how xAct does it (local copy at `reference/xAct/xAct/`)
   - Subagent 2 & 3: Independently suggest solutions
   - Then implement.
5. **MANDATORY REVIEW**: After implementing a code change, spawn a rigorous reviewer agent to carefully and skeptically review. No exceptions.
6. **PHYSICS IS GROUND TRUTH**: Ground truth is the physics, not pinned numbers in handoffs or anything else. Only string matches with local ground truth references (papers in `reference/ground_truth/`) are acceptable.
7. **TARGETED TESTING**: Tests take forever — only run targeted testing. OR run full tests in background.
8. **REPEAT RULES**: Repeat the rules occasionally to keep the attention mechanism working optimally.
9. **DO NOT UNDERESTIMATE**: This task is DEEPLY NONTRIVIAL.

### TGR-9ay Resolution (2026-03-18)

**Root cause**: `_distribute_derivs_sums(::TDeriv)` in `kernel_extraction.jl` did not call `expand_products` on the inner expression before checking for TSum. This meant `expand_products` later created `TDeriv(TSum)` structures from products-containing-sums inside derivative arguments, which `_unwrap_field_chain` could not traverse.

**Fix**: One line added — `inner = expand_products(inner)` — mirroring xAct/xTensor's automatic `CovD[expr_Plus, ders__]` distribution. Verified by 3-agent workflow (xAct research, 2 independent solution proposals, rigorous reviewer).

**Impact**: R□R and Ric□Ric kernel extraction now produces 36 terms each (was 0). All gauge sectors satisfy spin-1=0, spin-0w=0. No regressions. 29 new regression tests added.

## The 3 Remaining Issues

### 1. TGR-3q9 [P3, READY] — `solve_tov()` in DiffEq extension

**What**: Add `solve_tov` and `mass_radius_curve` to `ext/TensorGRDiffEqExt.jl`.

**Pattern to follow**: Copy the `integrate_geodesic` pattern already in that file (lines 28-63). The TOV version is structurally identical:

```julia
# In ext/TensorGRDiffEqExt.jl, add:

struct TOVSolution
    r_surface::Float64      # radius where p=0
    M_total::Float64        # m(r_surface)
    r::Vector{Float64}      # radial coordinate
    m::Vector{Float64}      # enclosed mass profile
    p::Vector{Float64}      # pressure profile
    rho::Vector{Float64}    # density profile
    raw::Any                # underlying ODE solution
end

function TensorGR.solve_tov(tov::TensorGR.TOVSystem, r_max::Real;
                             solver=Tsit5(), kwargs...)
    # 1. Build ContinuousCallback to stop when p(r) <= 0:
    #    condition(u, r, integrator) = u[2]  (pressure)
    #    affect!(integrator) = terminate!(integrator)
    #    cb = ContinuousCallback(condition, affect!)
    # 2. Build ODEProblem from tov_rhs!, tov.u0, (tov.r0, r_max), tov
    # 3. Solve with solver; merge callback with any user callbacks
    # 4. Extract profiles: r=sol.t, m=sol[1,:], p=sol[2,:], rho from EOS inversion
    # 5. r_surface = sol.t[end], M_total = sol[1,end]
    # 6. Return TOVSolution
end

function TensorGR.mass_radius_curve(eos::TensorGR.EquationOfState,
                                     rho_c_range::AbstractVector;
                                     r_max=50.0, kwargs...)
    # Sweep central densities, call solve_tov for each, collect (R, M) pairs
    # Return (R_vals, M_vals)
end
```

**Stubs needed in `src/solvers/tov.jl`** (or `src/geodesics/geodesic.jl` pattern):
- Add `TOVSolution` struct to base TensorGR (like `GeodesicSolution`)
- Add `function solve_tov end` and `function mass_radius_curve end` stubs
- Export both + `TOVSolution`

**Files to modify**:
- `src/solvers/tov.jl` — add `TOVSolution` struct + stubs
- `src/TensorGR.jl` — add exports
- `ext/TensorGRDiffEqExt.jl` — add implementation (~80 lines)

**Key detail**: The `_density_from_pressure` function already exists in `src/solvers/tov.jl` for EOS inversion.

**Testing**: The test for this is TGR-8ea (next issue).

---

### 2. TGR-8ea [P3, BLOCKED by TGR-3q9] — Validate TOV solver

**What**: Create `test/test_tov.jl` with 5 validation tests.

**Tests to write** (~200 lines):

1. **Constant density sphere**: `BarotropicEOS(0)` with rho=rho_c everywhere (dust). The analytic interior solution is:
   ```
   p(r) = rho_c * (sqrt(1 - 2Mr²/R³) - sqrt(1 - 2M/R)) / (3*sqrt(1 - 2M/R) - sqrt(1 - 2Mr²/R³))
   ```
   But simpler: just verify m(r) = (4/3)pi*rho_c*r³ grows linearly.

2. **Polytropic EOS** (K=100, Gamma=2): integrate, verify M_total > 0 and R_surface > 0. The solution should terminate at finite radius.

3. **Buchdahl limit**: For any solution, verify `2*M_total/R_surface < 8/9`.

4. **Mass-radius curve**: For polytropic EOS, sweep rho_c from 0.1 to 2.0, verify we get a curve of (R, M) points.

5. **Conservation check**: At the surface, pressure should be ~0 (within tolerance).

**Conditional on DifferentialEquations.jl**: Use `@eval using DifferentialEquations` with try/catch, skip if unavailable (same pattern as `test/test_geodesics.jl`).

**Files**: Create `test/test_tov.jl`, add `include("test_tov.jl")` to `test/runtests.jl`.

---

### 3. TGR-76k [P1, IN_PROGRESS] — dS crosscheck

**Status**: Flat-space crosscheck is COMPLETE and fully validated. The dS part is blocked by a **theoretical issue**, not a code bug.

**The Problem**: On de Sitter (MSS) background, the flat-space Barnes-Rivers projectors give **wrong results**:
- `spin_project` with flat θ_{μν} = g_{μν} - k_μk_ν/k² gives spin-2 = 2.5 - 5Λ at k²=1
- The Bueno-Cano prediction (analytic, correct) gives spin-2 = 2.5 (constant, independent of Λ)
- The discrepancy grows linearly with Λ: it's exactly -5Λ

**Root Cause**: On MSS, the Lichnerowicz operator eigenvalues differ from flat:
- Flat: ∇² → -k² for all spin sectors
- MSS: ∇² → -(k² - c_J Λ) where c_J depends on spin J
- The Barnes-Rivers projectors need to account for these Λ-dependent eigenvalue shifts

**What Would Fix It**: dS-adapted Barnes-Rivers projectors where θ_{μν} uses the MSS eigenvalue relation. Specifically, replace `k²` in the ω projector with `k² + (correction depending on spin and Λ)`. This is well-documented in the PSALTer literature (Barker et al. 2024).

**Pragmatic Closure Option**: Close this issue with a note that:
1. Flat crosscheck is fully validated (all 4 spins match FP at k²=0.5, 1.0, 2.0)
2. The 4-derivative spectrum matches analytic Buoninfante form factors for 5 parameter combinations
3. The dS extension requires a theoretical paper-level adaptation of the projectors
4. Create a follow-up issue for "dS-adapted spin projectors" if desired

**Files relevant to the dS projector extension**:
- `src/action/spin_projectors.jl` — current flat projectors (146 lines)
- `src/action/kernel_extraction.jl` — `spin_project` function (lines 332-370), `_standardize_h_indices`, `_kernel_build_projector`

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/solvers/tov.jl` | TOVSystem, setup_tov, tov_rhs! (194 lines) |
| `src/matter/eos.jl` | EquationOfState types (196 lines) |
| `ext/TensorGRDiffEqExt.jl` | DiffEq integration (65 lines, extend here) |
| `src/geodesics/geodesic.jl` | GeodesicSolution struct + stub pattern to follow |
| `src/action/spin_projectors.jl` | Barnes-Rivers projectors (for dS adaptation) |
| `src/action/kernel_extraction.jl` | spin_project, extract_kernel_direct |
| `test/test_geodesics.jl` | DiffEq-conditional test pattern to follow |

## Quick Commands

```bash
bd ready                    # see available work
bd update TGR-3q9 --status=in_progress   # claim solve_tov
julia --project -e 'using Pkg; Pkg.test()'  # full test suite (~5min)
git push                    # push to remote
bd close TGR-3q9 --reason="..." # close when done
```

## Session Close Protocol

```
[ ] git status
[ ] git add <files>
[ ] git commit -m "..."
[ ] git push
[ ] bd close <completed issues>
```
