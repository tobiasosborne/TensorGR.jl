# Next Agent Onboarding Prompt

Copy-paste this as your first message to the next Claude Code session:

---

Read HANDOFF.md completely, then read CLAUDE.md. Before doing ANY work:

1. Run `bd stats` and `bd ready -n 15` to see the current state
2. Run `git log --oneline -10` to see recent commits
3. Run `git status` to confirm clean working tree
4. Recite the 9 rules from HANDOFF.md back to me

## What you need to know

This is TensorGR.jl — abstract tensor algebra and GR in Julia (~15,000 LOC, ~3,500 tests). You are continuing a marathon build session. The project is at 235/369 issues closed. All tests pass. All code is pushed to master.

## How to work effectively

**The biggest mistake previous agents made**: trusting their own subagents without verification, and treating pinned test numbers as ground truth instead of physics papers. The ONLY real ground truth is equations from locally-downloaded physics papers in `reference/papers/` — string-matched against TensorGR output.

**What works well**:
- Pick issues from `bd ready`, mark `in_progress`, implement, test, close, commit, push
- Leaf modules (spinors, fermions, bimetric, metric-affine) are safe — implement freely
- Core pipeline changes (simplify, canonicalize, contraction) require the full 3-subagent workflow: xAct research + 2 independent solutions + reviewer
- Run targeted tests only (`julia --project -e 'using TensorGR, Test; ...'`), full suite in background at most once
- Close already-done duplicate issues aggressively — many were created speculatively

**What to avoid**:
- Don't run multiple full test suites in parallel
- Don't modify core modules without the full workflow
- Don't trust issue descriptions uncritically — the agent that created them was aggressive
- Don't add trace-free enforcement (TGR-avk is deferred — already handled by Weyl decomposition)
- Don't sort TSum terms or deriv chains in canonicalize

**API traps**:
- `FullySymmetric(4)` creates slots=[4], NOT [1,2,3,4]. Use `FullySymmetric(1,2,3,4)`
- `make_rule(lhs, rhs)` returns Vector{RewriteRule} but does NOT register them
- `symmetrize(expr, indices)` takes `Vector{Symbol}` not `Vector{TIndex}`
- For contraction rules on products, use function-based `RewriteRule` + `register_rule!` (see soldering_form.jl pattern)
- Test registries must use `with_registry(reg) do @manifold ... end` pattern

## Where the ground truth lives

12 physics papers with TeX source in `reference/papers/`:
- Hohmann (2021) xPPN: `2012_14984_src/xppnpaper.tex`
- Barker (2024) PSALTer: `barker_psalter_2406.09500.pdf`
- Brizuela (2009) xPert: `brizuela_src/xPert.tex`
- Nutma (2014) xTras: `1308_3493_src/xTras.tex`
- + 8 more (see HANDOFF.md for full list)

xAct Mathematica source: `reference/xAct/`

## Suggested work order

1. **TGR-6s5**: Bimetric mass eigenstates diagonalization (just unblocked, builds on linearize.jl)
2. **NP chain**: TGR-jvn (Bianchi in NP form), TGR-900 (18 field equations)
3. **Validation**: TGR-v8u (EFTofPNG 1PN EIH — needs Levi/Steinhoff paper research first)
4. **Metric-affine chain**: TGR-swh.7 (Brauer algebra 11-piece decomposition)
5. **Clifford chain**: TGR-dai.7 (gamma trace validation to order 6)

Avoid core pipeline issues (TGR-ulo.2, TGR-xlu.5) unless you have context budget for the full 3-agent workflow.

Now: recite the rules, then `bd ready` and get to work.
