# Handoff: 6-Derivative Gravity Independent Cross-Check

## Status: Covariant Pipeline Done, BC Param Extraction Blocked on √g

### What's Done (this session)
- `covariant_output=true` kwarg on `define_metric_perturbation!` — produces `∇h` instead of `∂h + Γ₀h`
- Modified `δchristoffel` (Palatini identity) and `δriemann` (covariant linear part + Γ₀ absorption)
- Extended `to_fourier` with `covd_names` kwarg for named CovDs
- End-to-end pipeline validated: expand → commute_covds → Fourier → kernel → spin_project
- All 5851 tests pass, bench_12 testset 12.7 green, examples/15 runs successfully
- Commit: `01d8f4a` pushed to master

### What's Blocking
Spin projection of `δ²(L)` (without `√g` determinant) gives **non-zero gauge sectors** (spin-1 ≠ 0, spin-0w ≠ 0) on MSS. The gauge-invariant quadratic Lagrangian is `δ²(√g · L)`, not `δ²(L)`. The `√g` cross-terms are needed to make gauge sectors vanish and enable clean BC param extraction.

### Three Approaches to Complete the Cross-Check

All three should give identical BC parameters, providing triple independent confirmation.

---

## Approach 1: √g Perturbation (Most Direct)

**Idea**: Manually add the `√g` perturbation terms to get the full `δ²(√g · L) / √g₀`.

**Formula** (exact for scalar Lagrangians on MSS):
```
δ²(√g · L) / √g₀ = δ²L + tr(h) · δ¹L + L₀ · [−½ h^{ab}h_{ab} + ¼(tr h)²]
```
where:
- `tr(h) = g^{ab} h_{ab}` (trace of perturbation)
- `L₀` = background value of L (e.g., `(4Λ)³` for R³)
- `δ¹L` = first-order perturbation of L (e.g., `3R₀² · δR = 48Λ² · δR` for R³)
- `h^{ab} h_{ab}` = contraction via background metric

**Implementation**:
1. Compute `δ²L` via `expand_perturbation(L, mp_cov, 2)` — already works
2. Compute `δ¹L` via `expand_perturbation(L, mp_cov, 1)` — already works
3. Evaluate `L₀` on background via `simplify(L; registry=reg)` — already works
4. Build `tr(h)`:
   ```julia
   trh = Tensor(:g, [up(:p), up(:q)]) * Tensor(:h, [down(:p), down(:q)])
   ```
5. Build `h² = h^{ab}h_{ab}`:
   ```julia
   hsq = Tensor(:h, [down(:r), down(:s)]) * Tensor(:g, [up(:r), up(:u)]) *
         Tensor(:g, [up(:s), up(:v)]) * Tensor(:h, [down(:u), down(:v)])
   ```
6. Assemble: `full_δ2 = δ²L + trh * δ¹L + L₀ * ((-1//2)*hsq + (1//4)*trh*trh)`
7. Simplify with `commute_covds_name=:∇g` → Fourier → kernel → spin_project
8. Verify spin-1 = 0, spin-0w = 0 (gauge invariance check)
9. Extract BC params from spin-2 and spin-0s form factors

**Key detail**: On MSS, `L₀` and `δ¹L` simplify to known Λ expressions:
- R³: `L₀ = 64Λ³`, `δ¹L = 48Λ² · δR + 0` (since R₀=4Λ)
- Actually: `δ¹(R³) = 3R₀² · δ¹R`, and `δ¹R = δricci_scalar(mp, 1)`

**Verification**: After spin projection, the form factors should be:
```
f₂(k², Λ) / 5 = polynomial matching bc_R3(1, Λ) params
f₀(k², Λ) / 1 = polynomial matching bc_R3(1, Λ) params
```
The mapping from BC params `(a,b,c,e)` to form factors is via Bueno-Cano Eqs. 17-19.

---

## Approach 2: Linearized EOM via `variational_derivative`

**Idea**: Use `variational_derivative` or `metric_variation` to get the linearized equations of motion on MSS, which IS the kinetic kernel.

**Implementation**:
1. Build the Lagrangian (e.g., `R³`) as a TensorExpr
2. `eom = metric_variation(L, :g, down(:a), down(:b))` — gives `(1/√g) δ(√g·L)/δg^{ab}`
   - **Note**: `var_lagrangian` is just a wrapper for `metric_variation`
3. This gives the EOM for the full metric. Linearize around MSS:
   - Replace `g → g₀ + εh` and expand to first order in h
   - The linearized EOM `G^{ab}_{cd} h^{cd} = 0` gives the kinetic operator
4. The kinetic kernel K = G^{ab}_{cd} can then be spin-projected

**Caveat**: `metric_variation` in `variation.jl:110-162` does a Leibniz rule on the metric factors, but may not handle curvature tensors' implicit metric dependence (Christoffel → metric → ∂g). Need to verify this works for curvature-dependent Lagrangians.

**Alternative**: Instead of `metric_variation`, use:
```julia
δ¹_eom = expand_perturbation(eom_background, mp_cov, 1)
```
where `eom_background` is the EOM tensor expression evaluated on MSS. This linearizes the EOM directly.

**Files**: `src/perturbation/variation.jl` — `variational_derivative`, `metric_variation`, `var_lagrangian`

---

## Approach 3: FP Form from BC Params (Analytic Construction)

**Idea**: Construct the Fierz-Pauli kinetic kernel analytically from the known BC params, Fourier-transform it, spin-project, and compare against the perturbation engine output.

**Implementation**:
1. Use `BuenoCanoParams` from `src/action/kernel_extraction.jl:432-476`
2. For each invariant, the BC params are known functions: `bc_R3(γ, Λ)`, etc.
3. Construct the kinetic kernel in momentum space:
   ```
   K^{abcd}(k², Λ) = e · K_EH^{abcd}(k²) + a · K_W^{abcd}(k²) + b · K_R²^{abcd}(k²) + c · K_Ric²^{abcd}(k²)
   ```
   where `K_EH`, `K_W`, `K_R²`, `K_Ric²` are the standard momentum-space kernels
4. The spin projections of these standard kernels are known:
   - `Tr(K_EH · P²) = 5k²`, `Tr(K_EH · P⁰ˢ) = -k²` (from bench_13 / examples/15 flat)
   - `Tr(K_R² · P²) = 0`, `Tr(K_R² · P⁰ˢ) = 3k⁴`
   - `Tr(K_Ric² · P²) = 5k⁴/4`, `Tr(K_Ric² · P⁰ˢ) = k⁴`
   - The Weyl² (= `a` param) kernel: `Tr(K_W · P²) = ?`, `Tr(K_W · P⁰ˢ) = 0`
5. Express the predicted form factors as polynomials in k² and Λ
6. Compare against perturbation engine's numeric output at test points

**Key advantage**: This is purely algebraic — no heavy computation needed.

**Files**:
- `src/action/kernel_extraction.jl:421-530` — `BuenoCanoParams`, `bc_R3`, `bc_RRicSq`, etc.
- `src/action/spin_projectors.jl` — Barnes-Rivers projectors
- `examples/13_6deriv_particle_spectrum.jl` — has the flat-space spectrum derivation

---

## Helper: Evaluating Expressions with Λ

The `_eval_spin_scalar` function doesn't handle `Tensor(:Λ, [])`. Use this helper (already in `examples/15`):

```julia
function _subst_lambda(expr::Tensor, Λ_val)
    expr.name == :Λ && isempty(expr.indices) && return TScalar(Λ_val)
    expr
end
_subst_lambda(s::TScalar, _) = s
function _subst_lambda(p::TProduct, Λ_val)
    tproduct(p.scalar, TensorExpr[_subst_lambda(f, Λ_val) for f in p.factors])
end
function _subst_lambda(s::TSum, Λ_val)
    tsum(TensorExpr[_subst_lambda(t, Λ_val) for t in s.terms])
end
function _subst_lambda(d::TDeriv, Λ_val)
    TDeriv(d.index, _subst_lambda(d.arg, Λ_val), d.covd)
end
```

Then: `_eval_spin_scalar(_subst_lambda(spin_result, 0.3), 1.7)` works.

---

## Key Files

| File | What | Relevance |
|------|------|-----------|
| `src/perturbation/metric_perturbation.jl` | MetricPerturbation struct + define | `covariant_output` kwarg ✓ |
| `src/perturbation/expand.jl` | δΓ, δR, δRic, expand_perturbation | Covariant δΓ/δR ✓ |
| `src/svt/fourier.jl` | to_fourier | `covd_names` kwarg ✓ |
| `src/perturbation/variation.jl` | variational_derivative, metric_variation | Approach 2 |
| `src/action/kernel_extraction.jl` | extract_kernel, spin_project, BC params | Approach 3 |
| `src/action/spin_projectors.jl` | Barnes-Rivers P², P¹, P⁰ˢ, P⁰ʷ | Spin projection |
| `src/gr/metric.jl:249-264` | metric_det_expr, sqrt_det_expr | √g helper |
| `examples/14_cubic_bc_params.jl` | Parametric Riemann BC verification | Ground truth |
| `examples/15_perturbation_spectrum_crosscheck.jl` | Current cross-check (structural) | Starting point |
| `benchmarks/bench_12_6deriv_dS.jl` | 6-deriv benchmark with testset 12.7 | Regression guard |

## Suggested Execution Order

1. **Approach 1** (√g): highest payoff, most direct — extend examples/15 with the √g correction formula, verify spin-1=0 and spin-0w=0, then extract BC params
2. **Approach 3** (FP analytic): quick algebraic check — construct predicted form factors from BC params, compare at numeric test points
3. **Approach 2** (variational): verify `metric_variation` works on curvature Lagrangians, use as third independent check

## Important Caveats

- `commute_covds` may need `maxiter=200` for large expressions (300-1500 terms)
- Spin projection: `Tr(K·P^J)`, NOT `f_J` — divide by `dim(sector)` = {5,3,1,1} to get form factors
- `δricci_scalar(mp, n)` returns the ε^n COEFFICIENT, not `(1/n!) d^n/dε^n`
- On MSS: `R₀ = 4Λ`, `Ric₀_{ab} = Λ g_{ab}`, `Riem₀_{abcd} = (Λ/3)(g_{ac}g_{bd} - g_{ad}g_{bc})`
- The Bueno-Cano convention uses `Λ_BC = Λ/3` (their Λ vs TGR's Λ)
