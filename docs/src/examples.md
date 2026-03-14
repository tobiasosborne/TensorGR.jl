# Examples

TensorGR.jl ships with 27 numbered example scripts in `examples/`, plus two
verification suites (`course_verification/` and `wald_verification/`).  Run any
example with:

```bash
julia --project examples/01_getting_started.jl
```

Some examples support multi-threaded execution (noted below).  Use
`julia -t4 --project` or `julia -t8 --project` where indicated.

---

## Quick Reference Table

| # | File | Description | Key functions | Lines |
|---|------|-------------|---------------|-------|
| 01 | `01_getting_started.jl` | Manifold setup, metric contraction, delta trace | `@manifold`, `simplify`, `to_unicode` | 54 |
| 02 | `02_covariant_derivatives.jl` | Covariant derivatives, Christoffel expansion, Bianchi identity | `@covd`, `covd_to_christoffel`, `commute_covds` | 58 |
| 03 | `03_curvature_decomposition.jl` | Weyl decomposition, Gauss-Bonnet, Kretschner scalar | `riemann_to_weyl`, `weyl_to_riemann`, `kretschmann_expr` | 71 |
| 04 | `04_perturbation_theory.jl` | Linearized gravity: metric perturbation, delta Ricci | `define_metric_perturbation!`, `expand_perturbation` | 67 |
| 05 | `05_schwarzschild.jl` | Schwarzschild Christoffels, Riemann, Kretschmann in coordinates | `define_chart!`, `CTensor`, `metric_riemann`, `metric_kretschmann` | 106 |
| 06 | `06_exterior_calculus.jl` | Differential forms, wedge products, Hodge dual, Cartan formula | `define_form!`, `wedge`, `exterior_d`, `hodge_dual` | 72 |
| 07 | `07_gauge_theory.jl` | SU(2) gauge bundle, mixed spacetime/gauge indices | `define_vbundle!`, `Tensor` with `:SU2` indices | 83 |
| 08 | `08_postquantum_gravity.jl` | Onsager-Machlup action for fourth-derivative gravity | `expand_perturbation`, `to_fourier`, `QuadraticForm` | 260 |
| 09 | `09_compare_with_reference.jl` | Cross-check fourth-derivative propagators against reference | `delta_ricci`, `spin2_projector`, `spin0s_projector` | 288 |
| 10 | `10_onsager_machlup_R2_RicciSq.jl` | Propagators for R^2 + Ricci^2 action (all SVT sectors) | `expand_perturbation`, `to_fourier`, `extract_kernel` | 396 |
| 11 | `11_6deriv_gravity_dS.jl` | Cubic curvature invariants on de Sitter (parallel support) | `maximally_symmetric_background!`, `expand_perturbation` | 225 |
| 12 | `12_product_manifolds.jl` | Product manifold M1 x M2 curvature decomposition | `define_product_manifold!`, `product_metric`, `product_einstein` | 132 |
| 13 | `13_6deriv_particle_spectrum.jl` | Full particle spectrum of six-derivative gravity | `spin2_projector`, `spin0s_projector`, Barnes-Rivers projectors | 455 |
| 14 | `14_cubic_bc_params.jl` | Bueno-Cano parameters for cubic curvature invariants | Pure LinearAlgebra (no TensorGR), parametric Riemann | 360 |
| 15 | `15_perturbation_spectrum_crosscheck.jl` | Cross-check Bueno-Cano spectrum via TGR perturbation engine on MSS | `expand_perturbation`, `to_fourier`, `extract_kernel` | 228 |
| 26 | `26_6deriv_spectrum_showcase.jl` | Three-path showcase: Barnes-Rivers, SVT, Bueno-Cano compared | All three spectrum computation paths combined | 278 |
| 27 | `27_geodesic_orbits.jl` | Schwarzschild geodesic orbits, circular orbit, energy conservation | `GeodesicEquation`, numerical integration | 228 |
| 16--25 | `16_*.jl` -- `25_*.jl` | Internal debug/development scripts (see below) | Various pipeline diagnostics | 91--274 |

---

## Getting Started (01--03)

These three examples introduce the core abstractions: manifolds, metrics,
indices, and the simplify pipeline.

### 01 -- Getting Started

Registers a 4D Lorentzian manifold with `@manifold`, then verifies the two most
fundamental tensor identities: metric contraction `g^{ab} g_{bc} = delta^a_c`
and the delta trace `delta^a_a = 4`.  This is the minimal example that
exercises the registry, AST construction, and the simplify fixed-point loop.

### 02 -- Covariant Derivatives

Defines a metric-compatible covariant derivative with `@covd` and expands
`nabla_a V^b` into partial derivatives plus Christoffel symbols.  It then
commutes two covariant derivatives on a vector to produce the Riemann
commutator term `[nabla_a, nabla_b] V^c = R^c_{dab} V^d`, and verifies the
contracted Bianchi identity `nabla^a G_{ab} = 0`.

### 03 -- Curvature Decomposition

Decomposes the Riemann tensor into its Weyl, Ricci, and scalar-curvature
parts and verifies the roundtrip `Riemann -> Weyl -> Riemann`.  Constructs the
Gauss-Bonnet topological invariant and the Kretschmann scalar `R_{abcd} R^{abcd}`,
demonstrating the curvature conversion and contraction APIs.

---

## Gravity Applications (04--05, 08--15, 26)

These examples demonstrate increasingly sophisticated computations in classical
and quantum gravity, from linearized perturbation theory through full particle
spectrum extraction for higher-derivative theories.

### 04 -- Perturbation Theory

Introduces linearized gravity by defining a metric perturbation `g -> g + eps*h`
and computing the first-order changes to the inverse metric, Christoffel
symbols, Ricci tensor, and Ricci scalar.  Verifies the standard formula
`delta(g^{ab}) = -g^{ac} g^{bd} h_{cd}` and the structure of the linearized
Einstein equations.

### 05 -- Schwarzschild Spacetime

Builds the Schwarzschild metric as a `CTensor` in coordinates `(t, r, theta, phi)`
and computes the Christoffel symbols, Riemann tensor, Ricci tensor (confirming
vacuum `R_{ab} = 0`), and the Kretschmann scalar `K = 48 M^2 / r^6`.  This
exercises the full component computation pipeline including `define_chart!`.

### 08 -- Postquantum Gravity (Onsager-Machlup Action)

Carries out the complete pipeline for the quadratic action of linearized
fourth-derivative gravity: perturbation engine to get linearized Ricci,
3+1 foliation to split spacetime indices, SVT decomposition into Bardeen
gauge fields, symbolic quadratic form via Symbolics.jl, and propagator
extraction.  This is the flagship end-to-end example.

### 09 -- Comparison with Reference

Cross-checks the results of example 08 against the analytic reference for
fourth-derivative gravity propagators.  Verifies the scalar kinetic matrix
entries `M_{Phi Phi}`, `M_{Phi psi}`, `M_{psi psi}` and the determinant
against Theorem 5.1 of the reference, including careful sign conventions.

### 10 -- Onsager-Machlup: R^2 + Ricci^2

Computes the full two-point functions (propagators) for all degrees of freedom
of linearized gravity with the quadratic-in-curvature action
`alpha R^2 - beta R_{mu nu} R^{mu nu}` around flat space.  Decomposes into
tensor, vector, and scalar SVT sectors and extracts the Bardeen-gauge kinetic
matrices for all polarizations.

### 11 -- Six-Derivative Gravity on de Sitter

Computes the second-order perturbation `delta^2[I_i]` for all six independent
cubic curvature monomials (including the Goroff-Sagnotti invariant
`R_{ab}^{cd} R_{cd}^{ef} R_{ef}^{ab}`) on a maximally symmetric de Sitter
background.  Supports parallel execution with `julia -t8`.

### 12 -- Product Manifolds

Demonstrates the product manifold feature for `M = AdS_2 x S^2` (the
near-horizon geometry of Reissner-Nordstrom).  Verifies the additive scalar
curvature `R = R_1 + R_2`, the block-diagonal Riemann decomposition, and the
physically significant cross-scalar term in the Einstein tensor
`G_{ij} = G_1_{ij} - (1/2) R_2 g_1_{ij}`.

### 13 -- Six-Derivative Particle Spectrum

Computes the complete tree-level particle spectrum for general six-derivative
gravity in 4D, verified against Buoninfante et al. (arXiv:2012.11829) and
the PSALTer package (arXiv:2406.09500).  Extracts spin-2 and spin-0 form
factors and identifies the mass poles and ghost conditions from the Barnes-Rivers
spin-projected propagator.

### 14 -- Cubic Bueno-Cano Parameters

Computes the Bueno-Cano parameters `(a, b, c, e)` for all six cubic curvature
invariants using the parametric Riemann tensor method of Bueno and Cano
(arXiv:1607.06463).  This is a pure linear-algebra computation (no TensorGR
AST) that serves as independent ground truth for examples 11 and 15.

### 15 -- Perturbation Spectrum Cross-Check

Cross-checks the Bueno-Cano parameters by computing `delta^2(sqrt(g) L)` for
each cubic invariant via the TensorGR perturbation engine on a maximally
symmetric background, spin-projecting the momentum-space kernel, and comparing
the resulting mass poles against the analytic predictions from example 14.

### 26 -- Six-Derivative Spectrum Showcase

Demonstrates three independent computation paths for the same physical
quantity -- the particle spectrum of six-derivative gravity: Path A uses
Barnes-Rivers spin projection in covariant momentum space, Path B uses SVT
quadratic forms in Bardeen gauge, and Path C uses the Bueno-Cano parametric
spectrum on de Sitter.  Cross-checks verify agreement between all three paths.

---

## Advanced Topics (06--07)

### 06 -- Exterior Calculus

Introduces differential forms and the exterior calculus engine: form
registration, antisymmetry, wedge products, the exterior derivative with
`d^2 = 0`, interior products, Hodge duality, and Cartan's magic formula
`L_v = i_v d + d i_v`.  Demonstrates the formalism needed for gauge theory
and topological invariants.

### 07 -- Gauge Theory with VBundles

Defines an SU(2) gauge vector bundle with `define_vbundle!` and constructs
field strength tensors carrying mixed spacetime (Tangent) and internal (SU2)
indices.  Verifies that index contraction respects bundle boundaries --
spacetime indices contract only with spacetime indices, and gauge indices
only with gauge indices.

---

## Geodesics (27)

### 27 -- Geodesic Orbits in Schwarzschild

Builds the Schwarzschild metric as a numerical function and sets up a
`GeodesicEquation` for numerical integration.  Integrates a circular orbit at
`r = 10M` for five orbital periods and checks energy conservation along the
trajectory.  Uses DifferentialEquations.jl for integration (falls back to a
simple Euler integrator if unavailable).  Symbolics.jl is optional for symbolic
Christoffel display.

---

## Course Verification Suite

The `examples/course_verification/` directory contains seven scripts that
follow a standard GR lecture course (modelled on Carroll's *Spacetime and
Geometry* and Wald's *General Relativity*).  Each script verifies a
lecture's worth of identities using TensorGR.

| File | Lecture Topic | Key identities verified |
|------|---------------|------------------------|
| `lec07_derivative_operators.jl` | Derivative operators | Christoffel formula, metric compatibility, connection difference tensor |
| `lec09_10_curvature.jl` | Curvature | Riemann from commutator, Riemann symmetries, Bianchi identity, Einstein tensor |
| `lec12_lie_derivatives.jl` | Lie derivatives | Lie bracket, Killing equation, Lie derivative of metric/vector/covector |
| `lec13_einstein_equations.jl` | Einstein equations | Einstein tensor structure, trace identity, dust stress-energy, linearized inverse metric |
| `lec14_gravitational_radiation.jl` | Gravitational radiation | Linearized Riemann/Ricci, trace-reversed perturbation, gauge transformation |
| `lec17_friedmann.jl` | Friedmann equations | FLRW Christoffels, Ricci, scalar curvature, Friedmann equations (Symbolics.jl) |
| `lec19_schwarzschild.jl` | Schwarzschild solution | Schwarzschild Christoffels, vacuum Ricci, Kretschmann `48M^2/r^6` (Symbolics.jl) |
| `lec21_22_geodesics.jl` | Geodesics | Killing vector conservation, effective potential, ISCO at `r=6M`, photon sphere at `r=3M` |

---

## Wald Verification Suite

The `examples/wald_verification/` directory contains eight scripts that
systematically verify identities from Wald's *General Relativity* (1984).

| File | Topic | What is verified |
|------|-------|------------------|
| `01_covariant_derivative_identities.jl` | Covariant derivatives | Riemann symmetries, Bianchi identities |
| `02_lie_derivatives.jl` | Lie derivatives | Lie derivative identities (Wald Appendix C.1) |
| `03_linearised_gravity.jl` | Linearized gravity | First-order perturbation of curvature tensors (Wald Section 7.5) |
| `04_schwarzschild.jl` | Schwarzschild | Vacuum Ricci, Kretschmann scalar via symbolic components |
| `06_flrw.jl` | FLRW cosmology | Friedmann equations via symbolic FLRW metric (Wald Appendix F) |
| `07_curvature_decomposition.jl` | Curvature decomposition | Weyl decomposition roundtrips, trace properties (Wald Section 3.2) |
| `08_exterior_calculus.jl` | Exterior calculus | `d^2 = 0`, Maxwell-Bianchi identity `dF = 0` (Wald Appendix B) |

---

## Debug and Development Scripts (16--25)

Examples 16 through 25 are internal development and debugging scripts created
during the implementation of the Fourier-space perturbation pipeline.  They
are not intended as user-facing examples and could be moved to a `dev/` or
`test/scripts/` directory in a future cleanup.

| # | File | Purpose | Status |
|---|------|---------|--------|
| 16 | `16_diagnostic_flat_pipeline.jl` | Diagnose Fourier sign conventions for bilinear actions | Debug script; documents the `i^N (-1)^{n_R}` sign correction |
| 17 | `17_pipeline_isolation_test.jl` | Isolate pipeline bug: position-space FP Lagrangian through Fourier and spin projection | Debug script; compares pipeline output against direct kernel builder |
| 18 | `18_expand_derivs_fix.jl` | Test fix for `expand_derivatives` before `to_fourier` | Bug-fix verification; the underlying issue is now resolved |
| 19 | `19_kernel_debug.jl` | Debug `extract_kernel` missing terms | Debug script; minimal reproducer for kernel extraction |
| 20 | `20_kernel_comparison.jl` | Compare perturbation-engine kernel against direct FP kernel | Debug script; side-by-side term comparison |
| 21 | `21_total_deriv_test.jl` | Test whether total derivatives vanish under spin projection | Pipeline validation; verifies total-derivative handling |
| 22 | `22_swap_symmetry_test.jl` | Test left-right h swap symmetry of `spin_project` | Pipeline validation; verifies projector symmetry `P^{mu nu,rho sigma} = P^{rho sigma,mu nu}` |
| 23 | `23_fourier_nested_test.jl` | Diagnose `to_fourier` on nested `TDeriv` and build analytic comparison | Debug script from Session 18; nested derivative Fourier transform |
| 24 | `24_simplify_vs_fourier.jl` | Diagnose why `simplify` changes Fourier transform results | Debug script; investigates simplify/Fourier non-commutativity |
| 25 | `25_minimal_fourier_bug.jl` | Minimal reproducer: does Fourier commute with simplify? | Bug reproducer; tests Fourier/simplify ordering |

**Recommendation:** These scripts served their purpose during development and
the bugs they target have been resolved.  They could be relocated to a `dev/`
directory to keep the `examples/` directory focused on user-facing
demonstrations.  Alternatively, the validation scripts (21, 22) could be
converted into proper unit tests.
