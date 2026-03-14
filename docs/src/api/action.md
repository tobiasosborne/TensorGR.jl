# Quadratic Action Analysis

Extract and manipulate the kinetic structure of quadratic Lagrangians: quadratic form matrices, kinetic kernel extraction from position-space bilinear actions, Barnes-Rivers spin projection, momentum-space kernel builders for standard gravity theories, Bueno-Cano parametrization for the particle spectrum of higher-derivative gravity on de Sitter, and SVT-decomposed quadratic forms.

## Quadratic Form

A quadratic Lagrangian `L(Phi) = Phi_i M_{ij}(k) Phi_j` defines a momentum-dependent matrix whose inverse is the propagator.

```julia
# Build a 2-field quadratic form with momentum-dependent entries
entries = Dict(
    (:phi, :phi) => :(k^2),
    (:phi, :psi) => :(alpha * k^2),
    (:psi, :psi) => :(beta * k^2 + m^2),
)
qf = quadratic_form(entries, [:phi, :psi])
```

```@docs
QuadraticForm
quadratic_form
```

## Extraction from Lagrangian

Given a tensor expression that is quadratic in a set of fields,
`extract_quadratic_form` Fourier-transforms derivatives to momenta,
expands the expression, identifies field bilinears, contracts momentum
indices, and collects coefficients into the kinetic matrix.

Momentum contractions are resolved symbolically:
- k\_{i} k\_{i} becomes :k^2
- k\_{0} k\_{0} becomes :omega^2
- k\_{a} k^{a} (abstract) becomes :p^2

```@docs
extract_quadratic_form
```

## Propagator & Determinant

Invert the kinetic matrix to obtain the propagator, or compute the
determinant to locate ghost poles.

```@docs
propagator
determinant
```

## Symbolic Matrix Algebra

Small-matrix symbolic operations (up to 3x3) for determinants and inverses,
operating on Julia `Number` and `Expr` trees. When Symbolics.jl is loaded,
these dispatch through the CAS for simplification.

```@docs
sym_det
sym_inv
sym_eval
```

## Kinetic Kernel Extraction

The `KineticKernel` type decomposes a bilinear action into per-term data: each term is a coefficient times two field factors with their index lists. This enables spin projection via Barnes-Rivers operators without requiring a single 4-index kernel tensor.

### From Fourier-Space Expressions

```julia
# Extract from a Fourier-transformed bilinear expression
K = extract_kernel(fourier_delta2S, :h)
```

```@docs
KineticKernel
extract_kernel
```

### From Position-Space Expressions

The `extract_kernel_direct` function handles the full two-momentum physics of quadratic forms under an integral, correctly assigning opposite momenta to the two field copies.

```julia
# Extract kernel directly from position-space bilinear:
# handles derivative → momentum conversion with correct phase
d1R_ab = simplify(delta_ricci(mp, down(:a), down(:b), 1); registry=reg)
d1R    = simplify(delta_ricci_scalar(mp, 1); registry=reg)
EH_bilinear = h_up * d1R_ab - (1//2) * trh * d1R
K = extract_kernel_direct(EH_bilinear, :h; registry=reg)
```

```@docs
extract_kernel_direct
```

## Spin Projection

Project a kinetic kernel onto spin sectors using Barnes-Rivers projectors. Returns the scalar form factor (function of k^2) for each spin sector.

```julia
# Project onto spin-2 sector
f2 = spin_project(K, :spin2; registry=reg)

# Available sectors: :spin2, :spin1, :spin0s, :spin0w
f0s = spin_project(K, :spin0s; registry=reg)
```

```@docs
spin_project
contract_momenta
```

## Momentum-Space Kernel Builders

Direct construction of bilinear kernels in Fourier space from known linearized curvature formulas, bypassing the position-space perturbation engine. These serve as verified ground truth for cross-checking.

### Fierz-Pauli (Einstein-Hilbert)

```julia
K_FP = build_FP_momentum_kernel(reg)
```

```@docs
build_FP_momentum_kernel
```

### R-squared

```julia
K_R2 = build_R2_momentum_kernel(reg)
```

```@docs
build_R2_momentum_kernel
```

### Ricci-squared

```julia
K_Ric2 = build_Ric2_momentum_kernel(reg)
```

```@docs
build_Ric2_momentum_kernel
```

### Combined 6-Derivative Kernel

Build the combined kinetic kernel for the action `S = integral d^4x sqrt(g) [kappa R + alpha_1 R^2 + alpha_2 Ric^2 + beta_1 R Box R + beta_2 Ric Box Ric]` on a flat background.

```julia
K = build_6deriv_flat_kernel(reg; kappa=1, alpha_1=0, alpha_2=0, beta_1=0, beta_2=0)
```

```@docs
build_6deriv_flat_kernel
```

### Kernel Manipulation

```@docs
scale_kernel
combine_kernels
```

## Spin Projection Evaluation

Evaluate fully-contracted spin projection results at numeric momentum values.

```julia
# Compute spin projections for 6-derivative gravity
projections = flat_6deriv_spin_projections(reg; kappa=1, alpha_1=0, alpha_2=1//10)

# Evaluate at k^2 = 1
f2_val = _eval_spin_scalar(projections.spin2, 1.0)
```

```@docs
flat_6deriv_spin_projections
_eval_spin_scalar
```

## Bueno-Cano Parametrization

The Bueno-Cano parameters `(a, b, c, e)` characterize the linearized field equations of a gravity theory on a maximally symmetric background. From these four numbers, the full particle spectrum (effective Newton constant, massive spin-2 mass, spin-0 mass) follows algebraically.

### Parameter Type

```@docs
BuenoCanoParams
```

### Parameters for Standard Lagrangian Terms

Each function returns the `BuenoCanoParams(a, b, c, e)` contribution from a single Lagrangian term on a maximally symmetric background with `Ric = Lambda g`.

```julia
# Einstein-Hilbert: kappa R
bc_EH(1, Lambda)

# 4-derivative terms
bc_R2(alpha_1, Lambda)      # alpha_1 R^2
bc_RicSq(alpha_2, Lambda)   # alpha_2 Ric_{ab} Ric^{ab}

# 6-derivative cubic invariants
bc_R3(gamma, Lambda)        # gamma R^3
bc_RRicSq(gamma, Lambda)    # gamma R Ric^2
bc_Ric3(gamma, Lambda)      # gamma Ric^3
bc_RRiem2(gamma, Lambda)    # gamma R Riem^2
bc_RicRiem2(gamma, Lambda)  # gamma Ric Riem^2
bc_Riem3(gamma, Lambda)     # gamma Riem^3
```

```@docs
bc_EH
bc_R2
bc_RicSq
bc_R3
bc_RRicSq
bc_Ric3
bc_RRiem2
bc_RicRiem2
bc_Riem3
```

### Spectrum and Form Factors

```julia
# Full spectrum of 6-derivative gravity on de Sitter
spec = dS_spectrum_6deriv(; kappa=1, alpha_1=0, alpha_2=0,
    gamma_1=0, gamma_2=0, gamma_3=0, gamma_4=0, gamma_5=0, gamma_6=0, Lambda=0.1)
spec.kappa_eff_inv    # inverse effective Newton constant
spec.m2_graviton     # massive spin-2 mass squared
spec.m2_scalar       # spin-0 mass squared
spec.flat_f2         # flat-space spin-2 form factor coefficients
spec.flat_f0         # flat-space spin-0 form factor coefficients

# Cross-check: predict form factors from BC parameters
ff = bc_to_form_factors(spec.params, 1.0, 0.1)
ff.f_spin2
ff.f_spin0s
```

```@docs
dS_spectrum_6deriv
bc_to_form_factors
```

## SVT-Decomposed Quadratic Forms

Build the scalar-vector-tensor decomposed kinetic matrix for higher-derivative gravity linearized on a flat background in Bardeen gauge. This is "Path B" -- direct computation from linearized curvature in SVT variables, complementing the spin-projection "Path A".

```julia
# Build SVT quadratic forms for 6-derivative gravity
qfs = svt_quadratic_forms_6deriv(; kappa=1, alpha_1=0, alpha_2=0, beta_1=0, beta_2=0)
qfs.tensor   # QuadraticForm for hTT sector
qfs.scalar   # QuadraticForm for (Phi, psi) sector
qfs.vector_vanishes  # true (vector sector vanishes in Bardeen gauge)
```

```@docs
svt_quadratic_forms_6deriv
```

## CAS Integration

When Symbolics.jl is loaded, additional functions are available for symbolic manipulation of quadratic forms. See [CAS Integration](@ref) in the Advanced Features page for details on `simplify_scalar`, `simplify_quadratic_form`, `symbolic_quadratic_form`, and `to_fourier_symbolic`.

## Example: Complete 4-Derivative Analysis

```julia
using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4, signature=lorentzian(4))

    # Build the combined EH + R^2 + Ric^2 kernel
    K = build_6deriv_flat_kernel(reg; kappa=1, alpha_1=1//10, alpha_2=1//20)

    # Spin-project to get form factors
    f2  = spin_project(K, :spin2; registry=reg)
    f0s = spin_project(K, :spin0s; registry=reg)

    # Evaluate at k^2 = 1
    println("f_2(1) = ", _eval_spin_scalar(f2, 1.0))
    println("f_0s(1) = ", _eval_spin_scalar(f0s, 1.0))

    # Cross-check via Bueno-Cano
    bc = bc_EH(1, 0) + bc_R2(1//10, 0) + bc_RicSq(1//20, 0)
    ff = bc_to_form_factors(bc, 1.0, 0.0)
    println("BC prediction: f2 = ", ff.f_spin2, ", f0s = ", ff.f_spin0s)
end
```
