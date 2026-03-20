# Spherical Harmonics

Abstract spherical harmonic decomposition framework for scalar, vector, and symmetric tensor fields on the 2-sphere S2. Provides harmonic types as TensorExpr subtypes with eigenvalue relations, Clebsch-Gordan coefficients and Wigner 3-j symbols, orthogonality inner products, angular integrals (Gaunt integrals), Laplacian eigenmodes, and decomposition of fields into harmonic mode expansions.

## Scalar Harmonics

Scalar spherical harmonics Y\_lm as abstract TensorExpr nodes with eigenvalue, conjugation, and inner product operations.

```@docs
ScalarHarmonic
conjugate
angular_laplacian
inner_product
```

## Clebsch-Gordan Coefficients

Wigner 3-j symbols and Clebsch-Gordan coefficients for coupling angular momenta, plus the linearization of products of scalar harmonics.

```@docs
wigner3j
clebsch_gordan
harmonic_product
```

## Vector Harmonics

Even (gradient-type) and odd (curl-type) vector spherical harmonics with parity, divergence/curl eigenvalues, and norms.

```@docs
EvenVectorHarmonic
OddVectorHarmonic
divergence_eigenvalue
curl_eigenvalue
norm_squared
```

## Tensor Harmonics

Even (Y and Z type) and odd tensor spherical harmonics for symmetric rank-2 fields, with trace, conjugation, inner product, and norm operations.

```@docs
EvenTensorHarmonicY
EvenTensorHarmonicZ
OddTensorHarmonic
trace
tensor_inner_product
```

## Orthogonality

Inner product and orthogonality relations between vector and tensor harmonics.

```@docs
vector_inner_product
```

## Angular Integrals

Gaunt integrals and angular selection rules for products of spherical harmonics.

```@docs
gaunt_integral
vector_gaunt
tensor_gaunt
angular_selection_rule
angular_integral
```

## Laplacian Eigenmodes

The Laplacian operator on S2 acting on harmonic types, with eigenvalue simplification.

```@docs
LaplacianS2
simplify_laplacian
laplacian_S2
```

## Decomposition

Decompose scalar, vector, and symmetric tensor fields into spherical harmonic mode expansions.

### Scalar Decomposition

```@docs
ScalarMode
HarmonicDecomposition
mode_count
get_mode
decompose_scalar
```

### Vector Decomposition

```@docs
VectorMode
VectorHarmonicDecomposition
decompose_vector
```

### Tensor Decomposition

```@docs
Parity
TensorMode
TensorHarmonicDecomposition
decompose_symmetric_tensor
```
