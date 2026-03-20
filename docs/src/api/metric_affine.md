# Metric-Affine Geometry

General affine connection framework for metric-affine gravity (MAG), where the connection is independent of the metric. Includes torsion and non-metricity tensors with their irreducible decompositions, the distortion tensor (contortion + disformation), metric-affine curvature tensors, and the Brauer algebra 11-piece irreducible decomposition of the curvature under GL(d,R).

## Connection

A general affine connection with no assumed lower-index symmetry, decomposing into the Levi-Civita part plus a distortion tensor.

```@docs
AffineConnection
define_affine_connection!
is_metric_compatible
is_torsion_free
set_metric_compatible!
set_torsion_free!
```

## Torsion

The torsion tensor and its irreducible decomposition into vector, axial, and tensor parts.

```@docs
TorsionDecomposition
decompose_torsion!
torsion_vector_expr
contortion_expr
```

## Nonmetricity

The non-metricity tensor and its irreducible decomposition into Weyl vector, second trace, and traceless parts.

```@docs
weyl_vector_expr
second_trace_expr
NonmetricityDecomposition
decompose_nonmetricity!
```

## Distortion Tensor

The distortion tensor N = K + L decomposes into contortion (from torsion) and disformation (from non-metricity).

```@docs
DistortionDecomposition
decompose_distortion!
contortion_from_torsion
disformation_from_nonmetricity
```

## Curvature

The metric-affine Riemann tensor for a general affine connection, with its decomposition into Riemannian and distortion contributions. Unlike the Riemannian case, the Ricci tensor is asymmetric and pair symmetry does not hold.

```@docs
MAFieldStrength
define_ma_curvature!
ma_riemann_decomposition
```

## Brauer Decomposition

The 11-piece irreducible decomposition of the metric-affine Riemann tensor under GL(d,R), organized into symmetric and antisymmetric sectors with pair-exchange sub-decompositions.

```@docs
BrauerDecomposition
brauer_piece_names
brauer_piece_dimensions
brauer_symmetric_split
brauer_decompose
```
