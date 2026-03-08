# SVT Decomposition & Projectors

Scalar-Vector-Tensor decomposition of symmetric rank-2 perturbations,
Fourier-space transforms, and spin-projection operators.

## SVT Fields

The metric perturbation h\_{ab} decomposes into 7 SVT fields on a flat
background:

- **Scalars**: Phi (lapse), B (shift longitudinal), psi (spatial trace), E (spatial longitudinal)
- **Vectors**: S\_i (shift transverse), F\_i (spatial transverse)
- **Tensor**: hTT\_{ij} (transverse-traceless)

```@docs
SVTFields
svt_substitute
```

## Fourier Transform

Replace partial derivatives with momentum factors for momentum-space analysis.
Each derivative d\_a becomes a momentum tensor k\_a. Sign conventions are
configurable via `FourierConvention`.

```@docs
to_fourier
FourierConvention
```

## Spatial Projectors

Projection operators in Fourier space for isolating transverse and
transverse-traceless components of spatial tensors.

The transverse projector removes the longitudinal part:

    P^T\_{ij}(k) = delta\_{ij} - k\_i k\_j / k^2

The TT projector isolates the transverse-traceless part of a symmetric
rank-2 spatial tensor:

    Pi^TT\_{ijkl} = 1/2 (P^T\_{ik} P^T\_{jl} + P^T\_{il} P^T\_{jk} - P^T\_{ij} P^T\_{kl})

```@docs
transverse_projector
tt_projector
```

## Barnes-Rivers Spin Projectors

The 4D Barnes-Rivers projectors decompose a symmetric rank-2 field h\_{ab}
into irreducible spin sectors under the Lorentz group. They are built from
the transverse projector theta\_{ab} and the longitudinal projector
omega\_{ab}:

    theta\_{ab} = eta\_{ab} - k\_a k\_b / k^2
    omega\_{ab} = k\_a k\_b / k^2

The six Barnes-Rivers operators satisfy completeness and orthogonality on
the space of symmetric rank-2 tensors.

```@docs
theta_projector
omega_projector
spin2_projector
spin1_projector
spin0s_projector
spin0w_projector
```

## Transfer Operators

Transfer operators mix the two spin-0 sectors (scalar and longitudinal).
They satisfy T^{sw} T^{ws} = (1/(d-1)) P^{0-s}.

```@docs
transfer_sw
transfer_ws
```
