# Hamiltonian & ADM Formalism

The Arnowitt-Deser-Misner (ADM) decomposition of spacetime into spatial slices and time evolution, providing the Hamiltonian formulation of general relativity. Includes lapse, shift, spatial metric, extrinsic curvature, conjugate momenta, constraint expressions, canonical Poisson brackets, and constraint algebra classification.

## ADM Decomposition

Decompose the spacetime metric into the ADM variables: lapse function N, shift vector N^i, and spatial metric gamma_{ij}. Registers all associated tensors including extrinsic curvature K_{ij}, its trace K, and the conjugate momentum pi^{ij}.

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    adm = define_adm!(reg; manifold=:M4)
end
```

```@docs
ADMDecomposition
define_adm!
```

## Constraints

The Hamiltonian and momentum constraints that must vanish on physical states.

```julia
H = hamiltonian_constraint(adm; registry=reg)    # pi^{ij}pi_{ij} - (1/2)pi^2 - R^(3)
Hi = momentum_constraint(adm; registry=reg)      # -2 D_j pi^j_i
```

```@docs
hamiltonian_constraint
momentum_constraint
```

## Poisson Brackets

Canonical Poisson bracket structure for the ADM phase space variables.

```@docs
CanonicalPair
adm_canonical_pair
fundamental_bracket
PoissonBracketResult
constraint_algebra_type
```
