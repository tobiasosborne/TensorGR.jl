# Fermions & Clifford Algebra

Dirac gamma matrices, Clifford algebra, trace identities, Fierz rearrangement, and charge conjugation for spinor fields in curved spacetime. Gamma matrices are represented as AST nodes with one spacetime index and implicit spinor indices, supporting abstract symbolic manipulation.

## Gamma Matrices

The Dirac gamma matrix gamma^a satisfies the Clifford algebra {gamma^a, gamma^b} = 2 g^{ab}. Includes the chirality matrix gamma^5, Feynman slash notation, and basic algebraic identities.

```julia
gamma_a = GammaMatrix(down(:a))
gamma_b = GammaMatrix(up(:b))
g5 = Gamma5()
v_slash = slash(Tensor(:p, [down(:a)]))
```

```@docs
GammaMatrix
Gamma5
gamma5
clifford_relation
gamma_trace
gamma5_trace
gamma5_anticommutator
gamma5_squared
slash
```

## Dirac Traces

Evaluate traces of products of gamma matrices using the standard recursive identities. Supports arbitrary-length chains via the Clifford algebra recursion.

```julia
# Tr(gamma^a gamma^b) = 4 g^{ab}
result = gamma_chain_trace([GammaMatrix(up(:a)), GammaMatrix(up(:b))]; metric=:g)

# Tr(gamma^a gamma^b gamma^c gamma^d) = 4(g^{ab}g^{cd} - g^{ac}g^{bd} + g^{ad}g^{bc})
result = trace_identity_4(up(:a), up(:b), up(:c), up(:d); metric=:g)
```

```@docs
gamma_chain_trace
trace_identity_2
trace_identity_4
```

## Fierz Identities

The Fierz rearrangement identity expresses products of spinor bilinears in a rearranged form using the completeness of the 16-dimensional Clifford algebra basis {I, gamma^a, sigma^{ab}, gamma^a gamma^5, gamma^5}.

```@docs
CliffordBasis
fierz_matrix
fierz_coefficient
fierz_identity_check
```

## Charge Conjugation

The charge conjugation matrix C satisfies C gamma^a C^{-1} = -(gamma^a)^T. It is antisymmetric and unitary, and defines the Majorana condition for self-conjugate spinors.

```@docs
ChargeConjugation
charge_conjugation_properties
majorana_condition
```
