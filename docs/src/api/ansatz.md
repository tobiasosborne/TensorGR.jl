# Tensor Ansatz Construction

Build the most general tensor expression with a given index structure by
enumerating all independent contractions and forming linear combinations
with symbolic coefficients. This is useful for constructing ans\"atze for
unknown tensors (e.g., the most general symmetric rank-2 tensor built
from the metric and Ricci tensor) and for solving tensor identities.

## Linear Combinations

Create a linear combination of tensor expressions with explicit or
auto-generated symbolic coefficients.

```julia
R = Tensor(:Ric, [down(:a), down(:b)])
g = Tensor(:g, [down(:a), down(:b)])

# With explicit coefficient names
ansatz = make_ansatz(TensorExpr[R, g], [:alpha, :beta])
# alpha * Ric_{ab} + beta * g_{ab}

# With auto-generated coefficients c1, c2, ...
ansatz = make_ansatz(TensorExpr[R, g])
# c1 * Ric_{ab} + c2 * g_{ab}
```

```@docs
make_ansatz
```

## Contraction Enumeration

Given a set of tensors and desired free indices, enumerate all independent
ways to contract the tensors' indices into the target free index structure.
The algorithm generates all perfect matchings of contractible slots,
assigns dummy names, canonicalizes via xperm.c, and deduplicates
(including sign-aware deduplication for antisymmetric tensors).

```julia
# All rank-2 symmetric contractions from Ricci and metric
Ric = Tensor(:Ric, [down(:a), down(:b)])
g = Tensor(:g, [down(:a), down(:b)])
contractions = all_contractions(Tensor[Ric, g], [down(:a), down(:b)])
# Returns independent contractions like Ric_{ab}, g_{ab} * RicScalar, etc.
```

```@docs
all_contractions
```

## Full Ansatz Construction

Combine contraction enumeration with linear combination to produce the
most general expression in one call.

```julia
# Most general symmetric rank-2 tensor built from Ricci and metric
full = contraction_ansatz(
    Tensor[Tensor(:Ric, [down(:a), down(:b)]),
           Tensor(:g, [down(:a), down(:b)])],
    [down(:a), down(:b)])
# c1 * Ric_{ab} + c2 * g_{ab} * RicScalar + ...
```

This is particularly useful in conjunction with `solve_tensors` for
determining coefficients from symmetry or dynamical constraints.

```@docs
contraction_ansatz
```

## Smooth Maps (Pullback & Pushforward)

Define smooth maps between manifolds and construct pullback/pushforward
operations using explicit Jacobian contractions.

### Defining a Map

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M dim=3 metric=gM
    @manifold N dim=4 metric=gN

    # Define phi: M -> N with Jacobian dphi^i_a
    define_mapping!(reg, :phi; domain=:M, codomain=:N)

    # With inverse Jacobian for pushforward
    define_mapping!(reg, :psi; domain=:M, codomain=:N,
                    inv_jacobian_name=:dpsi_inv)
end
```

### Pullback

The pullback contracts covariant (Down) indices with the Jacobian:

```julia
g = Tensor(:gN, [down(:i), down(:j)])
pb = pullback(g, :phi)
# dphi^{i}_{a} dphi^{j}_{b} gN_{ij}  (induced metric on M)

# Convenience for pullback metric
gamma = pullback_metric(:phi, :gN)
```

### Pushforward

The pushforward contracts contravariant (Up) indices with the inverse
Jacobian (requires `inv_jacobian_name` at definition time):

```julia
V = Tensor(:V, [up(:a)])
pf = pushforward(V, :psi)
# dpsi_inv^{i}_{a} V^{a}
```

```@docs
MappingProperties
define_mapping!
pullback
pushforward
pullback_metric
```

## Product Manifolds

Define direct product manifolds M = M1 x M2 with block-diagonal metrics
and automatic curvature decomposition.

### Defining a Product

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M1 dim=2 metric=g1 indices=[a,b,c,d]
    @manifold S2 dim=2 metric=g2 indices=[i,j,k,l]

    define_product_manifold!(reg, :M; factors=[:M1, :S2])
end
```

### Curvature Decomposition

For a direct product, curvature splits by factor with no cross terms:

```julia
# Scalar curvature: R = R_1 + R_2
R = product_scalar_curvature(:M)

# Ricci in factor sector (mixed components vanish)
Ric1 = product_ricci(:M, :M1)

# Einstein with cross-scalar contribution:
# G1_{ab} = Ein_g1_{ab} - (1/2) RicScalar_g2 * g1_{ab}
G1 = product_einstein(:M, :M1)

# All factor Einstein equations at once
eqs = product_einstein_equations(:M)
```

```@docs
ProductManifoldProperties
define_product_manifold!
product_metric
product_scalar_curvature
product_ricci
product_riemann
product_einstein
product_einstein_equations
```
