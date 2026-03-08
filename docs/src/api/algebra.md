# Algebra

The algebra module provides all operations for manipulating abstract tensor expressions: smart constructors that enforce normal forms, the full simplification pipeline, index utilities, symmetrization, ansatz construction, integration by parts, and linear equation solving.

## Smart Constructors

The normalized constructors `tproduct` and `tsum` enforce invariants on every construction: nested products/sums are flattened, scalar coefficients are absorbed, zeros are propagated, and identities are eliminated. All arithmetic operators (`*`, `+`, `-`) dispatch through these constructors.

```julia
# Products flatten and absorb scalars
tproduct(2//1, TensorExpr[tproduct(3//1, TensorExpr[T])]) # => 6//1 * T

# Sums flatten and remove zeros
tsum(TensorExpr[ZERO, R, ZERO])  # => R

# Operator overloads
expr = 2 * Ric - (1//2) * g * RicScalar
```

```@docs
tproduct
tsum
```

## Simplification Pipeline

The `simplify` function runs a fixed-point pipeline: expand products over sums, contract metrics and deltas, detect Riemann/Ricci traces, canonicalize index ordering via xperm.c, optionally commute covariant derivatives, collect like terms, and apply registered rewrite rules. Each step repeats until the expression stabilizes.

```julia
expr = g * Ric + Ric * g   # two terms that differ only in factor order
simplify(expr)               # => 2 * g * Ric  (after canonicalize + collect)

# With covariant derivative commutation
simplify(expr; commute_covds_name=:nabla)

# Parallel simplification for large sums (>20 terms)
simplify(expr; parallel=true)
```

```@docs
simplify
expand_products
collect_terms
contract_metrics
canonicalize
expand_derivatives
```

## Index Utilities

Functions for inspecting, renaming, and generating tensor indices. These are the building blocks used by the simplification pipeline and by user code that constructs expressions programmatically.

```julia
T = Tensor(:T, [up(:a), down(:b), up(:c), down(:c)])
free_indices(T)     # [up(:a), down(:b)]
dummy_pairs(T)      # [(up(:c), down(:c))]
indices(T)          # [up(:a), down(:b), up(:c), down(:c)]
```

```@docs
indices
free_indices
dummy_pairs
rename_dummy
rename_dummies
fresh_index
ensure_no_dummy_clash
index_sort
same_dummies
split_index
```

## AST Traversal

Generic tree-walking utilities for transforming, inspecting, and substituting within tensor expression trees.

```julia
# Bottom-up transformation
walk(expr) do node
    node isa Tensor && node.name == :Ric ? TScalar(0//1) : node
end

# Direct substitution
substitute(expr, old_tensor => new_tensor)

# Get immediate sub-expressions
children(product_expr)   # => product_expr.factors
```

```@docs
walk
substitute
children
```

## Expression Properties

Query structural properties of expressions: derivative depth, constancy, and derivative ordering.

```@docs
derivative_order
is_constant
is_sorted_covds
dagger
```

## Trace & Decomposition

Contract pairs of free indices (abstract trace) and decompose tensors into trace-free and trace parts.

```julia
# Trace over indices a, b
traced = abstract_trace(T_ab, :a, :b; metric=:g)

# Trace-free decomposition: T^TF_{ab} = T_{ab} - (1/d) g_{ab} T^c_c
tf = make_traceless(T_ab, :g, :a, :b; dim=4)
```

```@docs
abstract_trace
make_traceless
```

## Symmetrization

Symmetrize or antisymmetrize tensor expressions over specified index sets. The `impose_symmetry` function dispatches on symmetry type.

```julia
# T_{(ab)} = (1/2)(T_{ab} + T_{ba})
sym = symmetrize(T, [:a, :b])

# T_{[ab]} = (1/2)(T_{ab} - T_{ba})
asym = antisymmetrize(T, [:a, :b])

# Project onto a symmetry class
proj = impose_symmetry(T, :symmetric, [:a, :b])
```

```@docs
symmetrize
antisymmetrize
impose_symmetry
```

## Young Tableaux

Decompose tensors into irreducible representations of the symmetric group using Young symmetrizers and projectors.

```julia
# Standard Young tableau with shape [2,1]
yt = YoungTableau([[1,2],[3]])
young_shape(yt)  # [2, 1]

# Apply Young symmetrizer (symmetrize rows, antisymmetrize columns)
result = young_symmetrize(T_abc, yt, [:a, :b, :c])

# Apply Young projector with normalization
proj = young_project(T_abc, yt, [:a, :b, :c])
```

```@docs
YoungTableau
young_shape
young_symmetrize
young_project
```

## Tensor Collection

Group terms in a sum by their tensor structure and combine scalar coefficients. Also provides utilities to strip scalars or tensors from expressions.

```@docs
collect_tensors
remove_constants
remove_tensors
index_collect
```

## Ansatz Construction

Build the most general tensor expression with a given index structure by enumerating all independent contractions and forming linear combinations with symbolic coefficients.

```julia
# Linear combination with explicit coefficients
ansatz = make_ansatz([T1, T2, T3], [:c1, :c2, :c3])

# Enumerate all contractions of given tensors with specified free indices
contractions = all_contractions([Ric, g], [down(:a), down(:b)])

# Full ansatz: general linear combination of all contractions
full = contraction_ansatz([Ric, g], [down(:a), down(:b)])
```

```@docs
make_ansatz
all_contractions
contraction_ansatz
```

## Integration by Parts

Move derivatives off a specified field in product expressions, assuming an implicit integral with vanishing boundary terms.

```julia
# Move derivatives off phi: integral (d_a phi) psi -> - integral phi (d_a psi)
result = ibp(expr, :phi)

# Explicit product-level IBP (handles multi-derivative chains)
result = ibp_product(product_expr, :phi)
```

```@docs
ibp
ibp_product
```

## Solving Tensor Equations

Solve linear tensor equations for unknown tensors, returning rewrite rules. Supports systems of equations and optional trace decomposition.

```julia
# Solve G_{ab} + Lambda g_{ab} = 0 for Ric
rules = solve_tensors(equation, [:Ric])

# Solve a system of equations
rules = solve_tensors([eq1, eq2], [:T, :S]; take_traces=true)
```

```@docs
solve_tensors
```
