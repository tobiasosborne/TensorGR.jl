# Quadratic Action Analysis

Extract and manipulate the kinetic matrix of a quadratic Lagrangian. A quadratic
Lagrangian L(Phi) = Phi\_i M\_{ij}(k) Phi\_j defines a momentum-dependent matrix
whose inverse is the propagator.

## Quadratic Form

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

## Example

```julia
using TensorGR

# Build a 2-field quadratic form with momentum-dependent entries
entries = Dict(
    (:phi, :phi) => :(k^2),
    (:phi, :psi) => :(alpha * k^2),
    (:psi, :psi) => :(beta * k^2 + m^2),
)
qf = quadratic_form(entries, [:phi, :psi])

# Compute the propagator (inverse kinetic matrix)
prop = propagator(qf)

# Evaluate at a specific momentum point
vars = Dict(:k^2 => 1.0, :alpha => 0.5, :beta => 1.0, :m^2 => 0.1)
println(sym_eval(prop.matrix[1,1], vars))
```
