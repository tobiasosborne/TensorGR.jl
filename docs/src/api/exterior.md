# Exterior Calculus

This module provides differential forms and exterior algebra operations: form definition, wedge product, exterior derivative, interior product, Hodge dual, codifferential, Cartan's magic formula, and the Cartan structure equations for connection and curvature forms. Forms are represented as fully antisymmetric `TensorExpr` objects, so all standard simplification tools (canonicalize, simplify, etc.) apply.

## Differential Forms

A k-form is a completely antisymmetric rank-(0,k) tensor. `define_form!` registers a tensor with full antisymmetry in all index slots.

```julia
reg = TensorRegistry()
# ... register manifold :M4 ...

# Define a 1-form (e.g., electromagnetic potential)
define_form!(reg, :A; manifold=:M4, degree=1)

# Define a 2-form (e.g., field strength)
define_form!(reg, :F; manifold=:M4, degree=2)

# Query the degree
form_degree(reg, :F)  # => 2
```

```@docs
define_form!
form_degree
```

## Wedge Product

The wedge product of a p-form and a q-form produces a (p+q)-form. The combinatorial prefactor `(p+q)!/(p!q!)` is included automatically; antisymmetry is enforced by the form's symmetry properties during canonicalization.

```julia
A = Tensor(:A, [down(:a)])          # 1-form
B = Tensor(:B, [down(:b)])          # 1-form
AB = wedge(A, B, 1, 1)             # 2-form: A wedge B

# Wedge power: alpha^{wedge n} = alpha wedge ... wedge alpha (n times)
alpha = Tensor(:alpha, [down(:a)])
alpha3 = wedge_power(alpha, 1, 3)  # alpha wedge alpha wedge alpha
```

```@docs
wedge
wedge_power
```

## Exterior Derivative

The exterior derivative maps a p-form to a (p+1)-form. It is represented as a `TDeriv` wrapping the form expression.

```julia
A = Tensor(:A, [down(:a)])
# dA with derivative index b
dA = exterior_d(A, 1, down(:b))  # => TDeriv(down(:b), A)
```

```@docs
exterior_d
```

## Interior Product

The interior product (contraction) of a vector field with a form lowers the degree by one: for a vector `v` and a p-form `alpha`, the result is a (p-1)-form.

```julia
v = Tensor(:v, [up(:a)])
omega = Tensor(:omega, [down(:a), down(:b)])  # 2-form

# iota_v(omega): contract v^a with first index of omega
iv_omega = interior_product(v, omega)
```

```@docs
interior_product
```

## Hodge Dual

The Hodge star operator maps a p-form to a (d-p)-form using the Levi-Civita tensor. Requires the epsilon tensor to be defined (automatic with `define_metric!`).

```julia
F = Tensor(:F, [down(:a), down(:b)])  # 2-form in 4D

# Hodge dual: *F is a 2-form (4 - 2 = 2)
star_F = hodge_dual(F, :epsilon_g, 2, 4)
```

```@docs
hodge_dual
```

## Codifferential

The codifferential (adjoint of the exterior derivative) maps a p-form to a (p-1)-form. Defined as `delta = (-1)^{d(p+1)+1} * d *` where `*` is the Hodge dual.

```@docs
codifferential
```

## Cartan's Magic Formula

Cartan's formula expresses the Lie derivative of a form in terms of the exterior derivative and interior product: `L_v omega = d(iota_v omega) + iota_v(d omega)`.

```julia
v = Tensor(:v, [up(:a)])
omega = Tensor(:omega, [down(:a), down(:b)])

# Cartan's formula: L_v omega = d(iota_v omega) + iota_v(d omega)
lie_omega = cartan_lie_d(v, omega, 2, down(:c))
```

```@docs
cartan_lie_d
```

## Connection and Curvature Forms

Build the Cartan structure equations relating connection 1-forms, curvature 2-forms, torsion, and the coframe. These are the language of the vierbein/tetrad formalism.

### Connection 1-Form

The connection 1-form `omega^a_b` is built from the Christoffel symbols (or more generally, from any connection coefficients).

```julia
# omega^a_b = Gamma^a_{c b} dx^c
omega = connection_form(:Gamma, up(:a), down(:b), down(:c))
```

```@docs
connection_form
```

### Curvature 2-Form

The curvature 2-form `Omega^a_b` is computed via the second Cartan structure equation.

```julia
# Omega^a_b = d(omega^a_b) + omega^a_c wedge omega^c_b
Omega = curvature_form(:Gamma, up(:a), down(:b), down(:c), down(:d))
```

```@docs
curvature_form
```

### Cartan Structure Equations

The first structure equation relates torsion to the coframe and connection. The second structure equation relates curvature to the connection.

```julia
# First structure equation: T^a = d(theta^a) + omega^a_b wedge theta^b
T = cartan_first_structure(:T, :Gamma, :theta, up(:a), down(:b), down(:c))

# Second structure equation: Omega^a_b = d(omega^a_b) + omega^a_c wedge omega^c_b
Omega = cartan_second_structure(:Riem, :Gamma, up(:a), down(:b), down(:c), down(:d))
```

```@docs
cartan_first_structure
cartan_second_structure
```
