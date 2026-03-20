# Bimetric Gravity

Hassan-Rosen bimetric theory with two dynamical metrics on the same manifold, interacting through a ghost-free potential built from elementary symmetric polynomials of the matrix square root. Provides registration of dual curvature tensor sets, the interaction potential, Cayley-Hamilton reduction rules, linearized perturbation theory about proportional backgrounds, mass eigenstates, and the Higuchi bound for massive spin-2 on de Sitter.

## Bimetric Setup

Register two independent metrics on the same manifold, each with its own Christoffel symbols, Riemann, Ricci, Einstein, and Weyl tensors. The naming convention uses metric name as suffix (e.g., `Riem_g`, `Ric_f`).

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    bs = define_bimetric!(reg, :g, :f; manifold=:M4)
end
```

```@docs
BimetricSetup
define_bimetric!
bimetric_field_equations
```

## Hassan-Rosen Potential

The interaction potential couples the two metrics through the matrix square root S = sqrt(g^{-1}f) and the elementary symmetric polynomials e_n(S).

```julia
params = HassanRosenParams(m_sq=:m2, beta0=0, beta1=1, beta2=1, beta3=1, beta4=0)
V = hassan_rosen_potential(bs, params; registry=reg)
```

```@docs
HassanRosenParams
hassan_rosen_potential
```

## Elementary Symmetric Polynomials

The elementary symmetric polynomials e_n(S) of the matrix square root, which appear as coefficients in the Hassan-Rosen potential.

```@docs
elementary_symmetric
```

## Matrix Square Root

Algebraic identities for the matrix square root S = sqrt(g^{-1}f), including the defining identity S^2 = g^{-1}f, the Cayley-Hamilton theorem for reduction of S powers, and the Sylvester equation for variations.

```@docs
sqrt_matrix_identity
cayley_hamilton_S
register_sqrt_rules!
sqrt_matrix_variation
```

## Linearization

Linearized perturbation theory for bimetric gravity about proportional backgrounds f = c^2 g. Diagonalization of the mass matrix yields a massless graviton and a massive Fierz-Pauli spin-2 mode.

```@docs
BimetricPerturbation
define_bimetric_perturbation!
fierz_pauli_mass_squared
bimetric_mass_matrix
bimetric_mass_eigenvalues
```

## Mass Eigenstates

The mass eigenstates are linear combinations of the two metric perturbations that diagonalize the linearized field equations.

```@docs
bimetric_mass_eigenstates
bimetric_inverse_transform
```

## Higuchi Bound

The Higuchi bound constrains the mass of a spin-2 field on de Sitter: below m^2 = 2 Lambda / 3, the helicity-0 mode becomes a ghost.

```@docs
higuchi_bound
higuchi_coefficient
is_higuchi_healthy
```
