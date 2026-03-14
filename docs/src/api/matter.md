# Matter Fields

Stress-energy tensors for standard matter content in general relativity.
Currently provides the perfect fluid stress-energy tensor; additional
matter models (scalar field, electromagnetic, dust) can be built from
the core tensor algebra.

## Perfect Fluid

The perfect fluid stress-energy tensor is:

    T^{ab} = (rho + p) u^a u^b + p g^{ab}

where `rho` is the energy density, `p` is the pressure, and `u^a` is the
4-velocity field satisfying the normalization condition `g_{ab} u^a u^b = -1`.

### Defining a Perfect Fluid

`define_perfect_fluid!` registers the stress-energy tensor, energy density,
pressure, and 4-velocity as tensors in the registry. It also registers the
normalization rule `u_a u^a = -1`.

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    # Define a perfect fluid with default field names
    fp = define_perfect_fluid!(reg, :T; manifold=:M4, metric=:g)

    # Build the stress-energy expression
    expr = perfect_fluid_expr(up(:a), up(:b), fp)
    # (rho + p) u^a u^b + p g^{ab}

    # The normalization rule u_a u^a = -1 is automatically applied:
    uu = Tensor(:u, [down(:a)]) * Tensor(:u, [up(:a)])
    simplify(uu)  # => TScalar(-1//1)
end
```

### Custom Field Names

```julia
fp = define_perfect_fluid!(reg, :Tmatter;
    manifold=:M4, metric=:g,
    rho=:epsilon, p=:P, u=:U)
```

```@docs
PerfectFluidProperties
define_perfect_fluid!
perfect_fluid_expr
get_perfect_fluid
```

## Einstein's Equation with Matter

Combine the perfect fluid with curvature tensors to write Einstein's equation:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    fp = define_perfect_fluid!(reg, :T; manifold=:M4, metric=:g)

    # G_{ab} = 8 pi T_{ab}
    G = Tensor(:Ein, [down(:a), down(:b)])
    T = perfect_fluid_expr(up(:a), up(:b), fp)

    # Solve for the fluid variables using solve_tensors
    # or impose as a rewrite rule
end
```

## Scalar Field Stress-Energy

For a minimally coupled scalar field, the stress-energy tensor can be
constructed directly from the core algebra:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    @define_tensor phi on=M4 rank=(0,0)

    # T_{ab} = d_a(phi) d_b(phi) - (1/2) g_{ab} (d_c(phi) d^c(phi))
    dphi_a = TDeriv(down(:a), Tensor(:phi, TIndex[]))
    dphi_b = TDeriv(down(:b), Tensor(:phi, TIndex[]))
    kinetic = grad_squared(Tensor(:phi, TIndex[]), :g)
    T_ab = dphi_a * dphi_b - (1//2) * Tensor(:g, [down(:a), down(:b)]) * kinetic
end
```
