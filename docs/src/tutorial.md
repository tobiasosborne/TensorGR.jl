# Tutorial: Fourth-Derivative Gravity

This tutorial reproduces the main result from TensorGR's test suite: computing the propagator structure of fourth-derivative gravity.

## Setting Up

```julia
using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    # Define a 4D manifold with metric g
    @manifold M4 dim=4 metric=g

    # Register curvature tensors
    define_curvature_tensors!(reg, :M4, :g)

    # Define covariant derivative
    @covd D on=M4 metric=g
end
```

## Building Expressions

```julia
with_registry(reg) do
    # Riemann tensor
    R = @tensor Riem[-a, -b, -c, -d]

    # Ricci tensor and scalar
    Ric = @tensor Ric[-a, -b]
    RicS = @tensor RicScalar

    # Metric
    g_ab = @tensor g[-a, -b]
end
```

## Linearization

```julia
with_registry(reg) do
    # First-order perturbation of Ricci tensor
    dR = δRicci(down(:a), down(:b), :h)

    # First-order perturbation of Ricci scalar
    dRS = δRicciScalar(:h)
end
```

## Covariant Derivatives

```julia
with_registry(reg) do
    @define_tensor V on=M4 rank=(1,0)
    V_b = Tensor(:V, [up(:b)])

    # Expand covariant derivative
    expr = TDeriv(down(:a), V_b)
    expanded = covd_to_christoffel(expr, :D)
    # Result: ∂_a V^b + Γ^b_{ac} V^c
end
```

## Component Calculations

```julia
# Schwarzschild metric components at r=10, M=1
R_val = 10.0; M = 1.0; f = 1 - 2M/R_val

g_comp = zeros(4, 4)
g_comp[1,1] = -f
g_comp[2,2] = 1/f
g_comp[3,3] = R_val^2
g_comp[4,4] = R_val^2  # at θ=π/2

ginv_comp = zeros(4, 4)
ginv_comp[1,1] = -1/f
ginv_comp[2,2] = f
ginv_comp[3,3] = 1/R_val^2
ginv_comp[4,4] = 1/R_val^2

# Compute Christoffel symbols
Gamma = metric_christoffel(g_comp, ginv_comp, [:t,:r,:θ,:φ];
    deriv_fn=(expr, coord) -> coord == :r && expr ≈ -f ? -(2M/R_val^2) : 0.0)
```

## Simplification Pipeline

The `simplify` function chains:
1. `expand_products` — distribute `*` over `+`
2. `contract_metrics` — eliminate g and δ contractions
3. `canonicalize` — canonical index ordering via xperm.c
4. `collect_terms` — combine like terms (with dummy normalization)
5. `apply_rules` — registered rewrite rules

```julia
with_registry(reg) do
    g_up = Tensor(:g, [up(:a), up(:b)])
    g_dn = Tensor(:g, [down(:b), down(:c)])
    result = simplify(g_up * g_dn)
    # Result: δ^a_c
end
```
