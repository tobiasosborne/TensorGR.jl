# Metric Ansatz Generators

Generate symbolic metric tensors from spacetime symmetry assumptions. The `SymmetryAnsatz` type hierarchy encodes sets of isometries (Killing vector fields) that a metric must respect, reducing the number of independent components. The `metric_ansatz` function dispatches on the ansatz type to produce a symbolic metric (requires Symbolics.jl).

## Symmetry Ansatz Types

The abstract type `SymmetryAnsatz` is the root of the hierarchy. Concrete subtypes specify the symmetry group and its parameters.

```julia
# SO(3) rotational symmetry (Schwarzschild, Reissner-Nordstrom)
ans = SphericalSymmetry(:M4)

# U(1) axial symmetry (Kerr, Lewis-Papapetrou)
ans = AxialSymmetry(:M4)
ans = AxialSymmetry(:M4; axis=:y)

# Time-translation invariance (static spacetimes)
ans = StaticSymmetry(:M4)
ans = StaticSymmetry(:M4; time_coord=:tau)

# Spatial homogeneity and isotropy (FLRW cosmology)
ans = HomogeneousIsotropy(:M4)
ans = HomogeneousIsotropy(:M4; curvature=:K)
```

```@docs
SymmetryAnsatz
SphericalSymmetry
AxialSymmetry
StaticSymmetry
HomogeneousIsotropy
```

## Metric Generation

The `metric_ansatz` function generates a `SymbolicMetric` from a symmetry ansatz. It requires `Symbolics.jl` to be loaded (weak dependency extension).

Each ansatz type produces a characteristic line element:

**SphericalSymmetry**: Static spherically symmetric metric with free functions A(r), B(r):

    ds^2 = -A(r) dt^2 + B(r) dr^2 + r^2 (d theta^2 + sin^2 theta d phi^2)

**AxialSymmetry**: Stationary axisymmetric metric in Lewis-Papapetrou form with free functions N, g\_rr, g\_{theta theta}, g\_{phi phi}, omega:

    ds^2 = -N^2 dt^2 + g_rr dr^2 + g_{theta theta} d theta^2 + g_{phi phi} (d phi - omega dt)^2

**HomogeneousIsotropy**: FLRW metric with scale factor a(tau) and spatial curvature k:

    ds^2 = -d tau^2 + a(tau)^2 [dr^2/(1 - k r^2) + r^2 d Omega^2]

```julia
# Requires: using Symbolics
# result = metric_ansatz(reg, :M4, SphericalSymmetry(:M4))
# result.metric          # SymbolicMetric
# result.free_functions  # [A, B]
```

```@docs
metric_ansatz
```

## Example: Schwarzschild Solution

```julia
using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4, signature=lorentzian(4))

    # The SphericalSymmetry ansatz represents the most general
    # static, spherically symmetric metric
    ans = SphericalSymmetry(:M4)

    # With Symbolics.jl loaded, generate the symbolic metric:
    # result = metric_ansatz(reg, :M4, ans)
    # Then compute curvature and solve Einstein equations to
    # determine the free functions A(r), B(r).
end
```

## Example: FLRW Cosmology

```julia
using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4, signature=lorentzian(4))

    # Spatially flat FLRW
    ans = HomogeneousIsotropy(:M4; curvature=:k)

    # With Symbolics.jl loaded:
    # result = metric_ansatz(reg, :M4, ans; k=0)
    # result.metric          # SymbolicMetric for ds^2 = -dtau^2 + a(tau)^2 dx^2
    # result.free_functions  # [a]  (the scale factor)
    # result.time_coord      # tau symbol
end
```
