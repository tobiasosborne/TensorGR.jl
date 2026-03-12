#= Metric ansatz generators.

Stub function for metric_ansatz. Implementations dispatching on SymmetryAnsatz
subtypes live in ext/TensorGRSymbolicsExt.jl (requires Symbolics.jl).
=#

"""
    metric_ansatz(reg, manifold, ansatz; kwargs...) -> NamedTuple

Generate a symbolic metric from a symmetry ansatz. Requires `Symbolics.jl`.

Dispatches on `ansatz` type:

- `HomogeneousIsotropy` -- FLRW metric with scale factor `a(tau)` and spatial curvature `k`.
  Returns `(metric=SymbolicMetric, free_functions=[a], time_coord=tau_sym)`.

- `SphericalSymmetry` -- static spherically symmetric metric with free functions `A(r)`, `B(r)`.
  Returns `(metric=SymbolicMetric, free_functions=[A, B], radial_coord=r_sym)`.

# Keyword arguments

- `coords::Vector{Symbol}` -- coordinate names (defaults depend on ansatz type)
- `k::Int` -- spatial curvature for FLRW (0, +1, or -1; default 0)
"""
function metric_ansatz end
