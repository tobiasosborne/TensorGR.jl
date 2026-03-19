# One-liner spinor structure setup.
#
# Chains define_spinor_bundles!, define_spin_metric!, and define_soldering_form!
# into a single call, analogous to @manifold for the tensor case.
#
# Reference: Penrose & Rindler, Spinors and Space-Time Vol 1 (1984), Ch 3.
# xAct equivalent: DefSpinStructure[M, sigma]

"""
    define_spinor_structure!(reg::TensorRegistry; manifold::Symbol=:M4, metric::Symbol=:g)

Set up the complete spinor infrastructure on `manifold` in one call:

1. Register SL2C and SL2C_dot spinor VBundles
2. Register spin metrics epsilon_{AB} and epsilon_{A'B'} with deltas
3. Register soldering form sigma^a_{AA'} with completeness and metric rules

Requires that `manifold` and its `metric` are already registered
(via `@manifold` or `register_manifold!`).

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_spinor_structure!(reg; manifold=:M4, metric=:g)
    # Now ready for spinor calculus
end
```
"""
function define_spinor_structure!(reg::TensorRegistry;
                                  manifold::Symbol=:M4,
                                  metric::Symbol=:g)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")
    has_tensor(reg, metric) || error("Metric $metric not registered on $manifold")

    has_vbundle(reg, :SL2C) || define_spinor_bundles!(reg; manifold=manifold)
    has_tensor(reg, :eps_spin) || define_spin_metric!(reg; manifold=manifold)
    has_tensor(reg, :sigma) || define_soldering_form!(reg; manifold=manifold)

    nothing
end

"""
    @spinor_manifold M4 metric=g

Macro form of [`define_spinor_structure!`](@ref). Sets up the full spinor
infrastructure on the named manifold using the current registry.

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    @spinor_manifold M4 metric=g
end
```
"""
macro spinor_manifold(manifold_expr, kwargs...)
    manifold = manifold_expr
    metric = :g  # default

    for kw in kwargs
        if kw isa Expr && kw.head == :(=)
            key = kw.args[1]
            val = kw.args[2]
            if key == :metric
                metric = val
            end
        end
    end

    quote
        define_spinor_structure!(current_registry();
                                manifold=$(QuoteNode(manifold)),
                                metric=$(QuoteNode(metric)))
    end |> esc
end
