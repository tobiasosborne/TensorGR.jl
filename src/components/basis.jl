#= Coordinate charts and bases for component calculations.

A Chart defines a coordinate system on a manifold with named coordinates.
This enables evaluation of abstract tensor expressions to component arrays.
=#

"""
    ChartProperties(name, manifold, coords)

A coordinate chart on a manifold.
"""
struct ChartProperties
    name::Symbol
    manifold::Symbol
    coords::Vector{Symbol}
end

"""
    define_chart!(reg, name; manifold, coords) -> ChartProperties

Define a coordinate chart and register it.
"""
function define_chart!(reg::TensorRegistry, name::Symbol;
                       manifold::Symbol, coords::Vector{Symbol})
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")
    mp = get_manifold(reg, manifold)
    length(coords) == mp.dim || error("Chart needs $(mp.dim) coordinates, got $(length(coords))")

    chart = ChartProperties(name, manifold, coords)

    # Store in registry tensors with a marker
    register_tensor!(reg, TensorProperties(
        name=name, manifold=manifold, rank=(0, 0),
        symmetries=Any[],
        options=Dict{Symbol,Any}(:is_chart => true, :chart_props => chart)))

    chart
end

function get_chart(reg::TensorRegistry, name::Symbol)
    has_tensor(reg, name) || error("Chart $name not registered")
    props = get_tensor(reg, name)
    get(props.options, :is_chart, false) || error("$name is not a chart")
    props.options[:chart_props]::ChartProperties
end
