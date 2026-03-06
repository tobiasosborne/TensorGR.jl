#= 3+1 foliation of spacetime manifolds.

Splits spacetime indices into temporal and spatial components for
SVT (Scalar-Vector-Tensor) decomposition of perturbations.
=#

"""
    FoliationProperties(name, manifold, temporal_component, spatial_components, spatial_dim)

Properties of a 3+1 foliation: which component index is temporal and which are spatial.
"""
struct FoliationProperties
    name::Symbol
    manifold::Symbol
    temporal_component::Int
    spatial_components::Vector{Int}
    spatial_dim::Int
end

"""
    define_foliation!(reg, name; manifold, temporal=0, spatial=[1,2,3])

Register a 3+1 foliation on a manifold. The temporal component defaults to 0
and spatial to [1,2,3] for a standard 4D spacetime.

Validation: temporal must not be in spatial, manifold must exist,
and total components must match manifold dimension.
"""
function define_foliation!(reg::TensorRegistry, name::Symbol;
                           manifold::Symbol,
                           temporal::Int=0,
                           spatial::Vector{Int}=Int[1,2,3])
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")
    mp = get_manifold(reg, manifold)
    total = 1 + length(spatial)
    total == mp.dim || error("Foliation dimension $(total) != manifold dimension $(mp.dim)")
    temporal in spatial && error("Temporal component $temporal must not be in spatial $spatial")
    length(unique(spatial)) == length(spatial) || error("Spatial components must be unique")
    has_foliation(reg, name) && error("Foliation $name already registered")

    fol = FoliationProperties(name, manifold, temporal, spatial, length(spatial))
    reg.foliations[name] = fol
    fol
end

"""
    get_foliation(reg, name) -> FoliationProperties

Retrieve a registered foliation by name.
"""
function get_foliation(reg::TensorRegistry, name::Symbol)
    reg.foliations[name]
end

"""
    has_foliation(reg, name) -> Bool

Check if a foliation is registered.
"""
has_foliation(reg::TensorRegistry, name::Symbol) = haskey(reg.foliations, name)

"""
    classify_component(component::Int, fol::FoliationProperties) -> Symbol

Classify a component index as `:temporal` or `:spatial`.
"""
function classify_component(component::Int, fol::FoliationProperties)
    component == fol.temporal_component && return :temporal
    component in fol.spatial_components && return :spatial
    error("Component $component not in foliation $(fol.name)")
end

"""
    all_components(fol::FoliationProperties) -> Vector{Int}

Return all component indices (temporal first, then spatial) for this foliation.
"""
function all_components(fol::FoliationProperties)
    vcat([fol.temporal_component], fol.spatial_components)
end

"""
    foliate_and_decompose(expr, h_name; foliation, fields=DEFAULT_SVT, gauge=:bardeen)

End-to-end pipeline:
1. Split all spacetime indices into 3+1 components
2. Apply SVT substitution rules (Bardeen or full gauge)
3. Apply constraint rules (transversality, tracelessness)
4. Collect by SO(3) sector

Returns `Dict{Symbol, TensorExpr}` mapping sector names to expressions.
"""
function foliate_and_decompose(expr::TensorExpr, h_name::Symbol;
                                foliation::FoliationProperties,
                                fields::SVTFields=DEFAULT_SVT,
                                gauge::Symbol=:bardeen)
    # 1. Split all spacetime indices into 3+1 components
    split_expr = split_all_spacetime(expr, foliation)

    # 2. Apply SVT substitution rules
    substituted = apply_svt(split_expr, h_name, foliation; gauge=gauge, fields=fields)

    # 3. Apply constraint rules (transversality, tracelessness)
    crules = svt_constraint_rules(fields, foliation)
    constrained = apply_rules_fixpoint(substituted, crules)

    # 4. Collect by sector
    collect_sectors(constrained, fields)
end
