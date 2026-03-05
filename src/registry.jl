"""
    ManifoldProperties(name, dim, metric, derivative, indices)

Properties of a manifold: dimension, associated metric/derivative symbols, and index alphabet.
"""
struct ManifoldProperties
    name::Symbol
    dim::Int
    metric::Union{Symbol, Nothing}
    derivative::Union{Symbol, Nothing}
    indices::Vector{Symbol}
end

"""
    TensorProperties(; name, manifold, rank, symmetries, dependencies, weight, options)

Properties of a tensor stored in the registry.
"""
struct TensorProperties
    name::Symbol
    manifold::Symbol
    rank::Tuple{Int,Int}
    symmetries::Vector{Any}
    dependencies::Vector{Symbol}
    weight::Int
    options::Dict{Symbol,Any}
end

function TensorProperties(; name::Symbol, manifold::Symbol, rank::Tuple{Int,Int},
                           symmetries::Vector{Any}=Any[],
                           dependencies::Vector{Symbol}=Symbol[],
                           weight::Int=0,
                           options::Dict{Symbol,Any}=Dict{Symbol,Any}())
    TensorProperties(name, manifold, rank, symmetries, dependencies, weight, options)
end

"""
    TensorRegistry()

A mutable container for manifold and tensor metadata, plus rewrite rules.
"""
mutable struct TensorRegistry
    manifolds::Dict{Symbol, ManifoldProperties}
    tensors::Dict{Symbol, TensorProperties}
    rules::Vector{Any}  # Vector{RewriteRule}, Any to avoid forward ref
end

TensorRegistry() = TensorRegistry(
    Dict{Symbol,ManifoldProperties}(),
    Dict{Symbol,TensorProperties}(),
    Any[]
)

has_manifold(reg::TensorRegistry, name::Symbol) = haskey(reg.manifolds, name)
has_tensor(reg::TensorRegistry, name::Symbol) = haskey(reg.tensors, name)

function get_manifold(reg::TensorRegistry, name::Symbol)
    reg.manifolds[name]
end

function get_tensor(reg::TensorRegistry, name::Symbol)
    reg.tensors[name]
end

function register_manifold!(reg::TensorRegistry, mp::ManifoldProperties)
    has_manifold(reg, mp.name) && error("Manifold $(mp.name) already registered")
    reg.manifolds[mp.name] = mp
    mp
end

function register_tensor!(reg::TensorRegistry, tp::TensorProperties)
    has_tensor(reg, tp.name) && error("Tensor $(tp.name) already registered")
    reg.tensors[tp.name] = tp
    tp
end

"""
    register_rule!(reg, rule)

Add a rewrite rule to the registry. Rules are applied during `simplify`.
"""
function register_rule!(reg::TensorRegistry, rule)
    push!(reg.rules, rule)
    rule
end

"""
    get_rules(reg) -> Vector

Return all registered rewrite rules.
"""
get_rules(reg::TensorRegistry) = reg.rules

# Global registry with context-based scoping
const _GLOBAL_REGISTRY = TensorRegistry()
const _REGISTRY_STACK = TensorRegistry[_GLOBAL_REGISTRY]

"""
    current_registry()

Return the currently active TensorRegistry (top of the context stack).
"""
current_registry() = _REGISTRY_STACK[end]

"""
    with_registry(f, reg::TensorRegistry)

Execute `f` with `reg` as the active registry, then restore the previous one.
"""
function with_registry(f, reg::TensorRegistry)
    push!(_REGISTRY_STACK, reg)
    try
        f()
    finally
        pop!(_REGISTRY_STACK)
    end
end
