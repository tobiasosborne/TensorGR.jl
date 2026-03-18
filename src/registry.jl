"""
    VBundleProperties(name, manifold, dim, indices[, options])

Properties of a vector bundle: name, base manifold, fiber dimension, index alphabet,
and optional metadata (e.g., `:conjugate_bundle`, `:is_spinor`).
"""
struct VBundleProperties
    name::Symbol
    manifold::Symbol
    dim::Int
    indices::Vector{Symbol}
    options::Dict{Symbol,Any}
end

# Backward-compatible 4-arg positional constructor
VBundleProperties(name::Symbol, manifold::Symbol, dim::Int, indices::Vector{Symbol}) =
    VBundleProperties(name, manifold, dim, indices, Dict{Symbol,Any}())

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
mutable struct TensorProperties
    name::Symbol
    manifold::Symbol
    rank::Tuple{Int,Int}
    symmetries::Vector{SymmetrySpec}
    dependencies::Vector{Symbol}
    weight::Int
    # Hot-path boolean fields (avoid Dict lookup in contraction inner loop)
    is_metric::Bool
    is_delta::Bool
    frozen::Bool
    flat::Bool
    is_covd::Bool
    is_christoffel::Bool
    vanishing::Bool
    options::Dict{Symbol,Any}
end

function TensorProperties(; name::Symbol, manifold::Symbol, rank::Tuple{Int,Int},
                           symmetries=SymmetrySpec[],
                           dependencies::Vector{Symbol}=Symbol[],
                           weight::Int=0,
                           is_metric::Bool=false,
                           is_delta::Bool=false,
                           frozen::Bool=false,
                           options::Dict{Symbol,Any}=Dict{Symbol,Any}())
    # Infer struct fields from options for backward compatibility
    _is_metric = is_metric || get(options, :is_metric, false)
    _is_delta = is_delta || get(options, :is_delta, false)
    _frozen = frozen || get(options, :frozen, false)
    _flat = get(options, :flat, false)
    _is_covd = get(options, :is_covd, false)
    _is_christoffel = get(options, :is_christoffel, false)
    _vanishing = get(options, :vanishing, false)
    # Convert Any[] to SymmetrySpec[] for backward compatibility
    typed_syms = symmetries isa Vector{SymmetrySpec} ? symmetries :
                 SymmetrySpec[s for s in symmetries]
    TensorProperties(name, manifold, rank, typed_syms, dependencies, weight,
                     _is_metric, _is_delta, _frozen, _flat, _is_covd, _is_christoffel,
                     _vanishing, options)
end

"""
    TensorRegistry()

A mutable container for manifold and tensor metadata, plus rewrite rules.
"""
mutable struct TensorRegistry
    manifolds::Dict{Symbol, ManifoldProperties}
    tensors::Dict{Symbol, TensorProperties}
    rules::Vector{Any}  # Vector{RewriteRule}, Any to avoid forward ref
    vbundles::Dict{Symbol, VBundleProperties}
    foliations::Dict{Symbol, Any}  # Dict{Symbol, FoliationProperties}, Any to avoid forward ref
    mappings::Dict{Symbol, Any}    # Dict{Symbol, MappingProperties}, Any to avoid forward ref
    tex_aliases::Dict{Tuple{Symbol,Int}, Symbol}  # (tex_name, rank) => tensor_name; rank=-1 for any
    # Caches: metric/delta name per manifold (populated by register_tensor!)
    metric_cache::Dict{Symbol, Symbol}   # manifold => metric tensor name
    delta_cache::Dict{Symbol, Symbol}    # manifold => delta tensor name
end

TensorRegistry() = TensorRegistry(
    Dict{Symbol,ManifoldProperties}(),
    Dict{Symbol,TensorProperties}(),
    Any[],
    Dict{Symbol,VBundleProperties}(),
    Dict{Symbol,Any}(),
    Dict{Symbol,Any}(),
    Dict{Tuple{Symbol,Int}, Symbol}(),
    Dict{Symbol,Symbol}(),
    Dict{Symbol,Symbol}()
)

has_manifold(reg::TensorRegistry, name::Symbol) = haskey(reg.manifolds, name)
has_tensor(reg::TensorRegistry, name::Symbol) = haskey(reg.tensors, name)

function get_manifold(reg::TensorRegistry, name::Symbol)
    reg.manifolds[name]
end

function get_tensor(reg::TensorRegistry, name::Symbol)
    reg.tensors[name]
end

has_vbundle(reg::TensorRegistry, name::Symbol) = haskey(reg.vbundles, name)

function get_vbundle(reg::TensorRegistry, name::Symbol)
    reg.vbundles[name]
end

"""
    define_vbundle!(reg, name; manifold, dim, indices, conjugate_bundle=nothing)

Register a vector bundle on a manifold with given fiber dimension and index alphabet.
Optionally specify a conjugate bundle (e.g., for spinor SL2C/SL2C_dot pairs).
"""
function define_vbundle!(reg::TensorRegistry, name::Symbol;
                         manifold::Symbol, dim::Int,
                         indices::Vector{Symbol}=Symbol[],
                         conjugate_bundle::Union{Nothing,Symbol}=nothing)
    has_vbundle(reg, name) && error("VBundle $name already registered")
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")
    opts = Dict{Symbol,Any}()
    if conjugate_bundle !== nothing
        opts[:conjugate_bundle] = conjugate_bundle
    end
    vb = VBundleProperties(name, manifold, dim, indices, opts)
    reg.vbundles[name] = vb
    vb
end

"""
    conjugate_vbundle(reg, name) -> Union{Nothing, Symbol}

Return the conjugate bundle name for vbundle `name`, or `nothing` if none is defined.
"""
function conjugate_vbundle(reg::TensorRegistry, name::Symbol)
    has_vbundle(reg, name) || error("VBundle $name not registered")
    get(reg.vbundles[name].options, :conjugate_bundle, nothing)
end

function register_manifold!(reg::TensorRegistry, mp::ManifoldProperties)
    has_manifold(reg, mp.name) && error("Manifold $(mp.name) already registered")
    reg.manifolds[mp.name] = mp
    # Auto-register the tangent bundle (only if not already registered for a different manifold)
    if has_vbundle(reg, :Tangent)
        existing = get_vbundle(reg, :Tangent)
        if existing.manifold != mp.name
            @warn "Overwriting :Tangent vbundle (was on $(existing.manifold), now on $(mp.name))"
        end
    end
    reg.vbundles[:Tangent] = VBundleProperties(:Tangent, mp.name, mp.dim, mp.indices)
    mp
end

function register_tensor!(reg::TensorRegistry, tp::TensorProperties)
    has_tensor(reg, tp.name) && error("Tensor $(tp.name) already registered")
    reg.tensors[tp.name] = tp
    # Populate metric/delta caches
    tp.is_metric && (reg.metric_cache[tp.manifold] = tp.name)
    tp.is_delta && (reg.delta_cache[tp.manifold] = tp.name)
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

"""
    unregister_tensor!(reg, name)

Remove a tensor from the registry. Errors if other tensors depend on it.
"""
function unregister_tensor!(reg::TensorRegistry, name::Symbol)
    has_tensor(reg, name) || error("Tensor $name not registered")
    # Check for dependents
    for (tname, tp) in reg.tensors
        tname == name && continue
        if name in tp.dependencies
            error("Cannot remove $name: tensor $tname depends on it")
        end
        if get(tp.options, :metric, nothing) == name ||
           get(tp.options, :covd, nothing) == name
            error("Cannot remove $name: tensor $tname references it")
        end
    end
    delete!(reg.tensors, name)
    nothing
end

"""
    tex_alias!(reg, tex_name, tensor_name; rank=-1)

Register a LaTeX parser alias: `tex"tex_name_{...}"` with `rank` indices maps to `tensor_name`.
Use `rank=-1` for a catch-all alias regardless of index count.
"""
function tex_alias!(reg::TensorRegistry, tex_name::Symbol, tensor_name::Symbol; rank::Int=-1)
    reg.tex_aliases[(tex_name, rank)] = tensor_name
end

"""
    unregister_manifold!(reg, name)

Remove a manifold from the registry. Errors if tensors are defined on it.
"""
function unregister_manifold!(reg::TensorRegistry, name::Symbol)
    has_manifold(reg, name) || error("Manifold $name not registered")
    for (tname, tp) in reg.tensors
        tp.manifold == name && error("Cannot remove manifold $name: tensor $tname is defined on it")
    end
    delete!(reg.manifolds, name)
    nothing
end

"""
    unregister_covd!(reg, name)

Remove a covariant derivative and its Christoffel symbol.
"""
function unregister_covd!(reg::TensorRegistry, name::Symbol)
    has_tensor(reg, name) || error("CovD $name not registered")
    props = get_tensor(reg, name)
    props.is_covd || error("$name is not a CovD")
    christoffel = props.options[:covd_props].christoffel
    delete!(reg.tensors, name)
    haskey(reg.tensors, christoffel) && delete!(reg.tensors, christoffel)
    nothing
end

"""
    set_vanishing!(reg, name)

Mark a tensor as identically zero. Adds a rule that replaces it with ZERO.
"""
function set_vanishing!(reg::TensorRegistry, name::Symbol)
    has_tensor(reg, name) || error("Tensor $name not registered")
    tp = get_tensor(reg, name)
    tp.vanishing = true
    tp.options[:vanishing] = true
    register_rule!(reg, RewriteRule(
        expr -> expr isa Tensor && expr.name == name,
        _ -> TScalar(0 // 1)
    ))
    nothing
end

# Global registry with task-local scoping (thread-safe)
const _GLOBAL_REGISTRY = TensorRegistry()

"""
    current_registry()

Return the currently active TensorRegistry (top of the task-local stack).
Falls back to `_GLOBAL_REGISTRY` if no stack exists for the current task.
"""
function current_registry()
    stack = get(task_local_storage(), :_tgr_reg_stack, nothing)
    stack !== nothing && !isempty(stack) ? stack[end] : _GLOBAL_REGISTRY
end

"""
    with_registry(f, reg::TensorRegistry)

Execute `f` with `reg` as the active registry, then restore the previous one.
Each task maintains its own registry stack via `task_local_storage`.
"""
function with_registry(f, reg::TensorRegistry)
    stack = get!(task_local_storage(), :_tgr_reg_stack) do
        TensorRegistry[]
    end
    push!(stack, reg)
    try
        f()
    finally
        pop!(stack)
    end
end
