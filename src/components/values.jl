#= Component value storage.

Store and retrieve individual tensor component values, with symmetry awareness.
=#

"""
    ComponentStore

A storage container for tensor component values. Symmetry-aware:
only independent components are stored, and symmetry-related components
are computed on access.
"""
struct ComponentStore
    tensor::Symbol
    chart::Symbol
    dim::Int
    values::Dict{Vector{Int}, Any}
    symmetries::Vector{Any}
end

"""
    ComponentStore(tensor, chart, dim; symmetries=[])

Create a component store for the given tensor in the given chart.
"""
function ComponentStore(tensor::Symbol, chart::Symbol, dim::Int;
                        symmetries::Vector{Any}=Any[])
    ComponentStore(tensor, chart, dim, Dict{Vector{Int}, Any}(), symmetries)
end

"""
    set_component!(store, indices, value)

Set a component value. If the tensor has symmetries, also stores the value
for all symmetry-related index permutations.
"""
function set_component!(store::ComponentStore, indices::Vector{Int}, value)
    store.values[indices] = value

    # Store symmetry-related values
    for sym in store.symmetries
        if sym isa Symmetric
            perm_idx = copy(indices)
            perm_idx[sym.i], perm_idx[sym.j] = perm_idx[sym.j], perm_idx[sym.i]
            store.values[perm_idx] = value
        elseif sym isa AntiSymmetric
            perm_idx = copy(indices)
            perm_idx[sym.i], perm_idx[sym.j] = perm_idx[sym.j], perm_idx[sym.i]
            store.values[perm_idx] = isa(value, Number) ? -value : :(-$value)
        end
    end
    value
end

"""
    get_component(store, indices) -> Any

Retrieve a component value. Returns 0 if not set.
"""
function get_component(store::ComponentStore, indices::Vector{Int})
    get(store.values, indices, 0)
end

"""
    independent_components(store) -> Vector{Vector{Int}}

Return the list of independent component index tuples
(accounting for symmetries).
"""
function independent_components(store::ComponentStore)
    rank = 0
    for (k, _) in store.values
        rank = length(k)
        break
    end
    rank == 0 && return Vector{Int}[]

    all_indices = _all_index_tuples(store.dim, rank)
    seen = Set{Vector{Int}}()
    independent = Vector{Int}[]

    for idx in all_indices
        canonical = _canonicalize_component_index(idx, store.symmetries)
        if canonical ∉ seen
            push!(seen, canonical)
            push!(independent, canonical)
        end
    end
    independent
end

function _all_index_tuples(dim::Int, rank::Int)
    if rank == 0
        return [Int[]]
    end
    result = Vector{Int}[]
    _generate_tuples!(result, Int[], dim, rank)
    result
end

function _generate_tuples!(result, current, dim, remaining)
    if remaining == 0
        push!(result, copy(current))
        return
    end
    for i in 1:dim
        push!(current, i)
        _generate_tuples!(result, current, dim, remaining - 1)
        pop!(current)
    end
end

function _canonicalize_component_index(idx::Vector{Int}, symmetries::Vector{Any})
    result = copy(idx)
    changed = true
    while changed
        changed = false
        for sym in symmetries
            if sym isa Symmetric
                if result[sym.i] > result[sym.j]
                    result[sym.i], result[sym.j] = result[sym.j], result[sym.i]
                    changed = true
                end
            end
        end
    end
    result
end
