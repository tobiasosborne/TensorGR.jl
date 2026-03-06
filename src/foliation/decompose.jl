#= Index classification and spacetime splitting for 3+1 foliation.

Splits abstract spacetime indices into temporal (0) and spatial (i)
components, producing explicit sums over component terms.
=#

"""
    split_spacetime(expr::TensorExpr, idx_name::Symbol, fol::FoliationProperties) -> TensorExpr

Replace the abstract index `idx_name` with a sum over all components
(temporal + spatial) of the foliation. Each term has the index replaced
by a numeric component label `:_0`, `:_1`, etc.

Uses `_replace_index` from `src/components/to_basis.jl`.
"""
function split_spacetime(expr::TensorExpr, idx_name::Symbol, fol::FoliationProperties)
    components = all_components(fol)
    terms = TensorExpr[]
    for c in components
        push!(terms, _replace_index(expr, idx_name, c))
    end
    tsum(terms)
end

"""
    split_all_spacetime(expr::TensorExpr, fol::FoliationProperties) -> TensorExpr

Split all free indices in the expression into their 3+1 components.
For a rank-n expression in d dimensions, produces up to d^n terms.

Iterates over free indices, applying `split_spacetime` for each one.
"""
function split_all_spacetime(expr::TensorExpr, fol::FoliationProperties)
    fidx = free_indices(expr)
    isempty(fidx) && return expr

    # Get unique free index names (preserving order of first occurrence)
    seen = Set{Symbol}()
    idx_names = Symbol[]
    for idx in fidx
        if idx.name ∉ seen
            push!(seen, idx.name)
            push!(idx_names, idx.name)
        end
    end

    result = expr
    for name in idx_names
        result = split_spacetime(result, name, fol)
    end
    result
end

"""
    is_temporal_component(idx::TIndex, fol::FoliationProperties) -> Bool

Check if an index has been replaced with the temporal component marker.
"""
function is_temporal_component(idx::TIndex, fol::FoliationProperties)
    s = string(idx.name)
    startswith(s, "_") && length(s) > 1 || return false
    comp = tryparse(Int, s[2:end])
    comp !== nothing && comp == fol.temporal_component
end

"""
    is_spatial_component(idx::TIndex, fol::FoliationProperties) -> Bool

Check if an index has been replaced with a spatial component marker.
"""
function is_spatial_component(idx::TIndex, fol::FoliationProperties)
    s = string(idx.name)
    startswith(s, "_") && length(s) > 1 || return false
    comp = tryparse(Int, s[2:end])
    comp !== nothing && comp in fol.spatial_components
end

"""
    component_value(idx::TIndex) -> Union{Int, Nothing}

Extract the numeric component value from a component marker index like `:_0`, `:_1`.
Returns `nothing` if the index is not a component marker.
"""
function component_value(idx::TIndex)
    s = string(idx.name)
    startswith(s, "_") && length(s) > 1 || return nothing
    tryparse(Int, s[2:end])
end

"""
    component_pattern(t::Tensor, fol::FoliationProperties) -> Vector{Symbol}

Classify each index of a tensor as `:temporal`, `:spatial`, or `:abstract`.
Returns a vector of classifications.
"""
function component_pattern(t::Tensor, fol::FoliationProperties)
    map(t.indices) do idx
        if is_temporal_component(idx, fol)
            :temporal
        elseif is_spatial_component(idx, fol)
            :spatial
        else
            :abstract
        end
    end
end
