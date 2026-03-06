#= Basis evaluation: convert abstract tensor expressions to component arrays.

Given a Tensor with free indices and a ChartProperties (which defines coordinates
and dimension), produce CTensor or raw Array representations where each component
corresponds to a specific assignment of coordinate indices.
=#

"""
    _replace_index(expr::TensorExpr, idx_name::Symbol, component::Int) -> TensorExpr

Replace all occurrences of index `idx_name` with a numeric component label.
The resulting expression encodes the component number in the index name as
`:_1`, `:_2`, etc.
"""
function _replace_index(expr::Tensor, idx_name::Symbol, component::Int)
    new_indices = map(expr.indices) do idx
        if idx.name == idx_name
            TIndex(Symbol("_", component), idx.position, idx.vbundle)
        else
            idx
        end
    end
    Tensor(expr.name, new_indices)
end

function _replace_index(expr::TProduct, idx_name::Symbol, component::Int)
    TProduct(expr.scalar, TensorExpr[_replace_index(f, idx_name, component) for f in expr.factors])
end

function _replace_index(expr::TSum, idx_name::Symbol, component::Int)
    TSum(TensorExpr[_replace_index(t, idx_name, component) for t in expr.terms])
end

function _replace_index(expr::TDeriv, idx_name::Symbol, component::Int)
    new_idx = expr.index.name == idx_name ?
        TIndex(Symbol("_", component), expr.index.position, expr.index.vbundle) : expr.index
    TDeriv(new_idx, _replace_index(expr.arg, idx_name, component))
end

_replace_index(expr::TScalar, ::Symbol, ::Int) = expr

"""
    _component_indices(expr::TensorExpr) -> Vector{TIndex}

Return the free indices of the expression, used to determine which indices
need component substitution.
"""
_component_indices(expr::TensorExpr) = free_indices(expr)

"""
    to_basis(expr::TensorExpr, chart::ChartProperties) -> CTensor

Convert an abstract tensor expression to a component representation in the
given chart. Each entry of the resulting CTensor is the abstract expression
with free indices replaced by component numbers.

For a rank-n tensor in a d-dimensional chart, produces a d^n array.
"""
function to_basis(expr::TensorExpr, chart::ChartProperties)
    fidx = free_indices(expr)
    dim = length(chart.coords)
    rank = length(fidx)

    # Determine index positions for the CTensor
    positions = IndexPosition[idx.position for idx in fidx]

    if rank == 0
        # Scalar expression
        return CTensor(fill(expr), chart.name, IndexPosition[])
    end

    # Build the component array
    data = Array{TensorExpr}(undef, ntuple(_ -> dim, rank)...)
    idx_names = [idx.name for idx in fidx]

    for ci in CartesianIndices(data)
        component = expr
        for (slot, idx_name) in enumerate(idx_names)
            component = _replace_index(component, idx_name, ci[slot])
        end
        data[ci] = component
    end

    CTensor(data, chart.name, positions)
end

"""
    component_array(expr::TensorExpr, chart::ChartProperties,
                    reg::TensorRegistry) -> Array

Extract all components of a tensor expression into a raw Array.
Each entry is the abstract expression with indices replaced by component numbers.
The registry is available for looking up tensor properties if needed.
"""
function component_array(expr::TensorExpr, chart::ChartProperties,
                         reg::TensorRegistry)
    fidx = free_indices(expr)
    dim = length(chart.coords)
    rank = length(fidx)

    if rank == 0
        return fill(expr)
    end

    data = Array{TensorExpr}(undef, ntuple(_ -> dim, rank)...)
    idx_names = [idx.name for idx in fidx]

    for ci in CartesianIndices(data)
        component = expr
        for (slot, idx_name) in enumerate(idx_names)
            component = _replace_index(component, idx_name, ci[slot])
        end
        data[ci] = component
    end

    data
end

"""
    to_ctensor(expr::TensorExpr, chart::ChartProperties,
               values::Dict) -> CTensor

Evaluate a tensor expression to numerical components using a dictionary
of known component values.

`values` maps `(tensor_name::Symbol, component_indices::Vector{Int})` to
numerical values. For example:
    `(:g, [1,1]) => -1.0`  for g_{11} = -1.

Returns a CTensor with the evaluated numerical data.
"""
function to_ctensor(expr::TensorExpr, chart::ChartProperties,
                    values::Dict)
    fidx = free_indices(expr)
    dim = length(chart.coords)
    rank = length(fidx)
    positions = IndexPosition[idx.position for idx in fidx]

    if rank == 0
        val = _evaluate_component(expr, values, dim)
        T = typeof(val)
        return CTensor(fill(val), chart.name, IndexPosition[])
    end

    # First pass to determine element type
    idx_names = [idx.name for idx in fidx]
    first_comp = expr
    for (slot, idx_name) in enumerate(idx_names)
        first_comp = _replace_index(first_comp, idx_name, 1)
    end
    first_val = _evaluate_component(first_comp, values, dim)
    T = typeof(first_val)

    data = Array{T}(undef, ntuple(_ -> dim, rank)...)
    for ci in CartesianIndices(data)
        component = expr
        for (slot, idx_name) in enumerate(idx_names)
            component = _replace_index(component, idx_name, ci[slot])
        end
        data[ci] = _evaluate_component(component, values, dim)
    end

    CTensor(data, chart.name, positions)
end

"""
    _evaluate_component(expr::TensorExpr, values::Dict, dim::Int)

Recursively evaluate a fully-indexed expression using the values dictionary.
"""
function _evaluate_component(expr::Tensor, values::Dict, dim::Int)
    # Extract numeric indices from the index names (e.g., :_1 -> 1)
    comp_indices = Int[]
    for idx in expr.indices
        s = string(idx.name)
        if startswith(s, "_") && length(s) > 1
            push!(comp_indices, parse(Int, s[2:end]))
        else
            error("Cannot evaluate expression with unresolved index: $(idx.name)")
        end
    end
    key = (expr.name, comp_indices)
    haskey(values, key) || return zero(Float64)
    return values[key]
end

function _evaluate_component(expr::TProduct, values::Dict, dim::Int)
    result = Rational{Int}(expr.scalar)
    for f in expr.factors
        result *= _evaluate_component(f, values, dim)
    end
    result
end

function _evaluate_component(expr::TSum, values::Dict, dim::Int)
    s = zero(Float64)
    for t in expr.terms
        s += _evaluate_component(t, values, dim)
    end
    s
end

function _evaluate_component(expr::TScalar, values::Dict, dim::Int)
    v = expr.val
    v isa Number ? v : error("Cannot evaluate symbolic scalar: $v")
end

function _evaluate_component(expr::TDeriv, values::Dict, dim::Int)
    error("Cannot numerically evaluate derivative expressions; expand derivatives first")
end
