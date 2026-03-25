#= Full component evaluation: abstract TensorExpr → numeric CTensor.

Extends the to_basis/to_ctensor pipeline to handle dummy index summation
(Einstein convention), partial derivatives, and automatic metric inverse
computation.  This is TensorGR's equivalent of xCoba's ToBasis + TraceBasisDummy.

The algorithm is expression-level summation: for each assignment of free indices
(outer loop, → output array slots) and each assignment of dummy indices (inner
loop, Einstein sum), walk the expression tree and look up component values.
This reuses the existing `_replace_index` and `_evaluate_component` infrastructure.
=#

"""
    evaluate_components(expr::TensorExpr, chart::ChartProperties, values::Dict;
                        registry::TensorRegistry=current_registry(),
                        deriv_fn=nothing,
                        simplify_fn=identity) -> CTensor

Fully evaluate an abstract tensor expression to a numeric (or symbolic) CTensor.

Handles:
- Free index expansion (Cartesian product over chart dimension)
- Dummy index summation (Einstein convention), including per-term dummies in TSums
- Products, sums, scalars, nested contractions
- Partial derivatives (via `deriv_fn` callback or pre-computed values)

Note: Value lookup is position-agnostic — `T^{ab}` and `T_{ab}` look up the
same key `(:T, [i,j])`.  For expressions involving both `g_{ab}` and `g^{ab}`,
contract metrics at the abstract level first (via `contract_metrics`) so that
only one position variant needs values.

# Arguments
- `expr`: Abstract tensor expression to evaluate.
- `chart`: Coordinate chart (determines dimension and coordinate names).
- `values`: Maps `(tensor_name::Symbol, indices::Vector{Int})` to component
  values.  For scalars: `(:M, Int[]) => 1.0`.
- `registry`: For looking up tensor properties.
- `deriv_fn`: Optional `(value, coord::Symbol) -> derivative_value` for partial
  derivatives.  Required if `expr` contains unexpanded TDeriv nodes.
- `simplify_fn`: Applied to each output component after summation.

# Example
```julia
vals = Dict((:g, [1,1]) => -1.0, (:g, [2,2]) => 1.0,
            (:T, [1,1]) => 3.0,  (:T, [2,2]) => 5.0)
chart = define_chart!(reg, :cart; manifold=:M, coords=[:x, :y])
ct = evaluate_components(g^{ab}*T_{ab}, chart, vals)  # scalar trace = 3+5
```
"""
function evaluate_components(expr::TensorExpr, chart::ChartProperties,
                             values::Dict;
                             registry::TensorRegistry=current_registry(),
                             deriv_fn=nothing,
                             simplify_fn=identity)
    _, free_set, dpairs = _analyze_indices(expr)
    dim = length(chart.coords)
    dummy_names = Symbol[p[1].name for p in dpairs]

    # Use the expression's natural index order (from indices()) for deterministic
    # output axis ordering.  free_indices() uses Dict iteration which is
    # non-deterministic in Julia 1.12+.
    free_name_set = Set(idx.name for idx in free_set)
    all_idxs = indices(expr)
    fidx = TIndex[]
    seen = Set{Symbol}()
    for idx in all_idxs
        if idx.name in free_name_set && !(idx.name in seen)
            push!(fidx, idx)
            push!(seen, idx.name)
        end
    end

    positions = IndexPosition[idx.position for idx in fidx]
    idx_names = Symbol[idx.name for idx in fidx]

    if isempty(fidx)
        # Scalar expression: just evaluate with dummy summation
        val = _evaluate_with_dummies(expr, dummy_names, values, dim,
                                     chart.coords, deriv_fn)
        val = simplify_fn(val)
        return CTensor(fill(val), chart.name, IndexPosition[])
    end

    # Probe first component to determine output element type
    first_expr = expr
    for (slot, name) in enumerate(idx_names)
        first_expr = _replace_index(first_expr, name, 1)
    end
    first_val = simplify_fn(_evaluate_with_dummies(first_expr, dummy_names,
                                                    values, dim,
                                                    chart.coords, deriv_fn))
    T = typeof(first_val)

    data = Array{T}(undef, ntuple(_ -> dim, length(fidx))...)
    for ci in CartesianIndices(data)
        component = expr
        for (slot, name) in enumerate(idx_names)
            component = _replace_index(component, name, ci[slot])
        end
        data[ci] = simplify_fn(_evaluate_with_dummies(component, dummy_names,
                                                       values, dim,
                                                       chart.coords, deriv_fn))
    end

    CTensor(data, chart.name, positions)
end

"""
    _evaluate_with_dummies(expr, dummy_names, values, dim, coords, deriv_fn) -> Number

Evaluate an expression whose free indices have already been resolved to
component numbers.  Sums over the given `dummy_names` (Einstein convention),
then delegates to `_evaluate_component_full` for the fully-resolved leaf
evaluation.

IMPORTANT: `dummy_names` must come from the ORIGINAL expression (before free
index replacement).  Re-detecting dummies after replacement would mistake
resolved component indices (e.g., two `:_1` entries) for dummy pairs.
"""
function _evaluate_with_dummies(expr::TensorExpr, dummy_names::Vector{Symbol},
                                values::Dict, dim::Int,
                                coords::Vector{Symbol}, deriv_fn)
    if isempty(dummy_names)
        return _evaluate_component_full(expr, values, dim, coords, deriv_fn)
    end

    n_dummy = length(dummy_names)

    # Sum over all assignments of dummy indices
    # Use the first evaluation to determine the accumulator type
    first_fixed = expr
    for name in dummy_names
        first_fixed = _replace_index(first_fixed, name, 1)
    end
    total = _evaluate_component_full(first_fixed, values, dim, coords, deriv_fn)

    first = true
    for ci in CartesianIndices(ntuple(_ -> dim, n_dummy))
        if first
            first = false
            continue  # skip (1,1,...,1) already computed
        end
        fixed = expr
        for (k, name) in enumerate(dummy_names)
            fixed = _replace_index(fixed, name, ci[k])
        end
        total += _evaluate_component_full(fixed, values, dim, coords, deriv_fn)
    end
    total
end

"""
    _evaluate_component_full(expr, values, dim, coords, deriv_fn) -> Number

Recursively evaluate a fully-indexed expression (all indices resolved to
component numbers :_1, :_2, etc.).  Extends `_evaluate_component` with
TDeriv support via `deriv_fn`.
"""
function _evaluate_component_full(expr::Tensor, values::Dict, dim::Int,
                                  coords::Vector{Symbol}, deriv_fn)
    _evaluate_component(expr, values, dim)
end

function _evaluate_component_full(expr::TProduct, values::Dict, dim::Int,
                                  coords::Vector{Symbol}, deriv_fn)
    result = expr.scalar
    for f in expr.factors
        result *= _evaluate_component_full(f, values, dim, coords, deriv_fn)
    end
    result
end

function _evaluate_component_full(expr::TSum, values::Dict, dim::Int,
                                  coords::Vector{Symbol}, deriv_fn)
    # Each TSum term may have its own internal contractions (dummy pairs).
    # Detect and sum over per-term dummies independently.
    total = _eval_term_with_local_dummies(expr.terms[1], values, dim, coords, deriv_fn)
    for i in 2:length(expr.terms)
        total += _eval_term_with_local_dummies(expr.terms[i], values, dim, coords, deriv_fn)
    end
    total
end

"""
    _eval_term_with_local_dummies(term, values, dim, coords, deriv_fn) -> Number

Evaluate a single TSum term, detecting and summing over any per-term dummy
pairs that weren't handled at the outer level.  Component-marker indices
(:_1, :_2, ...) are excluded from dummy detection.
"""
function _eval_term_with_local_dummies(term::TensorExpr, values::Dict, dim::Int,
                                       coords::Vector{Symbol}, deriv_fn)
    dpairs = dummy_pairs(term)
    # Filter: only keep pairs where NEITHER index is a component marker (:_N)
    real_dummies = Symbol[]
    for (idx1, idx2) in dpairs
        _is_component_marker(idx1.name) && continue
        _is_component_marker(idx2.name) && continue
        push!(real_dummies, idx1.name)
    end
    if isempty(real_dummies)
        return _evaluate_component_full(term, values, dim, coords, deriv_fn)
    end
    _evaluate_with_dummies(term, real_dummies, values, dim, coords, deriv_fn)
end

"""Return true if a symbol is a component marker like :_1, :_2, etc."""
function _is_component_marker(name::Symbol)
    s = string(name)
    startswith(s, "_") && length(s) > 1 && all(isdigit, s[2:end])
end

function _evaluate_component_full(expr::TScalar, values::Dict, dim::Int,
                                  coords::Vector{Symbol}, deriv_fn)
    _evaluate_component(expr, values, dim)
end

function _evaluate_component_full(expr::TDeriv, values::Dict, dim::Int,
                                  coords::Vector{Symbol}, deriv_fn)
    expr.covd == :partial || error(
        "Covariant derivative :$(expr.covd) must be expanded before component " *
        "evaluation.  Use covd_to_christoffel() first.")

    # Extract the component number from the derivative index
    s = string(expr.index.name)
    if !(startswith(s, "_") && length(s) > 1)
        error("Derivative index $(expr.index.name) not resolved to component number")
    end
    deriv_comp = parse(Int, s[2:end])

    # Strategy 1: use deriv_fn callback for symbolic/numeric differentiation
    if deriv_fn !== nothing && !isempty(coords)
        inner_val = _evaluate_component_full(expr.arg, values, dim, coords, deriv_fn)
        return deriv_fn(inner_val, coords[deriv_comp])
    end

    # Strategy 2: look up pre-computed derivative value
    # Convention: ∂_k T_{ij} stored as (Symbol("∂", T_name), [k, i, j])
    _lookup_deriv_value(expr, values, dim)
end

"""
    _lookup_deriv_value(expr::TDeriv, values, dim) -> Number

Look up a pre-computed derivative value from the values dict.

Convention: `∂_k T_{ij}` is stored as `(Symbol("∂", :T), [k, i, j])`.
Nested derivatives: `∂_k ∂_l T_{ij}` as `(Symbol("∂∂", :T), [k, l, i, j])`.
"""
function _lookup_deriv_value(expr::TDeriv, values::Dict, dim::Int)
    # Peel off derivative layers to find the inner tensor
    deriv_indices = Int[]
    inner = expr
    prefix = ""
    while inner isa TDeriv
        s = string(inner.index.name)
        (startswith(s, "_") && length(s) > 1) || error(
            "Derivative index $(inner.index.name) not resolved to component number")
        push!(deriv_indices, parse(Int, s[2:end]))
        prefix *= "∂"
        inner = inner.arg
    end

    if inner isa Tensor
        tensor_indices = Int[]
        for idx in inner.indices
            s = string(idx.name)
            (startswith(s, "_") && length(s) > 1) || error(
                "Index $(idx.name) not resolved to component number")
            push!(tensor_indices, parse(Int, s[2:end]))
        end
        key = (Symbol(prefix, inner.name), vcat(deriv_indices, tensor_indices))
        return get(values, key, zero(Float64))
    end

    error("Cannot look up derivative value for non-Tensor argument: $(typeof(inner))")
end

"""
    prepare_values(chart::ChartProperties, metric_data::AbstractMatrix;
                   registry::TensorRegistry=current_registry(),
                   deriv_fn=nothing,
                   tensors::Dict{Symbol,<:AbstractArray}=Dict{Symbol,Array{Float64,0}}()) -> Dict

Build a values dictionary from a metric matrix and optional tensor arrays.

Automatically stores:
- `g_{ab}` from `metric_data`
- `g^{ab}` from `inv(metric_data)` (stored under `:g_inv` or `Symbol(metric, :_inv)`)
- Kronecker delta `δ_{ab}`
- All tensors from `tensors` dict
- If `deriv_fn` provided: `∂_c g_{ab}` and Christoffel symbols `Γ^a_{bc}`

# Example
```julia
chart = define_chart!(reg, :Schw; manifold=:M4, coords=[:t, :r, :θ, :φ])
g = [-f 0 0 0; 0 1/f 0 0; 0 0 r^2 0; 0 0 0 r^2*sin(θ)^2]
vals = prepare_values(chart, g; registry=reg, deriv_fn=sym_deriv)
```
"""
function prepare_values(chart::ChartProperties, metric_data::AbstractMatrix;
                        registry::TensorRegistry=current_registry(),
                        deriv_fn=nothing,
                        tensors::Dict{Symbol,<:AbstractArray}=Dict{Symbol,Array{Float64,0}}())
    dim = length(chart.coords)
    @assert size(metric_data) == (dim, dim) "Metric must be $(dim)x$(dim)"

    values = Dict{Any,Any}()

    # Find metric and delta names from registry
    metric_name = get(registry.metric_cache, chart.manifold, :g)
    delta_name = get(registry.delta_cache, chart.manifold, :δ)

    # Store g_{ab}
    for i in 1:dim, j in 1:dim
        values[(metric_name, [i, j])] = metric_data[i, j]
    end

    # Store g^{ab} (inverse metric)
    ginv = inv(Matrix(metric_data))
    inv_name = Symbol(metric_name, :_inv)
    for i in 1:dim, j in 1:dim
        values[(inv_name, [i, j])] = ginv[i, j]
    end

    # Store Kronecker delta
    for i in 1:dim, j in 1:dim
        values[(delta_name, [i, j])] = (i == j) ? 1 : 0
    end

    # Store user tensors
    for (name, arr) in tensors
        for ci in CartesianIndices(arr)
            values[(name, collect(Int, Tuple(ci)))] = arr[ci]
        end
    end

    # Optionally compute derivative-related values
    if deriv_fn !== nothing
        coords = chart.coords
        # ∂_k g_{ij}
        for i in 1:dim, j in 1:dim, k in 1:dim
            values[(Symbol("∂", metric_name), [k, i, j])] =
                deriv_fn(metric_data[i, j], coords[k])
        end

        # Christoffel Γ^a_{bc} via existing metric_christoffel
        Gamma = metric_christoffel(Matrix(metric_data), ginv, coords;
                                    deriv_fn=deriv_fn)
        for a in 1:dim, b in 1:dim, c in 1:dim
            values[(:Γ, [a, b, c])] = Gamma[a, b, c]
        end
    end

    values
end
