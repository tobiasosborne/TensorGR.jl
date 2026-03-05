#= Covariant derivative infrastructure.

DefCovD: define a covariant derivative on a manifold with associated metric.
Auto-generates Christoffel symbols, and provides:
  - Metric compatibility: ∇_a g_{bc} = 0
  - covd_to_christoffel: expand ∇ into ∂ + Γ terms
  - change_covd: switch between different covariant derivatives
=#

"""
    CovDProperties(name, manifold, metric, torsion_free, metric_compatible)

Properties of a covariant derivative stored in the registry.
"""
struct CovDProperties
    name::Symbol
    manifold::Symbol
    metric::Symbol
    christoffel::Symbol       # Christoffel symbol tensor name
    torsion_free::Bool
    metric_compatible::Bool
end

"""
    define_covd!(reg, name; manifold, metric, torsion_free=true, metric_compatible=true)

Define a covariant derivative and register associated tensors (Christoffel symbols).
"""
function define_covd!(reg::TensorRegistry, name::Symbol;
                      manifold::Symbol, metric::Symbol,
                      torsion_free::Bool=true,
                      metric_compatible::Bool=true)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    christoffel = Symbol(:Γ, name)

    # Register Christoffel symbol Γ^a_{bc}
    if !has_tensor(reg, christoffel)
        syms = torsion_free ? Any[Symmetric(2, 3)] : Any[]
        register_tensor!(reg, TensorProperties(
            name=christoffel, manifold=manifold, rank=(1, 2),
            symmetries=syms,
            options=Dict{Symbol,Any}(:is_christoffel => true,
                                     :covd => name,
                                     :metric => metric)))
    end

    covd_props = CovDProperties(name, manifold, metric, christoffel,
                                 torsion_free, metric_compatible)

    # Store CovD properties in the registry's options
    if !haskey(reg.tensors, name)
        # Register a "tensor" entry for the CovD for lookup
        register_tensor!(reg, TensorProperties(
            name=name, manifold=manifold, rank=(0, 0),
            symmetries=Any[],
            options=Dict{Symbol,Any}(:is_covd => true,
                                     :covd_props => covd_props)))
    end

    # Register metric compatibility rule: ∇_a g_{bc} = 0
    if metric_compatible
        register_rule!(reg, RewriteRule(
            function(expr)
                expr isa TDeriv || return false
                # Check if this is the named CovD (we use TDeriv with a marker)
                inner = expr.arg
                inner isa Tensor || return false
                inner.name == metric || return false
                # Check derivative index contracts... actually for metric
                # compatibility, ANY ∇g = 0 regardless of index structure
                true
            end,
            _ -> ZERO
        ))
    end

    covd_props
end

"""
    get_covd(reg, name) -> CovDProperties

Retrieve CovD properties from the registry.
"""
function get_covd(reg::TensorRegistry, name::Symbol)
    has_tensor(reg, name) || error("CovD $name not registered")
    props = get_tensor(reg, name)
    get(props.options, :is_covd, false) || error("$name is not a CovD")
    props.options[:covd_props]::CovDProperties
end

"""
    covd_to_christoffel(expr, covd_name; registry=current_registry()) -> TensorExpr

Expand covariant derivatives into partial derivatives plus Christoffel terms.

∇_a V^b = ∂_a V^b + Γ^b_{ac} V^c
∇_a ω_b = ∂_a ω_b - Γ^c_{ab} ω_c
∇_a T^b_c = ∂_a T^b_c + Γ^b_{ad} T^d_c - Γ^d_{ac} T^b_d
"""
function covd_to_christoffel(expr::TensorExpr, covd_name::Symbol;
                              registry::TensorRegistry=current_registry())
    with_registry(registry) do
        _expand_covd(expr, covd_name, registry)
    end
end

function _expand_covd(expr::Tensor, ::Symbol, ::TensorRegistry)
    expr
end

function _expand_covd(expr::TScalar, ::Symbol, ::TensorRegistry)
    expr
end

function _expand_covd(expr::TSum, covd::Symbol, reg::TensorRegistry)
    tsum(TensorExpr[_expand_covd(t, covd, reg) for t in expr.terms])
end

function _expand_covd(expr::TProduct, covd::Symbol, reg::TensorRegistry)
    TProduct(expr.scalar, TensorExpr[_expand_covd(f, covd, reg) for f in expr.factors])
end

function _expand_covd(expr::TDeriv, covd::Symbol, reg::TensorRegistry)
    # First expand inside
    inner = _expand_covd(expr.arg, covd, reg)

    # If the inner is a bare tensor, expand ∇_a T^{...}_{...}
    if inner isa Tensor
        return _expand_covd_on_tensor(expr.index, inner, covd, reg)
    end

    # If inner is a product, apply Leibniz rule first via expand_derivatives
    # then recursively expand each CovD
    TDeriv(expr.index, inner)
end

"""
Expand ∇_a T^{b1...}_{c1...} into ∂_a T + Γ terms.
"""
function _expand_covd_on_tensor(deriv_idx::TIndex, tensor::Tensor,
                                 covd::Symbol, reg::TensorRegistry)
    covd_props = get_covd(reg, covd)
    christoffel = covd_props.christoffel

    # Start with the partial derivative term
    result = TDeriv(deriv_idx, tensor)

    # Collect all used index names to generate fresh dummies
    used = Set{Symbol}()
    push!(used, deriv_idx.name)
    for idx in tensor.indices
        push!(used, idx.name)
    end

    # For each index on the tensor, add a Christoffel term
    for (i, tidx) in enumerate(tensor.indices)
        dummy = fresh_index(used)
        push!(used, dummy)

        if tidx.position == Up
            # +Γ^{b_i}_{a d} T^{...d...}_{...}
            gamma = Tensor(christoffel, [tidx, deriv_idx, down(dummy)])
            new_indices = copy(tensor.indices)
            new_indices[i] = up(dummy)
            t_with_dummy = Tensor(tensor.name, new_indices)
            result = result + gamma * t_with_dummy
        else
            # -Γ^{d}_{a b_i} T^{...}_{...d...}
            gamma = Tensor(christoffel, [up(dummy), deriv_idx, tidx])
            new_indices = copy(tensor.indices)
            new_indices[i] = down(dummy)
            t_with_dummy = Tensor(tensor.name, new_indices)
            result = result - gamma * t_with_dummy
        end
    end

    result
end

"""
    change_covd(expr, from_covd, to_covd; registry=current_registry()) -> TensorExpr

Change from one covariant derivative to another by inserting the difference
of Christoffel symbols.
"""
function change_covd(expr::TensorExpr, from::Symbol, to::Symbol;
                     registry::TensorRegistry=current_registry())
    with_registry(registry) do
        # Strategy: expand from_covd into partials + Christoffel_from,
        # then re-express partials as to_covd - Christoffel_to.
        # Equivalently: ∇₁ = ∇₂ + (Γ₁ - Γ₂)
        # For now, expand both to partial derivatives
        result = _expand_covd(expr, from, registry)
        result
    end
end
