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

    # Register torsion tensor if not torsion-free
    torsion_sym = Symbol(:T, name)
    if !torsion_free && !has_tensor(reg, torsion_sym)
        register_tensor!(reg, TensorProperties(
            name=torsion_sym, manifold=manifold, rank=(1, 2),
            symmetries=Any[AntiSymmetric(2, 3)],
            options=Dict{Symbol,Any}(:is_torsion => true,
                                     :covd => name)))
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
    # Fires for this CovD or for untagged (:partial) derivatives acting on its metric.
    # When multiple CovDs exist, only the tagged CovD's rule fires for its metric.
    if metric_compatible
        local _covd_name = name
        local _metric_name = metric
        register_rule!(reg, RewriteRule(
            function(expr)
                expr isa TDeriv || return false
                # Accept this CovD name or untagged :partial (backward compat)
                (expr.covd == _covd_name || expr.covd == :partial) || return false
                inner = expr.arg
                inner isa Tensor || return false
                inner.name == _metric_name || return false
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
    props.is_covd || error("$name is not a CovD")
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
    TDeriv(expr.index, inner, expr.covd)
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
of Christoffel symbols. ∇₁ T = ∇₂ T + (Γ₁ - Γ₂) terms.
"""
function change_covd(expr::TensorExpr, from::Symbol, to::Symbol;
                     registry::TensorRegistry=current_registry())
    with_registry(registry) do
        _change_covd_walk(expr, from, to, registry)
    end
end

function _change_covd_walk(expr::Tensor, ::Symbol, ::Symbol, ::TensorRegistry)
    expr
end
function _change_covd_walk(expr::TScalar, ::Symbol, ::Symbol, ::TensorRegistry)
    expr
end
function _change_covd_walk(expr::TSum, from::Symbol, to::Symbol, reg::TensorRegistry)
    tsum(TensorExpr[_change_covd_walk(t, from, to, reg) for t in expr.terms])
end
function _change_covd_walk(expr::TProduct, from::Symbol, to::Symbol, reg::TensorRegistry)
    TProduct(expr.scalar, TensorExpr[_change_covd_walk(f, from, to, reg) for f in expr.factors])
end
function _change_covd_walk(expr::TDeriv, from::Symbol, to::Symbol, reg::TensorRegistry)
    inner = _change_covd_walk(expr.arg, from, to, reg)

    if inner isa Tensor
        # ∇₁_a T = ∇₂_a T + (Γ₁ - Γ₂) terms
        from_props = get_covd(reg, from)
        to_props = get_covd(reg, to)
        Γ1 = from_props.christoffel
        Γ2 = to_props.christoffel

        # Start with ∇₂_a T (keep as TDeriv with target CovD)
        result = TDeriv(expr.index, inner, to)

        used = Set{Symbol}()
        push!(used, expr.index.name)
        for idx in inner.indices
            push!(used, idx.name)
        end

        # Add difference tensor terms for each index
        for (i, tidx) in enumerate(inner.indices)
            dummy = fresh_index(used)
            push!(used, dummy)

            if tidx.position == Up
                # +(Γ₁ - Γ₂)^{b_i}_{a d} T^{...d...}
                diff1 = Tensor(Γ1, [tidx, expr.index, down(dummy)])
                diff2 = Tensor(Γ2, [tidx, expr.index, down(dummy)])
                new_indices = copy(inner.indices)
                new_indices[i] = up(dummy)
                t_mod = Tensor(inner.name, new_indices)
                result = result + (diff1 - diff2) * t_mod
            else
                # -(Γ₁ - Γ₂)^{d}_{a b_i} T^{...}_{...d...}
                diff1 = Tensor(Γ1, [up(dummy), expr.index, tidx])
                diff2 = Tensor(Γ2, [up(dummy), expr.index, tidx])
                new_indices = copy(inner.indices)
                new_indices[i] = down(dummy)
                t_mod = Tensor(inner.name, new_indices)
                result = result - (diff1 - diff2) * t_mod
            end
        end

        return result
    end

    TDeriv(expr.index, inner, expr.covd)
end

"""
    christoffel_to_grad_metric(christoffel::Symbol, metric::Symbol,
                                a::TIndex, b::TIndex, c::TIndex) -> TensorExpr

Express Γ^a_{bc} = (1/2) g^{ad} (∂_b g_{cd} + ∂_c g_{bd} - ∂_d g_{bc}).
"""
function christoffel_to_grad_metric(metric::Symbol,
                                     a::TIndex, b::TIndex, c::TIndex)
    @assert a.position == Up
    @assert b.position == Down && c.position == Down

    used = Set{Symbol}([a.name, b.name, c.name])
    d = fresh_index(used)

    g_inv = Tensor(metric, [a, up(d)])
    g_cd = Tensor(metric, [c, down(d)])  # these get differentiated
    g_bd = Tensor(metric, [b, down(d)])
    g_bc = Tensor(metric, [b, c])

    (1 // 2) * g_inv * (TDeriv(b, Tensor(metric, [c, down(d)])) +
                          TDeriv(c, Tensor(metric, [b, down(d)])) -
                          TDeriv(down(d), g_bc))
end

"""
    grad_metric_to_christoffel(metric::Symbol, christoffel::Symbol,
                                a::TIndex, b::TIndex, c::TIndex) -> TensorExpr

Express ∂_a g_{bc} = Γ^d_{ab} g_{dc} + Γ^d_{ac} g_{db}.
(Metric compatibility: ∇_a g_{bc} = 0.)
"""
function grad_metric_to_christoffel(metric::Symbol, christoffel::Symbol,
                                     a::TIndex, b::TIndex, c::TIndex)
    @assert a.position == Down
    @assert b.position == Down && c.position == Down

    used = Set{Symbol}([a.name, b.name, c.name])
    d = fresh_index(used)

    Tensor(christoffel, [up(d), a, b]) * Tensor(metric, [down(d), c]) +
    Tensor(christoffel, [up(d), a, c]) * Tensor(metric, [down(d), b])
end
