#= Smooth maps between manifolds: pullback, pushforward, Jacobian.

Given a map φ: M → N, the differential dφ is a rank-(1,1) tensor that maps
tangent vectors on M to tangent vectors on N. The pullback φ* maps covariant
tensors on N back to M; the pushforward φ₊ maps contravariant tensors forward.

Design: Jacobians are ordinary registered tensors with cross-manifold vbundles.
pullback/pushforward construct explicit Jacobian contractions. No new AST types.
=#

"""
    MappingProperties

Metadata for a smooth map φ: domain → codomain.

Fields:
- `name`: symbol identifying the map
- `domain`: source manifold
- `codomain`: target manifold
- `jacobian`: name of the Jacobian tensor dφ^i_a (rank-(1,1))
- `inv_jacobian`: name of inverse Jacobian (dφ⁻¹)^a_i, or `nothing`
"""
struct MappingProperties
    name::Symbol
    domain::Symbol
    codomain::Symbol
    jacobian::Symbol
    inv_jacobian::Union{Symbol, Nothing}
end

has_mapping(reg::TensorRegistry, name::Symbol) = haskey(reg.mappings, name)

function get_mapping(reg::TensorRegistry, name::Symbol)
    reg.mappings[name]::MappingProperties
end

"""
    define_mapping!(reg, name; domain, codomain, jacobian_name=:dφ,
                    inv_jacobian_name=nothing) -> MappingProperties

Define a smooth map `name: domain → codomain` and register its Jacobian tensor.

The Jacobian `dφ^i_a` has its upper index on `Tangent_codomain` and lower index
on `Tangent_domain`, representing the differential ∂φ^i/∂x^a.

If `inv_jacobian_name` is given, also registers the inverse Jacobian (dφ⁻¹)^a_i
with reversed vbundle assignments.

# Example
```julia
define_mapping!(reg, :φ; domain=:M, codomain=:N)
# Registers tensor :dφ with rank (1,1)
# dφ^{i}_{a} where i ∈ Tangent_N, a ∈ Tangent_M
```
"""
function define_mapping!(reg::TensorRegistry, name::Symbol;
                          domain::Symbol, codomain::Symbol,
                          jacobian_name::Symbol=Symbol(:d, name),
                          inv_jacobian_name::Union{Symbol, Nothing}=nothing)
    has_manifold(reg, domain) || error("Domain manifold $domain not registered")
    has_manifold(reg, codomain) || error("Codomain manifold $codomain not registered")
    has_mapping(reg, name) && error("Mapping $name already registered")

    # VBundle names for cross-manifold indices
    vb_domain = _tangent_vbundle_name(reg, domain)
    vb_codomain = _tangent_vbundle_name(reg, codomain)

    # Register Jacobian: dφ^i_a (upper index on codomain, lower on domain)
    if !has_tensor(reg, jacobian_name)
        register_tensor!(reg, TensorProperties(
            name=jacobian_name, manifold=domain, rank=(1, 1),
            options=Dict{Symbol,Any}(:is_jacobian => true,
                                     :mapping => name,
                                     :up_vbundle => vb_codomain,
                                     :down_vbundle => vb_domain)))
    end

    # Optionally register inverse Jacobian
    if inv_jacobian_name !== nothing && !has_tensor(reg, inv_jacobian_name)
        register_tensor!(reg, TensorProperties(
            name=inv_jacobian_name, manifold=codomain, rank=(1, 1),
            options=Dict{Symbol,Any}(:is_jacobian => true,
                                     :mapping => name,
                                     :is_inverse => true,
                                     :up_vbundle => vb_domain,
                                     :down_vbundle => vb_codomain)))
    end

    mp = MappingProperties(name, domain, codomain, jacobian_name, inv_jacobian_name)
    reg.mappings[name] = mp
    mp
end

"""Return the tangent bundle name for a manifold (domain-specific if multi-manifold)."""
function _tangent_vbundle_name(reg::TensorRegistry, manifold::Symbol)
    # For single-manifold setups, everything is :Tangent.
    # For multi-manifold, use Tangent_manifold to distinguish.
    n_manifolds = length(reg.manifolds)
    n_manifolds <= 1 ? :Tangent : Symbol(:Tangent_, manifold)
end

"""
    pullback(T::TensorExpr, φ::Symbol; registry=current_registry()) -> TensorExpr

Construct the pullback φ*(T) by contracting each covariant (Down) index of `T`
with the Jacobian dφ^i_a. The result has domain-manifold indices.

For a rank-(0,k) tensor T_{i₁...iₖ}:
    φ*(T)_{a₁...aₖ} = dφ^{i₁}_{a₁} ⋯ dφ^{iₖ}_{aₖ} T_{i₁...iₖ}

# Example
```julia
g = Tensor(:g, [down(:i), down(:j)])
pullback(g, :φ)  # => dφ^{i}_{a} dφ^{j}_{b} g_{ij}  with fresh a,b
```
"""
function pullback(T::TensorExpr, φ::Symbol;
                  registry::TensorRegistry=current_registry())
    mp = get_mapping(registry, φ)
    _pullback_expr(T, mp, registry)
end

function _pullback_expr(T::Tensor, mp::MappingProperties, reg::TensorRegistry)
    down_indices = [idx for idx in T.indices if idx.position == Down]
    isempty(down_indices) && return T  # nothing to pull back

    used = Set(idx.name for idx in T.indices)
    result_expr::TensorExpr = T
    new_free = TIndex[]

    for idx in T.indices
        if idx.position == Down
            # Fresh index on the domain side
            a = fresh_index(used)
            push!(used, a)
            # Jacobian: dφ^{idx.name}_{a}
            J = Tensor(mp.jacobian, [up(idx.name), down(a)])
            result_expr = result_expr * J
            push!(new_free, down(a))
        else
            push!(new_free, idx)
        end
    end

    result_expr
end

# Pullback distributes over sums and products
function _pullback_expr(s::TSum, mp::MappingProperties, reg::TensorRegistry)
    tsum(TensorExpr[_pullback_expr(t, mp, reg) for t in s.terms])
end

function _pullback_expr(p::TProduct, mp::MappingProperties, reg::TensorRegistry)
    # Only pull back tensor factors that belong to the codomain
    tproduct(p.scalar, TensorExpr[_pullback_expr(f, mp, reg) for f in p.factors])
end

_pullback_expr(s::TScalar, ::MappingProperties, ::TensorRegistry) = s

function _pullback_expr(d::TDeriv, mp::MappingProperties, reg::TensorRegistry)
    TDeriv(d.index, _pullback_expr(d.arg, mp, reg), d.covd)
end

"""
    pushforward(T::TensorExpr, φ::Symbol; registry=current_registry()) -> TensorExpr

Construct the pushforward φ₊(T) by contracting each contravariant (Up) index
of `T` with the inverse Jacobian (dφ⁻¹)^a_i.

Requires that `φ` was defined with `inv_jacobian_name`.

For a rank-(k,0) tensor U^{a₁...aₖ}:
    φ₊(U)^{i₁...iₖ} = (dφ⁻¹)^{i₁}_{a₁} ⋯ (dφ⁻¹)^{iₖ}_{aₖ} U^{a₁...aₖ}
"""
function pushforward(T::TensorExpr, φ::Symbol;
                     registry::TensorRegistry=current_registry())
    mp = get_mapping(registry, φ)
    mp.inv_jacobian === nothing &&
        error("pushforward requires inverse Jacobian; define_mapping! with inv_jacobian_name")
    _pushforward_expr(T, mp, registry)
end

function _pushforward_expr(T::Tensor, mp::MappingProperties, reg::TensorRegistry)
    up_indices = [idx for idx in T.indices if idx.position == Up]
    isempty(up_indices) && return T

    used = Set(idx.name for idx in T.indices)
    result_expr::TensorExpr = T

    for idx in T.indices
        if idx.position == Up
            i = fresh_index(used)
            push!(used, i)
            # Inverse Jacobian: (dφ⁻¹)^{i}_{idx.name}
            Jinv = Tensor(mp.inv_jacobian, [up(i), down(idx.name)])
            result_expr = result_expr * Jinv
        end
    end

    result_expr
end

function _pushforward_expr(s::TSum, mp::MappingProperties, reg::TensorRegistry)
    tsum(TensorExpr[_pushforward_expr(t, mp, reg) for t in s.terms])
end

function _pushforward_expr(p::TProduct, mp::MappingProperties, reg::TensorRegistry)
    tproduct(p.scalar, TensorExpr[_pushforward_expr(f, mp, reg) for f in p.factors])
end

_pushforward_expr(s::TScalar, ::MappingProperties, ::TensorRegistry) = s

function _pushforward_expr(d::TDeriv, mp::MappingProperties, reg::TensorRegistry)
    TDeriv(d.index, _pushforward_expr(d.arg, mp, reg), d.covd)
end

"""
    pullback_metric(φ::Symbol, metric::Symbol;
                    registry=current_registry()) -> TensorExpr

Construct the pullback of a metric tensor:
    φ*(g)_{ab} = dφ^i_a dφ^j_b g_{ij}

This is the induced metric on the domain manifold.
"""
function pullback_metric(φ::Symbol, metric::Symbol;
                         registry::TensorRegistry=current_registry())
    mp = get_mapping(registry, φ)
    used = Set{Symbol}()
    i = fresh_index(used); push!(used, i)
    j = fresh_index(used); push!(used, j)
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)

    g = Tensor(metric, [down(i), down(j)])
    J1 = Tensor(mp.jacobian, [up(i), down(a)])
    J2 = Tensor(mp.jacobian, [up(j), down(b)])

    J1 * J2 * g
end
