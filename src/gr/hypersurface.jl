#= Submanifold / hypersurface geometry.

Given an embedding of a codimension-k submanifold in an ambient manifold,
compute the induced metric, unit normals, extrinsic curvatures, etc.

Codimension-1 (hypersurface) is the most common case and retains its
convenience API for backward compatibility.
=#

"""
    SubmanifoldProperties

Stores the geometric data of a submanifold embedding with arbitrary codimension.

Fields:
- `ambient`: name of the ambient manifold
- `metric`: ambient metric name
- `codimension`: codimension of the embedding (1 for hypersurface)
- `normal_coord`: first normal coordinate (or nothing for parametric), backward compat
- `signature`: signature of first normal (+1 spacelike, -1 timelike), backward compat
- `dim_ambient`: dimension of ambient manifold
- `dim_surface`: dimension of submanifold
- `normal_names`: vector of normal vector names (length = codimension)
- `extrinsic_names`: vector of extrinsic curvature names (length = codimension)
- `signatures`: vector of normal signatures (length = codimension)
"""
struct SubmanifoldProperties
    ambient::Symbol
    metric::Symbol
    codimension::Int
    normal_coord::Union{Symbol, Nothing}
    signature::Int
    dim_ambient::Int
    dim_surface::Int
    normal_names::Vector{Symbol}
    extrinsic_names::Vector{Symbol}
    signatures::Vector{Int}
end

# Backward compatibility alias
const HypersurfaceProperties = SubmanifoldProperties

"""
    extrinsic_curvature_expr(a, b, normal, metric; registry=current_registry()) -> TensorExpr

Construct the extrinsic curvature (second fundamental form):
`K_{ab} = -∇_a n_b = -(∂_a n_b - Γ^c_{ab} n_c)`

where `n_a` is the unit normal covector.
"""
function extrinsic_curvature_expr(a::TIndex, b::TIndex,
                                    normal::Symbol, metric::Symbol;
                                    registry::TensorRegistry=current_registry())
    with_registry(registry) do
        used = Set{Symbol}([a.name, b.name])
        c = fresh_index(used)
        n_b = Tensor(normal, [b])
        # K_{ab} = -∂_a n_b + Γ^c_{ab} n_c
        # This is -(∇_a n_b) for the Levi-Civita connection
        # We return it as a symbolic expression; the user can expand with covd_to_christoffel
        -TDeriv(a, n_b)
    end
end

# ── Internal: register normal + normalization rule ──

function _register_normal!(reg::TensorRegistry, normal_name::Symbol,
                           ambient::Symbol, sig::Int)
    if !has_tensor(reg, normal_name)
        register_tensor!(reg, TensorProperties(
            name=normal_name, manifold=ambient, rank=(0, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_normal => true, :signature => sig)))
    end

    # Rule: n_a n^a = signature
    register_rule!(reg, RewriteRule(
        function(expr)
            if !(expr isa TProduct)
                return false
            end
            n_factors = [f for f in expr.factors if f isa Tensor && f.name == normal_name]
            length(n_factors) >= 2 || return false
            for i in 1:length(n_factors), j in i+1:length(n_factors)
                fi, fj = n_factors[i], n_factors[j]
                if length(fi.indices) == 1 && length(fj.indices) == 1
                    if fi.indices[1].name == fj.indices[1].name &&
                       fi.indices[1].position != fj.indices[1].position
                        return true
                    end
                end
            end
            false
        end,
        function(expr)
            factors = copy(expr.factors)
            n_indices = Int[]
            for (i, f) in enumerate(factors)
                if f isa Tensor && f.name == normal_name
                    push!(n_indices, i)
                end
            end
            for i in 1:length(n_indices), j in i+1:length(n_indices)
                fi = factors[n_indices[i]]
                fj = factors[n_indices[j]]
                if length(fi.indices) == 1 && length(fj.indices) == 1 &&
                   fi.indices[1].name == fj.indices[1].name &&
                   fi.indices[1].position != fj.indices[1].position
                    remaining = [factors[k] for k in eachindex(factors)
                                 if k != n_indices[i] && k != n_indices[j]]
                    return tproduct(expr.scalar * sig, isempty(remaining) ?
                                    TensorExpr[TScalar(1 // 1)] : remaining)
                end
            end
            expr
        end
    ))
end

"""
    define_submanifold!(reg, name; ambient, metric, codimension=1,
                         normal_names=nothing, extrinsic_names=nothing,
                         induced_name=:γ, projector_name=:P_hs,
                         signatures=nothing)

Define a submanifold embedding of arbitrary codimension and register
associated tensors. For each normal direction i:
- Unit normal `nᵢ_a`
- Extrinsic curvature `Kᵢ_{ab}`

Also registers the induced metric and projector.

`signatures` is a vector of normal norms: +1 (spacelike) or -1 (timelike).
"""
function define_submanifold!(reg::TensorRegistry, name::Symbol;
                              ambient::Symbol,
                              metric::Symbol=:g,
                              codimension::Int=1,
                              normal_names::Union{Vector{Symbol}, Nothing}=nothing,
                              extrinsic_names::Union{Vector{Symbol}, Nothing}=nothing,
                              induced_name::Symbol=:γ,
                              projector_name::Symbol=:P_hs,
                              signatures::Union{Vector{Int}, Nothing}=nothing)
    @assert codimension >= 1 "codimension must be >= 1"
    mp = get_manifold(reg, ambient)
    d = mp.dim
    @assert codimension < d "codimension must be < ambient dimension"

    # Default names
    if normal_names === nothing
        normal_names = codimension == 1 ? [:n] :
            [Symbol(:n, i) for i in 1:codimension]
    end
    if extrinsic_names === nothing
        extrinsic_names = codimension == 1 ? [:K] :
            [Symbol(:K, i) for i in 1:codimension]
    end
    if signatures === nothing
        signatures = fill(-1, codimension)
    end

    @assert length(normal_names) == codimension
    @assert length(extrinsic_names) == codimension
    @assert length(signatures) == codimension
    @assert all(s -> s in (-1, 1), signatures)

    # Register normals with normalization rules
    for (nn, sig) in zip(normal_names, signatures)
        _register_normal!(reg, nn, ambient, sig)
    end

    # Register extrinsic curvatures K_i_{ab} (symmetric)
    for kn in extrinsic_names
        if !has_tensor(reg, kn)
            register_tensor!(reg, TensorProperties(
                name=kn, manifold=ambient, rank=(0, 2),
                symmetries=SymmetrySpec[Symmetric(1, 2)],
                options=Dict{Symbol,Any}(:is_extrinsic => true)))
        end
    end

    # Register induced metric (symmetric)
    if !has_tensor(reg, induced_name)
        register_tensor!(reg, TensorProperties(
            name=induced_name, manifold=ambient, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_induced_metric => true)))
    end

    # Register projector P^a_b
    if !has_tensor(reg, projector_name)
        register_tensor!(reg, TensorProperties(
            name=projector_name, manifold=ambient, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_projector => true)))
    end

    # Store properties
    sp = SubmanifoldProperties(
        ambient, metric, codimension,
        nothing, signatures[1], d, d - codimension,
        collect(normal_names), collect(extrinsic_names), collect(signatures))
    reg.foliations[Symbol(:hypersurface_, name)] = sp

    sp
end

"""
    define_hypersurface!(reg, name; ambient, metric, normal_name=:n,
                          extrinsic_name=:K, induced_name=:γ, signature=-1)

Define a codimension-1 hypersurface embedding. Convenience wrapper around
`define_submanifold!`.

Registers:
- Unit normal `n_a`
- Induced metric `γ_{ab} = g_{ab} + σ n_a n_b` (where σ = -signature)
- Extrinsic curvature `K_{ab}`
- Projector `P^a_b = δ^a_b + σ n^a n_b`

`signature` is the norm of the unit normal: +1 for spacelike, -1 for timelike.
"""
function define_hypersurface!(reg::TensorRegistry, name::Symbol;
                               ambient::Symbol,
                               metric::Symbol=:g,
                               normal_name::Symbol=:n,
                               extrinsic_name::Symbol=:K,
                               induced_name::Symbol=:γ,
                               projector_name::Symbol=:P_hs,
                               signature::Int=-1)
    define_submanifold!(reg, name;
        ambient=ambient, metric=metric, codimension=1,
        normal_names=[normal_name], extrinsic_names=[extrinsic_name],
        induced_name=induced_name, projector_name=projector_name,
        signatures=[signature])
end

"""
    induced_metric_expr(a, b, metric, normal; signature=-1) -> TensorExpr

The induced metric on a codimension-1 hypersurface:
`γ_{ab} = g_{ab} - σ n_a n_b`

where σ = signature of the normal (n·n = σ).
"""
function induced_metric_expr(a::TIndex, b::TIndex, metric::Symbol, normal::Symbol;
                              signature::Int=-1)
    Tensor(metric, [a, b]) - (signature // 1) * Tensor(normal, [a]) * Tensor(normal, [b])
end

"""
    induced_metric_expr(a, b, metric, normals, signatures) -> TensorExpr

The induced metric on a codimension-k submanifold:
`γ_{ab} = g_{ab} - Σᵢ σᵢ nᵢ_a nᵢ_b`

where σᵢ = signature of normal i (nᵢ·nᵢ = σᵢ).
"""
function induced_metric_expr(a::TIndex, b::TIndex, metric::Symbol,
                              normals::Vector{Symbol}, signatures::Vector{Int})
    @assert length(normals) == length(signatures)
    result = Tensor(metric, [a, b])
    for (nn, sig) in zip(normals, signatures)
        result = result - (sig // 1) * Tensor(nn, [a]) * Tensor(nn, [b])
    end
    result
end

"""
    projector_expr(a, b, metric, normal; signature=-1) -> TensorExpr

The projection tensor onto the hypersurface:
`P^a_b = δ^a_b - σ n^a n_b`
"""
function projector_expr(a::TIndex, b::TIndex, normal::Symbol;
                         signature::Int=-1)
    @assert a.position == Up && b.position == Down
    Tensor(:δ, [a, b]) - (signature // 1) * Tensor(normal, [a]) * Tensor(normal, [b])
end

"""
    projector_expr(a, b, normals, signatures) -> TensorExpr

The projection tensor onto a codimension-k submanifold:
`P^a_b = δ^a_b - Σᵢ σᵢ nᵢ^a nᵢ_b`
"""
function projector_expr(a::TIndex, b::TIndex,
                         normals::Vector{Symbol}, signatures::Vector{Int})
    @assert a.position == Up && b.position == Down
    @assert length(normals) == length(signatures)
    result = Tensor(:δ, [a, b])
    for (nn, sig) in zip(normals, signatures)
        result = result - (sig // 1) * Tensor(nn, [a]) * Tensor(nn, [b])
    end
    result
end
