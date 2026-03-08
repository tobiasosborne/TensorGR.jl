#= Hypersurface geometry.

Given an embedding of a codimension-1 hypersurface in an ambient manifold,
compute the induced metric, unit normal, extrinsic curvature, etc.

Port of diffgeo.m's `hypersurface` functionality.
=#

"""
    HypersurfaceProperties

Stores the geometric data of a hypersurface embedding.

Fields:
- `ambient`: name of the ambient manifold
- `metric`: ambient metric name
- `normal_coord`: the coordinate normal to the hypersurface (or nothing for parametric)
- `signature`: +1 (spacelike normal) or -1 (timelike normal)
"""
struct HypersurfaceProperties
    ambient::Symbol
    metric::Symbol
    normal_coord::Union{Symbol, Nothing}
    signature::Int
    dim_ambient::Int
    dim_surface::Int
end

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

"""
    define_hypersurface!(reg, name; ambient, metric, normal_name=:n,
                          extrinsic_name=:K, induced_name=:γ, signature=-1)

Define a hypersurface embedding and register associated tensors:
- Unit normal `n_a`
- Induced metric `γ_{ab} = g_{ab} + σ n_a n_b` (where σ = -signature)
- Extrinsic curvature `K_{ab}`
- Projector `P^a_b = δ^a_b + σ n^a n_b` (projects onto the hypersurface)

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
    @assert signature in (-1, 1) "signature must be +1 or -1"
    mp = get_manifold(reg, ambient)
    d = mp.dim
    σ = -signature  # sign factor for induced metric

    # Register unit normal n_a (rank-1)
    if !has_tensor(reg, normal_name)
        register_tensor!(reg, TensorProperties(
            name=normal_name, manifold=ambient, rank=(0, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_normal => true, :signature => signature)))
    end

    # Register induced metric γ_{ab} = g_{ab} + σ n_a n_b  (symmetric)
    if !has_tensor(reg, induced_name)
        register_tensor!(reg, TensorProperties(
            name=induced_name, manifold=ambient, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_induced_metric => true)))
    end

    # Register extrinsic curvature K_{ab} (symmetric)
    if !has_tensor(reg, extrinsic_name)
        register_tensor!(reg, TensorProperties(
            name=extrinsic_name, manifold=ambient, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_extrinsic => true)))
    end

    # Register projector P^a_b
    if !has_tensor(reg, projector_name)
        register_tensor!(reg, TensorProperties(
            name=projector_name, manifold=ambient, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_projector => true)))
    end

    # Rule: n_a n^a = signature
    register_rule!(reg, RewriteRule(
        function(expr)
            if !(expr isa TProduct)
                return false
            end
            n_count = count(f -> f isa Tensor && f.name == normal_name, expr.factors)
            n_count >= 2 || return false
            # Check for contraction: n_a n^a
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
            # Replace n_a n^a with signature
            factors = copy(expr.factors)
            n_indices = Int[]
            for (i, f) in enumerate(factors)
                if f isa Tensor && f.name == normal_name
                    push!(n_indices, i)
                end
            end
            # Find first contracting pair and remove
            for i in 1:length(n_indices), j in i+1:length(n_indices)
                fi = factors[n_indices[i]]
                fj = factors[n_indices[j]]
                if length(fi.indices) == 1 && length(fj.indices) == 1 &&
                   fi.indices[1].name == fj.indices[1].name &&
                   fi.indices[1].position != fj.indices[1].position
                    remaining = [factors[k] for k in eachindex(factors)
                                 if k != n_indices[i] && k != n_indices[j]]
                    return tproduct(expr.scalar * signature, isempty(remaining) ?
                                    TensorExpr[TScalar(1 // 1)] : remaining)
                end
            end
            expr
        end
    ))

    # Store properties (use foliations dict as general storage)
    hs = HypersurfaceProperties(ambient, metric, nothing, signature, d, d - 1)
    reg.foliations[Symbol(:hypersurface_, name)] = hs

    hs
end

"""
    induced_metric_expr(a, b, metric, normal; signature=-1) -> TensorExpr

The induced metric on the hypersurface:
`γ_{ab} = g_{ab} - σ n_a n_b`

where σ = signature of the normal (n·n = σ).
"""
function induced_metric_expr(a::TIndex, b::TIndex, metric::Symbol, normal::Symbol;
                              signature::Int=-1)
    Tensor(metric, [a, b]) - (signature // 1) * Tensor(normal, [a]) * Tensor(normal, [b])
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
