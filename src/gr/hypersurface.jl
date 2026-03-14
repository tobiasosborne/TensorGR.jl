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

# ── Gauss-Codazzi relations ──────────────────────────────────────────

"""
    gauss_equation(a, b, c, d; Riem=:Riem, K=:K, signature=-1) -> TensorExpr

The Gauss equation relating the ambient Riemann tensor projected onto the
hypersurface to the extrinsic curvature:

    ³R_{abcd} = R_{abcd} + σ (K_{ac} K_{bd} - K_{ad} K_{bc})

where σ = signature of the unit normal (n·n = σ). All indices are tangent
to the hypersurface. Returns the right-hand side (ambient Riemann + K²
terms) so that the intrinsic Riemann can be replaced by this expression.
"""
function gauss_equation(a::TIndex, b::TIndex, c::TIndex, d::TIndex;
                        Riem::Symbol=:Riem, K::Symbol=:K, signature::Int=-1)
    s = signature // 1
    # Build a flat 3-term sum: Riem_{abcd} + σ K_{ac}K_{bd} - σ K_{ad}K_{bc}
    tsum(TensorExpr[
        Tensor(Riem, [a, b, c, d]),
        tproduct(s, TensorExpr[Tensor(K, [a, c]), Tensor(K, [b, d])]),
        tproduct(-s, TensorExpr[Tensor(K, [a, d]), Tensor(K, [b, c])])
    ])
end

"""
    codazzi_equation(a, b, c; Riem=:Riem, K=:K, normal=:n, signature=-1) -> TensorExpr

The Codazzi(-Mainardi) equation relating the tangential derivative of the
extrinsic curvature to the normal projection of the ambient Riemann tensor:

    D_a K_{bc} - D_b K_{ac} = σ R_{dabc} n^d

where σ = signature of the unit normal and D is the induced connection on
the hypersurface. Returns the right-hand side (Riemann contracted with the
normal) so that the antisymmetric CovD-of-K pattern can be replaced.

The dummy index for the normal contraction is chosen automatically to avoid
clashes with a, b, c.
"""
function codazzi_equation(a::TIndex, b::TIndex, c::TIndex;
                          Riem::Symbol=:Riem, K::Symbol=:K,
                          normal::Symbol=:n, signature::Int=-1)
    used = Set{Symbol}([a.name, b.name, c.name])
    d = fresh_index(used)
    s = signature // 1
    s * Tensor(Riem, [down(d), a, b, c]) * Tensor(normal, [up(d)])
end

"""
    gauss_codazzi_rules(; Riem=:Riem, K=:K, normal=:n, signature=-1,
                          intrinsic_Riem=:Riem3) -> Vector{RewriteRule}

Create rewrite rules implementing the Gauss and Codazzi equations for a
codimension-1 hypersurface embedding.

Returns two rules:
1. **Gauss**: replaces the intrinsic Riemann `Riem3_{abcd}` with the ambient
   Riemann plus extrinsic curvature terms.
2. **Codazzi**: replaces `Riem_{abcd} n^a` (Riemann contracted with the
   normal in the first slot) with covariant derivatives of K.

The Codazzi rule is a functional rule that detects a TProduct containing
`Riem_{abcd}` contracted with `n^e` on the first index, and rewrites it to
`σ (D_c K_{bd} - D_d K_{bc})` using the specified covariant derivative.
"""
function gauss_codazzi_rules(; Riem::Symbol=:Riem, K::Symbol=:K,
                               normal::Symbol=:n, signature::Int=-1,
                               intrinsic_Riem::Symbol=:Riem3,
                               covd::Symbol=:partial)
    rules = RewriteRule[]
    s = signature // 1

    # ── Rule 1: Gauss equation ──
    # Riem3_{a_ b_ c_ d_} → Riem_{a_ b_ c_ d_} + σ(K_{a_ c_} K_{b_ d_} - K_{a_ d_} K_{b_ c_})
    gauss_lhs = Tensor(intrinsic_Riem, [down(:a_), down(:b_), down(:c_), down(:d_)])
    gauss_rhs = tsum(TensorExpr[
        Tensor(Riem, [down(:a_), down(:b_), down(:c_), down(:d_)]),
        tproduct(s, TensorExpr[Tensor(K, [down(:a_), down(:c_)]),
                                Tensor(K, [down(:b_), down(:d_)])]),
        tproduct(-s, TensorExpr[Tensor(K, [down(:a_), down(:d_)]),
                                 Tensor(K, [down(:b_), down(:c_)])])
    ])
    push!(rules, RewriteRule(gauss_lhs, gauss_rhs))

    # ── Rule 2: Codazzi equation ──
    # Detect Riem_{abcd} * n^e where e contracts with the first Riemann index.
    # Rewrite: σ Riem_{eabc} n^e → D_a K_{bc} - D_b K_{ac}
    push!(rules, RewriteRule(
        function(expr)
            expr isa TProduct || return false
            riem_idx = nothing
            norm_idx = nothing
            for (i, f) in enumerate(expr.factors)
                if f isa Tensor && f.name == Riem && length(f.indices) == 4
                    riem_idx = i
                elseif f isa Tensor && f.name == normal && length(f.indices) == 1
                    norm_idx = i
                end
            end
            (riem_idx === nothing || norm_idx === nothing) && return false
            riem_t = expr.factors[riem_idx]::Tensor
            norm_t = expr.factors[norm_idx]::Tensor
            # Check contraction: normal index matches first Riemann index
            ni = norm_t.indices[1]
            ri = riem_t.indices[1]
            ni.name == ri.name && ni.position != ri.position
        end,
        function(expr)
            riem_idx = nothing
            norm_idx = nothing
            for (i, f) in enumerate(expr.factors)
                if f isa Tensor && f.name == Riem && length(f.indices) == 4
                    riem_idx = i
                elseif f isa Tensor && f.name == normal && length(f.indices) == 1
                    norm_idx = i
                end
            end
            riem_t = expr.factors[riem_idx]::Tensor
            # Free indices: slots 2,3,4 of Riemann (slot 1 is contracted with n)
            idx_a = riem_t.indices[2]
            idx_b = riem_t.indices[3]
            idx_c = riem_t.indices[4]
            # Codazzi: σ R_{eabc} n^e = D_a K_{bc} - D_b K_{ac}
            codazzi_val = TDeriv(idx_a, Tensor(K, [idx_b, idx_c]), covd) -
                          TDeriv(idx_b, Tensor(K, [idx_a, idx_c]), covd)
            # Carry through the scalar and any remaining factors
            remaining = [expr.factors[k] for k in eachindex(expr.factors)
                         if k != riem_idx && k != norm_idx]
            result = (expr.scalar * (1 // s)) * codazzi_val
            if !isempty(remaining)
                result = tproduct(1 // 1, TensorExpr[result; remaining])
            end
            result
        end
    ))

    rules
end

# ── GHY boundary term ───────────────────────────────────────────────

"""
    ghy_boundary_term(reg, hypersurface::Symbol) -> TensorExpr

Return the Gibbons-Hawking-York boundary term as an abstract tensor expression:

    S_GHY = 2 K

where `K = g^{ab} K_{ab}` is the trace of the extrinsic curvature of the
named hypersurface.  Adding this to the Einstein-Hilbert action makes the
Dirichlet variational problem well-posed.

The hypersurface must have been registered with `define_hypersurface!` or
`define_submanifold!` (codimension 1).

# Example
```julia
reg = TensorRegistry()
@manifold M4 dim=4 metric=g
define_hypersurface!(reg, :Sigma; ambient=:M4, metric=:g, signature=-1)
S_ghy = ghy_boundary_term(reg, :Sigma)
```
"""
function ghy_boundary_term(reg::TensorRegistry, hypersurface::Symbol)
    key = Symbol(:hypersurface_, hypersurface)
    haskey(reg.foliations, key) ||
        error("No hypersurface ':$hypersurface' found in registry. " *
              "Call define_hypersurface! first.")
    sp = reg.foliations[key]
    sp.codimension == 1 ||
        error("GHY boundary term is defined only for codimension-1 " *
              "hypersurfaces (got codimension $(sp.codimension)).")

    K_name = sp.extrinsic_names[1]
    metric = sp.metric

    # K = g^{ab} K_{ab}  (trace of extrinsic curvature)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)

    K_trace = tproduct(1 // 1, TensorExpr[
        Tensor(metric, [up(a), up(b)]),
        Tensor(K_name, [down(a), down(b)])
    ])

    # S_GHY = 2 K
    tproduct(2 // 1, TensorExpr[K_trace])
end

# ── Integration by parts with boundary term ──────────────────────────

"""
    ibp_with_boundary(expr::TensorExpr, field::Symbol;
                      registry::TensorRegistry=current_registry()) -> (bulk, boundary)

Integration by parts that preserves the boundary term.

Like `ibp_product`, this moves all derivatives off tensors named `field` in a
product.  Instead of discarding the surface contribution it returns a
`(bulk, boundary)` tuple where:

- **bulk**: the IBP-transferred expression (same as `ibp_product`)
- **boundary**: the total-derivative surface contribution that `ibp_product`
  would discard.  Explicitly, for a factor `d_a1...d_an(field) * rest`:

      expr = bulk + boundary

  so `boundary = expr - bulk` (under the implicit integral sign).

For expressions that are not products, or products with no derivatives on
`field`, the result is trivially `(expr, 0)`.

# Example
```julia
# d_a(phi) T^a  -->  bulk = -phi d_a(T^a),  boundary = expr - bulk
phi = Tensor(:phi, TIndex[])
T   = Tensor(:T, [up(:a)])
expr = tproduct(1//1, TensorExpr[TDeriv(down(:a), phi), T])
bulk, bdry = ibp_with_boundary(expr, :phi)
```
"""
ibp_with_boundary(t::Tensor, ::Symbol;
                  registry::TensorRegistry=current_registry()) = (t, ZERO)
ibp_with_boundary(s::TScalar, ::Symbol;
                  registry::TensorRegistry=current_registry()) = (s, ZERO)

function ibp_with_boundary(s::TSum, field::Symbol;
                           registry::TensorRegistry=current_registry())
    bulks = TensorExpr[]
    bdrys = TensorExpr[]
    for t in s.terms
        b, d = ibp_with_boundary(t, field; registry=registry)
        push!(bulks, b)
        push!(bdrys, d)
    end
    (tsum(bulks), tsum(bdrys))
end

function ibp_with_boundary(d::TDeriv, field::Symbol;
                           registry::TensorRegistry=current_registry())
    # Standalone derivative (not inside a product) -- IBP is a no-op.
    (d, ZERO)
end

function ibp_with_boundary(p::TProduct, field::Symbol;
                           registry::TensorRegistry=current_registry())
    factors = p.factors
    for (i, fi) in enumerate(factors)
        base, idxs = _peel_all_derivs_of(fi, field)
        isempty(idxs) && continue

        # ── bulk: same logic as ibp_product ──
        rest_factors = TensorExpr[factors[j] for j in eachindex(factors) if j != i]
        rest = isempty(rest_factors) ? TScalar(1 // 1) : tproduct(1 // 1, rest_factors)

        d_rest = rest
        for idx in idxs
            d_rest = TDeriv(idx, d_rest)
        end
        d_rest = expand_derivatives(d_rest)

        sign = iseven(length(idxs)) ? 1 : -1
        bulk = tproduct(Rational{Int}(sign) * p.scalar, TensorExpr[base, d_rest])

        # ── boundary = original - bulk ──
        # This is the total-derivative surface contribution:
        #   d_a1(... field ... rest ...) type terms.
        # We compute it as  expr - bulk  so that  expr = bulk + boundary.
        boundary = tsum(TensorExpr[p, tproduct(-1 // 1, TensorExpr[bulk])])

        return (bulk, boundary)
    end
    # No derivative of `field` found -- trivially no boundary term.
    (p, ZERO)
end
