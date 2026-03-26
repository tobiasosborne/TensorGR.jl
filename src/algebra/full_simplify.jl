#= FullSimplification: comprehensive curvature simplification.

Higher-level than `simplify()`, composing all available tools:
  simplify + DDIs + Bianchi + basis comparison + CovD commutation

Inspired by xTras FullSimplification (Nutma, arXiv:1308.3493, §4),
extended with automatic basis comparison (Riemann vs Weyl).
=#

"""
    full_simplify(expr; registry, metric, dim, covd, basis, use_ddis, maxiter, parallel)

Comprehensive curvature simplification combining all available tools.

Higher-level than `simplify()`: automatically applies DDIs when dimension
is known, tries multiple curvature bases, and uses covariant derivative
commutation with Bianchi identities.

# Keyword arguments
- `registry::TensorRegistry`: registry with tensor definitions
- `metric::Union{Symbol,Nothing}=nothing`: metric name (auto-inferred if `nothing`)
- `dim::Union{Int,Nothing}=nothing`: manifold dimension (auto-inferred if `nothing`)
- `covd::Union{Symbol,Nothing}=nothing`: covariant derivative name for CovD commutation
- `basis::Symbol=:auto`: curvature basis — `:riemann`, `:weyl`, or `:auto` (tries both, picks shortest)
- `use_ddis::Bool=true`: apply dimensionally-dependent identities (requires `dim`)
- `maxiter::Int=20`: maximum simplification iterations
- `parallel::Bool=false`: enable TSum-level parallelism

# Examples
```julia
reg = TensorRegistry()
@manifold M4 dim=4 metric=g registry=reg
define_curvature_tensors!(reg, :M4, :g)

# Basic: simplify with DDIs in d=4
result = full_simplify(expr; registry=reg)

# With CovD commutation
@covd D on=M4 metric=g registry=reg
result = full_simplify(expr; registry=reg, covd=:D)

# Force Weyl basis
result = full_simplify(expr; registry=reg, basis=:weyl)
```

See also: [`simplify`](@ref), [`simplify_with_ddis`](@ref), [`to_riemann`](@ref)
"""
function full_simplify(expr::TensorExpr;
                        registry::TensorRegistry=current_registry(),
                        metric::Union{Symbol,Nothing}=nothing,
                        dim::Union{Int,Nothing}=nothing,
                        covd::Union{Symbol,Nothing}=nothing,
                        basis::Symbol=:auto,
                        use_ddis::Bool=true,
                        maxiter::Int=20,
                        parallel::Bool=false)

    basis in (:auto, :riemann, :weyl) ||
        throw(ArgumentError("basis must be :auto, :riemann, or :weyl (got :$basis)"))

    reg = registry

    # Auto-infer metric and dimension from registry
    met = something(metric, _fs_infer_metric(reg))
    d = something(dim, _fs_try_infer_dim(reg)...)

    # Phase 1: Normalize to Riemann+Ricci+RicScalar basis
    dim_val = d !== nothing ? d : 4
    result = to_riemann(expr; metric=met, dim=dim_val)

    # Phase 2: Core simplification with DDIs
    skw = Dict{Symbol,Any}(:registry => reg, :maxiter => maxiter, :parallel => parallel)
    if covd !== nothing
        skw[:commute_covds_name] = covd
    end

    if use_ddis && d !== nothing
        ddi_order = _fs_ddi_order(result, d)
        register_ddi_rules!(reg; dim=d, order=ddi_order, metric=met)
        result = simplify(result; pairs(skw)...)
    else
        result = simplify(result; pairs(skw)...)
    end

    # Phase 3: Basis comparison (try both, pick shortest)
    if basis == :auto && d !== nothing && d >= 3
        riemann_result = result
        weyl_candidate = _fs_to_weyl_basis(result; metric=met, dim=d)
        if weyl_candidate !== result  # only simplify if conversion changed anything
            weyl_result = simplify(weyl_candidate; pairs(skw)...)
            result = _fs_pick_shortest(riemann_result, weyl_result)
        end
    elseif basis == :weyl && d !== nothing && d >= 3
        result = _fs_to_weyl_basis(result; metric=met, dim=d)
        result = simplify(result; pairs(skw)...)
    end

    result
end

# ── Helpers ──────────────────────────────────────────────────────────

"""Infer metric name from registry."""
function _fs_infer_metric(reg::TensorRegistry)
    isempty(reg.metric_cache) ? :g : first(values(reg.metric_cache))
end

"""Infer dimension from registry, returns (dim,) or (nothing,) for `something`."""
function _fs_try_infer_dim(reg::TensorRegistry)
    isempty(reg.manifolds) ? (nothing,) : (first(values(reg.manifolds)).dim,)
end

"""Determine DDI order from expression degree and dimension."""
function _fs_ddi_order(expr::TensorExpr, dim)
    deg = count_riemann_degree(expr)
    # DDI order matches curvature degree (order 2 = quadratic DDIs, etc.)
    # Cap at dim÷2 (DDIs beyond that are trivially zero)
    dim isa Int ? clamp(deg, 2, dim ÷ 2) : deg
end

"""Convert Riemann tensors to Weyl + Ricci decomposition (expression-level)."""
function _fs_to_weyl_basis(expr::TensorExpr; metric::Symbol=:g, dim::Int=4)
    walk(expr) do node
        node isa Tensor || return node
        if node.name == :Riem && length(node.indices) == 4
            a, b, c, d = node.indices
            riemann_to_weyl(a, b, c, d, metric; dim=dim)
        else
            node
        end
    end
end

"""Pick the expression with fewer terms."""
function _fs_pick_shortest(e1::TensorExpr, e2::TensorExpr)
    _fs_term_count(e1) <= _fs_term_count(e2) ? e1 : e2
end

_fs_term_count(s::TSum) = length(s.terms)
_fs_term_count(::TensorExpr) = 1
