#= Extended metric infrastructure.

DefMetric full setup, signature tracking, epsilon tensor, flat metrics,
frozen metrics, conformal transformations, metric determinant.
=#

"""
    MetricSignature

Store the metric signature as a tuple of +1/-1 values.
"""
struct MetricSignature
    signs::Vector{Int}
end

MetricSignature(s::Tuple) = MetricSignature(collect(Int, s))

"""
    lorentzian(dim::Int) -> MetricSignature

Standard Lorentzian signature (-,+,+,...,+).
"""
lorentzian(dim::Int) = MetricSignature(vcat([-1], fill(1, dim - 1)))

"""
    euclidean(dim::Int) -> MetricSignature

Euclidean signature (+,+,...,+).
"""
euclidean(dim::Int) = MetricSignature(fill(1, dim))

"""
    sign_det(sig::MetricSignature) -> Int

Sign of the metric determinant: product of all signature entries.
"""
sign_det(sig::MetricSignature) = prod(sig.signs)

"""
    define_metric!(reg, name; manifold, signature=nothing, covd_name=nothing)

Full DefMetric one-liner: register metric, inverse, delta, epsilon,
Levi-Civita CovD with Christoffel, and all curvature tensors with Bianchi rules.
"""
function define_metric!(reg::TensorRegistry, name::Symbol;
                        manifold::Symbol,
                        signature::Union{MetricSignature, Nothing}=nothing,
                        covd_name::Union{Symbol, Nothing}=nothing)
    mp = get_manifold(reg, manifold)
    d = mp.dim

    # Store signature
    sig = signature !== nothing ? signature : lorentzian(d)

    # Register metric tensor g_{ab} (symmetric)
    if !has_tensor(reg, name)
        register_tensor!(reg, TensorProperties(
            name=name, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true,
            options=Dict{Symbol,Any}(:is_metric => true,
                                     :signature => sig)))
    else
        # Update signature on existing metric
        get_tensor(reg, name).options[:signature] = sig
    end

    # Register delta
    if !has_tensor(reg, :δ)
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            is_delta=true,
            options=Dict{Symbol,Any}(:is_delta => true)))
    end

    # Register epsilon tensor ε_{a1...ad} (fully antisymmetric)
    eps_name = Symbol(:ε, name)
    if !has_tensor(reg, eps_name)
        syms = SymmetrySpec[]
        for i in 1:d-1
            push!(syms, AntiSymmetric(i, i + 1))
        end
        register_tensor!(reg, TensorProperties(
            name=eps_name, manifold=manifold, rank=(0, d),
            symmetries=syms,
            options=Dict{Symbol,Any}(:is_epsilon => true,
                                     :metric => name,
                                     :sign_det => sign_det(sig))))
    end

    # Register curvature tensors
    define_curvature_tensors!(reg, manifold, name)

    # Register Levi-Civita CovD
    cd = covd_name !== nothing ? covd_name : Symbol(:∇, name)
    covd_props = define_covd!(reg, cd; manifold=manifold, metric=name)

    # Register Bianchi rules
    for rule in bianchi_rules(; manifold=manifold, metric=name)
        register_rule!(reg, rule)
    end

    nothing
end

"""
    metric_signature(reg, metric) -> MetricSignature

Retrieve the signature of a registered metric.
"""
function metric_signature(reg::TensorRegistry, metric::Symbol)
    props = get_tensor(reg, metric)
    get(props.options, :signature, lorentzian(get_manifold(reg, props.manifold).dim))
end

"""
    set_flat!(reg, metric)

Mark a metric as flat. Auto-registers rules: Riem=0, Ric=0, RicScalar=0,
Weyl=0, Ein=0, and Christoffel=0.
"""
function set_flat!(reg::TensorRegistry, metric::Symbol)
    has_tensor(reg, metric) || error("Metric $metric not registered")
    props = get_tensor(reg, metric)
    props.flat = true
    props.options[:flat] = true

    for tname in [:Riem, :Ric, :RicScalar, :Weyl, :Ein, :Sch]
        if has_tensor(reg, tname)
            register_rule!(reg, RewriteRule(
                expr -> expr isa Tensor && expr.name == tname,
                _ -> TScalar(0 // 1)
            ))
        end
    end

    # Christoffel = 0
    if has_tensor(reg, metric)
        tp = get_tensor(reg, metric)
        for (tname, tp2) in reg.tensors
            if tp2.is_christoffel &&
               get(tp2.options, :metric, nothing) == metric
                register_rule!(reg, RewriteRule(
                    expr -> expr isa Tensor && expr.name == tname,
                    _ -> TScalar(0 // 1)
                ))
            end
        end
    end
    nothing
end

"""
    is_flat(reg, metric) -> Bool

Check if a metric is marked as flat.
"""
function is_flat(reg::TensorRegistry, metric::Symbol)
    has_tensor(reg, metric) || return false
    get_tensor(reg, metric).flat
end

"""
    freeze_metric!(reg, metric)

Freeze a metric so it does not participate in contraction.
"""
function freeze_metric!(reg::TensorRegistry, metric::Symbol)
    has_tensor(reg, metric) || error("Metric $metric not registered")
    tp = get_tensor(reg, metric)
    tp.frozen = true
    tp.options[:frozen] = true
    nothing
end

"""
    unfreeze_metric!(reg, metric)

Unfreeze a metric to allow contraction again.
"""
function unfreeze_metric!(reg::TensorRegistry, metric::Symbol)
    has_tensor(reg, metric) || error("Metric $metric not registered")
    tp = get_tensor(reg, metric)
    tp.frozen = false
    tp.options[:frozen] = false
    nothing
end

"""
    is_frozen(reg, metric) -> Bool

Check if a metric is frozen.
"""
function is_frozen(reg::TensorRegistry, metric::Symbol)
    has_tensor(reg, metric) || return false
    get_tensor(reg, metric).frozen
end

"""
    separate_metric(expr::TensorExpr, idx::Symbol, metric::Symbol) -> TensorExpr

Insert a metric tensor to raise or lower a specific index.
T^a_b with idx=:a → g^{ac} T_c_b (lower a, insert metric to raise back)
"""
function separate_metric(expr::TensorExpr, idx::Symbol, metric::Symbol)
    _separate_metric_walk(expr, idx, metric)
end

function _separate_metric_walk(t::Tensor, idx::Symbol, metric::Symbol)
    slot = findfirst(i -> i.name == idx, t.indices)
    slot === nothing && return t

    tidx = t.indices[slot]
    used = Set{Symbol}(i.name for i in t.indices)
    d = fresh_index(used)

    new_indices = copy(t.indices)
    if tidx.position == Up
        # T^a... → g^{a d} T_d...
        new_indices[slot] = down(d)
        return Tensor(metric, [tidx, up(d)]) * Tensor(t.name, new_indices)
    else
        # T_a... → g_{a d} T^d...
        new_indices[slot] = up(d)
        return Tensor(metric, [tidx, down(d)]) * Tensor(t.name, new_indices)
    end
end

function _separate_metric_walk(p::TProduct, idx::Symbol, metric::Symbol)
    TProduct(p.scalar, TensorExpr[_separate_metric_walk(f, idx, metric) for f in p.factors])
end

function _separate_metric_walk(s::TSum, idx::Symbol, metric::Symbol)
    tsum(TensorExpr[_separate_metric_walk(t, idx, metric) for t in s.terms])
end

function _separate_metric_walk(d::TDeriv, idx::Symbol, metric::Symbol)
    TDeriv(d.index, _separate_metric_walk(d.arg, idx, metric), d.covd)
end

_separate_metric_walk(s::TScalar, ::Symbol, ::Symbol) = s

"""
    metric_det_expr(metric::Symbol) -> TScalar

Return a symbolic scalar representing the metric determinant det(g).
"""
function metric_det_expr(metric::Symbol)
    TScalar(Symbol(:det_, metric))
end

"""
    sqrt_det_expr(metric::Symbol; signature=nothing) -> TScalar

Return √|det(g)| or √(-det(g)) depending on signature.
"""
function sqrt_det_expr(metric::Symbol; neg::Bool=true)
    if neg
        TScalar(Symbol(:sqrt_neg_det_, metric))
    else
        TScalar(Symbol(:sqrt_det_, metric))
    end
end

"""
    gdelta(up_indices::Vector{TIndex}, down_indices::Vector{TIndex}) -> TensorExpr

Generalized Kronecker delta δ^{a1...ap}_{b1...bp} = det of δ matrix.
"""
function gdelta(up_indices::Vector{TIndex}, down_indices::Vector{TIndex})
    p = length(up_indices)
    @assert p == length(down_indices) "Same number of up and down indices required"
    @assert all(i -> i.position == Up, up_indices) "First argument must be all Up indices"
    @assert all(i -> i.position == Down, down_indices) "Second argument must be all Down indices"

    p == 1 && return Tensor(:δ, [up_indices[1], down_indices[1]])

    # Expand as determinant of δ^{ai}_{bj} matrix using Leibniz formula
    terms = TensorExpr[]
    for perm in _permutations_with_sign_impl(p)
        sigma, sgn = perm
        factors = TensorExpr[]
        for i in 1:p
            push!(factors, Tensor(:δ, [up_indices[i], down_indices[sigma[i]]]))
        end
        push!(terms, tproduct(Rational{Int}(sgn), factors))
    end
    tsum(terms)
end

"""
    expand_gdelta(expr::TensorExpr) -> TensorExpr

Expand generalized Kronecker deltas into products of ordinary deltas.
(Already expanded by gdelta constructor.)
"""
expand_gdelta(expr::TensorExpr) = expr

# Helper: generate permutations with signs
function _permutations_with_sign_impl(n::Int)
    if n == 0
        return [(Int[], 1)]
    end
    result = Tuple{Vector{Int}, Int}[]
    _perm_helper!(result, collect(1:n), 1, n)
    result
end

function _perm_helper!(result, arr, k, n)
    if k == n
        push!(result, (copy(arr), _perm_sign(arr)))
        return
    end
    for i in k:n
        arr[k], arr[i] = arr[i], arr[k]
        _perm_helper!(result, arr, k + 1, n)
        arr[k], arr[i] = arr[i], arr[k]
    end
end

"""
    set_conformal_to!(reg, g1, g2, factor)

Set g1 = e^{2f} g2 where factor is the conformal factor symbol.
Auto-generates conformal transformation rules for curvature tensors.
"""
function set_conformal_to!(reg::TensorRegistry, g1::Symbol, g2::Symbol, factor::Symbol)
    has_tensor(reg, g1) || error("Metric $g1 not registered")
    has_tensor(reg, g2) || error("Metric $g2 not registered")
    tp1 = get_tensor(reg, g1)
    tp1.options[:conformal_to] = g2
    tp1.options[:conformal_factor] = factor
    nothing
end

function _perm_sign(perm::Vector{Int})
    n = length(perm)
    visited = falses(n)
    sign = 1
    for i in 1:n
        visited[i] && continue
        visited[i] = true
        cycle_len = 1
        j = perm[i]
        while j != i
            visited[j] = true
            j = perm[j]
            cycle_len += 1
        end
        if iseven(cycle_len)
            sign = -sign
        end
    end
    sign
end
