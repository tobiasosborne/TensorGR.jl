#= Metric perturbation expansion.

DefMetricPerturbation: define g = g₀ + ε h + ε² h₂ + ...
Auto-generates perturbation rules for g⁻¹, Christoffel, Riemann, Ricci, etc.

At order n, the inverse metric perturbation is:
  δⁿ(g^{ab}) = Σ (-1)^k g^{a?} h^{??} ... g^{?b}  (partition-based)

The Christoffel perturbation at order n uses the "three-pert" recursion:
  δⁿΓ^a_{bc} = (1/2) g^{ad} Σ_{partitions} (∇_b δᵖg_{cd} + ∇_c δᵖg_{bd} - ∇_d δᵖg_{bc})
=#

"""
    MetricPerturbation(metric, perturbation, order_parameter)

Definition of a metric perturbation g = g₀ + ε h.
"""
struct MetricPerturbation
    metric::Symbol
    perturbation::Symbol
    background::Symbol
    order_param::Symbol
end

"""
    define_metric_perturbation!(reg, metric, perturbation;
        background=Symbol(metric, :₀), order_param=:ε) -> MetricPerturbation

Define a metric perturbation and register the perturbation tensor.
"""
function define_metric_perturbation!(reg::TensorRegistry, metric::Symbol,
                                     perturbation::Symbol;
                                     background::Symbol=Symbol(metric, :_bg),
                                     order_param::Symbol=:ε)
    mp = get_manifold(reg, get_tensor(reg, metric).manifold)

    # Register perturbation tensor h_{ab} with same symmetries as metric
    if !has_tensor(reg, perturbation)
        register_tensor!(reg, TensorProperties(
            name=perturbation, manifold=mp.name, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)]))
    end

    MetricPerturbation(metric, perturbation, background, order_param)
end

"""
    perturb(expr, mp::MetricPerturbation, order::Int) -> TensorExpr

Expand a tensor expression to the given perturbation order.
"""
function perturb(expr::TensorExpr, mp::MetricPerturbation, order::Int)
    _perturb(expr, mp, order)
end

function _perturb(t::Tensor, mp::MetricPerturbation, order::Int)
    if t.name == mp.metric && order == 1
        # δg_{ab} = h_{ab}
        return Tensor(mp.perturbation, t.indices)
    elseif t.name == mp.metric && order == 0
        return t  # zeroth order = background metric
    elseif t.name == mp.metric && order >= 2
        return ZERO  # only first-order perturbation of g itself
    end
    # Non-metric tensors: perturbation is zero at order >= 1 (they're background)
    order == 0 ? t : ZERO
end

function _perturb(s::TScalar, ::MetricPerturbation, order::Int)
    order == 0 ? s : ZERO
end

function _perturb(s::TSum, mp::MetricPerturbation, order::Int)
    tsum(TensorExpr[_perturb(t, mp, order) for t in s.terms])
end

function _perturb(p::TProduct, mp::MetricPerturbation, order::Int)
    # Perturbation of a product: sum over all ways to distribute order among factors
    factors = p.factors
    nf = length(factors)
    comps = all_compositions(order, nf)

    terms = TensorExpr[]
    for comp in comps
        parts = TensorExpr[]
        valid = true
        for (i, fi) in enumerate(factors)
            pi = _perturb(fi, mp, comp[i])
            if pi == ZERO
                valid = false
                break
            end
            push!(parts, pi)
        end
        valid || continue
        push!(terms, tproduct(p.scalar, parts))
    end
    tsum(terms)
end

function _perturb(d::TDeriv, mp::MetricPerturbation, order::Int)
    # Perturbation commutes with partial derivatives (on flat background)
    TDeriv(d.index, _perturb(d.arg, mp, order))
end

"""
    δinverse_metric(mp::MetricPerturbation, idx_a::TIndex, idx_b::TIndex, order::Int) -> TensorExpr

Compute the perturbation of the inverse metric at a given order.
At order 1: δ(g^{ab}) = -g^{ac} g^{bd} h_{cd}
At order n: uses the partition-based recursion.
"""
function δinverse_metric(mp::MetricPerturbation, idx_a::TIndex, idx_b::TIndex, order::Int)
    order == 0 && return Tensor(mp.metric, [idx_a, idx_b])
    order < 0 && return ZERO

    if order == 1
        used = Set{Symbol}([idx_a.name, idx_b.name])
        c = fresh_index(used)
        push!(used, c)
        d = fresh_index(used)
        # δ(g^{ab}) = -g^{ac} g^{bd} h_{cd}
        return tproduct(-1 // 1, TensorExpr[
            Tensor(mp.metric, [idx_a, up(c)]),
            Tensor(mp.metric, [idx_b, up(d)]),
            Tensor(mp.perturbation, [down(c), down(d)])
        ])
    end

    # Higher orders: δⁿ(g^{ab}) = -Σ_{k=1}^{n-1} δᵏ(g^{ac}) g^{bd} δⁿ⁻ᵏ(g_{cd})
    # This is the standard recursion from matrix perturbation theory
    used = Set{Symbol}([idx_a.name, idx_b.name])
    terms = TensorExpr[]
    for k in 1:order-1
        c = fresh_index(used)
        push!(used, c)
        d = fresh_index(used)
        push!(used, d)
        δk_inv = δinverse_metric(mp, idx_a, up(c), k)
        δnk_g = order - k == 1 ? Tensor(mp.perturbation, [down(c), down(d)]) : ZERO
        if δnk_g != ZERO
            term = tproduct(-1 // 1, TensorExpr[
                δk_inv,
                Tensor(mp.metric, [idx_b, up(d)]),
                δnk_g
            ])
            push!(terms, term)
        end
    end
    tsum(terms)
end
